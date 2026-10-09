import performax as px

px.enable_device_profiling()
px.enable_barriers()

import argparse
import time
from dataclasses import dataclass

import hydra
import jax
import jax.numpy as jnp
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from boonta.algorithms.wrappers import Wrapper as AlgorithmWrapper
from boonta.environments.wrappers import Wrapper as EnvironmentWrapper
from boonta.utils import Transition
from hangar import recipes

def compose(overrides):
    with hydra.initialize(version_base=None, config_path="../config"):
        cfg = hydra.compose(
            config_name="config", overrides=list(overrides), return_hydra_config=True
        )
    HydraConfig.instance().set_config(cfg)
    return cfg


def sample(key, action, space):
    if jnp.issubdtype(action.dtype, jnp.integer):
        return jax.random.randint(
            key, action.shape, space.low, jnp.asarray(space.high) + 1, action.dtype
        )
    return jax.random.uniform(key, action.shape, action.dtype, space.low, space.high)


@dataclass
class TrackedAlgorithm(AlgorithmWrapper):
    def step(self, state, key, timestep, temperature=1.0):
        return px.track(name="algorithm/step")(self.algorithm.step)(
            state, key, timestep, temperature
        )

    def update(self, state, key, transitions):
        return px.track(name="algorithm/update")(self.algorithm.update)(
            state, key, transitions
        )


class TrackedEnvironment(EnvironmentWrapper):
    def step(self, key, state, action):
        return px.track(name="environment/step")(self._env.step)(key, state, action)

    def update(self, state, key, **kwargs):
        return self._env.update(state, key, **kwargs)

    def action_mask(self, state):
        return self._env.action_mask(state)

    def observe(self, state):
        return self._env.observe(state)

    def render(self, state):
        return self._env.render(state)

    def close(self, state):
        return self._env.close(state)


def identity(state):
    return state


def benchmark(overrides, num_updates, seed):
    cfg = compose(overrides)
    namespace = cfg.environment.namespace
    suite = cfg.environment.get("suite", namespace)
    name = HydraConfig.get().runtime.choices["algorithm"]
    components = recipes.register[(name, namespace, suite)](cfg)
    components["algorithm"] = TrackedAlgorithm(components["algorithm"])
    if components.get("environment") is not None:
        components["environment"] = TrackedEnvironment(components["environment"])
    components["pit"] = px.track(name="pit")(components.get("pit", identity))
    components["lap"] = px.track(name="lap")(components.get("lap", identity))
    podracer = instantiate(cfg.podracer)(**components)
    algorithm, environment = podracer.algorithm, podracer.environment

    num_envs = podracer.config.num_envs
    num_steps = podracer.config.num_steps
    num_agents = getattr(environment, "num_agents", 1)
    batch_size = num_envs * num_steps * num_agents
    space = environment.action_space()

    env_key, collect_key, update_key, init_key, train_key = jax.random.split(
        jax.random.key(seed), 5
    )

    @jax.jit
    def rollout(environment_state, timestep, key):
        def step(carry, key):
            environment_state, timestep = carry
            action_key, step_key = jax.random.split(key)
            action = sample(action_key, timestep.action, space)
            return environment.step(step_key, environment_state, action), None

        keys = jax.random.split(key, num_steps)
        carry, _ = jax.lax.scan(step, (environment_state, timestep), keys)
        return carry

    @jax.jit
    def collect(algorithm_state, environment_state, timestep, key):
        def step(carry, key):
            algorithm_state, environment_state, timestep = carry
            algorithm_key, environment_key = jax.random.split(key)
            algorithm_state, action, aux = algorithm.step(
                algorithm_state, algorithm_key, timestep, 1.0
            )
            environment_state, next_timestep = environment.step(
                environment_key, environment_state, action
            )
            transition = Transition(first=timestep, second=next_timestep, aux=aux)
            return (algorithm_state, environment_state, next_timestep), transition

        keys = jax.random.split(key, num_steps)
        return jax.lax.scan(step, (algorithm_state, environment_state, timestep), keys)

    update = jax.jit(algorithm.update)

    environment_state, timestep = environment.init(env_key)
    jax.block_until_ready(rollout(environment_state, timestep, env_key))

    start = time.monotonic()
    jax.block_until_ready(rollout(environment_state, timestep, env_key))
    env_time = time.monotonic() - start

    algorithm_state = algorithm.init(init_key, timestep)
    carry, _ = collect(algorithm_state, environment_state, timestep, collect_key)
    carry, transitions = jax.block_until_ready(collect(*carry, collect_key))

    start = time.monotonic()
    carry, transitions = jax.block_until_ready(collect(*carry, collect_key))
    rollout_time = time.monotonic() - start

    algorithm_state, *_ = carry
    algorithm_state = update(algorithm_state, update_key, transitions)
    jax.block_until_ready(update(algorithm_state, update_key, transitions))

    start = time.monotonic()
    jax.block_until_ready(update(algorithm_state, update_key, transitions))
    update_time = time.monotonic() - start
    environment.close(environment_state)

    def train(state):
        state = podracer.train(state, train_key, num_updates)
        if isinstance(state, tuple):
            state, _ = state
        return state

    state = jax.block_until_ready(train(train(podracer.init(init_key))))

    clock = []

    def race(state):
        start = time.monotonic()
        state = jax.block_until_ready(train(state))
        clock.append(time.monotonic() - start)
        return state

    state, stats = px.profile(race)(state)
    (train_time,) = clock
    podracer.close(state)

    return {
        "num_envs": num_envs,
        "batch_size": batch_size,
        "environment/SPS": batch_size / env_time,
        "rollout/SPS": batch_size / rollout_time,
        "update/SPS": batch_size / update_time,
        "training/SPS": num_updates * batch_size / train_time,
    }, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--num-updates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    results, stats = benchmark(args.overrides, args.num_updates, args.seed)
    for name, value in results.items():
        print(f"{name}: {value:,.0f}")
    print(stats.device)
