import argparse
import time
from functools import partial

import hydra
import jax
from hydra.core.hydra_config import HydraConfig

from hangar import recipes


def compose(overrides):
    with hydra.initialize(version_base=None, config_path="../config"):
        cfg = hydra.compose(
            config_name="train", overrides=list(overrides), return_hydra_config=True
        )
    HydraConfig.instance().set_config(cfg)
    return cfg


def seconds(function, *args):
    jax.block_until_ready(function(*args))
    start = time.perf_counter()
    jax.block_until_ready(function(*args))
    return time.perf_counter() - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    podracer = recipes.make(compose(args.overrides))
    state = podracer.init(jax.random.key(args.seed))
    steps = podracer.config.num_steps
    environment, algorithm = podracer.environment, podracer.algorithm
    space = environment.action_space()
    key = jax.random.key(1)

    @jax.jit
    def rollout(state, key):
        return jax.lax.scan(
            partial(podracer.rollout, temperature=1.0), state, jax.random.split(key, steps)
        )

    @jax.jit
    def policy(state, key):
        def body(loop, key):
            algorithm_state, carry = loop
            algorithm_state, carry, action, _ = algorithm.step(
                algorithm_state, carry, key, state.timestep, 1.0
            )
            return (algorithm_state, carry), action

        loop = (state.algorithm_state, state.carry)
        return jax.lax.scan(body, loop, jax.random.split(key, steps))[1]

    @jax.jit
    def simulate(state, key):
        def body(environment_state, key):
            action_key, step_key = jax.random.split(key)
            action = jax.random.randint(
                action_key, state.timestep.action.shape, space.low, space.high + 1
            )
            environment_state, timestep = environment.step(step_key, environment_state, action)
            return environment_state, timestep.reward.sum()

        return jax.lax.scan(body, state.environment_state, jax.random.split(key, steps))[1]

    update = jax.jit(algorithm.update)

    _, transitions = rollout(state, key)
    timings = {
        "rollout (policy and environment)": seconds(rollout, state, key),
        "  policy only": seconds(policy, state, key),
        "  environment only": seconds(simulate, state, key),
        "update": seconds(update, state.algorithm_state, key, transitions),
    }
    batch = podracer.batch_size
    total = timings["rollout (policy and environment)"] + timings["update"]
    for name, duration in timings.items():
        print(f"{duration * 1e3:9.1f} ms  {duration / total:6.1%}  {name}")
    print(f"{batch / total:,.0f} agent-steps per second for one rollout and one update")
