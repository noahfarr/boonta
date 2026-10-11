from collections.abc import Callable

import jax
import jax.numpy as jnp

from boonta.algorithms import Algorithm
from boonta.environments.wrappers import Wrapper
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import Array, Key, PyTree, Timestep, broadcast
from boonta.utils.typing import Environment


class DomainRandomization(Wrapper):
    def __init__(self, env, sample: Callable[[Key], PyTree]):
        super().__init__(env)
        self._randomize = sample

    def randomize(self, key: Key, env_state, obs, chosen: Array):
        def start(state, key):
            sample_key, update_key = jax.random.split(key)
            return self._env.update(state, update_key, theta=self._randomize(sample_key))

        started = jax.vmap(start)(env_state, jax.random.split(key, self.num_envs))

        def select(fresh, leaf):
            return jnp.where(broadcast(chosen, leaf), fresh, leaf)

        env_state = jax.tree.map(select, started, env_state)
        return env_state, jax.tree.map(select, self._env.observe(started), obs)

    def init(self, key: Key) -> tuple[PyTree, Timestep]:
        init_key, randomize_key = jax.random.split(key)
        env_state, timestep = self._env.init(init_key)
        env_state, obs = self.randomize(
            randomize_key, env_state, timestep.obs, jnp.ones(self.num_envs, bool)
        )
        return env_state, timestep.replace(obs=obs)

    def step(self, key: Key, state: PyTree, action: Array) -> tuple[PyTree, Timestep]:
        env_state, timestep = self._env.step(key, state, action)
        done = timestep.done.reshape(self.num_envs, -1).all(axis=-1)
        env_state, obs = jax.lax.cond(
            done.any(),
            lambda: self.randomize(
                jax.random.fold_in(key, 1), env_state, timestep.obs, done
            ),
            lambda: (env_state, timestep.obs),
        )
        return env_state, timestep.replace(obs=obs)

    def update(self, state, key, **kwargs):
        return self._env.update(state, key, **kwargs)

    def action_mask(self, state):
        return self._env.action_mask(state)

    def observe(self, state):
        return self._env.observe(state)


def domain_randomization(
    algorithm: Algorithm,
    environment: Environment,
    sample: Callable[[Key], PyTree],
    **kwargs,
) -> tuple[Algorithm, DomainRandomization, Pit, Lap]:
    environment = DomainRandomization(environment, sample)
    return algorithm, environment, lambda state: state, lambda state: state
