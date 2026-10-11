from collections.abc import Callable

import jax
import jax.numpy as jnp

from boonta.algorithms import Algorithm
from boonta.environments.wrappers import UED, Wrapper
from boonta.environments.wrappers.ued import UEDState
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import Array, Key, PyTree, Timestep, broadcast
from boonta.utils.typing import Environment


class Resample(Wrapper):
    def __init__(self, env, sample: Callable[[Key], PyTree]):
        super().__init__(env)
        self._sample = sample

    def draw(self, key: Key) -> PyTree:
        return jax.vmap(self._sample)(jax.random.split(key, self.num_envs))

    def init(self, key: Key) -> tuple[UEDState, Timestep]:
        init_key, draw_key, update_key = jax.random.split(key, 3)
        state, timestep = self._env.init(init_key)
        state = self._env.update(
            state,
            update_key,
            theta=self.draw(draw_key),
            assign=jnp.arange(self.num_envs, dtype=jnp.int32),
        )
        return state, timestep

    def step(self, key: Key, state: UEDState, action: Array) -> tuple[UEDState, Timestep]:
        state, timestep = self._env.step(key, state, action)
        done = timestep.done.reshape(self.num_envs, -1).all(axis=-1)
        theta = jax.tree.map(
            lambda fresh, leaf: jnp.where(broadcast(done, leaf), fresh, leaf),
            self.draw(jax.random.fold_in(key, 3)),
            state.theta,
        )
        return self._env.update(state, key, theta=theta), timestep

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
) -> tuple[Algorithm, Resample, Pit, Lap]:
    environment = Resample(UED(environment), sample)
    return algorithm, environment, lambda state: state, lambda state: state
