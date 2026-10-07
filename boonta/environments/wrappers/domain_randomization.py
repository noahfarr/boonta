from collections.abc import Callable
from typing import Any

import jax

from boonta.utils import Array, Key, PyTree, Timestep

from .wrapper import Wrapper


class DomainRandomization(Wrapper):
    def __init__(self, env, randomize_fn: Callable[[Key, PyTree], PyTree]):
        super().__init__(env)
        self.randomize_fn = randomize_fn

    def init(self, key: Key) -> tuple[Any, Timestep]:
        sample_key, reset_key = jax.random.split(key)
        state, timestep = self._env.init(reset_key)
        params = self.randomize_fn(sample_key, state.params)
        return state.replace(params=params), timestep

    def step(self, key: Key, state: Any, action: Array) -> tuple[Any, Timestep]:
        return self._env.step(key, state, action)

    def update(self, state, **kwargs):
        return self._env.update(state, **kwargs)
