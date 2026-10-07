from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp
import lox

from boonta.utils import Array, Key

from .wrapper import Wrapper


class LogInfo(Wrapper):
    def __init__(
        self, env, keys: Iterable[str] | None = None, tags: Iterable[str] = ("training",)
    ):
        super().__init__(env)
        self.keys = None if keys is None else tuple(keys)
        self.tags = tags

    def step(self, key: Key, state: Any, action: Array) -> tuple[Any, Any]:
        state, timestep = self._env.step(key, state, action)
        lox.log(
            {
                f"info/{name}": jnp.asarray(value, jnp.float32)
                for name, value in timestep.info.items()
                if self.keys is None or name in self.keys
            },
            tags=self.tags,
        )
        return state, timestep

    def update(self, state, **kwargs):
        return self._env.update(state, **kwargs)
