from typing import Any

from boonta.utils import Array, Key, PyTree

from .vectorize import Vectorize


class Batched(Vectorize):
    def init(self, key: Key, group_size: int = 1) -> tuple[PyTree, Any]:
        if group_size == 1:
            return self._env.init(key)
        return self._env.init(key, group_size=group_size)

    def step(self, key: Key, state: PyTree, action: Array) -> tuple[PyTree, Any]:
        return self._env.step(key, state, action)

    def update(self, state: PyTree, key: Key, **kwargs) -> PyTree:
        return self._env.update(state, key, **kwargs)

    def spread(self, state: PyTree, keys: Key, **kwargs) -> PyTree:
        return self._env.update(state, keys, **kwargs)

    def action_mask(self, state: PyTree) -> Array | None:
        return self._env.action_mask(state)

    def observe(self, state: PyTree) -> PyTree:
        return self._env.observe(state)
