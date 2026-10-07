from typing import Any

from boonta.utils import Array, Key, PyTree

from .vectorize import Vectorize


class Batched(Vectorize):
    def init(self, key: Key, group_size: int = 1) -> tuple[PyTree, Any]:
        return self._env.init(key)

    def step(self, key: Key, state: PyTree, action: Array) -> tuple[PyTree, Any]:
        return self._env.step(key, state, action)
