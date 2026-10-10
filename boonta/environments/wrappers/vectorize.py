from typing import Any

import jax

from boonta.utils import Array, Key, PyTree

from .wrapper import Wrapper


class Vectorize(Wrapper):
    def __init__(self, env, num_envs: int):
        from .batched import Batched

        super().__init__(env)
        assert not self.wraps(
            Batched
        ), "Vectorize cannot wrap a Batched environment; it already produces batches"
        self.num_envs = num_envs

    def init(self, key: Key) -> tuple[PyTree, Any]:
        return jax.vmap(self._env.init)(jax.random.split(key, self.num_envs))

    def step(self, key: Key, state: PyTree, action: Array) -> tuple[PyTree, Any]:
        keys = jax.random.split(key, self.num_envs)
        return jax.vmap(self._env.step)(keys, state, action)

    def update(self, state: PyTree, key: Key, **kwargs) -> PyTree:
        return self._env.update(state, key, **kwargs)

    def action_mask(self, state: PyTree) -> Array | None:
        return jax.vmap(self._env.action_mask)(state)

    def observe(self, state: PyTree) -> PyTree:
        return jax.vmap(self._env.observe)(state)
