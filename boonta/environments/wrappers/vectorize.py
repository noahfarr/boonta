from typing import Any

import jax
import jax.numpy as jnp

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

    def init(self, key: Key, group_size: int = 1) -> tuple[PyTree, Any]:
        num_groups = self.num_envs // group_size
        group_keys = jax.random.split(key, num_groups)
        keys = jnp.repeat(group_keys, group_size, axis=0)
        return jax.vmap(self._env.init)(keys)

    def step(self, key: Key, state: PyTree, action: Array) -> tuple[PyTree, Any]:
        keys = jax.random.split(key, self.num_envs)
        return jax.vmap(self._env.step)(keys, state, action)

    def update(self, state: PyTree, **kwargs) -> PyTree:
        return self._env.update(state, **kwargs)

    def action_mask(self, state: PyTree) -> Array | None:
        return jax.vmap(self._env.action_mask)(state)

    def observe(self, state: PyTree) -> PyTree:
        return jax.vmap(self._env.observe)(state)
