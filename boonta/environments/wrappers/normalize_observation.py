from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, PyTree, Timestep, broadcast

from .vectorize import Vectorize
from .wrapper import Wrapper, WrapperState


@struct.dataclass
class NormalizeObservationState(WrapperState):
    mean: PyTree
    m2: PyTree
    count: Array


class NormalizeObservation(Wrapper):
    def __init__(
        self,
        env,
        epsilon: float = 1e-8,
        filter: Callable[[Any, Array], bool] | None = None,
    ):
        super().__init__(env)
        assert self.wraps(Vectorize), (
            "NormalizeObservation computes statistics over axis 0, so env must "
            "have a Vectorize wrapper somewhere underneath it."
        )
        self.epsilon = epsilon
        self.filter = filter

    def merge(
        self, obs: PyTree, mean: PyTree, m2: PyTree, count: Array
    ) -> tuple[PyTree, PyTree, Array]:
        count = count + 1
        delta = jax.tree.map(lambda o, m: o - m, obs, mean)
        mean = jax.tree.map(lambda m, d: m + d / broadcast(count, d), mean, delta)
        m2 = jax.tree.map(lambda s, d, o, m: s + d * (o - m), m2, delta, obs, mean)
        return mean, m2, count

    def moments(
        self, mean: PyTree, m2: PyTree, count: Array
    ) -> tuple[PyTree, PyTree]:
        batch_count = count.sum()
        batch_mean = jax.tree.map(
            lambda m: (broadcast(count, m) * m).sum(axis=0) / batch_count, mean
        )
        variance = jax.tree.map(
            lambda s, m, b: (
                s.sum(axis=0) + (broadcast(count, m) * (m - b) ** 2).sum(axis=0)
            )
            / batch_count,
            m2,
            mean,
            batch_mean,
        )
        return batch_mean, variance

    def normalize(self, obs: PyTree, mean: PyTree, m2: PyTree, count: Array) -> PyTree:
        batch_mean, variance = self.moments(mean, m2, count)

        def normalize_leaf(path, o, m, v):
            if self.filter is not None and not self.filter(path, o):
                return o
            return (o - m) / jnp.sqrt(v + self.epsilon)

        return jax.tree_util.tree_map_with_path(
            normalize_leaf, obs, batch_mean, variance
        )

    def init(
        self, key: Key
    ) -> tuple[NormalizeObservationState, Timestep]:
        env_state, timestep = self._env.init(key)
        obs = timestep.obs
        zeros = jax.tree.map(
            lambda o: jnp.zeros(o.shape, jnp.result_type(o, jnp.float32)), obs
        )
        leaf, *_ = jax.tree.leaves(obs)
        num_envs, *_ = leaf.shape
        count = jnp.zeros(num_envs, jnp.float32)
        mean, m2, count = self.merge(obs, zeros, zeros, count)
        state = NormalizeObservationState(env_state, mean, m2, count)
        return state, timestep.replace(obs=self.normalize(obs, mean, m2, count))

    def step(
        self,
        key: Key,
        state: NormalizeObservationState,
        action: Array,
    ) -> tuple[NormalizeObservationState, Timestep]:
        env_state, timestep = self._env.step(key, state.env_state, action)
        mean, m2, count = self.merge(timestep.obs, state.mean, state.m2, state.count)
        state = NormalizeObservationState(env_state, mean, m2, count)
        obs = self.normalize(timestep.obs, mean, m2, count)
        return state, timestep.replace(obs=obs)
