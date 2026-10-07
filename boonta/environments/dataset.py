from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.experimental import io_callback

from boonta.utils import Array, Key, Timestep, broadcast

from .environment import Environment
from .spaces import Space


@struct.dataclass
class DatasetState:
    stream: Timestep
    cursor: Array


class Dataset(Environment):
    def __init__(self, data: Timestep, num_envs: int):
        self.data = data
        self.num_envs = num_envs
        leaf, *_ = jax.tree.leaves(data)
        self.size, *_ = leaf.shape

    def stream(self) -> Timestep:
        return io_callback(
            lambda: jax.tree.map(np.asarray, self.data),
            jax.tree.map(
                lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), self.data
            ),
        )

    def gather(self, stream: Timestep, cursor: Array) -> Timestep:
        return jax.tree.map(lambda leaf: leaf[cursor], stream)

    def init(
        self, key: Key
    ) -> tuple[DatasetState, Timestep]:
        stream = self.stream()
        cursor = jax.random.randint(key, (self.num_envs,), 0, self.size)
        return DatasetState(stream=stream, cursor=cursor), self.gather(stream, cursor)

    def step(
        self,
        key: Key,
        state: DatasetState,
        action: Array,
    ) -> tuple[DatasetState, Timestep]:
        del action
        cursor = (state.cursor + 1) % self.size
        timestep = self.gather(state.stream, cursor)

        initial_cursor = jax.random.randint(key, (self.num_envs,), 0, self.size)
        initial_timestep = self.gather(state.stream, initial_cursor)

        done = timestep.done
        cursor = jnp.where(done, initial_cursor, cursor)
        obs = jax.tree.map(
            lambda initial, leaf: jnp.where(broadcast(done, leaf), initial, leaf),
            initial_timestep.obs,
            timestep.obs,
        )
        return DatasetState(stream=state.stream, cursor=cursor), timestep.replace(
            obs=obs
        )

    def observation_space(self) -> Space:
        leaf, *_ = jax.tree.leaves(self.data.obs)
        return Space(shape=leaf.shape[1:], dtype=leaf.dtype)

    def action_space(self) -> Space:
        if self.data.action is None:
            return Space(shape=())
        return Space(shape=self.data.action.shape[1:], dtype=self.data.action.dtype)

    def horizon(self) -> int:
        return self.size
