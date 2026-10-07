from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, PyTree, Timestep

from .wrapper import Wrapper, WrapperState


@struct.dataclass
class PBRSState(WrapperState):
    potential: Array


class PBRS(Wrapper):
    def __init__(
        self,
        env,
        potential: Callable[[PyTree], Array],
        gamma: float,
    ):
        super().__init__(env)
        self.potential = potential
        self.gamma = gamma

    def init(self, key: Key) -> tuple[Any, Timestep]:
        env_state, timestep = self._env.init(key)
        return PBRSState(env_state, self.potential(timestep.obs)), timestep

    def step(self, key: Key, state: Any, action: Array) -> tuple[Any, Timestep]:
        env_state, timestep = self._env.step(key, state.env_state, action)
        potential = self.potential(timestep.obs)
        alive = 1.0 - timestep.terminated.astype(potential.dtype)
        shaped = timestep.reward + self.gamma * alive * potential - state.potential
        return PBRSState(env_state, potential), timestep.replace(reward=shaped)
