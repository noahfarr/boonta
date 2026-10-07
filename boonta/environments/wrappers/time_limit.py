import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, Timestep

from .wrapper import Wrapper, WrapperState


@struct.dataclass
class TimeLimitState(WrapperState):
    time: Array


class TimeLimit(Wrapper):
    def __init__(self, env, time_limit: int):
        super().__init__(env)
        self._time_limit = time_limit

    def init(self, key: Key) -> tuple[TimeLimitState, Timestep]:
        env_state, timestep = self._env.init(key)
        return TimeLimitState(env_state, jnp.int32(0)), timestep

    def step(
        self, key: Key, state: TimeLimitState, action: Array
    ) -> tuple[TimeLimitState, Timestep]:
        env_state, timestep = self._env.step(key, state.env_state, action)
        time = state.time + 1
        truncated = timestep.truncated | (
            (time >= self._time_limit) & ~timestep.terminated
        )
        state = TimeLimitState(env_state, time)
        return state, timestep.replace(truncated=truncated)

    def time_limit(self) -> int:
        return self._time_limit
