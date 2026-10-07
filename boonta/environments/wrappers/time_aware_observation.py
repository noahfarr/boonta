import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, Timestep

from ..spaces import Space
from .wrapper import Wrapper, WrapperState


@struct.dataclass
class TimeAwareObservationState(WrapperState):
    time: Array


class TimeAwareObservation(Wrapper):

    def __init__(self, env, time_limit: int):
        super().__init__(env)
        self.max_steps = time_limit

    def observation_space(self) -> Space:
        space = self._env.observation_space()
        return Space(
            shape=(space.shape[0] + 1,), dtype=space.dtype, low=-jnp.inf, high=jnp.inf
        )

    def append_time(self, obs: Array, time: Array) -> Array:
        feature = time / self.max_steps - 0.5
        return jnp.concatenate([obs, feature[None].astype(obs.dtype)])

    def init(
        self, key: Key
    ) -> tuple[TimeAwareObservationState, Timestep]:
        env_state, timestep = self._env.init(key)
        time = jnp.int32(0)
        state = TimeAwareObservationState(env_state, time)
        return state, timestep.replace(obs=self.append_time(timestep.obs, time))

    def step(
        self,
        key: Key,
        state: TimeAwareObservationState,
        action: Array,
    ) -> tuple[TimeAwareObservationState, Timestep]:
        env_state, timestep = self._env.step(key, state.env_state, action)
        time = jnp.where(timestep.done, 0, state.time + 1)
        state = TimeAwareObservationState(env_state, time)
        return state, timestep.replace(obs=self.append_time(timestep.obs, time))
