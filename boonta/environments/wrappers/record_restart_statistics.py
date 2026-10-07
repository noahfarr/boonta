import jax.numpy as jnp
import lox
from flax import struct

from boonta.utils import Array, Key, Timestep

from .wrapper import Wrapper, WrapperState


@struct.dataclass
class RecordRestartStatisticsState(WrapperState):
    episode_returns: Array
    episode_lengths: Array
    visited: Array


class RecordRestartStatistics(Wrapper):
    def __init__(self, env, room_byte: int = 3):
        super().__init__(env)
        self._room_byte = room_byte

    def init(self, key: Key) -> tuple[RecordRestartStatisticsState, Timestep]:
        env_state, timestep = self._env.init(key)
        zeros = jnp.zeros_like(timestep.reward)
        state = RecordRestartStatisticsState(
            env_state,
            zeros,
            zeros.astype(jnp.int32),
            jnp.zeros(256, bool).at[timestep.obs[:, self._room_byte]].set(True),
        )
        return state, timestep

    def step(
        self,
        key: Key,
        state: RecordRestartStatisticsState,
        action: Array,
    ) -> tuple[RecordRestartStatisticsState, Timestep]:
        env_state, timestep = self._env.step(key, state.env_state, action)
        done = timestep.done
        warm = timestep.info["warm"]
        returns = state.episode_returns + timestep.reward
        lengths = state.episode_lengths + 1
        visited = state.visited.at[timestep.obs[:, self._room_byte]].set(True)
        cold = done & ~warm
        lox.log(
            {
                "episode_statistics/cold_return": jnp.where(cold, returns, jnp.nan),
                "episode_statistics/cold_length": jnp.where(
                    cold, lengths.astype(jnp.float32), jnp.nan
                ),
                "episode_statistics/warm_return": jnp.where(
                    done & warm, returns, jnp.nan
                ),
                "episode_statistics/rooms": jnp.full_like(
                    returns, visited.sum(), dtype=jnp.float32
                ),
            }
        )
        state = RecordRestartStatisticsState(
            env_state=env_state,
            episode_returns=returns * (1 - done),
            episode_lengths=lengths * (1 - done),
            visited=visited,
        )
        return state, timestep
