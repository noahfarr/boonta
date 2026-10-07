import jax.numpy as jnp
import lox
from flax import struct

from boonta.utils import Array, Key, Timestep

from .wrapper import Wrapper, WrapperState


@struct.dataclass
class RecordEpisodeStatisticsState(WrapperState):
    episode_returns: float
    discounted_episode_returns: float
    episode_discount: float
    episode_lengths: int


class RecordEpisodeStatistics(Wrapper):
    def __init__(self, env, gamma: float = 0.99):
        super().__init__(env)
        self._gamma = gamma

    def init(
        self, key: Key
    ) -> tuple[RecordEpisodeStatisticsState, Timestep]:
        env_state, timestep = self._env.init(key)
        zeros = jnp.zeros_like(timestep.reward)
        state = RecordEpisodeStatisticsState(
            env_state, zeros, zeros, jnp.ones_like(zeros), zeros.astype(jnp.int32)
        )
        return state, timestep

    def step(
        self,
        key: Key,
        state: RecordEpisodeStatisticsState,
        action: Array,
    ) -> tuple[RecordEpisodeStatisticsState, Timestep]:
        env_state, timestep = self._env.step(key, state.env_state, action)
        done = timestep.done
        new_episode_return = state.episode_returns + timestep.reward
        new_discounted_episode_return = (
            state.discounted_episode_returns + state.episode_discount * timestep.reward
        )
        new_episode_discount = state.episode_discount * self._gamma
        new_episode_length = state.episode_lengths + 1
        state = RecordEpisodeStatisticsState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            discounted_episode_returns=new_discounted_episode_return * (1 - done),
            episode_discount=new_episode_discount * (1 - done) + done,
            episode_lengths=new_episode_length * (1 - done),
        )
        lox.log(
            {
                "episode_statistics/episode_return": jnp.where(
                    done, new_episode_return, jnp.nan
                ),
                "episode_statistics/discounted_episode_return": jnp.where(
                    done, new_discounted_episode_return, jnp.nan
                ),
                "episode_statistics/episode_length": jnp.where(
                    done, new_episode_length, jnp.nan
                ),
            }
        )
        return state, timestep
