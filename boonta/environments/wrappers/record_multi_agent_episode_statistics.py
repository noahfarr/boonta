import jax.numpy as jnp
import lox
from flax import struct

from boonta.utils import Array, Key, Timestep

from .wrapper import Wrapper, WrapperState


@struct.dataclass
class RecordMultiAgentEpisodeStatisticsState(WrapperState):
    episode_returns: Array
    discounted_episode_returns: Array
    episode_discount: Array
    episode_lengths: Array


class RecordMultiAgentEpisodeStatistics(Wrapper):
    def __init__(self, env, gamma: float = 0.99):
        super().__init__(env)
        self._gamma = gamma

    def init(
        self, key: Key
    ) -> tuple[RecordMultiAgentEpisodeStatisticsState, Timestep]:
        env_state, timestep = self._env.init(key)
        returns = jnp.zeros_like(timestep.reward)
        batch_size = timestep.reward.shape[:-1]
        state = RecordMultiAgentEpisodeStatisticsState(
            env_state,
            returns,
            returns,
            jnp.ones(batch_size),
            jnp.zeros(batch_size, jnp.int32),
        )
        return state, timestep

    def step(
        self,
        key: Key,
        state: RecordMultiAgentEpisodeStatisticsState,
        action: Array,
    ) -> tuple[RecordMultiAgentEpisodeStatisticsState, Timestep]:
        env_state, timestep = self._env.step(key, state.env_state, action)
        done = timestep.done.all(axis=-1)
        new_episode_return = state.episode_returns + timestep.reward
        new_discounted_episode_return = (
            state.discounted_episode_returns
            + state.episode_discount[..., None] * timestep.reward
        )
        new_episode_discount = state.episode_discount * self._gamma
        new_episode_length = state.episode_lengths + 1
        state = RecordMultiAgentEpisodeStatisticsState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done[..., None]),
            discounted_episode_returns=new_discounted_episode_return
            * (1 - done[..., None]),
            episode_discount=new_episode_discount * (1 - done) + done,
            episode_lengths=new_episode_length * (1 - done),
        )
        logs = {
            "episode_statistics/episode_return": jnp.where(
                done, new_episode_return.mean(axis=-1), jnp.nan
            ),
            "episode_statistics/discounted_episode_return": jnp.where(
                done, new_discounted_episode_return.mean(axis=-1), jnp.nan
            ),
            "episode_statistics/episode_length": jnp.where(
                done, new_episode_length, jnp.nan
            ),
        }
        for agent in range(self._env.num_agents):
            logs[f"episode_statistics/agent_{agent}/episode_return"] = jnp.where(
                done, new_episode_return[..., agent], jnp.nan
            )
            logs[f"episode_statistics/agent_{agent}/discounted_episode_return"] = (
                jnp.where(done, new_discounted_episode_return[..., agent], jnp.nan)
            )
        lox.log(logs)
        return state, timestep
