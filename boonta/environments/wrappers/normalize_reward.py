import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, Timestep

from .vectorize import Vectorize
from .wrapper import Wrapper, WrapperState


@struct.dataclass
class NormalizeRewardState(WrapperState):
    returns: Array
    mean: Array
    m2: Array
    count: Array


class NormalizeReward(Wrapper):
    def __init__(
        self,
        env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        super().__init__(env)
        assert self.wraps(Vectorize), (
            "NormalizeReward computes statistics over axis 0, so env must have "
            "a Vectorize wrapper somewhere underneath it."
        )
        self.gamma = gamma
        self.epsilon = epsilon

    def merge(
        self, returns: Array, mean: Array, m2: Array, count: Array
    ) -> tuple[Array, Array, Array]:
        count = count + 1
        delta = returns - mean
        mean = mean + delta / count
        m2 = m2 + delta * (returns - mean)
        return mean, m2, count

    def variance(self, mean: Array, m2: Array, count: Array) -> Array:
        batch_count = count.sum(axis=0)
        batch_mean = (count * mean).sum(axis=0) / batch_count
        batch_m2 = m2.sum(axis=0) + (count * (mean - batch_mean) ** 2).sum(axis=0)
        return batch_m2 / batch_count

    def init(
        self, key: Key
    ) -> tuple[NormalizeRewardState, Timestep]:
        env_state, timestep = self._env.init(key)
        zeros = jnp.zeros(timestep.reward.shape, jnp.float32)
        return NormalizeRewardState(env_state, zeros, zeros, zeros, zeros), timestep

    def step(
        self,
        key: Key,
        state: NormalizeRewardState,
        action: Array,
    ) -> tuple[NormalizeRewardState, Timestep]:
        env_state, timestep = self._env.step(key, state.env_state, action)
        returns = state.returns * self.gamma * (1 - timestep.done) + timestep.reward
        mean, m2, count = self.merge(returns, state.mean, state.m2, state.count)
        state = NormalizeRewardState(env_state, returns, mean, m2, count)
        variance = self.variance(mean, m2, count)
        reward = timestep.reward / jnp.sqrt(variance + self.epsilon)
        return state, timestep.replace(reward=reward)
