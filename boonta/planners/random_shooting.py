from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils.typing import Array, Key

from .planner import RolloutFn, SampleFn


@struct.dataclass(frozen=True)
class RandomShootingState:
    action: Array


@dataclass
class RandomShooting:
    action_shape: tuple[int, ...]
    population_size: int

    def init(self, key: Key) -> RandomShootingState:
        return RandomShootingState(action=jnp.zeros(self.action_shape))

    def reset(self, state: RandomShootingState, done: Array) -> RandomShootingState:
        return state

    def plan(
        self,
        key: Key,
        state: RandomShootingState,
        rollout_fn: RolloutFn,
        sample_fn: SampleFn,
    ) -> RandomShootingState:
        sample_key, eval_key = jax.random.split(key)

        actions = sample_fn(
            sample_key,
            jnp.zeros(self.action_shape),
            jnp.ones(self.action_shape),
            self.population_size,
        )
        returns = rollout_fn(eval_key, actions)

        best_idx = jnp.argmax(returns, axis=-1, keepdims=True)
        best_actions = jnp.take_along_axis(
            actions, jnp.expand_dims(best_idx, axis=(-2, -1)), axis=-3
        )
        best_action = jax.lax.index_in_dim(best_actions, 0, axis=-3, keepdims=False)
        return RandomShootingState(action=best_action)

    def sample(self, key: Key, state: RandomShootingState, temperature: float) -> Array:
        return jax.lax.index_in_dim(state.action, 0, axis=-2, keepdims=False)
