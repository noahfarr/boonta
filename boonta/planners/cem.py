from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils.typing import Array, Key

from .planner import RolloutFn, SampleFn


@struct.dataclass(frozen=True)
class CEMState:
    mean: Array
    std: Array


@dataclass
class CEM:
    action_shape: tuple[int, ...]
    population_size: int
    num_elites: int
    num_iterations: int
    min_std: float
    max_std: float

    def init(self, key: Key) -> CEMState:
        return CEMState(
            mean=jnp.zeros(self.action_shape),
            std=jnp.full(self.action_shape, self.max_std),
        )

    def reset(self, state: CEMState, done: Array) -> CEMState:
        shifted_mean = jnp.roll(state.mean, shift=-1, axis=-2).at[..., -1, :].set(0.0)
        expanded_done = done.reshape(
            done.shape + (1,) * (shifted_mean.ndim - done.ndim)
        )
        mean = jnp.where(expanded_done, jnp.zeros_like(shifted_mean), shifted_mean)
        std = jnp.full_like(state.std, self.max_std)
        return CEMState(mean=mean, std=std)

    def plan(
        self, key: Key, state: CEMState, rollout_fn: RolloutFn, sample_fn: SampleFn
    ) -> CEMState:
        def iteration(
            carry: tuple[Array, Array], iteration_key: Key
        ) -> tuple[tuple[Array, Array], None]:
            mean, std = carry
            sample_key, eval_key = jax.random.split(iteration_key)

            actions = sample_fn(sample_key, mean, std, self.population_size)
            returns = rollout_fn(eval_key, actions)

            _, elite_idx = jax.lax.top_k(returns, self.num_elites)
            elite_actions = jnp.take_along_axis(
                actions, jnp.expand_dims(elite_idx, axis=(-2, -1)), axis=-3
            )

            new_mean = jnp.mean(elite_actions, axis=-3)
            new_std = jnp.clip(
                jnp.std(elite_actions, axis=-3), self.min_std, self.max_std
            )
            return (new_mean, new_std), None

        (mean, std), _ = jax.lax.scan(
            iteration,
            (state.mean, state.std),
            jax.random.split(key, self.num_iterations),
        )
        return CEMState(mean=mean, std=std)

    def sample(self, key: Key, state: CEMState, temperature: float) -> Array:
        mean = jax.lax.index_in_dim(state.mean, 0, axis=-2, keepdims=False)
        std = jax.lax.index_in_dim(state.std, 0, axis=-2, keepdims=False)
        noise = jax.random.normal(key, mean.shape)
        return mean + temperature * std * noise
