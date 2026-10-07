import jax
import jax.numpy as jnp

from boonta.utils import Transition
from boonta.utils.typing import Array


def generalized_advantage_estimation(
    transitions: Transition,
    values: Array,
    bootstrap: Array,
    gamma: float,
    gae_lambda: float,
    importance: Array | float = 1.0,
    trace: Array | float = 1.0,
) -> tuple[Array, Array]:
    rewards = transitions.second.reward
    dtype = jnp.result_type(values, bootstrap, rewards, jnp.float32)
    rewards, values, bootstrap = (
        rewards.astype(dtype),
        values.astype(dtype),
        bootstrap.astype(dtype),
    )
    importance = jnp.broadcast_to(jnp.asarray(importance, dtype), values.shape)
    trace = jnp.broadcast_to(jnp.asarray(trace, dtype), values.shape)

    def scan_fn(carry: tuple, x: tuple) -> tuple:
        advantage, next_value = carry
        reward, terminated, truncated, value, importance, trace = x
        delta = importance * (reward + gamma * (1.0 - terminated) * next_value - value)
        delta *= 1.0 - truncated
        advantage = (
            delta
            + gamma
            * gae_lambda
            * trace
            * (1.0 - terminated)
            * (1.0 - truncated)
            * advantage
        )
        return (advantage, value), advantage

    _, advantages = jax.lax.scan(
        scan_fn,
        (jnp.zeros_like(bootstrap), bootstrap),
        (
            rewards,
            transitions.second.terminated,
            transitions.second.truncated,
            values,
            importance,
            trace,
        ),
        reverse=True,
    )
    return advantages, advantages + values
