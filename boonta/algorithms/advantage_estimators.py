import jax
import jax.numpy as jnp

from boonta.utils import Transition
from boonta.utils.typing import Array


def generalized_advantage_estimator(
    transitions: Transition,
    values: Array,
    bootstrap: Array,
    gamma: float,
    gae_lambda: float,
    rho: Array | float = 1.0,
    c: Array | float = 1.0,
) -> tuple[Array, Array]:
    rewards = transitions.second.reward
    dtype = jnp.result_type(values, bootstrap, rewards, jnp.float32)
    rewards, values, bootstrap = (
        rewards.astype(dtype),
        values.astype(dtype),
        bootstrap.astype(dtype),
    )
    rho = jnp.broadcast_to(jnp.asarray(rho, dtype), values.shape)
    c = jnp.broadcast_to(jnp.asarray(c, dtype), values.shape)

    def scan_fn(carry: tuple, x: tuple) -> tuple:
        advantage, next_value = carry
        reward, terminated, truncated, value, rho, c = x
        delta = rho * (reward + gamma * (1.0 - terminated) * next_value - value)
        delta *= 1.0 - truncated
        advantage = (
            delta
            + gamma
            * gae_lambda
            * c
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
            rho,
            c,
        ),
        reverse=True,
    )
    return advantages, advantages + values


def group_relative_advantage_estimator(
    transitions: Transition, gamma: float, group_size: int
) -> tuple[Array, Array]:
    num_steps, num_envs = transitions.second.reward.shape
    rewards = transitions.second.reward
    dones = transitions.second.done

    episode_mask = (jnp.cumsum(dones, axis=0) - dones == 0).astype(rewards.dtype)
    discounts = gamma ** jnp.arange(num_steps, dtype=rewards.dtype)
    returns = discounts @ (rewards * episode_mask)

    groups = returns.reshape(num_envs // group_size, group_size)
    mean = groups.mean(axis=1, keepdims=True)
    std = groups.std(axis=1, keepdims=True)
    advantages = ((groups - mean) / (std + 1e-8)).reshape(-1)
    return jnp.broadcast_to(advantages, episode_mask.shape), episode_mask
