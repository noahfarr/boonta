import jax
import jax.numpy as jnp

from boonta.utils import Transition
from boonta.utils.typing import Array


def q_lambda(
    transitions: Transition, q_values: Array, gamma: float, lam: float
) -> Array:
    reward = transitions.second.reward
    terminated = transitions.second.terminated
    truncated = transitions.second.truncated

    def scan_fn(carry: tuple, x: tuple) -> tuple:
        lambda_returns, next_q_value = carry
        reward, terminated, truncated, q_values = x
        td_target = reward + gamma * (1.0 - terminated) * next_q_value
        delta = lambda_returns - next_q_value
        lambda_returns = td_target + gamma * lam * (1.0 - terminated) * delta
        q_value = jnp.max(q_values, axis=-1)
        lambda_returns = jnp.where(truncated, q_value, lambda_returns)
        return (lambda_returns, q_value), lambda_returns

    next_q_value = jnp.max(q_values, axis=-1)
    _, target_q_value = jax.lax.scan(
        scan_fn,
        (next_q_value, next_q_value),
        (reward, terminated, truncated, transitions.aux["q_values"]),
        reverse=True,
    )
    return target_q_value


def _lambda_returns(
    reward: Array,
    value: Array,
    terminated: Array,
    truncated: Array,
    gamma: float,
    td_lambda: float,
) -> Array:
    def scan_fn(carry: tuple, x: tuple) -> tuple[tuple, Array]:
        lambda_return, cut = carry
        reward, value, terminated, truncated = x
        bootstrap = jnp.where(
            cut, value, td_lambda * lambda_return + (1.0 - td_lambda) * value
        )
        lambda_return = reward + gamma * (1.0 - terminated) * bootstrap
        return (lambda_return, truncated), lambda_return

    last_value = jnp.take(value, -1, axis=0)
    _, target_values = jax.lax.scan(
        scan_fn,
        (last_value, jnp.zeros_like(last_value, bool)),
        (reward, value, terminated, truncated),
        reverse=True,
    )
    return target_values


def td_lambda(transitions: Transition, gamma: float, td_lambda: float) -> Array:
    second = transitions.second
    return _lambda_returns(
        second.reward,
        transitions.aux["value"],
        second.terminated,
        second.truncated,
        gamma,
        td_lambda,
    )


def soft_td_lambda(
    transitions: Transition, gamma: float, td_lambda: float
) -> Array:
    second = transitions.second
    return _lambda_returns(
        transitions.aux["soft_reward"],
        transitions.aux["value"],
        second.terminated,
        second.truncated,
        gamma,
        td_lambda,
    )
