from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import (
    Timestep,
    Transition,
    add_feature_axis,
    canonicalize_dtype,
    flatten,
    remove_feature_axis,
)
from boonta.utils.typing import Array, Key, PyTree


@struct.dataclass(frozen=True)
class PQNConfig:
    num_minibatches: int
    update_epochs: int
    gamma: float
    q_lambda: float


@struct.dataclass(frozen=True)
class PQNState:
    step: Array
    params: PyTree
    optimizer_state: optax.OptState


@dataclass
class PQN:
    cfg: PQNConfig
    network: nn.Module
    exploration_schedule: optax.Schedule
    optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> PQNState:
        params = self.network.init(key, timestep.obs, temperature=1.0)
        optimizer_state = self.optimizer.init(params["params"])

        return PQNState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            params=params,
            optimizer_state=optimizer_state,
        )

    def step(
        self, state: PQNState, key: Key, timestep: Timestep, temperature: float = 1.0
    ) -> tuple[PQNState, Array, PyTree]:
        dist = self.network.apply(
            state.params,
            timestep.obs,
            temperature=self.exploration_schedule(state.step) * temperature,
        )
        action = dist.sample(seed=key)
        return state, action, {"q_values": dist.preferences}

    def update(self, state: PQNState, key: Key, transitions: Transition) -> PQNState:
        def q_lambda(transitions: Transition, q_values: Array) -> Array:
            gamma, q_lambda_ = self.cfg.gamma, self.cfg.q_lambda
            reward = transitions.second.reward
            terminated = transitions.second.terminated
            truncated = transitions.second.truncated

            def scan_fn(carry: tuple, x: tuple) -> tuple:
                lambda_returns, next_q_value = carry
                reward, terminated, truncated, q_values = x
                td_target = reward + gamma * (1.0 - terminated) * next_q_value
                delta = lambda_returns - next_q_value
                lambda_returns = (
                    td_target + gamma * q_lambda_ * (1.0 - terminated) * delta
                )
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

        def loss_fn(params: PyTree, transitions: Transition) -> tuple[Array, Array]:
            q_values = self.network.apply(
                params, transitions.first.obs, temperature=1.0
            ).preferences
            q_value = remove_feature_axis(
                jnp.take_along_axis(
                    q_values, add_feature_axis(transitions.second.action), axis=-1
                )
            )
            target_q_value = transitions.aux["target_q_value"]
            td_error = (q_value - target_q_value) * (1.0 - transitions.second.truncated)
            loss = 0.5 * (td_error**2).mean()
            for auxiliary_loss in self.auxiliary_losses:
                loss = loss + auxiliary_loss(
                    params=params, transitions=transitions, q_values=q_values
                )
            return loss, q_value

        num_steps, num_envs = transitions.second.reward.shape
        batch_size = num_steps * num_envs

        *_, obs = transitions.second.obs
        q_values = self.network.apply(
            state.params, obs, temperature=1.0
        ).preferences
        target_q_value = q_lambda(transitions, q_values)
        transitions = transitions.replace(
            aux={**transitions.aux, "target_q_value": target_q_value}
        )

        transitions = jax.tree.map(
            lambda x: flatten(x, start_dim=0, end_dim=1), transitions
        )

        def minibatch_fn(state: PQNState, indices: Array) -> tuple[PQNState, tuple]:
            minibatch = jax.tree.map(
                lambda leaf: jnp.take(leaf, indices, axis=0), transitions
            )
            target_q_value = minibatch.aux["target_q_value"]

            (loss, q_value), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, minibatch
            )
            explained_variance = 1 - jnp.var(target_q_value - q_value) / (
                jnp.var(target_q_value) + 1e-8
            )
            lox.log(
                {
                    "critic/loss": loss,
                    "critic/explained_variance": explained_variance,
                    "critic/q_value": q_value.mean(),
                }
            )

            updates, optimizer_state = self.optimizer.update(
                grads["params"], state.optimizer_state, state.params["params"]
            )
            params = {
                **state.params,
                "params": optax.apply_updates(state.params["params"], updates),
            }

            state = state.replace(
                params=params,
                optimizer_state=optimizer_state,
            )
            return state, None

        def epoch_fn(state, key):
            permutation = jax.random.permutation(key, batch_size)
            minibatch_indices = permutation.reshape(self.cfg.num_minibatches, -1)

            state, _ = jax.lax.scan(minibatch_fn, state, minibatch_indices)

            return state, None

        keys = jax.random.split(key, self.cfg.update_epochs)
        state, _ = jax.lax.scan(epoch_fn, state, keys)

        return state
