from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import (Timestep, Transition, add_feature_axis,
                          canonicalize_dtype, remove_feature_axis,
                          remove_time_axis)
from boonta.utils.typing import Array, Key, PyTree


@struct.dataclass(frozen=True)
class RecurrentPQNConfig:
    num_minibatches: int
    update_epochs: int
    gamma: float
    q_lambda: float


@struct.dataclass(frozen=True)
class RecurrentPQNState:
    step: Array
    carry: PyTree = struct.field(metadata={"axis": "data"})
    rollout_carry: PyTree = struct.field(metadata={"axis": "data"})
    params: PyTree
    optimizer_state: optax.OptState


@dataclass
class RecurrentPQN:
    cfg: RecurrentPQNConfig
    network: nn.Module
    exploration_schedule: optax.Schedule
    optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> RecurrentPQNState:
        network_key, carry_key = jax.random.split(key)

        carry = self.network.initialize_carry(
            carry_key, (*timestep.reward.shape, 1)
        )
        sequence = timestep.to_sequence()
        params = self.network.init(
            network_key,
            sequence.obs,
            sequence.action,
            sequence.reward,
            sequence.done,
            carry=carry,
            temperature=1.0,
        )
        optimizer_state = self.optimizer.init(params["params"])

        return RecurrentPQNState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            carry=carry,
            rollout_carry=carry,
            params=params,
            optimizer_state=optimizer_state,
        )

    def step(
        self, state: RecurrentPQNState, key: Key, timestep: Timestep, temperature: float = 1.0
    ) -> tuple[RecurrentPQNState, Array, PyTree]:
        sequence = timestep.to_sequence()
        carry, dist = self.network.apply(
            state.params,
            sequence.obs,
            sequence.action,
            sequence.reward,
            sequence.done,
            carry=state.carry,
            temperature=self.exploration_schedule(state.step) * temperature,
        )
        action = remove_time_axis(dist.sample(seed=key))

        aux = {"q_values": remove_time_axis(dist.preferences)}
        return state.replace(carry=carry), action, aux

    def update(
        self, state: RecurrentPQNState, key: Key, transitions: Transition
    ) -> RecurrentPQNState:
        def q_lambda(trajectory: Transition, q_values: Array) -> Array:
            gamma, q_lambda_ = self.cfg.gamma, self.cfg.q_lambda
            reward = trajectory.second.reward
            terminated = trajectory.second.terminated
            truncated = trajectory.second.truncated

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
                (reward, terminated, truncated, trajectory.aux["q_values"]),
                reverse=True,
            )
            return target_q_value

        def loss_fn(
            params: PyTree, trajectory: Transition, carry: PyTree
        ) -> tuple[Array, tuple[PyTree, Array]]:
            timesteps = trajectory.first
            (_, dist), variables = self.network.apply(
                params,
                timesteps.obs,
                timesteps.action,
                timesteps.reward,
                timesteps.done,
                carry=carry,
                temperature=1.0,
                mutable=True,
            )
            q_values = dist.preferences
            q_value = remove_feature_axis(
                jnp.take_along_axis(
                    q_values, add_feature_axis(trajectory.second.action), axis=-1
                )
            )
            target_q_value = trajectory.aux["target_q_value"]
            td_error = (q_value - target_q_value) * (1.0 - trajectory.second.truncated)
            loss = 0.5 * (td_error**2).mean()

            def apply(params: PyTree) -> PyTree:
                _, dist = self.network.apply(
                    params,
                    timesteps.obs,
                    timesteps.action,
                    timesteps.reward,
                    timesteps.done,
                    carry=carry,
                    temperature=1.0,
                )
                return dist

            for auxiliary_loss in self.auxiliary_losses:
                loss = loss + auxiliary_loss(
                    params=params,
                    apply=apply,
                    transitions=trajectory,
                    dist=dist,
                    q_values=q_values,
                    carry=carry,
                    variables=variables,
                )
            return loss, (variables, q_value)

        num_steps, num_envs = transitions.second.reward.shape

        last_timestep = jax.tree.map(
            lambda x: jnp.take(x, -1, axis=0), transitions.second
        )
        sequence = last_timestep.to_sequence()
        _, dist = self.network.apply(
            state.params,
            sequence.obs,
            sequence.action,
            sequence.reward,
            sequence.done,
            carry=state.carry,
            temperature=1.0,
        )
        q_values = remove_time_axis(dist.preferences)
        target_q_value = q_lambda(transitions, q_values)
        transitions = transitions.replace(
            aux={**transitions.aux, "target_q_value": target_q_value}
        )
        initial_carry = state.rollout_carry

        def minibatch_fn(
            state: RecurrentPQNState, indices: Array
        ) -> tuple[RecurrentPQNState, tuple]:
            trajectory = jax.tree.map(
                lambda leaf: jnp.moveaxis(jnp.take(leaf, indices, axis=1), 0, 1),
                transitions,
            )
            carry = jax.tree.map(
                lambda leaf: jnp.take(leaf, indices, axis=0), initial_carry
            )
            target_q_value = trajectory.aux["target_q_value"]

            (loss, (returned, q_value)), grads = jax.value_and_grad(
                loss_fn, has_aux=True, allow_int=True
            )(state.params, trajectory, carry)
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
            variables = {
                name: returned.get(name, value) for name, value in state.params.items()
            }
            params = {
                **variables,
                "params": optax.apply_updates(state.params["params"], updates),
            }

            state = state.replace(
                params=params,
                optimizer_state=optimizer_state,
            )
            return state, None

        def epoch_fn(state, key):
            permutation = jax.random.permutation(key, num_envs)
            minibatch_indices = permutation.reshape(self.cfg.num_minibatches, -1)
            state, _ = jax.lax.scan(minibatch_fn, state, minibatch_indices)

            return state, None

        keys = jax.random.split(key, self.cfg.update_epochs)
        state, _ = jax.lax.scan(epoch_fn, state, keys)

        return state.replace(rollout_carry=state.carry)
