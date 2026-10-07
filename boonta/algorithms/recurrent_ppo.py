from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import (Timestep, Transition, canonicalize_dtype,
                          remove_feature_axis, remove_time_axis)
from boonta.utils.typing import Array, Key, PyTree

from .advantage_estimators import generalized_advantage_estimation


@struct.dataclass(frozen=True)
class RecurrentPPOConfig:
    num_minibatches: int
    update_epochs: int
    clip_coefficient: float
    clip_value_loss: bool
    entropy_coefficient: float
    value_coefficient: float
    gamma: float
    gae_lambda: float
    normalize_advantage: bool = False


@struct.dataclass(frozen=True)
class RecurrentPPOState:
    step: Array
    carry: PyTree = struct.field(metadata={"axis": "data"})
    rollout_carry: PyTree = struct.field(metadata={"axis": "data"})
    params: PyTree
    optimizer_state: optax.OptState


@dataclass
class RecurrentPPO:
    cfg: RecurrentPPOConfig
    network: nn.Module
    optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> RecurrentPPOState:
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

        return RecurrentPPOState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            carry=carry,
            rollout_carry=carry,
            params=params,
            optimizer_state=optimizer_state,
        )

    def step(
        self, state: RecurrentPPOState, key: Key, timestep: Timestep, temperature: float = 1.0
    ) -> tuple[RecurrentPPOState, Array, PyTree]:
        sequence = timestep.to_sequence()
        carry, (dist, values) = self.network.apply(
            state.params,
            sequence.obs,
            sequence.action,
            sequence.reward,
            sequence.done,
            carry=state.carry,
            temperature=temperature,
        )
        action, log_prob = dist.sample_and_log_prob(seed=key)
        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)
        value = remove_time_axis(remove_feature_axis(values))

        aux = {"log_prob": log_prob, "value": value}
        return state.replace(carry=carry), action, aux

    def update(
        self, state: RecurrentPPOState, key: Key, transitions: Transition
    ) -> RecurrentPPOState:
        def loss_fn(
            params: PyTree, trajectory: Transition, carry: PyTree
        ) -> tuple[Array, tuple]:
            advantages = trajectory.aux["advantages"]
            returns = trajectory.aux["returns"]

            timesteps = trajectory.first
            (_, (dist, values)), intermediates = self.network.apply(
                params,
                timesteps.obs,
                timesteps.action,
                timesteps.reward,
                timesteps.done,
                carry=carry,
                temperature=1.0,
                mutable="intermediates",
            )

            log_probs = dist.log_prob(trajectory.second.action)
            entropy = dist.entropy().mean()
            ratio = jnp.exp(log_probs - trajectory.aux["log_prob"])
            approximate_kl = jnp.mean(trajectory.aux["log_prob"] - log_probs)
            clip_fraction = jnp.mean(
                (jnp.abs(ratio - 1.0) > self.cfg.clip_coefficient).astype(jnp.float32)
            )
            actor_loss = -jnp.minimum(
                ratio * advantages,
                jnp.clip(
                    ratio,
                    1.0 - self.cfg.clip_coefficient,
                    1.0 + self.cfg.clip_coefficient,
                )
                * advantages,
            ).mean()

            values = remove_feature_axis(values)
            critic_loss = 0.5 * ((values - returns) ** 2)
            if self.cfg.clip_value_loss:
                clipped_values = trajectory.aux["value"] + jnp.clip(
                    values - trajectory.aux["value"],
                    -self.cfg.clip_coefficient,
                    self.cfg.clip_coefficient,
                )
                clipped_critic_loss = 0.5 * ((clipped_values - returns) ** 2)
                critic_loss = jnp.maximum(critic_loss, clipped_critic_loss)
            critic_loss = critic_loss.mean()

            loss = (
                actor_loss
                - self.cfg.entropy_coefficient * entropy
                + self.cfg.value_coefficient * critic_loss
            )

            def apply(params: PyTree) -> PyTree:
                _, (dist, _) = self.network.apply(
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
                    value=values,
                    carry=carry,
                    intermediates=intermediates,
                )
            return loss, (
                actor_loss,
                critic_loss,
                entropy,
                approximate_kl,
                clip_fraction,
                values,
            )

        num_steps, num_envs, *_ = transitions.second.reward.shape

        last_timestep = jax.tree.map(
            lambda x: jnp.take(x, -1, axis=0), transitions.second
        )
        sequence = last_timestep.to_sequence()
        _, (_, values) = self.network.apply(
            state.params,
            sequence.obs,
            sequence.action,
            sequence.reward,
            sequence.done,
            carry=state.carry,
            temperature=1.0,
        )
        value = remove_time_axis(remove_feature_axis(values))
        advantages, returns = generalized_advantage_estimation(
            transitions,
            transitions.aux["value"],
            value,
            self.cfg.gamma,
            self.cfg.gae_lambda,
        )
        if self.cfg.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        transitions = transitions.replace(
            aux={**transitions.aux, "advantages": advantages, "returns": returns}
        )
        initial_carry = state.rollout_carry

        def minibatch_fn(
            state: RecurrentPPOState, indices: Array
        ) -> tuple[RecurrentPPOState, None]:
            trajectory = jax.tree.map(
                lambda leaf: jnp.moveaxis(jnp.take(leaf, indices, axis=1), 0, 1),
                transitions,
            )
            carry = jax.tree.map(
                lambda leaf: jnp.take(leaf, indices, axis=0), initial_carry
            )
            returns = trajectory.aux["returns"]
            advantages = trajectory.aux["advantages"]

            (_, aux), grads = jax.value_and_grad(
                loss_fn, has_aux=True, allow_int=True
            )(state.params, trajectory, carry)
            (
                actor_loss,
                critic_loss,
                entropy,
                approximate_kl,
                clip_fraction,
                values,
            ) = aux
            explained_variance = 1 - jnp.var(returns - values) / (
                jnp.var(returns) + 1e-8
            )
            lox.log(
                {
                    "actor/loss": actor_loss,
                    "actor/entropy": entropy,
                    "actor/approximate_kl": approximate_kl,
                    "actor/clip_fraction": clip_fraction,
                    "actor/advantage": advantages.mean(),
                    "critic/loss": critic_loss,
                    "critic/explained_variance": explained_variance,
                    "critic/value": values.mean(),
                }
            )

            updates, optimizer_state = self.optimizer.update(
                grads["params"], state.optimizer_state, state.params["params"]
            )
            params = {
                **state.params,
                "params": optax.apply_updates(state.params["params"], updates),
            }

            state = state.replace(params=params, optimizer_state=optimizer_state)
            return state, None

        def epoch_fn(
            state: RecurrentPPOState, key: Key
        ) -> tuple[RecurrentPPOState, None]:
            permutation = jax.random.permutation(key, num_envs)
            minibatch_indices = permutation.reshape(self.cfg.num_minibatches, -1)
            state, _ = jax.lax.scan(minibatch_fn, state, minibatch_indices)

            return state, None

        keys = jax.random.split(key, self.cfg.update_epochs)
        state, _ = jax.lax.scan(epoch_fn, state, keys)

        return state.replace(rollout_carry=state.carry)
