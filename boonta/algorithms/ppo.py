from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import Timestep, Transition, canonicalize_dtype, flatten, remove_feature_axis
from boonta.utils.typing import Array, Key, PyTree

from .advantage_estimators import generalized_advantage_estimation


@struct.dataclass(frozen=True)
class PPOConfig:
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
class PPOState:
    step: Array
    params: PyTree
    optimizer_state: optax.OptState


@dataclass
class PPO:
    cfg: PPOConfig
    network: nn.Module
    optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> PPOState:
        params = self.network.init(key, timestep.obs, temperature=1.0)
        optimizer_state = self.optimizer.init(params["params"])

        return PPOState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            params=params,
            optimizer_state=optimizer_state,
        )

    def step(
        self, state: PPOState, key: Key, timestep: Timestep, temperature: float = 1.0
    ) -> tuple[PPOState, Array, PyTree]:
        dist, value = self.network.apply(
            state.params, timestep.obs, temperature=temperature
        )
        action, log_prob = dist.sample_and_log_prob(seed=key)
        return (
            state,
            action,
            {"log_prob": log_prob, "value": remove_feature_axis(value)},
        )

    def update(self, state: PPOState, key: Key, transitions: Transition) -> PPOState:
        def loss_fn(params: PyTree, transitions: Transition) -> tuple[Array, tuple]:
            advantages = transitions.aux["advantages"]
            returns = transitions.aux["returns"]

            (dist, value), intermediates = self.network.apply(
                params,
                transitions.first.obs,
                temperature=1.0,
                mutable="intermediates",
            )

            log_probs = dist.log_prob(transitions.second.action)
            entropy = dist.entropy().mean()
            ratio = jnp.exp(log_probs - transitions.aux["log_prob"])
            approximate_kl = jnp.mean(transitions.aux["log_prob"] - log_probs)
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

            value = remove_feature_axis(value)
            critic_loss = 0.5 * ((value - returns) ** 2)
            if self.cfg.clip_value_loss:
                clipped_value = transitions.aux["value"] + jnp.clip(
                    value - transitions.aux["value"],
                    -self.cfg.clip_coefficient,
                    self.cfg.clip_coefficient,
                )
                clipped_critic_loss = 0.5 * ((clipped_value - returns) ** 2)
                critic_loss = jnp.maximum(critic_loss, clipped_critic_loss)
            critic_loss = critic_loss.mean()

            loss = (
                actor_loss
                - self.cfg.entropy_coefficient * entropy
                + self.cfg.value_coefficient * critic_loss
            )

            def apply(params: PyTree) -> PyTree:
                dist, _ = self.network.apply(
                    params, transitions.first.obs, temperature=1.0
                )
                return dist

            for auxiliary_loss in self.auxiliary_losses:
                loss = loss + auxiliary_loss(
                    params=params,
                    apply=apply,
                    transitions=transitions,
                    dist=dist,
                    value=value,
                    intermediates=intermediates,
                )
            return loss, (
                actor_loss,
                critic_loss,
                entropy,
                approximate_kl,
                clip_fraction,
                value,
            )

        def minibatch_fn(state: PPOState, indices: Array) -> tuple[PPOState, None]:
            minibatch = jax.tree.map(
                lambda leaf: jnp.take(leaf, indices, axis=0), transitions
            )
            returns = minibatch.aux["returns"]
            advantages = minibatch.aux["advantages"]

            (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, minibatch
            )
            (
                actor_loss,
                critic_loss,
                entropy,
                approximate_kl,
                clip_fraction,
                value,
            ) = aux
            explained_variance = 1 - jnp.var(returns - value) / (
                jnp.var(returns) + 1e-8
            )
            logs = {
                "actor/loss": actor_loss,
                "actor/entropy": entropy,
                "actor/approximate_kl": approximate_kl,
                "actor/clip_fraction": clip_fraction,
                "actor/advantage": advantages.mean(),
                "critic/loss": critic_loss,
                "critic/explained_variance": explained_variance,
                "critic/value": value.mean(),
            }
            lox.log(logs)

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

        def epoch_fn(state: PPOState, key: Key) -> tuple[PPOState, None]:
            permutation = jax.random.permutation(key, batch_size)
            minibatch_indices = permutation.reshape(self.cfg.num_minibatches, -1)
            state, _ = jax.lax.scan(minibatch_fn, state, minibatch_indices)
            return state, None

        num_steps, num_envs, *_ = transitions.second.reward.shape
        batch_size = num_steps * num_envs

        obs = jax.tree.map(
            lambda leaf: jnp.take(leaf, -1, axis=0), transitions.second.obs
        )
        _, value = self.network.apply(state.params, obs, temperature=1.0)
        value = remove_feature_axis(value)
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

        transitions = jax.tree.map(
            lambda x: flatten(x, start_dim=0, end_dim=1), transitions
        )

        keys = jax.random.split(key, self.cfg.update_epochs)
        state, _ = jax.lax.scan(epoch_fn, state, keys)

        return state
