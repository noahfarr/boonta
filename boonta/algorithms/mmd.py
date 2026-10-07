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


@struct.dataclass(frozen=True)
class MMDConfig:
    num_minibatches: int
    update_epochs: int
    clip_coefficient: float
    clip_value_loss: bool
    entropy_coefficient: float
    value_coefficient: float
    gamma: float
    gae_lambda: float
    magnet_coefficient: float
    magnet_decay: float
    magnet_anneal: float
    normalize_advantage: bool = False


@struct.dataclass(frozen=True)
class MMDState:
    step: Array
    params: PyTree
    optimizer_state: optax.OptState
    magnet_params: PyTree
    alpha: Array


@dataclass
class MMD:
    cfg: MMDConfig
    network: nn.Module
    optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> MMDState:
        params = self.network.init(key, timestep.obs, temperature=1.0)
        optimizer_state = self.optimizer.init(params["params"])

        return MMDState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            params=params,
            optimizer_state=optimizer_state,
            magnet_params=params,
            alpha=jnp.asarray(self.cfg.magnet_coefficient, dtype=jnp.float32),
        )

    def step(
        self, state: MMDState, key: Key, timestep: Timestep, temperature: float = 1.0
    ) -> tuple[MMDState, Array, PyTree]:
        dist, value = self.network.apply(
            state.params, timestep.obs, temperature=temperature
        )
        action, log_prob = dist.sample_and_log_prob(seed=key)
        return (
            state,
            action,
            {"log_prob": log_prob, "value": remove_feature_axis(value)},
        )

    def update(self, state: MMDState, key: Key, transitions: Transition) -> MMDState:
        def generalized_advantage_estimation(
            transitions: Transition, value: Array
        ) -> tuple[Array, Array]:
            gamma, gae_lambda = self.cfg.gamma, self.cfg.gae_lambda
            values = transitions.aux["value"]

            def scan_fn(carry: tuple, x: tuple) -> tuple:
                advantage, next_value = carry
                reward, terminated, truncated, value = x
                delta = reward + gamma * (1.0 - terminated) * next_value - value
                delta *= 1.0 - truncated
                advantage = (
                    delta
                    + gamma
                    * gae_lambda
                    * (1.0 - terminated)
                    * (1.0 - truncated)
                    * advantage
                )
                return (advantage, value), advantage

            _, advantages = jax.lax.scan(
                scan_fn,
                (jnp.zeros_like(value), value),
                (
                    transitions.second.reward,
                    transitions.second.terminated,
                    transitions.second.truncated,
                    values,
                ),
                reverse=True,
            )
            return advantages, advantages + values

        def loss_fn(
            params: PyTree,
            magnet_params: PyTree,
            alpha: Array,
            transitions: Transition,
        ) -> tuple[Array, tuple]:
            advantages = transitions.aux["advantages"]
            returns = transitions.aux["returns"]

            dist, value = self.network.apply(
                params, transitions.first.obs, temperature=1.0
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

            magnet, _ = self.network.apply(
                magnet_params, transitions.first.obs, temperature=1.0
            )
            log_policy = dist.logits
            log_magnet = jax.lax.stop_gradient(magnet.logits)
            probabilities = jnp.exp(log_policy)
            magnet_kl = jnp.sum(
                probabilities * (log_policy - log_magnet), axis=-1
            ).mean()

            loss = (
                actor_loss
                - self.cfg.entropy_coefficient * entropy
                + self.cfg.value_coefficient * critic_loss
                + alpha * magnet_kl
            )
            for auxiliary_loss in self.auxiliary_losses:
                loss = loss + auxiliary_loss(
                    params=params, transitions=transitions, dist=dist, value=value
                )
            return loss, (
                actor_loss,
                critic_loss,
                entropy,
                approximate_kl,
                clip_fraction,
                magnet_kl,
                value,
            )

        num_steps, num_envs, *_ = transitions.second.reward.shape
        batch_size = num_steps * num_envs

        obs = jax.tree.map(
            lambda leaf: jnp.take(leaf, -1, axis=0), transitions.second.obs
        )
        _, value = self.network.apply(state.params, obs, temperature=1.0)
        value = remove_feature_axis(value)
        advantages, returns = generalized_advantage_estimation(transitions, value)
        if self.cfg.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        transitions = transitions.replace(
            aux={**transitions.aux, "advantages": advantages, "returns": returns}
        )

        transitions = jax.tree.map(
            lambda x: flatten(x, start_dim=0, end_dim=1), transitions
        )

        def minibatch_fn(state: MMDState, indices: Array) -> tuple[MMDState, tuple]:
            minibatch = jax.tree.map(
                lambda leaf: jnp.take(leaf, indices, axis=0), transitions
            )
            returns = minibatch.aux["returns"]
            advantages = minibatch.aux["advantages"]

            (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, state.magnet_params, state.alpha, minibatch
            )
            (
                actor_loss,
                critic_loss,
                entropy,
                approximate_kl,
                clip_fraction,
                magnet_kl,
                value,
            ) = aux
            explained_variance = 1 - jnp.var(returns - value) / (
                jnp.var(returns) + 1e-8
            )
            lox.log(
                {
                    "actor/loss": actor_loss,
                    "actor/entropy": entropy,
                    "actor/approximate_kl": approximate_kl,
                    "actor/clip_fraction": clip_fraction,
                    "actor/advantage": advantages.mean(),
                    "actor/magnet_kl": magnet_kl,
                    "actor/magnet_coefficient": state.alpha,
                    "critic/loss": critic_loss,
                    "critic/explained_variance": explained_variance,
                    "critic/value": value.mean(),
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

        magnet_params = jax.tree.map(
            lambda magnet, param: self.cfg.magnet_decay * magnet
            + (1.0 - self.cfg.magnet_decay) * param,
            state.magnet_params,
            state.params,
        )
        alpha = state.alpha * self.cfg.magnet_anneal
        return state.replace(magnet_params=magnet_params, alpha=alpha)
