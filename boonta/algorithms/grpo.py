from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import Timestep, Transition, canonicalize_dtype
from boonta.utils.typing import Array, Key, PyTree


@struct.dataclass(frozen=True)
class GRPOConfig:
    group_size: int
    num_minibatches: int
    update_epochs: int
    clip_coefficient: float
    kl_coefficient: float
    gamma: float


@struct.dataclass(frozen=True)
class GRPOState:
    step: Array
    params: PyTree
    reference_params: PyTree
    optimizer_state: optax.OptState


@dataclass
class GRPO:
    cfg: GRPOConfig
    network: nn.Module
    optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> GRPOState:
        params = self.network.init(key, timestep.obs, temperature=1.0)
        return GRPOState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            params=params,
            reference_params=params,
            optimizer_state=self.optimizer.init(params["params"]),
        )

    def step(
        self, state: GRPOState, key: Key, timestep: Timestep, temperature: float = 1.0
    ) -> tuple[GRPOState, Array, PyTree]:
        dist = self.network.apply(state.params, timestep.obs, temperature=temperature)
        action, log_prob = dist.sample_and_log_prob(seed=key)
        return state, action, {"log_prob": log_prob}

    def update(self, state: GRPOState, key: Key, transitions: Transition) -> GRPOState:
        num_steps, num_envs = transitions.second.reward.shape
        num_groups = num_envs // self.cfg.group_size
        batch_size = num_steps * num_envs

        def group_relative_advantage_estimation(
            transitions: Transition,
        ) -> tuple[Array, Array]:
            rewards = transitions.second.reward
            dones = transitions.second.done

            valid = (jnp.cumsum(dones, axis=0) - dones == 0).astype(rewards.dtype)
            discounts = self.cfg.gamma ** jnp.arange(num_steps, dtype=rewards.dtype)
            returns = discounts @ (rewards * valid)

            groups = returns.reshape(num_groups, self.cfg.group_size)
            mean = groups.mean(axis=1, keepdims=True)
            std = groups.std(axis=1, keepdims=True)
            advantages = ((groups - mean) / (std + 1e-8)).reshape(-1)
            return jnp.broadcast_to(advantages, valid.shape), valid

        def loss_fn(params: PyTree, transitions: Transition) -> tuple[Array, tuple]:
            advantages = transitions.aux["advantages"]
            valid = transitions.aux["valid"]
            num_valid = jnp.maximum(valid.sum(), 1.0)

            dist = self.network.apply(params, transitions.first.obs, temperature=1.0)

            log_probs = dist.log_prob(transitions.second.action)
            entropy = (dist.entropy() * valid).sum() / num_valid
            ratio = jnp.exp(log_probs - transitions.aux["log_prob"])
            approximate_kl = (
                (transitions.aux["log_prob"] - log_probs) * valid
            ).sum() / num_valid
            clip_fraction = (
                (jnp.abs(ratio - 1.0) > self.cfg.clip_coefficient).astype(jnp.float32)
                * valid
            ).sum() / num_valid
            actor_loss = (
                -(
                    jnp.minimum(
                        ratio * advantages,
                        jnp.clip(
                            ratio,
                            1.0 - self.cfg.clip_coefficient,
                            1.0 + self.cfg.clip_coefficient,
                        )
                        * advantages,
                    )
                    * valid
                ).sum()
                / num_valid
            )

            log_ratio = transitions.aux["reference_log_prob"] - log_probs
            reference_kl = (
                (jnp.exp(log_ratio) - log_ratio - 1.0) * valid
            ).sum() / num_valid

            loss = actor_loss + self.cfg.kl_coefficient * reference_kl

            def apply(params: PyTree) -> PyTree:
                return self.network.apply(
                    params, transitions.first.obs, temperature=1.0
                )

            for auxiliary_loss in self.auxiliary_losses:
                loss = loss + auxiliary_loss(
                    params=params,
                    apply=apply,
                    transitions=transitions,
                    dist=dist,
                )
            return loss, (
                actor_loss,
                reference_kl,
                entropy,
                approximate_kl,
                clip_fraction,
            )

        advantages, valid = group_relative_advantage_estimation(transitions)
        transitions = transitions.replace(
            aux={**transitions.aux, "advantages": advantages, "valid": valid}
        )

        transitions = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), transitions)

        reference = self.network.apply(
            state.reference_params, transitions.first.obs, temperature=1.0
        )
        reference_log_prob = reference.log_prob(transitions.second.action)
        transitions = transitions.replace(
            aux={**transitions.aux, "reference_log_prob": reference_log_prob}
        )

        def minibatch_fn(state: GRPOState, indices: Array) -> tuple[GRPOState, tuple]:
            minibatch = jax.tree.map(
                lambda leaf: jnp.take(leaf, indices, axis=0), transitions
            )
            advantages = minibatch.aux["advantages"]
            valid = minibatch.aux["valid"]
            num_valid = jnp.maximum(valid.sum(), 1.0)
            advantage = (advantages * valid).sum() / num_valid

            (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, minibatch
            )
            (
                actor_loss,
                reference_kl,
                entropy,
                approximate_kl,
                clip_fraction,
            ) = aux
            lox.log(
                {
                    "actor/loss": actor_loss,
                    "actor/reference_kl": reference_kl,
                    "actor/entropy": entropy,
                    "actor/approximate_kl": approximate_kl,
                    "actor/clip_fraction": clip_fraction,
                    "actor/advantage": advantage,
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
