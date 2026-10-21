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

from .advantage_estimators import group_relative_advantage_estimator
from .algorithm import Carries


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
    advantage_estimator: Callable = group_relative_advantage_estimator

    def init(self, key: Key, timestep: Timestep) -> tuple[GRPOState, PyTree]:
        params = self.network.init(key, timestep.obs, temperature=1.0)
        state = GRPOState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            params=params,
            reference_params=params,
            optimizer_state=self.optimizer.init(params["params"]),
        )

        return state, None

    def step(
        self,
        state: GRPOState,
        carry: PyTree,
        key: Key,
        timestep: Timestep,
        temperature: float = 1.0,
    ) -> tuple[GRPOState, PyTree, Array, PyTree]:
        dist = self.network.apply(state.params, timestep.obs, temperature=temperature)
        action, log_prob = dist.sample_and_log_prob(seed=key)
        return state, carry, action, {"log_prob": log_prob}

    def update(
        self, state: GRPOState, key: Key, transitions: Transition, carries: Carries
    ) -> GRPOState:
        num_steps, num_envs = transitions.second.reward.shape
        batch_size = num_steps * num_envs

        def loss_fn(params: PyTree, transitions: Transition) -> tuple[Array, tuple]:
            advantages = transitions.aux["advantages"]
            episode_mask = transitions.aux["episode_mask"]

            dist, variables = self.network.apply(
                params,
                transitions.first.obs,
                temperature=1.0,
                mutable=True,
            )

            log_probs = dist.log_prob(transitions.second.action)
            entropy = (dist.entropy() * episode_mask).sum() / jnp.maximum(
                episode_mask.sum(), 1.0
            )
            ratio = jnp.exp(log_probs - transitions.aux["log_prob"])
            approximate_kl = (
                (transitions.aux["log_prob"] - log_probs) * episode_mask
            ).sum() / jnp.maximum(episode_mask.sum(), 1.0)
            clip_fraction = (
                (jnp.abs(ratio - 1.0) > self.cfg.clip_coefficient).astype(jnp.float32)
                * episode_mask
            ).sum() / jnp.maximum(episode_mask.sum(), 1.0)
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
                    * episode_mask
                ).sum()
                / jnp.maximum(episode_mask.sum(), 1.0)
            )

            log_ratio = transitions.aux["reference_log_prob"] - log_probs
            reference_kl = (
                (jnp.exp(log_ratio) - log_ratio - 1.0) * episode_mask
            ).sum() / jnp.maximum(episode_mask.sum(), 1.0)

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
                    variables=variables,
                )
            return loss, (
                variables,
                actor_loss,
                reference_kl,
                entropy,
                approximate_kl,
                clip_fraction,
            )

        advantages, episode_mask = self.advantage_estimator(
            transitions, self.cfg.gamma, self.cfg.group_size
        )
        transitions = transitions.replace(
            aux={
                **transitions.aux,
                "advantages": advantages,
                "episode_mask": episode_mask,
            }
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
            episode_mask = minibatch.aux["episode_mask"]
            advantage = (advantages * episode_mask).sum() / jnp.maximum(
                episode_mask.sum(), 1.0
            )

            (_, aux), grads = jax.value_and_grad(
                loss_fn, has_aux=True, allow_int=True
            )(state.params, minibatch)
            (
                variables,
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
            variables = {
                name: variables.get(name, value) for name, value in state.params.items()
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
            permutation = jax.random.permutation(key, batch_size)
            minibatch_indices = permutation.reshape(self.cfg.num_minibatches, -1)

            state, _ = jax.lax.scan(minibatch_fn, state, minibatch_indices)

            return state, None

        keys = jax.random.split(key, self.cfg.update_epochs)
        state, _ = jax.lax.scan(epoch_fn, state, keys)

        return state
