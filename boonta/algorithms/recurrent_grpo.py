from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import (Timestep, Transition, canonicalize_dtype,
                          remove_time_axis)


from boonta.utils.typing import Array, Key, PyTree


@struct.dataclass(frozen=True)
class RecurrentGRPOConfig:
    group_size: int
    num_minibatches: int
    update_epochs: int
    clip_coefficient: float
    kl_coefficient: float
    gamma: float


@struct.dataclass(frozen=True)
class RecurrentGRPOState:
    step: Array
    carry: PyTree = struct.field(metadata={"axis": "data"})
    rollout_carry: PyTree = struct.field(metadata={"axis": "data"})
    params: PyTree
    reference_params: PyTree
    optimizer_state: optax.OptState


@dataclass
class RecurrentGRPO:
    cfg: RecurrentGRPOConfig
    network: nn.Module
    optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> RecurrentGRPOState:
        network_key, carry_key = jax.random.split(key)

        carry = self.network.initialize_carry(carry_key, (*timestep.reward.shape, 1))
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

        return RecurrentGRPOState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            carry=carry,
            rollout_carry=carry,
            params=params,
            reference_params=params,
            optimizer_state=self.optimizer.init(params["params"]),
        )

    def step(
        self,
        state: RecurrentGRPOState,
        key: Key,
        timestep: Timestep,
        temperature: float = 1.0,
    ) -> tuple[RecurrentGRPOState, Array, PyTree]:
        sequence = timestep.to_sequence()
        carry, dist = self.network.apply(
            state.params,
            sequence.obs,
            sequence.action,
            sequence.reward,
            sequence.done,
            carry=state.carry,
            temperature=temperature,
        )
        action, log_prob = dist.sample_and_log_prob(seed=key)

        aux = {"log_prob": remove_time_axis(log_prob)}
        return state.replace(carry=carry), remove_time_axis(action), aux

    def update(
        self, state: RecurrentGRPOState, key: Key, transitions: Transition
    ) -> RecurrentGRPOState:
        num_steps, num_envs = transitions.second.reward.shape
        num_groups = num_envs // self.cfg.group_size

        def group_relative_advantage_estimation(
            trajectory: Transition,
        ) -> tuple[Array, Array]:
            rewards = trajectory.second.reward
            dones = trajectory.second.done

            valid = (jnp.cumsum(dones, axis=0) - dones == 0).astype(rewards.dtype)
            discounts = self.cfg.gamma ** jnp.arange(num_steps, dtype=rewards.dtype)
            returns = discounts @ (rewards * valid)

            groups = returns.reshape(num_groups, self.cfg.group_size)
            mean = groups.mean(axis=1, keepdims=True)
            std = groups.std(axis=1, keepdims=True)
            advantages = ((groups - mean) / (std + 1e-8)).reshape(-1)
            return jnp.broadcast_to(advantages, valid.shape), valid

        def loss_fn(
            params: PyTree, trajectory: Transition, carry: PyTree
        ) -> tuple[Array, tuple]:
            advantages = trajectory.aux["advantages"]
            valid = trajectory.aux["valid"]
            num_valid = jnp.maximum(valid.sum(), 1.0)

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

            log_probs = dist.log_prob(trajectory.second.action)
            entropy = (dist.entropy() * valid).sum() / num_valid
            ratio = jnp.exp(log_probs - trajectory.aux["log_prob"])
            approximate_kl = (
                (trajectory.aux["log_prob"] - log_probs) * valid
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

            log_ratio = trajectory.aux["reference_log_prob"] - log_probs
            reference_kl = (
                (jnp.exp(log_ratio) - log_ratio - 1.0) * valid
            ).sum() / num_valid

            loss = actor_loss + self.cfg.kl_coefficient * reference_kl

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
                    carry=carry,
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

        initial_carry = state.rollout_carry
        advantages, valid = group_relative_advantage_estimation(transitions)
        trajectory = transitions.replace(
            aux={**transitions.aux, "advantages": advantages, "valid": valid}
        )

        sequences = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), trajectory.first)
        _, reference = self.network.apply(
            state.reference_params,
            sequences.obs,
            sequences.action,
            sequences.reward,
            sequences.done,
            carry=initial_carry,
            temperature=1.0,
        )
        reference_log_prob = reference.log_prob(
            jnp.swapaxes(trajectory.second.action, 0, 1)
        )
        trajectory = trajectory.replace(
            aux={
                **trajectory.aux,
                "reference_log_prob": jnp.swapaxes(reference_log_prob, 0, 1),
            }
        )

        def minibatch_fn(
            state: RecurrentGRPOState, indices: Array
        ) -> tuple[RecurrentGRPOState, tuple]:
            minibatch = jax.tree.map(
                lambda leaf: jnp.moveaxis(jnp.take(leaf, indices, axis=1), 0, 1),
                trajectory,
            )
            carry = jax.tree.map(
                lambda leaf: jnp.take(leaf, indices, axis=0), initial_carry
            )
            advantages = minibatch.aux["advantages"]
            valid = minibatch.aux["valid"]
            num_valid = jnp.maximum(valid.sum(), 1.0)
            advantage = (advantages * valid).sum() / num_valid

            (_, aux), grads = jax.value_and_grad(
                loss_fn, has_aux=True, allow_int=True
            )(state.params, minibatch, carry)
            (
                returned,
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
