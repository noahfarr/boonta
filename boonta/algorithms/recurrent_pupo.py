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


@struct.dataclass(frozen=True)
class RecurrentPuPOConfig:
    num_minibatches: int
    update_epochs: int
    clip_coefficient: float
    clip_value_loss: bool
    entropy_coefficient: float
    value_coefficient: float
    gamma: float
    gae_lambda: float
    advantage_clip: float = 1.0
    trace_clip: float = 1.0
    priority_exponent: float = 0.8


@struct.dataclass(frozen=True)
class RecurrentPuPOState:
    step: Array
    params: PyTree
    optimizer_state: optax.OptState
    carry: PyTree
    rollout_carry: PyTree = struct.field(metadata={"axis": "data"})


@dataclass
class RecurrentPuPO:
    cfg: RecurrentPuPOConfig
    network: nn.Module
    optimizer: optax.GradientTransformation
    importance_exponent: optax.Schedule = optax.constant_schedule(0.2)
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> RecurrentPuPOState:
        sequence = timestep.to_sequence()
        carry = self.network.initialize_carry(key, (*timestep.reward.shape, 1))
        params = self.network.init(
            key,
            sequence.obs,
            sequence.action,
            sequence.reward,
            sequence.done,
            carry=carry,
            temperature=1.0,
        )
        return RecurrentPuPOState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            params=params,
            optimizer_state=self.optimizer.init(params["params"]),
            carry=carry,
            rollout_carry=carry,
        )

    def step(
        self,
        state: RecurrentPuPOState,
        key: Key,
        timestep: Timestep,
        temperature: float = 1.0,
    ) -> tuple[RecurrentPuPOState, Array, PyTree]:
        sequence = timestep.to_sequence()
        carry, (dist, value) = self.network.apply(
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
        aux = {
            "log_prob": remove_time_axis(log_prob),
            "value": remove_time_axis(remove_feature_axis(value)),
        }
        return state.replace(carry=carry), action, aux

    def update(
        self, state: RecurrentPuPOState, key: Key, transitions: Transition
    ) -> RecurrentPuPOState:
        def loss_fn(
            params: PyTree, trajectory: Transition, carry: PyTree
        ) -> tuple[Array, tuple]:
            advantages = trajectory.aux["advantages"]
            returns = trajectory.aux["returns"]

            timesteps = trajectory.first
            _, (dist, values) = self.network.apply(
                params,
                timesteps.obs,
                timesteps.action,
                timesteps.reward,
                timesteps.done,
                carry=carry,
                temperature=1.0,
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
                )
            return loss, (
                actor_loss,
                critic_loss,
                entropy,
                approximate_kl,
                clip_fraction,
                values,
                ratio,
            )

        num_steps, num_envs, *num_agents = transitions.second.reward.shape
        width = num_envs // self.cfg.num_minibatches

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
        bootstrap = remove_time_axis(remove_feature_axis(values))

        initial_carry = state.rollout_carry

        beta = self.importance_exponent(state.step)

        def vtrace(ratio: Array) -> tuple[Array, Array]:
            gamma, gae_lambda = self.cfg.gamma, self.cfg.gae_lambda
            rewards = transitions.second.reward
            values = transitions.aux["value"]
            dtype = jnp.result_type(values, bootstrap, rewards, jnp.float32)
            rewards, values = rewards.astype(dtype), values.astype(dtype)
            value = bootstrap.astype(dtype)
            ratio = jnp.moveaxis(ratio, 0, 1).astype(dtype)
            rho = jnp.minimum(ratio, self.cfg.advantage_clip)
            c = jnp.minimum(ratio, self.cfg.trace_clip)

            def scan_fn(carry: tuple, x: tuple) -> tuple:
                advantage, next_value = carry
                reward, terminated, truncated, value, rho_t, c_t = x
                delta = rho_t * (
                    reward + gamma * (1.0 - terminated) * next_value - value
                )
                delta *= 1.0 - truncated
                advantage = (
                    delta
                    + gamma
                    * gae_lambda
                    * c_t
                    * (1.0 - terminated)
                    * (1.0 - truncated)
                    * advantage
                )
                return (advantage, value), advantage

            _, advantages = jax.lax.scan(
                scan_fn,
                (jnp.zeros_like(value), value),
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

        def minibatch_fn(
            carry_state: tuple, key: Key
        ) -> tuple[tuple, None]:
            state, ratio = carry_state
            advantages, returns = vtrace(ratio)

            priority = (
                jnp.abs(advantages).sum(axis=(0, *range(2, advantages.ndim)))
                ** self.cfg.priority_exponent
            )
            probabilities = (priority + 1e-6) / (priority.sum() + 1e-6)
            indices = jax.random.choice(
                key, num_envs, shape=(width,), replace=True, p=probabilities
            )
            weight = (num_envs * probabilities[indices]) ** -beta

            epoch = transitions.replace(
                aux={**transitions.aux, "advantages": advantages, "returns": returns}
            )
            trajectory = jax.tree.map(
                lambda leaf: jnp.moveaxis(jnp.take(leaf, indices, axis=1), 0, 1), epoch
            )
            scaled = trajectory.aux["advantages"]
            scaled = (scaled - scaled.mean()) / (scaled.std() + 1e-8)
            scaled = jnp.reshape(weight, (-1,) + (1,) * (scaled.ndim - 1)) * scaled
            trajectory = trajectory.replace(
                aux={**trajectory.aux, "advantages": scaled}
            )
            carry = jax.tree.map(
                lambda leaf: jnp.take(leaf, indices, axis=0), initial_carry
            )

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
                new_ratio,
            ) = aux
            mb_returns = trajectory.aux["returns"]
            explained_variance = 1 - jnp.var(mb_returns - values) / (
                jnp.var(mb_returns) + 1e-8
            )
            lox.log(
                {
                    "actor/loss": actor_loss,
                    "actor/entropy": entropy,
                    "actor/approximate_kl": approximate_kl,
                    "actor/clip_fraction": clip_fraction,
                    "actor/advantage": trajectory.aux["advantages"].mean(),
                    "actor/priority_weight": weight.mean(),
                    "actor/priority_beta": beta,
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
            ratio = ratio.at[indices].set(jax.lax.stop_gradient(new_ratio))
            return (
                state.replace(params=params, optimizer_state=optimizer_state),
                ratio,
            ), None

        keys = jax.random.split(
            key, self.cfg.update_epochs * self.cfg.num_minibatches
        )
        ratio = jnp.ones((num_envs, num_steps, *num_agents))
        (state, _), _ = jax.lax.scan(minibatch_fn, (state, ratio), keys)
        return state.replace(rollout_carry=state.carry)
