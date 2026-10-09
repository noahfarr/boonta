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
    canonicalize_dtype,
    conditional_update,
    remove_batch_axis,
    remove_feature_axis,
)
from boonta.utils.typing import Array, Buffer, BufferState, Key, PyTree


@struct.dataclass(frozen=True)
class SACConfig:
    updates_per_step: int
    gamma: float
    tau: float
    target_entropy: float


@struct.dataclass(frozen=True)
class SACState:
    step: Array
    params: PyTree
    critic_params: PyTree
    target_critic_params: PyTree
    alpha_params: PyTree
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    buffer_state: BufferState = struct.field(metadata={"axis": "data"})


@dataclass
class SAC:
    cfg: SACConfig
    actor: nn.Module
    critic: nn.Module
    alpha: nn.Module
    buffer: Buffer
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    alpha_optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> SACState:
        actor_key, critic_key, alpha_key = jax.random.split(key, 3)

        actor_params = self.actor.init(actor_key, timestep.obs, temperature=1.0)
        critic_params = self.critic.init(critic_key, timestep.obs, timestep.action)
        alpha_params = self.alpha.init(alpha_key)

        return SACState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            params=actor_params,
            critic_params=critic_params,
            target_critic_params=critic_params,
            alpha_params=alpha_params,
            actor_optimizer_state=self.actor_optimizer.init(actor_params["params"]),
            critic_optimizer_state=self.critic_optimizer.init(critic_params["params"]),
            alpha_optimizer_state=self.alpha_optimizer.init(alpha_params["params"]),
            buffer_state=self.buffer.init(
                jax.tree.map(
                    remove_batch_axis,
                    Transition(first=timestep, second=timestep, aux={}),
                )
            ),
        )

    def step(
        self, state: SACState, key: Key, timestep: Timestep, temperature: float = 1.0
    ) -> tuple[SACState, Array, PyTree]:
        dist = self.actor.apply(state.params, timestep.obs, temperature=temperature)
        action = dist.sample(seed=key)
        return state, action, {}

    def update(self, state: SACState, key: Key, transitions: Transition) -> SACState:
        def critic_loss_fn(
            critic_params: PyTree, state: SACState, transitions: Transition, key: Key
        ) -> tuple[Array, tuple[PyTree, Array]]:
            alpha = jnp.exp(self.alpha.apply(state.alpha_params))

            dist = self.actor.apply(
                state.params, transitions.second.obs, temperature=1.0
            )
            next_action, next_log_prob = dist.sample_and_log_prob(seed=key)

            next_q_value = remove_feature_axis(
                self.critic.apply(
                    state.target_critic_params, transitions.second.obs, next_action
                )
            )
            target_q_value = transitions.second.reward + self.cfg.gamma * (
                1.0 - transitions.second.terminated
            ) * (jnp.min(next_q_value, axis=0) - alpha * next_log_prob)

            q_value, variables = self.critic.apply(
                critic_params,
                transitions.first.obs,
                transitions.second.action,
                mutable=True,
            )
            q_value = remove_feature_axis(q_value)
            td_error = (q_value - target_q_value) * (1.0 - transitions.second.truncated)
            loss = 0.5 * (td_error**2).mean()
            return loss, (variables, q_value)

        def actor_loss_fn(
            params: PyTree, state: SACState, transitions: Transition, key: Key
        ) -> tuple[Array, tuple[PyTree, Array]]:
            alpha = jnp.exp(self.alpha.apply(state.alpha_params))

            dist, variables = self.actor.apply(
                params,
                transitions.first.obs,
                temperature=1.0,
                mutable=True,
            )
            action, log_prob = dist.sample_and_log_prob(seed=key)

            q_value = remove_feature_axis(
                self.critic.apply(state.critic_params, transitions.first.obs, action)
            )
            loss = (alpha * log_prob - jnp.min(q_value, axis=0)).mean()

            def apply(params: PyTree) -> PyTree:
                return self.actor.apply(
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
            return loss, (variables, log_prob)

        def alpha_loss_fn(
            alpha_params: PyTree, log_prob: Array
        ) -> tuple[Array, PyTree]:
            log_alpha, variables = self.alpha.apply(alpha_params, mutable=True)
            loss = -(log_alpha * (log_prob + self.cfg.target_entropy)).mean()
            return loss, variables

        transitions = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), transitions)
        state = state.replace(
            buffer_state=self.buffer.add(state.buffer_state, transitions)
        )

        def minibatch_fn(state: SACState, batch, key: Key) -> SACState:
            critic_key, actor_key = jax.random.split(key)
            transitions = batch

            (critic_loss, (variables, q_value)), grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True, allow_int=True
            )(state.critic_params, state, transitions, critic_key)
            updates, critic_optimizer_state = self.critic_optimizer.update(
                grads["params"],
                state.critic_optimizer_state,
                state.critic_params["params"],
            )
            variables = {
                name: variables.get(name, value)
                for name, value in state.critic_params.items()
            }
            state = state.replace(
                critic_params={
                    **variables,
                    "params": optax.apply_updates(
                        state.critic_params["params"], updates
                    ),
                },
                critic_optimizer_state=critic_optimizer_state,
            )

            (actor_loss, (variables, log_prob)), grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True, allow_int=True
            )(state.params, state, transitions, actor_key)
            updates, actor_optimizer_state = self.actor_optimizer.update(
                grads["params"], state.actor_optimizer_state, state.params["params"]
            )
            variables = {
                name: variables.get(name, value) for name, value in state.params.items()
            }
            state = state.replace(
                params={
                    **variables,
                    "params": optax.apply_updates(state.params["params"], updates),
                },
                actor_optimizer_state=actor_optimizer_state,
            )

            (alpha_loss, variables), grads = jax.value_and_grad(
                alpha_loss_fn, has_aux=True, allow_int=True
            )(state.alpha_params, jax.lax.stop_gradient(log_prob))
            updates, alpha_optimizer_state = self.alpha_optimizer.update(
                grads["params"],
                state.alpha_optimizer_state,
                state.alpha_params["params"],
            )
            variables = {
                name: variables.get(name, value)
                for name, value in state.alpha_params.items()
            }
            state = state.replace(
                alpha_params={
                    **variables,
                    "params": optax.apply_updates(
                        state.alpha_params["params"], updates
                    ),
                },
                alpha_optimizer_state=alpha_optimizer_state,
                target_critic_params={
                    **state.critic_params,
                    "params": optax.incremental_update(
                        state.critic_params["params"],
                        state.target_critic_params["params"],
                        self.cfg.tau,
                    ),
                },
            )

            lox.log(
                {
                    "actor/loss": actor_loss,
                    "actor/entropy": -log_prob.mean(),
                    "critic/loss": critic_loss,
                    "critic/q_value": q_value.mean(),
                    "alpha/loss": alpha_loss,
                    "alpha/value": jnp.exp(self.alpha.apply(state.alpha_params)),
                }
            )

            return state

        sample_key, update_key = jax.random.split(key)
        sample_keys = jax.random.split(sample_key, self.cfg.updates_per_step)
        update_keys = jax.random.split(update_key, self.cfg.updates_per_step)

        def draw(state: SACState, keys: tuple[Key, Key]) -> tuple[SACState, None]:
            sample_key, update_key = keys
            batch = jax.tree.map(
                lambda leaf: jnp.squeeze(leaf, axis=1),
                self.buffer.sample(state.buffer_state, sample_key).experience,
            )
            return minibatch_fn(state, batch, update_key), None

        updated, _ = jax.lax.scan(draw, state, (sample_keys, update_keys))
        return conditional_update(
            updated, state, self.buffer.can_sample(state.buffer_state)
        )
