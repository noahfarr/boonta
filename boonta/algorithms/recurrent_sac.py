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
    add_time_axis,
    canonicalize_dtype,
    conditional_update,
    remove_batch_axis,
    remove_feature_axis,
    remove_time_axis,
)
from boonta.utils.typing import Array, Buffer, BufferState, Key, PyTree


@struct.dataclass(frozen=True)
class RecurrentSACConfig:
    updates_per_step: int
    gamma: float
    tau: float
    target_entropy: float


@struct.dataclass(frozen=True)
class RecurrentSACState:
    step: Array
    carry: PyTree = struct.field(metadata={"axis": "data"})
    params: PyTree
    critic_params: PyTree
    target_critic_params: PyTree
    alpha_params: PyTree
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    buffer_state: BufferState = struct.field(metadata={"axis": "data"})


@dataclass
class RecurrentSAC:
    cfg: RecurrentSACConfig
    actor: nn.Module
    critic: nn.Module
    alpha: nn.Module
    buffer: Buffer
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    alpha_optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> RecurrentSACState:
        actor_key, critic_key, alpha_key, carry_key = jax.random.split(key, 4)

        carry = self.actor.initialize_carry(carry_key, (*timestep.reward.shape, 1))
        sequence = timestep.to_sequence()
        actor_params = self.actor.init(
            actor_key,
            sequence.obs,
            sequence.action,
            sequence.reward,
            sequence.done,
            carry=carry,
            temperature=1.0,
        )
        critic_params = self.critic.init(
            critic_key,
            sequence.obs,
            sequence.action,
            sequence.reward,
            sequence.done,
            carry=self.critic.initialize_carry(
                carry_key, (*timestep.reward.shape, 1)
            ),
        )
        alpha_params = self.alpha.init(alpha_key)

        return RecurrentSACState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            carry=carry,
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
        self,
        state: RecurrentSACState,
        key: Key,
        timestep: Timestep,
        temperature: float = 1.0,
    ) -> tuple[RecurrentSACState, Array, PyTree]:
        sequence = timestep.to_sequence()
        carry, dist = self.actor.apply(
            state.params,
            sequence.obs,
            sequence.action,
            sequence.reward,
            sequence.done,
            carry=state.carry,
            temperature=temperature,
        )
        action = dist.sample(seed=key)
        action = remove_time_axis(action)
        return state.replace(carry=carry), action, {}

    def update(
        self, state: RecurrentSACState, key: Key, transitions: Transition
    ) -> RecurrentSACState:
        def critic_loss_fn(
            critic_params: PyTree,
            state: RecurrentSACState,
            trajectory: Transition,
            key: Key,
        ) -> tuple[Array, Array]:
            actor_key, target_key, critic_key, action_key = jax.random.split(key, 4)
            alpha = jnp.exp(self.alpha.apply(state.alpha_params))

            transitions = jax.tree.map(lambda x: x[:, :-1], trajectory)
            next_transitions = jax.tree.map(lambda x: x[:, 1:], trajectory)

            batch, _, *num_agents = transitions.second.reward.shape
            carry = self.actor.initialize_carry(actor_key, (batch, *num_agents, 1))
            timesteps = next_transitions.first
            _, dist = self.actor.apply(
                state.params,
                timesteps.obs,
                timesteps.action,
                timesteps.reward,
                timesteps.done,
                carry=carry,
                temperature=1.0,
            )
            next_action, next_log_prob = dist.sample_and_log_prob(seed=action_key)

            carry = self.critic.initialize_carry(target_key, (batch, *num_agents, 1))
            _, next_q_value = self.critic.apply(
                state.target_critic_params,
                next_transitions.first.obs,
                next_action,
                next_transitions.first.reward,
                next_transitions.first.done,
                carry=carry,
            )
            next_q_value = remove_feature_axis(next_q_value)
            target_q_value = transitions.second.reward + self.cfg.gamma * (
                1.0 - transitions.second.terminated
            ) * (jnp.min(next_q_value, axis=0) - alpha * next_log_prob)

            carry = self.critic.initialize_carry(critic_key, (batch, *num_agents, 1))
            _, q_value = self.critic.apply(
                critic_params,
                transitions.first.obs,
                transitions.second.action,
                transitions.first.reward,
                transitions.first.done,
                carry=carry,
            )
            q_value = remove_feature_axis(q_value)
            td_error = (q_value - target_q_value) * (1.0 - transitions.second.truncated)
            loss = 0.5 * (td_error**2).mean()
            return loss, q_value

        def actor_loss_fn(
            params: PyTree,
            state: RecurrentSACState,
            trajectory: Transition,
            key: Key,
        ) -> tuple[Array, Array]:
            actor_key, critic_key, action_key = jax.random.split(key, 3)
            alpha = jnp.exp(self.alpha.apply(state.alpha_params))

            timesteps = trajectory.first
            batch, _, *num_agents = trajectory.second.reward.shape
            carry = self.actor.initialize_carry(actor_key, (batch, *num_agents, 1))
            (_, dist), intermediates = self.actor.apply(
                params,
                timesteps.obs,
                timesteps.action,
                timesteps.reward,
                timesteps.done,
                carry=carry,
                temperature=1.0,
                mutable="intermediates",
            )
            action, log_prob = dist.sample_and_log_prob(seed=action_key)

            critic_carry = self.critic.initialize_carry(
                critic_key, (batch, *num_agents, 1)
            )
            _, q_value = self.critic.apply(
                state.critic_params,
                timesteps.obs,
                action,
                timesteps.reward,
                timesteps.done,
                carry=critic_carry,
            )
            q_value = remove_feature_axis(q_value)
            loss = (alpha * log_prob - jnp.min(q_value, axis=0)).mean()

            def apply(params: PyTree) -> PyTree:
                _, dist = self.actor.apply(
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
                    intermediates=intermediates,
                )
            return loss, log_prob

        def alpha_loss_fn(alpha_params: PyTree, log_prob: Array) -> Array:
            log_alpha = self.alpha.apply(alpha_params)
            return -(log_alpha * (log_prob + self.cfg.target_entropy)).mean()

        buffer_state, _ = jax.lax.scan(
            lambda buffer_state, transition: (
                self.buffer.add(buffer_state, jax.tree.map(add_time_axis, transition)),
                None,
            ),
            state.buffer_state,
            transitions,
        )
        state = state.replace(buffer_state=buffer_state)

        def minibatch_fn(
            state: RecurrentSACState, batch, key: Key
        ) -> RecurrentSACState:
            critic_key, actor_key = jax.random.split(key)
            trajectory = batch

            (critic_loss, q_value), grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True
            )(state.critic_params, state, trajectory, critic_key)
            updates, critic_optimizer_state = self.critic_optimizer.update(
                grads["params"],
                state.critic_optimizer_state,
                state.critic_params["params"],
            )
            state = state.replace(
                critic_params={
                    **state.critic_params,
                    "params": optax.apply_updates(
                        state.critic_params["params"], updates
                    ),
                },
                critic_optimizer_state=critic_optimizer_state,
            )

            (actor_loss, log_prob), grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(state.params, state, trajectory, actor_key)
            updates, actor_optimizer_state = self.actor_optimizer.update(
                grads["params"], state.actor_optimizer_state, state.params["params"]
            )
            state = state.replace(
                params={
                    **state.params,
                    "params": optax.apply_updates(state.params["params"], updates),
                },
                actor_optimizer_state=actor_optimizer_state,
            )

            alpha_loss, grads = jax.value_and_grad(alpha_loss_fn)(
                state.alpha_params, jax.lax.stop_gradient(log_prob)
            )
            updates, alpha_optimizer_state = self.alpha_optimizer.update(
                grads["params"],
                state.alpha_optimizer_state,
                state.alpha_params["params"],
            )
            state = state.replace(
                alpha_params={
                    **state.alpha_params,
                    "params": optax.apply_updates(
                        state.alpha_params["params"], updates
                    ),
                },
                alpha_optimizer_state=alpha_optimizer_state,
                target_critic_params=optax.incremental_update(
                    state.critic_params, state.target_critic_params, self.cfg.tau
                ),
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

        def draw(
            state: RecurrentSACState, keys: tuple[Key, Key]
        ) -> tuple[RecurrentSACState, None]:
            sample_key, update_key = keys
            batch = self.buffer.sample(state.buffer_state, sample_key).experience
            return minibatch_fn(state, batch, update_key), None

        updated, _ = jax.lax.scan(draw, state, (sample_keys, update_keys))
        return conditional_update(
            updated, state, self.buffer.can_sample(state.buffer_state)
        )
