from dataclasses import dataclass
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import Timestep, Transition, canonicalize_dtype, remove_feature_axis
from boonta.utils.typing import Array, Key, PyTree


@struct.dataclass(frozen=True)
class IQLConfig:
    gamma: float
    tau: float
    batch_size: int
    expectile: float = 0.7
    beta: float = 3.0
    max_advantage_weight: float = 100.0


@struct.dataclass(frozen=True)
class IQLState:
    step: Array
    params: PyTree
    critic_params: PyTree
    target_critic_params: PyTree
    value_params: PyTree
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    value_optimizer_state: optax.OptState


@dataclass
class IQL:
    cfg: IQLConfig
    actor: nn.Module
    critic: nn.Module
    value: nn.Module
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    value_optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> IQLState:
        actor_key, critic_key, value_key = jax.random.split(key, 3)

        actor_params = self.actor.init(actor_key, timestep.obs, temperature=1.0)
        critic_params = self.critic.init(critic_key, timestep.obs, timestep.action)
        value_params = self.value.init(value_key, timestep.obs)

        return IQLState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            params=actor_params,
            critic_params=critic_params,
            target_critic_params=critic_params,
            value_params=value_params,
            actor_optimizer_state=self.actor_optimizer.init(actor_params["params"]),
            critic_optimizer_state=self.critic_optimizer.init(critic_params["params"]),
            value_optimizer_state=self.value_optimizer.init(value_params["params"]),
        )

    def step(
        self, state: IQLState, key: Key, timestep: Timestep, temperature: float = 1.0
    ) -> tuple[IQLState, Array, PyTree]:
        dist = self.actor.apply(state.params, timestep.obs, temperature=temperature)
        action = dist.sample(seed=key)
        return state, action, {}

    def update(self, state: IQLState, key: Key, transitions: Transition) -> IQLState:
        del key

        def value_loss_fn(
            value_params: PyTree, state: IQLState, transitions: Transition
        ) -> tuple[Array, tuple[PyTree, Array]]:
            q_value = remove_feature_axis(
                self.critic.apply(
                    state.target_critic_params,
                    transitions.first.obs,
                    transitions.second.action,
                )
            )
            q_value = jnp.min(q_value, axis=0)

            value, variables = self.value.apply(
                value_params, transitions.first.obs, mutable=True
            )
            variables.pop("intermediates", None)
            value = remove_feature_axis(value)

            advantage = q_value - value
            weight = jnp.where(
                advantage > 0.0, self.cfg.expectile, 1.0 - self.cfg.expectile
            )
            loss = (weight * advantage**2).mean()
            return loss, (variables, value)

        def critic_loss_fn(
            critic_params: PyTree, state: IQLState, transitions: Transition
        ) -> tuple[Array, tuple[PyTree, Array]]:
            next_value = remove_feature_axis(
                self.value.apply(state.value_params, transitions.second.obs)
            )
            target_q_value = (
                transitions.second.reward
                + self.cfg.gamma * (1.0 - transitions.second.terminated) * next_value
            )

            q_value, variables = self.critic.apply(
                critic_params,
                transitions.first.obs,
                transitions.second.action,
                mutable=True,
            )
            variables.pop("intermediates", None)
            q_value = remove_feature_axis(q_value)
            td_error = (q_value - target_q_value) * (1.0 - transitions.second.truncated)
            loss = 0.5 * (td_error**2).mean()
            return loss, (variables, q_value)

        def actor_loss_fn(
            params: PyTree, state: IQLState, transitions: Transition
        ) -> tuple[Array, tuple[PyTree, Array]]:
            q_value = remove_feature_axis(
                self.critic.apply(
                    state.target_critic_params,
                    transitions.first.obs,
                    transitions.second.action,
                )
            )
            q_value = jnp.min(q_value, axis=0)

            value = remove_feature_axis(
                self.value.apply(state.value_params, transitions.first.obs)
            )

            advantage = q_value - value
            weight = jnp.minimum(
                jnp.exp(self.cfg.beta * advantage), self.cfg.max_advantage_weight
            )

            dist, variables = self.actor.apply(
                params,
                transitions.first.obs,
                temperature=1.0,
                mutable=True,
            )
            intermediates = {"intermediates": variables.pop("intermediates", {})}
            log_prob = dist.log_prob(transitions.second.action)

            loss = -(weight * log_prob).mean()

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
                    intermediates=intermediates,
                )
            return loss, (variables, log_prob)

        (value_loss, (variables, value)), grads = jax.value_and_grad(
            value_loss_fn, has_aux=True, allow_int=True
        )(state.value_params, state, transitions)
        updates, value_optimizer_state = self.value_optimizer.update(
            grads["params"], state.value_optimizer_state, state.value_params["params"]
        )
        state = state.replace(
            value_params={
                **state.value_params,
                **variables,
                "params": optax.apply_updates(state.value_params["params"], updates),
            },
            value_optimizer_state=value_optimizer_state,
        )

        (critic_loss, (variables, q_value)), grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True, allow_int=True
        )(state.critic_params, state, transitions)
        updates, critic_optimizer_state = self.critic_optimizer.update(
            grads["params"], state.critic_optimizer_state, state.critic_params["params"]
        )
        state = state.replace(
            critic_params={
                **state.critic_params,
                **variables,
                "params": optax.apply_updates(state.critic_params["params"], updates),
            },
            critic_optimizer_state=critic_optimizer_state,
        )

        (actor_loss, (variables, log_prob)), grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True, allow_int=True
        )(state.params, state, transitions)
        updates, actor_optimizer_state = self.actor_optimizer.update(
            grads["params"], state.actor_optimizer_state, state.params["params"]
        )
        state = state.replace(
            params={
                **state.params,
                **variables,
                "params": optax.apply_updates(state.params["params"], updates),
            },
            actor_optimizer_state=actor_optimizer_state,
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
                "actor/log_prob": log_prob.mean(),
                "critic/loss": critic_loss,
                "critic/q_value": q_value.mean(),
                "value/loss": value_loss,
                "value/value": value.mean(),
            }
        )

        return state
