from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import Timestep, Transition, canonicalize_dtype, flatten
from boonta.utils.typing import Array, Key, PyTree


@struct.dataclass(frozen=True)
class REPPOConfig:
    num_minibatches: int
    update_epochs: int
    gamma: float
    td_lambda: float
    target_kl: float
    target_entropy_scale: float
    num_kl_samples: int
    vmin: float
    vmax: float
    num_bins: int
    action_dim: int

    @property
    def support(self):
        return jnp.linspace(self.vmin, self.vmax, self.num_bins + 1)

    @property
    def bin_centers(self):
        half_width = (self.vmax - self.vmin) / (2 * self.num_bins)
        return jnp.linspace(
            self.vmin + half_width, self.vmax - half_width, self.num_bins
        )

    @property
    def sigma(self):
        return 0.75 * (self.vmax - self.vmin) / self.num_bins


@struct.dataclass(frozen=True)
class REPPOState:
    step: Array
    params: PyTree
    actor_target_params: PyTree
    critic_params: PyTree
    alpha_params: PyTree
    lagrangian_params: PyTree
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    lagrangian_optimizer_state: optax.OptState


@dataclass
class REPPO:
    cfg: REPPOConfig
    actor: nn.Module
    critic: nn.Module
    alpha: nn.Module
    lagrangian: nn.Module
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    alpha_optimizer: optax.GradientTransformation
    lagrangian_optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> REPPOState:
        actor_key, critic_key, alpha_key, lagrangian_key = jax.random.split(key, 4)

        actor_params = self.actor.init(actor_key, timestep.obs, temperature=1.0)
        critic_params = self.critic.init(critic_key, timestep.obs, timestep.action)
        alpha_params = self.alpha.init(alpha_key)
        lagrangian_params = self.lagrangian.init(lagrangian_key)

        return REPPOState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            params=actor_params,
            actor_target_params=actor_params,
            critic_params=critic_params,
            alpha_params=alpha_params,
            lagrangian_params=lagrangian_params,
            actor_optimizer_state=self.actor_optimizer.init(actor_params["params"]),
            critic_optimizer_state=self.critic_optimizer.init(critic_params["params"]),
            alpha_optimizer_state=self.alpha_optimizer.init(alpha_params["params"]),
            lagrangian_optimizer_state=self.lagrangian_optimizer.init(
                lagrangian_params["params"]
            ),
        )

    def step(
        self, state: REPPOState, key: Key, timestep: Timestep, temperature: float = 1.0
    ) -> tuple[REPPOState, Array, PyTree]:
        dist = self.actor.apply(state.params, timestep.obs, temperature=temperature)
        action = dist.sample(seed=key)
        return state, action, {}

    def update(
        self, state: REPPOState, key: Key, transitions: Transition
    ) -> REPPOState:
        def bootstrap(
            state: REPPOState, transitions: Transition, key: Key
        ) -> tuple[Array, Array]:
            def bootstrap_fn(timestep: Timestep, key: Key) -> tuple[Array, Array]:
                next_dist = self.actor.apply(
                    state.params, timestep.obs, temperature=1.0
                )
                next_action, next_log_prob = next_dist.sample_and_log_prob(seed=key)
                next_critic_logits = self.critic.apply(
                    state.critic_params, timestep.obs, next_action
                )
                next_value = jnp.sum(
                    jax.nn.softmax(next_critic_logits, axis=-1) * self.cfg.bin_centers,
                    axis=-1,
                )
                alpha = jnp.exp(self.alpha.apply(state.alpha_params))
                soft_reward = timestep.reward - self.cfg.gamma * (
                    1.0 - timestep.terminated
                ) * alpha * next_log_prob
                return soft_reward, next_value

            num_steps, *_ = transitions.second.reward.shape
            keys = jax.random.split(key, num_steps)
            return jax.vmap(bootstrap_fn)(transitions.second, keys)

        def lambda_returns(transitions: Transition) -> Array:
            gamma, td_lambda = self.cfg.gamma, self.cfg.td_lambda
            soft_reward = transitions.aux["soft_reward"]
            value = transitions.aux["value"]
            terminated = transitions.second.terminated
            truncated = transitions.second.truncated

            def scan_fn(carry: tuple, x: tuple) -> tuple[tuple, Array]:
                lambda_return, cut = carry
                soft_reward, value, terminated, truncated = x
                bootstrap = jnp.where(
                    cut, value, td_lambda * lambda_return + (1.0 - td_lambda) * value
                )
                lambda_return = soft_reward + gamma * (1.0 - terminated) * bootstrap
                return (lambda_return, truncated), lambda_return

            last_value = jnp.take(value, -1, axis=0)
            _, target_values = jax.lax.scan(
                scan_fn,
                (last_value, jnp.zeros_like(last_value, bool)),
                (soft_reward, value, terminated, truncated),
                reverse=True,
            )
            return target_values

        def critic_loss_fn(
            critic_params: PyTree, transitions: Transition
        ) -> tuple[Array, tuple[PyTree, Array]]:
            logits, variables = self.critic.apply(
                critic_params,
                transitions.first.obs,
                transitions.second.action,
                mutable=True,
            )

            target = jnp.clip(
                transitions.aux["target_values"], self.cfg.vmin, self.cfg.vmax
            )
            cdf = jax.scipy.special.erf(
                (self.cfg.support - target[..., None]) / (jnp.sqrt(2) * self.cfg.sigma)
            )
            probs = jnp.diff(cdf, axis=-1)
            target_probs = probs / probs.sum(axis=-1, keepdims=True)

            loss = optax.softmax_cross_entropy(logits, target_probs)
            loss = (loss * (1.0 - transitions.second.truncated)).mean()

            q_value = jnp.sum(
                jax.nn.softmax(logits, axis=-1) * self.cfg.bin_centers, axis=-1
            )
            return loss, (variables, q_value)

        def actor_loss_fn(
            params: PyTree, state: REPPOState, transitions: Transition, key: Key
        ) -> tuple[Array, tuple[PyTree, Array, Array]]:
            action_key, kl_key = jax.random.split(key)
            alpha = jax.lax.stop_gradient(jnp.exp(self.alpha.apply(state.alpha_params)))
            lagrangian = jax.lax.stop_gradient(
                jnp.exp(self.lagrangian.apply(state.lagrangian_params))
            )

            dist, variables = self.actor.apply(
                params,
                transitions.first.obs,
                temperature=1.0,
                mutable=True,
            )
            action, log_prob = dist.sample_and_log_prob(seed=action_key)

            logits = self.critic.apply(
                state.critic_params, transitions.first.obs, action
            )
            q_value = jnp.sum(
                jax.nn.softmax(logits, axis=-1) * self.cfg.bin_centers, axis=-1
            )

            target_dist = self.actor.apply(
                state.actor_target_params, transitions.first.obs, temperature=1.0
            )
            target_actions = target_dist.sample(
                seed=kl_key, sample_shape=(self.cfg.num_kl_samples,)
            )
            target_actions = jnp.clip(target_actions, -1.0 + 1e-4, 1.0 - 1e-4)
            target_log_prob = target_dist.log_prob(target_actions).mean(axis=0)
            actor_log_prob = dist.log_prob(target_actions).mean(axis=0)
            kl = target_log_prob - actor_log_prob

            pathwise_loss = alpha * log_prob - q_value
            kl_loss = kl * lagrangian
            actor_loss = jnp.where(kl < self.cfg.target_kl, pathwise_loss, kl_loss)

            loss = actor_loss.mean()

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
            return loss, (variables, log_prob, kl)

        def alpha_loss_fn(
            alpha_params: PyTree, log_prob: Array
        ) -> tuple[Array, PyTree]:
            log_alpha, variables = self.alpha.apply(alpha_params, mutable=True)
            alpha = jnp.exp(log_alpha)
            target_entropy = (
                self.cfg.action_dim * self.cfg.target_entropy_scale - log_prob
            )
            loss = (alpha * jax.lax.stop_gradient(target_entropy)).mean()
            return loss, variables

        def lagrangian_loss_fn(
            lagrangian_params: PyTree, kl: Array
        ) -> tuple[Array, PyTree]:
            log_lagrangian, variables = self.lagrangian.apply(
                lagrangian_params, mutable=True
            )
            loss = -(log_lagrangian * (kl - self.cfg.target_kl)).mean()
            return loss, variables

        num_steps, num_envs = transitions.second.reward.shape
        batch_size = num_steps * num_envs

        state = state.replace(actor_target_params=state.params)

        bootstrap_key, key = jax.random.split(key)
        soft_reward, value = bootstrap(state, transitions, bootstrap_key)
        transitions = transitions.replace(
            aux={**transitions.aux, "soft_reward": soft_reward, "value": value}
        )

        target_values = lambda_returns(transitions)
        transitions = transitions.replace(
            aux={**transitions.aux, "target_values": target_values}
        )
        transitions = jax.tree.map(
            lambda x: flatten(x, start_dim=0, end_dim=1), transitions
        )

        def minibatch_fn(
            state: REPPOState, x: tuple[Array, Key]
        ) -> tuple[REPPOState, None]:
            indices, key = x
            minibatch = jax.tree.map(
                lambda leaf: jnp.take(leaf, indices, axis=0), transitions
            )

            (critic_loss, (variables, q_value)), grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True, allow_int=True
            )(state.critic_params, minibatch)
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

            (actor_loss, (variables, log_prob, kl)), grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True, allow_int=True
            )(state.params, state, minibatch, key)
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
            )

            (lagrangian_loss, variables), grads = jax.value_and_grad(
                lagrangian_loss_fn, has_aux=True, allow_int=True
            )(state.lagrangian_params, jax.lax.stop_gradient(kl))
            updates, lagrangian_optimizer_state = self.lagrangian_optimizer.update(
                grads["params"],
                state.lagrangian_optimizer_state,
                state.lagrangian_params["params"],
            )
            variables = {
                name: variables.get(name, value)
                for name, value in state.lagrangian_params.items()
            }
            state = state.replace(
                lagrangian_params={
                    **variables,
                    "params": optax.apply_updates(
                        state.lagrangian_params["params"], updates
                    ),
                },
                lagrangian_optimizer_state=lagrangian_optimizer_state,
            )

            lox.log(
                {
                    "actor/loss": actor_loss,
                    "actor/entropy": -log_prob.mean(),
                    "actor/kl": kl.mean(),
                    "critic/loss": critic_loss,
                    "critic/q_value": q_value.mean(),
                    "alpha/loss": alpha_loss,
                    "alpha/value": jnp.exp(self.alpha.apply(state.alpha_params)),
                    "lagrangian/loss": lagrangian_loss,
                    "lagrangian/value": jnp.exp(
                        self.lagrangian.apply(state.lagrangian_params)
                    ),
                }
            )
            return state, None

        def epoch_fn(state, key):
            permutation_key, minibatch_key = jax.random.split(key)
            permutation = jax.random.permutation(permutation_key, batch_size)
            minibatch_indices = permutation.reshape(self.cfg.num_minibatches, -1)
            minibatch_keys = jax.random.split(minibatch_key, self.cfg.num_minibatches)

            state, _ = jax.lax.scan(
                minibatch_fn, state, (minibatch_indices, minibatch_keys)
            )

            return state, None

        keys = jax.random.split(key, self.cfg.update_epochs)
        state, _ = jax.lax.scan(epoch_fn, state, keys)

        return state
