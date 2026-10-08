from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.planners import Controller
from boonta.utils import (Timestep, Transition, add_time_axis, canonicalize_dtype,
                          conditional_update, remove_batch_axis)
from boonta.utils.typing import Array, Buffer, BufferState, Key, PyTree


def symlog(x: Array) -> Array:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: Array) -> Array:
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)


@struct.dataclass(frozen=True)
class TDMPC2Config:
    updates_per_step: int
    horizon: int
    gamma: float
    tau: float
    temporal_decay: float
    consistency_coefficient: float
    reward_coefficient: float
    value_coefficient: float
    entropy_coefficient: float
    num_critics: int
    num_target_critics: int
    vmin: float
    vmax: float
    num_bins: int
    num_policy_trajectories: int
    termination_coefficient: float
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
class TDMPC2State:
    step: Array
    params: PyTree
    world_model_params: PyTree
    target_critic_params: PyTree
    world_model_optimizer_state: optax.OptState
    actor_optimizer_state: optax.OptState
    controller_state: PyTree = struct.field(metadata={"axis": "data"})
    buffer_state: BufferState = struct.field(metadata={"axis": "data"})


@dataclass
class TDMPC2:
    cfg: TDMPC2Config
    encoder: nn.Module
    dynamics: nn.Module
    reward: nn.Module
    critic: nn.Module
    actor: nn.Module
    termination: nn.Module
    controller: Controller
    buffer: Buffer
    world_model_optimizer: optax.GradientTransformation
    actor_optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()

    def part(self, world_model_params: PyTree, name: str) -> PyTree:
        return {"params": world_model_params["params"][name]}

    def init(self, key: Key, timestep: Timestep) -> TDMPC2State:
        (
            encoder_key,
            dynamics_key,
            reward_key,
            critic_key,
            actor_key,
            termination_key,
            controller_key,
        ) = jax.random.split(key, 7)

        encoder_params = self.encoder.init(encoder_key, timestep.obs)
        z = self.encoder.apply(encoder_params, timestep.obs)

        dynamics_params = self.dynamics.init(dynamics_key, z, timestep.action)
        reward_params = self.reward.init(reward_key, z, timestep.action)
        critic_params = self.critic.init(critic_key, z, timestep.action)
        actor_params = self.actor.init(actor_key, z, temperature=1.0)
        termination_params = self.termination.init(termination_key, z)

        world_model_params = {
            "params": {
                "encoder": encoder_params["params"],
                "dynamics": dynamics_params["params"],
                "reward": reward_params["params"],
                "critic": critic_params["params"],
                "termination": termination_params["params"],
            }
        }

        return TDMPC2State(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            params=actor_params,
            world_model_params=world_model_params,
            target_critic_params=critic_params,
            world_model_optimizer_state=self.world_model_optimizer.init(
                world_model_params["params"]
            ),
            actor_optimizer_state=self.actor_optimizer.init(actor_params["params"]),
            buffer_state=self.buffer.init(
                jax.tree.map(
                    remove_batch_axis,
                    Transition(first=timestep, second=timestep, aux={}),
                )
            ),
            controller_state=self.controller.init(controller_key),
        )

    def step(
        self, state: TDMPC2State, key: Key, timestep: Timestep, temperature: float = 1.0
    ) -> tuple[TDMPC2State, Array, PyTree]:
        z = self.encoder.apply(
            self.part(state.world_model_params, "encoder"), timestep.obs
        )
        controller_state, action = self.plan(
            state.world_model_params,
            state.params,
            state.target_critic_params,
            z,
            state.controller_state,
            timestep.done,
            key,
            temperature,
        )
        return state.replace(controller_state=controller_state), action, {}

    def plan(
        self,
        world_model_params: PyTree,
        actor_params: PyTree,
        target_critic_params: PyTree,
        z0: Array,
        controller_state: PyTree,
        done: Array,
        key: Key,
        temperature: float,
    ) -> tuple[PyTree, Array]:
        dynamics_params = self.part(world_model_params, "dynamics")
        reward_params = self.part(world_model_params, "reward")
        termination_params = self.part(world_model_params, "termination")

        num_envs, latent_dim = z0.shape
        horizon = self.cfg.horizon
        num_policy_trajectories = self.cfg.num_policy_trajectories

        def sample_fn(key: Key, mean: Array, std: Array, num_proposals: int) -> Array:
            policy_key, gaussian_key = jax.random.split(key)
            num_random_trajectories = num_proposals - num_policy_trajectories

            z_policy = jnp.broadcast_to(
                jnp.expand_dims(z0, axis=1),
                (num_envs, num_policy_trajectories, latent_dim),
            )

            def policy_step(z: Array, step_key: Key) -> tuple[Array, Array]:
                dist = self.actor.apply(actor_params, z, temperature=1.0)
                action = dist.sample(seed=step_key)
                return self.dynamics.apply(dynamics_params, z, action), action

            _, policy_actions = jax.lax.scan(
                policy_step, z_policy, jax.random.split(policy_key, horizon)
            )
            policy_actions = jnp.moveaxis(policy_actions, 0, 2)

            noise = jax.random.normal(
                gaussian_key,
                (num_envs, num_random_trajectories, horizon, self.cfg.action_dim),
            )
            random_actions = jnp.clip(
                jnp.expand_dims(mean, axis=1) + jnp.expand_dims(std, axis=1) * noise,
                -1.0,
                1.0,
            )

            return jnp.concatenate([policy_actions, random_actions], axis=1)

        def rollout_fn(key: Key, actions: Array) -> Array:
            num_candidates = actions.shape[1]
            z = jnp.broadcast_to(
                jnp.expand_dims(z0, axis=1), (num_envs, num_candidates, latent_dim)
            )
            actions_tm = jnp.moveaxis(actions, 2, 0)

            def imagine_step(
                carry: tuple[Array, Array, Array, Array], a_t: Array
            ) -> tuple[tuple[Array, Array, Array, Array], None]:
                z, returns, discount, terminated = carry
                reward_logits = self.reward.apply(reward_params, z, a_t)
                reward_hat = self.expected_value(reward_logits)
                z_next = self.dynamics.apply(dynamics_params, z, a_t)
                termination_prob = jax.nn.sigmoid(
                    jnp.squeeze(
                        self.termination.apply(termination_params, z_next), axis=-1
                    )
                )
                next_terminated = jnp.clip(
                    terminated + (termination_prob > 0.5).astype(jnp.float32), max=1.0
                )
                return (
                    z_next,
                    returns + discount * (1.0 - terminated) * reward_hat,
                    discount * self.cfg.gamma,
                    next_terminated,
                ), None

            initial_carry = (
                z,
                jnp.zeros((num_envs, num_candidates)),
                jnp.ones((num_envs, num_candidates)),
                jnp.zeros((num_envs, num_candidates)),
            )
            (z, returns, discount_final, terminated), _ = jax.lax.scan(
                imagine_step, initial_carry, actions_tm
            )

            next_action = self.actor.apply(actor_params, z, temperature=1.0).sample(
                seed=key
            )
            next_q_logits = self.critic.apply(target_critic_params, z, next_action)
            next_q_value = self.expected_value(next_q_logits).mean(axis=0)
            return returns + discount_final * (1.0 - terminated) * next_q_value

        controller_state, action = self.controller.step(
            key, controller_state, rollout_fn, sample_fn, done, temperature
        )
        action = jnp.clip(action, -1.0, 1.0)
        return controller_state, action

    def expected_value(self, logits: Array) -> Array:
        probs = jax.nn.softmax(logits, axis=-1)
        return symexp(jnp.sum(probs * self.cfg.bin_centers, axis=-1))

    def update(
        self, state: TDMPC2State, key: Key, transitions: Transition
    ) -> TDMPC2State:
        def extract_window(
            trajectory: Transition,
        ) -> tuple[Array, Array, Array, Array, Array]:
            obs = trajectory.first.obs
            actions = jax.lax.slice_in_dim(trajectory.second.action, 0, -1, axis=1)
            rewards = jax.lax.slice_in_dim(trajectory.second.reward, 0, -1, axis=1)
            terminated = jax.lax.slice_in_dim(
                trajectory.second.terminated, 0, -1, axis=1
            )
            truncated = jax.lax.slice_in_dim(
                trajectory.second.truncated, 0, -1, axis=1
            )
            return jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1),
                (obs, actions, rewards, terminated, truncated),
            )

        def target_probs(target: Array) -> Array:
            target = jnp.clip(symlog(target), self.cfg.vmin, self.cfg.vmax)
            cdf = jax.scipy.special.erf(
                (self.cfg.support - jnp.expand_dims(target, axis=-1))
                / (jnp.sqrt(2) * self.cfg.sigma)
            )
            probs = jnp.diff(cdf, axis=-1)
            return probs / probs.sum(axis=-1, keepdims=True)

        def reward_loss_fn(reward_logits: Array, rewards: Array) -> Array:
            return optax.softmax_cross_entropy(reward_logits, target_probs(rewards))

        def value_loss_fn(q_logits: Array, target: Array) -> Array:
            target_probs_ = jnp.expand_dims(target_probs(target), axis=1)
            return optax.softmax_cross_entropy(q_logits, target_probs_)

        def world_model_loss_fn(
            world_model_params: PyTree, state: TDMPC2State, window: tuple, key: Key
        ) -> tuple[Array, PyTree]:
            encoder_params = self.part(world_model_params, "encoder")
            dynamics_params = self.part(world_model_params, "dynamics")
            reward_params = self.part(world_model_params, "reward")
            critic_params = self.part(world_model_params, "critic")
            termination_params = self.part(world_model_params, "termination")
            next_action_key, subset_key = jax.random.split(key)

            obs, actions, rewards, terminated, truncated = window
            terminated = terminated.astype(jnp.float32)
            truncated = truncated.astype(jnp.float32)
            within = jnp.cumprod(1.0 - jnp.maximum(terminated, truncated), axis=0)
            before = jnp.concatenate([jnp.ones_like(within[:1]), within[:-1]], axis=0)

            z0 = self.encoder.apply(encoder_params, obs[0])

            def predict_step(
                z: Array, a: Array
            ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
                reward_logits = self.reward.apply(reward_params, z, a)
                q_logits = self.critic.apply(critic_params, z, a)
                z_next = self.dynamics.apply(dynamics_params, z, a)
                termination_logits = self.termination.apply(termination_params, z_next)
                return z_next, (z_next, reward_logits, q_logits, termination_logits)

            _, (z_imagined, reward_logits, q_logits, termination_logits) = jax.lax.scan(
                predict_step, z0, actions
            )

            z_target = jax.lax.stop_gradient(
                self.encoder.apply(encoder_params, obs[1:])
            )
            consistency_loss = within * jnp.mean((z_imagined - z_target) ** 2, axis=-1)

            reward_loss = before * reward_loss_fn(reward_logits, rewards)

            next_action, _ = self.actor.apply(
                state.params, z_target, temperature=1.0
            ).sample_and_log_prob(seed=next_action_key)
            next_q_logits = self.critic.apply(
                state.target_critic_params, z_target, next_action
            )
            subset_idx = jax.random.choice(
                subset_key,
                self.cfg.num_critics,
                (self.cfg.num_target_critics,),
                replace=False,
            )
            next_q_value = self.expected_value(next_q_logits[subset_idx])
            bootstrap = jnp.min(next_q_value, axis=0)
            target = rewards + self.cfg.gamma * (1.0 - terminated) * bootstrap
            target = jax.lax.stop_gradient(target)

            value_loss = value_loss_fn(q_logits, target).mean(axis=1)
            value_loss = before * value_loss * (1.0 - truncated)

            termination_loss = jnp.sum(
                before
                * optax.sigmoid_binary_cross_entropy(
                    jnp.squeeze(termination_logits, axis=-1), terminated
                )
            ) / jnp.maximum(before.sum(), 1.0)

            temporal_weights = self.cfg.temporal_decay ** jnp.arange(self.cfg.horizon)
            per_step_loss = (
                self.cfg.consistency_coefficient * consistency_loss
                + self.cfg.reward_coefficient * reward_loss
                + self.cfg.value_coefficient * value_loss
            )
            loss = (jnp.expand_dims(temporal_weights, axis=1) * per_step_loss).mean()
            loss = loss + self.cfg.termination_coefficient * termination_loss

            z_all = jnp.concatenate([z0[None], z_imagined], axis=0)
            q_value = self.expected_value(q_logits)
            aux = {
                "z": jax.lax.stop_gradient(z_all),
                "q_value": q_value.mean(),
                "consistency_loss": consistency_loss.mean(),
                "reward_loss": reward_loss.mean(),
                "value_loss": value_loss.mean(),
                "termination_loss": termination_loss,
            }
            return loss, aux

        def actor_loss_fn(
            params: PyTree,
            state: TDMPC2State,
            z: Array,
            transitions: Transition,
            key: Key,
        ) -> tuple[Array, Array]:
            action_key, subset_key = jax.random.split(key)
            dist, intermediates = self.actor.apply(
                params, z, temperature=1.0, mutable="intermediates"
            )
            action, log_prob = dist.sample_and_log_prob(seed=action_key)
            q_logits = self.critic.apply(
                self.part(state.world_model_params, "critic"), z, action
            )
            subset_idx = jax.random.choice(
                subset_key,
                self.cfg.num_critics,
                (self.cfg.num_target_critics,),
                replace=False,
            )
            q_value = self.expected_value(q_logits[subset_idx]).mean(axis=0)

            temporal_weights = self.cfg.temporal_decay ** jnp.arange(z.shape[0])
            scaled_entropy = self.cfg.action_dim * log_prob
            per_step_loss = self.cfg.entropy_coefficient * scaled_entropy - q_value
            loss = (jnp.expand_dims(temporal_weights, axis=1) * per_step_loss).mean()

            def apply(params: PyTree) -> PyTree:
                return self.actor.apply(params, z, temperature=1.0)

            for auxiliary_loss in self.auxiliary_losses:
                loss = loss + auxiliary_loss(
                    params=params,
                    apply=apply,
                    transitions=transitions,
                    dist=dist,
                    intermediates=intermediates,
                )
            return loss, log_prob

        buffer_state, _ = jax.lax.scan(
            lambda buffer_state, transition: (
                self.buffer.add(buffer_state, jax.tree.map(add_time_axis, transition)),
                None,
            ),
            state.buffer_state,
            transitions,
        )
        state = state.replace(buffer_state=buffer_state)

        def minibatch_fn(state: TDMPC2State, batch, key: Key) -> TDMPC2State:
            world_model_key, actor_key = jax.random.split(key)
            window = extract_window(batch)

            (world_model_loss, aux), grads = jax.value_and_grad(
                world_model_loss_fn, has_aux=True
            )(state.world_model_params, state, window, world_model_key)
            updates, world_model_optimizer_state = self.world_model_optimizer.update(
                grads["params"],
                state.world_model_optimizer_state,
                state.world_model_params["params"],
            )
            world_model_params = {
                "params": optax.apply_updates(state.world_model_params["params"], updates)
            }
            state = state.replace(
                world_model_params=world_model_params,
                world_model_optimizer_state=world_model_optimizer_state,
                target_critic_params=optax.incremental_update(
                    self.part(world_model_params, "critic"),
                    state.target_critic_params,
                    self.cfg.tau,
                ),
            )

            (actor_loss, log_prob), grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(
                state.params,
                state,
                aux["z"],
                jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), batch),
                actor_key,
            )
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

            lox.log(
                {
                    "world_model/loss": world_model_loss,
                    "world_model/consistency_loss": aux["consistency_loss"],
                    "world_model/reward_loss": aux["reward_loss"],
                    "world_model/termination_loss": aux["termination_loss"],
                    "critic/loss": aux["value_loss"],
                    "critic/q_value": aux["q_value"],
                    "actor/loss": actor_loss,
                    "actor/entropy": -log_prob.mean(),
                }
            )

            return state

        sample_key, update_key = jax.random.split(key)
        sample_keys = jax.random.split(sample_key, self.cfg.updates_per_step)
        update_keys = jax.random.split(update_key, self.cfg.updates_per_step)
        batches = jax.vmap(self.buffer.sample, in_axes=(None, 0))(
            state.buffer_state, sample_keys
        ).experience
        updated, _ = jax.lax.scan(
            lambda state, inputs: (minibatch_fn(state, *inputs), None),
            state,
            (batches, update_keys),
        )
        return conditional_update(
            updated, state, self.buffer.can_sample(state.buffer_state)
        )
