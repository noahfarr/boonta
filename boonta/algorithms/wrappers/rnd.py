import dataclasses
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import (EMA, Array, Key, PyTree, Timestep, Transition,
                          remove_feature_axis)

from ..advantage_estimators import generalized_advantage_estimator
from ..algorithm import Carries
from .wrapper import Wrapper, WrapperState


@struct.dataclass
class RNDState(WrapperState):
    step: Array
    random_network_params: PyTree
    predictor_params: PyTree
    critic_params: PyTree
    curiosity_state: optax.OptState
    intrinsic_returns: Array = struct.field(metadata={"axis": "data"})
    obs_normalizer: EMA
    reward_normalizer: EMA


@dataclass
class RND(Wrapper):
    random_network: nn.Module = None
    predictor_network: nn.Module = None
    critic_network: nn.Module = None
    optimizer: optax.GradientTransformation = None
    intrinsic_coefficient: float = 1.0
    intrinsic_gamma: float = 0.99
    intrinsic_lambda: float = 0.95
    statistics_decay: float = 0.99
    update_proportion: float = 1.0
    update_epochs: int = 1
    num_minibatches: int = 1
    observation_clip: float = 5.0

    def distillation_error(
        self,
        random_network_params: PyTree,
        predictor_params: PyTree,
        obs: PyTree,
        obs_normalizer: EMA,
    ) -> Array:
        def step_fn(network_params: PyTree, step_obs: PyTree) -> Array:
            whitened_obs = obs_normalizer.whiten(step_obs, self.observation_clip)
            target = jax.lax.stop_gradient(
                self.random_network.apply(random_network_params, whitened_obs)
            )
            prediction = self.predictor_network.apply(network_params, whitened_obs)
            return jnp.mean((prediction - target) ** 2, axis=-1)

        checkpointed_fn = jax.checkpoint(step_fn)
        return jax.lax.map(
            lambda step_obs: checkpointed_fn(predictor_params, step_obs), obs
        )

    def intrinsic_value(
        self, critic_params: PyTree, obs: PyTree, obs_normalizer: EMA
    ) -> Array:
        def step_fn(network_params: PyTree, step_obs: PyTree) -> Array:
            whitened_obs = obs_normalizer.whiten(step_obs, self.observation_clip)
            return remove_feature_axis(
                self.critic_network.apply(network_params, whitened_obs)
            )

        checkpointed_fn = jax.checkpoint(step_fn)
        return jax.lax.map(
            lambda step_obs: checkpointed_fn(critic_params, step_obs), obs
        )

    def accumulate(self, state: RNDState, errors: Array) -> tuple[Array, Array]:
        def scan_fn(running_return: Array, step_error: Array) -> tuple[Array, Array]:
            running_return = running_return * self.intrinsic_gamma + step_error
            return running_return, running_return

        return jax.lax.scan(scan_fn, state.intrinsic_returns, errors)

    def intrinsic_reward(
        self, state: RNDState, transitions: Transition, errors: Array
    ) -> Array:
        return state.reward_normalizer.scale(errors) * (1.0 - transitions.second.done)

    def intrinsic_advantage_estimator(
        self, rewards: Array, values: Array, bootstrap: Array
    ) -> tuple[Array, Array]:
        never_done = jnp.zeros(rewards.shape, bool)
        transitions = Transition(
            second=Timestep(
                obs=None, reward=rewards, terminated=never_done, truncated=never_done
            )
        )
        return generalized_advantage_estimator(
            transitions,
            values,
            bootstrap,
            self.intrinsic_gamma,
            self.intrinsic_lambda,
        )

    def synchronize(self, state: RNDState) -> RNDState:
        return state.replace(
            algorithm_state=state.algorithm_state.replace(step=state.step)
        )

    def step(
        self, state: RNDState, carry: PyTree, key: Key, timestep: Timestep, temperature: float = 1.0
    ) -> tuple[RNDState, PyTree, Array, PyTree]:
        state = self.synchronize(state)
        algorithm_state, carry, action, aux = self.algorithm.step(
            state.algorithm_state, carry, key, timestep, temperature
        )
        return state.replace(algorithm_state=algorithm_state), carry, action, aux

    def init(self, key: Key, timestep: Timestep) -> tuple[RNDState, PyTree]:
        algorithm_key, random_key, predictor_key, critic_key = jax.random.split(key, 4)
        algorithm_state, carry = self.algorithm.init(algorithm_key, timestep)
        sample_obs = jax.tree.map(lambda leaf: leaf[:1], timestep.obs)
        random_network_params = self.random_network.init(random_key, sample_obs)
        predictor_params = self.predictor_network.init(predictor_key, sample_obs)
        critic_params = self.critic_network.init(critic_key, sample_obs)
        state = RNDState(
            algorithm_state=algorithm_state,
            step=algorithm_state.step,
            random_network_params=random_network_params,
            predictor_params=predictor_params,
            critic_params=critic_params,
            curiosity_state=self.optimizer.init(
                {"predictor": predictor_params, "critic": critic_params}
            ),
            intrinsic_returns=jnp.zeros_like(timestep.reward, jnp.float32),
            obs_normalizer=EMA.blank(
                timestep.obs, axis=(0,), decay=self.statistics_decay
            ),
            reward_normalizer=EMA.blank(
                jnp.zeros(()), axis=(), decay=self.statistics_decay
            ),
        )
        return state, carry

    def update(
        self, state: RNDState, key: Key, transitions: Transition, carries: Carries
    ) -> RNDState:
        state = self.synchronize(state)
        num_steps, num_envs = transitions.second.reward.shape
        assert num_envs % self.num_minibatches == 0, (
            f"the {num_envs} environments do not divide into "
            f"{self.num_minibatches} minibatches. Pick a divisor of {num_envs}."
        )
        obs_normalizer = state.obs_normalizer.absorb(
            transitions.second.obs, axis=(0, 1)
        )
        next_obs = transitions.second.obs
        obs = transitions.first.obs

        errors = jax.lax.stop_gradient(
            self.distillation_error(
                state.random_network_params,
                state.predictor_params,
                next_obs,
                obs_normalizer,
            )
        )
        intrinsic_returns, running_returns = self.accumulate(state, errors)
        reward_normalizer = state.reward_normalizer.absorb(running_returns, axis=(0, 1))
        intrinsic_rewards = self.intrinsic_reward(state, transitions, errors)
        intrinsic_values = self.intrinsic_value(
            state.critic_params, obs, obs_normalizer
        )
        intrinsic_bootstrap, *_ = self.intrinsic_value(
            state.critic_params,
            jax.tree.map(lambda leaf: leaf[-1:], next_obs),
            obs_normalizer,
        )
        intrinsic_advantages, value_targets = self.intrinsic_advantage_estimator(
            intrinsic_rewards, intrinsic_values, intrinsic_bootstrap
        )
        value_targets = jax.lax.stop_gradient(value_targets)

        def curiosity_loss(
            curiosity_params: PyTree,
            next_obs: PyTree,
            obs: PyTree,
            value_targets: Array,
            loss_mask: Array,
        ) -> Array:
            errors = self.distillation_error(
                state.random_network_params,
                curiosity_params["predictor"],
                next_obs,
                obs_normalizer,
            )
            distillation_loss = jnp.sum(errors * loss_mask) / jnp.maximum(
                jnp.sum(loss_mask), 1.0
            )
            predicted_values = self.intrinsic_value(
                curiosity_params["critic"], obs, obs_normalizer
            )
            return distillation_loss + 0.5 * jnp.mean(
                (predicted_values - value_targets) ** 2
            )

        def minibatch_fn(carry: tuple, minibatch: tuple) -> tuple[tuple, None]:
            curiosity_params, curiosity_state = carry
            indices, mask_key = minibatch
            loss_mask = jax.random.bernoulli(
                mask_key, self.update_proportion, (num_steps, indices.shape[0])
            )
            curiosity_grads = jax.grad(curiosity_loss)(
                curiosity_params,
                jax.tree.map(lambda leaf: jnp.take(leaf, indices, axis=1), next_obs),
                jax.tree.map(lambda leaf: jnp.take(leaf, indices, axis=1), obs),
                jnp.take(value_targets, indices, axis=1),
                loss_mask,
            )
            curiosity_updates, curiosity_state = self.optimizer.update(
                curiosity_grads, curiosity_state, curiosity_params
            )
            curiosity_params = optax.apply_updates(curiosity_params, curiosity_updates)
            return (curiosity_params, curiosity_state), None

        def epoch_fn(carry: tuple, key: Key) -> tuple[tuple, None]:
            permute_key, mask_key = jax.random.split(key)
            indices = jax.random.permutation(permute_key, num_envs).reshape(
                self.num_minibatches, -1
            )
            mask_keys = jax.random.split(mask_key, self.num_minibatches)
            carry, _ = jax.lax.scan(minibatch_fn, carry, (indices, mask_keys))
            return carry, None

        key, epoch_key = jax.random.split(key)
        (curiosity_params, curiosity_state), _ = jax.lax.scan(
            epoch_fn,
            (
                {
                    "predictor": state.predictor_params,
                    "critic": state.critic_params,
                },
                state.curiosity_state,
            ),
            jax.random.split(epoch_key, self.update_epochs),
        )

        lox.log(
            {
                "rnd/distillation_error": jnp.mean(errors),
                "rnd/intrinsic_reward": jnp.mean(intrinsic_rewards),
                "rnd/reward_scale": reward_normalizer.spread(),
                "rnd/intrinsic_advantage": jnp.mean(jnp.abs(intrinsic_advantages)),
                "rnd/intrinsic_value": jnp.mean(intrinsic_values),
            }
        )

        extrinsic_estimator = self.algorithm.advantage_estimator

        def advantage_estimator(
            transitions: Transition, values: Array, bootstrap: Array, *args, **kwargs
        ) -> tuple[Array, Array]:
            extrinsic_advantages, returns = extrinsic_estimator(
                transitions, values, bootstrap, *args, **kwargs
            )
            return (
                extrinsic_advantages
                + self.intrinsic_coefficient * intrinsic_advantages,
                returns,
            )

        curious_algorithm = dataclasses.replace(
            self.algorithm, advantage_estimator=advantage_estimator
        )
        algorithm_state = curious_algorithm.update(
            state.algorithm_state, key, transitions, carries
        )
        return state.replace(
            algorithm_state=algorithm_state,
            predictor_params=curiosity_params["predictor"],
            critic_params=curiosity_params["critic"],
            curiosity_state=curiosity_state,
            intrinsic_returns=intrinsic_returns,
            obs_normalizer=obs_normalizer,
            reward_normalizer=reward_normalizer,
        )
