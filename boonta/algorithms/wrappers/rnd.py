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

from ..advantage_estimators import generalized_advantage_estimation
from .wrapper import Wrapper, WrapperState


@struct.dataclass
class RNDState(WrapperState):
    step: Array
    random_network_params: PyTree
    predictor_params: PyTree
    critic_params: PyTree
    curiosity_state: optax.OptState
    intrinsic_returns: Array = struct.field(metadata={"axis": "data"})
    sight: EMA
    surprise: EMA


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
        sight: EMA,
    ) -> Array:
        def frame_fn(params: PyTree, frame: PyTree) -> Array:
            whitened = sight.whiten(frame, self.observation_clip)
            fixed = jax.lax.stop_gradient(
                self.random_network.apply(random_network_params, whitened)
            )
            tried = self.predictor_network.apply(params, whitened)
            return jnp.mean((tried - fixed) ** 2, axis=-1)

        step = jax.checkpoint(frame_fn)
        return jax.lax.map(lambda frame: step(predictor_params, frame), obs)

    def intrinsic_value(self, critic_params: PyTree, obs: PyTree, sight: EMA) -> Array:
        def frame_fn(params: PyTree, frame: PyTree) -> Array:
            whitened = sight.whiten(frame, self.observation_clip)
            return remove_feature_axis(self.critic_network.apply(params, whitened))

        step = jax.checkpoint(frame_fn)
        return jax.lax.map(lambda frame: step(critic_params, frame), obs)

    def accumulate(self, state: RNDState, errors: Array) -> tuple[Array, Array]:
        def scan_fn(carried: Array, step: Array) -> tuple[Array, Array]:
            carried = carried * self.intrinsic_gamma + step
            return carried, carried

        return jax.lax.scan(scan_fn, state.intrinsic_returns, errors)

    def intrinsic_reward(
        self, state: RNDState, transitions: Transition, errors: Array
    ) -> Array:
        return state.surprise.scale(errors) * (1.0 - transitions.second.done)

    def intrinsic_advantage_estimator(
        self, rewards: Array, values: Array, bootstrap: Array
    ) -> tuple[Array, Array]:
        endless = jnp.zeros(rewards.shape, bool)
        transitions = Transition(
            second=Timestep(
                obs=None, reward=rewards, terminated=endless, truncated=endless
            )
        )
        return generalized_advantage_estimation(
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
        self, state: RNDState, key: Key, timestep: Timestep, temperature: float = 1.0
    ) -> tuple[RNDState, Array, PyTree]:
        state = self.synchronize(state)
        algorithm_state, action, aux = self.algorithm.step(
            state.algorithm_state, key, timestep, temperature
        )
        return state.replace(algorithm_state=algorithm_state), action, aux

    def init(self, key: Key, timestep: Timestep) -> RNDState:
        algorithm_key, random_key, predictor_key, critic_key = jax.random.split(key, 4)
        algorithm_state = self.algorithm.init(algorithm_key, timestep)
        sample = jax.tree.map(lambda leaf: leaf[:1], timestep.obs)
        random_network_params = self.random_network.init(random_key, sample)
        predictor_params = self.predictor_network.init(predictor_key, sample)
        critic_params = self.critic_network.init(critic_key, sample)
        return RNDState(
            algorithm_state=algorithm_state,
            step=algorithm_state.step,
            random_network_params=random_network_params,
            predictor_params=predictor_params,
            critic_params=critic_params,
            curiosity_state=self.optimizer.init(
                {"predictor": predictor_params, "critic": critic_params}
            ),
            intrinsic_returns=jnp.zeros_like(timestep.reward, jnp.float32),
            sight=EMA.blank(timestep.obs, axis=(0,), decay=self.statistics_decay),
            surprise=EMA.blank(jnp.zeros(()), axis=(), decay=self.statistics_decay),
        )

    def update(self, state: RNDState, key: Key, transitions: Transition) -> RNDState:
        state = self.synchronize(state)
        steps, num_envs = transitions.second.reward.shape
        assert num_envs % self.num_minibatches == 0, (
            f"the {num_envs} environments do not divide into "
            f"{self.num_minibatches} minibatches. Pick a divisor of {num_envs}."
        )
        sight = state.sight.absorb(transitions.second.obs, axis=(0, 1))
        arrived = transitions.second.obs
        departed = transitions.first.obs

        errors = jax.lax.stop_gradient(
            self.distillation_error(
                state.random_network_params, state.predictor_params, arrived, sight
            )
        )
        intrinsic_returns, returns = self.accumulate(state, errors)
        surprise = state.surprise.absorb(returns, axis=(0, 1))
        rewards = self.intrinsic_reward(state, transitions, errors)
        values = self.intrinsic_value(state.critic_params, departed, sight)
        bootstrap, *_ = self.intrinsic_value(
            state.critic_params,
            jax.tree.map(lambda leaf: leaf[-1:], arrived),
            sight,
        )
        advantages, targets = self.intrinsic_advantage_estimator(
            rewards, values, bootstrap
        )
        targets = jax.lax.stop_gradient(targets)

        def loss(
            params: PyTree,
            arrived: PyTree,
            departed: PyTree,
            targets: Array,
            kept: Array,
        ) -> Array:
            errors = self.distillation_error(
                state.random_network_params, params["predictor"], arrived, sight
            )
            distilled = jnp.sum(errors * kept) / jnp.maximum(jnp.sum(kept), 1.0)
            rated = self.intrinsic_value(params["critic"], departed, sight)
            return distilled + 0.5 * jnp.mean((rated - targets) ** 2)

        def minibatch_fn(carry: tuple, x: tuple) -> tuple[tuple, None]:
            params, curiosity_state = carry
            indices, mask = x
            crop = lambda leaf: jnp.take(leaf, indices, 1)
            kept = jax.random.bernoulli(
                mask, self.update_proportion, (steps, indices.shape[0])
            )
            grads = jax.grad(loss)(
                params,
                jax.tree.map(crop, arrived),
                jax.tree.map(crop, departed),
                crop(targets),
                kept,
            )
            updates, curiosity_state = self.optimizer.update(
                grads, curiosity_state, params
            )
            return (optax.apply_updates(params, updates), curiosity_state), None

        def epoch_fn(carry: tuple, key: Key) -> tuple[tuple, None]:
            permute, mask = jax.random.split(key)
            indices = jax.random.permutation(permute, num_envs).reshape(
                self.num_minibatches, -1
            )
            masks = jax.random.split(mask, self.num_minibatches)
            carry, _ = jax.lax.scan(minibatch_fn, carry, (indices, masks))
            return carry, None

        key, pick = jax.random.split(key)
        (params, curiosity_state), _ = jax.lax.scan(
            epoch_fn,
            (
                {
                    "predictor": state.predictor_params,
                    "critic": state.critic_params,
                },
                state.curiosity_state,
            ),
            jax.random.split(pick, self.update_epochs),
        )

        lox.log(
            {
                "rnd/distillation_error": jnp.mean(errors),
                "rnd/intrinsic_reward": jnp.mean(rewards),
                "rnd/reward_scale": surprise.spread(),
                "rnd/intrinsic_advantage": jnp.mean(jnp.abs(advantages)),
                "rnd/intrinsic_value": jnp.mean(values),
            }
        )

        extrinsic = self.algorithm.advantage_estimator

        def advantage_estimator(
            transitions: Transition, values: Array, bootstrap: Array, *args, **kwargs
        ) -> tuple[Array, Array]:
            extrinsic_advantages, returns = extrinsic(
                transitions, values, bootstrap, *args, **kwargs
            )
            return extrinsic_advantages + self.intrinsic_coefficient * advantages, returns

        algorithm = dataclasses.replace(
            self.algorithm, advantage_estimator=advantage_estimator
        )
        algorithm_state = algorithm.update(state.algorithm_state, key, transitions)
        return state.replace(
            algorithm_state=algorithm_state,
            predictor_params=params["predictor"],
            critic_params=params["critic"],
            curiosity_state=curiosity_state,
            intrinsic_returns=intrinsic_returns,
            sight=sight,
            surprise=surprise,
        )
