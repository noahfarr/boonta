from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import Timestep, Transition, canonicalize_dtype, conditional_update, remove_batch_axis
from boonta.utils.typing import Array, Buffer, BufferState, Key, PyTree


@struct.dataclass(frozen=True)
class SuperMimicConfig:
    horizon: int
    window: int
    gamma: float
    gae_lambda: float
    world_model_updates: int
    policy_updates: int


@struct.dataclass(frozen=True)
class SuperMimicState:
    step: Array
    params: PyTree
    critic_params: PyTree
    world_model_params: PyTree
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    world_model_optimizer_state: optax.OptState
    buffer_state: BufferState = struct.field(metadata={"axis": "data"})


@struct.dataclass(frozen=True)
class Prediction:
    state: Array
    reward: Array
    observation: Array
    termination: Array


class WorldModel(nn.Module):
    dynamics: nn.Module
    reward: nn.Module
    observation: nn.Module
    termination: nn.Module

    def __call__(self, state: Array, action: Array, context: Array) -> Prediction:
        inputs = jnp.concatenate([state, action, context], axis=-1)
        following = state + self.dynamics(inputs)
        return Prediction(
            state=following,
            reward=jnp.squeeze(self.reward(inputs), axis=-1),
            observation=self.observation(jnp.concatenate([following, context], axis=-1)),
            termination=jnp.squeeze(self.termination(following), axis=-1),
        )

    def observe(self, state: Array, context: Array) -> Array:
        return self.observation(jnp.concatenate([state, context], axis=-1))


def observed(timestep: Timestep) -> Array:
    return timestep.obs


def contextless(timestep: Timestep) -> Array:
    batch = jnp.shape(timestep.reward)
    return jnp.zeros((*batch, 0), jnp.float32)


@dataclass
class SuperMimic:
    cfg: SuperMimicConfig
    actor: nn.Module
    critic: nn.Module
    world_model: WorldModel
    buffer: Buffer
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    world_model_optimizer: optax.GradientTransformation
    state_of: Callable[[Timestep], Array] = observed
    observation_of: Callable[[Timestep], Array] = observed
    context_of: Callable[[Timestep], Array] = contextless
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> tuple[SuperMimicState, PyTree]:
        actor_key, critic_key, world_model_key = jax.random.split(key, 3)
        state, context = self.state_of(timestep), self.context_of(timestep)
        params = self.actor.init(actor_key, self.observation_of(timestep), temperature=1.0)
        critic_params = self.critic.init(critic_key, jnp.concatenate([state, context], axis=-1))
        world_model_params = self.world_model.init(world_model_key, state, timestep.action, context)
        return (
            SuperMimicState(
                step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
                params=params,
                critic_params=critic_params,
                world_model_params=world_model_params,
                actor_optimizer_state=self.actor_optimizer.init(params["params"]),
                critic_optimizer_state=self.critic_optimizer.init(critic_params["params"]),
                world_model_optimizer_state=self.world_model_optimizer.init(world_model_params["params"]),
                buffer_state=self.buffer.init(
                    jax.tree.map(remove_batch_axis, Transition(first=timestep, second=timestep, aux={}))
                ),
            ),
            None,
        )

    def step(
        self,
        state: SuperMimicState,
        carry: PyTree,
        key: Key,
        timestep: Timestep,
        temperature: float = 1.0,
    ) -> tuple[SuperMimicState, PyTree, Array, PyTree]:
        dist = self.actor.apply(state.params, self.observation_of(timestep), temperature=temperature)
        return state, carry, dist.sample(seed=key), {}

    def value(self, critic_params: PyTree, state: Array, context: Array) -> Array:
        *batch, _ = jnp.shape(state)
        context = jnp.broadcast_to(context, (*batch, jnp.shape(context)[-1]))
        return jnp.squeeze(self.critic.apply(critic_params, jnp.concatenate([state, context], axis=-1)), axis=-1)

    def fit(self, world_model_params: PyTree, window: Transition) -> tuple[Array, dict]:
        first, second = window.first, window.second
        context = self.context_of(jax.tree.map(lambda leaf: leaf[:, 0], first))
        done = (second.terminated | second.truncated).astype(jnp.float32)
        alive = jnp.concatenate([jnp.ones_like(done[:, :1]), jnp.cumprod(1.0 - done, axis=1)[:, :-1]], axis=1)
        settled = alive * (1.0 - done)

        def roll(state, step):
            action, = step
            prediction = self.world_model.apply(world_model_params, state, action, context)
            return prediction.state, prediction

        _, predictions = jax.lax.scan(
            roll,
            self.state_of(jax.tree.map(lambda leaf: leaf[:, 0], first)),
            (jnp.swapaxes(second.action, 0, 1),),
        )
        predictions = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), predictions)

        def mean(error, weight):
            return jnp.sum(weight * error) / jnp.maximum(jnp.sum(weight), 1.0)

        terms = {
            "state": mean(jnp.sum((predictions.state - self.state_of(second)) ** 2, axis=-1), settled),
            "reward": mean((predictions.reward - second.reward) ** 2, alive),
            "observation": mean(
                jnp.sum((predictions.observation - self.observation_of(second)) ** 2, axis=-1), settled
            ),
            "termination": mean(
                optax.sigmoid_binary_cross_entropy(predictions.termination, second.terminated.astype(jnp.float32)),
                alive,
            ),
        }
        return sum(terms.values()), terms

    def imagine(
        self, params: PyTree, state: SuperMimicState, starts: Array, observations: Array, context: Array
    ) -> tuple[Array, Array, Array, Array]:
        def act(carry, _):
            imagined, observation, survival = carry
            action = self.actor.apply(params, observation, temperature=1.0).mean()
            prediction = self.world_model.apply(state.world_model_params, imagined, action, context)
            termination = jax.nn.sigmoid(prediction.termination)
            return (prediction.state, prediction.observation, survival * (1.0 - termination)), (
                imagined,
                prediction.reward,
                termination,
                survival,
            )

        survival = jnp.ones(jnp.shape(starts)[:-1])
        (last, _, last_survival), (states, rewards, terminations, survivals) = jax.lax.scan(
            act, (starts, observations, survival), None, length=self.cfg.horizon
        )
        states = jnp.concatenate([states, last[None]], axis=0)
        survivals = jnp.concatenate([survivals, last_survival[None]], axis=0)
        return states, rewards, terminations, survivals

    def returns(self, params: PyTree, state: SuperMimicState, starts: Array, observations: Array, context: Array):
        states, rewards, terminations, survivals = self.imagine(params, state, starts, observations, context)
        values = self.value(state.critic_params, states, context)
        discounts = self.cfg.gamma ** jnp.arange(self.cfg.horizon + 1, dtype=jnp.float32)
        objective = jnp.sum(discounts[:-1, None] * survivals[:-1] * rewards, axis=0)
        objective = objective + discounts[-1] * survivals[-1] * values[-1]
        return jnp.mean(objective), (states, rewards, terminations, values)

    def targets(self, rewards: Array, terminations: Array, values: Array) -> Array:
        def back(following, step):
            reward, termination, value = step
            target = reward + self.cfg.gamma * (1.0 - termination) * (
                (1.0 - self.cfg.gae_lambda) * value + self.cfg.gae_lambda * following
            )
            return target, target

        _, targets = jax.lax.scan(back, values[-1], (rewards, terminations, values[1:]), reverse=True)
        return targets

    def update(self, state: SuperMimicState, key: Key, transitions: Transition) -> SuperMimicState:
        transitions = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), transitions)
        state = state.replace(buffer_state=self.buffer.add(state.buffer_state, transitions))
        world_model_key, policy_key = jax.random.split(key)

        def fit_world_model(state: SuperMimicState, key: Key) -> tuple[SuperMimicState, None]:
            window = self.buffer.sample(state.buffer_state, key).experience
            (loss, terms), grads = jax.value_and_grad(self.fit, has_aux=True)(state.world_model_params, window)
            updates, optimizer_state = self.world_model_optimizer.update(
                grads["params"], state.world_model_optimizer_state, state.world_model_params["params"]
            )
            lox.log({"world_model/loss": loss, **{f"world_model/{name}": term for name, term in terms.items()}})
            return state.replace(
                world_model_params={
                    **state.world_model_params,
                    "params": optax.apply_updates(state.world_model_params["params"], updates),
                },
                world_model_optimizer_state=optimizer_state,
            ), None

        def improve_policy(state: SuperMimicState, key: Key) -> tuple[SuperMimicState, None]:
            window = self.buffer.sample(state.buffer_state, key).experience
            first = jax.tree.map(lambda leaf: leaf[:, 0], window.first)
            starts, context = self.state_of(first), self.context_of(first)

            def actor_loss(params):
                objective, imagined = self.returns(params, state, starts, self.observation_of(first), context)
                loss = -objective
                dist, variables = self.actor.apply(
                    params, self.observation_of(first), temperature=1.0, mutable=True
                )

                def apply(params):
                    return self.actor.apply(params, self.observation_of(first), temperature=1.0)

                for auxiliary_loss in self.auxiliary_losses:
                    loss = loss + auxiliary_loss(
                        params=params, apply=apply, transitions=window, dist=dist, variables=variables
                    )
                return loss, (objective, imagined)

            (_, (objective, (states, rewards, terminations, _))), grads = jax.value_and_grad(
                actor_loss, has_aux=True
            )(state.params)
            updates, actor_optimizer_state = self.actor_optimizer.update(
                grads["params"], state.actor_optimizer_state, state.params["params"]
            )
            states, rewards, terminations = jax.lax.stop_gradient((states, rewards, terminations))
            targets = self.targets(rewards, terminations, self.value(state.critic_params, states, context))

            survivals = jax.lax.stop_gradient(jnp.cumprod(jnp.concatenate([jnp.ones_like(terminations[:1]), 1.0 - terminations[:-1]]), axis=0))

            def critic_loss(critic_params):
                values = self.value(critic_params, states[:-1], context)
                return jnp.mean(jnp.sum(survivals * (values - targets) ** 2, axis=0))

            loss, critic_grads = jax.value_and_grad(critic_loss)(state.critic_params)
            critic_updates, critic_optimizer_state = self.critic_optimizer.update(
                critic_grads["params"], state.critic_optimizer_state, state.critic_params["params"]
            )
            lox.log(
                {
                    "actor/imagined_return": objective,
                    "actor/imagined_reward": jnp.mean(rewards),
                    "actor/survival": jnp.mean(jnp.prod(1.0 - terminations, axis=0)),
                    "critic/loss": loss,
                    "critic/value": jnp.mean(targets),
                }
            )
            return state.replace(
                params={**state.params, "params": optax.apply_updates(state.params["params"], updates)},
                critic_params={
                    **state.critic_params,
                    "params": optax.apply_updates(state.critic_params["params"], critic_updates),
                },
                actor_optimizer_state=actor_optimizer_state,
                critic_optimizer_state=critic_optimizer_state,
            ), None

        updated, _ = jax.lax.scan(
            fit_world_model, state, jax.random.split(world_model_key, self.cfg.world_model_updates)
        )
        updated, _ = jax.lax.scan(improve_policy, updated, jax.random.split(policy_key, self.cfg.policy_updates))
        return conditional_update(updated, state, self.buffer.can_sample(state.buffer_state))
