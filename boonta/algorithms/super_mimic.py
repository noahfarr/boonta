from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.networks import Prediction, WorldModel
from boonta.utils import Timestep, Transition, canonicalize_dtype, conditional_update, remove_batch_axis
from boonta.utils.typing import Array, Buffer, BufferState, Key, PyTree

from .action_value_estimators import td_lambda


@struct.dataclass(frozen=True)
class SuperMimicConfig:
    horizon: int
    window: int
    history: int
    gamma: float
    td_lambda: float
    world_model_updates: int
    policy_updates: int
    policy_batches: int
    critic_epochs: int
    critic_minibatches: int


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
class Imagination:
    states: Array
    goals: Array
    context: Array
    rewards: Array
    terminations: Array
    survivals: Array


def _observed(timestep: Timestep) -> Array:
    return timestep.obs


def _acted(timestep: Timestep) -> Array:
    return timestep.action


def _noiseless(key: Key, observation: Array) -> Array:
    return observation


def _nothing(timestep: Timestep) -> Array:
    batch = jnp.shape(timestep.reward)
    return jnp.zeros((*batch, 0), jnp.float32)


def _unchanged(command: Array) -> Array:
    return command


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
    state_of: Callable[[Timestep], Array] = _observed
    observation_of: Callable[[Timestep], Array] = _observed
    action_of: Callable[[Timestep], Array] = _acted
    context_of: Callable[[Timestep], Array] = _nothing
    command_of: Callable[[Timestep], Array] = _nothing
    goal: Callable[[Array], Array] = _unchanged
    advance: Callable[[Array], Array] = _unchanged
    sensor_noise: Callable[[Key, Array], Array] = _noiseless
    value_estimator: Callable = td_lambda
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> tuple[SuperMimicState, Array]:
        actor_key, critic_key, world_model_key, noise_key = jax.random.split(key, 4)
        state, context, command = self.state_of(timestep), self.context_of(timestep), self.command_of(timestep)
        goal = self.goal(command)
        history = jnp.tile(self.sensor_noise(noise_key, self.observation_of(timestep)), self.cfg.history)
        params = self.actor.init(actor_key, jnp.concatenate([history, goal], axis=-1), temperature=1.0)
        critic_params = self.critic.init(critic_key, jnp.concatenate([state, goal, context], axis=-1))
        world_model_params = self.world_model.init(world_model_key, state, timestep.action, context, command, goal)
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
            history,
        )

    def step(
        self,
        state: SuperMimicState,
        carry: Array,
        key: Key,
        timestep: Timestep,
        temperature: float = 1.0,
    ) -> tuple[SuperMimicState, Array, Array, PyTree]:
        noise_key, action_key = jax.random.split(key)
        observation = self.sensor_noise(noise_key, self.observation_of(timestep))
        *_, width = jnp.shape(observation)
        history = jnp.where(
            timestep.done[..., None],
            jnp.tile(observation, self.cfg.history),
            jnp.concatenate([carry[..., width:], observation], axis=-1),
        )
        goal = self.goal(self.command_of(timestep))
        dist = self.actor.apply(state.params, jnp.concatenate([history, goal], axis=-1), temperature=temperature)
        return state, history, dist.sample(seed=action_key), {}

    def update(self, state: SuperMimicState, key: Key, transitions: Transition) -> SuperMimicState:
        learned = state.world_model_params["params"]

        def state_values_fn(critic_params: PyTree, states: Array, goals: Array, context: Array) -> Array:
            *batch, _ = jnp.shape(states)
            context = jnp.broadcast_to(context, (*batch, jnp.shape(context)[-1]))
            values = self.critic.apply(critic_params, jnp.concatenate([states, goals, context], axis=-1))
            return jnp.squeeze(values, axis=-1)

        def mean_fn(error: Array, weight: Array) -> Array:
            return jnp.sum(weight * error) / jnp.maximum(jnp.sum(weight), 1.0)

        def history_fn(window: Transition, key: Key) -> Array:
            first = window.first
            _, length = jnp.shape(first.reward)
            positions = jnp.arange(length)
            episode_start = jax.lax.cummax(jnp.where(first.done, positions, 0), axis=1)[:, -1]
            recent = jnp.maximum(positions[-self.cfg.history :], episode_start[:, None])
            observations = jnp.take_along_axis(self.observation_of(first), recent[..., None], axis=1)
            batch, *_ = jnp.shape(observations)
            return jnp.reshape(self.sensor_noise(key, observations), (batch, -1))

        def world_model_loss_fn(world_model_params: PyTree, window: Transition) -> tuple[Array, dict]:
            first, second = jax.tree.map(lambda leaf: leaf[:, : self.cfg.window], (window.first, window.second))
            start = jax.tree.map(lambda leaf: leaf[:, 0], first)
            context = self.context_of(start)
            done = (second.terminated | second.truncated).astype(jnp.float32)
            alive = jnp.concatenate([jnp.ones_like(done[:, :1]), jnp.cumprod(1.0 - done, axis=1)[:, :-1]], axis=1)
            settled = alive * (1.0 - done)

            def predict_fn(predicted: Array, inputs: tuple) -> tuple[Array, Prediction]:
                action, command = inputs
                prediction = self.world_model.apply(
                    world_model_params, predicted, action, context, command, self.goal(command)
                )
                return prediction.state, prediction

            _, predictions = jax.lax.scan(
                predict_fn,
                self.state_of(start),
                (jnp.swapaxes(self.action_of(second), 0, 1), jnp.swapaxes(self.command_of(first), 0, 1)),
            )
            predictions = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), predictions)

            target = self.state_of(second)
            change = target - self.state_of(first)
            weight, count = settled[..., None], jnp.maximum(jnp.sum(settled), 1.0)
            average = jnp.sum(weight * change, axis=(0, 1)) / count
            scale = jax.lax.stop_gradient(jnp.sum(weight * (change - average) ** 2, axis=(0, 1)) / count) + 1e-6
            terms = {"state": mean_fn(jnp.mean((predictions.state - target) ** 2 / scale, axis=-1), settled)}
            if "reward" in learned:
                terms["reward"] = mean_fn((predictions.reward - second.reward) ** 2, alive)
            if "observation" in learned:
                observations = self.world_model.apply(
                    world_model_params,
                    predictions.state,
                    jnp.broadcast_to(context[:, None], (*jnp.shape(settled), jnp.shape(context)[-1])),
                    self.command_of(second),
                    method=WorldModel.observe,
                )
                terms["observation"] = mean_fn(
                    jnp.mean((observations - self.observation_of(second)) ** 2, axis=-1), settled
                )
            if "termination" in learned:
                labels = second.terminated.astype(jnp.float32)
                terms["termination"] = mean_fn(
                    optax.sigmoid_binary_cross_entropy(predictions.termination, labels), alive
                )
            return sum(terms.values()), terms

        def imagine_fn(
            params: PyTree, state: SuperMimicState, start: Timestep, history: Array, key: Key
        ) -> Imagination:
            context = self.context_of(start)

            def act_fn(carry: tuple, key: Key) -> tuple:
                imagined, history, command, survival = carry
                goal = self.goal(command)
                action = self.actor.apply(params, jnp.concatenate([history, goal], axis=-1), temperature=1.0).mean()
                prediction = self.world_model.apply(state.world_model_params, imagined, action, context, command, goal)
                termination = jax.nn.sigmoid(prediction.termination)
                following = self.advance(command)
                observation = self.world_model.apply(
                    state.world_model_params, prediction.state, context, following, method=WorldModel.observe
                )
                *_, width = jnp.shape(observation)
                history = jnp.concatenate([history[..., width:], self.sensor_noise(key, observation)], axis=-1)
                carry = (prediction.state, history, following, survival * (1.0 - termination))
                return carry, (imagined, goal, prediction.reward, termination, survival)

            carry = (self.state_of(start), history, self.command_of(start), jnp.ones(jnp.shape(start.reward)))
            (last, _, last_command, last_survival), (states, goals, rewards, terminations, survivals) = jax.lax.scan(
                jax.checkpoint(act_fn), carry, jax.random.split(key, self.cfg.horizon)
            )
            return Imagination(
                states=jnp.concatenate([states, last[None]], axis=0),
                goals=jnp.concatenate([goals, self.goal(last_command)[None]], axis=0),
                context=context,
                rewards=rewards,
                terminations=terminations,
                survivals=jnp.concatenate([survivals, last_survival[None]], axis=0),
            )

        def actor_loss_fn(
            params: PyTree, state: SuperMimicState, window: Transition, key: Key
        ) -> tuple[Array, tuple[Array, Imagination]]:
            history_key, imagine_key = jax.random.split(key)
            history = history_fn(window, history_key)
            start = jax.tree.map(lambda leaf: leaf[:, -1], window.first)
            imagination = imagine_fn(params, state, start, history, imagine_key)
            values = state_values_fn(state.critic_params, imagination.states, imagination.goals, imagination.context)
            discounts = self.cfg.gamma ** jnp.arange(self.cfg.horizon + 1, dtype=jnp.float32)
            survivals = imagination.survivals
            imagined_return = jnp.sum(discounts[:-1, None] * survivals[:-1] * imagination.rewards, axis=0)
            imagined_return = jnp.mean(imagined_return + discounts[-1] * survivals[-1] * values[-1])
            loss = -imagined_return

            inputs = jnp.concatenate([history, imagination.goals[0]], axis=-1)
            dist, variables = self.actor.apply(params, inputs, temperature=1.0, mutable=True)

            def apply(params: PyTree) -> PyTree:
                return self.actor.apply(params, inputs, temperature=1.0)

            for auxiliary_loss in self.auxiliary_losses:
                loss = loss + auxiliary_loss(
                    params=params, apply=apply, transitions=window, dist=dist, variables=variables
                )
            return loss, (imagined_return, imagination)

        def critic_loss_fn(
            critic_params: PyTree, states: Array, goals: Array, context: Array, targets: Array
        ) -> Array:
            values = state_values_fn(critic_params, states[:-1], goals[:-1], context)
            return jnp.mean(jnp.sum((values - targets) ** 2, axis=0))

        def calibrate_fn(state: SuperMimicState, key: Key) -> SuperMimicState:
            sample_key, noise_key = jax.random.split(key)
            window = self.buffer.sample(state.buffer_state, sample_key).experience
            start = jax.tree.map(lambda leaf: leaf[:, -1], window.first)
            goal = self.goal(self.command_of(start))
            first, second = jax.tree.map(
                lambda leaf: jnp.reshape(leaf, (-1, *jnp.shape(leaf)[2:])), (window.first, window.second)
            )
            commands = self.command_of(first)
            _, actor_statistics = self.actor.apply(
                state.params,
                jnp.concatenate([history_fn(window, noise_key), goal], axis=-1),
                temperature=1.0,
                mutable=["statistics"],
            )
            _, critic_statistics = self.critic.apply(
                state.critic_params,
                jnp.concatenate([self.state_of(start), goal, self.context_of(start)], axis=-1),
                mutable=["statistics"],
            )
            _, world_model_statistics = self.world_model.apply(
                state.world_model_params,
                self.state_of(first),
                self.action_of(second),
                self.context_of(first),
                commands,
                self.goal(commands),
                mutable=["statistics"],
            )
            return state.replace(
                params={**state.params, **actor_statistics},
                critic_params={**state.critic_params, **critic_statistics},
                world_model_params={**state.world_model_params, **world_model_statistics},
            )

        def world_model_minibatch_fn(state: SuperMimicState, key: Key) -> tuple[SuperMimicState, None]:
            window = self.buffer.sample(state.buffer_state, key).experience
            (loss, terms), grads = jax.value_and_grad(world_model_loss_fn, has_aux=True)(
                state.world_model_params, window
            )
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

        def policy_minibatch_fn(state: SuperMimicState, key: Key) -> tuple[SuperMimicState, None]:
            sample_key, imagine_key, shuffle_key = jax.random.split(key, 3)
            windows = jax.vmap(lambda key: self.buffer.sample(state.buffer_state, key).experience)(
                jax.random.split(sample_key, self.cfg.policy_batches)
            )
            window = jax.tree.map(lambda leaf: jnp.reshape(leaf, (-1, *jnp.shape(leaf)[2:])), windows)
            (_, (imagined_return, imagination)), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
                state.params, state, window, imagine_key
            )
            updates, actor_optimizer_state = self.actor_optimizer.update(
                grads["params"], state.actor_optimizer_state, state.params["params"]
            )
            imagination = jax.lax.stop_gradient(imagination)
            states, goals, context = imagination.states, imagination.goals, imagination.context
            values = state_values_fn(state.critic_params, states, goals, context)
            imagined = Transition(
                first=Timestep(obs=states[:-1]),
                second=Timestep(
                    obs=states[1:],
                    reward=imagination.rewards,
                    terminated=imagination.terminations,
                    truncated=jnp.zeros_like(imagination.terminations, bool),
                ),
                aux={"value": values[1:]},
            )
            targets = self.value_estimator(imagined, self.cfg.gamma, self.cfg.td_lambda)
            num_rollouts, *_ = jnp.shape(context)
            minibatch_size = num_rollouts // self.cfg.critic_minibatches

            def critic_minibatch_fn(carry: tuple, indices: Array) -> tuple[tuple, Array]:
                critic_params, critic_optimizer_state = carry
                loss, grads = jax.value_and_grad(critic_loss_fn)(
                    critic_params, states[:, indices], goals[:, indices], context[indices], targets[:, indices]
                )
                updates, critic_optimizer_state = self.critic_optimizer.update(
                    grads["params"], critic_optimizer_state, critic_params["params"]
                )
                critic_params = {**critic_params, "params": optax.apply_updates(critic_params["params"], updates)}
                return (critic_params, critic_optimizer_state), loss

            def critic_epoch_fn(carry: tuple, key: Key) -> tuple[tuple, Array]:
                order = jax.random.permutation(key, num_rollouts)[: minibatch_size * self.cfg.critic_minibatches]
                return jax.lax.scan(
                    critic_minibatch_fn, carry, jnp.reshape(order, (self.cfg.critic_minibatches, minibatch_size))
                )

            (critic_params, critic_optimizer_state), critic_losses = jax.lax.scan(
                critic_epoch_fn,
                (state.critic_params, state.critic_optimizer_state),
                jax.random.split(shuffle_key, self.cfg.critic_epochs),
            )
            lox.log(
                {
                    "actor/imagined_return": imagined_return,
                    "actor/imagined_reward": jnp.mean(imagination.rewards),
                    "actor/survival": jnp.mean(jnp.prod(1.0 - imagination.terminations, axis=0)),
                    "critic/loss": jnp.mean(critic_losses),
                    "critic/value": jnp.mean(targets),
                }
            )
            return state.replace(
                params={**state.params, "params": optax.apply_updates(state.params["params"], updates)},
                critic_params=critic_params,
                actor_optimizer_state=actor_optimizer_state,
                critic_optimizer_state=critic_optimizer_state,
            ), None

        transitions = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), transitions.replace(carry=None))
        state = state.replace(buffer_state=self.buffer.add(state.buffer_state, transitions))
        calibrate_key, world_model_key, policy_key = jax.random.split(key, 3)
        updated, _ = jax.lax.scan(
            world_model_minibatch_fn,
            calibrate_fn(state, calibrate_key),
            jax.random.split(world_model_key, self.cfg.world_model_updates),
        )
        updated, _ = jax.lax.scan(policy_minibatch_fn, updated, jax.random.split(policy_key, self.cfg.policy_updates))
        return conditional_update(updated, state, self.buffer.can_sample(state.buffer_state))
