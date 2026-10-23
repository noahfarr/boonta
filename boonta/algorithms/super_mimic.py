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
class Prediction:
    state: Array
    reward: Array
    observation: Array
    termination: Array


class Residual(nn.Module):
    network: nn.Module

    def __call__(self, state: Array, action: Array, context: Array) -> Array:
        return state + self.network(jnp.concatenate([state, action, context], axis=-1))


class Learned(nn.Module):
    network: nn.Module
    inputs: tuple[str, ...]

    def __call__(self, **inputs: Array) -> Array:
        features = jnp.concatenate([inputs[name] for name in self.inputs], axis=-1)
        return jnp.squeeze(self.network(features), axis=-1)


class Analytic(nn.Module):
    function: Callable

    def __call__(self, **inputs: Array) -> Array:
        return self.function(**inputs)


class WorldModel(nn.Module):
    dynamics: nn.Module
    reward: nn.Module
    observation: nn.Module
    termination: nn.Module

    def __call__(self, state: Array, action: Array, context: Array, command: Array, goal: Array) -> Prediction:
        following = self.dynamics(state, action, context)
        return Prediction(
            state=following,
            reward=self.reward(state=state, action=action, following=following, command=command, goal=goal),
            observation=self.observe(following, context),
            termination=self.termination(state=following, command=command, goal=goal),
        )

    def observe(self, state: Array, context: Array) -> Array:
        return self.observation(state=state, context=context)


def fully_observed(state: Array, **_) -> Array:
    return state


def _observed(timestep: Timestep) -> Array:
    return timestep.obs


def _noiseless(key: Key, observation: Array) -> Array:
    return observation


def _nothing(timestep: Timestep) -> Array:
    batch = jnp.shape(timestep.reward)
    return jnp.zeros((*batch, 0), jnp.float32)


def _unchanged(command: Array) -> Array:
    return command


def _pushed(history: Array, observation: Array) -> Array:
    return jnp.concatenate([history[..., 1:, :], observation[..., None, :]], axis=-2)


def _flattened(history: Array) -> Array:
    *batch, length, width = jnp.shape(history)
    return jnp.reshape(history, (*batch, length * width))


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
    context_of: Callable[[Timestep], Array] = _nothing
    command_of: Callable[[Timestep], Array] = _nothing
    goal: Callable[[Array], Array] = _unchanged
    advance: Callable[[Array], Array] = _unchanged
    sensor_noise: Callable[[Key, Array], Array] = _noiseless
    value_estimator: Callable = td_lambda
    auxiliary_losses: tuple[Callable, ...] = ()

    def _act(self, params: PyTree, history: Array, goal: Array, temperature: float):
        return self.actor.apply(params, jnp.concatenate([_flattened(history), goal], axis=-1), temperature=temperature)

    def init(self, key: Key, timestep: Timestep) -> tuple[SuperMimicState, Array]:
        actor_key, critic_key, world_model_key, noise_key = jax.random.split(key, 4)
        state, context = self.state_of(timestep), self.context_of(timestep)
        command = self.command_of(timestep)
        goal = self.goal(command)
        observation = self.sensor_noise(noise_key, self.observation_of(timestep))
        history = jnp.repeat(observation[..., None, :], self.cfg.history, axis=-2)
        params = self.actor.init(actor_key, jnp.concatenate([_flattened(history), goal], axis=-1), temperature=1.0)
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
        fresh = jnp.repeat(observation[..., None, :], self.cfg.history, axis=-2)
        history = jnp.where(timestep.done[..., None, None], fresh, _pushed(carry, observation))
        dist = self._act(state.params, history, self.goal(self.command_of(timestep)), temperature)
        return state, history, dist.sample(seed=action_key), {}

    def update(self, state: SuperMimicState, key: Key, transitions: Transition) -> SuperMimicState:
        learned = state.world_model_params["params"]

        def state_values_fn(critic_params: PyTree, states: Array, goals: Array, context: Array) -> Array:
            *batch, _ = jnp.shape(states)
            context = jnp.broadcast_to(context, (*batch, jnp.shape(context)[-1]))
            values = self.critic.apply(critic_params, jnp.concatenate([states, goals, context], axis=-1))
            return jnp.squeeze(values, axis=-1)

        def world_model_loss_fn(
            world_model_params: PyTree, state: SuperMimicState, window: Transition
        ) -> tuple[Array, dict]:
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
                (jnp.swapaxes(second.action, 0, 1), jnp.swapaxes(self.command_of(first), 0, 1)),
            )
            predictions = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), predictions)

            def mean_fn(error: Array, weight: Array) -> Array:
                return jnp.sum(weight * error) / jnp.maximum(jnp.sum(weight), 1.0)

            terms = {"state": mean_fn(jnp.mean((predictions.state - self.state_of(second)) ** 2, axis=-1), settled)}
            if "reward" in learned:
                terms["reward"] = mean_fn((predictions.reward - second.reward) ** 2, alive)
            if "observation" in learned:
                terms["observation"] = mean_fn(
                    jnp.mean((predictions.observation - self.observation_of(second)) ** 2, axis=-1), settled
                )
            if "termination" in learned:
                terms["termination"] = mean_fn(
                    optax.sigmoid_binary_cross_entropy(
                        predictions.termination, second.terminated.astype(jnp.float32)
                    ),
                    alive,
                )
            return sum(terms.values()), terms

        def history_fn(window: Transition, key: Key) -> Array:
            first = window.first
            _, length = jnp.shape(first.reward)
            positions = jnp.arange(length)
            episode_start = jax.lax.cummax(jnp.where(first.done, positions, 0), axis=1)[:, -1]
            recent = jnp.maximum(positions[-self.cfg.history :], episode_start[:, None])
            observations = jnp.take_along_axis(self.observation_of(first), recent[..., None], axis=1)
            return self.sensor_noise(key, observations)

        def imagine_fn(params: PyTree, state: SuperMimicState, start: Timestep, history: Array, key: Key) -> tuple:
            context = self.context_of(start)

            def act_fn(carry: tuple, key: Key) -> tuple:
                imagined, history, command, survival = carry
                goal = self.goal(command)
                action = self._act(params, history, goal, 1.0).mean()
                prediction = self.world_model.apply(state.world_model_params, imagined, action, context, command, goal)
                termination = jax.nn.sigmoid(prediction.termination)
                history = _pushed(history, self.sensor_noise(key, prediction.observation))
                return (prediction.state, history, self.advance(command), survival * (1.0 - termination)), (
                    imagined,
                    goal,
                    prediction.reward,
                    termination,
                    survival,
                )

            survival = jnp.ones(jnp.shape(start.reward))
            (last, _, last_command, last_survival), (states, goals, rewards, terminations, survivals) = jax.lax.scan(
                jax.checkpoint(act_fn),
                (self.state_of(start), history, self.command_of(start), survival),
                jax.random.split(key, self.cfg.horizon),
            )
            states = jnp.concatenate([states, last[None]], axis=0)
            goals = jnp.concatenate([goals, self.goal(last_command)[None]], axis=0)
            survivals = jnp.concatenate([survivals, last_survival[None]], axis=0)
            return states, goals, context, rewards, terminations, survivals

        def actor_loss_fn(
            params: PyTree, state: SuperMimicState, window: Transition, key: Key
        ) -> tuple[Array, tuple]:
            history_key, imagine_key = jax.random.split(key)
            history = history_fn(window, history_key)
            start = jax.tree.map(lambda leaf: leaf[:, -1], window.first)
            states, goals, context, rewards, terminations, survivals = imagine_fn(
                params, state, start, history, imagine_key
            )
            values = state_values_fn(state.critic_params, states, goals, context)
            discounts = self.cfg.gamma ** jnp.arange(self.cfg.horizon + 1, dtype=jnp.float32)
            imagined_return = jnp.sum(discounts[:-1, None] * survivals[:-1] * rewards, axis=0)
            imagined_return = jnp.mean(imagined_return + discounts[-1] * survivals[-1] * values[-1])
            loss = -imagined_return

            inputs = jnp.concatenate([_flattened(history), goals[0]], axis=-1)
            dist, variables = self.actor.apply(params, inputs, temperature=1.0, mutable=True)

            def apply(params: PyTree) -> PyTree:
                return self.actor.apply(params, inputs, temperature=1.0)

            for auxiliary_loss in self.auxiliary_losses:
                loss = loss + auxiliary_loss(
                    params=params, apply=apply, transitions=window, dist=dist, variables=variables
                )
            return loss, (imagined_return, states, goals, context, rewards, terminations)

        def critic_loss_fn(
            critic_params: PyTree, states: Array, goals: Array, context: Array, targets: Array
        ) -> Array:
            values = state_values_fn(critic_params, states[:-1], goals[:-1], context)
            return jnp.mean(jnp.sum((values - targets) ** 2, axis=0))

        def world_model_minibatch_fn(state: SuperMimicState, key: Key) -> tuple[SuperMimicState, None]:
            window = self.buffer.sample(state.buffer_state, key).experience
            (loss, terms), grads = jax.value_and_grad(world_model_loss_fn, has_aux=True)(
                state.world_model_params, state, window
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
            (_, (imagined_return, states, goals, context, rewards, terminations)), grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(state.params, state, window, imagine_key)
            updates, actor_optimizer_state = self.actor_optimizer.update(
                grads["params"], state.actor_optimizer_state, state.params["params"]
            )
            states, goals, rewards, terminations = jax.lax.stop_gradient((states, goals, rewards, terminations))
            values = state_values_fn(state.critic_params, states, goals, context)
            imagined = Transition(
                first=Timestep(obs=states[:-1]),
                second=Timestep(
                    obs=states[1:],
                    reward=rewards,
                    terminated=terminations,
                    truncated=jnp.zeros_like(terminations, bool),
                ),
                aux={"value": values[1:]},
            )
            targets = self.value_estimator(imagined, self.cfg.gamma, self.cfg.td_lambda)
            num_rollouts, *_ = jnp.shape(context)
            minibatch_size = num_rollouts // self.cfg.critic_minibatches

            def critic_minibatch_fn(carry: tuple, indices: Array) -> tuple[tuple, Array]:
                critic_params, critic_optimizer_state = carry
                loss, grads = jax.value_and_grad(critic_loss_fn)(
                    critic_params,
                    states[:, indices],
                    goals[:, indices],
                    context[indices],
                    targets[:, indices],
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
            critic_loss = jnp.mean(critic_losses)
            lox.log(
                {
                    "actor/imagined_return": imagined_return,
                    "actor/imagined_reward": jnp.mean(rewards),
                    "actor/survival": jnp.mean(jnp.prod(1.0 - terminations, axis=0)),
                    "critic/loss": critic_loss,
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
        world_model_key, policy_key = jax.random.split(key)
        updated, _ = jax.lax.scan(
            world_model_minibatch_fn, state, jax.random.split(world_model_key, self.cfg.world_model_updates)
        )
        updated, _ = jax.lax.scan(policy_minibatch_fn, updated, jax.random.split(policy_key, self.cfg.policy_updates))
        return conditional_update(updated, state, self.buffer.can_sample(state.buffer_state))
