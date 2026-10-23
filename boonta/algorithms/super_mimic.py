from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
class Imagination:
    states: Array
    goals: Array
    contexts: Array
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
    batch = jnp.shape(timestep.terminated)
    return jnp.zeros((*batch, 0), jnp.float32)


@dataclass
class SuperMimic:
    cfg: SuperMimicConfig
    actor: nn.Module
    critic: nn.Module
    world_model: Any
    buffer: Buffer
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    world_model_optimizer: optax.GradientTransformation
    state_of: Callable[[Timestep], Array] = _observed
    observation_of: Callable[[Timestep], Array] = _observed
    action_of: Callable[[Timestep], Array] = _acted
    context_of: Callable[[Timestep], Array] = _nothing
    command_of: Callable[[Timestep], Array] = _nothing
    sensor_noise: Callable[[Key, Array], Array] = _noiseless
    value_estimator: Callable = td_lambda
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> tuple[SuperMimicState, Array]:
        actor_key, critic_key, world_model_key, noise_key = jax.random.split(key, 4)
        state, context, command = self.state_of(timestep), self.context_of(timestep), self.command_of(timestep)
        goal = self.world_model.goal(command)
        history = jnp.tile(self.sensor_noise(noise_key, self.observation_of(timestep)), self.cfg.history)
        params = self.actor.init(actor_key, jnp.concatenate([history, goal], axis=-1), temperature=1.0)
        critic_params = self.critic.init(critic_key, jnp.concatenate([state, goal, context], axis=-1))
        world_model_params = self.world_model.initialize(world_model_key, (state, context, command), timestep.action)
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
        goal = self.world_model.goal(self.command_of(timestep))
        dist = self.actor.apply(state.params, jnp.concatenate([history, goal], axis=-1), temperature=temperature)
        return state, history, dist.sample(seed=action_key), {}

    def update(self, state: SuperMimicState, key: Key, transitions: Transition) -> SuperMimicState:
        learned = self.world_model.learned(state.world_model_params)

        def state_values_fn(critic_params: PyTree, states: Array, goals: Array, contexts: Array) -> Array:
            values = self.critic.apply(critic_params, jnp.concatenate([states, goals, contexts], axis=-1))
            return jnp.squeeze(values, axis=-1)

        def mean_fn(error: Array, weight: Array) -> Array:
            return jnp.sum(weight * error) / jnp.maximum(jnp.sum(weight), 1.0)

        def history_fn(window: Transition, key: Key) -> Array:
            first = window.first
            _, length = jnp.shape(first.reward)
            positions = jnp.arange(length)
            episode_start = jax.lax.cummax(jnp.where(first.done, positions, 0), axis=1)[:, -1]
            recent = jnp.maximum(positions[-self.cfg.history - 1 :], episode_start[:, None])
            observations = jnp.take_along_axis(self.observation_of(first), recent[..., None], axis=1)
            batch, *_ = jnp.shape(observations)
            return jnp.reshape(self.sensor_noise(key, observations), (batch, -1))

        def begin_fn(params: PyTree, start: Timestep, key: Key) -> tuple[PyTree, Timestep]:
            world, _ = self.world_model.init(key)
            theta = (self.state_of(start), self.context_of(start), self.command_of(start))
            world = self.world_model.update(world, key, params=params, theta=theta)
            flags = jnp.zeros(jnp.shape(start.reward), bool)
            return world, Timestep(obs=self.world_model.observe(world), terminated=flags, truncated=flags)

        def world_model_loss_fn(world_model_params: PyTree, window: Transition, key: Key) -> tuple[Array, dict]:
            first, second = jax.tree.map(lambda leaf: leaf[:, : self.cfg.window], (window.first, window.second))
            start = jax.tree.map(lambda leaf: leaf[:, 0], first)
            done = (second.terminated | second.truncated).astype(jnp.float32)
            alive = jnp.concatenate([jnp.ones_like(done[:, :1]), jnp.cumprod(1.0 - done, axis=1)[:, :-1]], axis=1)
            settled = alive * (1.0 - done)

            def predict_fn(world: PyTree, inputs: tuple) -> tuple[PyTree, Timestep]:
                action, key = inputs
                return self.world_model.step(key, world, action)

            begin_key, step_key = jax.random.split(key)
            world, _ = begin_fn(world_model_params, start, begin_key)
            _, predicted = jax.lax.scan(
                predict_fn,
                world,
                (jnp.swapaxes(self.action_of(second), 0, 1), jax.random.split(step_key, self.cfg.window)),
            )
            predicted = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), predicted)

            target = self.state_of(second)
            change = target - self.state_of(first)
            weight, count = settled[..., None], jnp.maximum(jnp.sum(settled), 1.0)
            average = jnp.sum(weight * change, axis=(0, 1)) / count
            scale = jax.lax.stop_gradient(jnp.sum(weight * (change - average) ** 2, axis=(0, 1)) / count) + 1e-6
            error = jnp.mean((self.state_of(predicted) - target) ** 2 / scale, axis=-1)
            terms = {"state": mean_fn(error, settled)}
            if "reward" in learned:
                terms["reward"] = mean_fn((predicted.reward - second.reward) ** 2, alive)
            if "observation" in learned:
                error = jnp.mean((self.observation_of(predicted) - self.observation_of(second)) ** 2, axis=-1)
                terms["observation"] = mean_fn(error, settled)
            if "termination" in learned:
                labels = second.terminated.astype(jnp.float32)
                terms["termination"] = mean_fn(
                    optax.sigmoid_binary_cross_entropy(predicted.info["termination"], labels), alive
                )
            return sum(terms.values()), terms

        def imagine_fn(
            params: PyTree, state: SuperMimicState, start: Timestep, history: Array, key: Key
        ) -> Imagination:
            acting = state.replace(params=params)

            def act_fn(carry: tuple, key: Key) -> tuple:
                world, timestep, history, survival = carry
                act_key, step_key = jax.random.split(key)
                _, history, action, _ = self.step(acting, history, act_key, timestep, temperature=0.0)
                world, following = self.world_model.step(step_key, world, action)
                termination = jax.nn.sigmoid(following.info["termination"])
                carry = (
                    world,
                    Timestep(obs=following.obs, terminated=following.terminated, truncated=following.truncated),
                    history,
                    survival * (1.0 - termination),
                )
                return carry, (timestep, following.reward, termination, survival)

            begin_key, key = jax.random.split(key)
            world, timestep = begin_fn(state.world_model_params, start, begin_key)
            carry = (world, timestep, history, jnp.ones(jnp.shape(start.reward)))
            (_, last, _, last_survival), (timesteps, rewards, terminations, survivals) = jax.lax.scan(
                jax.checkpoint(act_fn), carry, jax.random.split(key, self.cfg.horizon)
            )
            timesteps = jax.tree.map(lambda steps, final: jnp.concatenate([steps, final[None]]), timesteps, last)
            return Imagination(
                states=self.state_of(timesteps),
                goals=jax.vmap(self.world_model.goal)(self.command_of(timesteps)),
                contexts=self.context_of(timesteps),
                rewards=rewards,
                terminations=terminations,
                survivals=jnp.concatenate([survivals, last_survival[None]], axis=0),
            )

        def actor_loss_fn(
            params: PyTree, state: SuperMimicState, window: Transition, key: Key
        ) -> tuple[Array, tuple[Array, Imagination]]:
            history_key, imagine_key = jax.random.split(key)
            start = jax.tree.map(lambda leaf: leaf[:, -1], window.first)
            observations = history_fn(window, history_key)
            *_, width = jnp.shape(self.observation_of(start))
            imagination = imagine_fn(params, state, start, observations[..., :-width], imagine_key)
            values = state_values_fn(
                state.critic_params, imagination.states, imagination.goals, imagination.contexts
            )
            discounts = self.cfg.gamma ** jnp.arange(self.cfg.horizon + 1, dtype=jnp.float32)
            survivals = imagination.survivals
            imagined_return = jnp.sum(discounts[:-1, None] * survivals[:-1] * imagination.rewards, axis=0)
            imagined_return = jnp.mean(imagined_return + discounts[-1] * survivals[-1] * values[-1])
            loss = -imagined_return

            inputs = jnp.concatenate([observations[..., width:], imagination.goals[0]], axis=-1)
            dist, variables = self.actor.apply(params, inputs, temperature=1.0, mutable=True)

            def apply(params: PyTree) -> PyTree:
                return self.actor.apply(params, inputs, temperature=1.0)

            for auxiliary_loss in self.auxiliary_losses:
                loss = loss + auxiliary_loss(
                    params=params, apply=apply, transitions=window, dist=dist, variables=variables
                )
            return loss, (imagined_return, imagination)

        def critic_loss_fn(
            critic_params: PyTree, states: Array, goals: Array, contexts: Array, targets: Array
        ) -> Array:
            values = state_values_fn(critic_params, states[:-1], goals[:-1], contexts[:-1])
            return jnp.mean(jnp.sum((values - targets) ** 2, axis=0))

        def calibrate_fn(state: SuperMimicState, key: Key) -> SuperMimicState:
            sample_key, noise_key = jax.random.split(key)
            window = self.buffer.sample(state.buffer_state, sample_key).experience
            start = jax.tree.map(lambda leaf: leaf[:, -1], window.first)
            goal = self.world_model.goal(self.command_of(start))
            first, second = jax.tree.map(
                lambda leaf: jnp.reshape(leaf, (-1, *jnp.shape(leaf)[2:])), (window.first, window.second)
            )
            *_, width = jnp.shape(self.observation_of(start))
            history = history_fn(window, noise_key)[..., width:]
            _, actor_statistics = self.actor.apply(
                state.params, jnp.concatenate([history, goal], axis=-1), temperature=1.0, mutable=["statistics"]
            )
            _, critic_statistics = self.critic.apply(
                state.critic_params,
                jnp.concatenate([self.state_of(start), goal, self.context_of(start)], axis=-1),
                mutable=["statistics"],
            )
            return state.replace(
                params={**state.params, **actor_statistics},
                critic_params={**state.critic_params, **critic_statistics},
                world_model_params=self.world_model.calibrate(
                    state.world_model_params,
                    (self.state_of(first), self.context_of(first), self.command_of(first)),
                    self.action_of(second),
                ),
            )

        def world_model_minibatch_fn(state: SuperMimicState, key: Key) -> tuple[SuperMimicState, None]:
            sample_key, step_key = jax.random.split(key)
            window = self.buffer.sample(state.buffer_state, sample_key).experience
            (loss, terms), grads = jax.value_and_grad(world_model_loss_fn, has_aux=True)(
                state.world_model_params, window, step_key
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
            states, goals, contexts = imagination.states, imagination.goals, imagination.contexts
            values = state_values_fn(state.critic_params, states, goals, contexts)
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
            _, num_rollouts = jnp.shape(imagination.rewards)
            minibatch_size = num_rollouts // self.cfg.critic_minibatches

            def critic_minibatch_fn(carry: tuple, indices: Array) -> tuple[tuple, Array]:
                critic_params, critic_optimizer_state = carry
                loss, grads = jax.value_and_grad(critic_loss_fn)(
                    critic_params, states[:, indices], goals[:, indices], contexts[:, indices], targets[:, indices]
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
