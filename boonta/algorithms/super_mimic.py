from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.networks import WorldModel
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


def _noiseless(key: Key, observation: Array) -> Array:
    return observation


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
    observation_key: str = "observation"
    state_key: str = "state"
    context_key: str = "context"
    command_key: str = "command"
    sensor_noise: Callable[[Key, Array], Array] = _noiseless
    value_estimator: Callable = td_lambda
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> tuple[SuperMimicState, Array]:
        actor_key, critic_key, world_model_key, noise_key = jax.random.split(key, 4)
        obs = timestep.obs
        state, context, command = obs[self.state_key], obs[self.context_key], obs[self.command_key]
        goal = self.world_model.goal(command)
        history = jnp.tile(self.sensor_noise(noise_key, timestep.obs[self.observation_key]), self.cfg.history)
        params = self.actor.init(actor_key, jnp.concatenate([history, goal], axis=-1), temperature=1.0)
        critic_params = self.critic.init(critic_key, jnp.concatenate([state, goal, context], axis=-1))
        world_model_params = self.world_model.init(world_model_key, state, timestep.action, context, command)
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
        observation = self.sensor_noise(noise_key, timestep.obs[self.observation_key])
        *_, width = jnp.shape(observation)
        history = jnp.where(
            timestep.done[..., None],
            jnp.tile(observation, self.cfg.history),
            jnp.concatenate([carry[..., width:], observation], axis=-1),
        )
        goal = self.world_model.goal(timestep.obs[self.command_key])
        dist = self.actor.apply(state.params, jnp.concatenate([history, goal], axis=-1), temperature=temperature)
        return state, history, dist.sample(seed=action_key), {}

    def update(self, state: SuperMimicState, key: Key, transitions: Transition) -> SuperMimicState:
        learned = state.world_model_params["params"]

        def state_values_fn(critic_params: PyTree, states: Array, goals: Array, contexts: Array) -> Array:
            values = self.critic.apply(critic_params, jnp.concatenate([states, goals, contexts], axis=-1))
            return jnp.squeeze(values, axis=-1)

        def predict_fn(world_model_params: PyTree, obs: PyTree, action: Array) -> tuple[PyTree, Array, Array]:
            following, observation, command, reward, termination = self.world_model.apply(
                world_model_params, obs[self.state_key], action, obs[self.context_key], obs[self.command_key]
            )
            obs = {**obs, self.state_key: following, self.observation_key: observation, self.command_key: command}
            return obs, reward, termination

        def merge_fn(leaf: Array) -> Array:
            outer, inner, *rest = jnp.shape(leaf)
            return jnp.reshape(leaf, (outer * inner, *rest))

        def mean_fn(error: Array, weight: Array) -> Array:
            return jnp.sum(weight * error) / jnp.maximum(jnp.sum(weight), 1.0)

        def history_fn(window: Transition, key: Key) -> Array:
            first = window.first
            _, length = jnp.shape(first.reward)
            positions = jnp.arange(length)
            episode_start = jax.lax.cummax(jnp.where(first.done, positions, 0), axis=1)[:, -1]
            recent = jnp.maximum(positions[-self.cfg.history - 1 :], episode_start[:, None])
            observations = jnp.take_along_axis(first.obs[self.observation_key], recent[..., None], axis=1)
            batch, *_ = jnp.shape(observations)
            return jnp.reshape(self.sensor_noise(key, observations), (batch, -1))

        def world_model_loss_fn(world_model_params: PyTree, window: Transition) -> tuple[Array, dict]:
            first, second = jax.tree.map(lambda leaf: leaf[:, : self.cfg.window], (window.first, window.second))
            start = jax.tree.map(lambda leaf: leaf[:, 0], first)
            done = (second.terminated | second.truncated).astype(jnp.float32)
            alive = jnp.concatenate([jnp.ones_like(done[:, :1]), jnp.cumprod(1.0 - done, axis=1)[:, :-1]], axis=1)
            settled = alive * (1.0 - done)

            def rollout_fn(obs: PyTree, action: Array) -> tuple[PyTree, tuple]:
                prediction = predict_fn(world_model_params, obs, action)
                obs, *_ = prediction
                return obs, prediction

            _, predicted = jax.lax.scan(rollout_fn, start.obs, jnp.swapaxes(second.action, 0, 1))
            predictions, rewards, terminations = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), predicted)

            target = second.obs[self.state_key]
            change = target - first.obs[self.state_key]
            weight, count = settled[..., None], jnp.maximum(jnp.sum(settled), 1.0)
            average = jnp.sum(weight * change, axis=(0, 1)) / count
            scale = jax.lax.stop_gradient(jnp.sum(weight * (change - average) ** 2, axis=(0, 1)) / count) + 1e-6
            error = jnp.mean((predictions[self.state_key] - target) ** 2 / scale, axis=-1)
            terms = {"state": mean_fn(error, settled)}
            if "reward_fn" in learned:
                terms["reward"] = mean_fn((rewards - second.reward) ** 2, alive)
            if "observation_fn" in learned:
                error = jnp.mean((predictions[self.observation_key] - second.obs[self.observation_key]) ** 2, axis=-1)
                terms["observation"] = mean_fn(error, settled)
            if "termination_fn" in learned:
                labels = second.terminated.astype(jnp.float32)
                terms["termination"] = mean_fn(optax.sigmoid_binary_cross_entropy(terminations, labels), alive)
            return sum(terms.values()), terms

        def imagine_fn(
            params: PyTree, state: SuperMimicState, start: Timestep, history: Array, key: Key
        ) -> Imagination:
            acting = state.replace(params=params)

            def act_fn(carry: tuple, key: Key) -> tuple:
                timestep, history, survival = carry
                _, history, action, _ = self.step(acting, history, key, timestep, temperature=0.0)
                obs, reward, termination = predict_fn(state.world_model_params, timestep.obs, action)
                termination = jax.nn.sigmoid(termination)
                flags = jnp.zeros_like(reward, bool)
                following = Timestep(obs=obs, terminated=flags, truncated=flags)
                return (following, history, survival * (1.0 - termination)), (timestep, reward, termination, survival)

            timestep = Timestep(obs=start.obs, terminated=start.terminated, truncated=start.truncated)
            carry = (timestep, history, jnp.ones(jnp.shape(start.reward)))
            (last, _, last_survival), (timesteps, rewards, terminations, survivals) = jax.lax.scan(
                jax.checkpoint(act_fn), carry, jax.random.split(key, self.cfg.horizon)
            )
            timesteps = jax.tree.map(lambda steps, final: jnp.concatenate([steps, final[None]]), timesteps, last)
            return Imagination(
                states=timesteps.obs[self.state_key],
                goals=jax.vmap(self.world_model.goal)(timesteps.obs[self.command_key]),
                contexts=timesteps.obs[self.context_key],
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
            *_, width = jnp.shape(start.obs[self.observation_key])
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
            goal = self.world_model.goal(start.obs[self.command_key])
            first, second = jax.tree.map(merge_fn, (window.first, window.second))
            *_, width = jnp.shape(start.obs[self.observation_key])
            history = history_fn(window, noise_key)[..., width:]
            _, actor_statistics = self.actor.apply(
                state.params, jnp.concatenate([history, goal], axis=-1), temperature=1.0, mutable=["statistics"]
            )
            _, critic_statistics = self.critic.apply(
                state.critic_params,
                jnp.concatenate([start.obs[self.state_key], goal, start.obs[self.context_key]], axis=-1),
                mutable=["statistics"],
            )
            _, world_model_statistics = self.world_model.apply(
                state.world_model_params,
                first.obs[self.state_key],
                second.action,
                first.obs[self.context_key],
                first.obs[self.command_key],
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
            window = jax.tree.map(merge_fn, windows)
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
