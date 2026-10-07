from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import mctx
import optax
from flax import struct

from boonta.utils import (Timestep, Transition, canonicalize_dtype,
                          conditional_update, remove_batch_axis)
from boonta.utils.typing import Array, Buffer, BufferState, Key, PyTree


def transform(x: Array, epsilon: float = 1e-3) -> Array:
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1.0) - 1.0) + epsilon * x


def inverse(x: Array, epsilon: float = 1e-3) -> Array:
    magnitude = (jnp.sqrt(1.0 + 4.0 * epsilon * (jnp.abs(x) + 1.0 + epsilon)) - 1.0) / (
        2.0 * epsilon
    )
    return jnp.sign(x) * (magnitude**2 - 1.0)


def two_hot(x: Array, support: Array) -> Array:
    bins = support.shape[0]
    delta = support[1] - support[0]
    x = jnp.clip(x, support[0], support[-1])
    index = jnp.clip(jnp.floor((x - support[0]) / delta), 0, bins - 2).astype(jnp.int32)
    weight = (x - support[index]) / delta
    return (
        jax.nn.one_hot(index, bins) * (1.0 - weight)[..., None]
        + jax.nn.one_hot(index + 1, bins) * weight[..., None]
    )


def scale_gradient(x: PyTree, scale: float) -> PyTree:
    return jax.tree.map(
        lambda leaf: leaf * scale + jax.lax.stop_gradient(leaf) * (1.0 - scale), x
    )


def cross_entropy(logits: Array, target: Array) -> Array:
    return -jnp.sum(target * jax.nn.log_softmax(logits, axis=-1), axis=-1)


@struct.dataclass(frozen=True)
class MuZeroConfig:
    num_actions: int
    unroll_steps: int
    bootstrap_steps: int
    num_simulations: int
    max_considered_actions: int
    updates_per_step: int
    gamma: float
    vmin: float
    vmax: float
    num_bins: int
    value_coefficient: float
    reward_coefficient: float
    policy_coefficient: float

    @property
    def support(self) -> Array:
        return jnp.linspace(self.vmin, self.vmax, self.num_bins)

    @property
    def sequence_length(self) -> int:
        return self.unroll_steps + self.bootstrap_steps + 1


@struct.dataclass(frozen=True)
class MuZeroState:
    step: Array
    params: PyTree
    optimizer_state: optax.OptState
    buffer_state: BufferState = struct.field(metadata={"axis": "data"})


@dataclass
class MuZero:
    cfg: MuZeroConfig
    representation: nn.Module
    dynamics: nn.Module
    prediction: nn.Module
    buffer: Buffer
    optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()

    def scalar(self, logits: Array) -> Array:
        return inverse(jnp.sum(jax.nn.softmax(logits, axis=-1) * self.cfg.support, -1))

    def target(self, value: Array) -> Array:
        return two_hot(transform(value), self.cfg.support)

    def root(self, params: PyTree, obs: PyTree) -> mctx.RootFnOutput:
        embedding = self.representation.apply(params["representation"], obs)
        logits, value = self.prediction.apply(params["prediction"], embedding)
        return mctx.RootFnOutput(
            prior_logits=logits, value=self.scalar(value), embedding=embedding
        )

    def recurrent_fn(self, params: PyTree, key: Key, action: Array, embedding: PyTree):
        del key
        embedding, reward = self.dynamics.apply(params["dynamics"], embedding, action)
        logits, value = self.prediction.apply(params["prediction"], embedding)
        reward = self.scalar(reward)
        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=jnp.full_like(reward, self.cfg.gamma),
                prior_logits=logits,
                value=self.scalar(value),
            ),
            embedding,
        )

    def search(self, params: PyTree, key: Key, obs: PyTree):
        return mctx.gumbel_muzero_policy(
            params=params,
            rng_key=key,
            root=self.root(params, obs),
            recurrent_fn=self.recurrent_fn,
            num_simulations=self.cfg.num_simulations,
            max_num_considered_actions=self.cfg.max_considered_actions,
        )

    def init(self, key: Key, timestep: Timestep) -> MuZeroState:
        representation_key, dynamics_key, prediction_key = jax.random.split(key, 3)
        representation = self.representation.init(representation_key, timestep.obs)
        embedding = self.representation.apply(representation, timestep.obs)
        envs = embedding.shape[0]
        params = {
            "representation": representation,
            "dynamics": self.dynamics.init(
                dynamics_key, embedding, jnp.zeros(envs, jnp.int32)
            ),
            "prediction": self.prediction.init(prediction_key, embedding),
        }
        experience = Transition(
            first=timestep,
            second=timestep,
            aux={
                "policy": jnp.zeros((envs, self.cfg.num_actions)),
                "value": jnp.zeros((envs,)),
            },
        )
        return MuZeroState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            params=params,
            optimizer_state=self.optimizer.init(params),
            buffer_state=self.buffer.init(
                jax.tree.map(remove_batch_axis, experience)
            ),
        )

    def step(
        self, state: MuZeroState, key: Key, timestep: Timestep, temperature: float = 1.0
    ) -> tuple[MuZeroState, Array, PyTree]:
        output = self.search(state.params, key, timestep.obs)
        weights = output.action_weights
        action = jnp.where(
            temperature > 0.0, output.action, jnp.argmax(weights, axis=-1)
        )
        return state, action, {
            "policy": weights,
            "value": output.search_tree.summary().value,
        }

    def bootstrapped(self, batch: Transition, start: int) -> Array:
        rewards = batch.second.reward
        values = batch.aux["value"]
        terminated = batch.second.terminated.astype(jnp.float32)
        truncated = batch.second.truncated.astype(jnp.float32)

        total = jnp.zeros_like(values[:, 0])
        discount = jnp.ones_like(total)
        live = jnp.ones_like(total)
        for offset in range(self.cfg.bootstrap_steps):
            index = start + offset
            total = total + discount * live * rewards[:, index]
            total = total + discount * self.cfg.gamma * live * truncated[
                :, index
            ] * values[:, index + 1]
            live = live * (1.0 - jnp.maximum(terminated[:, index], truncated[:, index]))
            discount = discount * self.cfg.gamma
        return total + discount * live * values[:, start + self.cfg.bootstrap_steps]

    def loss(self, params: PyTree, batch: Transition) -> tuple[Array, dict]:
        done = jnp.maximum(
            batch.second.terminated.astype(jnp.float32),
            batch.second.truncated.astype(jnp.float32),
        )
        embedding = self.representation.apply(
            params["representation"],
            jax.tree.map(lambda leaf: leaf[:, 0], batch.first.obs),
        )
        policy_loss = value_loss = reward_loss = jnp.zeros(())
        alive = jnp.ones_like(done[:, 0])

        for offset in range(self.cfg.unroll_steps + 1):
            logits, value = self.prediction.apply(params["prediction"], embedding)
            weight = alive / jnp.maximum(alive.sum(), 1.0)
            scale = 1.0 if offset == 0 else 1.0 / self.cfg.unroll_steps

            policy_loss += scale * jnp.sum(
                weight * cross_entropy(logits, batch.aux["policy"][:, offset])
            )
            value_loss += scale * jnp.sum(
                weight
                * cross_entropy(value, self.target(self.bootstrapped(batch, offset)))
            )

            if offset < self.cfg.unroll_steps:
                embedding, reward = self.dynamics.apply(
                    params["dynamics"],
                    scale_gradient(embedding, 0.5),
                    batch.second.action[:, offset],
                )
                reward_loss += (1.0 / self.cfg.unroll_steps) * jnp.sum(
                    weight
                    * cross_entropy(
                        reward, self.target(batch.second.reward[:, offset])
                    )
                )
                alive = alive * (1.0 - done[:, offset])

        loss = (
            self.cfg.policy_coefficient * policy_loss
            + self.cfg.value_coefficient * value_loss
            + self.cfg.reward_coefficient * reward_loss
        )
        for auxiliary_loss in self.auxiliary_losses:
            loss = loss + auxiliary_loss(params=params, transitions=batch)
        return loss, {
            "muzero/loss": loss,
            "muzero/policy_loss": policy_loss,
            "muzero/value_loss": value_loss,
            "muzero/reward_loss": reward_loss,
        }

    def update(
        self, state: MuZeroState, key: Key, transitions: Transition
    ) -> MuZeroState:
        sequence = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), transitions)
        state = state.replace(
            buffer_state=self.buffer.add(state.buffer_state, sequence)
        )

        def minibatch_fn(carry: MuZeroState, key: Key) -> tuple[MuZeroState, None]:
            batch = self.buffer.sample(carry.buffer_state, key).experience
            (_, logs), grads = jax.value_and_grad(self.loss, has_aux=True)(
                carry.params, batch
            )
            updates, optimizer_state = self.optimizer.update(
                grads, carry.optimizer_state, carry.params
            )
            lox.log(logs)
            return carry.replace(
                params=optax.apply_updates(carry.params, updates),
                optimizer_state=optimizer_state,
            ), None

        updated, _ = jax.lax.scan(
            minibatch_fn, state, jax.random.split(key, self.cfg.updates_per_step)
        )
        return conditional_update(
            updated, state, self.buffer.can_sample(state.buffer_state)
        )
