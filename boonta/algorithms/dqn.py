from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import (
    Timestep,
    Transition,
    add_feature_axis,
    canonicalize_dtype,
    conditional_update,
    remove_batch_axis,
    remove_feature_axis,
)
from boonta.utils.typing import Array, Buffer, BufferState, Key, PyTree


@struct.dataclass(frozen=True)
class DQNConfig:
    updates_per_step: int
    gamma: float
    tau: float


@struct.dataclass(frozen=True)
class DQNState:
    step: Array
    params: PyTree
    target_params: PyTree
    optimizer_state: optax.OptState
    buffer_state: BufferState = struct.field(metadata={"axis": "data"})


@dataclass
class DQN:
    cfg: DQNConfig
    network: nn.Module
    exploration_schedule: optax.Schedule
    buffer: Buffer
    optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> DQNState:
        params = self.network.init(key, timestep.obs, temperature=1.0)

        return DQNState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            params=params,
            target_params=params,
            optimizer_state=self.optimizer.init(params["params"]),
            buffer_state=self.buffer.init(
                jax.tree.map(
                    remove_batch_axis,
                    Transition(first=timestep, second=timestep, aux={}),
                )
            ),
        )

    def step(
        self, state: DQNState, key: Key, timestep: Timestep, temperature: float = 1.0
    ) -> tuple[DQNState, Array, PyTree]:
        dist = self.network.apply(
            state.params,
            timestep.obs,
            temperature=self.exploration_schedule(state.step) * temperature,
        )
        action = dist.sample(seed=key)
        return state, action, {}

    def update(self, state: DQNState, key: Key, transitions: Transition) -> DQNState:
        def loss_fn(
            params: PyTree, state: DQNState, transitions: Transition
        ) -> tuple[Array, tuple[PyTree, Array]]:
            next_q_values = self.network.apply(
                state.target_params, transitions.second.obs, temperature=1.0
            ).preferences
            target_q_value = transitions.second.reward + self.cfg.gamma * (
                1.0 - transitions.second.terminated
            ) * jnp.max(next_q_values, axis=-1)

            dist, variables = self.network.apply(
                params,
                transitions.first.obs,
                temperature=1.0,
                mutable=True,
            )
            intermediates = {"intermediates": variables.pop("intermediates", {})}
            q_values = dist.preferences
            q_value = remove_feature_axis(
                jnp.take_along_axis(
                    q_values, add_feature_axis(transitions.second.action), axis=-1
                )
            )
            td_error = (q_value - target_q_value) * (1.0 - transitions.second.truncated)
            loss = 0.5 * (td_error**2).mean()

            def apply(params: PyTree) -> PyTree:
                return self.network.apply(
                    params, transitions.first.obs, temperature=1.0
                )

            for auxiliary_loss in self.auxiliary_losses:
                loss = loss + auxiliary_loss(
                    params=params,
                    apply=apply,
                    transitions=transitions,
                    dist=dist,
                    q_values=q_values,
                    intermediates=intermediates,
                )
            return loss, (variables, q_value)

        transitions = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), transitions)
        state = state.replace(
            buffer_state=self.buffer.add(state.buffer_state, transitions)
        )

        def minibatch_fn(state: DQNState, batch, key: Key) -> DQNState:
            del key
            transitions = batch

            (loss, (variables, q_value)), grads = jax.value_and_grad(
                loss_fn, has_aux=True, allow_int=True
            )(state.params, state, transitions)
            updates, optimizer_state = self.optimizer.update(
                grads["params"], state.optimizer_state, state.params["params"]
            )
            params = {
                **state.params,
                **variables,
                "params": optax.apply_updates(state.params["params"], updates),
            }
            state = state.replace(
                params=params,
                optimizer_state=optimizer_state,
                target_params={
                    **params,
                    "params": optax.incremental_update(
                        params["params"], state.target_params["params"], self.cfg.tau
                    ),
                },
            )

            lox.log(
                {
                    "critic/loss": loss,
                    "critic/q_value": q_value.mean(),
                }
            )

            return state

        sample_key, update_key = jax.random.split(key)
        sample_keys = jax.random.split(sample_key, self.cfg.updates_per_step)
        update_keys = jax.random.split(update_key, self.cfg.updates_per_step)

        def draw(state: DQNState, keys: tuple[Key, Key]) -> tuple[DQNState, None]:
            sample_key, update_key = keys
            batch = jax.tree.map(
                lambda leaf: jnp.squeeze(leaf, axis=1),
                self.buffer.sample(state.buffer_state, sample_key).experience,
            )
            return minibatch_fn(state, batch, update_key), None

        updated, _ = jax.lax.scan(draw, state, (sample_keys, update_keys))
        return conditional_update(
            updated, state, self.buffer.can_sample(state.buffer_state)
        )
