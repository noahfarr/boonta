from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import (Timestep, Transition, add_feature_axis,
                          add_time_axis, canonicalize_dtype,
                          conditional_update, remove_batch_axis,
                          remove_feature_axis, remove_time_axis)
from boonta.utils.typing import Array, Buffer, BufferState, Key, PyTree


@struct.dataclass(frozen=True)
class RecurrentDQNConfig:
    updates_per_step: int
    gamma: float
    tau: float


@struct.dataclass(frozen=True)
class RecurrentDQNState:
    step: Array
    carry: PyTree = struct.field(metadata={"axis": "data"})
    params: PyTree
    target_params: PyTree
    optimizer_state: optax.OptState
    buffer_state: BufferState = struct.field(metadata={"axis": "data"})


@dataclass
class RecurrentDQN:
    cfg: RecurrentDQNConfig
    network: nn.Module
    exploration_schedule: optax.Schedule
    buffer: Buffer
    optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()

    def init(self, key: Key, timestep: Timestep) -> RecurrentDQNState:
        network_key, carry_key = jax.random.split(key)

        carry = self.network.initialize_carry(
            carry_key, (*timestep.reward.shape, 1)
        )
        sequence = timestep.to_sequence()
        params = self.network.init(
            network_key,
            sequence.obs,
            sequence.action,
            sequence.reward,
            sequence.done,
            carry=carry,
            temperature=1.0,
        )

        return RecurrentDQNState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            carry=carry,
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
        self,
        state: RecurrentDQNState,
        key: Key,
        timestep: Timestep,
        temperature: float = 1.0,
    ) -> tuple[RecurrentDQNState, Array, PyTree]:
        sequence = timestep.to_sequence()
        carry, dist = self.network.apply(
            state.params,
            sequence.obs,
            sequence.action,
            sequence.reward,
            sequence.done,
            carry=state.carry,
            temperature=self.exploration_schedule(state.step) * temperature,
        )
        action = dist.sample(seed=key)
        action = remove_time_axis(action)
        return state.replace(carry=carry), action, {}

    def update(
        self, state: RecurrentDQNState, key: Key, transitions: Transition
    ) -> RecurrentDQNState:
        def loss_fn(
            params: PyTree,
            state: RecurrentDQNState,
            trajectory: Transition,
            key: Key,
        ) -> tuple[Array, tuple[PyTree, Array]]:
            transitions = jax.tree.map(lambda x: x[:, :-1], trajectory)
            next_transitions = jax.tree.map(lambda x: x[:, 1:], trajectory)

            batch, _, *num_agents = transitions.second.reward.shape
            carry = self.network.initialize_carry(key, (batch, *num_agents, 1))

            timesteps = next_transitions.first
            _, next_dist = self.network.apply(
                state.target_params,
                timesteps.obs,
                timesteps.action,
                timesteps.reward,
                timesteps.done,
                carry=carry,
                temperature=1.0,
            )
            next_q_values = next_dist.preferences
            target_q_value = transitions.second.reward + self.cfg.gamma * (
                1.0 - transitions.second.terminated
            ) * jnp.max(next_q_values, axis=-1)

            timesteps = transitions.first
            (_, dist), variables = self.network.apply(
                params,
                timesteps.obs,
                timesteps.action,
                timesteps.reward,
                timesteps.done,
                carry=carry,
                temperature=1.0,
                mutable=True,
            )
            q_values = dist.preferences
            q_value = remove_feature_axis(
                jnp.take_along_axis(
                    q_values, add_feature_axis(transitions.second.action), axis=-1
                )
            )
            td_error = (q_value - target_q_value) * (1.0 - transitions.second.truncated)
            loss = 0.5 * (td_error**2).mean()

            def apply(params: PyTree) -> PyTree:
                _, dist = self.network.apply(
                    params,
                    timesteps.obs,
                    timesteps.action,
                    timesteps.reward,
                    timesteps.done,
                    carry=carry,
                    temperature=1.0,
                )
                return dist

            for auxiliary_loss in self.auxiliary_losses:
                loss = loss + auxiliary_loss(
                    params=params,
                    apply=apply,
                    transitions=transitions,
                    dist=dist,
                    q_values=q_values,
                    carry=carry,
                    variables=variables,
                )
            return loss, (variables, q_value)

        buffer_state, _ = jax.lax.scan(
            lambda buffer_state, transition: (
                self.buffer.add(buffer_state, jax.tree.map(add_time_axis, transition)),
                None,
            ),
            state.buffer_state,
            transitions,
        )
        state = state.replace(buffer_state=buffer_state)

        def minibatch_fn(
            state: RecurrentDQNState, batch, key: Key
        ) -> RecurrentDQNState:
            loss_key = key
            trajectory = batch

            (loss, (variables, q_value)), grads = jax.value_and_grad(
                loss_fn, has_aux=True, allow_int=True
            )(state.params, state, trajectory, loss_key)
            updates, optimizer_state = self.optimizer.update(
                grads["params"], state.optimizer_state, state.params["params"]
            )
            variables = {
                name: variables.get(name, value) for name, value in state.params.items()
            }
            params = {
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

        def draw(
            state: RecurrentDQNState, keys: tuple[Key, Key]
        ) -> tuple[RecurrentDQNState, None]:
            sample_key, update_key = keys
            batch = self.buffer.sample(state.buffer_state, sample_key).experience
            return minibatch_fn(state, batch, update_key), None

        updated, _ = jax.lax.scan(draw, state, (sample_keys, update_keys))
        return conditional_update(
            updated, state, self.buffer.can_sample(state.buffer_state)
        )
