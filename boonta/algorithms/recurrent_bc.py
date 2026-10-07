from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import Timestep, Transition, canonicalize_dtype, remove_time_axis
from boonta.utils.typing import Array, Key, PyTree


@struct.dataclass(frozen=True)
class RecurrentBCConfig:
    batch_size: int
    entropy_coefficient: float = 0.0


@struct.dataclass(frozen=True)
class RecurrentBCState:
    step: Array
    carry: PyTree = struct.field(metadata={"axis": "data"})
    params: PyTree
    optimizer_state: optax.OptState


@dataclass
class RecurrentBC:
    cfg: RecurrentBCConfig
    network: nn.Module
    optimizer: optax.GradientTransformation

    def init(self, key: Key, timestep: Timestep) -> RecurrentBCState:
        network_key, carry_key = jax.random.split(key)
        carry = self.network.initialize_carry(carry_key, (*timestep.reward.shape, 1))
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
        return RecurrentBCState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            carry=carry,
            params=params,
            optimizer_state=self.optimizer.init(params["params"]),
        )

    def step(
        self,
        state: RecurrentBCState,
        key: Key,
        timestep: Timestep,
        temperature: float = 1.0,
    ) -> tuple[RecurrentBCState, Array, PyTree]:
        sequence = timestep.to_sequence()
        carry, dist = self.network.apply(
            state.params,
            sequence.obs,
            sequence.action,
            sequence.reward,
            sequence.done,
            carry=state.carry,
            temperature=temperature,
        )
        action = remove_time_axis(dist.sample(seed=key))
        return state.replace(carry=carry), action, {}

    def update(
        self, state: RecurrentBCState, key: Key, transitions: Transition
    ) -> RecurrentBCState:
        timesteps = transitions.first
        batch_size, *_ = jax.tree.leaves(timesteps.obs)[0].shape
        carry = self.network.initialize_carry(key, (batch_size, 1))

        def loss_fn(params: PyTree) -> tuple[Array, tuple[Array, Array]]:
            _, dist = self.network.apply(
                params,
                timesteps.obs,
                timesteps.action,
                timesteps.reward,
                timesteps.done,
                carry=carry,
                temperature=1.0,
            )
            log_prob = dist.log_prob(transitions.second.action)
            weight = (transitions.aux or {}).get("weight", jnp.ones_like(log_prob))
            weight = weight.astype(log_prob.dtype)
            total = jnp.maximum(jnp.sum(weight), 1.0)
            likelihood = -jnp.sum(log_prob * weight) / total
            entropy = jnp.sum(dist.entropy() * weight) / total
            return likelihood - self.cfg.entropy_coefficient * entropy, (
                likelihood,
                entropy,
            )

        (loss, (likelihood, entropy)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        lox.log(
            {
                "actor/loss": loss,
                "actor/likelihood": likelihood,
                "actor/entropy": entropy,
            }
        )

        updates, optimizer_state = self.optimizer.update(
            grads["params"], state.optimizer_state, state.params["params"]
        )
        return state.replace(
            step=state.step + 1,
            params={
                **state.params,
                "params": optax.apply_updates(state.params["params"], updates),
            },
            optimizer_state=optimizer_state,
        )
