from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import Timestep, Transition, canonicalize_dtype
from boonta.utils.typing import Array, Key, PyTree


@struct.dataclass(frozen=True)
class BCConfig:
    batch_size: int
    entropy_coefficient: float = 0.0


@struct.dataclass(frozen=True)
class BCState:
    step: Array
    params: PyTree
    optimizer_state: optax.OptState


@dataclass
class BC:
    cfg: BCConfig
    network: nn.Module
    optimizer: optax.GradientTransformation

    def init(self, key: Key, timestep: Timestep) -> BCState:
        params = self.network.init(key, timestep.obs, temperature=1.0)
        return BCState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            params=params,
            optimizer_state=self.optimizer.init(params["params"]),
        )

    def step(
        self, state: BCState, key: Key, timestep: Timestep, temperature: float = 1.0
    ) -> tuple[BCState, Array, PyTree]:
        dist = self.network.apply(state.params, timestep.obs, temperature=temperature)
        return state, dist.sample(seed=key), {}

    def update(self, state: BCState, key: Key, transitions: Transition) -> BCState:
        del key

        def loss_fn(params: PyTree) -> tuple[Array, tuple[Array, Array]]:
            dist = self.network.apply(params, transitions.first.obs, temperature=1.0)
            likelihood = -jnp.mean(dist.log_prob(transitions.second.action))
            entropy = jnp.mean(dist.entropy())
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
        params = {
            **state.params,
            "params": optax.apply_updates(state.params["params"], updates),
        }
        return state.replace(params=params, optimizer_state=optimizer_state)
