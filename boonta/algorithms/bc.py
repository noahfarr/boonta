from collections.abc import Callable
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
    auxiliary_losses: tuple[Callable, ...] = ()

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

        def loss_fn(params: PyTree) -> tuple[Array, tuple[PyTree, Array, Array]]:
            dist, variables = self.network.apply(
                params,
                transitions.first.obs,
                temperature=1.0,
                mutable=True,
            )
            likelihood = -jnp.mean(dist.log_prob(transitions.second.action))
            entropy = jnp.mean(dist.entropy())
            loss = likelihood - self.cfg.entropy_coefficient * entropy

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
                    variables=variables,
                )
            return loss, (variables, likelihood, entropy)

        (loss, (returned, likelihood, entropy)), grads = jax.value_and_grad(
            loss_fn, has_aux=True, allow_int=True
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
        variables = {
            name: returned.get(name, value) for name, value in state.params.items()
        }
        params = {
            **variables,
            "params": optax.apply_updates(state.params["params"], updates),
        }
        return state.replace(params=params, optimizer_state=optimizer_state)
