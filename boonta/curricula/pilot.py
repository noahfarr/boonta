from dataclasses import dataclass

import jax
import jax.numpy as jnp

from boonta.utils import Key, PyTree, Timestep


@dataclass(frozen=True)
class Pilot:
    algorithm: object
    state: PyTree

    def start(self, key: Key, timestep: Timestep) -> PyTree:
        shape = jax.eval_shape(self.algorithm.init, key, timestep).carry
        return jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shape)

    def act(self, key: Key, timestep: Timestep, carry: PyTree, temperature: float):
        state, action, _ = self.algorithm.step(
            self.state.replace(carry=carry), key, timestep, temperature
        )
        return action, state.carry
