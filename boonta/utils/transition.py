from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from .timestep import Timestep
from .typing import PyTree


@struct.dataclass(frozen=True)
class Transition:
    first: Timestep | None = None
    second: Timestep | None = None
    aux: PyTree | None = None
    carry: PyTree | None = None

    @property
    def start(self) -> PyTree:
        return jax.tree.map(lambda leaf: leaf[0], self.carry)

    def tail(self) -> tuple[Timestep, PyTree]:
        last = jax.tree.map(
            lambda first, second: jnp.stack([first[-1], second[-1]], axis=1),
            self.first,
            self.second,
        )
        return last, jax.tree.map(lambda leaf: leaf[-1], self.carry)


def unroll(rollout: Callable, state: PyTree, keys: PyTree) -> tuple[PyTree, Transition]:
    def body(scanned, key):
        state, _ = scanned
        entering = state.carry
        state, transition = rollout(state, key)
        return (state, entering), transition

    start = state.carry
    (state, last), transitions = jax.lax.scan(body, (state, start), keys)
    carry = jax.tree.map(lambda first, final: jnp.stack([first, final]), start, last)
    return state, transitions.replace(carry=carry)
