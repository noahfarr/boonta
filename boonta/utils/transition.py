import jax
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
        final_timestep = jax.tree.map(lambda leaf: leaf[-1][:, None], self.second)
        return final_timestep, jax.tree.map(lambda leaf: leaf[-1], self.carry)
