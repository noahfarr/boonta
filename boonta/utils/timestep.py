import jax
from flax import struct

from .axis import add_time_axis, remove_time_axis
from .typing import Array, PyTree


@struct.dataclass(frozen=True)
class Timestep:
    obs: PyTree
    action: Array | None = None
    reward: Array | None = None
    terminated: Array | None = None
    truncated: Array | None = None
    info: PyTree | None = None

    @property
    def done(self) -> Array:
        return self.terminated | self.truncated

    def to_sequence(self) -> "Timestep":
        return jax.tree.map(add_time_axis, self)

    def from_sequence(self) -> "Timestep":
        return jax.tree.map(remove_time_axis, self)

    def __iter__(self):
        return iter(
            (
                self.obs,
                self.action,
                self.reward,
                self.terminated,
                self.truncated,
                self.info,
            )
        )
