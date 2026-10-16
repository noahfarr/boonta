from flax import struct

from .timestep import Timestep
from .typing import PyTree


@struct.dataclass(frozen=True)
class Transition:
    first: Timestep | None = None
    second: Timestep | None = None
    aux: PyTree | None = None


@struct.dataclass(frozen=True)
class Trajectory:
    transitions: Transition
    carry: PyTree | None = None
