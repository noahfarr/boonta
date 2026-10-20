from flax import struct

from boonta.utils import Array

MAX_ENERGY = 1.0


@struct.dataclass
class Cell:
    wall: Array
    goal: Array
    energy: Array


@struct.dataclass
class Agent:
    position: Array
    direction: Array
    energy: Array
    broadcast: Array
    inventory: Array


@struct.dataclass
class DuneSeaState:
    cells: Cell
    agent: Agent
    time: Array
