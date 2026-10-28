from flax import struct

from boonta.utils import Array

MAX_ENERGY = 1.0
ENERGY_GAIN = 0.3
MAX_PLANT_ENERGY = 0.5
SEED_ENERGY = 0.05
MATURE_ENERGY = 0.3
GROWTH_STEPS = 100


@struct.dataclass
class Cell:
    wall: Array
    energy: Array
    bush: Array


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
