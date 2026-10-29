from flax import struct

from boonta.utils import Array

MAX_ENERGY = 250.0
STARTING_ENERGY = 125.0
MAX_PLANT_ENERGY = 125.0
MATURE_ENERGY = 75.0
SEED_ENERGY = 12.5
GROWTH_STEPS = 100
THRESHOLD = 200.0
GIFT = 100.0
MIN_GIFT = 10.0


@struct.dataclass
class Cell:
    wall: Array
    energy: Array
    bush: Array
    scent: Array


@struct.dataclass
class Agent:
    position: Array
    direction: Array
    energy: Array
    broadcast: Array
    inventory: Array
    threshold: Array
    gift: Array
    age: Array


@struct.dataclass
class JundlandState:
    cells: Cell
    agent: Agent
    time: Array
