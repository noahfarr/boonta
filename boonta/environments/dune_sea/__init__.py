from .commons import Commons
from .dune_sea import DuneSea
from .generators import Generator, cave_generator, uniform_generator
from .state import Agent, Cell, DuneSeaState

registry = {
    "dune_sea": DuneSea,
    "commons": Commons,
}


def make(env_id: str, **kwargs):
    return registry[env_id](**kwargs)


__all__ = [
    "Agent",
    "Cell",
    "Commons",
    "DuneSea",
    "DuneSeaState",
    "Generator",
    "cave_generator",
    "make",
    "uniform_generator",
]
