from functools import partial

from .commons import Commons
from .dune_sea import DuneSea
from .hide_and_seek import HideAndSeek
from .generators import Generator, cave_generator, uniform_generator
from .state import Agent, Cell, DuneSeaState

SIGHT_DRAIN = 0.01

registry = {
    "dune_sea": DuneSea,
    "commons": Commons,
    "hide_and_seek": HideAndSeek,
    "hide_and_seek_sight": partial(HideAndSeek, sight_drain=SIGHT_DRAIN, catch_amount=0.0),
}


def make(env_id: str, **kwargs):
    return registry[env_id](**kwargs)


__all__ = [
    "Agent",
    "Cell",
    "Commons",
    "DuneSea",
    "DuneSeaState",
    "HideAndSeek",
    "Generator",
    "cave_generator",
    "make",
    "uniform_generator",
]
