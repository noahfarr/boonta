from .dune_sea import DuneSea
from .hide_and_seek import HideAndSeek
from .generators import Generator, cave_generator
from .tag import Tag
from .state import Agent, Cell, DuneSeaState

registry = {
    "hide_and_seek": HideAndSeek,
    "tag": Tag,
}


def make(env_id: str, **kwargs):
    return registry[env_id](**kwargs)


__all__ = [
    "Agent",
    "Cell",
    "DuneSea",
    "DuneSeaState",
    "HideAndSeek",
    "Generator",
    "cave_generator",
    "Tag",
    "make",
]
