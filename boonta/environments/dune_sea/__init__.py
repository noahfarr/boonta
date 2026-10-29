from .dune_sea import DuneSea
from .forage import Forage
from .hide_and_seek import HideAndSeek
from .generators import Generator, cave_generator, maze_generator
from .maze import Maze
from .shelter import Shelter
from .tag import Tag
from .state import Agent, Cell, DuneSeaState

registry = {
    "forage": Forage,
    "hide_and_seek": HideAndSeek,
    "maze": Maze,
    "shelter": Shelter,
    "tag": Tag,
}


def make(env_id: str, **kwargs):
    return registry[env_id](**kwargs)


__all__ = [
    "Agent",
    "Cell",
    "DuneSea",
    "Forage",
    "DuneSeaState",
    "HideAndSeek",
    "Generator",
    "cave_generator",
    "Maze",
    "maze_generator",
    "Shelter",
    "Tag",
    "make",
]
