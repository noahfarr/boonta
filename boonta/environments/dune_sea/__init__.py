from .generators import Generator, cave_generator
from .hide_and_seek import (HideAndSeek, HideAndSeekState, Layout, random_layout,
                            shelter_layout)
from .jundland import Jundland
from .maze import Level, Maze, MazeState, kruskal_generator
from .state import Agent, Cell, JundlandState

registry = {
    "hide_and_seek": HideAndSeek,
    "jundland": Jundland,
    "maze": Maze,
}


def make(env_id: str, **kwargs):
    return registry[env_id](**kwargs)


__all__ = [
    "Agent",
    "Cell",
    "Generator",
    "HideAndSeek",
    "HideAndSeekState",
    "Jundland",
    "JundlandState",
    "Layout",
    "Level",
    "Maze",
    "MazeState",
    "cave_generator",
    "kruskal_generator",
    "make",
    "random_layout",
    "shelter_layout",
]
