from .generators import Generator, cave_generator
from .hide_and_seek import HideAndSeek, HideAndSeekState, Layout, random_layout
from .jundland import Jundland
from .state import Agent, Cell, JundlandState

registry = {
    "hide_and_seek": HideAndSeek,
    "jundland": Jundland,
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
    "cave_generator",
    "make",
    "random_layout",
]
