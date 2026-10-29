from .generators import Generator, cave_generator
from .jundland import Jundland
from .state import Agent, Cell, JundlandState

registry = {
    "jundland": Jundland,
}


def make(env_id: str, **kwargs):
    return registry[env_id](**kwargs)


__all__ = [
    "Agent",
    "Cell",
    "Generator",
    "Jundland",
    "JundlandState",
    "cave_generator",
    "make",
]
