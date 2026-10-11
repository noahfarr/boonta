from . import wordle
from .chain_of_forks import ChainOfForks, ChainOfForksState
from .connectx import ConnectX, ConnectXState
from .t_maze import TMaze, TMazeState
from .wordle import Wordle

registry = {
    "chain_of_forks": ChainOfForks,
    "connectx": ConnectX,
    "t_maze": TMaze,
    "wordle": wordle.make,
}


def make(env_id: str, **kwargs):
    return registry[env_id](**kwargs)


__all__ = [
    "ChainOfForks",
    "ChainOfForksState",
    "ConnectX",
    "ConnectXState",
    "TMaze",
    "TMazeState",
    "Wordle",
    "make",
]
