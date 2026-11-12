from . import wordle
from .boxoban_tower import BoxobanTower, BoxobanTowerState
from .chain import Chain, ChainState
from .chain_of_forks import ChainOfForks, ChainOfForksState
from .combination_lock import CombinationLock, CombinationLockState
from .connectx import ConnectX, ConnectXState
from .hotspots import Hotspots, HotspotsState
from .maze import Maze, MazeState
from .t_maze import TMaze, TMazeState
from .wordle import Wordle

registry = {
    "boxoban_tower": BoxobanTower,
    "chain": Chain,
    "chain_of_forks": ChainOfForks,
    "combination_lock": CombinationLock,
    "connectx": ConnectX,
    "hotspots": Hotspots,
    "maze": Maze,
    "t_maze": TMaze,
    "wordle": wordle.make,
}


def make(env_id: str, **kwargs):
    return registry[env_id](**kwargs)


__all__ = [
    "BoxobanTower",
    "BoxobanTowerState",
    "Chain",
    "ChainState",
    "ChainOfForks",
    "ChainOfForksState",
    "CombinationLock",
    "CombinationLockState",
    "ConnectX",
    "ConnectXState",
    "Hotspots",
    "HotspotsState",
    "Maze",
    "MazeState",
    "TMaze",
    "TMazeState",
    "Wordle",
    "make",
]
