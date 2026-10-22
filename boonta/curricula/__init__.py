from boonta.algorithms import Algorithm
from boonta.utils.typing import Environment

from boonta.podracers.podracer import Lap, Pit

from . import erd, neyman, plr, prd, psro, sgs
from .curriculum import Curriculum


def default(
    algorithm: Algorithm, environment: Environment, **kwargs
) -> tuple[Algorithm, Environment, Pit | None, Lap | None]:
    return algorithm, environment, None, None


__all__ = [
    "Curriculum",
    "default",
    "erd",
    "neyman",
    "plr",
    "prd",
    "psro",
    "sgs",
]
