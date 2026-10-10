from boonta.algorithms import Algorithm
from boonta.utils.typing import Environment

from boonta.podracers.podracer import Lap, Pit

from .curriculum import Curriculum
from .erd import erd
from .league import league
from .plr import maximum_monte_carlo, plr, positive_value_loss
from .prd import prd


def default(
    algorithm: Algorithm, environment: Environment, **kwargs
) -> tuple[Algorithm, Environment, Pit, Lap]:
    return algorithm, environment, lambda state, transitions: state, lambda state: state


__all__ = [
    "Curriculum",
    "default",
    "erd",
    "league",
    "maximum_monte_carlo",
    "plr",
    "positive_value_loss",
    "prd",
]
