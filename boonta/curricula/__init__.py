from boonta.algorithms import Algorithm
from boonta.utils.typing import Environment

from boonta.podracers.podracer import Lap, Pit

from .curriculum import Curriculum, within
from .league import league
from .plr import maximum_monte_carlo, plr, positive_value_loss


def default(
    algorithm: Algorithm, environment: Environment, **kwargs
) -> tuple[Algorithm, Environment, Pit, Lap]:
    return algorithm, environment, lambda state: state, lambda state: state


__all__ = [
    "Curriculum",
    "default",
    "league",
    "maximum_monte_carlo",
    "plr",
    "positive_value_loss",
    "within",
]
