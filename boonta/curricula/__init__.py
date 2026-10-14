from boonta.algorithms import Algorithm
from boonta.utils.typing import Environment

from boonta.podracers.podracer import Lap, Pit

from .curriculum import Curriculum
from .plr import maximum_monte_carlo, plr, positive_value_loss
from .psro import (Learner, always, either, even, forgotten, fsp, hard, ladder,
                   latest, mixture, nash, periodic, pfsp, pfsp_vs_lineage,
                   psro, self_play, threshold)


def default(
    algorithm: Algorithm, environment: Environment, **kwargs
) -> tuple[Algorithm, Environment, Pit | None, Lap | None]:
    return algorithm, environment, None, None


__all__ = [
    "Curriculum",
    "Learner",
    "always",
    "default",
    "either",
    "even",
    "forgotten",
    "fsp",
    "hard",
    "ladder",
    "latest",
    "maximum_monte_carlo",
    "mixture",
    "nash",
    "periodic",
    "pfsp",
    "pfsp_vs_lineage",
    "plr",
    "positive_value_loss",
    "psro",
    "self_play",
    "threshold",
]
