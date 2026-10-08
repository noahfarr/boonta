from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class Distribution(Enum):
    uniform = 1
    int_uniform = 2
    uniform_pow2 = 3
    log_normal = 4
    logit_normal = 5


class Direction(Enum):
    minimize = 1
    maximize = 2


@dataclass
class ParamConfig:
    distribution: Distribution = MISSING
    min: float = MISSING
    max: float = MISSING
    center: Optional[float] = None
    scale: Optional[float] = None
    rounding_factor: int = 1


@dataclass
class CarbsSweeperConf:
    _target_: str = "hydra_plugins.hydra_carbs_sweeper.sweeper.CarbsSweeper"

    metric: str = "score"
    cost: str = "cost"

    direction: Direction = Direction.maximize

    n_trials: int = 100
    n_jobs: int = 1
    max_failure_rate: float = 1.0

    seed: int = 0
    num_random_samples: int = 4
    max_suggestion_cost: Optional[float] = None
    resample_frequency: int = 5
    num_candidates_for_suggestion_per_dim: int = 100
    initial_search_radius: float = 0.3
    exploration_bias: float = 1.0

    checkpoint_dir: Optional[str] = None
    warm_start_from: Optional[str] = None
    is_saved_on_every_observation: bool = True
    is_wandb_logging_enabled: bool = False

    params: Optional[Dict[str, ParamConfig]] = "${oc.select:search_space,null}"


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="carbs",
    node=CarbsSweeperConf,
    provider="carbs_sweeper",
)
