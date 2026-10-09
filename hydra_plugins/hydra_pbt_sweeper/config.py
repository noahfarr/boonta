from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from hydra_plugins.hydra_carbs_sweeper.config import ParamConfig


@dataclass
class PbtSweeperConf:
    _target_: str = "hydra_plugins.hydra_pbt_sweeper.sweeper.PbtSweeper"

    metric: str = "score"

    generation: str = MISSING
    checkpoint: str = MISSING
    num_epochs: int = MISSING
    overrides: List[str] = field(default_factory=list)

    members: int = 4
    seeds: int = 4
    generations: int = 5

    fraction: float = 0.25
    threshold: float = 2.0
    factors: List[float] = field(default_factory=lambda: [0.8, 1.25])
    resample_probability: float = 0.25

    seed: int = 0

    params: Optional[Dict[str, ParamConfig]] = "${oc.select:search_space,null}"


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="pbt_schema",
    node=PbtSweeperConf,
    provider="pbt_sweeper",
)
