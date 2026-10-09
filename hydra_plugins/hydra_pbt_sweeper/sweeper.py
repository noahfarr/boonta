from typing import Any, List

from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig


class PbtSweeper(Sweeper):
    def __init__(self, **settings: Any) -> None:
        from ._population import Population

        self.population = Population(**settings)

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.population.setup(
            hydra_context=hydra_context, task_function=task_function, config=config
        )

    def sweep(self, arguments: List[str]) -> Any:
        return self.population.sweep(arguments)
