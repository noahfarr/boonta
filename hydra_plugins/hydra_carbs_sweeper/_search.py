import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hydra.core.plugins import Plugins
from hydra.core.utils import JobStatus
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf

from ._spaces import center, space
from .config import Direction, ParamConfig

log = logging.getLogger(__name__)


@contextmanager
def trusted():
    import torch

    original = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original


def typed(entry: Any) -> DictConfig:
    plain = OmegaConf.to_container(OmegaConf.create(entry), resolve=True, enum_to_str=True)
    return OmegaConf.merge(OmegaConf.structured(ParamConfig), plain)


def newest(path: Path) -> Path:
    from carbs import CARBS_CHECKPOINT_PREFIX, CARBS_CHECKPOINT_SUFFIX, get_checkpoint_obs_count

    if path.is_file():
        return path
    found = sorted(
        path.rglob(f"{CARBS_CHECKPOINT_PREFIX}*{CARBS_CHECKPOINT_SUFFIX}"),
        key=lambda item: get_checkpoint_obs_count(item.name),
    )
    if not found:
        raise FileNotFoundError(
            f"no {CARBS_CHECKPOINT_PREFIX}*{CARBS_CHECKPOINT_SUFFIX} checkpoint "
            f"under {path}, so there is nothing to warm start from"
        )
    return found[-1]


class Search(Sweeper):
    def __init__(
        self,
        *,
        metric: str,
        cost: str,
        direction: Direction,
        n_trials: int,
        n_jobs: int,
        max_failure_rate: float,
        seed: int,
        num_random_samples: int,
        max_suggestion_cost: Optional[float],
        resample_frequency: int,
        num_candidates_for_suggestion_per_dim: int,
        initial_search_radius: float,
        exploration_bias: float,
        checkpoint_dir: Optional[str],
        warm_start_from: Optional[str],
        is_saved_on_every_observation: bool,
        is_wandb_logging_enabled: bool,
        params: Optional[Dict[str, Any]],
    ) -> None:
        assert 0 <= max_failure_rate <= 1
        self.metric = metric
        self.cost = cost
        self.direction = direction
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.max_failure_rate = max_failure_rate
        self.seed = seed
        self.num_random_samples = num_random_samples
        self.max_suggestion_cost = max_suggestion_cost
        self.resample_frequency = resample_frequency
        self.num_candidates_for_suggestion_per_dim = num_candidates_for_suggestion_per_dim
        self.initial_search_radius = initial_search_radius
        self.exploration_bias = exploration_bias
        self.checkpoint_dir = checkpoint_dir
        self.warm_start_from = warm_start_from
        self.is_saved_on_every_observation = is_saved_on_every_observation
        self.is_wandb_logging_enabled = is_wandb_logging_enabled
        self.params = {name: typed(entry) for name, entry in (params or {}).items()}

        self.hydra_context: Optional[HydraContext] = None
        self.config: Optional[DictConfig] = None
        self.launcher = None
        self.job_idx = 0

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.hydra_context = hydra_context
        self.config = config
        self.launcher = Plugins.instance().instantiate_launcher(
            hydra_context=hydra_context, task_function=task_function, config=config
        )

    def build(self, sweep_dir: Path):
        from carbs import CARBS, CARBSParams, Param

        parameters = [
            Param(name=name, space=space(entry), search_center=center(entry))
            for name, entry in self.params.items()
        ]
        settings = CARBSParams(
            better_direction_sign=1 if self.direction == Direction.maximize else -1,
            seed=self.seed,
            num_random_samples=self.num_random_samples,
            max_suggestion_cost=self.max_suggestion_cost,
            resample_frequency=self.resample_frequency,
            num_candidates_for_suggestion_per_dim=self.num_candidates_for_suggestion_per_dim,
            initial_search_radius=self.initial_search_radius,
            exploration_bias=self.exploration_bias,
            is_wandb_logging_enabled=self.is_wandb_logging_enabled,
            is_saved_on_every_observation=self.is_saved_on_every_observation,
            checkpoint_dir=str(self.checkpoint_dir or sweep_dir / "carbs"),
        )
        optimizer = CARBS(settings, parameters)
        if self.warm_start_from:
            prior = newest(Path(self.warm_start_from))
            with trusted():
                optimizer.warm_start(str(prior), is_prior_observation_valid=True)
            log.info(
                "CarbsSweeper warm started from %s, %d observations carried over",
                prior,
                optimizer.observation_count,
            )
        return optimizer

    def sweep(self, arguments: List[str]) -> Any:
        assert self.config is not None
        assert self.launcher is not None
        assert self.params, "CarbsSweeper requires a non-empty `params` search space"

        from carbs import ObservationInParam

        maximize = self.direction == Direction.maximize
        sweep_dir = Path(self.config.hydra.sweep.dir)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, sweep_dir / "multirun.yaml")
        carbs = self.build(sweep_dir)

        log.info(
            "CarbsSweeper running %d trials, %d at a time, over %d parameters",
            self.n_trials,
            self.n_jobs,
            len(self.params),
        )

        lock = threading.Lock()
        best_value: Optional[float] = None
        best_params: Optional[Dict[str, Any]] = None
        remaining = self.n_trials
        completed = 0
        failures = 0
        aborted: List[Any] = []

        def claim() -> Optional[Tuple[Dict[str, Any], Tuple[str, ...], int]]:
            nonlocal remaining
            with lock:
                if remaining <= 0 or aborted:
                    return None
                remaining -= 1
                suggestion = carbs.suggest().suggestion
                override = tuple(
                    list(arguments)
                    + [f"{key}={value}" for key, value in suggestion.items() if key in self.params]
                )
                self.validate_batch_is_legal([override])
                idx = self.job_idx
                self.job_idx += 1
                return suggestion, override, idx

        def settle(suggestion: Dict[str, Any], ret: Any) -> None:
            nonlocal best_value, best_params, completed, failures
            with lock:
                completed += 1
                if ret is not None and ret.status == JobStatus.COMPLETED:
                    value = ret.return_value
                    score = float(value[self.metric])
                    cost = float(value[self.cost])
                    carbs.observe(ObservationInParam(input=suggestion, output=score, cost=cost))
                    if best_value is None or (score > best_value if maximize else score < best_value):
                        best_value, best_params = score, dict(suggestion)
                    return
                failures += 1
                carbs.observe(
                    ObservationInParam(input=suggestion, output=0.0, cost=0.0, is_failure=True)
                )
                if (
                    completed >= self.n_jobs
                    and failures / completed > self.max_failure_rate
                    and not aborted
                ):
                    log.error(
                        "failure rate %.2f over %d trials exceeds max %.2f; stopping",
                        failures / completed,
                        completed,
                        self.max_failure_rate,
                    )
                    aborted.append(ret)

        def work() -> None:
            while True:
                claimed = claim()
                if claimed is None:
                    return
                suggestion, override, idx = claimed
                try:
                    returns = self.launcher.launch([override], initial_job_idx=idx)
                    ret = returns[0] if returns else None
                except Exception:
                    log.exception("job %d failed to launch", idx)
                    ret = None
                settle(suggestion, ret)

        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            for future in [pool.submit(work) for _ in range(self.n_jobs)]:
                future.result()

        for ret in aborted:
            if ret is not None and ret.status != JobStatus.COMPLETED:
                ret.return_value

        results = {"name": "carbs", "best_value": best_value, "best_params": best_params}
        OmegaConf.save(OmegaConf.create(results), sweep_dir / "optimization_results.yaml")
        log.info("best value: %s", best_value)
        log.info("best params: %s", best_params)
        return results
