import csv
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf

from hydra_plugins.hydra_carbs_sweeper.config import Distribution, ParamConfig

log = logging.getLogger(__name__)

STRUCTURAL = (
    "total_timesteps",
    "environment.num_envs",
    "rollout.num_steps",
    "replay.capacity",
    "network.",
    "cell.",
    "torso.",
    "stack.",
)
CONTINUOUS = ("uniform", "log_normal", "logit_normal")


def typed(entry: Any) -> DictConfig:
    plain = OmegaConf.to_container(OmegaConf.create(entry), resolve=True, enum_to_str=True)
    return OmegaConf.merge(OmegaConf.structured(ParamConfig), plain)


def kind(entry) -> str:
    if isinstance(entry.distribution, Distribution):
        return entry.distribution.name
    return str(entry.distribution)


def structural(name: str) -> bool:
    return any(name == key or (key.endswith(".") and name.startswith(key)) for key in STRUCTURAL)


def sample(entry, rng: np.random.Generator):
    low, high = float(entry.min), float(entry.max)
    name = kind(entry)
    if name == "uniform":
        return float(rng.uniform(low, high))
    if name == "int_uniform":
        return int(rng.integers(int(low), int(high) + 1))
    if name == "log_normal":
        return float(math.exp(rng.uniform(math.log(low), math.log(high))))
    if name == "logit_normal":
        logit = rng.uniform(math.log(low / (1 - low)), math.log(high / (1 - high)))
        return float(1 / (1 + math.exp(-logit)))
    if name == "uniform_pow2":
        return int(2 ** rng.integers(round(math.log2(low)), round(math.log2(high)) + 1))
    raise ValueError(f"unknown distribution {name!r}")


def perturb(
    values: Dict[str, Any],
    params: Dict[str, Any],
    rng: np.random.Generator,
    factors: Sequence[float],
    resample_probability: float,
) -> Dict[str, Any]:
    perturbed = {}
    for name, value in values.items():
        entry = params[name]
        low, high = float(entry.min), float(entry.max)
        if rng.random() < resample_probability:
            perturbed[name] = sample(entry, rng)
        elif kind(entry) in CONTINUOUS:
            perturbed[name] = float(np.clip(value * rng.choice(factors), low, high))
        elif kind(entry) == "int_uniform":
            perturbed[name] = int(np.clip(round(value * rng.choice(factors)), low, high))
        else:
            exponent = round(math.log2(value)) + rng.choice([-1, 1])
            perturbed[name] = int(2 ** np.clip(exponent, round(math.log2(low)), round(math.log2(high))))
    return perturbed


def choose(
    fitness: np.ndarray, rng: np.random.Generator, fraction: float, threshold: float
) -> np.ndarray:
    members, seeds = fitness.shape
    count = min(math.ceil(fraction * members), members // 2)
    mean = fitness.mean(axis=1)
    variance = np.zeros(members)
    if seeds > 1:
        variance = fitness.var(axis=1, ddof=1)
    order = np.argsort(mean, kind="stable")
    losers, winners = order[:count], order[::-1][:count]
    sources = np.arange(members)
    for loser in losers:
        winner = rng.choice(winners)
        if winner == loser:
            continue
        gap = mean[winner] - mean[loser]
        error = math.sqrt((variance[winner] + variance[loser]) / seeds)
        if not np.isfinite(mean[loser]) or gap > threshold * error:
            sources[loser] = winner
    return sources


def key(argument: str) -> str:
    name, *_ = argument.split("=", 1)
    return name.lstrip("+~")


class Population(Sweeper):
    def __init__(
        self,
        *,
        metric: str,
        members: int,
        seeds: int,
        generations: int,
        interval: Optional[int],
        fraction: float,
        threshold: float,
        factors: Sequence[float],
        resample_probability: float,
        seed: int,
        params: Optional[Dict[str, Any]],
    ) -> None:
        self.metric = metric
        self.members = members
        self.seeds = seeds
        self.generations = generations
        self.interval = interval
        self.fraction = fraction
        self.threshold = threshold
        self.factors = [float(factor) for factor in factors]
        self.resample_probability = resample_probability
        self.seed = seed
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

    def validate(self) -> None:
        assert self.params, "PbtSweeper requires a non-empty `params` search space"
        blocked = [name for name in self.params if structural(name)]
        if blocked:
            raise ValueError(
                f"PbtSweeper cannot search {blocked}: a member resumes from another "
                f"member's checkpoint, so it can only change parameters that keep the "
                f"shapes of the algorithm state, and the sweeper sets total_timesteps "
                f"to each generation's interval itself"
            )

    def budget(self) -> int:
        if self.interval is not None:
            return int(self.interval)
        return int(self.config.total_timesteps) // self.generations

    def launch(self, arguments, generation, values, parents, checkpoints, interval):
        kept = [argument for argument in arguments if key(argument) not in self.reserved]
        overrides = []
        for member in range(self.members):
            for seed in range(self.seeds):
                override = kept + [f"{name}={value}" for name, value in values[member].items()]
                override += [
                    f"total_timesteps={interval}",
                    "scoring=final",
                    f"save={checkpoints / f'generation_{generation}/member_{member}/seed_{seed}'}",
                ]
                if generation > 0:
                    parent = checkpoints / f"generation_{generation - 1}/member_{parents[member]}/seed_{seed}"
                    override.append(f"checkpoint={parent}")
                overrides.append(tuple(override))
        self.validate_batch_is_legal(overrides)
        returns = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
        self.job_idx += len(overrides)
        scores = [float(ret.return_value[self.metric]) for ret in returns]
        fitness = np.array(scores, dtype=np.float64).reshape(self.members, self.seeds)
        return np.where(np.isnan(fitness), -np.inf, fitness)

    @property
    def reserved(self):
        return {*self.params, "total_timesteps", "scoring", "save", "checkpoint"}

    def sweep(self, arguments: List[str]) -> Any:
        assert self.config is not None
        assert self.launcher is not None
        self.validate()

        sweep_dir = Path(self.config.hydra.sweep.dir).absolute()
        sweep_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, sweep_dir / "multirun.yaml")
        checkpoints = sweep_dir / "checkpoints"
        interval = self.budget()
        assert interval >= 1, "PbtSweeper needs an interval of at least one step"

        log.info(
            "PbtSweeper running %d generations of %d steps, %d members x %d seeds, over %d parameters",
            self.generations,
            interval,
            self.members,
            self.seeds,
            len(self.params),
        )

        rng = np.random.default_rng(self.seed)
        values = [
            {name: sample(entry, rng) for name, entry in self.params.items()}
            for _ in range(self.members)
        ]
        parents = list(range(self.members))
        rows = []

        for generation in range(self.generations):
            fitness = self.launch(arguments, generation, values, parents, checkpoints, interval)
            for member in range(self.members):
                rows.append(
                    {
                        "generation": generation,
                        "member": member,
                        "parent": parents[member] if generation > 0 else "",
                        "score": float(fitness[member].mean()),
                        **{f"score_{seed}": float(fitness[member, seed]) for seed in range(self.seeds)},
                        **values[member],
                    }
                )
            self.record(sweep_dir / "population.csv", rows)
            log.info(
                "generation %d: member scores %s",
                generation,
                np.round(fitness.mean(axis=1), 3).tolist(),
            )

            if generation + 1 < self.generations:
                sources = choose(fitness, rng, self.fraction, self.threshold)
                values = [
                    values[member]
                    if source == member
                    else perturb(values[source], self.params, rng, self.factors, self.resample_probability)
                    for member, source in enumerate(sources.tolist())
                ]
                parents = sources.tolist()
                for member, source in enumerate(parents):
                    if source != member:
                        log.info("member %d copies member %d", member, source)

        results = self.summarize(rows, checkpoints)
        OmegaConf.save(OmegaConf.create(results), sweep_dir / "optimization_results.yaml")
        log.info("best value: %s", results["best_value"])
        log.info("best params: %s", results["best_params"])
        return results

    def record(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def summarize(self, rows: List[Dict[str, Any]], checkpoints: Path) -> Dict[str, Any]:
        last = self.generations - 1
        table = {(row["generation"], row["member"]): row for row in rows}
        final = [table[(last, member)] for member in range(self.members)]
        best = max(final, key=lambda row: row["score"])
        schedule = []
        member = best["member"]
        for generation in range(last, -1, -1):
            row = table[(generation, member)]
            schedule.append(
                {
                    "generation": generation,
                    "member": member,
                    "score": row["score"],
                    "params": {name: row[name] for name in self.params},
                }
            )
            if generation > 0:
                member = row["parent"]
        return {
            "name": "pbt",
            "best_value": best["score"],
            "best_member": best["member"],
            "best_params": {name: best[name] for name in self.params},
            "best_checkpoint": str(checkpoints / f"generation_{last}/member_{best['member']}"),
            "schedule": schedule[::-1],
        }
