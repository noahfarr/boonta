import csv
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.utils import JobStatus
from hydra.utils import instantiate
from omegaconf import OmegaConf

from boonta.utils import load_checkpoint
from hangar import resolvers  # noqa: F401
from hydra_plugins.hydra_pbt_sweeper._population import (choose, perturb,
                                                         sample, typed)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "hangar/config"

SPACE = {
    "lr": {"distribution": "log_normal", "min": 1e-4, "max": 1e-2},
    "clip": {"distribution": "uniform", "min": 0.1, "max": 0.3},
    "epochs": {"distribution": "int_uniform", "min": 1, "max": 8},
    "minibatches": {"distribution": "uniform_pow2", "min": 1, "max": 32},
    "lambda": {"distribution": "logit_normal", "min": 0.8, "max": 0.99},
}


def params():
    return {name: typed(entry) for name, entry in SPACE.items()}


class Launcher:
    def __init__(self, score):
        self.score = score
        self.batches = []

    def launch(self, overrides, initial_job_idx):
        self.batches.append([dict(item.split("=", 1) for item in override) for override in overrides])
        return [
            SimpleNamespace(status=JobStatus.COMPLETED, return_value={"score": self.score(job), "cost": 1.0})
            for job in self.batches[-1]
        ]


def population(tmp_path, score, **settings):
    with initialize_config_dir(config_dir=str(CONFIG), version_base=None):
        cfg = compose(
            "config",
            overrides=["hydra/sweeper=pbt", f"hydra.sweep.dir={tmp_path}", "total_timesteps=1000", "training.num_epochs=6"],
            return_hydra_config=True,
        )
    for name, value in settings.items():
        cfg.hydra.sweeper[name] = value
    built = instantiate(cfg.hydra.sweeper).population
    built.config = cfg
    built.launcher = Launcher(score)
    built.validate_batch_is_legal = lambda batch: None
    return built


def test_samples_stay_inside_the_space():
    rng = np.random.default_rng(0)
    for _ in range(200):
        drawn = {name: sample(entry, rng) for name, entry in params().items()}
        for name, entry in params().items():
            assert entry.min <= drawn[name] <= entry.max
        assert isinstance(drawn["epochs"], int)
        assert drawn["minibatches"] in (1, 2, 4, 8, 16, 32)


def test_perturbation_scales_by_a_factor_and_clips_to_the_bounds():
    rng = np.random.default_rng(0)
    values = {"lr": 1e-3, "clip": 0.29, "epochs": 4, "minibatches": 8, "lambda": 0.95}
    for _ in range(50):
        moved = perturb(values, params(), rng, [0.8, 1.25], 0.0)
        assert moved["lr"] in (pytest.approx(8e-4), pytest.approx(1.25e-3))
        assert moved["clip"] in (pytest.approx(0.232), 0.3)
        assert moved["epochs"] in (3, 5)
        assert moved["lambda"] in (0.8, 0.99)


def test_categoricals_shift_to_a_neighbour_and_stop_at_the_ends():
    rng = np.random.default_rng(0)
    space = {"minibatches": params()["minibatches"]}
    shifted = {perturb({"minibatches": 8}, space, rng, [0.8, 1.25], 0.0)["minibatches"] for _ in range(100)}
    lowest = {perturb({"minibatches": 1}, space, rng, [0.8, 1.25], 0.0)["minibatches"] for _ in range(100)}
    highest = {perturb({"minibatches": 32}, space, rng, [0.8, 1.25], 0.0)["minibatches"] for _ in range(100)}
    assert shifted == {4, 16}
    assert lowest == {1, 2}
    assert highest == {16, 32}


@pytest.mark.parametrize("probability", [0.0, 0.25, 1.0])
def test_every_parameter_resamples_with_the_resample_probability(probability):
    rng = np.random.default_rng(0)
    values = {"lr": 1e-3, "clip": 0.2, "epochs": 4, "minibatches": 8, "lambda": 0.9}
    neighbours = {
        "lr": [8e-4, 1.25e-3],
        "clip": [0.16, 0.25],
        "epochs": [3, 5],
        "minibatches": [4, 16],
        "lambda": [0.8, 0.99],
    }
    trials = 4000
    resampled = {name: 0 for name in values}
    for _ in range(trials):
        moved = perturb(values, params(), rng, [0.8, 1.25], probability)
        for name, value in moved.items():
            resampled[name] += not np.isclose(value, neighbours[name]).any()
    for name in ("lr", "clip", "lambda"):
        assert resampled[name] / trials == pytest.approx(probability, abs=0.04), name
    for name in ("epochs", "minibatches"):
        assert (resampled[name] > 0) == (probability > 0), name


def test_the_bottom_fraction_copies_a_significantly_better_member():
    fitness = np.array([[1.0, 1.1], [5.0, 5.1], [9.0, 9.1], [3.0, 3.1]])
    sources = choose(fitness, np.random.default_rng(0), fraction=0.25, threshold=2.0)
    np.testing.assert_array_equal(sources, [2, 1, 2, 3])


@pytest.mark.parametrize(
    "members, fraction, count",
    [(4, 0.25, 1), (10, 0.25, 3), (5, 0.2, 1), (8, 0.5, 4), (3, 0.5, 1), (2, 0.25, 1), (1, 0.25, 0)],
)
def test_the_quantile_holds_the_ceiling_of_the_fraction_and_at_most_half(members, fraction, count):
    fitness = np.arange(members, dtype=np.float64).reshape(members, 1) * 100
    copied = set()
    for seed in range(50):
        sources = choose(fitness, np.random.default_rng(seed), fraction=fraction, threshold=0.0)
        losers = {member for member, source in enumerate(sources) if source != member}
        assert losers == set(range(count))
        assert all(source >= members - count for source in sources[:count])
        copied |= set(sources[:count].tolist())
    assert copied == set(range(members - count, members))


def test_a_gap_inside_the_noise_copies_nothing():
    fitness = np.array([[0.0, 10.0], [5.0, 5.0], [6.0, 14.0], [4.0, 6.0]])
    sources = choose(fitness, np.random.default_rng(0), fraction=0.25, threshold=2.0)
    np.testing.assert_array_equal(sources, [0, 1, 2, 3])


def test_a_member_without_a_score_always_copies():
    fitness = np.array([[-np.inf], [1.0], [2.0], [1.5]])
    sources = choose(fitness, np.random.default_rng(0), fraction=0.25, threshold=1e9)
    np.testing.assert_array_equal(sources, [2, 1, 2, 3])


def test_structural_keys_are_rejected_up_front(tmp_path):
    built = population(tmp_path, lambda job: 0.0)
    built.params["environment.num_envs"] = typed({"distribution": "uniform_pow2", "min": 8, "max": 64})
    with pytest.raises(ValueError, match="environment.num_envs"):
        built.sweep([])


def test_generations_must_divide_the_epochs(tmp_path):
    built = population(tmp_path, lambda job: 0.0, generations=4)
    with pytest.raises(ValueError, match="must divide training.num_epochs"):
        built.sweep([])
    assert built.launcher.batches == []


@pytest.mark.parametrize(
    "choice, settings, expected",
    [
        ("default", [], [False, False, False, False]),
        ("epochs", ["early_stopping.at=2"], [False, True, True, True]),
        ("steps", ["early_stopping.at=100"], [False, True, True, True]),
        ("plateau", ["early_stopping.patience=2"], [False, False, False, True]),
    ],
)
def test_early_stopping_stops_on_its_condition(choice, settings, expected):
    with initialize_config_dir(config_dir=str(CONFIG), version_base=None):
        cfg = compose("config", overrides=[f"early_stopping={choice}", *settings])
    stop = instantiate(cfg.early_stopping)
    epochs = [(1, 50, 1.0), (2, 100, 2.0), (3, 150, 1.5), (4, 200, 1.9)]
    assert [stop(epoch, step, {cfg.score: np.array([[value]])}) for epoch, step, value in epochs] == expected


def test_each_generation_resumes_from_the_one_before(tmp_path):
    built = population(tmp_path, lambda job: float(job["optimizer.lr"]) * 1000, members=4, seeds=2, generations=3)
    results = built.sweep(["algorithm=ppo", "total_timesteps=5"])

    batches = built.launcher.batches
    assert len(batches) == 3
    for generation, batch in enumerate(batches):
        assert len(batch) == 8
        for index, job in enumerate(batch):
            member, seed = divmod(index, 2)
            assert job["total_timesteps"] == "5"
            assert job["early_stopping"] == "epochs"
            assert job["early_stopping.at"] == str(2 * (generation + 1))
            assert job["scoring"] == "final"
            assert job["save"] == str(tmp_path / f"checkpoints/generation_{generation}/member_{member}/seed_{seed}")
            if generation == 0:
                assert "checkpoint" not in job
            else:
                assert job["checkpoint"].endswith(f"seed_{seed}")
                assert f"generation_{generation - 1}/" in job["checkpoint"]

    with open(tmp_path / "population.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert {"generation", "member", "parent", "score", "score_0", "score_1", "optimizer.lr"} <= set(rows[0])
    assert len(results["schedule"]) == 3
    assert results["best_value"] == max(float(row["score"]) for row in rows if row["generation"] == "2")


def test_a_copy_loads_the_winners_checkpoint_and_perturbs_its_parameters(tmp_path):
    built = population(tmp_path, lambda job: float(job["optimizer.lr"]) * 1000, members=4, seeds=1, generations=2, threshold=0.0, resample_probability=0.0)
    built.sweep([])
    first, second = built.launcher.batches
    scores = [float(job["optimizer.lr"]) for job in first]
    best, worst = int(np.argmax(scores)), int(np.argmin(scores))
    assert second[worst]["checkpoint"].endswith(f"generation_0/member_{best}/seed_0")
    assert float(second[worst]["optimizer.lr"]) / scores[best] in (pytest.approx(0.8), pytest.approx(1.25))
    for member in set(range(4)) - {worst}:
        assert second[member]["checkpoint"].endswith(f"generation_0/member_{member}/seed_0")
        assert second[member]["optimizer.lr"] == first[member]["optimizer.lr"]


def test_pbt_trains_ppo_and_resumes_each_member(tmp_path):
    path = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "hangar/main.py"),
            "-m",
            "hydra/sweeper=pbt",
            "algorithm=ppo",
            "environment=gymnax/minatar/breakout",
            "logger=file",
            "environment.num_envs=8",
            "rollout.num_steps=4",
            "algorithm.num_minibatches=1",
            "training.num_epochs=2",
            "evaluation.num_steps=8",
            "total_timesteps=128",
            "hydra.sweeper.members=2",
            "hydra.sweeper.seeds=1",
            "hydra.sweeper.generations=2",
            f"hydra.sweep.dir={tmp_path}",
        ],
        check=True,
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": path, "JAX_PLATFORMS": "cpu"},
    )
    for member in range(2):
        first, second = (
            load_checkpoint(tmp_path / f"checkpoints/generation_{generation}/member_{member}/seed_0/algorithm_state")
            for generation in range(2)
        )
        assert int(first["step"]) == 64
        assert int(second["step"]) == 128
    results = OmegaConf.load(tmp_path / "optimization_results.yaml")
    assert set(results.best_params) == {"optimizer.lr", "algorithm.entropy_coefficient"}
