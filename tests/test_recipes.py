import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

from boonta.datasets.disk import write
from hangar import recipes

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "hangar/config"
SMALL = ["environment.num_envs=32", "++rollout.num_steps=4", "++replay.capacity=1024"]


def installed(*modules):
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    return f"needs {', '.join(missing)}" if missing else None


def rom(path):
    return None if (ROOT / path).exists() else f"needs {path}, which is not committed"


def cached(repo_id):
    from huggingface_hub import try_to_load_from_cache

    found = isinstance(try_to_load_from_cache(repo_id, "config.json"), str)
    return None if found else f"needs {repo_id} in the Hugging Face cache"


def hopper_dataset(directory):
    generator = np.random.default_rng(0)
    write(
        directory,
        [
            SimpleNamespace(
                observations=generator.normal(size=(9, 11)),
                actions=generator.uniform(-1, 1, size=(8, 3)),
                rewards=generator.normal(size=8),
                terminations=np.zeros(8, bool),
                truncations=np.eye(8, dtype=bool)[-1],
            )
            for _ in range(16)
        ],
    )
    return ["dataset=disk", f"dataset.dataset_id={directory}", "algorithm.batch_size=64"]


MINATAR = "gymnax/minatar/breakout"
HOPPER = "brax/mujoco/hopper"

CASES = [
    ("ppo", "gymnasium/cartpole", installed("gymnasium"), False),
    ("ppo", "gymnax/classic_control/cartpole", None, False),
    ("ppo", "gymnax/bsuite/deep_sea", None, False),
    ("ppo", MINATAR, None, False),
    ("dqn", MINATAR, None, False),
    ("pqn", MINATAR, None, False),
    ("grpo", MINATAR, None, False),
    ("recurrent_ppo", MINATAR, None, False),
    ("recurrent_dqn", MINATAR, None, False),
    ("recurrent_pqn", MINATAR, None, False),
    ("recurrent_pupo", MINATAR, None, False),
    ("ppo", HOPPER, installed("brax"), False),
    ("sac", HOPPER, installed("brax"), False),
    ("reppo", HOPPER, installed("brax"), False),
    ("recurrent_sac", HOPPER, installed("brax"), False),
    ("bc", HOPPER, installed("brax"), True),
    ("iql", HOPPER, installed("brax"), True),
    ("ppo", "jumanji/sokoban", installed("jumanji"), False),
    ("recurrent_pupo", "jumanji/sokoban", installed("jumanji"), False),
    ("ppo", "mujoco_playground/dm_control_suite/cartpole_balance", installed("mujoco_playground"), False),
    ("ppo", "craftax/craftax_classic/symbolic", installed("craftax"), False),
    ("recurrent_pupo", "craftax/craftax_classic/symbolic", installed("craftax"), False),
    ("ppo", "xland_minigrid/minigrid/empty_5x5", installed("xminigrid"), False),
    ("ippo", "connectx/connectx", None, False),
    ("ippo", "jaxmarl/smax/3m", installed("jaxmarl"), False),
    ("mappo", "jaxmarl/smax/3m", installed("jaxmarl"), False),
    ("ippo", "mapox/find_return", installed("mapox"), False),
    ("recurrent_pupo", "mapox/find_return", installed("mapox"), False),
    ("recurrent_pupo", "ale/montezuma", installed("ale_py"), False),
    ("ppo", "isaaclab/classic/cartpole", installed("isaaclab"), False),
    (
        "recurrent_ppo",
        "peanut_gb/pokemon_red",
        rom("boonta/environments/peanut_gb/roms/pokemon_red.gb"),
        False,
    ),
    ("recurrent_ppo", "wordle/wordle", cached("Qwen/Qwen3-0.6B-Base"), False),
    ("recurrent_grpo", "wordle/wordle", cached("Qwen/Qwen3-0.6B-Base"), False),
]


def configure(algorithm, environment, *overrides):
    with initialize_config_dir(config_dir=str(CONFIG), version_base=None):
        overrides = [f"algorithm={algorithm}", f"environment={environment}", *overrides]
        return compose("config", overrides=overrides, return_hydra_config=True)


def test_every_recipe_has_a_smoke_case():
    covered = set()
    for algorithm, environment, *_ in CASES:
        namespace = configure(algorithm, environment).environment.namespace
        suite = configure(algorithm, environment).environment.get("suite", namespace)
        covered.add(recipes.register.get((algorithm, namespace, suite)))
    assert covered >= set(recipes.register.values())


@pytest.mark.parametrize(
    "algorithm, environment, missing, offline",
    [
        pytest.param(*case, id=f"{case[0]}-{case[1]}", marks=pytest.mark.skipif(case[2] is not None, reason=str(case[2])))
        for case in CASES
    ],
)
def test_every_recipe_builds_and_runs_one_update(algorithm, environment, missing, offline, tmp_path):
    overrides = SMALL + (hopper_dataset(tmp_path) if offline else [])
    cfg = configure(algorithm, environment, *overrides)
    HydraConfig.instance().set_config(cfg)
    podracer = recipes.make(cfg)

    state = podracer.init(jax.random.key(0))
    state, logs = podracer.train(state, jax.random.key(1), 1)
    for name, values in logs.items():
        values = np.asarray(values, dtype=np.float64)
        assert not np.isinf(values).any(), name
    podracer.close(state)
