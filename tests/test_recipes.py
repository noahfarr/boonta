import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from dummies import publish
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
    dataset_id = publish(
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
        "dummy/hopper/expert-v0",
    )
    return [f"dataset.dataset_id={dataset_id}", "algorithm.batch_size=64"]


def hopper_pool(directory):
    return hopper_dataset(directory) + ["dataset.kwargs.pool_size=64"]


def kinetix_environment(env_id=None):
    from kinetix.environment import StaticEnvParams

    from boonta import environments

    static = StaticEnvParams(
        num_polygons=6, num_circles=3, num_joints=2, num_thrusters=2, frame_skip=2
    )
    env = environments.make(
        "kinetix",
        env_id,
        kwargs={
            "action_type": "multi_discrete",
            "observation_type": "symbolic_entity",
            "static_env_params": static,
        },
    )
    return static, env


def kinetix_dataset(directory):
    import zarr

    trajectories, steps = 8, 8
    static, env = kinetix_environment()
    env_state, _ = env.init(jax.random.key(0))
    width = static.num_motor_bindings + static.num_thruster_bindings
    fields = {
        "env_state/" + jax.tree_util.keystr(path, simple=True, separator="/"): np.stack(
            [np.asarray(leaf)] * steps
        )
        for path, leaf in jax.tree_util.tree_flatten_with_path(env_state)[0]
    }
    fields |= {
        "action": np.ones((steps, width), np.int32),
        "action_mask": np.ones((steps, width), bool),
        "done": np.eye(steps, dtype=bool)[-1],
    }
    records = np.zeros(
        trajectories,
        [(name, value.dtype, (steps, 1, *value.shape[1:])) for name, value in fields.items()],
    )
    for name, value in fields.items():
        records[name] = value[:, None]
    group = zarr.open_group(str(directory), mode="w")
    for shard in ("shard_000", "shard_001"):
        group.array(shard, records)
    return [
        f"dataset.dataset_id={directory}",
        "dataset.kwargs.batch_size=4",
        f"podracer.config.batch_shape=[4,{steps}]",
    ]


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
    ("bc", HOPPER, installed("brax"), hopper_dataset),
    ("bc", HOPPER, installed("brax"), hopper_pool),
    ("iql", HOPPER, installed("brax"), hopper_dataset),
    ("recurrent_bc", "kinetix/kinetix", installed("kinetix", "zarr"), kinetix_dataset),
    ("ppo", "kinetix/kinetix", installed("kinetix"), False),
    ("ppo", "jumanji/sokoban", installed("jumanji"), False),
    ("recurrent_pupo", "jumanji/sokoban", installed("jumanji"), False),
    ("ppo", "mujoco_playground/dm_control_suite/cartpole_balance", installed("mujoco_playground"), False),
    ("ppo", "craftax/craftax_classic/symbolic", installed("craftax"), False),
    ("recurrent_pupo", "craftax/craftax_classic/symbolic", installed("craftax"), False),
    ("ppo", "xland_minigrid/minigrid/empty_5x5", installed("xminigrid"), False),
    ("ippo", "connectx/connectx", None, False),
    ("mmd", "connectx/connectx", None, False),
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


def test_hyperparameters_cascade_to_everything_below_them():
    assert configure("ppo", "gymnax/minatar/asterix").total_timesteps == 20_000_000
    assert configure("ppo", "isaaclab/classic/ant").algorithm.num_minibatches == 4
    assert configure("ppo", "isaaclab/classic/ant").optimizer.lr == 5e-4


@pytest.mark.parametrize("overrides, seed", [((), 0), (("hydra.job.num=3",), 3)])
def test_a_sweep_seeds_each_trial_with_its_number(overrides, seed):
    cfg = configure("ppo", MINATAR, *overrides)
    HydraConfig.instance().set_config(cfg)
    assert cfg.seed == seed


def test_only_carbs_sweeps_the_search_space():
    assert "search_space" not in configure("ppo", MINATAR)
    plain = configure("ippo", "connectx/connectx", "hydra.mode=MULTIRUN")
    assert len(plain.search_space) == 13
    assert plain.hydra.sweeper.params is None


@pytest.mark.skipif(bool(installed("carbs")), reason="needs carbs")
def test_carbs_reads_the_search_space():
    swept = configure("ippo", "connectx/connectx", "hydra/sweeper=carbs")
    params = instantiate(swept.hydra.sweeper).search.params
    assert len(params) == 13
    assert params["optimizer.lr"].center == 1.46e-3


@pytest.mark.parametrize(
    "algorithm, environment, missing, offline",
    [
        pytest.param(*case, id=f"{case[0]}-{case[1]}", marks=pytest.mark.skipif(case[2] is not None, reason=str(case[2])))
        for case in CASES
    ],
)
def test_every_recipe_builds_and_runs_one_update(
    algorithm, environment, missing, offline, tmp_path, monkeypatch
):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path))
    overrides = SMALL + (offline(tmp_path) if offline else [])
    cfg = configure(algorithm, environment, *overrides)
    HydraConfig.instance().set_config(cfg)
    podracer = recipes.make(cfg)

    state = podracer.init(jax.random.key(0))
    state, logs = podracer.train(state, jax.random.key(1), 1)
    for name, values in logs.items():
        values = np.asarray(values, dtype=np.float64)
        assert not np.isinf(values).any(), name
    podracer.close(state)


@pytest.mark.parametrize("algorithm", ["recurrent_ppo", "recurrent_grpo"])
def test_a_wordle_recipe_loads_the_model_its_config_names(algorithm, monkeypatch):
    cfg = configure(algorithm, "wordle/wordle", "network.repo_id=someone/else")
    recipe = importlib.import_module(f"hangar.recipes.{algorithm}_wordle")
    loaded = []

    def refuse(repo_id):
        loaded.append(repo_id)
        raise LookupError(repo_id)

    monkeypatch.setattr(recipe, "load_config", refuse)
    with pytest.raises(LookupError):
        recipe.make(cfg)
    assert loaded == ["someone/else"]


def test_every_recipe_takes_its_learning_rate_from_one_schedule():
    copies = [
        path.name
        for path in (ROOT / "hangar/recipes").glob("*.py")
        if path.name != "schedules.py" and "cosine_decay_schedule" in path.read_text()
    ]
    assert copies == []


def test_an_annealed_learning_rate_decays_over_every_gradient_step():
    from omegaconf import OmegaConf

    from hangar.recipes.schedules import learning_rate

    cfg = OmegaConf.create(
        {
            "total_timesteps": 100,
            "optimizer": {"lr": 1.0, "anneal": True, "alpha": 0.25},
            "algorithm": {"update_epochs": 2, "num_minibatches": 3},
        }
    )
    schedule = learning_rate(cfg, batch_size=10)
    np.testing.assert_allclose([schedule(0), schedule(60)], [1.0, 0.25])

    cfg.optimizer.anneal = False
    assert learning_rate(cfg, batch_size=10) == 1.0


KINETIX = pytest.mark.skipif(
    installed("kinetix", "zarr") is not None, reason=str(installed("kinetix", "zarr"))
)


@KINETIX
@pytest.mark.parametrize("prior", [False, True])
def test_kinetix_entities_see_the_previous_action_only_when_asked(prior):
    import jax.numpy as jnp

    from hangar.recipes.recurrent_bc_kinetix import Entities, cardinality

    static, env = kinetix_environment()
    _, timestep = env.init(jax.random.key(0))
    values = cardinality(static)
    obs = {
        "entities": jax.tree.map(lambda leaf: leaf[None, None], timestep.obs),
        "mask": jnp.ones((1, 1, len(values)), bool),
    }
    action = jnp.zeros((1, 1, len(values)), jnp.int32)
    entities = Entities(features=16, num_layers=1, num_heads=2, values=values, prior=prior)
    params = entities.init(jax.random.key(0), obs, action)
    assert ("Embed_0" in params["params"]) == prior


@KINETIX
def test_kinetix_evaluation_reports_a_success_rate():
    import jax.numpy as jnp
    import lox

    from boonta.environments.wrappers import (RecordEpisodeStatistics,
                                              SameStepAutoReset, Vectorize)
    from hangar.recipes.recurrent_bc_kinetix import Scored, cardinality

    static, env = kinetix_environment()
    time_limit = env.time_limit()
    env = Vectorize(Scored(RecordEpisodeStatistics(SameStepAutoReset(env))), num_envs=4)
    action = jnp.zeros((4, len(cardinality(static))), jnp.int32)

    def roll(key):
        state, _ = env.init(key)

        def once(state, key):
            state, _ = env.step(key, state, action)
            return state, None

        state, _ = jax.lax.scan(once, state, jax.random.split(key, time_limit + 1))
        return state

    _, logs = jax.jit(lox.spool(roll))(jax.random.key(0))
    rates = np.asarray(logs["episode_statistics/success_rate"])
    finished = rates[~np.isnan(rates)]
    assert finished.size
    assert np.all(np.isin(finished, [0.0, 1.0]))


@KINETIX
def test_kinetix_restarts_at_the_level_it_is_given_and_observes_it():
    _, env = kinetix_environment()
    state, _ = env.init(jax.random.key(0))
    level, timestep = env.init(jax.random.key(1))
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(jax.tree.leaves(state), jax.tree.leaves(level))
    )
    state = env.update(state, theta=level)
    for started, given in zip(jax.tree.leaves(state), jax.tree.leaves(level)):
        np.testing.assert_array_equal(started, given)
    for seen, shown in zip(jax.tree.leaves(env.observe(state)), jax.tree.leaves(timestep.obs)):
        np.testing.assert_array_equal(seen, shown)


@KINETIX
def test_kinetix_holdout_draws_distinct_levels():
    from omegaconf import OmegaConf

    names = OmegaConf.load(CONFIG / "environment/kinetix/holdout_m.yaml").env_id
    assert len(names) == 24
    _, env = kinetix_environment(list(names))
    seen = [np.asarray(env.init(jax.random.key(seed))[1].obs.polygons) for seed in range(12)]
    assert any(not np.array_equal(seen[0], other) for other in seen[1:])


@KINETIX
def test_kinetix_evaluates_on_the_levels_the_dataset_reserved(tmp_path):
    overrides = kinetix_dataset(tmp_path) + [
        "environment.num_envs=4",
        "+dataset.kwargs.val_shards=1",
    ]
    cfg = configure("recurrent_bc", "kinetix/holdout_levels", *overrides)
    HydraConfig.instance().set_config(cfg)
    podracer = recipes.make(cfg)

    _, reference = kinetix_environment()[1].init(jax.random.key(0))
    state = podracer.init(jax.random.key(1))
    polygons = np.asarray(state.timestep.obs["entities"].polygons)
    np.testing.assert_allclose(polygons, np.broadcast_to(reference.obs.polygons, polygons.shape))

    state, _ = podracer.evaluate(state, jax.random.key(2), 4)
    podracer.close(state)
