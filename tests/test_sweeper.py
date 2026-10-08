import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

pytest.importorskip("carbs")

from hydra_plugins.hydra_carbs_sweeper._search import typed  # noqa: E402
from hydra_plugins.hydra_carbs_sweeper._spaces import center, space  # noqa: E402
from hydra_plugins.hydra_carbs_sweeper.config import Distribution, ParamConfig  # noqa: E402

ROOT = Path(__file__).parents[1]
APP = Path(__file__).parent / "sweeper"


def entry(distribution, low, high, **kwargs):
    return ParamConfig(distribution=distribution, min=low, max=high, **kwargs)


def search(config_name="sphere", overrides=()):
    with initialize_config_dir(config_dir=str(APP / "conf"), version_base=None):
        cfg = compose(config_name, overrides=list(overrides), return_hydra_config=True)
        return instantiate(cfg.hydra.sweeper).search


def test_center_is_geometric_for_log_spaces():
    assert center(entry(Distribution.log_normal, 1e-4, 1e-2)) == 1e-3
    assert center(entry(Distribution.uniform, 0.0, 1.0)) == 0.5


def test_explicit_center_wins():
    assert center(entry(Distribution.uniform, 0.0, 1.0, center=0.25)) == 0.25


@pytest.mark.parametrize(
    "distribution, low, high, value",
    [
        (Distribution.uniform, -1.0, 1.0, 0.25),
        (Distribution.log_normal, 1e-4, 1e-1, 1e-3),
        (Distribution.logit_normal, 0.01, 0.99, 0.4),
    ],
)
def test_round_trip_through_basic(distribution, low, high, value):
    built = space(entry(distribution, low, high))
    assert math.isclose(built.param_from_basic(built.basic_from_param(value)), value, rel_tol=1e-6)


def test_pow2_snaps_to_powers_of_two():
    built = space(entry(Distribution.uniform_pow2, 32, 1024))
    assert built.param_from_basic(built.basic_from_param(256)) == 256


def test_plain_dict_entries_get_param_defaults():
    plain = typed({"distribution": "int_uniform", "min": 1, "max": 8})
    assert plain.distribution == Distribution.int_uniform
    assert (plain.center, plain.scale, plain.rounding_factor) == (None, None, 1)
    assert center(plain) == 4.5
    assert space(plain).is_integer


def test_a_top_level_search_space_is_the_default(tmp_path):
    params = search().params
    assert set(params) == {"x", "y"}
    assert params["x"].distribution == Distribution.uniform
    assert params["x"].rounding_factor == 1
    assert params["y"].center == 0.5
    search().build(tmp_path)


def test_without_a_search_space_there_is_nothing_to_sweep():
    assert search("bare").params == {}


def test_explicit_params_win():
    params = search("explicit").params
    assert set(params) == {"x", "y"}
    assert params["y"].distribution == Distribution.uniform
    assert (params["y"].min, params["y"].max) == (0.0, 1.0)
    replaced = search(overrides=["++hydra.sweeper.params={y:{distribution:uniform,min:0,max:1}}"])
    assert set(replaced.params) == {"y"}


def test_sweeps_the_search_space_end_to_end(tmp_path):
    path = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
    subprocess.run(
        [
            sys.executable,
            str(APP / "app.py"),
            "-m",
            "hydra.sweeper.n_trials=2",
            "hydra.sweeper.num_random_samples=2",
            f"hydra.sweep.dir={tmp_path}",
        ],
        check=True,
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": path},
    )
    results = OmegaConf.load(tmp_path / "optimization_results.yaml")
    assert set(results.best_params) >= {"x", "y"}
    assert len(list(tmp_path.glob("[0-9]*"))) == 2
