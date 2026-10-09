import dataclasses
import signal
import subprocess
import sys
import time
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from omegaconf import OmegaConf

import zoo
from boonta.algorithms.wrappers.population import Population
from boonta.artisans import Checkpointer
from boonta.loggers import FileLogger, OrbaxLogger
from boonta.artisans import Checkpoint, Video
from boonta.utils import load_checkpoint, newest, sharded
from dummies import corridor
from hangar.brief import brief

NUM_ENVS, NUM_STEPS = 8, 4
online = partial(zoo.online, devices=2)


def pair(environment, num_envs, num_steps, podracer):
    algorithm = zoo.ppo(environment, num_envs=num_envs, num_steps=num_steps).algorithm
    return podracer(
        Population(algorithm, count=2), zoo.wrap(environment, num_envs), num_envs, num_steps
    )


BUILDS = [
    pytest.param(zoo.ppo, id="ppo"),
    pytest.param(zoo.recurrent_ppo, id="recurrent_ppo"),
    pytest.param(pair, id="population"),
]


def build_podracer(build, num_envs=NUM_ENVS):
    return build(corridor(), num_envs=num_envs, num_steps=NUM_STEPS, podracer=online)


def train(podracer):
    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), 3)
    return state


def save(directory, state, **options):
    logger = OrbaxLogger(str(directory / "checkpoints"))
    logs = {"episode_statistics/episode_return": np.ones(4)}
    logger.log_artifact(Checkpointer(**options).craft(None, None, state, logs), step=96)
    logger.finish()


def assert_restored(saved, loaded):
    assert jax.tree.structure(loaded) == jax.tree.structure(saved)
    for before, after in zip(jax.tree.leaves(saved), jax.tree.leaves(loaded)):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
        assert after.dtype == before.dtype
        assert after.sharding == before.sharding


def moved(before, after):
    return any(
        not np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(jax.tree.leaves(before), jax.tree.leaves(after))
    )


@pytest.mark.parametrize("build", BUILDS)
def test_resume_restores_the_whole_algorithm_state(tmp_path, build):
    podracer = build_podracer(build)
    trained = train(podracer)
    fresh = podracer.init(jax.random.key(0))
    save(tmp_path, trained)

    restored = load_checkpoint(newest(tmp_path), fresh)

    assert moved(fresh.algorithm_state.params, trained.algorithm_state.params)
    assert_restored(trained.algorithm_state, restored.algorithm_state)
    steps = int(restored.algorithm_state.step)
    resumed, _ = podracer.train(restored, jax.random.key(2), 1)
    assert int(resumed.algorithm_state.step) == steps + podracer.batch_size


@pytest.mark.parametrize(
    "fields",
    [("algorithm_state",), ("algorithm_state", "environment_state")],
    ids=["default", "with-environment"],
)
def test_resume_restores_each_saved_field_and_leaves_the_rest_fresh(tmp_path, fields):
    podracer = build_podracer(zoo.ppo)
    trained = train(podracer)
    fresh = podracer.init(jax.random.key(0))
    save(tmp_path, trained, fields=fields)

    restored = load_checkpoint(newest(tmp_path), fresh)

    items = {item.name for item in (tmp_path / "checkpoints/latest/96").iterdir()}
    assert items == {*fields, "_CHECKPOINT_METADATA"}
    for field in ("algorithm_state", "environment_state", "timestep"):
        assert moved(getattr(fresh, field), getattr(trained, field))
        expected = trained if field in fields else fresh
        assert_restored(getattr(expected, field), getattr(restored, field))


def test_resume_fails_when_a_saved_field_changed_shape(tmp_path):
    saved = build_podracer(zoo.recurrent_ppo)
    wider = build_podracer(zoo.recurrent_ppo, num_envs=2 * NUM_ENVS)
    save(tmp_path, saved.init(jax.random.key(0)))

    with pytest.raises(ValueError, match="stored shape"):
        load_checkpoint(newest(tmp_path), wider.init(jax.random.key(0)))


def test_newest_resolves_a_run_directory_to_its_latest_or_best_checkpoint(tmp_path):
    log_scores(tmp_path / "checkpoints", [1.0, 3.0, 2.0])

    assert newest(tmp_path) == str(tmp_path / "checkpoints/latest/3")
    assert newest(tmp_path, "best") == str(tmp_path / "checkpoints/best/2")
    assert newest(tmp_path / "checkpoints/best/2") == str(tmp_path / "checkpoints/best/2")
    assert newest(None) is None


def test_the_checkpointer_saves_exactly_the_fields_it_names():
    state = SimpleNamespace(algorithm_state="learned", environment_state="played")

    default = Checkpointer().craft(None, None, state, {})
    both = Checkpointer(fields=["algorithm_state", "environment_state"]).craft(
        None, None, state, {}
    )

    assert default.data == {"algorithm_state": "learned"}
    assert both.data == {"algorithm_state": "learned", "environment_state": "played"}


def test_the_checkpointer_scores_by_mean_episode_return_unless_told_otherwise():
    state = SimpleNamespace(algorithm_state="learned")
    logs = {"episode_statistics/episode_return": np.array([1.0, np.nan, 5.0]), "bonus": [7.0]}

    assert Checkpointer().craft(None, None, state, logs).score == 3.0
    assert Checkpointer(score=lambda logs: logs["bonus"]).craft(None, None, state, logs).score == 7.0
    assert Checkpointer().craft(None, None, state, {}).score == float("-inf")


def log_scores(directory, scores, **options):
    logger = OrbaxLogger(directory=str(directory), max_to_keep=len(scores), **options)
    for step, score in enumerate(scores, start=1):
        weights = {"params": {"w": jnp.full((2,), score)}}
        logger.log_artifact(Checkpoint("checkpoint", weights, score=score), step)
    logger.finish()


def kept(directory):
    return sorted(path.name for path in directory.iterdir() if path.name.isdigit())


@pytest.mark.parametrize(
    "options, best",
    [({}, ["2"]), ({"best_mode": "min"}, ["1"]), ({"best_to_keep": 2}, ["2", "3"])],
    ids=["max", "min", "two"],
)
def test_the_orbax_logger_counts_its_best_checkpoints_apart_from_its_latest(
    tmp_path, options, best
):
    log_scores(tmp_path, [1.0, 3.0, 2.0], **options)

    assert kept(tmp_path / "latest") == ["1", "2", "3"]
    assert kept(tmp_path / "best") == best


def test_the_orbax_logger_saves_only_checkpoints(tmp_path):
    logger = OrbaxLogger(directory=str(tmp_path))
    logger.log_artifact(Video("episode", np.zeros((3, 8, 8), np.uint8)), step=1)
    logger.finish()

    assert kept(tmp_path / "latest") == []
    assert kept(tmp_path / "best") == []


def test_the_file_logger_gives_every_metric_one_row_per_logged_step(tmp_path):
    logger = FileLogger(directory=tmp_path)
    logger.log({"monitor/SPS": np.array([[0.0]])}, steps=np.array([0, 0]))
    logger.log(
        {"monitor/SPS": np.array([[5.0]]), "training/episode_return": np.array([[1.0, 3.0]])},
        steps=np.array([0, 100]),
    )
    logger.log({"training/episode_return": np.array([[4.0]])}, steps=np.array([100, 200]))
    logger.finish()

    metrics = np.load(tmp_path / "metrics.npz")
    np.testing.assert_array_equal(metrics["steps"], [0, 100, 200])
    np.testing.assert_array_equal(metrics["monitor/SPS"].ravel(), [0.0, 5.0, np.nan])
    np.testing.assert_array_equal(metrics["training/episode_return"].ravel(), [np.nan, 2.0, 4.0])


def test_the_file_logger_writes_each_video_as_a_gif_and_nothing_else(tmp_path):
    logger = FileLogger(directory=tmp_path)
    logger.log_artifact(Video("episode", np.zeros((3, 8, 8), np.uint8)), step=100)
    logger.log_artifact(Checkpoint("checkpoint", {}), step=100)

    assert [path.name for path in tmp_path.iterdir()] == ["episode-100.gif"]


RUN = """
import time
from boonta.loggers import DashboardLogger
logger = DashboardLogger()
print("ready", flush=True)
time.sleep(60)
"""

SHOW_CURSOR = b"\x1b[?25h"


@pytest.mark.parametrize(
    "received, status",
    [(signal.SIGTERM, 128 + signal.SIGTERM), (signal.SIGINT, -signal.SIGINT)],
    ids=["sigterm", "sigint"],
)
def test_a_signal_stops_a_run_using_the_dashboard_and_gives_the_cursor_back(received, status):
    process = subprocess.Popen(
        [sys.executable, "-c", RUN], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    while b"ready" not in process.stdout.readline():
        assert process.poll() is None, process.stderr.read().decode()
    time.sleep(0.2)

    process.send_signal(received)

    assert process.wait(timeout=10) == status
    assert SHOW_CURSOR in process.stdout.read()


@struct.dataclass
class CriticState:
    step: jnp.ndarray
    params: dict
    critic_params: dict
    target_critic_params: dict
    optimizer_state: dict


def variables(width):
    return {"params": {"Dense_0": {"kernel": jnp.zeros((width, width))}}}


def critic_state():
    return CriticState(
        step=jnp.array(0),
        params=variables(2),
        critic_params=variables(3),
        target_critic_params=variables(3),
        optimizer_state={"mu": jnp.zeros((100,))},
    )


@pytest.mark.parametrize(
    "wrap",
    [lambda state: state, lambda state: SimpleNamespace(algorithm_state=state)],
    ids=["algorithm_state", "podracer_state"],
)
def test_brief_counts_every_network(capsys, wrap):
    brief(OmegaConf.create({"seed": 0}), wrap(critic_state()))

    printed = capsys.readouterr().out.splitlines()
    headers = [line.split("┃")[1].strip() for line in printed if "┃" in line]
    totals = [line.split("│")[3].strip() for line in printed if "│ total" in line]
    assert headers == ["module", "critic_params", "target_critic_params", "setting"]
    assert totals == ["4", "9", "9"]


def test_brief_lists_the_run_settings_but_not_hydra_loggers_or_artisans(capsys):
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "algorithm": {"gamma": 0.99, "lr": 0.0003},
            "hydra": {"job_name": "launch"},
            "loggers": {"wandb": {"project": "elsewhere"}},
            "artisans": {"checkpointer": {"interval": 7777}},
        }
    )

    brief(cfg, critic_state())

    printed = capsys.readouterr().out
    for shown in ("seed", "algorithm", "gamma", "0.99", "lr"):
        assert shown in printed
    for hidden in ("job_name", "launch", "project", "elsewhere", "interval", "7777"):
        assert hidden not in printed


class Layout:
    def __init__(self, array):
        self.array = array

    def __hash__(self):
        return hash(self.array.tobytes())

    def __eq__(self, other):
        return np.array_equal(self.array, other.array)


@dataclasses.dataclass
class Packed:
    values: jax.Array
    layout: np.ndarray


jax.tree_util.register_pytree_with_keys(
    Packed,
    lambda node: (
        ((jax.tree_util.GetAttrKey("values"), node.values),),
        Layout(node.layout) if isinstance(node.layout, np.ndarray) else node.layout,
    ),
    lambda layout, values: Packed(values=values[0], layout=layout.array),
)


def test_sharded_keeps_fields_that_are_not_pytree_children():
    node = Packed(values=jnp.zeros((4, 2)), layout=np.arange(3))

    axes = sharded(node, "data")
    roundtrip = jax.tree.map(lambda axis: axis, axes)

    assert roundtrip.values == "data"
    np.testing.assert_array_equal(roundtrip.layout, np.arange(3))


@struct.dataclass
class Weights:
    kernel: jax.Array


@struct.dataclass
class Seat:
    carry: jax.Array = struct.field(metadata={"axis": "data"})
    opponents: jax.Array = struct.field(metadata={"axis": None})
    weights: Weights
    step: jax.Array


@pytest.mark.parametrize("root", [False, "data"], ids=["replicated", "sharded"])
def test_sharded_follows_a_fields_axis_and_inherits_its_parents_otherwise(root):
    seat = Seat(
        carry=jnp.zeros((4, 2)),
        opponents=jnp.zeros((3, 2)),
        weights=Weights(kernel=jnp.zeros((2, 2))),
        step=jnp.array(0),
    )

    axes = sharded(seat, root)

    assert axes.carry == "data"
    assert axes.opponents is None
    assert axes.weights.kernel == root
    assert axes.step is False
