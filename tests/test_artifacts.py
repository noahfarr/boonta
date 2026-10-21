import dataclasses
import signal
import subprocess
import sys
import time
from functools import partial
from types import SimpleNamespace

from pathlib import Path

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
from boonta.artisans import Checkpoint, Render, Transcript, Video
from boonta.utils import load_checkpoint, resolve_checkpoint, sharded
from dummies import corridor
from hangar.brief import brief, debrief, evaluation_command

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
    logger.log_artifact(Checkpointer(**options).craft(None, None, None, state, logs), step=96)
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

    restored = load_checkpoint(resolve_checkpoint(tmp_path), fresh)

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

    restored = load_checkpoint(resolve_checkpoint(tmp_path), fresh)

    items = {item.name for item in (tmp_path / "checkpoints/latest/96").iterdir()}
    assert items == {*fields, "_CHECKPOINT_METADATA"}
    for field in ("algorithm_state", "environment_state", "timestep"):
        assert moved(getattr(fresh, field), getattr(trained, field))
        expected = trained if field in fields else fresh
        assert_restored(getattr(expected, field), getattr(restored, field))


def test_resume_fails_when_a_saved_parameter_changed_shape(tmp_path):
    podracer = build_podracer(zoo.recurrent_ppo)
    save(tmp_path, podracer.init(jax.random.key(0)))
    fresh = podracer.init(jax.random.key(0))
    wrong = jax.tree.map(lambda leaf: jnp.zeros((leaf.shape[0] + 1, *leaf.shape[1:]), leaf.dtype) if leaf.ndim else leaf, fresh.algorithm_state.params)

    with pytest.raises(ValueError, match="stored shape"):
        load_checkpoint(resolve_checkpoint(tmp_path), fresh.replace(algorithm_state=fresh.algorithm_state.replace(params=wrong)))


def test_a_recurrent_checkpoint_holds_no_carry_and_loads_into_a_different_env_count(tmp_path):
    saved = build_podracer(zoo.recurrent_ppo)
    wider = build_podracer(zoo.recurrent_ppo, num_envs=2 * NUM_ENVS)
    trained = train(saved)
    save(tmp_path, trained)

    stored = {item.name for item in (tmp_path / "checkpoints/latest/96/algorithm_state").iterdir()}
    loaded = load_checkpoint(resolve_checkpoint(tmp_path), wider.init(jax.random.key(0)))

    assert not any("carry" in name for name in stored)
    assert_restored(trained.algorithm_state.params, loaded.algorithm_state.params)
    assert int(loaded.algorithm_state.step) == int(trained.algorithm_state.step)
    assert loaded.carry.shape[0] == 2 * NUM_ENVS


def test_a_checkpoint_saved_with_the_carry_inside_the_algorithm_state_still_loads(tmp_path):
    import orbax.checkpoint as ocp

    podracer = build_podracer(zoo.recurrent_ppo)
    trained = train(podracer)
    legacy = {
        **{field: getattr(trained.algorithm_state, field) for field in ("step", "params", "optimizer_state")},
        "carry": trained.carry,
        "rollout_carry": trained.carry,
    }
    directory = tmp_path / "checkpoints/latest/96"
    directory.mkdir(parents=True)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save((directory / "algorithm_state").resolve(), legacy)
    checkpointer.wait_until_finished()

    loaded = load_checkpoint(resolve_checkpoint(tmp_path), podracer.init(jax.random.key(0)))

    assert_restored(trained.algorithm_state.params, loaded.algorithm_state.params)


def test_a_host_and_path_is_recognised_as_a_remote_checkpoint_and_a_local_path_is_not(tmp_path):
    from boonta.utils.checkpoint import remote_parts

    assert remote_parts("ias:/path/to/checkpoint") == ("ias", "/path/to/checkpoint")
    assert remote_parts("me@host:~/runs/a") == ("me@host", "~/runs/a")
    assert remote_parts(str(tmp_path)) is None
    assert remote_parts("C:\\runs") is None
    assert remote_parts("https://example.org/a") is None
    assert remote_parts("a/b:c") is None


def test_a_remote_run_directory_is_pulled_once_to_the_cache_at_its_latest_checkpoint(tmp_path, monkeypatch):
    import shutil
    import subprocess

    remote = tmp_path / "remote" / "run"
    for step in (1, 12, 3):
        (remote / "checkpoints/latest" / str(step) / "algorithm_state").mkdir(parents=True)
        (remote / "checkpoints/latest" / str(step) / "algorithm_state/data").write_text(str(step))
    calls = []

    def fake(command, **kwargs):
        calls.append(command[0])
        if command[0] == "ssh":
            return subprocess.CompletedProcess(command, 0, stdout=str(remote / "checkpoints/latest/12") + "\n", stderr="")
        shutil.copytree(remote / "checkpoints/latest/12", command[-1], dirs_exist_ok=True)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("BOONTA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(subprocess, "run", fake)

    first = resolve_checkpoint(f"cluster:{remote}")
    second = resolve_checkpoint(f"cluster:{remote}")

    assert first == second
    assert first.endswith("checkpoints/latest/12")
    assert (Path(first) / "algorithm_state/data").read_text() == "12"
    assert str(tmp_path / "cache") in first
    assert len([call for call in calls if call != "ssh"]) == 1


def test_resolve_checkpoint_finds_the_latest_checkpoint_of_a_run_directory(tmp_path):
    log_scores(tmp_path / "checkpoints", [1.0, 3.0, 2.0])

    assert resolve_checkpoint(tmp_path) == str(tmp_path / "checkpoints/latest/3")
    assert resolve_checkpoint(tmp_path / "checkpoints/best/2") == str(tmp_path / "checkpoints/best/2")
    assert resolve_checkpoint(None) is None


def test_the_checkpointer_saves_exactly_the_fields_it_names():
    state = SimpleNamespace(algorithm_state="learned", environment_state="played")

    default = Checkpointer().craft(None, None, None, state, {})
    both = Checkpointer(fields=["algorithm_state", "environment_state"]).craft(
        None, None, None, state, {}
    )

    assert default.data == {"algorithm_state": "learned"}
    assert both.data == {"algorithm_state": "learned", "environment_state": "played"}


def test_the_checkpointer_scores_by_mean_episode_return_unless_told_otherwise():
    state = SimpleNamespace(algorithm_state="learned")
    logs = {"episode_statistics/episode_return": np.array([1.0, np.nan, 5.0]), "bonus": [7.0]}

    assert Checkpointer().craft(None, None, None, state, logs).score == 3.0
    assert Checkpointer(score=lambda logs: logs["bonus"]).craft(None, None, None, state, logs).score == 7.0
    assert Checkpointer().craft(None, None, None, state, {}).score == float("-inf")


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


def test_the_render_artisan_plays_and_draws_the_first_environment():
    from boonta.environments import gymnax

    environment = zoo.wrap(gymnax.make("Breakout-MinAtar"), NUM_ENVS)
    space = environment.action_space()
    num_actions = space.num_actions

    def step(state, carry, key, timestep, temperature=1.0):
        action = jax.random.randint(key, (NUM_ENVS,), 0, num_actions, space.dtype)
        return state, carry, action, {}

    algorithm = SimpleNamespace(step=step, init=lambda key, timestep: (None, None))
    state = SimpleNamespace(algorithm_state=None)
    video = Render(max_frames=16).craft(jax.random.key(0), algorithm, environment, state, {})

    frames, _, _, channels = video.data.shape
    assert (frames, channels) == (16, 3)
    assert len(np.unique(video.data.reshape(16, -1), axis=0)) > 1


def test_the_transcript_artisan_plays_its_own_episode_and_decodes_each_environment(
    monkeypatch,
):
    class Tokenizer:
        def decode(self, tokens):
            return " ".join(map(str, tokens))

    monkeypatch.setattr("boonta.artisans.transcript.load_tokenizer", lambda repo_id: Tokenizer())
    environment = zoo.wrap(corridor(), NUM_ENVS)
    environment.time_limit = lambda: 3

    def step(state, carry, key, timestep, temperature=1.0):
        return state, carry, jnp.arange(NUM_ENVS, dtype=jnp.int32) % 2, {}

    algorithm = SimpleNamespace(step=step, init=lambda key, timestep: (None, None))
    state = SimpleNamespace(algorithm_state=None)
    text = Transcript("repo", num_examples=3).craft(
        jax.random.key(0), algorithm, environment, state, {}
    )

    assert text.data == "0: 0 0 0 0\n\n1: 1 1 1 1\n\n2: 0 0 0 0"


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
            "artisans": {"checkpoint": {"interval": 7777}},
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


TRAINED_WITH = [
    "algorithm=ippo",
    "environment=dune_sea/oasis",
    "environment.kwargs.num_agents=8",
    "logger=[dashboard,orbax]",
    "artisan=checkpoint",
    "seed=3",
    "num_steps=1_000",
    "hydra/launcher=submitit",
]


def test_the_evaluation_command_keeps_what_shapes_the_model_and_drops_what_only_trains():
    command = evaluation_command("/runs/a", TRAINED_WITH)

    assert command == (
        "uv run boonta evaluate checkpoint=/runs/a algorithm=ippo "
        "environment=dune_sea/oasis environment.kwargs.num_agents=8"
    )


def test_the_evaluation_command_quotes_what_the_shell_would_expand():
    command = evaluation_command("/runs/with space", ["network.sizes=[1,2]"])

    assert command == "uv run boonta evaluate 'checkpoint=/runs/with space' 'network.sizes=[1,2]'"


def test_debrief_prints_the_evaluation_command_of_a_run_that_saved_a_checkpoint(capsys, tmp_path):
    (tmp_path / "checkpoints").mkdir()

    debrief(OmegaConf.create({"score": "training/return"}), tmp_path, TRAINED_WITH, 20_000_000, 3.5, 4000.0)

    printed = capsys.readouterr().out
    for shown in ("20,000,000", "training/return", "3.5", "1h 06m 40s", "saved"):
        assert shown in printed
    assert f"uv run boonta evaluate checkpoint={tmp_path} algorithm=ippo" in printed


def test_debrief_says_so_when_no_checkpoint_was_saved(capsys, tmp_path):
    debrief(OmegaConf.create({"score": "training/return"}), tmp_path, TRAINED_WITH, 1_000, float("-inf"), 30.0)

    printed = capsys.readouterr().out
    assert "not saved" in printed and "n/a" in printed
    assert "boonta evaluate" not in printed
    assert "logger=[dashboard,orbax] artisan=checkpoint" in printed


def test_the_wandb_logger_keeps_a_host_copy_of_the_checkpoint_it_will_upload(monkeypatch):
    import sys
    import types

    import jax.numpy as jnp

    monkeypatch.setitem(sys.modules, "wandb", types.ModuleType("wandb"))
    monkeypatch.delitem(sys.modules, "boonta.loggers.wandb", raising=False)

    from boonta.artisans import Checkpoint
    from boonta.loggers.wandb import WandbLogger

    logger = object.__new__(WandbLogger)
    logger.runs, logger.checkpoint = {}, None
    weights = jnp.arange(4.0)

    logger.log_artifact(Checkpoint("model", {"weights": weights}, 1.0), step=1)
    weights.delete()

    assert logger.checkpoint.data["weights"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_an_artisan_seed_can_be_built_from_a_step_count_beyond_32_bits():
    import jax

    key = jax.random.key(0)
    step = 4_303_355_904

    folded = jax.random.fold_in(key, step & 0xFFFFFFFF)

    assert folded.shape == key.shape

