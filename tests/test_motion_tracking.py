import importlib.util
import shutil
from pathlib import Path

import jax
import numpy as np
import pytest

LIBRARY = Path("~/.cache/boonta/lafan1_g1/g1/library_50hz.npz").expanduser()


def missing(package):
    return importlib.util.find_spec(package) is None


simulated = pytest.mark.skipif(
    missing("mjlab") or shutil.which("nvidia-smi") is None or not LIBRARY.exists(),
    reason="needs mjlab, an NVIDIA GPU and the LAFAN1 library from hangar/scripts/convert_motions.py",
)


@pytest.fixture(scope="module")
def tracking():
    from boonta import environments
    from hangar.recipes.super_mimic_mjlab import TASK, describe, register

    register()
    kwargs = {"num_envs": 16, "motion_file": str(LIBRARY), "executed": "executed"}
    environment = environments.make("mjlab", TASK, kwargs=kwargs)
    return environment, describe(environment)


@pytest.mark.skipif(missing("scipy"), reason="needs scipy")
def test_the_motion_library_interpolates_rotations_like_scipy():
    from scipy.spatial.transform import Rotation, Slerp

    from hangar.recipes.super_mimic_mjlab import _resample

    count, source_rate = 40, 30
    rotations = Rotation.random(count, random_state=0)
    frames = np.concatenate([np.zeros((count, 3)), rotations.as_quat(), np.zeros((count, 29))], axis=-1)
    qpos, _ = _resample(frames, source_rate, 50)
    steps, _ = qpos.shape
    times = np.arange(steps) / 50
    expected = Slerp(np.arange(count) / source_rate, rotations)(times).as_quat()[:, [3, 0, 1, 2]]
    expected = np.where(np.sum(expected * qpos[:, 3:7], axis=-1, keepdims=True) < 0, -expected, expected)
    np.testing.assert_allclose(qpos[:, 3:7], expected, atol=1e-4)


@simulated
def test_the_jax_task_observes_what_mjlab_observes(tracking):
    environment, task = tracking
    state, timestep = environment.init(jax.random.key(0))
    for step in range(5):
        action = 0.3 * jax.random.normal(jax.random.key(step), timestep.action.shape)
        state, timestep = environment.step(jax.random.key(step), state, action)
        observed = jax.vmap(task.observe)(timestep.obs["state"], timestep.obs["context"])
        np.testing.assert_allclose(observed, timestep.obs["observation"], atol=1e-4)


@simulated
def test_mjlab_reports_the_executed_action_and_shows_the_policy_the_clean_one(tracking):
    environment, _ = tracking
    assert "executed" not in environment.observation_space()
    state, timestep = environment.init(jax.random.key(0))
    action = 0.3 * jax.random.normal(jax.random.key(1), timestep.action.shape)
    state, timestep = environment.step(jax.random.key(2), state, action)
    alive = ~np.asarray(timestep.done)
    _, width = action.shape
    noise = np.asarray(timestep.action - action)[alive]
    assert 0.05 < noise.std() < 0.2
    previous = np.asarray(timestep.obs["observation"][:, -width:])
    np.testing.assert_allclose(previous[alive], np.asarray(action)[alive], atol=1e-6)
