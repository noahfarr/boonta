import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
RUN = [
    "algorithm=ppo",
    "environment=gymnax/minatar/breakout",
    "environment.num_envs=64",
    "num_steps=2048",
    "num_epochs=2",
]
LEARNING = ("actor/", "critic/", "training/")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        _, port = sock.getsockname()
        return port


def launch(directory: Path, devices: int, overrides: list[str], output: Path, **slurm):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")} | {
        "JAX_PLATFORMS": "cpu",
        "JAX_CPU_COLLECTIVES_IMPLEMENTATION": "gloo",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])),
        **slurm,
    }
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "hangar/train.py"), *RUN, *overrides]
    with output.open("w") as log:
        return subprocess.Popen(command, cwd=directory, env=environment, stdout=log, stderr=log)


def task(index: int, port: int) -> dict[str, str]:
    return {
        "SLURM_JOB_ID": "4242",
        "SLURM_STEP_NODELIST": "localhost",
        "SLURM_NTASKS": "2",
        "SLURM_STEP_NUM_TASKS": "2",
        "SLURM_PROCID": str(index),
        "SLURM_LOCALID": str(index),
        "JAX_COORDINATOR_ADDRESS": f"localhost:{port}",
    }


def evaluate(directory: Path, overrides: list[str]) -> np.ndarray:
    environment = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")} | {
        "JAX_PLATFORMS": "cpu",
        "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])),
    }
    command = [sys.executable, str(ROOT / "hangar/evaluate.py"), *RUN[:2], "~artisan", "logger=file", f"hydra.run.dir={directory}", *overrides]
    finished = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=600)
    assert finished.returncode == 0, finished.stderr[-4000:]
    with np.load(directory / "metrics.npz") as metrics:
        return metrics["evaluation/episode_return"]


def test_a_run_keeps_its_untrained_policy_which_evaluate_plays_without_a_checkpoint(tmp_path):
    output = tmp_path / "train.log"
    run = launch(tmp_path / "train", 1, ["logger=[file,orbax]", "hydra.run.dir=."], output)
    assert run.wait(timeout=600) == 0, output.read_text()[-4000:]
    checkpoints = tmp_path / "train" / "checkpoints" / "latest"
    assert "0" in {path.name for path in checkpoints.iterdir()}

    untrained = evaluate(tmp_path / "untrained", [])
    saved = evaluate(tmp_path / "saved", [f"checkpoint={checkpoints / '0'}"])
    np.testing.assert_array_equal(untrained, saved)


def test_a_run_across_two_processes_equals_the_same_run_in_one(tmp_path):
    port = free_port()
    outputs = [tmp_path / f"process{index}.log" for index in range(2)] + [tmp_path / "single.log"]
    runs = [
        launch(tmp_path / "cluster", 2, ["num_processes=2", "logger=[file,orbax]", "+artisan=[checkpoint]"], output, **task(index, port))
        for index, output in zip(range(2), outputs)
    ] + [launch(tmp_path / "single", 4, ["logger=file", "hydra.run.dir=."], outputs[-1])]

    for run, output in zip(runs, outputs):
        assert run.wait(timeout=900) == 0, output.read_text()[-4000:]

    shared = tmp_path / "cluster/outputs/4242-0"
    together = np.load(shared / "metrics.npz")
    alone = np.load(tmp_path / "single/metrics.npz")
    learning = [key for key in alone.files if key.startswith(LEARNING)]
    assert learning
    for key in learning:
        # Gradients are summed across processes in a different order than within
        # one process, so the runs agree up to rounding, which training amplifies.
        np.testing.assert_allclose(together[key], alone[key], rtol=1e-4, err_msg=key)

    latest = {path.name for path in (shared / "checkpoints/latest").iterdir()}
    assert "2048" in latest
    assert any((shared / "checkpoints/best").iterdir())

    first, second, _ = (output.read_text() for output in outputs)
    assert "│" in first and "│" not in second
