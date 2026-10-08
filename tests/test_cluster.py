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
    "total_timesteps=40960",
    "training.num_epochs=2",
    "evaluation.num_steps=16",
]
LEARNING = ("actor/", "critic/", "training/", "evaluation/")


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
    command = [sys.executable, str(ROOT / "hangar/main.py"), *RUN, *overrides]
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


def test_a_run_across_two_processes_equals_the_same_run_in_one(tmp_path):
    port = free_port()
    outputs = [tmp_path / f"process{index}.log" for index in range(2)] + [tmp_path / "single.log"]
    runs = [
        launch(tmp_path / "cluster", 2, ["logger=[file,orbax]", "+artisan=[checkpointer]"], output, **task(index, port))
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
        np.testing.assert_array_equal(together[key], alone[key], err_msg=key)

    latest = {path.name for path in (shared / "checkpoints/latest").iterdir()}
    assert "40960" in latest
    assert any((shared / "checkpoints/best").iterdir())

    first, second, _ = (output.read_text() for output in outputs)
    assert "│" in first and "│" not in second
