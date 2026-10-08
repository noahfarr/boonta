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


def launch(directory: Path, devices: int, overrides: list[str], output: Path):
    environment = os.environ | {
        "JAX_PLATFORMS": "cpu",
        "JAX_CPU_COLLECTIVES_IMPLEMENTATION": "gloo",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
    }
    command = [sys.executable, str(ROOT / "hangar/main.py"), *RUN, f"hydra.run.dir={directory}"]
    with output.open("w") as log:
        return subprocess.Popen([*command, *overrides], env=environment, stdout=log, stderr=log)


def test_a_run_across_two_processes_equals_the_same_run_in_one(tmp_path):
    cluster = [
        "+cluster._target_=jax.distributed.initialize",
        f"+cluster.coordinator_address=localhost:{free_port()}",
        "+cluster.num_processes=2",
        "logger=[file,orbax]",
        "+artisan=[checkpointer]",
    ]
    outputs = [tmp_path / f"process{index}.log" for index in range(2)] + [tmp_path / "single.log"]
    runs = [
        launch(tmp_path / "cluster", 2, [*cluster, f"+cluster.process_id={index}"], output)
        for index, output in zip(range(2), outputs)
    ] + [launch(tmp_path / "single", 4, ["logger=file"], outputs[-1])]

    for run, output in zip(runs, outputs):
        assert run.wait(timeout=900) == 0, output.read_text()[-4000:]

    together = np.load(tmp_path / "cluster/metrics.npz")
    alone = np.load(tmp_path / "single/metrics.npz")
    learning = [key for key in alone.files if key.startswith(LEARNING)]
    assert learning
    for key in learning:
        np.testing.assert_array_equal(together[key], alone[key], err_msg=key)

    latest = {path.name for path in (tmp_path / "cluster/checkpoints/latest").iterdir()}
    assert "40960" in latest
    assert any((tmp_path / "cluster/checkpoints/best").iterdir())

    first, second, _ = (output.read_text() for output in outputs)
    assert "│" in first and "│" not in second
