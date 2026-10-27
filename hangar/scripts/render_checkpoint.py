import os
import shutil
import subprocess
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from hangar import resolvers  # noqa: F401


def load(run: Path, overrides: list[str]):
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    hydra = OmegaConf.structured(HydraConf)
    OmegaConf.set_struct(hydra, False)
    hydra = OmegaConf.merge(hydra, OmegaConf.load(run / ".hydra" / "hydra.yaml").hydra)
    HydraConfig.instance().set_config(OmegaConf.create({"hydra": hydra}))
    return OmegaConf.merge(
        cfg, OmegaConf.from_dotlist(["environment.num_envs=1", *overrides])
    )


def main():
    run, *overrides = sys.argv[1:]
    run = Path(run).resolve()
    cfg = load(run, overrides)

    from boonta.artisans import Video
    from boonta.utils import load_checkpoint, resolve_checkpoint
    from hangar import recipes

    podracer = recipes.make(cfg)
    algorithm, environment = podracer.algorithm, podracer.environment
    init_key, key = jax.random.split(jax.random.key(cfg.seed))
    path = Path(resolve_checkpoint(run))
    algorithm_state = load_checkpoint(
        path / "algorithm_state", podracer.init(init_key)
    ).algorithm_state

    act = jax.jit(partial(algorithm.step, temperature=1.0))
    step = jax.jit(environment.step)
    environment_state, timestep = environment.init(key)
    try:
        horizon = environment.time_limit()
    except NotImplementedError:
        horizon = None
    frames = []
    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        episode = progress.add_task(cfg.environment.env_id, total=horizon)
        while True:
            env_state = jax.tree.map(
                lambda leaf: jnp.take(leaf, 0, axis=0), environment_state.unwrapped
            )
            frames.append(
                np.asarray(environment.unwrapped.render(env_state), dtype=np.uint8)
            )
            progress.advance(episode)
            key, algorithm_key, environment_key = jax.random.split(key, 3)
            algorithm_state, action, _ = act(algorithm_state, algorithm_key, timestep)
            environment_state, timestep = step(
                environment_key, environment_state, action
            )
            if timestep.done.any():
                progress.update(episode, total=len(frames))
                break

    directory = Path("media") / cfg.environment.env_id
    directory.mkdir(parents=True, exist_ok=True)
    height, width, *_ = frames[-1].shape
    frames = [
        np.asarray(Image.fromarray(frame).resize((width, height)))
        for frame in frames
    ]
    path = Video("episode", np.stack(frames), fps=30).encode(
        directory / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.{Video.suffix()}"
    )
    if shutil.which("xdg-open") and os.environ.get("DISPLAY"):
        subprocess.Popen(["xdg-open", path])


if __name__ == "__main__":
    main()
