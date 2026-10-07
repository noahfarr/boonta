import jax
import numpy as np
import wandb

from boonta.artisans import Checkpoint, Text, Video
from boonta.utils import PyTree


class WandbLogger:
    def __init__(
        self,
        entity=None,
        project=None,
        name=None,
        group=None,
        mode="disabled",
        directory=".",
        cfg=None,
        seed=0,
        num_seeds=1,
        **kwargs,
    ):
        cfg = cfg or {}
        self.runs = {
            i: wandb.init(
                entity=entity,
                project=project,
                name=name,
                group=group,
                mode=mode,
                dir=directory,
                config={
                    **cfg,
                    "seed": seed + i,
                },
                reinit="create_new",
                settings=wandb.Settings(quiet=True, silent=True),
            )
            for i in range(num_seeds)
        }
        self.checkpoint = None

    def log(self, data: PyTree, steps: PyTree, **kwargs) -> None:
        steps = np.asarray(jax.device_get(steps)).reshape(-1)
        start, end = int(steps.min()), int(steps.max())
        span = end - start
        data = jax.device_get(data)

        rows: list[dict[int, dict]] = [{} for _ in self.runs]
        for k, v in data.items():
            for seed, row in enumerate(v):
                row = np.asarray(row)
                length = len(row)
                if length == 0:
                    continue
                grid = start + (np.arange(length) + 1) * span // length
                finite = np.isfinite(row)
                for step, value in zip(grid[finite].tolist(), row[finite].tolist()):
                    rows[seed].setdefault(step, {})[k] = value

        for seed, run in self.runs.items():
            for step in sorted(rows[seed]):
                run.log(rows[seed][step], step=step)

    def log_summary(self, data: PyTree, **kwargs) -> None:
        data = jax.device_get(data)
        for seed, run in self.runs.items():
            for k, v in data.items():
                value = np.asarray(v)[seed]
                if np.isfinite(value):
                    run.summary[k] = float(value)

    def log_artifact(self, artifact, step: int, **kwargs) -> None:
        if isinstance(artifact, Checkpoint):
            self.checkpoint = artifact
        elif isinstance(artifact, Text):
            html = wandb.Html(f"<pre>{artifact.data}</pre>")
            for run in self.runs.values():
                run.log({artifact.name: html}, step=step)
        elif isinstance(artifact, Video):
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                path = artifact.encode(f"{directory}/{artifact.name}.gif")
                for run in self.runs.values():
                    run.log({artifact.name: wandb.Video(path)}, step=step)

    def finish(self) -> None:
        if self.checkpoint is not None:
            from flax import serialization

            data = jax.device_get(self.checkpoint.data)
            for run in self.runs.values():
                model = wandb.Artifact(f"model-{run.id}", type="model")
                with model.new_file("model.msgpack", mode="wb") as f:
                    f.write(serialization.to_bytes(data))
                run.log_artifact(model)
        for run in self.runs.values():
            run.finish()
