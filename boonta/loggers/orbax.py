import operator
import os

import orbax.checkpoint as ocp

from boonta.artisans import Checkpoint
from boonta.utils import PyTree


def barriers(name: str) -> ocp.options.MultiprocessingOptions:
    return ocp.options.MultiprocessingOptions(barrier_sync_key_prefix=name)


class OrbaxLogger:
    def __init__(
        self,
        directory: str,
        max_to_keep: int = 1,
        best_to_keep: int = 1,
        best_mode: str = "max",
        interval: int = 1,
        best: bool = True,
        **kwargs,
    ):
        self.latest = ocp.CheckpointManager(
            os.path.join(directory, "latest"),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=max_to_keep,
                save_interval_steps=interval,
                multiprocessing_options=barriers("latest"),
            ),
        )
        self.best = None
        if best:
            self.best = ocp.CheckpointManager(
                os.path.join(directory, "best"),
                options=ocp.CheckpointManagerOptions(
                    max_to_keep=best_to_keep,
                    save_interval_steps=interval,
                    best_fn=operator.itemgetter("score"),
                    best_mode=best_mode,
                    multiprocessing_options=barriers("best"),
                ),
            )

    def log(self, data: PyTree, steps: PyTree, **kwargs) -> None:
        pass

    def log_summary(self, data: PyTree, **kwargs) -> None:
        pass

    def bundle(self, artifact: Checkpoint):
        return ocp.args.Composite(
            **{
                name: ocp.args.StandardSave(tree)
                for name, tree in artifact.data.items()
            }
        )

    def log_artifact(self, artifact, step: int, **kwargs) -> None:
        if not isinstance(artifact, Checkpoint):
            return
        self.latest.save(step, args=self.bundle(artifact))
        if self.best is not None:
            self.best.save(
                step, args=self.bundle(artifact), metrics={"score": artifact.score}
            )

    def finish(self) -> None:
        self.latest.wait_until_finished()
        if self.best is not None:
            self.best.wait_until_finished()
