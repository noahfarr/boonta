from functools import partial

import jax
import numpy as np

from .artifact import Video
from .rollout import rollout


class Render:
    def __init__(
        self,
        name: str = "render",
        max_frames: int = 256,
        fps: int = 16,
        kwargs: dict | None = None,
    ):
        self.name = name
        self.max_frames = max_frames
        self.fps = fps
        self.kwargs = kwargs or {}

    def craft(self, key, algorithm, environment, state, logs) -> Video:
        play = jax.jit(
            partial(
                rollout,
                algorithm,
                environment,
                num_steps=self.max_frames,
                record=lambda environment_state, action: jax.tree.map(
                    lambda leaf: leaf[0], environment_state
                ),
            )
        )
        trajectory = play(state.algorithm_state, key)
        frames = [
            np.asarray(
                environment.render(
                    jax.tree.map(lambda leaf: leaf[step], trajectory), **self.kwargs
                ),
                dtype=np.uint8,
            )
            for step in range(self.max_frames)
        ]
        return Video(self.name, np.stack(frames), fps=self.fps)
