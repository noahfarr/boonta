import jax
import numpy as np

from .artifact import Video


class Render:
    def __init__(
        self,
        name: str = "render",
        seed: int = 0,
        max_frames: int = 256,
        fps: int = 16,
    ):
        self.name = name
        self.seed = seed
        self.max_frames = max_frames
        self.fps = fps

    def craft(self, algorithm, environment, state, logs) -> Video:
        states = logs["env_state"]
        example, *_ = jax.tree.leaves(states)
        steps = min(len(example), self.max_frames)
        frames = [
            np.asarray(
                environment.render(jax.tree.map(lambda leaf: leaf[step], states)),
                dtype=np.uint8,
            )
            for step in range(steps)
        ]
        return Video(self.name, np.stack(frames), fps=self.fps)
