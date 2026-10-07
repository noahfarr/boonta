import jax
import numpy as np

from .artifact import Video


class Render:
    def __init__(
        self,
        name: str = "render",
        seed: int = 0,
        env: int = 0,
        max_frames: int = 256,
        fps: int = 16,
    ):
        self.name = name
        self.seed = seed
        self.env = env
        self.max_frames = max_frames
        self.fps = fps

    def craft(self, algorithm, environment, state, logs) -> Video:
        num_envs = environment.num_envs
        example, *_ = jax.tree.leaves(logs["env_state"])
        entries, *_ = example.shape
        steps = min(entries // num_envs, self.max_frames)

        def trajectory(leaf):
            leaf = leaf[: steps * num_envs, 0]
            return leaf.reshape(steps, num_envs, *leaf.shape[1:])[:, self.env]

        states = jax.tree.map(trajectory, logs["env_state"])
        frames = [
            np.asarray(
                environment.render(jax.tree.map(lambda leaf: leaf[step], states)),
                dtype=np.uint8,
            )
            for step in range(steps)
        ]
        return Video(self.name, np.stack(frames), fps=self.fps)
