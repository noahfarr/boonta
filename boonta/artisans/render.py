from functools import partial

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
        kwargs: dict | None = None,
    ):
        self.name = name
        self.key = jax.random.key(seed)
        self.max_frames = max_frames
        self.fps = fps
        self.kwargs = kwargs or {}
        self.epoch = 0

    def play(self, algorithm, environment, algorithm_state, key):
        init_key, key = jax.random.split(key)
        environment_state, timestep = environment.init(init_key)

        def step(carry, key):
            algorithm_state, environment_state, timestep = carry
            algorithm_key, environment_key = jax.random.split(key)
            algorithm_state, action, _ = algorithm.step(
                algorithm_state, algorithm_key, timestep, temperature=0.0
            )
            environment_state, timestep = environment.step(
                environment_key, environment_state, action
            )
            game = getattr(environment_state, "unwrapped", environment_state)
            first = jax.tree.map(lambda leaf: leaf[0], game)
            return (algorithm_state, environment_state, timestep), first

        (_, environment_state, _), games = jax.lax.scan(
            step,
            (algorithm_state, environment_state, timestep),
            jax.random.split(key, self.max_frames),
        )
        environment.close(environment_state)
        return games

    def craft(self, algorithm, environment, state, logs) -> Video:
        key = jax.random.fold_in(self.key, self.epoch)
        self.epoch += 1
        play = jax.jit(partial(self.play, algorithm, environment))
        games = play(state.algorithm_state, key)
        frames = [
            np.asarray(
                environment.render(
                    jax.tree.map(lambda leaf: leaf[step], games), **self.kwargs
                ),
                dtype=np.uint8,
            )
            for step in range(self.max_frames)
        ]
        return Video(self.name, np.stack(frames), fps=self.fps)
