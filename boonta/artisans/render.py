from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .artifact import Video


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

    def rollout(self, algorithm, environment, algorithm_state, key):
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
            env_state = jax.tree.map(
                lambda leaf: jnp.take(leaf, 0, axis=0), environment_state
            )
            return (algorithm_state, environment_state, timestep), env_state

        (_, environment_state, _), trajectory = jax.lax.scan(
            step,
            (algorithm_state, environment_state, timestep),
            jax.random.split(key, self.max_frames),
        )
        environment.close(environment_state)
        return trajectory

    def craft(self, key, algorithm, environment, state, logs) -> Video:
        trajectory = jax.jit(partial(self.rollout, algorithm, environment))(
            state.algorithm_state, key
        )
        frames = [
            np.asarray(
                environment.render(
                    jax.tree.map(lambda leaf: jnp.take(leaf, step, axis=0), trajectory),
                    **self.kwargs,
                ),
                dtype=np.uint8,
            )
            for step in range(self.max_frames)
        ]
        return Video(self.name, np.stack(frames), fps=self.fps)
