from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .artifact import Video


class Render:
    def __init__(
        self,
        name: str = "render",
        fps: int = 16,
        kwargs: dict | None = None,
    ):
        self.name = name
        self.fps = fps
        self.kwargs = kwargs or {}

    def rollout(self, algorithm, environment, algorithm_state, key):
        init_key, key = jax.random.split(key)
        environment_state, timestep = environment.init(init_key)
        _, carry = algorithm.init(init_key, timestep)
        num_steps = environment.time_limit() + 1

        def step(loop, key):
            algorithm_state, carry, environment_state, timestep = loop
            algorithm_key, environment_key = jax.random.split(key)
            algorithm_state, carry, action, _ = algorithm.step(
                algorithm_state, carry, algorithm_key, timestep, temperature=0.0
            )
            environment_state, timestep = environment.step(
                environment_key, environment_state, action
            )
            return (algorithm_state, carry, environment_state, timestep), (
                jax.tree.map(lambda leaf: jnp.take(leaf, 0, axis=0), environment_state),
                jnp.all(jnp.take(timestep.done, 0, axis=0)),
            )

        (_, _, environment_state, _), (trajectory, done) = jax.lax.scan(
            step,
            (algorithm_state, carry, environment_state, timestep),
            jax.random.split(key, num_steps),
        )
        environment.close(environment_state)
        return trajectory, done

    def craft(self, key, algorithm, environment, state, logs) -> Video:
        trajectory, done = jax.device_get(
            jax.jit(partial(self.rollout, algorithm, environment))(
                state.algorithm_state, key
            )
        )
        num_frames = int(np.argmax(done)) + 1 if done.any() else len(done)
        frames = [
            np.asarray(
                environment.render(
                    jax.tree.map(lambda leaf: leaf[step], trajectory), **self.kwargs
                ),
                dtype=np.uint8,
            )
            for step in range(num_frames)
        ]
        return Video(self.name, np.stack(frames), fps=self.fps)
