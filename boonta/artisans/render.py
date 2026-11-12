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
        num_episodes: int = 1,
    ):
        assert num_episodes >= 1, num_episodes
        self.name = name
        self.fps = fps
        self.kwargs = kwargs or {}
        # Episodes of the first environment, played back to back into one video.
        self.num_episodes = num_episodes

    def rollout(self, algorithm, environment, algorithm_state, key):
        init_key, key = jax.random.split(key)
        environment_state, timestep = environment.init(init_key)
        _, carry = algorithm.init(init_key, timestep)
        num_steps = self.num_episodes * (environment.time_limit() + 1)

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
                jax.tree.map(
                    lambda leaf: jnp.take(leaf, 0, axis=0),
                    getattr(environment_state, "unwrapped", environment_state),
                ),
                jnp.all(jnp.take(timestep.done, 0, axis=0)),
            )

        (_, _, environment_state, _), (trajectory, done) = jax.lax.scan(
            step,
            (algorithm_state, carry, environment_state, timestep),
            jax.random.split(key, num_steps),
        )
        environment.close(environment_state)
        return trajectory, done

    def craft(self, key, podracer, podracer_state, logs) -> Video:
        algorithm, environment = podracer.algorithm, podracer.environment
        try:
            from matplotlib import pyplot

            pyplot.switch_backend("Agg")
        except ImportError:
            pass
        trajectory, done = jax.device_get(
            jax.jit(partial(self.rollout, algorithm, environment))(
                podracer_state.algorithm_state, key
            )
        )
        ends = np.flatnonzero(done)
        num_frames = int(ends[self.num_episodes - 1]) + 1 if len(ends) >= self.num_episodes else len(done)
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
