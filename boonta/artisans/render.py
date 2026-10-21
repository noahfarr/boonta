import os
from concurrent.futures import ThreadPoolExecutor
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
        smooth: int = 1,
        kwargs: dict | None = None,
    ):
        self.name = name
        self.fps = fps
        self.smooth = smooth
        self.kwargs = kwargs or {}

    def rollout(self, algorithm, environment, algorithm_state, key):
        init_key, key = jax.random.split(key)
        environment_state, timestep = environment.init(init_key)
        _, carry = algorithm.init(init_key, timestep)
        steps = environment.time_limit() + 1

        def first(state):
            return jax.tree.map(lambda leaf: jnp.take(leaf, 0, axis=0), state)

        trajectory = jax.tree.map(
            lambda leaf: jnp.zeros((steps, *leaf.shape), leaf.dtype), first(environment_state)
        )

        def running(loop):
            *_, count, finished = loop
            return (count < steps) & ~finished

        def advance(loop):
            algorithm_state, carry, environment_state, timestep, key, trajectory, count, _ = loop
            key, algorithm_key, environment_key = jax.random.split(key, 3)
            algorithm_state, carry, action, _ = algorithm.step(
                algorithm_state, carry, algorithm_key, timestep, temperature=0.0
            )
            environment_state, timestep = environment.step(
                environment_key, environment_state, action
            )
            trajectory = jax.tree.map(
                lambda frames, leaf: frames.at[count].set(leaf),
                trajectory,
                first(environment_state),
            )
            finished = jnp.all(timestep.done[0])
            return (
                algorithm_state, carry, environment_state, timestep, key, trajectory, count + 1, finished,
            )

        _, _, environment_state, _, _, trajectory, count, _ = jax.lax.while_loop(
            running,
            advance,
            (
                algorithm_state,
                carry,
                environment_state,
                timestep,
                key,
                trajectory,
                jnp.int32(0),
                jnp.bool_(False),
            ),
        )
        environment.close(environment_state)
        return trajectory, count

    def craft(self, key, algorithm, environment, state, logs) -> Video:
        trajectory, count = jax.device_get(
            jax.jit(partial(self.rollout, algorithm, environment))(
                state.algorithm_state, key
            )
        )

        def state_at(step):
            return jax.tree.map(lambda leaf: leaf[step], trajectory)

        def draw(step, extra):
            return np.asarray(
                environment.render(state_at(step), **self.kwargs, **extra), dtype=np.uint8
            )

        def frames_of(step):
            if self.smooth > 1 and step > 0:
                previous = state_at(step - 1)
                return [
                    draw(step, {"previous": previous, "blend": sub / self.smooth})
                    for sub in range(1, self.smooth + 1)
                ]
            return [draw(step, {}) for _ in range(max(self.smooth, 1))]

        num_frames = int(count)
        if getattr(environment, "thread_safe_render", False):
            with ThreadPoolExecutor(min(4, os.cpu_count() or 1)) as pool:
                groups = list(pool.map(frames_of, range(num_frames)))
        else:
            groups = [frames_of(step) for step in range(num_frames)]
        frames = [frame for group in groups for frame in group]
        return Video(self.name, np.stack(frames), fps=self.fps * max(self.smooth, 1))
