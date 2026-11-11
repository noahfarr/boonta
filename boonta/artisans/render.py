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
        environment_state, timestep = jax.jit(environment.init)(init_key)
        carry = jax.jit(lambda key, timestep: algorithm.init(key, timestep)[1])(init_key, timestep)
        act = jax.jit(partial(algorithm.step, temperature=0.0))
        advance = jax.jit(environment.step)
        frames = []
        for step_key in jax.random.split(key, environment.time_limit() + 1):
            algorithm_key, environment_key = jax.random.split(step_key)
            algorithm_state, carry, action, _ = act(algorithm_state, carry, algorithm_key, timestep)
            environment_state, timestep = advance(environment_key, environment_state, action)
            frame_state = jax.device_get(jax.tree.map(lambda leaf: leaf[0] if jnp.ndim(leaf) else leaf, environment_state))
            frames.append(np.asarray(environment.render(frame_state, **self.kwargs), dtype=np.uint8))
            if bool(jax.device_get(jnp.all(timestep.done[0]))):
                break
        environment.close(environment_state)
        return np.stack(frames)

    def craft(self, key, algorithm, environment, state, logs) -> Video:
        frames = self.rollout(algorithm, environment, state.algorithm_state, key)
        return Video(self.name, frames, fps=self.fps)
