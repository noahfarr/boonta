import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .artifact import Video


def _first(state):
    return jax.tree.map(lambda leaf: jnp.take(leaf, 0, axis=0), state)


class Render:
    def __init__(
        self,
        name: str = "render",
        fps: int = 16,
        smooth: int = 1,
        kwargs: dict | None = None,
        num_episodes: int = 1,
        temperature: float = 0.0,
        chunk: int = 256,
        max_bytes: int = 2 * 1024**3,
    ):
        if chunk < 1 or max_bytes < 1 or num_episodes < 1:
            raise ValueError("chunk, max_bytes and num_episodes must be positive")
        self.name = name
        self.fps = fps
        self.smooth = max(smooth, 1)
        self.kwargs = kwargs or {}
        self.num_episodes = num_episodes
        self.temperature = temperature
        self.chunk = chunk
        self.max_bytes = max_bytes

    def start(self, algorithm, environment, algorithm_state, key):
        init_key, key = jax.random.split(key)
        environment_state, timestep = environment.init(init_key)
        _, carry = algorithm.init(init_key, timestep)
        return algorithm_state, carry, environment_state, timestep, key, jnp.int32(0)

    def lap(self, algorithm, environment, loop):
        def advance(loop, _):
            algorithm_state, carry, environment_state, timestep, key, episodes = loop
            key, algorithm_key, environment_key = jax.random.split(key, 3)
            algorithm_state, carry, action, _ = algorithm.step(
                algorithm_state, carry, algorithm_key, timestep, temperature=self.temperature
            )
            environment_state, timestep = environment.step(environment_key, environment_state, action)
            recorded = episodes < self.num_episodes
            episodes = episodes + jnp.all(timestep.done[0]).astype(jnp.int32)
            loop = algorithm_state, carry, environment_state, timestep, key, episodes
            return loop, (_first(environment_state), recorded)

        return jax.lax.scan(advance, loop, length=self.chunk)

    def craft(self, key, algorithm, environment, state, logs) -> Video:
        lap = jax.jit(partial(self.lap, algorithm, environment))
        loop = jax.jit(partial(self.start, algorithm, environment))(state.algorithm_state, key)

        def draw(current, previous, blend):
            extra = {} if previous is None else {"previous": previous, "blend": blend}
            return np.asarray(environment.render(current, **self.kwargs, **extra), dtype=np.uint8)

        def frames_of(pair):
            current, previous = pair
            if previous is None or self.smooth == 1:
                return [draw(current, None, 1.0)] * self.smooth
            return [draw(current, previous, sub / self.smooth) for sub in range(1, self.smooth + 1)]

        threaded = getattr(environment, "thread_safe_render", False)
        workers = min(4, os.cpu_count() or 1) if threaded else 1
        chunks, num_bytes, previous, done = [], 0, None, False
        with ThreadPoolExecutor(workers) as pool:
            while not done:
                loop, (states, recorded) = lap(loop)
                states, recorded = jax.device_get((states, recorded))
                currents = [jax.tree.map(lambda leaf: leaf[step], states) for step in np.flatnonzero(recorded)]
                pairs = list(zip(currents, [previous, *currents[:-1]]))
                previous = currents[-1] if currents else previous
                groups = []
                for group in pool.map(frames_of, pairs):
                    size = sum(frame.nbytes for frame in group)
                    if num_bytes + size > self.max_bytes:
                        warnings.warn(
                            f"Render clip stopped at {sum(map(len, chunks)) + len(groups)} steps, "
                            f"before the episode ended, at max_bytes={self.max_bytes}. "
                            "Raise artisans.render.max_bytes or lower the render resolution.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        done = True
                        break
                    num_bytes += size
                    groups.append(group)
                if groups:
                    chunks.append(groups)
                done = done or int(jax.device_get(loop[-1])) >= self.num_episodes

        frames = [frame for groups in chunks for group in groups for frame in group]
        if not frames:
            raise ValueError("Render rollout produced no frames; raise artisans.render.max_bytes")
        return Video(self.name, np.stack(frames), fps=self.fps * self.smooth)
