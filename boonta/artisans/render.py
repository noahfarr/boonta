from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from boonta.podracers.sebulba import SebulbaState

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

    def rollout(self, podracer, state, key):
        reset_key, lap_key, call_key, key = jax.random.split(key, 4)
        environment_state, timestep = podracer.environment.init(reset_key)
        _, carry = podracer.algorithm.init(reset_key, timestep)
        state = state.replace(
            timestep=timestep, environment_state=environment_state, carry=carry
        )
        state = podracer.pit(call_key, podracer.lap(lap_key, state))
        num_steps = podracer.environment.time_limit() + 1

        def step(state, key):
            state, transition = podracer.rollout(state, key, temperature=0.0)
            return state, (
                jax.tree.map(lambda leaf: jnp.take(leaf, 0, axis=0), state.environment_state),
                jnp.all(jnp.take(transition.second.done, 0, axis=0)),
            )

        state, (trajectory, done) = jax.lax.scan(
            step, state, jax.random.split(key, num_steps)
        )
        podracer.environment.close(state.environment_state)
        return trajectory, done

    def craft(self, key, podracer, state, logs) -> Video:
        if isinstance(state, SebulbaState):
            device, *_ = podracer.config.actor.devices.flat
            actor, *_ = state.actors
            state, key = jax.device_put(
                (actor.board(state.algorithm_state, state.pit_state), key), device
            )
        trajectory, done = jax.device_get(
            jax.jit(partial(self.rollout, podracer))(state, key)
        )
        num_frames = int(np.argmax(done)) + 1 if done.any() else len(done)
        frames = [
            np.asarray(
                podracer.environment.render(
                    jax.tree.map(lambda leaf: leaf[step], trajectory), **self.kwargs
                ),
                dtype=np.uint8,
            )
            for step in range(num_frames)
        ]
        return Video(self.name, np.stack(frames), fps=self.fps)
