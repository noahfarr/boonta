from typing import Any

import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space


class Brax(Environment):
    def __init__(self, env):
        self._env = env

    def init(self, key: Key) -> tuple[Any, Timestep]:
        state = self._env.reset(key)
        action_space = self.action_space()
        action = jnp.zeros(action_space.shape, action_space.dtype)
        _, shapes = jax.eval_shape(self.step, key, state, action)
        timestep = Timestep(
            obs=state.obs,
            action=action,
            reward=jnp.zeros(shapes.reward.shape, shapes.reward.dtype),
            terminated=jnp.ones(shapes.terminated.shape, shapes.terminated.dtype),
            truncated=jnp.zeros(shapes.truncated.shape, shapes.truncated.dtype),
            info=jax.tree.map(
                lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shapes.info
            ),
        )
        return state, timestep

    def step(
        self, key: Key, state: Any, action: Array
    ) -> tuple[Any, Timestep]:
        state = self._env.step(state, action)
        truncated = state.info["truncation"].astype(jnp.bool)
        done = state.done.astype(jnp.bool)
        terminated = done & ~truncated
        timestep = Timestep(
            obs=state.obs,
            action=action,
            reward=state.reward,
            terminated=terminated,
            truncated=truncated,
            info=state.info,
        )
        return state, timestep

    def observation_space(self) -> Space:
        return Space(
            shape=(self._env.observation_size,),
            dtype=jnp.float32,
            low=-jnp.inf,
            high=jnp.inf,
        )

    def action_space(self) -> Space:
        low, high = self._env.sys.actuator.ctrl_range.T
        return Space(
            shape=(self._env.action_size,), dtype=jnp.float32, low=low, high=high
        )

    def time_limit(self) -> int:
        return int(self._env.episode_length)

    def render(
        self,
        state: Any,
        height: int = 240,
        width: int = 320,
        camera: str | None = None,
    ) -> Array:
        from brax.io import image

        return image.render_array(
            self._env.sys,
            state.pipeline_state,
            height=height,
            width=width,
            camera=camera,
        )


def make(env_id, backend="generalized", episode_length=1000, action_repeat=1, **kwargs):
    from brax import envs
    from brax.envs.wrappers.training import EpisodeWrapper

    env = envs.get_environment(env_name=env_id, backend=backend, **kwargs)
    env = EpisodeWrapper(
        env, episode_length=episode_length, action_repeat=action_repeat
    )
    return Brax(env)
