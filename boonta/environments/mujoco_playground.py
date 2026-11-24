from typing import Any

import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space


class MuJoCoPlayground(Environment):
    def __init__(self, env):
        self._env = env

    def init(self, key: Key) -> tuple[Any, Timestep]:
        state = self._env.reset(key)
        steps = jnp.zeros(key.shape[:-1], dtype=jnp.int32)
        state = state.replace(info={**state.info, "steps": steps})
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
        steps = state.info["steps"] + 1
        terminated = state.done.astype(jnp.bool)
        horizon = jnp.array(self._env._config.episode_length, dtype=jnp.int32)
        truncated = (steps >= horizon) & ~terminated
        state = state.replace(info={**state.info, "steps": steps})
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
        low, high = self._env.mjx_model.actuator_ctrlrange.T
        return Space(
            shape=(self._env.action_size,), dtype=jnp.float32, low=low, high=high
        )

    def time_limit(self) -> int:
        return int(self._env._config.episode_length)

    def render(
        self,
        state: Any,
        height: int = 240,
        width: int = 320,
        camera: str | None = None,
    ) -> Array:
        frames = self._env.render([state], height=height, width=width, camera=camera)
        return frames[0]


def make(env_id, config_overrides=None, **kwargs):
    from mujoco_playground import registry

    env = registry.load(env_id, config_overrides=config_overrides, **kwargs)
    return MuJoCoPlayground(env)
