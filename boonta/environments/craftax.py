import importlib
from typing import Any

import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space


class Craftax(Environment):
    def __init__(self, env):
        self._env = env
        self._renderer = None

    def init(self, key: Key) -> tuple[Any, Timestep]:
        params = self._env.default_params
        obs, state = self._env.reset_env(key, params)
        action_space = self.action_space()
        action = jnp.zeros(action_space.shape, action_space.dtype)
        _, shapes = jax.eval_shape(self.step, key, state, action)
        timestep = Timestep(
            obs=obs,
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
        params = self._env.default_params
        obs, state, reward, done, info = self._env.step_env(key, state, action, params)
        timestep = Timestep(
            obs=obs,
            action=action,
            reward=reward,
            terminated=done,
            truncated=jnp.zeros_like(done),
            info=info,
        )
        return state, timestep

    def observation_space(self) -> Space:
        params = self._env.default_params
        space = self._env.observation_space(params)
        return Space(
            shape=space.shape, dtype=space.dtype, low=space.low, high=space.high
        )

    def action_space(self) -> Space:
        params = self._env.default_params
        space = self._env.action_space(params)
        return Space(shape=space.shape, dtype=space.dtype, low=0, high=space.n - 1)

    def horizon(self) -> int:
        params = self._env.default_params
        return int(params.max_timesteps)

    def render(self, state: Any, block: int = 16) -> Array:
        if self._renderer is None:
            module = type(self._env).__module__
            package = "craftax_classic" if "classic" in module else "craftax"
            renderer = importlib.import_module(f"craftax.{package}.renderer")
            self._renderer = renderer.make_craftax_pixel_renderer(block)
        return jnp.clip(self._renderer(state), 0, 255).astype(jnp.uint8)


def make(env_id, **kwargs):
    from craftax import craftax_env

    env = craftax_env.make_craftax_env_from_name(env_id, auto_reset=False, **kwargs)
    return Craftax(env)
