import importlib
from typing import Any

import jax
import jax.numpy as jnp
import lox

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space
from .wrappers.wrapper import Wrapper


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

    def observe(self, state: Any) -> Array:
        return self._env.get_obs(state)

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

    def time_limit(self) -> int:
        params = self._env.default_params
        return int(params.max_timesteps)

    def render(self, state: Any, block: int = 16) -> Array:
        if self._renderer is None:
            module = type(self._env).__module__
            package = "craftax_classic" if "classic" in module else "craftax"
            renderer = importlib.import_module(f"craftax.{package}.renderer")
            self._renderer = jax.jit(
                renderer.render_craftax_pixels, static_argnames="block_pixel_size"
            )
        return jnp.clip(self._renderer(state, block_pixel_size=block), 0, 255).astype(jnp.uint8)


def make(env_id, **kwargs):
    from craftax import craftax_env

    env = craftax_env.make_craftax_env_from_name(env_id, auto_reset=False, **kwargs)
    return Craftax(env)


def avalanche(key):
    key = (key ^ (key >> 16)) * jnp.uint32(0x85EBCA6B)
    key = (key ^ (key >> 13)) * jnp.uint32(0xC2B2AE35)
    return key ^ (key >> 16)


def progress(env, survival: bool = False, counts: bool = False):
    def cell(env_state):
        inventory = env_state.inventory
        held = jnp.stack(
            [
                inventory.wood,
                inventory.stone,
                inventory.coal,
                inventory.iron,
                inventory.diamond,
                inventory.sapling,
            ],
            axis=-1,
        )
        held = jnp.digitize(held, jnp.array([1, 2, 5])) if counts else held > 0
        features = [env_state.achievements.astype(jnp.int32), held.astype(jnp.int32)]
        if survival:
            vitals = jnp.stack(
                [
                    env_state.player_health,
                    env_state.player_food,
                    env_state.player_drink,
                    env_state.player_energy,
                ],
                axis=-1,
            )
            features.append(jnp.digitize(vitals, jnp.array([4, 7])).astype(jnp.int32))
            features.append((env_state.light_level < 0.5).astype(jnp.int32)[..., None])
        key = jnp.zeros(env_state.achievements.shape[:-1], jnp.uint32)
        for value in jnp.unstack(jnp.concatenate(features, axis=-1), axis=-1):
            key = avalanche(key * jnp.uint32(16777619) ^ value.astype(jnp.uint32))
        return avalanche(key).astype(jnp.int32)

    return cell


class LogAchievements(Wrapper):
    def step(self, key: Key, state: Any, action: Array) -> tuple[Any, Timestep]:
        state, timestep = self._env.step(key, state, action)
        info = timestep.info
        finished = timestep.done & info.get("rho", True)
        lox.log(
            {
                f"rho/{name.lower()}": jnp.where(finished, value / 100.0, jnp.nan)
                for name, value in info.items()
                if name.startswith("Achievements/")
            }
            | {"rho/score": jnp.where(finished, info["score"], jnp.nan)}
        )
        return state, timestep

    def update(self, state, key, **kwargs):
        return self._env.update(state, key, **kwargs)

    def action_mask(self, state):
        return self._env.action_mask(state)

    def observe(self, state):
        return self._env.observe(state)
