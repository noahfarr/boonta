from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import GetAttrKey

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space


def _shared(env) -> Callable[[tuple], bool]:
    if env._config.get("impl") != "warp":
        return lambda path: False
    from mujoco.mjx._src.types import tree_path_to_attr_str
    from mujoco.mjx.warp import types

    table = types._BATCH_DIM["Data"]
    return lambda path: path[0] == GetAttrKey("data") and not table.get(tree_path_to_attr_str(path[1:]), True)


class MuJoCoPlayground(Environment):
    def __init__(self, env):
        self._env = env
        flat, self.structure = jax.tree_util.tree_flatten_with_path(jax.eval_shape(self._begin, jax.random.key(0)))
        shared = _shared(env)
        self.scratch = tuple(leaf if shared(path) else None for path, leaf in flat)

    def _begin(self, key: Key) -> Any:
        state = self._env.reset(key)
        steps = jnp.zeros(key.shape[:-1], dtype=jnp.int32)
        return state.replace(info={**state.info, "steps": steps})

    def _strip(self, state: Any) -> Any:
        leaves = jax.tree.leaves(state)
        return jax.tree.unflatten(
            self.structure, [leaf if spare is None else None for leaf, spare in zip(leaves, self.scratch)]
        )

    def _fill(self, state: Any) -> Any:
        kept = iter(jax.tree.leaves(state))
        return jax.tree.unflatten(
            self.structure,
            [next(kept) if spare is None else jnp.zeros(spare.shape, spare.dtype) for spare in self.scratch],
        )

    def init(self, key: Key) -> tuple[Any, Timestep]:
        state = self._strip(self._begin(key))
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
        state = self._env.step(self._fill(state), action)
        steps = state.info["steps"] + 1
        terminated = state.done.astype(jnp.bool)
        horizon = jnp.array(self._env._config.episode_length, dtype=jnp.int32)
        truncated = (steps >= horizon) & ~terminated
        state = self._strip(state.replace(info={**state.info, "steps": steps}))
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
        frames = self._env.render([self._fill(state)], height=height, width=width, camera=camera)
        return frames[0]


def make(env_id, config_overrides=None, **kwargs):
    from mujoco_playground import registry

    env = registry.load(env_id, config_overrides=config_overrides, **kwargs)
    if env._config.get("impl") == "warp":
        model = env._mjx_model
        env._mjx_model = model.replace(opt=model.opt.replace(_impl=model.opt._impl.replace(warn_overflow=0)))
    return MuJoCoPlayground(env)
