from collections.abc import Mapping
from operator import attrgetter
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, PyTree, Timestep

from .environment import Environment
from .spaces import Space
from .wrappers import WrapperState


@struct.dataclass
class MuJoCoPlaygroundState(WrapperState):
    model: PyTree


class MuJoCoPlayground(Environment):
    def __init__(self, env, fields: tuple[str, ...] = ()):
        self._env = env
        self.fields = fields

    def under(self, model: PyTree, fn, *args):
        base = self._env._mjx_model
        self._env._mjx_model = base.tree_replace(model)
        try:
            return fn(*args)
        finally:
            self._env._mjx_model = base

    def start(self, key: Key, model: PyTree) -> MuJoCoPlaygroundState:
        env_state = self.under(model, self._env.reset, key)
        steps = jnp.zeros(key.shape[:-1], dtype=jnp.int32)
        env_state = env_state.replace(info={**env_state.info, "steps": steps})
        return MuJoCoPlaygroundState(env_state=env_state, model=model)

    def init(self, key: Key) -> tuple[MuJoCoPlaygroundState, Timestep]:
        model = {name: attrgetter(name)(self._env.mjx_model) for name in self.fields}
        state = self.start(key, model)
        action_space = self.action_space()
        action = jnp.zeros(action_space.shape, action_space.dtype)
        _, shapes = jax.eval_shape(self.step, key, state, action)
        timestep = Timestep(
            obs=self.observe(state),
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
        self, key: Key, state: MuJoCoPlaygroundState, action: Array
    ) -> tuple[MuJoCoPlaygroundState, Timestep]:
        env_state = self.under(state.model, self._env.step, state.env_state, action)
        steps = env_state.info["steps"] + 1
        terminated = env_state.done.astype(jnp.bool)
        horizon = jnp.array(self._env._config.episode_length, dtype=jnp.int32)
        truncated = (steps >= horizon) & ~terminated
        env_state = env_state.replace(info={**env_state.info, "steps": steps})
        state = state.replace(env_state=env_state)
        timestep = Timestep(
            obs=self.observe(state),
            action=action,
            reward=env_state.reward,
            terminated=terminated,
            truncated=truncated,
            info=env_state.info,
        )
        return state, timestep

    def update(
        self, state: MuJoCoPlaygroundState, key: Key, theta: PyTree = None, **kwargs
    ) -> MuJoCoPlaygroundState:
        if theta is None:
            return state
        return self.start(key, theta)

    def observe(self, state: MuJoCoPlaygroundState) -> PyTree:
        return state.env_state.obs

    def observation_space(self) -> Space | dict[str, Space]:
        size = self._env.observation_size
        if isinstance(size, Mapping):
            return {name: unbounded(shape) for name, shape in size.items()}
        return unbounded((size,))

    def action_space(self) -> Space:
        low, high = self._env.mjx_model.actuator_ctrlrange.T
        return Space(
            shape=(self._env.action_size,), dtype=jnp.float32, low=low, high=high
        )

    def time_limit(self) -> int:
        return int(self._env._config.episode_length)

    def render(
        self,
        state: MuJoCoPlaygroundState,
        height: int = 240,
        width: int = 320,
        camera: str | None = None,
    ) -> Array:
        frames = self._env.render(
            [state.env_state], height=height, width=width, camera=camera
        )
        return frames[0]


def unbounded(shape: tuple[int, ...]) -> Space:
    return Space(shape=shape, dtype=jnp.float32, low=-jnp.inf, high=jnp.inf)


def randomized(models, in_axes) -> dict[str, Array]:
    leaves, _ = jax.tree_util.tree_flatten_with_path(in_axes)
    names = [jax.tree_util.keystr(path, simple=True, separator=".") for path, _ in leaves]
    return {name: attrgetter(name)(models) for name in names}


def model_sampler(env_id, config_overrides=None, **kwargs):
    from mujoco_playground import registry

    model = registry.load(env_id, config_overrides=config_overrides, **kwargs).mjx_model
    randomize = registry.get_domain_randomizer(env_id)

    def sample(key):
        models, in_axes = randomize(model, key[None])
        return {name: leaf[0] for name, leaf in randomized(models, in_axes).items()}

    return sample


def make(env_id, config_overrides=None, **kwargs):
    from mujoco_playground import registry

    env = registry.load(env_id, config_overrides=config_overrides, **kwargs)
    randomize = registry.get_domain_randomizer(env_id)
    fields = ()
    if randomize is not None:
        _, in_axes = randomize(env.mjx_model, jax.random.split(jax.random.PRNGKey(0), 1))
        fields = tuple(randomized(env.mjx_model, in_axes))
    return MuJoCoPlayground(env, fields)
