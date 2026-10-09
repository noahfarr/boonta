from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from gymnax.environments import spaces as gymnax_spaces

from boonta.utils import Array, Key, PyTree, Timestep

from .environment import Environment
from .spaces import Space


def strong_leaf(leaf: Any) -> Array:
    return jnp.asarray(leaf, jnp.asarray(leaf).dtype)


def convert_space(space) -> Space:
    if isinstance(space, gymnax_spaces.Discrete):
        return Space(shape=(), dtype=jnp.int32, low=0, high=space.n - 1)
    return Space(shape=space.shape, dtype=space.dtype, low=space.low, high=space.high)


class JaxUED(Environment):
    def __init__(self, env, params, sample: Callable[[Key], PyTree]):
        self._env = env
        self._params = params
        self._sample = sample

    def start(self, level: PyTree, key: Key) -> Any:
        _, state = self._env.reset_env_to_level(key, level, self._params)
        return self.settle(state)

    def settle(self, state: Any) -> Any:
        return jax.tree.map(strong_leaf, state)

    def cut(self, state: Any, reward: Array, done: Array) -> Array:
        unlimited = self._params.replace(
            max_steps_in_episode=jnp.iinfo(jnp.int32).max
        )
        return done & ~self._env.is_terminal(state, unlimited)

    def init(self, key: Key) -> tuple[Any, Timestep]:
        state = self.start(self._sample(key), jax.random.fold_in(key, 1))
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

    def step(self, key: Key, state: Any, action: Array) -> tuple[Any, Timestep]:
        _, state, reward, done, info = self._env.step_env(
            key, state, action, self._params
        )
        state = self.settle(state)
        reward = jnp.asarray(reward, jnp.float32)
        truncated = self.cut(state, reward, done)
        timestep = Timestep(
            obs=self.observe(state),
            action=action,
            reward=reward,
            terminated=done & ~truncated,
            truncated=truncated,
            info=info,
        )
        return state, timestep

    def update(self, state: Any, key: Key, theta: PyTree = None, **kwargs) -> Any:
        if theta is None:
            return state
        return self.start(theta, key)

    def observe(self, state: Any) -> PyTree:
        return self._env.get_obs(state)

    def observation_space(self) -> Space:
        if hasattr(self._env, "observation_space"):
            return convert_space(self._env.observation_space(self._params))
        shape = jax.eval_shape(self.init, jax.random.key(0))[1].obs
        return Space(shape=shape.shape, dtype=shape.dtype, low=-jnp.inf, high=jnp.inf)

    def action_space(self) -> Space:
        return convert_space(self._env.action_space(self._params))

    def time_limit(self) -> int:
        return int(self._params.max_steps_in_episode)


class Maze(JaxUED):
    def start(self, level: PyTree, key: Key) -> Any:
        return self.settle(self._env.init_state_from_level(level))

    def cut(self, state: Any, reward: Array, done: Array) -> Array:
        reached = reward > 0
        return done & (state.time >= self._params.max_steps_in_episode) & ~reached

    def observe(self, state: Any) -> dict[str, Array]:
        observation = self._env.get_obs(state)
        return {"image": observation.image, "agent_dir": observation.agent_dir}

    def observation_space(self) -> dict[str, Space]:
        if self._env.fully_obs:
            shape = (self._env.max_height, self._env.max_width, 3)
        else:
            shape = (self._env.agent_view_size, self._env.agent_view_size, 3)
        if self._env.normalize_obs:
            image = Space(shape=shape, dtype=jnp.float32, low=0.0, high=1.0)
        else:
            image = Space(shape=shape, dtype=jnp.uint8, low=0, high=10)
        return {
            "image": image,
            "agent_dir": Space(shape=(), dtype=jnp.uint8, low=0, high=3),
        }


CONTROL = {"CartPole": "cartpole", "Acrobot": "acrobot", "Pendulum": "pendulum"}


def maze_generator(
    height: int = 13, width: int = 13, n_walls: int = 25
) -> Callable[[Key], PyTree]:
    from jaxued.environments.maze import make_level_generator

    return make_level_generator(height, width, n_walls)


def control_generator(env_id: str) -> Callable[[Key], PyTree]:
    import importlib

    module = importlib.import_module(f"jaxued.environments.gymnax.{CONTROL[env_id]}")
    return module.make_level_generator()


def make_maze(
    env_id: str = "Maze",
    height: int = 13,
    width: int = 13,
    n_walls: int = 25,
    agent_view_size: int = 5,
    see_agent: bool = False,
    normalize_obs: bool = True,
    fully_obs: bool = False,
    penalize_time: bool = True,
    max_steps_in_episode: int = 250,
) -> Maze:
    from jaxued.environments import maze

    kind = getattr(maze, env_id)
    env = kind(
        max_height=height,
        max_width=width,
        agent_view_size=agent_view_size,
        see_agent=see_agent,
        normalize_obs=normalize_obs,
        fully_obs=fully_obs,
        penalize_time=penalize_time,
    )
    params = env.default_params.replace(max_steps_in_episode=max_steps_in_episode)
    return Maze(env, params, maze_generator(height, width, n_walls))


def make_control(env_id: str, max_steps_in_episode: int | None = None) -> JaxUED:
    import importlib

    module = importlib.import_module(f"jaxued.environments.gymnax.{CONTROL[env_id]}")
    env = getattr(module, env_id)()
    params = env.default_params
    if max_steps_in_episode is not None:
        params = params.replace(max_steps_in_episode=max_steps_in_episode)
    return JaxUED(env, params, control_generator(env_id))


def make(env_id: str = "Maze", **kwargs):
    if env_id in CONTROL:
        return make_control(env_id, **kwargs)
    return make_maze(env_id, **kwargs)
