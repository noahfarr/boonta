from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, PyTree, Timestep

from .environment import Environment
from .spaces import Space


def unpack_observation(observation) -> dict[str, Array]:
    return {"image": observation.image, "agent_dir": observation.agent_dir}


class JaxUED(Environment):
    def __init__(self, env, params, sample: Callable[[Key], PyTree]):
        self._env = env
        self._params = params
        self._sample = sample

    def begin(self, level: PyTree) -> Any:
        state = self._env.init_state_from_level(level)
        return self.settle(state)

    def settle(self, state: Any) -> Any:
        return state.replace(
            time=jnp.asarray(state.time, jnp.int32),
            terminal=jnp.asarray(state.terminal, jnp.bool_),
        )

    def init(self, key: Key) -> tuple[Any, Timestep]:
        state = self.begin(self._sample(key))
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
        obs, state, reward, done, info = self._env.step_env(
            key, state, action, self._params
        )
        state = self.settle(state)
        reached = reward > 0
        truncated = done & (state.time >= self._params.max_steps_in_episode) & ~reached
        timestep = Timestep(
            obs=unpack_observation(obs),
            action=action,
            reward=reward,
            terminated=done & ~truncated,
            truncated=truncated,
            info=info,
        )
        return state, timestep

    def update(self, state: Any, theta: PyTree = None, **kwargs) -> Any:
        if theta is None:
            return state
        return self.begin(theta)

    def observe(self, state: Any) -> dict[str, Array]:
        return unpack_observation(self._env.get_obs(state))

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

    def action_space(self) -> Space:
        space = self._env.action_space(self._params)
        return Space(shape=(), dtype=jnp.int32, low=0, high=space.n - 1)

    def time_limit(self) -> int:
        return int(self._params.max_steps_in_episode)


def maze_generator(
    height: int = 13, width: int = 13, n_walls: int = 25
) -> Callable[[Key], PyTree]:
    from jaxued.environments.maze import make_level_generator

    return make_level_generator(height, width, n_walls)


def maze_mutator(
    max_num_edits: int = 100, num_edits: int = 5
) -> Callable[[Key, PyTree], PyTree]:
    from jaxued.environments.maze import make_level_mutator_minimax

    mutate = make_level_mutator_minimax(max_num_edits)

    def edit_level(key: Key, level: PyTree) -> PyTree:
        return mutate(key, level, num_edits)

    return edit_level


def make(
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
):
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
    return JaxUED(env, params, maze_generator(height, width, n_walls))
