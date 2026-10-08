import warnings
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces as gymnax_spaces

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space
from .wrappers import WrapperState

warnings.filterwarnings(
    "ignore",
    message="scatter inputs have incompatible types: cannot safely cast value from dtype=.* to dtype=bool",
    category=FutureWarning,
)


@struct.dataclass
class GymnaxState(WrapperState):
    params: Any


class Gymnax(Environment):

    def __init__(self, env, params):
        self._env = env
        self._params = params

    def init(self, key: Key) -> tuple[GymnaxState, Timestep]:
        obs, env_state = self._env.reset_env(key, self._params)
        state = GymnaxState(env_state=env_state, params=self._params)
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
        self, key: Key, state: GymnaxState, action: Array
    ) -> tuple[GymnaxState, Timestep]:
        obs, env_state, reward, done, info = self._env.step_env(
            key, state.env_state, action, state.params
        )
        timestep = Timestep(
            obs=obs,
            action=action,
            reward=reward,
            terminated=done,
            truncated=jnp.zeros_like(done),
            info=info,
        )
        return GymnaxState(env_state=env_state, params=state.params), timestep

    def observation_space(self) -> Space:
        space = self._env.observation_space(self._params)
        return Space(
            shape=space.shape, dtype=space.dtype, low=space.low, high=space.high
        )

    def action_space(self) -> Space:
        space = self._env.action_space(self._params)

        if isinstance(space, gymnax_spaces.Discrete):
            return Space(shape=space.shape, dtype=space.dtype, low=0, high=space.n - 1)
        return Space(
            shape=space.shape, dtype=space.dtype, low=space.low, high=space.high
        )

    def time_limit(self) -> int:
        return int(self._params.max_steps_in_episode)


def make(env_id, params=None, **kwargs):
    import gymnax

    env, env_params = gymnax.make(env_id, **kwargs)
    if params:
        env_params = env_params.replace(**params)
    return Gymnax(env, env_params)
