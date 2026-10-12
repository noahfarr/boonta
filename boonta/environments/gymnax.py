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
        env_state = jax.tree.map(
            lambda leaf: jnp.asarray(leaf, jnp.result_type(leaf)), env_state
        )
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
            reward=jnp.asarray(reward, jnp.result_type(reward)),
            terminated=done,
            truncated=jnp.zeros_like(done),
            info=jax.tree.map(lambda leaf: jnp.asarray(leaf, leaf.dtype), info),
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

    def render(self, state: Any, block: int = 16) -> Array:
        grid = self._env.get_obs(getattr(state, "unwrapped", state), self._params)
        assert grid.ndim == 3, "only MinAtar's grid observations can be rendered"
        *_, num_channels = grid.shape
        palette = jnp.concatenate(
            [jnp.zeros((1, 3)), cubehelix(jnp.linspace(0, 1, num_channels + 2)[1:-1])]
        )
        channel = jnp.max(grid * jnp.arange(1, num_channels + 1), axis=-1)
        image = (palette[channel.astype(jnp.int32)] * 255).astype(jnp.uint8)
        return jnp.repeat(jnp.repeat(image, block, axis=0), block, axis=1)


def cubehelix(x: Array, start: float = 0.5, rotations: float = -1.5, hue: float = 1.0) -> Array:
    angle = 2 * jnp.pi * (start / 3 + rotations * x)
    amplitude = hue * x * (1 - x) / 2
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    rgb = x[:, None] + amplitude[:, None] * jnp.stack(
        [
            -0.14861 * cos + 1.78277 * sin,
            -0.29227 * cos - 0.90649 * sin,
            1.97294 * cos,
        ],
        axis=-1,
    )
    return jnp.clip(rgb, 0, 1)


def make(env_id, params=None, **kwargs):
    import gymnax

    env, env_params = gymnax.make(env_id, **kwargs)
    if params:
        env_params = env_params.replace(**params)
    return Gymnax(env, env_params)
