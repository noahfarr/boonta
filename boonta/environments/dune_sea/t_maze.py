import jax
import jax.numpy as jnp
from flax import struct

from boonta.environments.environment import Environment
from boonta.environments.spaces import Space
from boonta.utils import Array, Key, Timestep

CORRIDOR, JUNCTION = 2, 3


@struct.dataclass
class TMazeState:
    cue: Array
    position: Array


class TMaze(Environment):
    def __init__(self, length: int = 10):
        assert length >= 1
        self.length = length

    def observation_space(self) -> Space:
        return Space((4,), jnp.float32, 0.0, 1.0)

    def action_space(self) -> Space:
        return Space((), jnp.int32, 0, 1)

    def time_limit(self) -> int:
        return self.length + 1

    def observe(self, state: TMazeState) -> Array:
        sign = jnp.where(
            state.position == 0,
            state.cue,
            jnp.where(state.position == self.length, JUNCTION, CORRIDOR),
        )
        return jax.nn.one_hot(sign, 4)

    def init(self, key: Key) -> tuple[TMazeState, Timestep]:
        state = TMazeState(cue=jax.random.randint(key, (), 0, 2), position=jnp.int32(0))
        return state, Timestep(
            obs=self.observe(state),
            action=jnp.int32(0),
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(True),
            truncated=jnp.bool_(False),
        )

    def step(self, key: Key, state: TMazeState, action: Array) -> tuple[TMazeState, Timestep]:
        turned = state.position == self.length
        reward = jnp.where(turned & (action == state.cue), 1.0, 0.0)
        state = state.replace(position=jnp.minimum(state.position + 1, self.length))
        return state, Timestep(
            obs=self.observe(state),
            action=action,
            reward=reward.astype(jnp.float32),
            terminated=turned,
            truncated=jnp.bool_(False),
        )
