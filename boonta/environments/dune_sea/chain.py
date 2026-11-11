import jax
import jax.numpy as jnp
from flax import struct

from boonta.environments.environment import Environment
from boonta.environments.spaces import Space
from boonta.utils import Array, Key, Timestep


@struct.dataclass
class ChainState:
    position: Array


def cell(env):
    return lambda env_state: env_state.position


class Chain(Environment):
    def __init__(self, length: int = 100, num_actions: int = 2):
        assert length >= 1 and num_actions >= 2
        self.length = length
        self.num_actions = num_actions
        self.num_cells = length + 1
        self.start = 0
        self.secret = num_actions - 1
        self.cell_grid = jnp.arange(self.num_cells, dtype=jnp.int32)[None]

    def observation_space(self) -> Space:
        return Space((self.num_cells,), jnp.float32, 0.0, 1.0)

    def action_space(self) -> Space:
        return Space((), jnp.int32, 0, self.num_actions - 1)

    def time_limit(self) -> int:
        return self.num_cells

    def expected_return(self, policy: Array) -> Array:
        return policy[self.length, self.secret]

    def advantages(self, policy: Array) -> Array:
        payoff = jnp.zeros_like(policy).at[self.length, self.secret].set(1.0)
        return payoff - jnp.sum(policy * payoff, axis=-1, keepdims=True)

    def transitions(self, policy: Array) -> Array:
        return jnp.roll(jnp.eye(self.num_cells), 1, axis=-1)

    def place(self, state: ChainState, cell: Array) -> ChainState:
        return state.replace(position=jnp.int32(cell))

    def observe(self, state: ChainState) -> Array:
        return jax.nn.one_hot(state.position, self.num_cells)

    def init(self, key: Key) -> tuple[ChainState, Timestep]:
        state = ChainState(position=jnp.int32(self.start))
        return state, Timestep(
            obs=self.observe(state),
            action=jnp.int32(0),
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(True),
            truncated=jnp.bool_(False),
        )

    def step(self, key: Key, state: ChainState, action: Array) -> tuple[ChainState, Timestep]:
        deciding = state.position == self.length
        reward = deciding & (action == self.secret)
        state = ChainState(position=jnp.minimum(state.position + 1, self.length))
        return state, Timestep(
            obs=self.observe(state),
            action=action,
            reward=reward.astype(jnp.float32),
            terminated=deciding,
            truncated=jnp.bool_(False),
        )

    def render(self, state: ChainState, tile: int = 32) -> Array:
        down, columns = jnp.meshgrid(
            (jnp.arange(tile) + 0.5) / tile,
            (jnp.arange(self.num_cells * tile) + 0.5) / tile,
            indexing="ij",
        )
        cell = jnp.floor(columns).astype(jnp.int32)
        across = columns - cell
        line = 1.0 / tile
        edge = (down < line) | (across < line) | (down > 1 - line) | (across > 1 - line)
        triangle = (
            (across > 0.12)
            & (across < 0.87)
            & (jnp.abs(down - 0.5) < 0.31 * (0.87 - across) / 0.75)
        )
        colour = jnp.full((*down.shape, 3), 100.0)
        colour = jnp.where((cell == self.length)[..., None], jnp.array([0.0, 255.0, 0.0]), colour)
        colour = jnp.where(edge[..., None], jnp.zeros(3), colour)
        colour = jnp.where(((cell == state.position) & triangle)[..., None], jnp.array([255.0, 0.0, 0.0]), colour)
        return colour.astype(jnp.uint8)
