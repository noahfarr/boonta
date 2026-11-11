import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from boonta.environments.environment import Environment
from boonta.environments.spaces import Space
from boonta.utils import Array, Key, Timestep

MOVES = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)], np.int32)


@struct.dataclass
class MazeState:
    position: Array
    clock: Array


def layout(size: int, seed: int) -> np.ndarray:
    """Randomized depth-first search: rooms on the odd tiles, one route between any two."""
    rng = np.random.default_rng(seed)
    walls = np.ones((2 * size + 1, 2 * size + 1), bool)
    walls[1, 1] = False
    trail = [(1, 1)]
    while trail:
        row, column = trail[-1]
        unvisited = [
            (down, across)
            for down, across in MOVES
            if 0 < row + 2 * down < 2 * size
            and 0 < column + 2 * across < 2 * size
            and walls[row + 2 * down, column + 2 * across]
        ]
        if not unvisited:
            trail.pop()
            continue
        down, across = unvisited[rng.integers(len(unvisited))]
        walls[row + down, column + across] = False
        walls[row + 2 * down, column + 2 * across] = False
        trail.append((row + 2 * down, column + 2 * across))
    return walls


def cell(env):
    return lambda env_state: env_state.position


class Maze(Environment):
    def __init__(self, size: int = 8, horizon: int = 500, seed: int = 0):
        assert size >= 2
        walls = layout(size, seed)
        rows, columns = np.nonzero(~walls)
        self.num_cells = len(rows)
        cell_grid = np.full(walls.shape, -1, np.int32)
        cell_grid[rows, columns] = np.arange(self.num_cells)
        landing = cell_grid[rows[:, None] + MOVES[:, 0], columns[:, None] + MOVES[:, 1]]
        successor = np.where(landing >= 0, landing, np.arange(self.num_cells)[:, None])
        goal = cell_grid[-2, -2]
        distances = np.hypot(rows - rows[goal], columns - columns[goal])
        self.size = size
        self.horizon = horizon
        self.start = int(cell_grid[1, 1])
        self.goal = int(goal)
        self.rows = jnp.asarray(rows, jnp.int32)
        self.columns = jnp.asarray(columns, jnp.int32)
        self.cell_grid = jnp.asarray(cell_grid)
        self.successor = jnp.asarray(successor, jnp.int32)
        self.reward = jnp.asarray(-distances / distances.max(), jnp.float32)

    def observation_space(self) -> Space:
        return Space((self.num_cells,), jnp.float32, 0.0, 1.0)

    def action_space(self) -> Space:
        return Space((), jnp.int32, 0, len(MOVES) - 1)

    def time_limit(self) -> int:
        return self.horizon

    def place(self, state: MazeState, cell: Array) -> MazeState:
        return state.replace(position=jnp.int32(cell), clock=jnp.zeros_like(state.clock))

    def observe(self, state: MazeState) -> Array:
        return jax.nn.one_hot(state.position, self.num_cells)

    def init(self, key: Key) -> tuple[MazeState, Timestep]:
        state = MazeState(position=jnp.int32(self.start), clock=jnp.int32(0))
        return state, Timestep(
            obs=self.observe(state),
            action=jnp.int32(0),
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(True),
            truncated=jnp.bool_(False),
        )

    def step(self, key: Key, state: MazeState, action: Array) -> tuple[MazeState, Timestep]:
        position = self.successor[state.position, action]
        clock = state.clock + 1
        arrived = position == self.goal
        state = MazeState(position=position, clock=clock)
        return state, Timestep(
            obs=self.observe(state),
            action=action,
            reward=self.reward[position],
            terminated=arrived,
            truncated=~arrived & (clock >= self.horizon),
        )

    def render(self, state: MazeState, tile: int = 32) -> Array:
        num_rows, num_columns = self.cell_grid.shape
        rows, columns = jnp.meshgrid(
            (jnp.arange(num_rows * tile) + 0.5) / tile,
            (jnp.arange(num_columns * tile) + 0.5) / tile,
            indexing="ij",
        )
        row, column = jnp.floor(rows).astype(jnp.int32), jnp.floor(columns).astype(jnp.int32)
        down, across = rows - row, columns - column
        cell = self.cell_grid[row, column]
        floor = cell >= 0
        line = 1.0 / tile
        edge = (down < line) | (across < line) | (down > 1 - line) | (across > 1 - line)
        triangle = (
            (across > 0.12)
            & (across < 0.87)
            & (jnp.abs(down - 0.5) < 0.31 * (0.87 - across) / 0.75)
        )
        colour = jnp.full((*rows.shape, 3), 100.0)
        colour = jnp.where(floor[..., None], jnp.zeros(3), colour)
        colour = jnp.where((cell == self.goal)[..., None], jnp.array([0.0, 255.0, 0.0]), colour)
        colour = jnp.where((floor & edge)[..., None], jnp.array([100.0, 100.0, 100.0]), colour)
        colour = jnp.where(((cell == state.position) & triangle)[..., None], jnp.array([255.0, 0.0, 0.0]), colour)
        return colour.astype(jnp.uint8)
