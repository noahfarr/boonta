import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from boonta.environments.environment import Environment
from boonta.environments.spaces import Space
from boonta.utils import Array, Key, Timestep


@struct.dataclass
class ChainOfForksState:
    cell: Array


def layout(depth: int, length: int, lure: float, seed: int):
    rng = np.random.default_rng(seed)
    top = depth + 2
    start = (top, 0)
    forward = {(top, column): (top, column + 1) for column in range(length)}
    branches, payouts, terminals = {}, {}, set()
    junction = (top, length)
    for _ in range(depth):
        row, column = junction
        good = -1 if rng.random() < 0.5 else 1
        bad = -good
        arms = {
            side: [(row + side, column + offset) for offset in range(length)]
            for side in (-1, 1)
        }
        for arm in arms.values():
            forward.update(zip(arm[:-1], arm[1:]))
        branches[junction] = (arms[-1][0], arms[1][0])
        dead_end = arms[bad][-1]
        pocket = (dead_end[0] + bad, dead_end[1])
        forward[dead_end] = pocket
        exits = ((pocket[0] + bad, pocket[1]), (pocket[0], pocket[1] + 1))
        paying = int(rng.random() < 0.5)
        branches[pocket] = exits
        payouts[exits[paying]] = lure
        payouts[exits[1 - paying]] = 0.0
        terminals.update(exits)
        ahead = (row + good, column + length)
        forward[arms[good][-1]] = ahead
        payouts[ahead] = 1.0
        junction = ahead
    terminals.add(junction)
    cells = sorted(set(forward) | set(forward.values()) | set(branches) | terminals)
    shift = min(row for row, _ in cells)
    index = {cell: position for position, cell in enumerate(cells)}
    successor = np.zeros((len(cells), 2), np.int32)
    for cell, position in index.items():
        if cell in branches:
            successor[position] = [index[target] for target in branches[cell]]
        elif cell in forward:
            successor[position] = index[forward[cell]]
        else:
            successor[position] = position
    reward = np.array([payouts.get(cell, 0.0) for cell in cells], np.float32)
    terminal = np.array([cell in terminals for cell in cells], bool)
    rows = np.array([row - shift for row, _ in cells], np.int32)
    columns = np.array([column for _, column in cells], np.int32)
    return index[start], successor, reward, terminal, rows, columns


def longest_episode(start: int, successor: np.ndarray, terminal: np.ndarray) -> int:
    steps_to_end = np.zeros(len(terminal), np.int64)
    for _ in range(len(terminal)):
        steps_to_end = np.where(terminal, 0, 1 + steps_to_end[successor].max(axis=-1))
    return int(steps_to_end[start])


def cell(env):
    return lambda env_state: env_state.cell


class ChainOfForks(Environment):
    def __init__(
        self,
        depth: int = 6,
        length: int = 4,
        lure: float = 0.8,
        hazard: float = 0.005,
        seed: int = 0,
    ):
        start, successor, reward, terminal, rows, columns = layout(depth, length, lure, seed)
        self.depth = depth
        self.hazard = hazard
        self.start = start
        self.successor = jnp.asarray(successor)
        self.reward = jnp.asarray(reward)
        self.terminal = jnp.asarray(terminal)
        self.rows = jnp.asarray(rows)
        self.columns = jnp.asarray(columns)
        self.num_cells = len(reward)
        self.longest_episode = longest_episode(start, successor, terminal)
        cell_grid = np.full((rows.max() + 1, columns.max() + 1), -1, np.int32)
        cell_grid[rows, columns] = np.arange(self.num_cells)
        self.cell_grid = jnp.asarray(cell_grid)

    def observation_space(self) -> Space:
        return Space((self.num_cells,), jnp.float32, 0.0, 1.0)

    def action_space(self) -> Space:
        return Space((), jnp.int32, 0, 1)

    def time_limit(self) -> int:
        return self.longest_episode

    def state_values(self, policy: Array) -> Array:
        survival = (1.0 - self.hazard) * ~self.terminal

        def backup(values, _):
            landing = self.successor
            return jnp.sum(policy * (self.reward[landing] + survival[landing] * values[landing]), axis=-1), None

        values, _ = jax.lax.scan(backup, jnp.zeros(self.num_cells), None, length=self.time_limit())
        return values

    def expected_return(self, policy: Array) -> Array:
        return self.state_values(policy)[self.start]

    def advantages(self, policy: Array) -> Array:
        survival = (1.0 - self.hazard) * ~self.terminal
        landing = self.successor
        action_values = self.reward[landing] + survival[landing] * self.state_values(policy)[landing]
        return action_values - jnp.sum(policy * action_values, axis=-1, keepdims=True)

    def transitions(self, policy: Array) -> Array:
        continuing = (1.0 - self.hazard) * ~self.terminal[self.successor]
        cells = jnp.arange(self.num_cells)[:, None]
        moves = jnp.zeros((self.num_cells, self.num_cells)).at[cells, self.successor].add(policy * continuing)
        return moves.at[:, self.start].add(jnp.sum(policy * (1.0 - continuing), axis=-1))

    def place(self, state: ChainOfForksState, cell: Array) -> ChainOfForksState:
        return state.replace(cell=jnp.int32(cell))

    def observe(self, state: ChainOfForksState) -> Array:
        return jax.nn.one_hot(state.cell, self.num_cells)

    def init(self, key: Key) -> tuple[ChainOfForksState, Timestep]:
        state = ChainOfForksState(cell=jnp.int32(self.start))
        return state, Timestep(
            obs=self.observe(state),
            action=jnp.int32(0),
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(True),
            truncated=jnp.bool_(False),
        )

    def step(self, key: Key, state: ChainOfForksState, action: Array) -> tuple[ChainOfForksState, Timestep]:
        cell = self.successor[state.cell, action]
        perished = jax.random.bernoulli(key, self.hazard)
        state = ChainOfForksState(cell=cell)
        return state, Timestep(
            obs=self.observe(state),
            action=action,
            reward=self.reward[cell],
            terminated=self.terminal[cell] | perished,
            truncated=jnp.bool_(False),
        )

    def render(self, state: ChainOfForksState, tile: int = 32) -> Array:
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
        reward = self.reward[jnp.maximum(cell, 0)]
        line = 1.0 / tile
        edge = (down < line) | (across < line) | (down > 1 - line) | (across > 1 - line)
        triangle = (
            (across > 0.12)
            & (across < 0.87)
            & (jnp.abs(down - 0.5) < 0.31 * (0.87 - across) / 0.75)
        )
        colour = jnp.zeros((*rows.shape, 3))
        colour = jnp.where(floor[..., None], jnp.array([100.0, 100.0, 100.0]), colour)
        colour = jnp.where((floor & (reward > 0))[..., None], jnp.array([255.0, 200.0, 0.0]), colour)
        colour = jnp.where((floor & (reward >= 1))[..., None], jnp.array([0.0, 255.0, 0.0]), colour)
        colour = jnp.where((floor & edge)[..., None], jnp.zeros(3), colour)
        colour = jnp.where(((cell == state.cell) & triangle)[..., None], jnp.array([255.0, 0.0, 0.0]), colour)
        return colour.astype(jnp.uint8)


def on_optimal_path(env):
    game = getattr(env, "unwrapped", env)
    successor = np.asarray(game.successor)
    reward = np.asarray(game.reward)
    terminal = np.asarray(game.terminal)
    survival = (1.0 - game.hazard) * (~terminal)
    values = np.zeros(game.num_cells)
    for _ in range(4 * game.num_cells):
        landing = reward[successor] + survival[successor] * values[successor]
        values = landing.max(axis=-1)
    path, cell = np.zeros(game.num_cells, bool), game.start
    while not path[cell]:
        path[cell] = True
        if terminal[cell]:
            break
        landing = reward[successor[cell]] + survival[successor[cell]] * values[successor[cell]]
        cell = successor[cell, np.argmax(landing)]
    path = jnp.asarray(path)
    return lambda env_state: path[env_state.cell]
