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
        self.height = int(rows.max()) + 1
        self.width = int(columns.max()) + 1

    def observation_space(self) -> Space:
        return Space((self.height + self.width,), jnp.float32, 0.0, 1.0)

    def action_space(self) -> Space:
        return Space((), jnp.int32, 0, 1)

    def observe(self, state: ChainOfForksState) -> Array:
        return jnp.concatenate(
            [
                jax.nn.one_hot(self.rows[state.cell], self.height),
                jax.nn.one_hot(self.columns[state.cell], self.width),
            ]
        )

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
