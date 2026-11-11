"""A fixed, seeded tower of Boxoban levels played by Jumanji Sokoban."""

from pathlib import Path
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from boonta.environments.environment import Environment
from boonta.environments.spaces import Space
from boonta.utils import Array, Key, PyTree, Timestep


def puzzles(path: str):
    """Read Boxoban's semicolon-delimited format without stripping floor spaces."""
    root = Path(path).expanduser()
    files = [root] if root.is_file() else sorted(root.rglob('*.txt'))
    if not files:
        raise FileNotFoundError(
            f'No Boxoban .txt files found at {root.resolve()}. '
            'Download the dataset with: git clone --depth 1 '
            'https://github.com/google-deepmind/boxoban-levels.git data/boxoban-levels '
            '(run from the project root), or set environment.kwargs.dataset_path '
            'to an existing Boxoban file or split directory.'
        )
    for file in files:
        rows = []
        for line in file.read_text().splitlines():
            if line.startswith(';') or not line:
                if rows:
                    yield '\n'.join(rows)
                    rows = []
            else:
                rows.append(line)
        if rows:
            yield '\n'.join(rows)


def parse_level(text: str):
    rows = text.splitlines()
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('Boxoban levels must be nonempty rectangular grids')
    grid = np.array([list(row) for row in rows])
    if not np.isin(grid, list('# @$.+*')).all():
        raise ValueError('Unknown Boxoban tile')
    walls = grid == '#'
    goals = np.isin(grid, list('.+*'))
    boxes = np.isin(grid, list('$*'))
    players = np.argwhere(np.isin(grid, list('@+')))
    if len(players) != 1 or boxes.sum() == 0 or boxes.sum() != goals.sum():
        raise ValueError('A level needs one player and equally many boxes and goals')
    if np.array_equal(boxes, goals):
        raise ValueError('A level must start unsolved')
    return walls, goals, boxes, players[0].astype(np.int32)


def sample_levels(source, count: int, seed: int):
    # Reservoir sampling keeps memory bounded even for the full training split.
    rng = np.random.default_rng(seed)
    selected = []
    for index, text in enumerate(source):
        if index < count:
            selected.append(text)
        else:
            slot = rng.integers(index + 1)
            if slot < count:
                selected[slot] = text
    if len(selected) < count:
        raise ValueError(f'Requested {count} floors, found only {len(selected)} levels')
    rng.shuffle(selected)
    parsed = [parse_level(text) for text in selected]
    if len({level[0].shape for level in parsed}) != 1:
        raise ValueError('All sampled floors must have the same grid shape')
    return tuple(jnp.asarray(np.stack(parts)) for parts in zip(*parsed))


@struct.dataclass
class BoxobanTowerState:
    floor: Array
    puzzle: PyTree
    steps: Array
    ended: Array

    @property
    def player(self):
        return self.puzzle.agent_location

    @property
    def boxes(self):
        return self.puzzle.variable_grid == 4

    @property
    def floor_steps(self):
        return self.puzzle.step_count


def cell(env):
    """Exact puzzle keys, supporting both single and batched states."""
    def key(state):
        boxes = state.boxes.reshape((*state.boxes.shape[:-2], -1))
        return jnp.concatenate((state.floor[..., None], state.player, boxes), axis=-1).astype(jnp.int32)
    return key


class BoxobanTower(Environment):
    """Fixed Boxoban floors played by Jumanji Sokoban: up, right, down, left."""

    def __init__(self, dataset_path: str | None = None, num_floors: int = 5,
                 seed: int = 0, horizon: int = 2000,
                 levels: Sequence[str] | None = None, floor_horizon: int = 120):
        from jumanji.environments.routing.sokoban.env import Sokoban
        from jumanji.environments.routing.sokoban.types import State

        if num_floors < 1 or horizon < 1 or floor_horizon < 1:
            raise ValueError('num_floors, horizon and floor_horizon must be positive')
        if (dataset_path is None) == (levels is None):
            raise ValueError('Supply exactly one of dataset_path or levels')
        self.num_floors, self.horizon = num_floors, horizon
        self.floor_horizon = floor_horizon
        source = puzzles(dataset_path) if levels is None else iter(levels)
        self.walls, self.goals, self.initial_boxes, self.initial_players = sample_levels(source, num_floors, seed)
        self.height, self.width = self.walls.shape[1:]
        if (self.height, self.width) != (10, 10):
            raise ValueError('Jumanji Sokoban requires 10 by 10 levels')
        if not bool(jnp.all(self.initial_boxes.sum(axis=(-2, -1)) == 4)):
            raise ValueError('Jumanji Sokoban requires four boxes and four targets per floor')
        self.fixed_grids = jnp.where(self.walls, 1, jnp.where(self.goals, 2, 0)).astype(jnp.uint8)
        self.variable_grids = self.initial_boxes.astype(jnp.uint8) * 4
        indices = jnp.arange(num_floors)
        self.variable_grids = self.variable_grids.at[
            indices, self.initial_players[:, 0], self.initial_players[:, 1]
        ].set(3)
        self._puzzle_state = State
        # Supplying a generator keeps native reset local and avoids dataset downloads.
        self.sokoban = Sokoban(
            generator=lambda key: self._floor_state(jnp.int32(0), key),
            time_limit=floor_horizon,
        )
        self._observe_puzzle = getattr(self.sokoban, 'observe', None)
        if self._observe_puzzle is None:
            self._observe_puzzle = self.sokoban._state_to_observation

    def _floor_state(self, floor, key):
        return self._puzzle_state(
            key=key, fixed_grid=self.fixed_grids[floor],
            variable_grid=self.variable_grids[floor],
            agent_location=self.initial_players[floor], step_count=jnp.int32(0),
        )

    def observation_space(self) -> Space:
        return Space((self.height, self.width, 6), jnp.float32, 0.0, 1.0)

    def action_space(self) -> Space:
        spec = self.sokoban.action_spec
        return Space((), jnp.int32, 0, spec.num_values - 1)

    def time_limit(self) -> int:
        return min(self.horizon, self.num_floors * self.floor_horizon)

    def observe(self, state: BoxobanTowerState) -> Array:
        observation = self._observe_puzzle(state.puzzle)
        variable, fixed = observation.grid[..., 0], observation.grid[..., 1]
        progress = jnp.full((self.height, self.width), state.floor / self.num_floors)
        clock = jnp.full((self.height, self.width), observation.step_count / self.floor_horizon)
        return jnp.stack((fixed == 1, fixed == 2, variable == 4, variable == 3,
                          progress, clock), axis=-1).astype(jnp.float32)

    def init(self, key: Key) -> tuple[BoxobanTowerState, Timestep]:
        puzzle, _ = self.sokoban.reset(key)
        state = BoxobanTowerState(jnp.int32(0), puzzle, jnp.int32(0), jnp.bool_(False))
        return state, Timestep(obs=self.observe(state), action=jnp.int32(0), reward=jnp.float32(0),
                               terminated=jnp.bool_(True), truncated=jnp.bool_(False))

    def step(self, key: Key, state: BoxobanTowerState, action: Array) -> tuple[BoxobanTowerState, Timestep]:
        def advance(_):
            puzzle, native = self.sokoban.step(state.puzzle, action)
            solved = self.sokoban.level_complete(puzzle)
            floor = state.floor + solved.astype(jnp.int32)
            steps = state.steps + 1
            ended = (native.last() & (~solved | (floor == self.num_floors))) | (steps >= self.horizon)
            ascend = solved & ~ended
            puzzle = jax.lax.cond(
                ascend, lambda: self._floor_state(jnp.minimum(floor, self.num_floors - 1), key),
                lambda: puzzle,
            )
            after = BoxobanTowerState(floor, puzzle, steps, ended)
            return after, Timestep(obs=self.observe(after), action=action,
                                   reward=jnp.asarray(native.reward, jnp.float32),
                                   terminated=ended, truncated=jnp.bool_(False))

        def finished(_):
            return state, Timestep(obs=self.observe(state), action=action, reward=jnp.float32(0),
                                   terminated=jnp.bool_(True), truncated=jnp.bool_(False))

        return jax.lax.cond(state.ended, finished, advance, operand=None)

    def render(self, state: BoxobanTowerState, tile: int = 32) -> Array:
        fixed = state.puzzle.fixed_grid
        goals = fixed == 2
        colour = jnp.broadcast_to(jnp.array([225, 215, 190], jnp.uint8), (self.height, self.width, 3))
        for mask, rgb in ((fixed == 1, [45, 50, 60]), (goals, [80, 180, 100]),
                          (state.boxes, [175, 110, 45]), (state.boxes & goals, [230, 190, 50])):
            colour = jnp.where(mask[..., None], jnp.array(rgb, jnp.uint8), colour)
        colour = colour.at[tuple(state.player)].set(jnp.array([65, 115, 230], jnp.uint8))
        return jnp.repeat(jnp.repeat(colour, tile, axis=0), tile, axis=1)
