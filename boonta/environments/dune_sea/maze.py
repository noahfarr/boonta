import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from boonta.utils import Array, Key, Timestep

from ..environment import Environment
from ..spaces import Space

NOOP, FORWARD, BACKWARD, LEFT, RIGHT, TURN_LEFT, TURN_RIGHT = range(7)
NUM_ACTIONS = 7
DIRECTIONS = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32)
MOVE_OFFSET = jnp.array([0, 0, 2, 3, 1, 0, 0], dtype=jnp.int32)
STEP_SIZE = jnp.array([0, 1, 1, 1, 1, 0, 0], dtype=jnp.int32)
TURN_OFFSET = jnp.array([0, 0, 0, 0, 0, 3, 1], dtype=jnp.int32)
NUM_CHANNELS = 3
TIME_PENALTY = 0.9
LAZINESS = NUM_ACTIONS / 4
LIMIT_SAMPLES = 256
LIMIT_QUANTILE = 0.9

TILE = 32
SUBDIVISIONS = 3
GRID_LINE = 0.031
GREY = np.array([100, 100, 100], np.float32)
GREEN = np.array([0, 255, 0], np.float32)
RED = np.array([255, 0, 0], np.float32)
WHITE = np.array([255, 255, 255], np.float32)
HIGHLIGHT = 0.3
AGENT_TRIANGLE = ((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))


@struct.dataclass
class Level:
    wall: Array
    start: Array
    direction: Array
    goal: Array


@struct.dataclass
class MazeState:
    level: Level
    position: Array
    direction: Array
    time: Array


def sample(plane: Array, cells: Array, outside: float) -> Array:
    height, width = plane.shape
    rows, columns = cells[..., 0], cells[..., 1]
    inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    return jnp.where(inside, plane[jnp.clip(rows, 0, height - 1), jnp.clip(columns, 0, width - 1)], outside)


def view_offsets(view_size: int) -> np.ndarray:
    pixels = np.arange(view_size**2).reshape(view_size, view_size)
    sources = np.stack([np.rot90(pixels, k) for k in range(4)])
    return (np.stack(np.divmod(sources, view_size), -1) - view_size // 2).astype(np.int32)


def grow(cells: Array) -> Array:
    size, _ = cells.shape
    padded = jnp.pad(cells, 1)
    return functools.reduce(
        jnp.logical_or,
        [padded[row : row + size, column : column + size] for row in range(3) for column in range(3)],
    )


def spread(clear: Array) -> Array:
    size, _ = clear.shape
    radius = size // 2
    seen = jnp.zeros(clear.shape, bool).at[radius, radius].set(True)
    for _ in range(radius):
        seen = seen | grow(seen & clear)
    return seen | (grow(seen & clear) & ~clear)


def passages(height: int, width: int) -> np.ndarray:
    cells = np.arange(height * width).reshape(height, width)
    across = np.stack([cells[:, :-1].ravel(), cells[:, 1:].ravel()], -1)
    down = np.stack([cells[:-1].ravel(), cells[1:].ravel()], -1)
    return np.concatenate([across, down])


def kruskal(key: Key, edges: Array, num_cells: int) -> Array:
    order = jax.random.permutation(key, edges.shape[0])

    def join(label, edge):
        first, second = label[edge[0]], label[edge[1]]
        joined = first != second
        return jnp.where(joined & (label == second), first, label), joined

    _, carved = jax.lax.scan(join, jnp.arange(num_cells), edges[order])
    return jnp.zeros(edges.shape[0], bool).at[order].set(carved)


def distances(wall: Array, start: Array) -> Array:
    far = wall.size
    open_ground = wall == 0
    distance = jnp.full(wall.shape, far, jnp.int32).at[start[0], start[1]].set(0)

    def relax(carry):
        distance, _ = carry
        padded = jnp.pad(distance, 1, constant_values=far)
        nearest = jnp.stack(
            [padded[2:, 1:-1], padded[:-2, 1:-1], padded[1:-1, 2:], padded[1:-1, :-2]]
        ).min(0)
        relaxed = jnp.where(open_ground, jnp.minimum(distance, nearest + 1), far)
        return relaxed, (relaxed != distance).any()

    distance, _ = jax.lax.while_loop(lambda carry: carry[1], relax, (distance, jnp.bool_(True)))
    return distance


def kruskal_generator(height: int = 13, width: int = 13, braid: float = 0.0) -> Callable[[Key], Level]:
    edges = passages(height, width)
    cells = np.stack(np.divmod(np.arange(height * width), width), -1) * 2 + 1
    middle = jnp.asarray((cells[edges[:, 0]] + cells[edges[:, 1]]) // 2)
    cells, edges = jnp.asarray(cells), jnp.asarray(edges)
    rows, columns = 2 * height + 1, 2 * width + 1

    def generate(key: Key) -> Level:
        tree_key, braid_key, start_key, direction_key = jax.random.split(key, 4)
        carved = kruskal(tree_key, edges, height * width)
        carved = carved | jax.random.bernoulli(braid_key, braid, carved.shape)
        wall = jnp.ones((rows, columns), jnp.float32).at[cells[:, 0], cells[:, 1]].set(0.0)
        wall = wall.at[middle[:, 0], middle[:, 1]].set(jnp.where(carved, 0.0, 1.0))
        start = cells[jax.random.randint(start_key, (), 0, height * width)]
        distance = distances(wall, start)[1::2, 1::2]
        return Level(
            wall=wall,
            start=start.astype(jnp.int32),
            direction=jax.random.randint(direction_key, (), 0, 4),
            goal=cells[jnp.argmax(distance)].astype(jnp.int32),
        )

    return generate


def exploration_limit(generator: Callable[[Key], Level]) -> int:
    levels = jax.vmap(generator)(jax.random.split(jax.random.key(0), LIMIT_SAMPLES))

    def bound(level: Level) -> Array:
        open_squares = (level.wall == 0).sum()
        path = distances(level.wall, level.start)[level.goal[0], level.goal[1]]
        return 4 * LAZINESS * open_squares * path

    return int(np.ceil(np.quantile(np.asarray(jax.vmap(bound)(levels)), LIMIT_QUANTILE)))


@functools.cache
def default_limit(height: int, width: int, braid: float) -> int:
    return exploration_limit(kruskal_generator(height, width, braid))


class Maze(Environment):
    thread_safe_render = True

    def __init__(
        self,
        height: int = 13,
        width: int = 13,
        braid: float = 0.0,
        view_size: int = 5,
        time_limit: int | None = None,
        generator: Callable[[Key], Level] | None = None,
    ):
        self.height = height
        self.width = width
        self.rows = 2 * height + 1
        self.columns = 2 * width + 1
        self.view_size = view_size
        self.generator = generator or kruskal_generator(height, width, braid)
        if time_limit is None:
            time_limit = default_limit(height, width, braid) if generator is None else exploration_limit(generator)
        self._time_limit = time_limit
        self.offsets = jnp.asarray(view_offsets(view_size))

    def start(self, level: Level) -> MazeState:
        return MazeState(
            level=level,
            position=jnp.asarray(level.start, jnp.int32),
            direction=jnp.asarray(level.direction, jnp.int32),
            time=jnp.int32(0),
        )

    def init(self, key: Key) -> tuple[MazeState, Timestep]:
        state = self.start(self.generator(key))
        return state, Timestep(
            obs=self.observe(state),
            action=jnp.int32(NOOP),
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(True),
            truncated=jnp.bool_(False),
            info={},
        )

    def step(self, key: Key, state: MazeState, action: Array) -> tuple[MazeState, Timestep]:
        move = jnp.clip(jnp.asarray(action, jnp.int32), 0, NUM_ACTIONS - 1)
        heading = DIRECTIONS[(state.direction + MOVE_OFFSET[move]) % 4]
        target = state.position + STEP_SIZE[move] * heading
        position = jnp.where(sample(state.level.wall, target, 1.0) == 0, target, state.position)
        time = state.time + 1
        state = state.replace(
            position=position, direction=(state.direction + TURN_OFFSET[move]) % 4, time=time
        )
        reached = jnp.all(position == state.level.goal)
        reward = jnp.where(reached, 1.0 - TIME_PENALTY * time / self._time_limit, 0.0)
        return state, Timestep(
            obs=self.observe(state),
            action=action,
            reward=reward.astype(jnp.float32),
            terminated=reached,
            truncated=~reached & (time >= self._time_limit),
            info={},
        )

    def update(self, state: MazeState, key: Key, theta: Level | None = None, **kwargs) -> MazeState:
        if theta is None:
            return state
        return self.start(theta)

    def observe(self, state: MazeState) -> dict[str, Array]:
        offsets = self.offsets[(state.direction + 1) % 4]
        cells = state.position + offsets
        wall = sample(state.level.wall, cells, 1.0)
        visible = spread(wall == 0)
        goal = jnp.all(cells == state.level.goal, -1)
        terrain = jnp.stack([wall, goal.astype(jnp.float32)], -1)
        image = jnp.concatenate([terrain * visible[..., None], visible[..., None]], -1)
        return {"image": image.astype(jnp.float32), "agent_dir": state.direction}

    def observation_space(self) -> dict[str, Space]:
        return {
            "image": Space((self.view_size, self.view_size, NUM_CHANNELS), jnp.float32, 0.0, 1.0),
            "agent_dir": Space((), jnp.int32, 0, 3),
        }

    def action_space(self) -> Space:
        return Space((), jnp.int32, 0, NUM_ACTIONS - 1)

    def time_limit(self) -> int:
        return self._time_limit

    def field_of_view(self, state: MazeState) -> Array:
        radius = self.view_size // 2
        window = spread(sample(state.level.wall, state.position + self.offsets[0], 1.0) == 0)
        height, width = state.level.wall.shape
        rows = jnp.arange(height)[:, None] - state.position[0]
        columns = jnp.arange(width)[None, :] - state.position[1]
        inside = (jnp.abs(rows) <= radius) & (jnp.abs(columns) <= radius)
        seen = window[jnp.clip(rows + radius, 0, 2 * radius), jnp.clip(columns + radius, 0, 2 * radius)]
        return inside & seen

    def render(self, state: MazeState, tile: int = TILE, **kwargs) -> np.ndarray:
        wall = np.asarray(state.level.wall) > 0
        rows, columns = wall.shape
        (row, column), (goal_row, goal_column) = np.asarray(state.position), np.asarray(state.level.goal)
        cells = np.broadcast_to(empty_tile(tile), (rows, columns, tile, tile, 3)).copy()
        cells[wall] = GREY
        cells[goal_row, goal_column] = GREEN
        coverage = agent_tile(tile, int(state.direction))[..., None]
        cells[row, column] = cells[row, column] * (1.0 - coverage) + RED * coverage
        seen = np.asarray(self.field_of_view(state))
        cells[seen] += HIGHLIGHT * (WHITE - cells[seen])
        image = cells.transpose(0, 2, 1, 3, 4).reshape(rows * tile, columns * tile, 3)
        return np.round(image).astype(np.uint8)


@functools.cache
def empty_tile(tile: int) -> np.ndarray:
    cell = np.zeros((tile, tile, 3), np.float32)
    line = max(1, round(GRID_LINE * tile))
    cell[:line] = GREY
    cell[:, :line] = GREY
    return cell


@functools.cache
def agent_tile(tile: int, direction: int) -> np.ndarray:
    size = tile * SUBDIVISIONS
    y, x = (np.mgrid[0:size, 0:size] + 0.5) / size - 0.5
    angle = -0.5 * np.pi * direction
    x, y = 0.5 + x * np.cos(angle) - y * np.sin(angle), 0.5 + y * np.cos(angle) + x * np.sin(angle)
    (ax, ay), (bx, by), (cx, cy) = AGENT_TRIANGLE
    denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    first = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / denominator
    second = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / denominator
    inside = (first >= 0) & (second >= 0) & (first + second <= 1)
    return inside.reshape(tile, SUBDIVISIONS, tile, SUBDIVISIONS).mean((1, 3)).astype(np.float32)
