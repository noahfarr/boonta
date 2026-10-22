import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from boonta.environments.environment import Environment
from boonta.environments.spaces import Space
from boonta.utils import Array, Key, Timestep

MOVES = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)], np.int32)


@struct.dataclass
class HotspotsState:
    position: Array
    clock: Array


def layout(width: int, num_hotspots: int, nearest: int, farthest: int, seed: int):
    rng = np.random.default_rng(seed)
    start = (width // 2, width // 2)
    rows, columns = np.divmod(np.arange(width * width), width)
    distances = np.abs(rows - start[0]) + np.abs(columns - start[1])
    targets = np.round(np.linspace(nearest, farthest, num_hotspots)).astype(int)
    hotspots = []
    allowed = (distances >= nearest) & (distances <= farthest)
    assert num_hotspots <= allowed.sum(), (
        f"only {allowed.sum()} cells lie {nearest} to {farthest} steps from the start "
        f"of a {width}x{width} grid, too few for {num_hotspots} hotspots"
    )
    for target in targets:
        free = allowed & ~np.isin(np.arange(width * width), hotspots)
        gaps = np.where(free, np.abs(distances - target), np.inf)
        hotspots.append(rng.choice(np.flatnonzero(gaps == gaps.min())))
    secrets = rng.integers(0, len(MOVES), num_hotspots)
    return start[0] * width + start[1], np.array(hotspots, np.int32), secrets.astype(np.int32)


def cell(env):
    return lambda env_state: env_state.position


class Hotspots(Environment):
    def __init__(
        self,
        width: int = 15,
        num_hotspots: int = 8,
        nearest: int = 2,
        farthest: int = 12,
        control: float = 0.0,
        horizon: int = 200,
        seed: int = 0,
    ):
        assert 1 <= nearest <= farthest <= 2 * (width // 2)
        start, hotspots, secrets = layout(width, num_hotspots, nearest, farthest, seed)
        self.width = width
        self.num_cells = width * width
        self.control = control
        self.horizon = horizon
        self.start = start
        self.hotspots = jnp.asarray(hotspots)
        self.secrets = jnp.asarray(secrets)
        self.hotspot_index = jnp.full(self.num_cells, -1, jnp.int32).at[self.hotspots].set(
            jnp.arange(num_hotspots, dtype=jnp.int32)
        )
        cell_grid = np.full((width + 2, width + 2), -1, np.int32)
        cell_grid[1:-1, 1:-1] = np.arange(self.num_cells).reshape(width, width)
        self.cell_grid = jnp.asarray(cell_grid)

    def observation_space(self) -> Space:
        return Space((self.num_cells,), jnp.float32, 0.0, 1.0)

    def action_space(self) -> Space:
        return Space((), jnp.int32, 0, len(MOVES) - 1)

    def time_limit(self) -> int:
        return self.horizon

    def expected_return(self, policy: Array) -> Array:
        assert self.control == 0.0, "the random walk's reach is the policy's only while it cannot steer"
        return jnp.sum(jnp.asarray(reach(self)) * policy[self.hotspots, self.secrets])

    def place(self, state: HotspotsState, cell: Array) -> HotspotsState:
        return state.replace(position=jnp.int32(cell), clock=jnp.zeros_like(state.clock))

    def observe(self, state: HotspotsState) -> Array:
        return jax.nn.one_hot(state.position, self.num_cells)

    def init(self, key: Key) -> tuple[HotspotsState, Timestep]:
        state = HotspotsState(position=jnp.int32(self.start), clock=jnp.int32(0))
        return state, Timestep(
            obs=self.observe(state),
            action=jnp.int32(0),
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(True),
            truncated=jnp.bool_(False),
        )

    def step(self, key: Key, state: HotspotsState, action: Array) -> tuple[HotspotsState, Timestep]:
        slip_key, direction_key = jax.random.split(key)
        hotspot = self.hotspot_index[state.position]
        guessed = hotspot >= 0
        controlled = jax.random.bernoulli(slip_key, self.control)
        direction = jnp.where(
            controlled, action, jax.random.randint(direction_key, (), 0, len(MOVES))
        )
        row, column = jnp.divmod(state.position, self.width)
        row = jnp.clip(row + jnp.asarray(MOVES)[direction, 0], 0, self.width - 1)
        column = jnp.clip(column + jnp.asarray(MOVES)[direction, 1], 0, self.width - 1)
        position = jnp.where(guessed, state.position, row * self.width + column)
        reward = guessed & (action == self.secrets[jnp.maximum(hotspot, 0)])
        clock = state.clock + 1
        state = HotspotsState(position=position, clock=clock)
        return state, Timestep(
            obs=self.observe(state),
            action=action,
            reward=reward.astype(jnp.float32),
            terminated=guessed,
            truncated=~guessed & (clock >= self.horizon),
        )


    def render(self, state: HotspotsState, tile: int = 32) -> Array:
        span = self.width + 2
        pixels = (jnp.arange(span * tile) + 0.5) / tile
        rows, columns = jnp.meshgrid(pixels, pixels, indexing="ij")
        row, column = jnp.floor(rows).astype(jnp.int32), jnp.floor(columns).astype(jnp.int32)
        down, across = rows - row, columns - column
        wall = (row == 0) | (row == span - 1) | (column == 0) | (column == span - 1)
        inside = jnp.clip(row - 1, 0, self.width - 1) * self.width + jnp.clip(column - 1, 0, self.width - 1)
        hotspot = ~wall & (self.hotspot_index[inside] >= 0)
        line = 1.0 / tile
        edge = (down < line) | (across < line) | (down > 1 - line) | (across > 1 - line)
        agent_row, agent_column = jnp.divmod(state.position, self.width)
        occupied = (row == agent_row + 1) & (column == agent_column + 1)
        triangle = (
            (across > 0.12)
            & (across < 0.87)
            & (jnp.abs(down - 0.5) < 0.31 * (0.87 - across) / 0.75)
        )
        colour = jnp.zeros((*rows.shape, 3))
        colour = jnp.where(hotspot[..., None], jnp.array([0.0, 255.0, 0.0]), colour)
        colour = jnp.where((edge & ~wall)[..., None], jnp.array([100.0, 100.0, 100.0]), colour)
        colour = jnp.where(wall[..., None], jnp.array([100.0, 100.0, 100.0]), colour)
        colour = jnp.where((occupied & triangle)[..., None], jnp.array([255.0, 0.0, 0.0]), colour)
        return colour.astype(jnp.uint8)


def reach(env) -> np.ndarray:
    game = getattr(env, "unwrapped", env)
    width, num_cells = game.width, game.num_cells
    hotspots = np.asarray(game.hotspots)
    rows, columns = np.divmod(np.arange(num_cells), width)
    walk = np.zeros((num_cells, num_cells))
    for row_move, column_move in MOVES:
        landing = (
            np.clip(rows + row_move, 0, width - 1) * width
            + np.clip(columns + column_move, 0, width - 1)
        )
        np.add.at(walk, (np.arange(num_cells), landing), 1.0 / len(MOVES))
    walk[hotspots] = 0.0
    occupancy = np.zeros(num_cells)
    occupancy[game.start] = 1.0
    arrived = np.zeros(len(hotspots))
    for _ in range(game.horizon - 1):
        occupancy = occupancy @ walk
        arrived += occupancy[hotspots]
        occupancy[hotspots] = 0.0
    return arrived


class Scorecard:
    def __init__(self, name: str = "hotspots"):
        self.name = name
        self.reached = {}

    def craft(self, key, podracer, podracer_state, logs):
        algorithm, environment = podracer.algorithm, podracer.environment
        from boonta.artisans.artifact import Metrics

        game = getattr(environment, "unwrapped", environment)
        if id(game) not in self.reached:
            self.reached[id(game)] = jnp.asarray(reach(game))
        reached = self.reached[id(game)]
        num_hotspots, *_ = jnp.shape(game.hotspots)
        empty = jnp.zeros(num_hotspots)
        timestep = Timestep(
            obs=jax.nn.one_hot(game.hotspots, game.num_cells),
            action=empty.astype(jnp.int32),
            reward=empty,
            terminated=empty.astype(bool),
            truncated=empty.astype(bool),
        )
        _, _, _, aux = algorithm.step(podracer_state.algorithm_state, None, key, timestep, temperature=1.0)
        logits = aux["logits"]
        correct = jnp.take_along_axis(jax.nn.softmax(logits), game.secrets[:, None], axis=-1)[:, 0]
        solved = jnp.argmax(logits, axis=-1) == game.secrets
        data = {
            f"{self.name}/return": jnp.sum(reached * correct),
            f"{self.name}/greedy_return": jnp.sum(reached * solved),
            f"{self.name}/best_return": jnp.sum(reached),
        }
        data |= {f"{self.name}/correct_{index}": correct[index] for index in range(num_hotspots)}
        return Metrics(self.name, jax.device_get(data))
