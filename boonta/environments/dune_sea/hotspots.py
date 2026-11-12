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

    def advantages(self, policy: Array) -> Array:
        assert self.control == 0.0, "actions matter only on hotspots while the agent cannot steer"
        payoff = jnp.zeros_like(policy).at[self.hotspots, self.secrets].set(1.0)
        return payoff - jnp.sum(policy * payoff, axis=-1, keepdims=True)

    def transitions(self, policy: Array) -> Array:
        assert self.control == 0.0, "the walk is the policy's only while the agent cannot steer"
        walk = jnp.asarray(random_walk(self), jnp.float32)
        return walk.at[self.hotspots].set(0.0).at[self.hotspots, self.start].set(1.0)

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
        """The floor is shaded by how often the walk from the start visits each cell, bright near
        and dark where it is rare. Hotspots are gold discs whose arrow points the way of their
        secret, the start has a blue outline, the agent is a red ring, and a bar along the top
        wall fills up as the episode runs out of time.
        """
        span = self.width + 2
        pixels = (jnp.arange(span * tile) + 0.5) / tile
        rows, columns = jnp.meshgrid(pixels, pixels, indexing="ij")
        row, column = jnp.floor(rows).astype(jnp.int32), jnp.floor(columns).astype(jnp.int32)
        across, down = columns - column - 0.5, rows - row - 0.5
        wall = (row == 0) | (row == span - 1) | (column == 0) | (column == span - 1)
        inside = jnp.clip(row - 1, 0, self.width - 1) * self.width + jnp.clip(column - 1, 0, self.width - 1)
        radius = jnp.hypot(across, down)

        shade = jnp.asarray(visit_shade(self), jnp.float32)[inside][..., None]
        colour = (1.0 - shade) * jnp.array([22.0, 26.0, 44.0]) + shade * jnp.array([226.0, 232.0, 242.0])
        line = 0.5 - 1.0 / tile
        colour = jnp.where(((jnp.abs(across) > line) | (jnp.abs(down) > line))[..., None], 0.8 * colour, colour)

        start = (inside == self.start) & ((jnp.abs(across) > 0.36) | (jnp.abs(down) > 0.36))
        colour = jnp.where((~wall & start)[..., None], jnp.array([40.0, 110.0, 230.0]), colour)

        index = self.hotspot_index[inside]
        hotspot = ~wall & (index >= 0)
        move = jnp.asarray(MOVES, jnp.float32)[self.secrets[jnp.maximum(index, 0)]]
        along = jnp.clip(down * move[..., 0] + across * move[..., 1], -0.2, 0.2)
        shaft = jnp.hypot(down - along * move[..., 0], across - along * move[..., 1]) < 0.05
        forward = down * move[..., 0] + across * move[..., 1]
        sideways = jnp.abs(down * move[..., 1] - across * move[..., 0])
        head = (forward > 0.1) & (forward < 0.3) & (sideways < 0.3 - forward)
        colour = jnp.where((hotspot & (radius < 0.4))[..., None], jnp.array([240.0, 190.0, 40.0]), colour)
        colour = jnp.where((hotspot & (radius < 0.4) & (radius > 0.34))[..., None], jnp.array([150.0, 105.0, 15.0]), colour)
        colour = jnp.where((hotspot & (shaft | head))[..., None], jnp.array([70.0, 45.0, 5.0]), colour)

        agent = ~wall & (inside == state.position) & (radius > 0.38) & (radius < 0.49)
        colour = jnp.where(agent[..., None], jnp.array([235.0, 30.0, 30.0]), colour)

        elapsed = jnp.clip(state.clock / self.horizon, 0.0, 1.0)
        bar = (row == 0) & (jnp.abs(down) < 0.22) & (columns > 1.0) & (columns < span - 1.0)
        filled = bar & ((columns - 1.0) / (span - 2.0) < elapsed)
        colour = jnp.where(wall[..., None], jnp.full(3, 100.0), colour)
        colour = jnp.where(bar[..., None], jnp.full(3, 60.0), colour)
        colour = jnp.where(filled[..., None], jnp.array([235.0, 160.0, 40.0]), colour)
        return colour.astype(jnp.uint8)


def visits(env) -> np.ndarray:
    """Expected visits to each cell by an episode of the random walk from the start."""
    game = getattr(env, "unwrapped", env)
    walk = random_walk(game)
    walk[np.asarray(game.hotspots)] = 0.0
    occupancy = np.zeros(game.num_cells)
    occupancy[game.start] = 1.0
    total = occupancy.copy()
    for _ in range(game.horizon - 1):
        occupancy = occupancy @ walk
        total += occupancy
    return total


def visit_shade(env) -> np.ndarray:
    """Visits on a log scale from 0, the rarest cell, to 1, the most visited."""
    game = getattr(env, "unwrapped", env)
    if getattr(game, "_shade", None) is None:
        logged = np.log(np.maximum(visits(game), 1e-12))
        game._shade = (logged - logged.min()) / max(logged.max() - logged.min(), 1e-12)
    return game._shade


def random_walk(game) -> np.ndarray:
    width, num_cells = game.width, game.num_cells
    rows, columns = np.divmod(np.arange(num_cells), width)
    walk = np.zeros((num_cells, num_cells))
    for row_move, column_move in MOVES:
        landing = (
            np.clip(rows + row_move, 0, width - 1) * width
            + np.clip(columns + column_move, 0, width - 1)
        )
        np.add.at(walk, (np.arange(num_cells), landing), 1.0 / len(MOVES))
    return walk


def reach(env) -> np.ndarray:
    game = getattr(env, "unwrapped", env)
    num_cells, hotspots = game.num_cells, np.asarray(game.hotspots)
    walk = random_walk(game)
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
