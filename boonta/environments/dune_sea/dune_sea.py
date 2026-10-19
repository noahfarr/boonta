import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from boonta.utils import Array, Key, Timestep

from ..environment import Environment
from ..spaces import Space
from .sprites import wanderer_sprites
from .tileset import (TILE_PIXELS, anchor_sprite, draw_objects, load_tileset, paint_scenery,
                      tileset_available)

NOOP, FORWARD, BACKWARD, STEP_LEFT, STEP_RIGHT, TURN_LEFT, TURN_RIGHT = range(7)
NUM_ACTIONS = 7
STEP_X = np.array([0, 1, 0, -1])
STEP_Y = np.array([-1, 0, 1, 0])
MOVE_HEADING = np.array([-1, 0, 2, 3, 1, -1, -1])
TURN_OFFSET = np.array([0, 0, 0, 0, 0, 3, 1])
VIEW_RADIUS = 3
VIEW_SIZE = 2 * VIEW_RADIUS + 1
SAND, ROCK, FLORA, WATER = 0, 1, 2, 3
FLORA_COLOR = np.array([96, 128, 64], np.uint8)
WATER_FILL_COLOR = np.array([95, 132, 128], np.uint8)
OASIS_REACH = 3
OASIS_CLEARANCE = 3
OASIS_PALM_CHANCE = 0.18
REED_CHANCE = 0.35
PATCH_FILL = 0.42
CAVE_THRESHOLD = 5
FLOOD_SEEDS = 4
GROUND_COLOR = np.array([233, 214, 175], np.uint8)
ROCK_COLOR = np.array([120, 82, 58], np.uint8)
HIGHLIGHT_COLOR = np.array([231, 111, 81], np.uint8)


def view_offsets() -> np.ndarray:
    cells_ahead = VIEW_RADIUS - np.arange(VIEW_SIZE)[:, None]
    cells_right = np.arange(VIEW_SIZE)[None, :] - VIEW_RADIUS
    offsets = np.zeros((4, VIEW_SIZE, VIEW_SIZE, 2), np.int32)
    for facing in range(4):
        right_side = (facing + 1) % 4
        offsets[facing, ..., 0] = (
            cells_ahead * STEP_X[facing] + cells_right * STEP_X[right_side]
        )
        offsets[facing, ..., 1] = (
            cells_ahead * STEP_Y[facing] + cells_right * STEP_Y[right_side]
        )
    return offsets


def count_neighbours(grid: Array) -> Array:
    return sum(
        jnp.roll(grid, (dy, dx), axis=(0, 1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
    )


def carve_caves(key: Key, height: int, width: int, rock_fill: float, smooth_passes: int) -> Array:
    rock = jax.random.bernoulli(key, rock_fill, (height, width)).astype(jnp.int32)

    def smooth_rock(_, rock):
        return (count_neighbours(rock) >= CAVE_THRESHOLD).astype(jnp.int32)

    return jax.lax.fori_loop(0, smooth_passes, smooth_rock, rock)


def flood_region(open_mask: Array, seed_cell: Array) -> Array:
    height, width = open_mask.shape
    reached = jnp.zeros_like(open_mask).at[seed_cell // width, seed_cell % width].set(True)
    reached = reached & open_mask

    def grow_region(carry):
        reached, _ = carry
        grown = (
            reached
            | jnp.roll(reached, 1, axis=0)
            | jnp.roll(reached, -1, axis=0)
            | jnp.roll(reached, 1, axis=1)
            | jnp.roll(reached, -1, axis=1)
        ) & open_mask
        return grown, jnp.any(grown != reached)

    reached, _ = jax.lax.while_loop(lambda carry: carry[1], grow_region, (reached, True))
    return reached


def connect_caves(key: Key, cells: Array, min_open: int) -> Array:
    open_mask = cells == SAND
    weights = open_mask.ravel() / jnp.maximum(open_mask.sum(), 1)
    seeds = jax.random.choice(key, open_mask.size, (FLOOD_SEEDS,), p=weights)
    regions = jax.vmap(lambda seed: flood_region(open_mask, seed))(seeds)
    largest = regions[jnp.argmax(regions.sum(axis=(1, 2)))]
    connected = jnp.where(largest | ~open_mask, cells, ROCK)
    return jnp.where(largest.sum() >= min_open, connected, jnp.full_like(cells, SAND))


def spread_mask(mask: Array, steps: int) -> Array:
    def grow(_, mask):
        return mask | jnp.roll(mask, 1, 0) | jnp.roll(mask, -1, 0) | jnp.roll(mask, 1, 1) | jnp.roll(mask, -1, 1)

    return jax.lax.fori_loop(0, steps, grow, mask)


def dig_oases(key: Key, cells: Array, num_oases: int, max_radius: float) -> Array:
    if num_oases == 0:
        return cells
    height, width = cells.shape
    centre_key, size_key, edge_key = jax.random.split(key, 3)
    is_rock = cells == ROCK
    is_clear = ~spread_mask(is_rock, OASIS_CLEARANCE)
    weights = is_clear.ravel() / jnp.maximum(is_clear.sum(), 1)
    centres = jax.random.choice(centre_key, height * width, (num_oases,), replace=False, p=weights)
    centre_rows = (centres // width)[:, None, None]
    centre_cols = (centres % width)[:, None, None]
    radii = jax.random.uniform(size_key, (num_oases, 1, 1), minval=1.5, maxval=max_radius)
    rows = jnp.arange(height)[None, :, None]
    cols = jnp.arange(width)[None, None, :]
    row_gap = jnp.minimum(jnp.abs(rows - centre_rows), height - jnp.abs(rows - centre_rows))
    col_gap = jnp.minimum(jnp.abs(cols - centre_cols), width - jnp.abs(cols - centre_cols))
    wobble = jax.random.uniform(edge_key, (height, width), minval=-0.7, maxval=0.7)
    is_pond = (jnp.sqrt(row_gap**2 + col_gap**2) + wobble < radii).any(axis=0)
    is_water = is_pond & ~spread_mask(is_rock, 1) & (cells == SAND) & is_clear.any()
    return jnp.where(is_water, WATER, cells)


def scatter_flora(key: Key, cells: Array, flora_fill: float) -> Array:
    cluster_key, plant_key = jax.random.split(key)
    clusters = carve_caves(cluster_key, *cells.shape, 0.5, 2).astype(jnp.float32)
    is_water = cells == WATER
    shoreline = spread_mask(is_water, 1)
    near_water = spread_mask(is_water, OASIS_REACH) & ~shoreline
    chance = flora_fill * (0.4 + 1.2 * clusters) + OASIS_PALM_CHANCE * near_water
    planted = jax.random.bernoulli(plant_key, chance) & (cells == SAND) & ~shoreline
    return jnp.where(planted, FLORA, cells)


def scenery_layers(scenery_seed: Array, height: int, width: int) -> tuple[Array, Array]:
    key = jax.random.wrap_key_data(scenery_seed)
    patch_key, variety_key = jax.random.split(key)
    is_patch = carve_caves(patch_key, height, width, PATCH_FILL, 4) == 1
    variety = jax.random.bits(variety_key, (height, width), jnp.uint32)
    return is_patch, variety


@struct.dataclass
class DuneSeaState:
    cells: Array
    scenery_seed: Array
    position: Array
    direction: Array
    time: Array


class DuneSea(Environment):
    def __init__(
        self,
        width: int = 16,
        height: int = 16,
        num_agents: int = 4,
        time_limit: int = 256,
        rock_fill: float = 0.45,
        smooth_passes: int = 4,
        flora_fill: float = 0.03,
        oasis_density: float = 0.0012,
        oasis_radius: float = 3.5,
    ):
        if num_agents > width * height:
            raise ValueError("dune_sea: more agents than cells")
        self.width, self.height = width, height
        self._num_agents = num_agents
        self.view_table = view_offsets()
        self._time_limit = time_limit
        self.rock_fill = rock_fill
        self.smooth_passes = smooth_passes
        self.flora_fill = flora_fill
        self.num_oases = int(round(oasis_density * width * height))
        self.oasis_radius = oasis_radius
        self.agent_sprites = wanderer_sprites()

    @property
    def num_agents(self) -> int:
        return self._num_agents

    def generate_map(self, key: Key) -> Array:
        cave_key, oasis_key, flora_key, flood_key = jax.random.split(key, 4)
        cells = carve_caves(cave_key, self.height, self.width, self.rock_fill, self.smooth_passes)
        cells = dig_oases(oasis_key, cells, self.num_oases, self.oasis_radius)
        cells = scatter_flora(flora_key, cells, self.flora_fill)
        return connect_caves(flood_key, cells, self._num_agents)

    def init(self, key: Key) -> tuple[DuneSeaState, Timestep]:
        map_key, place_key, face_key, scenery_key = jax.random.split(key, 4)
        cells = self.generate_map(map_key)
        open_cells = (cells == SAND).ravel()
        start_cells = jax.random.choice(
            place_key,
            self.width * self.height,
            (self._num_agents,),
            replace=False,
            p=open_cells / open_cells.sum(),
        )
        position = jnp.stack(
            [start_cells % self.width, start_cells // self.width], axis=-1
        )
        state = DuneSeaState(
            cells=cells,
            scenery_seed=jax.random.key_data(scenery_key),
            position=position.astype(jnp.int32),
            direction=jax.random.randint(face_key, (self._num_agents,), 0, 4),
            time=jnp.int32(0),
        )
        return state, Timestep(
            obs=self.observe(state),
            action=jnp.zeros((self._num_agents,), jnp.int32),
            reward=jnp.zeros((self._num_agents,), jnp.float32),
            terminated=jnp.ones((self._num_agents,), bool),
            truncated=jnp.zeros((self._num_agents,), bool),
            info={},
        )

    def move_agents(self, cells: Array, position: Array, heading: Array) -> Array:
        num_cells = self.width * self.height
        is_moving = heading >= 0
        bearing = jnp.maximum(heading, 0)
        step = jnp.stack(
            [jnp.asarray(STEP_X)[bearing], jnp.asarray(STEP_Y)[bearing]], axis=-1
        )
        target = (position + step) % jnp.asarray([self.width, self.height])
        is_moving = is_moving & (cells[target[:, 1], target[:, 0]] == SAND)

        current_cells = position[:, 1] * self.width + position[:, 0]
        target_cells = target[:, 1] * self.width + target[:, 0]
        target_claims = (
            jnp.zeros(num_cells, jnp.int32)
            .at[target_cells]
            .add(is_moving.astype(jnp.int32))
        )
        is_moving = is_moving & (target_claims[target_cells] == 1)
        is_swap = (
            (target_cells[:, None] == current_cells[None, :])
            & (target_cells[None, :] == current_cells[:, None])
            & is_moving[:, None]
            & is_moving[None, :]
        )
        is_moving = is_moving & ~is_swap.any(axis=1)

        def block_stayers(_, is_moving):
            stayer_counts = (
                jnp.zeros(num_cells, jnp.int32)
                .at[current_cells]
                .add((~is_moving).astype(jnp.int32))
            )
            return is_moving & (stayer_counts[target_cells] == 0)

        is_moving = jax.lax.fori_loop(0, self._num_agents, block_stayers, is_moving)
        return jnp.where(is_moving[:, None], target, position)

    def step(
        self, key: Key, state: DuneSeaState, action: Array
    ) -> tuple[DuneSeaState, Timestep]:
        action = jnp.clip(jnp.asarray(action, jnp.int32), 0, NUM_ACTIONS - 1)
        direction = (state.direction + jnp.asarray(TURN_OFFSET)[action]) % 4
        heading_offset = jnp.asarray(MOVE_HEADING)[action]
        heading = jnp.where(heading_offset >= 0, (direction + heading_offset) % 4, -1)
        time = state.time + 1
        state = DuneSeaState(
            cells=state.cells,
            scenery_seed=state.scenery_seed,
            position=self.move_agents(state.cells, state.position, heading),
            direction=direction,
            time=time,
        )
        n = self._num_agents
        return state, Timestep(
            obs=self.observe(state),
            action=action,
            reward=jnp.zeros((n,), jnp.float32),
            terminated=jnp.zeros((n,), bool),
            truncated=jnp.broadcast_to(time >= self._time_limit, (n,)),
            info={},
        )

    def observe(self, state: DuneSeaState) -> Array:
        agent_x, agent_y = state.position[:, 0], state.position[:, 1]
        occupancy = jnp.zeros((self.height, self.width)).at[agent_y, agent_x].add(1.0)
        offsets = jnp.asarray(self.view_table)[state.direction]
        view_x = (agent_x[:, None, None] + offsets[..., 0]) % self.width
        view_y = (agent_y[:, None, None] + offsets[..., 1]) % self.height
        cells = state.cells[view_y, view_x]
        rock = (cells == ROCK).astype(jnp.float32)
        flora = (cells == FLORA).astype(jnp.float32)
        water = (cells == WATER).astype(jnp.float32)
        return jnp.stack([occupancy[view_y, view_x], rock, flora, water], axis=-1)

    def observation_space(self) -> Space:
        return Space(
            shape=(self._num_agents, VIEW_SIZE, VIEW_SIZE, 4),
            dtype=jnp.float32,
            low=0.0,
            high=1.0,
        )

    def action_space(self) -> Space:
        return Space(
            shape=(self._num_agents,), dtype=jnp.int32, low=0, high=NUM_ACTIONS - 1
        )

    def time_limit(self) -> int:
        return self._time_limit

    def paint_ground(self, state: DuneSeaState) -> tuple[Array, Array]:
        is_patch, variety = scenery_layers(state.scenery_seed, self.height, self.width)
        if tileset_available():
            return paint_scenery(state.cells, is_patch, variety, load_tileset()), variety
        rows = jnp.arange(TILE_PIXELS)[:, None]
        cols = jnp.arange(TILE_PIXELS)[None, :]
        edge = (rows == 0) | (cols == 0)
        cell_colors = jnp.stack([GROUND_COLOR, ROCK_COLOR, FLORA_COLOR, WATER_FILL_COLOR]).astype(jnp.float32)
        tiles = cell_colors[state.cells][:, :, None, None, :]
        edge_shade = jnp.where(edge, 0.93, 1.0)[None, None, :, :, None]
        tiles = jnp.broadcast_to(tiles * edge_shade, (self.height, self.width, TILE_PIXELS, TILE_PIXELS, 3))
        frame = tiles.transpose(0, 2, 1, 3, 4).reshape(self.height * TILE_PIXELS, self.width * TILE_PIXELS, 3)
        return frame, variety

    def sprite_bank(self) -> tuple[Array, int, int]:
        agents = self.agent_sprites.reshape(-1, *self.agent_sprites.shape[2:])
        if not tileset_available():
            return jnp.asarray(agents), 0, 0
        tileset = load_tileset()
        anchored = jnp.asarray(np.stack([anchor_sprite(sprite) for sprite in agents]))
        bank = jnp.concatenate([tileset["flora"], tileset["reeds"], anchored])
        return bank, tileset["flora"].shape[0], tileset["reeds"].shape[0]

    def render(self, state: DuneSeaState, scale: int = TILE_PIXELS) -> Array:
        frame, variety = self.paint_ground(state)
        bank, num_flora, num_reeds = self.sprite_bank()
        num_cells = self.width * self.height
        num_poses = self.agent_sprites.shape[1]
        robes = jnp.arange(self._num_agents) % self.agent_sprites.shape[0]
        agent_ids = num_flora + num_reeds + robes * num_poses + state.direction
        agent_rows, agent_cols = state.position[:, 1], state.position[:, 0]
        rows, cols, ids = agent_rows, agent_cols, agent_ids
        present = jnp.ones(self._num_agents, bool)
        if num_flora:
            cell_rows = jnp.arange(num_cells) // self.width
            cell_cols = jnp.arange(num_cells) % self.width
            flora_ids = ((variety.ravel() >> 20) % num_flora).astype(jnp.int32)
            rows = jnp.concatenate([cell_rows, rows])
            cols = jnp.concatenate([cell_cols, cols])
            ids = jnp.concatenate([flora_ids, ids])
            present = jnp.concatenate([(state.cells == FLORA).ravel(), present])
        if num_reeds:
            shore = spread_mask(state.cells == WATER, 1) & (state.cells == SAND)
            reed_roll = ((variety >> 16) % 1000).astype(jnp.float32) / 1000.0
            has_reed = (shore & (reed_roll < REED_CHANCE)).ravel()
            reed_ids = num_flora + ((variety.ravel() >> 24) % num_reeds).astype(jnp.int32)
            rows = jnp.concatenate([cell_rows, rows])
            cols = jnp.concatenate([cell_cols, cols])
            ids = jnp.concatenate([reed_ids, ids])
            present = jnp.concatenate([has_reed, present])
        order = jnp.argsort(jnp.where(present, rows, num_cells + 1), stable=True)
        frame = draw_objects(frame, rows[order], cols[order], ids[order], present.sum(), bank)
        if scale != TILE_PIXELS:
            size = (self.height * scale, self.width * scale, 3)
            frame = jax.image.resize(frame, size, "nearest")
        return jnp.clip(frame, 0, 255).astype(jnp.uint8)
