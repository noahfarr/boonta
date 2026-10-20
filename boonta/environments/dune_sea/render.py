from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from boonta.utils import Array

from .sprites import team_sprites, wanderer_sprites
from .state import MATURE_ENERGY, MAX_ENERGY

if TYPE_CHECKING:
    from .state import DuneSeaState

DIRECTORY = Path(__file__).parent / "assets" / "glionox_desert"
TILE = 16
BROADCAST_COLOR = jnp.array([255, 255, 255], dtype=jnp.float32)
FILL_ROWS = (12, 13, 14)
SHEET_COLUMNS = 10
GROUND_SHEET = "sand.png"
MOUNTAIN_SHEET = "dry_dirt.png"
OVERHANG_ROWS = 6
DETAIL_OFFSET = 6
FULL_MASK = 255
NEIGHBOUR_STEPS = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
REGION_SLICES = (
    (slice(0, 2), slice(6, 10)),
    (slice(0, 2), slice(14, 16)),
    (slice(6, 10), slice(14, 16)),
    (slice(14, 16), slice(14, 16)),
    (slice(14, 16), slice(6, 10)),
    (slice(14, 16), slice(0, 2)),
    (slice(6, 10), slice(0, 2)),
    (slice(0, 2), slice(0, 2)),
)
ENERGY_SPRITE = "decor_cactus_small1.png"
RIPE_SPRITE = "decor_cactus_small1_pink.png"
GOAL_SPRITE = "decor_cactus1_pink.png"
ROW_STRIDE, COLUMN_STRIDE = 7, 13
POSE_OF_DIRECTION = jnp.array([1, 2, 3, 0])
TEAM_COLORS = jnp.array(
    [[60, 120, 255], [230, 60, 50], [60, 190, 90], [240, 190, 40]], dtype=jnp.float32
)
BAR_INNER = 12
BAR_HEIGHT = 4
BAR_FULL = jnp.array([90, 200, 80], dtype=jnp.float32)
BAR_HALF = jnp.array([235, 190, 50], dtype=jnp.float32)
BAR_LOW = jnp.array([215, 60, 50], dtype=jnp.float32)
BAR_EMPTY = jnp.array([60, 50, 45], dtype=jnp.float32)


def read_image(name: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(DIRECTORY / name).convert("RGBA"))


@cache
def fill_tiles(sheet: str) -> np.ndarray:
    image = read_image(sheet)
    rows = [
        image[row * TILE : (row + 1) * TILE, col * TILE : (col + 1) * TILE]
        for row in FILL_ROWS
        for col in range(SHEET_COLUMNS)
    ]
    return np.stack(rows)[..., :3]


@cache
def sprite_tiles(name: str) -> np.ndarray:
    image = read_image(name)
    num_tiles = image.shape[0] // TILE
    tiles = [image[k * TILE : (k + 1) * TILE] for k in range(num_tiles)]
    return np.stack(tiles[::-1]).astype(np.float32)


def cut_tile(sheet: np.ndarray, row: int, col: int) -> np.ndarray:
    return sheet[row * TILE : (row + 1) * TILE, col * TILE : (col + 1) * TILE]


def stack_layers(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    lower_alpha, upper_alpha = lower[..., 3:] / 255.0, upper[..., 3:] / 255.0
    alpha = upper_alpha + lower_alpha * (1.0 - upper_alpha)
    color = upper[..., :3] * upper_alpha + lower[..., :3] * lower_alpha * (1.0 - upper_alpha)
    color = np.where(alpha > 0, color / np.maximum(alpha, 1e-6), 0.0)
    return np.concatenate([color, alpha * 255.0], axis=-1)


def visual_mask(tile: np.ndarray) -> int:
    covered = tile[..., 3] > 0
    return sum(1 << bit for bit, region in enumerate(REGION_SLICES) if covered[region].mean() > 0.5)


def reachable_masks() -> list[int]:
    masks = set()
    for raw in range(256):
        mask = raw
        for bit in (1, 3, 5, 7):
            if raw & (1 << (bit - 1)) or raw & (1 << ((bit + 1) % 8)):
                mask |= 1 << bit
        masks.add(mask)
    return sorted(masks)


@cache
def mountain_edges(sheet: str) -> np.ndarray:
    image = read_image(sheet).astype(np.float32)
    table = np.zeros((256, TILE, TILE, 4), np.float32)
    filled = set()
    for row in range(OVERHANG_ROWS):
        for col in range(SHEET_COLUMNS):
            tile = stack_layers(cut_tile(image, row + DETAIL_OFFSET, col), cut_tile(image, row, col))
            mask = visual_mask(tile)
            centre_covered = (tile[6:10, 6:10, 3] > 0).mean() > 0.9
            if mask == FULL_MASK and centre_covered:
                continue
            table[mask] = tile
            filled.add(mask)
    for mask in reachable_masks():
        if mask and mask not in filled:
            nearest = min(filled, key=lambda other: (bin(mask ^ other).count("1"), other))
            table[mask] = table[nearest]
    return table


def neighbour_mask(wall: Array) -> Array:
    height, width = wall.shape
    padded = jnp.pad(wall, 1)
    flags = [padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width] for dx, dy in NEIGHBOUR_STEPS]
    mask = jnp.zeros(wall.shape, jnp.int32)
    for bit, flag in enumerate(flags):
        if bit % 2 == 1:
            flag = flag | flags[bit - 1] | flags[(bit + 1) % 8]
        mask = mask | (flag.astype(jnp.int32) << bit)
    return mask


def blend_over(base: Array, layer: Array) -> Array:
    alpha = layer[..., 3:] / 255.0
    return base * (1.0 - alpha) + layer[..., :3] * alpha


def choose_fill(tiles: Array, height: int, width: int) -> Array:
    rows, columns = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    return jnp.asarray(tiles)[(rows * ROW_STRIDE + columns * COLUMN_STRIDE) % tiles.shape[0]]


def to_image(cells: Array) -> Array:
    height, width, rows, columns, channels = cells.shape
    return cells.transpose(0, 2, 1, 3, 4).reshape(height * rows, width * columns, channels)


def paint_sprite(image: Array, name: str, mask: Array) -> Array:
    height, width = mask.shape
    canvas = jnp.zeros((height * TILE, width * TILE, 4), jnp.float32)
    for level, tile in enumerate(sprite_tiles(name)):
        layer = jnp.where(mask[..., None, None, None], tile, 0.0)
        layer = to_image(layer)
        shift = level * TILE
        layer = jnp.concatenate([layer[shift:], jnp.zeros((shift, *layer.shape[1:]))])
        canvas = layer_over(canvas, layer)
    return layer_over(image, canvas)


def layer_over(lower: Array, upper: Array) -> Array:
    alpha = upper[..., 3:] / 255.0
    color = upper[..., :3] * alpha + lower[..., :3] * (1.0 - alpha)
    lower_alpha = lower[..., 3:] / 255.0 if lower.shape[-1] == 4 else 1.0
    total = alpha + lower_alpha * (1.0 - alpha)
    if lower.shape[-1] == 3:
        return color
    return jnp.concatenate([color, total * 255.0], axis=-1)


@cache
def agent_bank() -> np.ndarray:
    return wanderer_sprites()


@cache
def team_bank() -> np.ndarray:
    return team_sprites()


def energy_bar(energy: Array, frame: Array | None = None) -> Array:
    fraction = jnp.clip(energy / MAX_ENERGY, 0.0, 1.0)
    filled = jnp.arange(BAR_INNER) < jnp.round(fraction * BAR_INNER)
    color = jnp.where(fraction > 0.5, BAR_FULL, jnp.where(fraction > 0.25, BAR_HALF, BAR_LOW))
    inner = jnp.where(filled[None, :, None], color, BAR_EMPTY)
    inner = jnp.broadcast_to(inner, (BAR_HEIGHT - 2, BAR_INNER, 3))
    bar = jnp.pad(inner, ((1, 1), (1, 1), (0, 0)))
    if frame is None:
        return bar
    inside = jnp.pad(jnp.ones((BAR_HEIGHT - 2, BAR_INNER), bool), ((1, 1), (1, 1)))
    return jnp.where(inside[..., None], bar, frame)


def paint_agents(image: Array, state: "DuneSeaState", teams: Array | None = None) -> Array:
    marked = teams is not None and bool(np.asarray(teams).any())
    bank = jnp.asarray(team_bank() if marked else agent_bank())
    coords = (jnp.arange(TILE) + 0.5) / TILE
    radius = jnp.hypot(coords[:, None] - 0.5, coords[None, :] - 0.5)
    ring = (radius >= 0.40) & (radius < 0.46)
    num_agents = state.agent.position.shape[0]
    robes = jnp.arange(num_agents) % bank.shape[0]
    if marked:
        team_ids = np.asarray(teams, np.int32)
        within = np.array([(team_ids[:i] == team_ids[i]).sum() for i in range(num_agents)])
        robes = jnp.asarray((team_ids % 2) * 2 + within % 2)
    rows, columns = jnp.meshgrid(jnp.arange(TILE), jnp.arange(TILE), indexing="ij")
    base = ((columns - 7.5) / 7.2) ** 2 + ((rows - 13.0) / 2.3) ** 2
    marker = (base >= 0.3) & (base <= 1.25)
    teams = jnp.zeros(num_agents, jnp.int32) if teams is None else jnp.asarray(teams, jnp.int32)

    def draw(image, agent):
        position, direction, broadcast, robe, energy, team = agent
        y, x = position
        region = jax.lax.dynamic_slice(image, (y * TILE, x * TILE, 0), (TILE, TILE, 3))
        if marked:
            region = jnp.where(marker[..., None], TEAM_COLORS[team % len(TEAM_COLORS)], region)
        region = jnp.where((ring & broadcast)[..., None], BROADCAST_COLOR, region)
        region = layer_over(region, bank[robe, POSE_OF_DIRECTION[direction]])
        drawn = jax.lax.dynamic_update_slice(image, region, (y * TILE, x * TILE, 0))
        top, left = jnp.maximum(y * TILE - BAR_HEIGHT, 0), x * TILE + 1
        frame = TEAM_COLORS[team % len(TEAM_COLORS)] if marked else None
        drawn = jax.lax.dynamic_update_slice(drawn, energy_bar(energy, frame), (top, left, 0))
        return jnp.where(energy > 0, drawn, image), None

    agents = (
        state.agent.position,
        state.agent.direction,
        state.agent.broadcast,
        robes,
        state.agent.energy,
        teams,
    )
    image, _ = jax.lax.scan(draw, image, agents)
    return image


def render(state: "DuneSeaState", scale: int = 2, teams: Array | None = None) -> Array:
    height, width = state.cells.wall.shape
    wall = state.cells.wall > 0
    ground = choose_fill(fill_tiles(GROUND_SHEET).astype(np.float32), height, width)
    ground = blend_over(ground, jnp.asarray(mountain_edges(MOUNTAIN_SHEET))[neighbour_mask(wall)])
    mountain = choose_fill(fill_tiles(MOUNTAIN_SHEET).astype(np.float32), height, width)
    image = to_image(jnp.where(wall[..., None, None, None], mountain, ground))
    image = paint_sprite(image, GOAL_SPRITE, state.cells.goal > 0)
    energy = state.cells.energy
    image = paint_sprite(image, ENERGY_SPRITE, (energy > 0) & (energy < MATURE_ENERGY))
    image = paint_sprite(image, RIPE_SPRITE, energy >= MATURE_ENERGY)
    image = paint_agents(image, state, teams)
    image = jnp.repeat(jnp.repeat(image, scale, axis=0), scale, axis=1)
    return jnp.clip(image, 0, 255).astype(jnp.uint8)
