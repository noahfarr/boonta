import functools
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .sprites import team_sprites, wanderer_sprites
from .state import MATURE_ENERGY, MAX_ENERGY

if TYPE_CHECKING:
    from .state import Agent, JundlandState

DIRECTORY = Path(__file__).parent / "assets" / "glionox_desert"
TILE = 16
PATCH_SHEET = "sandstone.png"
DEEP_ROCK_SHEET = "slate.png"
PATCH_FEATURE = 7.0
PATCH_COVERAGE = 0.30
SCENERY_STONES = tuple(f"decor_stone_small{i}.png" for i in range(1, 5))
SCENERY_BRUSH = tuple(f"decor_dead_bush{i}.png" for i in range(1, 5)) + ("decor_dead_bush_small.png",)
SCENERY_CACTI = tuple(f"decor_cactus{i}.png" for i in range(1, 6))
BUSH_SPRITES = ("decor_iris1.png", "decor_iris2.png", "decor_iris3.png", "decor_iris4.png")
BUSH_ALPHA = 0.85
BUSH_FOREGROUND_ROWS = (10, 16)
BROADCAST_COLOR = np.array([255, 255, 255], dtype=np.float32)
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
ROW_STRIDE, COLUMN_STRIDE = 7, 13
POSE_OF_DIRECTION = np.array([1, 2, 3, 0])
TEAM_COLORS = np.array(
    [[60, 120, 255], [230, 60, 50], [60, 190, 90], [240, 190, 40]], dtype=np.float32
)
FIELD_OF_VIEW_ALPHA = 0.25
BAR_INNER = 12
BAR_HEIGHT = 4
BAR_FULL = np.array([90, 200, 80], dtype=np.float32)
BAR_HALF = np.array([235, 190, 50], dtype=np.float32)
BAR_LOW = np.array([215, 60, 50], dtype=np.float32)
BAR_EMPTY = np.array([60, 50, 45], dtype=np.float32)



def read_image(name: str, directory: Path = DIRECTORY) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(directory / name).convert("RGBA"))


@cache
def fill_tiles(sheet: str) -> np.ndarray:
    image = read_image(sheet)
    rows = [
        image[row * TILE : (row + 1) * TILE, col * TILE : (col + 1) * TILE]
        for row in FILL_ROWS
        for col in range(SHEET_COLUMNS)
    ]
    return np.stack(rows)[..., :3].astype(np.float32)


@cache
def sprite_tiles(name: str, directory: Path = DIRECTORY) -> np.ndarray:
    image = read_image(name, directory)
    num_tiles = image.shape[0] // TILE
    tiles = [image[k * TILE : (k + 1) * TILE] for k in range(num_tiles)]
    return np.stack(tiles[::-1]).astype(np.float32)


@cache
def smooth_noise(height: int, width: int, feature: float, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    frequency_y = np.fft.fftfreq(height)[:, None]
    frequency_x = np.fft.rfftfreq(width)[None, :]
    falloff = np.exp(-2.0 * (np.pi * feature / 2.0) ** 2 * (frequency_x**2 + frequency_y**2))
    spectrum = generator.normal(size=(height, width // 2 + 1)) + 1j * generator.normal(size=(height, width // 2 + 1))
    field = np.fft.irfft2(spectrum * falloff, s=(height, width))
    return ((field - field.mean()) / (field.std() + 1e-9)).astype(np.float32)


@cache
def scenery_rolls(height: int, width: int) -> np.ndarray:
    return np.random.default_rng(7).random((2, height, width)).astype(np.float32)


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
def mountain_edges(sheet: str, layered: bool = True) -> np.ndarray:
    image = read_image(sheet).astype(np.float32)
    table = np.zeros((256, TILE, TILE, 4), np.float32)
    filled = set()
    for row in range(OVERHANG_ROWS):
        for col in range(SHEET_COLUMNS):
            tile = cut_tile(image, row, col)
            if layered:
                tile = stack_layers(cut_tile(image, row + DETAIL_OFFSET, col), tile)
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


@cache
def agent_bank() -> np.ndarray:
    return wanderer_sprites()


@cache
def team_bank() -> np.ndarray:
    return team_sprites()


def neighbour_mask(wall: np.ndarray) -> np.ndarray:
    height, width = wall.shape
    padded = np.pad(np.asarray(wall, bool), 1)
    flags = [padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width] for dx, dy in NEIGHBOUR_STEPS]
    mask = np.zeros(wall.shape, np.int32)
    for bit, flag in enumerate(flags):
        if bit % 2 == 1:
            flag = flag | flags[bit - 1] | flags[(bit + 1) % 8]
        mask |= flag.astype(np.int32) << bit
    return mask


def blend_over(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    alpha = layer[..., 3:] / 255.0
    return base * (1.0 - alpha) + layer[..., :3] * alpha


def choose_fill(tiles: np.ndarray, height: int, width: int) -> np.ndarray:
    rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return tiles[(rows * ROW_STRIDE + columns * COLUMN_STRIDE) % tiles.shape[0]]


def to_image(cells: np.ndarray) -> np.ndarray:
    height, width, rows, columns, channels = cells.shape
    return np.ascontiguousarray(cells.transpose(0, 2, 1, 3, 4)).reshape(height * rows, width * columns, channels)


def as_tiles(image: np.ndarray, height: int, width: int) -> np.ndarray:
    return image.reshape(height, TILE, width, TILE, 3).transpose(0, 2, 1, 3, 4)


@functools.lru_cache(maxsize=4)
def terrain(wall_bytes: bytes, height: int, width: int) -> np.ndarray:
    wall = np.frombuffer(wall_bytes, bool).reshape(height, width)
    noise = smooth_noise(height, width, PATCH_FEATURE, 3)
    patch = (noise > np.quantile(noise, 1.0 - PATCH_COVERAGE)) & ~wall
    ground = choose_fill(fill_tiles(GROUND_SHEET), height, width)
    ground = blend_over(ground, mountain_edges(PATCH_SHEET)[neighbour_mask(patch)])
    ground = np.where(patch[..., None, None, None], choose_fill(fill_tiles(PATCH_SHEET), height, width), ground)
    edges = neighbour_mask(wall)
    ground = blend_over(ground, mountain_edges(MOUNTAIN_SHEET)[edges])
    mountain = choose_fill(fill_tiles(MOUNTAIN_SHEET), height, width)
    mountain = np.where((edges == 255)[..., None, None, None], choose_fill(fill_tiles(DEEP_ROCK_SHEET), height, width), mountain)
    image = to_image(np.where(wall[..., None, None, None], mountain, ground)).astype(np.float32)
    image.flags.writeable = False
    return image


def paint_sprite(
    image: np.ndarray,
    name: str,
    mask: np.ndarray,
    alpha: float = 1.0,
    directory: Path = DIRECTORY,
    levels: tuple[int, ...] | None = None,
    rows: tuple[int, int] | None = None,
    opacity: np.ndarray | None = None,
) -> np.ndarray:
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return image
    fade = np.full(TILE, alpha, np.float32)
    if rows is not None:
        index = np.arange(TILE)
        fade = np.where((index >= rows[0]) & (index < rows[1]), alpha, 0.0).astype(np.float32)
    chosen = [
        (level, tile) for level, tile in enumerate(sprite_tiles(name, directory)) if levels is None or level in levels
    ]
    targets = []
    for level, _ in chosen:
        valid = ys - level >= 0
        targets.append(((ys - level)[valid], xs[valid], valid))
    cells = np.unique(np.concatenate([row * width + column for row, column, _ in targets]))
    if not len(cells):
        return image
    slot = np.full(height * width, -1)
    slot[cells] = np.arange(len(cells))
    colour = np.zeros((len(cells), TILE, TILE, 3), np.float32)
    cover = np.zeros((len(cells), TILE, TILE, 1), np.float32)
    for (level, tile), (row, column, _) in zip(chosen, targets):
        index = slot[row * width + column]
        strength = tile[..., 3:] * fade[:, None, None] / 255.0
        colour[index] = tile[..., :3] * strength + colour[index] * (1.0 - strength)
        cover[index] = strength + cover[index] * (1.0 - strength)
    tiles = as_tiles(image, height, width)
    rows_, columns_ = cells // width, cells % width
    below = tiles[rows_, columns_]
    drawn = colour * cover + below * (1.0 - cover)
    if opacity is not None:
        drawn = below + (drawn - below) * opacity[rows_, columns_][:, None, None, None]
    tiles[rows_, columns_] = drawn
    return image


def paint_scenery(image: np.ndarray, state: "JundlandState", window: tuple[slice, slice]) -> np.ndarray:
    height, width = state.cells.wall.shape
    wall = np.asarray(state.cells.wall) > 0
    rolls = scenery_rolls(height, width)
    near_rock = ~wall & (neighbour_mask(wall) > 0)
    free = (
        ~wall
        & (np.asarray(state.cells.bush) == 0)
        & (np.asarray(state.cells.energy) == 0)
    )[window]
    stone_chance = np.where(near_rock, 0.12, 0.012)[window]
    brush_chance, cactus_chance = 0.022, 0.006
    roll, pick = rolls[0][window], rolls[1][window]
    kinds = (
        (SCENERY_STONES, free & (roll < stone_chance)),
        (SCENERY_BRUSH, free & (roll >= stone_chance) & (roll < stone_chance + brush_chance)),
        (SCENERY_CACTI, free & (roll >= stone_chance + brush_chance) & (roll < stone_chance + brush_chance + cactus_chance)),
    )
    for names, where_ in kinds:
        variant = np.floor(pick * len(names)).astype(np.int32)
        for index, name in enumerate(names):
            image = paint_sprite(image, name, where_ & (variant == index))
    return image


def energy_bar(energy: float, frame: np.ndarray | None = None) -> np.ndarray:
    fraction = float(np.clip(energy / MAX_ENERGY, 0.0, 1.0))
    filled = np.arange(BAR_INNER) < np.round(fraction * BAR_INNER)
    color = BAR_FULL if fraction > 0.5 else BAR_HALF if fraction > 0.25 else BAR_LOW
    inner = np.where(filled[None, :, None], color, BAR_EMPTY)
    inner = np.broadcast_to(inner, (BAR_HEIGHT - 2, BAR_INNER, 3))
    bar = np.pad(inner, ((1, 1), (1, 1), (0, 0)))
    if frame is None:
        return bar
    inside = np.pad(np.ones((BAR_HEIGHT - 2, BAR_INNER), bool), ((1, 1), (1, 1)))
    return np.where(inside[..., None], bar, frame)


def paint_agents(
    image: np.ndarray, agent: "Agent", teams: np.ndarray | None = None, offsets: np.ndarray | None = None
) -> np.ndarray:
    marked = teams is not None and bool(np.asarray(teams).any())
    bank = team_bank() if marked else agent_bank()
    coords = (np.arange(TILE) + 0.5) / TILE
    radius = np.hypot(coords[:, None] - 0.5, coords[None, :] - 0.5)
    ring = (radius >= 0.40) & (radius < 0.46)
    position = np.asarray(agent.position)
    num_agents = position.shape[0]
    robes = np.zeros(num_agents, np.int32)
    if marked:
        team_ids = np.asarray(teams, np.int32)
        within = np.array([(team_ids[:i] == team_ids[i]).sum() for i in range(num_agents)])
        robes = (team_ids % 2) * 2 + within % 2
    rows, columns = np.meshgrid(np.arange(TILE), np.arange(TILE), indexing="ij")
    base = ((columns - 7.5) / 7.2) ** 2 + ((rows - 13.0) / 2.3) ** 2
    marker = (base >= 0.3) & (base <= 1.25)
    team_ids = np.zeros(num_agents, np.int32) if teams is None else np.asarray(teams, np.int32)
    offsets = np.zeros((num_agents, 2), np.int32) if offsets is None else np.asarray(offsets)
    limit = np.array(image.shape[:2]) - TILE
    direction, broadcast, energy = (np.asarray(x) for x in (agent.direction, agent.broadcast, agent.energy))
    for agent in range(num_agents):
        if not energy[agent] > 0:
            continue
        y, x = np.clip(position[agent] * TILE + offsets[agent], 0, limit)
        color = TEAM_COLORS[team_ids[agent] % len(TEAM_COLORS)]
        region = image[y : y + TILE, x : x + TILE]
        if marked:
            region[marker] = color
        if broadcast[agent]:
            region[ring] = BROADCAST_COLOR
        sprite = bank[robes[agent], POSE_OF_DIRECTION[direction[agent]]]
        alpha = sprite[..., 3:] / 255.0
        region[:] = sprite[..., :3] * alpha + region * (1.0 - alpha)
        top, left = max(y - BAR_HEIGHT, 0), min(x + 1, image.shape[1] - BAR_INNER - 2)
        image[top : top + BAR_HEIGHT, left : left + BAR_INNER + 2] = energy_bar(energy[agent], color if marked else None)
    return image


def paint_field_of_view(image: np.ndarray, field_of_view: np.ndarray, teams: np.ndarray) -> np.ndarray:
    field_of_view = np.asarray(field_of_view, bool)
    team_ids = np.asarray(teams, np.int32)
    order = np.argsort(team_ids, kind="stable")
    unique, starts = np.unique(team_ids[order], return_index=True)
    seen = np.logical_or.reduceat(field_of_view[order], starts, axis=0).astype(np.float32)
    later = np.cumsum(seen[::-1], axis=0)[::-1] - seen
    weights = seen * FIELD_OF_VIEW_ALPHA * (1.0 - FIELD_OF_VIEW_ALPHA) ** later
    keep = (1.0 - FIELD_OF_VIEW_ALPHA) ** seen.sum(0)
    tint = np.einsum("thw,tc->hwc", weights, TEAM_COLORS[unique % len(TEAM_COLORS)])
    height, width = keep.shape
    tiles = image.reshape(height, TILE, width, TILE, 3)
    tiles *= keep[:, None, :, None, None].astype(np.float32)
    tiles += tint[:, None, :, None, :].astype(np.float32)
    return image


def render(
    state: "JundlandState",
    scale: int,
    teams: np.ndarray,
    field_of_view: np.ndarray,
    offsets: np.ndarray | None = None,
    corner: tuple[int, int] = (0, 0),
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    cells = state.cells
    wall = np.asarray(cells.wall) > 0
    height, width = wall.shape
    top, left = corner
    rows_shown, columns_shown = size or (height, width)
    window = (slice(top, top + rows_shown), slice(left, left + columns_shown))
    frame = (slice(top * TILE, (top + rows_shown) * TILE), slice(left * TILE, (left + columns_shown) * TILE))
    bush = np.asarray(cells.bush)[window] > 0
    energy = np.asarray(cells.energy)[window]
    image = terrain(wall.tobytes(), height, width)[frame].copy()
    image = paint_scenery(image, state, window)
    rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    image = paint_sprite(image, ENERGY_SPRITE, (energy > 0) & (energy < MATURE_ENERGY))
    image = paint_sprite(image, RIPE_SPRITE, energy >= MATURE_ENERGY)
    variety = ((rows * ROW_STRIDE + columns * COLUMN_STRIDE) % len(BUSH_SPRITES))[window]
    position = np.asarray(state.agent.position) - np.array(corner)
    inside = ((position >= 0) & (position < np.array([rows_shown, columns_shown]))).all(-1)
    shown = (np.asarray(state.agent.energy) > 0) & inside
    occupied = np.zeros((rows_shown, columns_shown), bool)
    occupied[position[shown, 0], position[shown, 1]] = True
    for index, name in enumerate(BUSH_SPRITES):
        image = paint_sprite(image, name, bush & occupied & (variety == index), BUSH_ALPHA)
    image = paint_field_of_view(image, np.asarray(field_of_view)[:, window[0], window[1]], teams)
    visible = state.agent.replace(position=position, energy=np.where(shown, np.asarray(state.agent.energy), 0.0))
    image = paint_agents(image, visible, teams, offsets)
    for index, name in enumerate(BUSH_SPRITES):
        here = bush & (variety == index)
        image = paint_sprite(image, name, here & ~occupied, BUSH_ALPHA)
        image = paint_sprite(image, name, here & occupied, BUSH_ALPHA, levels=(0,), rows=BUSH_FOREGROUND_ROWS)
    image = np.clip(image, 0, 255, out=image).astype(np.uint8)
    if scale > 1:
        from PIL import Image

        size = (image.shape[1] * scale, image.shape[0] * scale)
        image = np.array(Image.fromarray(image).resize(size, Image.NEAREST))
    return image
