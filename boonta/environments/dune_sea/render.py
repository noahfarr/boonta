import functools
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from boonta.utils import Array

from .sprites import team_sprites, wanderer_sprites
from .state import MATURE_ENERGY, MAX_ENERGY, MIRAGE_STEPS

if TYPE_CHECKING:
    from .state import DuneSeaState

DIRECTORY = Path(__file__).parent / "assets" / "glionox_desert"
TILE = 16
PATCH_SHEET = "sandstone.png"
SHORE_SHEET = "sand_to_water.png"
FOREST_DIRECTORY = DIRECTORY.parent / "glionox_forest"
WATER_TILE = (3, 9)
WET_SAND = (0.80, 0.76, 0.66)
WATER_BLUR = 6
SHIMMER_REACH = 2
FADE_STEPS = 30.0
RISE_OPACITY = 0.15
SHIMMER_AMPLITUDE = 2.0
SHIMMER_FREQUENCY = 0.45
SHIMMER_SPEED = 0.7
EDGE_ROUGHNESS = 0.2
DAMP_START = 0.12
REED_SPRITES = tuple(f"decor_reed{i}.png" for i in range(1, 9))
LILYPAD_SPRITES = tuple(f"decor_lilypad{i}.png" for i in range(1, 9))
GRASS_SPRITES = tuple(f"decor_grass{i}.png" for i in range(1, 5))
REED_CHANCE = 0.34
LILYPAD_CHANCE = 0.12
STORM_COLOR = (232, 198, 142)
WIND = (1.4, 5.0)
DUST_FAR = 0.11
DUST_NEAR = 0.22
HIGH_HAZE = 0.07
HIGH_STREAKS = 0.10
SHADOW_STRENGTH = 0.20
SHADOW_OFFSET = (12, 8)
PALM_SPRITES = tuple(f"tree_palm{i}.png" for i in range(1, 5))
PALM_HEIGHT, PALM_WIDTH = 112, 48
PALMS_PER_POND = 3
BELT_REACH = 4
PALM_REACH = 2
MAX_PALMS = 48
DEEP_ROCK_SHEET = "slate.png"
PATCH_FEATURE = 7.0
PATCH_COVERAGE = 0.30
SCENERY_STONES = tuple(f"decor_stone_small{i}.png" for i in range(1, 5))
SCENERY_BRUSH = tuple(f"decor_dead_bush{i}.png" for i in range(1, 5)) + ("decor_dead_bush_small.png",)
SCENERY_CACTI = tuple(f"decor_cactus{i}.png" for i in range(1, 6))
BUSH_SPRITES = ("decor_iris1.png", "decor_iris2.png", "decor_iris3.png", "decor_iris4.png")
BUSH_ALPHA = 0.85
BUSH_FOREGROUND_ROWS = (10, 16)
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
FOV_ALPHA = 0.25
BAR_INNER = 12
BAR_HEIGHT = 4
BAR_FULL = jnp.array([90, 200, 80], dtype=jnp.float32)
BAR_HALF = jnp.array([235, 190, 50], dtype=jnp.float32)
BAR_LOW = jnp.array([215, 60, 50], dtype=jnp.float32)
BAR_EMPTY = jnp.array([60, 50, 45], dtype=jnp.float32)


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
    return np.stack(rows)[..., :3]


@cache
def palm_bank() -> np.ndarray:
    return np.stack([read_image(name).astype(np.float32) for name in PALM_SPRITES])


@cache
def water_tile() -> np.ndarray:
    sheet = read_image("water.png", FOREST_DIRECTORY).astype(np.float32)
    tile = cut_tile(sheet, *WATER_TILE)[..., :3]
    return tile * 0.55 + np.array([50.0, 130.0, 210.0], np.float32) * 0.45


@cache
def edge_noise(height: int, width: int) -> np.ndarray:
    generator = np.random.default_rng(41)
    frequency_y = np.fft.fftfreq(height)[:, None]
    frequency_x = np.fft.rfftfreq(width)[None, :]
    falloff = np.exp(-2.0 * np.pi**2 * (3.2**2) * (frequency_x**2 + frequency_y**2))
    spectrum = generator.normal(size=(height, width // 2 + 1)) + 1j * generator.normal(size=(height, width // 2 + 1))
    values = np.fft.irfft2(spectrum * falloff, s=(height, width))
    return ((values - values.mean()) / (values.std() + 1e-9)).astype(np.float32)


def blur(field: Array, reach: int) -> Array:
    offsets = range(-reach, reach + 1, 2)
    total = jnp.zeros_like(field)
    for dy in offsets:
        for dx in offsets:
            total = total + jnp.roll(field, (dy, dx), axis=(0, 1))
    return total / (len(offsets) ** 2)


def paint_water(image: Array, water: Array, wall: Array, opacity: Array) -> tuple[Array, Array]:
    height, width = water.shape
    pixels = lambda mask: jnp.repeat(jnp.repeat(mask.astype(jnp.float32), TILE, axis=0), TILE, axis=1)
    smooth = blur(pixels(water), WATER_BLUR)
    solid = blur(pixels(jnp.where(water, opacity, 0.0)), WATER_BLUR) / jnp.maximum(smooth, 1e-3)
    near_edge = jnp.clip(smooth * 4.0, 0.0, 1.0) * jnp.clip((1.0 - smooth) * 4.0, 0.0, 1.0)
    field = smooth + EDGE_ROUGHNESS * near_edge * jnp.asarray(edge_noise(height * TILE, width * TILE))
    on_rock = pixels(wall) > 0
    lake = (field > 0.5) & ~on_rock
    colour = jnp.tile(jnp.asarray(water_tile()), (height, width, 1))
    damp = jnp.clip((field - DAMP_START) / (0.5 - DAMP_START), 0.0, 1.0) * (~lake & ~on_rock) * solid
    wet = image * (1.0 - damp[..., None] * (1.0 - jnp.array(WET_SAND, jnp.float32)))
    filled = image + (colour - image) * solid[..., None]
    return jnp.where(lake[..., None], filled, wet), lake


def near_maximum(field: Array, reach: int) -> Array:
    height, width = field.shape
    padded = jnp.pad(field, reach)
    return functools.reduce(
        jnp.maximum,
        [
            padded[dy : dy + height, dx : dx + width]
            for dy in range(2 * reach + 1)
            for dx in range(2 * reach + 1)
        ],
    )


def count_neighbours(mask: Array) -> Array:
    height, width = mask.shape
    padded = jnp.pad(mask.astype(jnp.float32), 1)
    return sum(
        padded[dy : dy + height, dx : dx + width]
        for dy in range(3)
        for dx in range(3)
        if (dy, dx) != (1, 1)
    )


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
def storm_fields(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def field(feature_y: float, feature_x: float, seed: int) -> np.ndarray:
        generator = np.random.default_rng(seed)
        frequency_y = np.fft.fftfreq(height)[:, None]
        frequency_x = np.fft.rfftfreq(width)[None, :]
        falloff = np.exp(-2.0 * np.pi**2 * ((feature_y * frequency_y) ** 2 + (feature_x * frequency_x) ** 2))
        spectrum = generator.normal(size=(height, width // 2 + 1)) + 1j * generator.normal(size=(height, width // 2 + 1))
        values = np.fft.irfft2(spectrum * falloff, s=(height, width))
        return (values - values.mean()) / (values.std() + 1e-9)

    def along_the_wind(values: np.ndarray) -> np.ndarray:
        slope = WIND[0] / WIND[1]
        return np.stack([np.roll(values[:, x], int(round(slope * x))) for x in range(width)], axis=1)

    far = np.clip(along_the_wind(field(1.1, 10.0, 21)) - 1.0, 0.0, 1.0)
    near = np.clip(along_the_wind(field(0.9, 16.0, 23)) - 1.1, 0.0, 1.0)
    cloud = 0.5 + 0.5 * np.tanh(field(30.0, 80.0, 22))
    return far.astype(np.float32), near.astype(np.float32), cloud.astype(np.float32)


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


def paint_sprite(
    image: Array,
    name: str,
    mask: Array,
    alpha: float = 1.0,
    directory: Path = DIRECTORY,
    levels: tuple[int, ...] | None = None,
    rows: tuple[int, int] | None = None,
    opacity: Array | None = None,
) -> Array:
    height, width = mask.shape
    canvas = jnp.zeros((height * TILE, width * TILE, 4), jnp.float32)
    for level, tile in enumerate(sprite_tiles(name, directory)):
        if levels is not None and level not in levels:
            continue
        fade = jnp.full((TILE,), alpha)
        if rows is not None:
            fade = jnp.where((jnp.arange(TILE) >= rows[0]) & (jnp.arange(TILE) < rows[1]), alpha, 0.0)
        tile = jnp.asarray(tile).at[..., 3].multiply(fade[:, None])
        layer = jnp.where(mask[..., None, None, None], tile, 0.0)
        layer = to_image(layer)
        shift = level * TILE
        layer = jnp.concatenate([layer[shift:], jnp.zeros((shift, *layer.shape[1:]))])
        canvas = layer_over(canvas, layer)
    drawn = layer_over(image, canvas)
    if opacity is None:
        return drawn
    pixels = jnp.repeat(jnp.repeat(opacity, TILE, axis=0), TILE, axis=1)[..., None]
    return image + (drawn - image) * pixels


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


def pond_labels(water: Array, iterations: int = 12) -> Array:
    height, width = water.shape
    none = height * width
    labels = jnp.where(water, jnp.arange(none).reshape(height, width), none)
    for _ in range(iterations):
        padded = jnp.pad(labels, 1, constant_values=none)
        smallest = functools.reduce(
            jnp.minimum,
            [padded[dy : dy + height, dx : dx + width] for dy in range(3) for dx in range(3)],
        )
        labels = jnp.where(water, smallest, none)
    return labels


def palm_sites(water: Array, land: Array, rolls: Array) -> Array:
    height, width = water.shape
    none = height * width
    labels = pond_labels(water)
    padded = jnp.pad(labels, PALM_REACH, constant_values=none)
    nearest = functools.reduce(
        jnp.minimum,
        [
            padded[dy : dy + height, dx : dx + width]
            for dy in range(2 * PALM_REACH + 1)
            for dx in range(2 * PALM_REACH + 1)
        ],
    )
    pond = jnp.where(land & (nearest < none), nearest, none).reshape(-1)
    order = jnp.argsort(pond.astype(jnp.float32) * 2.0 + (1.0 - rolls[1].reshape(-1)))
    ordered = pond[order]
    position = jnp.arange(none)
    starts = jnp.concatenate([jnp.array([True]), ordered[1:] != ordered[:-1]])
    rank = position - jax.lax.cummax(jnp.where(starts, position, 0))
    keep = (ordered < none) & (rank < PALMS_PER_POND + ordered % 2)
    return jnp.zeros(none, bool).at[order].set(keep).reshape(height, width)


def paint_palms(
    image: Array, sites: Array, rolls: Array, behind_only: bool = False, opacity: Array | None = None
) -> Array:
    height, width = sites.shape
    bank = jnp.asarray(palm_bank())
    score = jnp.where(sites, rolls[1], -1.0)
    values, flat = jax.lax.top_k(score.reshape(-1), min(MAX_PALMS, height * width))
    rows, columns = flat // width, flat % width
    order = jnp.argsort(rows)
    variants = jnp.floor(rolls[0].reshape(-1)[flat] * len(PALM_SPRITES)).astype(jnp.int32) % len(PALM_SPRITES)
    drawn = jnp.pad(image, ((PALM_HEIGHT, 0), (PALM_WIDTH // 2, PALM_WIDTH // 2), (0, 0)))

    def draw(canvas, item):
        row, column, variant, active = item
        top, left = (row + 1) * TILE, column * TILE + TILE // 2
        region = jax.lax.dynamic_slice(canvas, (top, left, 0), (PALM_HEIGHT, PALM_WIDTH, 3))
        sprite = bank[variant]
        alpha = sprite[..., 3:] / 255.0 * active
        if behind_only:
            alpha = alpha * (jnp.arange(PALM_HEIGHT) < PALM_HEIGHT - TILE)[:, None, None]
        blended = region * (1.0 - alpha) + sprite[..., :3] * alpha
        return jax.lax.dynamic_update_slice(canvas, blended, (top, left, 0)), None

    presence = (values >= 0).astype(jnp.float32)
    if opacity is not None:
        presence = presence * opacity.reshape(-1)[flat]
    items = (rows[order], columns[order], variants[order], presence[order])
    drawn, _ = jax.lax.scan(draw, drawn, items)
    return drawn[PALM_HEIGHT:, PALM_WIDTH // 2 : -PALM_WIDTH // 2]


def paint_scenery(image: Array, state: "DuneSeaState", water: Array) -> Array:
    height, width = state.cells.wall.shape
    wall = state.cells.wall > 0
    rolls = jnp.asarray(scenery_rolls(height, width))
    near_rock = ~wall & (neighbour_mask(wall) > 0)
    free = ~wall & ~water & (state.cells.bush == 0) & (state.cells.energy == 0) & (state.cells.goal == 0)
    stone_chance = jnp.where(near_rock, 0.12, 0.012)
    brush_chance, cactus_chance = 0.022, 0.006
    roll, pick = rolls[0], rolls[1]
    kinds = (
        (SCENERY_STONES, free & (roll < stone_chance)),
        (SCENERY_BRUSH, free & (roll >= stone_chance) & (roll < stone_chance + brush_chance)),
        (SCENERY_CACTI, free & (roll >= stone_chance + brush_chance) & (roll < stone_chance + brush_chance + cactus_chance)),
    )
    for names, where_ in kinds:
        variant = jnp.floor(pick * len(names)).astype(jnp.int32)
        for index, name in enumerate(names):
            image = paint_sprite(image, name, where_ & (variant == index))
    return image


def drift(layer: Array, phase: Array, speed: float) -> Array:
    shift = jnp.round(phase * speed * jnp.array(WIND)).astype(jnp.int32)
    return jnp.roll(layer, (shift[0], shift[1]), axis=(0, 1))


def shimmer(image: Array, haze: Array, phase: Array) -> Array:
    height, width = haze.shape
    padded = jnp.pad(haze, SHIMMER_REACH)
    near = functools.reduce(
        jnp.maximum,
        [
            padded[dy : dy + height, dx : dx + width]
            for dy in range(2 * SHIMMER_REACH + 1)
            for dx in range(2 * SHIMMER_REACH + 1)
        ],
    )
    strength = jnp.repeat(jnp.repeat(near, TILE, axis=0), TILE, axis=1)
    rows = jnp.arange(height * TILE)[:, None]
    columns = jnp.arange(width * TILE)[None, :]
    sway = jnp.round(strength * SHIMMER_AMPLITUDE * jnp.sin(rows * SHIMMER_FREQUENCY + phase * SHIMMER_SPEED))
    source = jnp.clip(columns + sway.astype(jnp.int32), 0, width * TILE - 1)
    wavy = jnp.take_along_axis(image, jnp.broadcast_to(source[..., None], image.shape), axis=1)
    return jnp.where((strength > 0)[..., None], wavy, image)


def water_phases(state: "DuneSeaState", phase: Array) -> tuple[Array, Array]:
    fake = state.cells.mirage < 0
    risen = (state.cells.spare < 0) & (state.cells.oasis > 0)
    lifetime = jnp.clip((phase + state.cells.mirage) / MIRAGE_STEPS, 0.0, 1.0)
    rising = jnp.clip((phase + state.cells.spare) / FADE_STEPS, 0.0, 1.0)
    opacity = jnp.where(
        fake,
        1.0 - lifetime,
        jnp.where(risen, RISE_OPACITY + (1.0 - RISE_OPACITY) * rising, 1.0),
    )
    haze = jnp.where(fake, jnp.sin(jnp.pi * lifetime), jnp.where(risen, 1.0 - rising, 0.0))
    return opacity, haze


def tint(image: Array, alpha: Array) -> Array:
    return image * (1.0 - alpha[..., None]) + jnp.array(STORM_COLOR, jnp.float32) * alpha[..., None]


def paint_ground_storm(image: Array, phase: Array, strength: float) -> Array:
    far, near, cloud = (jnp.asarray(layer) for layer in storm_fields(*image.shape[:2]))
    overhead = drift(cloud, phase, 0.4)
    shadow = jnp.roll(overhead, SHADOW_OFFSET, axis=(0, 1))
    image = image * (1.0 - strength * SHADOW_STRENGTH * jnp.clip((shadow - 0.55) / 0.35, 0.0, 1.0)[..., None])
    return tint(image, strength * (DUST_FAR * drift(far, phase, 0.7) + DUST_NEAR * drift(near, phase, 1.0)))


def paint_high_storm(image: Array, phase: Array, strength: float) -> Array:
    _, near, cloud = (jnp.asarray(layer) for layer in storm_fields(*image.shape[:2]))
    return tint(image, strength * (HIGH_HAZE * drift(cloud, phase, 0.4) + HIGH_STREAKS * drift(near, phase, 1.6)))


def paint_agents(
    image: Array, state: "DuneSeaState", teams: Array | None = None, offsets: Array | None = None
) -> Array:
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
    offsets = jnp.zeros((num_agents, 2), jnp.int32) if offsets is None else offsets
    limit = jnp.array(image.shape[:2]) - TILE

    def draw(image, agent):
        position, direction, broadcast, robe, energy, team, offset = agent
        y, x = jnp.clip(position * TILE + offset, 0, limit)
        region = jax.lax.dynamic_slice(image, (y, x, 0), (TILE, TILE, 3))
        if marked:
            region = jnp.where(marker[..., None], TEAM_COLORS[team % len(TEAM_COLORS)], region)
        region = jnp.where((ring & broadcast)[..., None], BROADCAST_COLOR, region)
        region = layer_over(region, bank[robe, POSE_OF_DIRECTION[direction]])
        drawn = jax.lax.dynamic_update_slice(image, region, (y, x, 0))
        top, left = jnp.maximum(y - BAR_HEIGHT, 0), jnp.minimum(x + 1, image.shape[1] - BAR_INNER - 2)
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
        offsets,
    )
    image, _ = jax.lax.scan(draw, image, agents)
    return image


def paint_fov(image: Array, fov: Array, teams: Array | None) -> Array:
    team_ids = np.zeros(fov.shape[0], np.int32) if teams is None else np.asarray(teams, np.int32)
    for team in sorted(set(team_ids.tolist())):
        seen = (fov & jnp.asarray(team_ids == team)[:, None, None]).any(axis=0)
        mask = jnp.repeat(jnp.repeat(seen, TILE, axis=0), TILE, axis=1)[..., None]
        color = TEAM_COLORS[team % len(TEAM_COLORS)]
        image = jnp.where(mask, image * (1.0 - FOV_ALPHA) + color * FOV_ALPHA, image)
    return image


def render(
    state: "DuneSeaState",
    scale: int = 2,
    teams: Array | None = None,
    fov: Array | None = None,
    storm: float = 1.0,
    offsets: Array | None = None,
    blend: float = 1.0,
) -> Array:
    height, width = state.cells.wall.shape
    wall = state.cells.wall > 0
    noise = smooth_noise(height, width, PATCH_FEATURE, 3)
    patch = jnp.asarray(noise > np.quantile(noise, 1.0 - PATCH_COVERAGE)) & ~wall
    ground = choose_fill(fill_tiles(GROUND_SHEET).astype(np.float32), height, width)
    ground = blend_over(ground, jnp.asarray(mountain_edges(PATCH_SHEET))[neighbour_mask(patch)])
    ground = jnp.where(patch[..., None, None, None], choose_fill(fill_tiles(PATCH_SHEET).astype(np.float32), height, width), ground)
    ground = blend_over(ground, jnp.asarray(mountain_edges(MOUNTAIN_SHEET))[neighbour_mask(wall)])
    mountain = choose_fill(fill_tiles(MOUNTAIN_SHEET).astype(np.float32), height, width)
    deep = (neighbour_mask(wall) == 255)[..., None, None, None]
    mountain = jnp.where(deep, choose_fill(fill_tiles(DEEP_ROCK_SHEET).astype(np.float32), height, width), mountain)
    mirage = state.cells.mirage < 0
    water = ((state.cells.oasis > 0) | mirage) & ~wall
    neighbours = count_neighbours(water)
    wet = ~water & ~wall & (neighbours > 0)
    image = to_image(jnp.where(wall[..., None, None, None], mountain, ground))
    phase = state.time.astype(jnp.float32) - 1.0 + blend
    opacity, haze = water_phases(state, phase)
    image, lake = paint_water(image, water, wall, opacity)
    if storm:
        image = paint_ground_storm(image, phase, storm)
    image = paint_scenery(image, state, water)
    rolls = jnp.asarray(scenery_rolls(height, width))
    pond_opacity = jnp.where(water, opacity, 0.0)
    shore_opacity = near_maximum(pond_opacity, 1)
    belt_opacity = near_maximum(pond_opacity, BELT_REACH)
    reed_here = wet & (state.cells.bush == 0) & (rolls[1] < REED_CHANCE)
    reed_kind = jnp.floor(rolls[0] * len(REED_SPRITES)).astype(jnp.int32) % len(REED_SPRITES)
    lily_here = water & (neighbours >= 7) & (rolls[0] > 1.0 - LILYPAD_CHANCE)
    lily_kind = jnp.floor(rolls[1] * len(LILYPAD_SPRITES)).astype(jnp.int32) % len(LILYPAD_SPRITES)
    for index, name in enumerate(LILYPAD_SPRITES):
        image = paint_sprite(image, name, lily_here & (lily_kind == index), directory=FOREST_DIRECTORY, opacity=opacity)
    for index, name in enumerate(REED_SPRITES):
        image = paint_sprite(image, name, reed_here & (reed_kind == index), directory=FOREST_DIRECTORY, opacity=shore_opacity)
    belt = jnp.zeros((height, width), jnp.float32)
    close = jnp.zeros((height, width), jnp.float32)
    padded_belt = jnp.pad(water.astype(jnp.float32), BELT_REACH)
    for dy in range(2 * BELT_REACH + 1):
        for dx in range(2 * BELT_REACH + 1):
            shifted = padded_belt[dy : dy + height, dx : dx + width]
            belt = jnp.maximum(belt, shifted)
            if abs(dy - BELT_REACH) <= 2 and abs(dx - BELT_REACH) <= 2:
                close = jnp.maximum(close, shifted)
    tuft_chance = jnp.where(close > 0, 0.5, 0.2)
    tufts = (belt > 0) & ~water & ~wall & ~reed_here & (state.cells.bush == 0) & (rolls[0] < tuft_chance)
    tuft_kind = jnp.floor(rolls[1] * len(GRASS_SPRITES) * 3.0).astype(jnp.int32) % len(GRASS_SPRITES)
    for index, name in enumerate(GRASS_SPRITES):
        image = paint_sprite(image, name, tufts & (tuft_kind == index), directory=FOREST_DIRECTORY, opacity=belt_opacity)
    sites = palm_sites(water, ~water & ~wall & (state.cells.bush == 0), rolls)
    palm_opacity = near_maximum(pond_opacity, PALM_REACH)
    image = paint_palms(image, sites, rolls, opacity=palm_opacity)
    image = shimmer(image, haze, phase)
    rows, columns = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    image = paint_sprite(image, GOAL_SPRITE, state.cells.goal > 0)
    energy = state.cells.energy
    image = paint_sprite(image, ENERGY_SPRITE, (energy > 0) & (energy < MATURE_ENERGY))
    image = paint_sprite(image, RIPE_SPRITE, energy >= MATURE_ENERGY)
    variety = (rows * ROW_STRIDE + columns * COLUMN_STRIDE) % len(BUSH_SPRITES)
    y, x = state.agent.position[:, 0], state.agent.position[:, 1]
    occupied = jnp.zeros((height, width), jnp.float32).at[y, x].add((state.agent.energy > 0).astype(jnp.float32)) > 0
    bush = state.cells.bush > 0
    for index, name in enumerate(BUSH_SPRITES):
        image = paint_sprite(image, name, bush & occupied & (variety == index), BUSH_ALPHA)
    if fov is not None:
        image = paint_fov(image, fov, teams)
    image = paint_agents(image, state, teams, offsets)
    image = paint_palms(image, sites, rolls, behind_only=True, opacity=palm_opacity)
    for index, name in enumerate(BUSH_SPRITES):
        here = bush & (variety == index)
        image = paint_sprite(image, name, here & ~occupied, BUSH_ALPHA)
        image = paint_sprite(image, name, here & occupied, BUSH_ALPHA, levels=(0,), rows=BUSH_FOREGROUND_ROWS)
    if storm:
        image = paint_high_storm(image, phase, storm)
    image = jnp.repeat(jnp.repeat(image, scale, axis=0), scale, axis=1)
    return jnp.clip(image, 0, 255).astype(jnp.uint8)
