import functools
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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
FADE_STEPS = 30.0
RISE_OPACITY = 0.15
SHIMMER_REACH = 2
SHIMMER_SMOOTHING = 2
SHIMMER_AMPLITUDE = 2.0
SHIMMER_FREQUENCY = 0.45
SHIMMER_SPEED = 0.7
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
GOAL_SPRITE = "decor_cactus1_pink.png"
ROW_STRIDE, COLUMN_STRIDE = 7, 13
POSE_OF_DIRECTION = np.array([1, 2, 3, 0])
TEAM_COLORS = np.array(
    [[60, 120, 255], [230, 60, 50], [60, 190, 90], [240, 190, 40]], dtype=np.float32
)
FOV_ALPHA = 0.25
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
def palm_bank() -> np.ndarray:
    return np.stack([read_image(name).astype(np.float32) for name in PALM_SPRITES])


@cache
def water_tile() -> np.ndarray:
    sheet = read_image("water.png", FOREST_DIRECTORY).astype(np.float32)
    tile = cut_tile(sheet, *WATER_TILE)[..., :3]
    return tile * 0.55 + np.array([50.0, 130.0, 210.0], np.float32) * 0.45


@cache
def shore_table() -> np.ndarray:
    return mountain_edges(SHORE_SHEET, layered=False)


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


@cache
def agent_bank() -> np.ndarray:
    return wanderer_sprites()


@cache
def team_bank() -> np.ndarray:
    return team_sprites()


def shifted(field: np.ndarray, reach: int, fill=0):
    height, width = field.shape
    padded = np.pad(field, reach, constant_values=fill)
    return [
        padded[dy : dy + height, dx : dx + width]
        for dy in range(2 * reach + 1)
        for dx in range(2 * reach + 1)
    ]


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


def count_neighbours(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    padded = np.pad(mask.astype(np.float32), 1)
    return sum(
        padded[dy : dy + height, dx : dx + width]
        for dy in range(3)
        for dx in range(3)
        if (dy, dx) != (1, 1)
    )


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


def paint_pond(image: np.ndarray, water: np.ndarray, opacity: np.ndarray) -> np.ndarray:
    height, width = water.shape
    rows, columns = np.nonzero(water)
    if not len(rows):
        return image
    tiles = as_tiles(image, height, width)
    base = tiles[rows, columns]
    filled = base + (water_tile() - base) * opacity[rows, columns][:, None, None, None]
    shore = shore_table()[neighbour_mask(~water)[rows, columns]]
    tiles[rows, columns] = blend_over(filled, shore)
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


def pond_labels(water: np.ndarray, iterations: int = 12) -> np.ndarray:
    height, width = water.shape
    none = height * width
    labels = np.where(water, np.arange(none).reshape(height, width), none)
    for _ in range(iterations):
        smallest = functools.reduce(np.minimum, shifted(labels, 1, fill=none))
        labels = np.where(water, smallest, none)
    return labels


def palm_ponds(water: np.ndarray, land: np.ndarray) -> np.ndarray:
    height, width = water.shape
    none = height * width
    labels = pond_labels(water)
    nearest = functools.reduce(np.minimum, shifted(labels, PALM_REACH, fill=none))
    return np.where(land & (nearest < none), nearest, none)


def pond_presence(mirage: np.ndarray, phase: np.float32) -> np.ndarray:
    age = phase - mirage
    fading = np.clip(1.0 - age / np.float32(MIRAGE_STEPS), 0.0, 1.0)
    rising = np.clip((phase + mirage) / np.float32(FADE_STEPS), 0.0, 1.0)
    presence = np.where(mirage > 0, np.where(age >= 0, fading, 1.0), np.where(mirage < 0, rising, 1.0))
    return presence.astype(np.float32)


def palm_presence(ponds: np.ndarray, mirage: np.ndarray, phase: np.float32) -> np.ndarray:
    none = ponds.size
    presence = pond_presence(mirage, phase).reshape(-1)[np.minimum(ponds.reshape(-1), none - 1)]
    return (presence * (ponds.reshape(-1) < none)).reshape(ponds.shape)


def palm_sites(
    water: np.ndarray, land: np.ndarray, rolls: np.ndarray, ponds: np.ndarray | None = None
) -> np.ndarray:
    water, land, rolls = np.asarray(water, bool), np.asarray(land, bool), np.asarray(rolls)
    height, width = water.shape
    none = height * width
    pond = (palm_ponds(water, land) if ponds is None else ponds).reshape(-1)
    order = np.argsort(pond.astype(np.float32) * 2.0 + (1.0 - rolls[1].reshape(-1)), kind="stable")
    ordered = pond[order]
    position = np.arange(none)
    starts = np.concatenate([[True], ordered[1:] != ordered[:-1]])
    rank = position - np.maximum.accumulate(np.where(starts, position, 0))
    keep = (ordered < none) & (rank < PALMS_PER_POND + ordered % 2)
    sites = np.zeros(none, bool)
    sites[order] = keep
    return sites.reshape(height, width)


def paint_palms(
    image: np.ndarray,
    sites: np.ndarray,
    rolls: np.ndarray,
    behind_only: bool = False,
    presence: np.ndarray | None = None,
) -> np.ndarray:
    height, width = sites.shape
    bank = palm_bank()
    score = np.where(sites, rolls[1], -1.0).astype(np.float32).reshape(-1)
    flat = np.argsort(-score, kind="stable")[: min(MAX_PALMS, height * width)]
    rows, columns = flat // width, flat % width
    order = np.argsort(rows, kind="stable")
    variants = np.floor(rolls[0].reshape(-1)[flat] * len(PALM_SPRITES)).astype(np.int32) % len(PALM_SPRITES)
    trunk = (np.arange(PALM_HEIGHT) < PALM_HEIGHT - TILE)[:, None, None] if behind_only else 1.0
    size_y, size_x = image.shape[:2]
    for item in order:
        if score[flat[item]] < 0:
            continue
        strength = 1.0 if presence is None else presence[rows[item], columns[item]]
        if strength <= 0:
            continue
        top, left = (rows[item] + 1) * TILE - PALM_HEIGHT, columns[item] * TILE + TILE // 2 - PALM_WIDTH // 2
        y0, y1 = max(top, 0), min(top + PALM_HEIGHT, size_y)
        x0, x1 = max(left, 0), min(left + PALM_WIDTH, size_x)
        sprite = bank[variants[item]][y0 - top : y1 - top, x0 - left : x1 - left]
        alpha = sprite[..., 3:] / 255.0 * strength
        if behind_only:
            alpha = alpha * trunk[y0 - top : y1 - top]
        region = image[y0:y1, x0:x1]
        image[y0:y1, x0:x1] = region * (1.0 - alpha) + sprite[..., :3] * alpha
    return image


def paint_scenery(image: np.ndarray, state: "DuneSeaState", water: np.ndarray) -> np.ndarray:
    height, width = state.cells.wall.shape
    wall = np.asarray(state.cells.wall) > 0
    rolls = scenery_rolls(height, width)
    near_rock = ~wall & (neighbour_mask(wall) > 0)
    free = (
        ~wall
        & ~water
        & (np.asarray(state.cells.bush) == 0)
        & (np.asarray(state.cells.energy) == 0)
        & (np.asarray(state.cells.goal) == 0)
    )
    stone_chance = np.where(near_rock, 0.12, 0.012)
    brush_chance, cactus_chance = 0.022, 0.006
    roll, pick = rolls[0], rolls[1]
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


def drift(layer: np.ndarray, phase: np.float32, speed: float) -> np.ndarray:
    shift = np.round(phase * np.float32(speed) * np.array(WIND, np.float32)).astype(np.int32)
    return np.roll(layer, (shift[0], shift[1]), axis=(0, 1))


def shimmer(image: np.ndarray, haze: np.ndarray, phase: np.float32) -> np.ndarray:
    if not (haze > 0).any():
        return image
    height, width = haze.shape
    field = functools.reduce(np.maximum, shifted(haze, SHIMMER_REACH))
    for _ in range(SHIMMER_SMOOTHING):
        padded = np.pad(field, 1, mode="edge")
        field = sum(padded[dy : dy + height, dx : dx + width] for dy in range(3) for dx in range(3)) / 9.0
    strength = upsample(field, height * TILE, width * TILE)
    active = np.nonzero((strength > 0).any(axis=1))[0]
    rows = np.arange(height * TILE, dtype=np.float32)[:, None]
    columns = np.arange(width * TILE)[None, :]
    sway = np.round(strength[active] * np.float32(SHIMMER_AMPLITUDE) * np.sin(rows[active] * np.float32(SHIMMER_FREQUENCY) + phase * np.float32(SHIMMER_SPEED)))
    source = np.clip(columns + sway.astype(np.int32), 0, width * TILE - 1)
    wavy = np.take_along_axis(image[active], np.broadcast_to(source[..., None], (len(active), width * TILE, 3)), axis=1)
    image[active] = np.where((strength[active] > 0)[..., None], wavy, image[active])
    return image


def upsample(field: np.ndarray, height: int, width: int) -> np.ndarray:
    def weights(size: int, target: int) -> np.ndarray:
        centre = np.clip((np.arange(target) + 0.5) * size / target - 0.5, 0.0, size - 1.0)
        kernel = np.maximum(0.0, 1.0 - np.abs(centre[:, None] - np.arange(size)[None, :]))
        return (kernel / kernel.sum(axis=1, keepdims=True)).astype(np.float32)

    return weights(field.shape[0], height) @ field.astype(np.float32) @ weights(field.shape[1], width).T


def water_phases(mirage: np.ndarray, oasis: np.ndarray, phase: np.float32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    age = phase - mirage
    fake = (mirage > 0) & (age >= 0) & (age < MIRAGE_STEPS)
    lifetime = np.clip(age / np.float32(MIRAGE_STEPS), 0.0, 1.0)
    risen = (mirage < 0) & (oasis > 0)
    rising = np.clip((phase + mirage) / np.float32(FADE_STEPS), 0.0, 1.0)
    opacity = np.where(fake, 1.0 - lifetime, np.where(risen, RISE_OPACITY + (1.0 - RISE_OPACITY) * rising, 1.0))
    haze = np.where(fake, np.sin(np.pi * lifetime), np.where(risen, 1.0 - rising, 0.0))
    return fake, opacity.astype(np.float32), haze.astype(np.float32)


def veil(image: np.ndarray, keep: np.ndarray, haze: np.ndarray) -> np.ndarray:
    for channel, colour in enumerate(STORM_COLOR):
        plane = image[..., channel]
        plane *= keep
        plane += haze * np.float32(colour)
    return image


def paint_ground_storm(image: np.ndarray, phase: np.float32, strength: float) -> np.ndarray:
    far, near, cloud = storm_fields(*image.shape[:2])
    strength = np.float32(strength)
    shadow = np.roll(drift(cloud, phase, 0.4), SHADOW_OFFSET, axis=(0, 1))
    dark = 1.0 - strength * np.float32(SHADOW_STRENGTH) * np.clip((shadow - 0.55) / 0.35, 0.0, 1.0)
    dust = strength * (np.float32(DUST_FAR) * drift(far, phase, 0.7) + np.float32(DUST_NEAR) * drift(near, phase, 1.0))
    return veil(image, dark * (1.0 - dust), dust)


def paint_high_storm(image: np.ndarray, phase: np.float32, strength: float) -> np.ndarray:
    _, near, cloud = storm_fields(*image.shape[:2])
    dust = np.float32(strength) * (np.float32(HIGH_HAZE) * drift(cloud, phase, 0.4) + np.float32(HIGH_STREAKS) * drift(near, phase, 1.6))
    return veil(image, 1.0 - dust, dust)


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
    image: np.ndarray, state: "DuneSeaState", teams: np.ndarray | None = None, offsets: np.ndarray | None = None
) -> np.ndarray:
    marked = teams is not None and bool(np.asarray(teams).any())
    bank = team_bank() if marked else agent_bank()
    coords = (np.arange(TILE) + 0.5) / TILE
    radius = np.hypot(coords[:, None] - 0.5, coords[None, :] - 0.5)
    ring = (radius >= 0.40) & (radius < 0.46)
    position = np.asarray(state.agent.position)
    num_agents = position.shape[0]
    robes = np.arange(num_agents) % bank.shape[0]
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
    direction, broadcast, energy = (np.asarray(x) for x in (state.agent.direction, state.agent.broadcast, state.agent.energy))
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


def paint_fov(image: np.ndarray, fov: np.ndarray, teams: np.ndarray | None) -> np.ndarray:
    fov = np.asarray(fov, bool)
    team_ids = np.zeros(fov.shape[0], np.int32) if teams is None else np.asarray(teams, np.int32)
    for team in sorted(set(team_ids.tolist())):
        seen = (fov & (team_ids == team)[:, None, None]).any(axis=0)
        mask = np.repeat(np.repeat(seen, TILE, axis=0), TILE, axis=1)
        color = TEAM_COLORS[team % len(TEAM_COLORS)]
        image[mask] = image[mask] * (1.0 - FOV_ALPHA) + color * FOV_ALPHA
    return image


def render(
    state: "DuneSeaState",
    scale: int = 1,
    teams: np.ndarray | None = None,
    fov: np.ndarray | None = None,
    storm: float = 1.0,
    offsets: np.ndarray | None = None,
    blend: float = 1.0,
) -> np.ndarray:
    cells = state.cells
    wall = np.asarray(cells.wall) > 0
    height, width = wall.shape
    oasis, mirage, bush = np.asarray(cells.oasis), np.asarray(cells.mirage), np.asarray(cells.bush)
    energy, goal = np.asarray(cells.energy), np.asarray(cells.goal)
    phase = np.float32(int(state.time)) - np.float32(1.0) + np.float32(blend)
    fake, opacity, haze = water_phases(mirage, oasis, phase)
    water = ((oasis > 0) | fake) & ~wall
    habitat = water | ((mirage != 0) & ~wall)
    neighbours = count_neighbours(habitat)
    wet = ~habitat & ~wall & (neighbours > 0)
    image = terrain(wall.tobytes(), height, width).copy()
    image = paint_pond(image, water, opacity)
    if storm:
        image = paint_ground_storm(image, phase, storm)
    image = paint_scenery(image, state, habitat)
    rolls = scenery_rolls(height, width)
    reed_here = wet & (bush == 0) & (rolls[1] < REED_CHANCE)
    reed_kind = np.floor(rolls[0] * len(REED_SPRITES)).astype(np.int32) % len(REED_SPRITES)
    lily_here = water & (neighbours >= 7) & (rolls[0] > 1.0 - LILYPAD_CHANCE)
    lily_kind = np.floor(rolls[1] * len(LILYPAD_SPRITES)).astype(np.int32) % len(LILYPAD_SPRITES)
    for index, name in enumerate(LILYPAD_SPRITES):
        image = paint_sprite(image, name, lily_here & (lily_kind == index), directory=FOREST_DIRECTORY, opacity=opacity)
    for index, name in enumerate(REED_SPRITES):
        image = paint_sprite(image, name, reed_here & (reed_kind == index), directory=FOREST_DIRECTORY)
    belt = np.zeros((height, width), np.float32)
    close = np.zeros((height, width), np.float32)
    for index, near in enumerate(shifted(habitat.astype(np.float32), BELT_REACH)):
        dy, dx = divmod(index, 2 * BELT_REACH + 1)
        belt = np.maximum(belt, near)
        if abs(dy - BELT_REACH) <= 2 and abs(dx - BELT_REACH) <= 2:
            close = np.maximum(close, near)
    tuft_chance = np.where(close > 0, 0.5, 0.2)
    tufts = (belt > 0) & ~habitat & ~wall & ~reed_here & (bush == 0) & (rolls[0] < tuft_chance)
    tuft_kind = np.floor(rolls[1] * len(GRASS_SPRITES) * 3.0).astype(np.int32) % len(GRASS_SPRITES)
    for index, name in enumerate(GRASS_SPRITES):
        image = paint_sprite(image, name, tufts & (tuft_kind == index), directory=FOREST_DIRECTORY)
    ponds = palm_ponds(habitat, ~habitat & ~wall & (bush == 0))
    sites = palm_sites(habitat, ~habitat & ~wall & (bush == 0), rolls, ponds)
    standing = palm_presence(ponds, mirage, phase)
    image = paint_palms(image, sites, rolls, presence=standing)
    image = shimmer(image, haze, phase)
    rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    image = paint_sprite(image, GOAL_SPRITE, goal > 0)
    image = paint_sprite(image, ENERGY_SPRITE, (energy > 0) & (energy < MATURE_ENERGY))
    image = paint_sprite(image, RIPE_SPRITE, energy >= MATURE_ENERGY)
    variety = (rows * ROW_STRIDE + columns * COLUMN_STRIDE) % len(BUSH_SPRITES)
    position = np.asarray(state.agent.position)
    alive = np.asarray(state.agent.energy) > 0
    occupied = np.zeros((height, width), bool)
    occupied[position[alive, 0], position[alive, 1]] = True
    bush = bush > 0
    for index, name in enumerate(BUSH_SPRITES):
        image = paint_sprite(image, name, bush & occupied & (variety == index), BUSH_ALPHA)
    if fov is not None:
        image = paint_fov(image, fov, teams)
    image = paint_agents(image, state, teams, offsets)
    image = paint_palms(image, sites, rolls, behind_only=True, presence=standing)
    for index, name in enumerate(BUSH_SPRITES):
        here = bush & (variety == index)
        image = paint_sprite(image, name, here & ~occupied, BUSH_ALPHA)
        image = paint_sprite(image, name, here & occupied, BUSH_ALPHA, levels=(0,), rows=BUSH_FOREGROUND_ROWS)
    if storm:
        image = paint_high_storm(image, phase, storm)
    image = np.clip(image, 0, 255, out=image).astype(np.uint8)
    if scale > 1:
        from PIL import Image

        size = (image.shape[1] * scale, image.shape[0] * scale)
        image = np.array(Image.fromarray(image).resize(size, Image.NEAREST))
    return image
