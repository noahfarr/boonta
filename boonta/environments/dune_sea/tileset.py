from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from boonta.utils import Array

TILESET_DIRECTORY = Path(__file__).parent / "assets" / "glionox_desert"
TILE_PIXELS = 16
OVERHANG_ROWS = 6
DETAIL_OFFSET = 6
FILL_ROWS = (12, 13, 14)
SHEET_COLUMNS = 10
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
FULL_MASK = 255


def tileset_available() -> bool:
    return (TILESET_DIRECTORY / "sand.png").exists() and (TILESET_DIRECTORY / "sandstone.png").exists()


def read_sheet(name: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(TILESET_DIRECTORY / name).convert("RGBA")).astype(np.float32)


def cut_tile(sheet: np.ndarray, row: int, col: int) -> np.ndarray:
    return sheet[row * TILE_PIXELS : (row + 1) * TILE_PIXELS, col * TILE_PIXELS : (col + 1) * TILE_PIXELS]


def stack_layers(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    lower_alpha, upper_alpha = lower[..., 3:] / 255.0, upper[..., 3:] / 255.0
    alpha = upper_alpha + lower_alpha * (1.0 - upper_alpha)
    color = upper[..., :3] * upper_alpha + lower[..., :3] * lower_alpha * (1.0 - upper_alpha)
    color = np.where(alpha > 0, color / np.maximum(alpha, 1e-6), 0.0)
    return np.concatenate([color, alpha * 255.0], axis=-1)


def visual_mask(tile: np.ndarray) -> int:
    covered = tile[..., 3] > 0
    return sum(1 << bit for bit, region in enumerate(REGION_SLICES) if covered[region].mean() > 0.5)


def fill_tiles(sheet: np.ndarray) -> np.ndarray:
    return np.stack([cut_tile(sheet, row, col) for row in FILL_ROWS for col in range(SHEET_COLUMNS)])


def reachable_masks() -> list[int]:
    masks = set()
    for raw in range(256):
        mask = raw
        for bit in (1, 3, 5, 7):
            if raw & (1 << (bit - 1)) or raw & (1 << ((bit + 1) % 8)):
                mask |= 1 << bit
        masks.add(mask)
    return sorted(masks)


def overhang_tiles(sheet: np.ndarray) -> np.ndarray:
    table = np.zeros((256, TILE_PIXELS, TILE_PIXELS, 4), np.float32)
    filled = set()
    for row in range(OVERHANG_ROWS):
        for col in range(SHEET_COLUMNS):
            tile = stack_layers(cut_tile(sheet, row + DETAIL_OFFSET, col), cut_tile(sheet, row, col))
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


FLORA_SPRITES = (
    ("tree_palm1.png", 3),
    ("tree_palm2.png", 3),
    ("tree_palm3.png", 3),
    ("tree_palm4.png", 3),
    ("decor_cactus1.png", 2),
    ("decor_cactus2.png", 2),
    ("decor_cactus3.png", 1),
    ("decor_cactus4.png", 1),
    ("decor_cactus5.png", 1),
    ("decor_cactus2_pink.png", 1),
    ("decor_cactus4_white.png", 1),
    ("decor_stone_large1.png", 2),
    ("decor_stone_large2.png", 2),
)
DETAIL_SPRITES = (
    "decor_stone_small1.png",
    "decor_stone_small2.png",
    "decor_stone_small3.png",
    "decor_stone_small4.png",
    "decor_dead_bush1.png",
    "decor_dead_bush2.png",
    "decor_dead_bush3.png",
    "decor_dead_bush4.png",
    "decor_dead_bush_small.png",
    "decor_cactus_small1.png",
    "decor_cactus_small2.png",
    "decor_cactus_small1_pink.png",
)
SPRITE_HEIGHT = 112
SPRITE_WIDTH = 48
DETAIL_CHANCE = 0.06


def anchor_sprite(sprite: np.ndarray) -> np.ndarray:
    canvas = np.zeros((SPRITE_HEIGHT, SPRITE_WIDTH, 4), np.float32)
    height, width = sprite.shape[:2]
    left = (SPRITE_WIDTH - width) // 2
    canvas[SPRITE_HEIGHT - height :, left : left + width] = sprite
    return canvas


@cache
def load_tileset() -> dict:
    rock_sheet = read_sheet("dry_dirt.png")
    patch_sheet = read_sheet("sandstone.png")
    flora = [anchor_sprite(read_sheet(name)) for name, weight in FLORA_SPRITES for _ in range(weight)]
    details = [np.zeros((TILE_PIXELS, TILE_PIXELS, 4), np.float32)]
    details += [read_sheet(name) for name in DETAIL_SPRITES]
    return {
        "sand": jnp.asarray(fill_tiles(read_sheet("sand.png"))[..., :3]),
        "rock": jnp.asarray(fill_tiles(rock_sheet)[..., :3]),
        "overhang": jnp.asarray(overhang_tiles(rock_sheet)),
        "patch": jnp.asarray(fill_tiles(patch_sheet)[..., :3]),
        "patch_edges": jnp.asarray(overhang_tiles(patch_sheet)),
        "flora": jnp.asarray(np.stack(flora)),
        "details": jnp.asarray(np.stack(details)),
    }


def neighbour_mask(is_rock: Array) -> Array:
    flags = [jnp.roll(is_rock, (-dy, -dx), axis=(0, 1)) for dx, dy in NEIGHBOUR_STEPS]
    orthogonal = {0: flags[0], 2: flags[2], 4: flags[4], 6: flags[6]}
    mask = jnp.zeros(is_rock.shape, jnp.int32)
    for bit, flag in enumerate(flags):
        if bit % 2 == 1:
            flag = flag | orthogonal[bit - 1] | orthogonal[(bit + 1) % 8]
        mask = mask | (flag.astype(jnp.int32) << bit)
    return mask


def blend_over(base: Array, layer: Array) -> Array:
    alpha = layer[..., 3:] / 255.0
    return base * (1.0 - alpha) + layer[..., :3] * alpha


def tile_frame(tiles: Array) -> Array:
    height, width = tiles.shape[:2]
    return tiles.transpose(0, 2, 1, 3, 4).reshape(height * TILE_PIXELS, width * TILE_PIXELS, -1)


def draw_objects(frame: Array, rows: Array, cols: Array, sprite_ids: Array, count: Array, bank: Array) -> Array:
    sprite_height, sprite_width = bank.shape[1:3]
    pad_top, pad_side = sprite_height - TILE_PIXELS, (sprite_width - TILE_PIXELS) // 2
    canvas = jnp.pad(frame, ((pad_top, 0), (pad_side, pad_side), (0, 0)))

    def draw_object(index, canvas):
        top, left = rows[index] * TILE_PIXELS, cols[index] * TILE_PIXELS
        sprite = bank[sprite_ids[index]]
        patch = jax.lax.dynamic_slice(canvas, (top, left, 0), (sprite_height, sprite_width, 3))
        return jax.lax.dynamic_update_slice(canvas, blend_over(patch, sprite), (top, left, 0))

    canvas = jax.lax.fori_loop(0, count, draw_object, canvas)
    height, width = frame.shape[:2]
    return canvas[pad_top:, pad_side : pad_side + width]


def paint_scenery(cells: Array, is_patch: Array, variety: Array, tileset: dict) -> Array:
    is_rock = cells == 1
    is_patch = is_patch & ~is_rock
    fill_variant = (variety % tileset["sand"].shape[0]).astype(jnp.int32)
    ground = tileset["sand"][fill_variant]
    ground = blend_over(ground, tileset["patch_edges"][neighbour_mask(is_patch)])
    ground = jnp.where(is_patch[..., None, None, None], tileset["patch"][fill_variant], ground)
    detail_roll = ((variety >> 8) % 1000).astype(jnp.float32) / 1000.0
    num_details = tileset["details"].shape[0] - 1
    detail_kind = 1 + ((variety >> 4) % num_details).astype(jnp.int32)
    has_detail = (detail_roll < DETAIL_CHANCE) & (cells == 0)
    ground = blend_over(ground, tileset["details"][jnp.where(has_detail, detail_kind, 0)])
    ground = blend_over(ground, tileset["overhang"][neighbour_mask(is_rock)])
    tiles = jnp.where(is_rock[..., None, None, None], tileset["rock"][fill_variant], ground)
    return tile_frame(tiles)

