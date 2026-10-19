from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from boonta.utils import Array

TILESET_DIRECTORY = Path(__file__).parent / "assets" / "glionox_desert"
WATER_DIRECTORY = Path(__file__).parent / "assets" / "glionox_forest"
PLAIN_WATER_TILE = (3, 9)
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


def water_available() -> bool:
    return (WATER_DIRECTORY / "water.png").exists()


def read_sheet(name: str, directory: Path = TILESET_DIRECTORY) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(directory / name).convert("RGBA")).astype(np.float32)


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


def overhang_tiles(sheet: np.ndarray, layered: bool = True) -> np.ndarray:
    table = np.zeros((256, TILE_PIXELS, TILE_PIXELS, 4), np.float32)
    filled = set()
    for row in range(OVERHANG_ROWS):
        for col in range(SHEET_COLUMNS):
            tile = cut_tile(sheet, row, col)
            if layered:
                tile = stack_layers(cut_tile(sheet, row + DETAIL_OFFSET, col), tile)
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


PALM_SPRITES = ("tree_palm1.png", "tree_palm2.png", "tree_palm3.png", "tree_palm4.png")
CACTUS_SPRITES = (
    "decor_cactus1.png",
    "decor_cactus2.png",
    "decor_cactus3.png",
    "decor_cactus4.png",
    "decor_cactus5.png",
    "decor_cactus2_pink.png",
    "decor_cactus4_white.png",
)
BOULDER_SPRITES = ("decor_stone_large1.png", "decor_stone_large2.png", "decor_stone_medium1.png", "decor_stone_medium2.png")
STONE_DETAILS = ("decor_stone_small1.png", "decor_stone_small2.png", "decor_stone_small3.png", "decor_stone_small4.png")
DRY_DETAILS = (
    "decor_dead_bush1.png",
    "decor_dead_bush2.png",
    "decor_dead_bush3.png",
    "decor_dead_bush4.png",
    "decor_dead_bush_small.png",
)
GREEN_DETAILS = ("decor_cactus_small1.png", "decor_cactus_small2.png", "decor_cactus_small1_pink.png")
SPRITE_HEIGHT = 112
SPRITE_WIDTH = 48
DETAIL_CHANCE = 0.06
REED_SPRITES = ("decor_reed1.png", "decor_reed2.png", "decor_reed3.png", "decor_reed4.png")
LILYPAD_SPRITES = tuple(f"decor_lilypad{index}.png" for index in range(1, 9))
LILYPAD_CHANCE = 0.12
WATER_COLOR = np.array([95, 132, 128], np.float32)


def pad_detail(sprite: np.ndarray) -> np.ndarray:
    return sprite[-TILE_PIXELS:, :TILE_PIXELS]


def sprite_groups() -> dict:
    counts = {"palm": len(PALM_SPRITES), "cactus": len(CACTUS_SPRITES), "boulder": len(BOULDER_SPRITES)}
    details = {"stone": len(STONE_DETAILS), "dry": len(DRY_DETAILS), "green": len(GREEN_DETAILS)}
    return {"flora": counts, "details": details}


def anchor_sprite(sprite: np.ndarray) -> np.ndarray:
    canvas = np.zeros((SPRITE_HEIGHT, SPRITE_WIDTH, 4), np.float32)
    height, width = sprite.shape[:2]
    left = (SPRITE_WIDTH - width) // 2
    canvas[SPRITE_HEIGHT - height :, left : left + width] = sprite
    return canvas


@cache
def read_tileset() -> dict:
    rock_sheet = read_sheet("dry_dirt.png")
    patch_sheet = read_sheet("sandstone.png")
    flora = [anchor_sprite(read_sheet(name)) for name in PALM_SPRITES + CACTUS_SPRITES + BOULDER_SPRITES]
    details = [np.zeros((TILE_PIXELS, TILE_PIXELS, 4), np.float32)]
    details += [pad_detail(read_sheet(name)) for name in STONE_DETAILS + DRY_DETAILS + GREEN_DETAILS]
    water = {
        "water": np.asarray(np.broadcast_to(WATER_COLOR, (TILE_PIXELS, TILE_PIXELS, 3)).copy()),
        "shore": np.asarray(overhang_tiles(read_sheet("sand_to_water.png"), layered=False)),
        "reeds": np.zeros((0, SPRITE_HEIGHT, SPRITE_WIDTH, 4)),
        "lilypads": np.zeros((1, TILE_PIXELS, TILE_PIXELS, 4)),
    }
    if water_available():
        water["water"] = np.asarray(cut_tile(read_sheet("water.png", WATER_DIRECTORY), *PLAIN_WATER_TILE)[..., :3])
        reeds = [anchor_sprite(read_sheet(name, WATER_DIRECTORY)) for name in REED_SPRITES]
        water["reeds"] = np.asarray(np.stack(reeds))
        pads = [np.zeros((TILE_PIXELS, TILE_PIXELS, 4), np.float32)]
        pads += [read_sheet(name, WATER_DIRECTORY) for name in LILYPAD_SPRITES]
        water["lilypads"] = np.asarray(np.stack(pads))
    return {
        **water,
        "sand": np.asarray(fill_tiles(read_sheet("sand.png"))[..., :3]),
        "rock": np.asarray(fill_tiles(rock_sheet)[..., :3]),
        "overhang": np.asarray(overhang_tiles(rock_sheet)),
        "patch": np.asarray(fill_tiles(patch_sheet)[..., :3]),
        "patch_edges": np.asarray(overhang_tiles(patch_sheet)),
        "flora": np.asarray(np.stack(flora)),
        "details": np.asarray(np.stack(details)),
    }


def load_tileset() -> dict:
    return {name: jnp.asarray(value) for name, value in read_tileset().items()}


def distance_steps(mask: Array, max_distance: int) -> Array:
    def grow_ring(step, carry):
        reached, distance = carry
        grown = (
            reached
            | jnp.roll(reached, 1, 0)
            | jnp.roll(reached, -1, 0)
            | jnp.roll(reached, 1, 1)
            | jnp.roll(reached, -1, 1)
        )
        return grown, jnp.where(grown & ~reached, step + 1, distance)

    _, distance = jax.lax.fori_loop(0, max_distance - 1, grow_ring, (mask, jnp.where(mask, 0, max_distance)))
    return distance


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


def fold_overhang(canvas: Array, original: Array, pad_top: int, pad_side: int) -> Array:
    changed = jnp.any(canvas != original, axis=-1, keepdims=True)
    if pad_top:
        canvas = canvas.at[-pad_top:].set(
            jnp.where(changed[:pad_top], canvas[:pad_top], canvas[-pad_top:])
        )
        changed = changed.at[-pad_top:].set(changed[:pad_top] | changed[-pad_top:])
    if pad_side:
        width = canvas.shape[1] - 2 * pad_side
        left, right = slice(0, pad_side), slice(pad_side + width, None)
        wrapped_left, wrapped_right = slice(width, width + pad_side), slice(pad_side, 2 * pad_side)
        canvas = canvas.at[:, wrapped_left].set(
            jnp.where(changed[:, left], canvas[:, left], canvas[:, wrapped_left])
        )
        canvas = canvas.at[:, wrapped_right].set(
            jnp.where(changed[:, right], canvas[:, right], canvas[:, wrapped_right])
        )
    return canvas


def draw_objects(frame: Array, rows: Array, cols: Array, sprite_ids: Array, count: Array, bank: Array) -> Array:
    sprite_height, sprite_width = bank.shape[1:3]
    pad_top, pad_side = sprite_height - TILE_PIXELS, (sprite_width - TILE_PIXELS) // 2
    original = jnp.pad(frame, ((pad_top, 0), (pad_side, pad_side), (0, 0)), mode="wrap")

    def draw_object(index, canvas):
        top, left = rows[index] * TILE_PIXELS, cols[index] * TILE_PIXELS
        sprite = bank[sprite_ids[index]]
        patch = jax.lax.dynamic_slice(canvas, (top, left, 0), (sprite_height, sprite_width, 3))
        return jax.lax.dynamic_update_slice(canvas, blend_over(patch, sprite), (top, left, 0))

    canvas = jax.lax.fori_loop(0, count, draw_object, original)
    canvas = fold_overhang(canvas, original, pad_top, pad_side)
    height, width = frame.shape[:2]
    return canvas[pad_top:, pad_side : pad_side + width]


def paint_scenery(cells: Array, is_patch: Array, variety: Array, tileset: dict) -> Array:
    is_rock, is_water = cells == 1, cells == 3
    is_patch = is_patch & ~is_rock
    fill_variant = (variety % tileset["sand"].shape[0]).astype(jnp.int32)
    ground = tileset["sand"][fill_variant]
    ground = blend_over(ground, tileset["patch_edges"][neighbour_mask(is_patch)])
    ground = jnp.where(is_patch[..., None, None, None], tileset["patch"][fill_variant], ground)
    water_distance = distance_steps(cells == 3, 8)
    rock_distance = distance_steps(is_rock, 3)
    groups = sprite_groups()["details"]
    first_dry = 1 + groups["stone"]
    first_green = first_dry + groups["dry"]
    pick = (variety >> 4).astype(jnp.int32)
    stone_kind = 1 + pick % groups["stone"]
    dry_kind = first_dry + pick % groups["dry"]
    green_kind = first_green + pick % groups["green"]
    detail_kind = jnp.where(rock_distance <= 2, stone_kind, jnp.where(water_distance <= 6, green_kind, dry_kind))
    detail_chance = jnp.where(rock_distance <= 2, 2.0 * DETAIL_CHANCE, DETAIL_CHANCE)
    detail_roll = ((variety >> 8) % 1000).astype(jnp.float32) / 1000.0
    has_detail = (detail_roll < detail_chance) & (cells == 0)
    ground = blend_over(ground, tileset["details"][jnp.where(has_detail, detail_kind, 0)])
    ground = blend_over(ground, tileset["overhang"][neighbour_mask(is_rock)])
    tiles = jnp.where(is_rock[..., None, None, None], tileset["rock"][fill_variant], ground)
    num_pads = tileset["lilypads"].shape[0] - 1
    pad_roll = ((variety >> 12) % 1000).astype(jnp.float32) / 1000.0
    pad_kind = jnp.where((pad_roll < LILYPAD_CHANCE) & (num_pads > 0), 1 + (variety >> 2) % max(num_pads, 1), 0)
    pond = blend_over(jnp.broadcast_to(tileset["water"], tiles.shape), tileset["lilypads"][pad_kind.astype(jnp.int32)])
    pond = blend_over(pond, tileset["shore"][neighbour_mask(~is_water)])
    tiles = jnp.where(is_water[..., None, None, None], pond, tiles)
    return tile_frame(tiles)

