import numpy as np

FACING_NORTH = (
    "................",
    "......oooo......",
    ".....ohhhho.....",
    "....ohhhhhho....",
    "....ohhHHhho....",
    "....ohhHHhho....",
    "....oHhhhhHo....",
    "...ooHhhhhHoo...",
    "...oRrRRRRrRo...",
    "..oRrrRRRRrrRo..",
    "..ordrrbbrrdro..",
    "..ordrrrrrrdro..",
    "..odrrrrrrrrdo..",
    "..oddrrrrrrddo..",
    "...ooddddddoo...",
    "................",
)
FACING_SOUTH = (
    "................",
    "......oooo......",
    ".....ohhhho.....",
    "....ohhhhhho....",
    "....ohffffho....",
    "....ofeffefo....",
    "....ohffffho....",
    "...ooHhhhhHoo...",
    "...oRrRRRRrRo...",
    "..oRrrRRRRrrRo..",
    "..ordrrbbrrdro..",
    "..ordrrrrrrdro..",
    "..odrrrrrrrrdo..",
    "..oddrrrrrrddo..",
    "...ooddddddoo...",
    "................",
)
FACING_EAST = (
    "................",
    ".....oooo.......",
    "....ohhhhoo.....",
    "...ohhhhhhho....",
    "...ohhhhhffo....",
    "...ohhhhhfeo....",
    "...ohhhhhffo....",
    "...oHhhhhhoo....",
    "...oRRrRRRo.....",
    "..ooRrrrRRRo....",
    "..odrrrrbbRo....",
    "..odrrrrrrRo....",
    "..odrrrrrrrRo...",
    "..oddrrrrrrro...",
    "...oodddddoo....",
    "................",
)
SHADOW = (
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "..............ss",
    "...ssssssssssss.",
    "....ssssssssss..",
)
OUTLINE_COLOR = (69, 42, 27)
HOOD_COLORS = {
    "h": (221, 206, 191),
    "H": (189, 164, 153),
    "f": (51, 40, 34),
    "e": (248, 197, 58),
    "b": (198, 184, 88),
}
ROBE_PALETTES = (
    {"R": (140, 91, 62), "r": (114, 75, 44), "d": (79, 52, 47)},
    {"R": (127, 142, 68), "r": (88, 99, 53), "d": (51, 60, 36)},
    {"R": (136, 110, 106), "r": (110, 88, 86), "d": (89, 77, 77)},
    {"R": (176, 96, 64), "r": (146, 72, 50), "d": (104, 50, 38)},
)
SHADOW_COLOR = (40, 28, 18, 90)


def paint_pixels(rows: tuple[str, ...], palette: dict) -> np.ndarray:
    sprite = np.zeros((len(rows), len(rows[0]), 4), np.float32)
    for y, row in enumerate(rows):
        for x, symbol in enumerate(row):
            if symbol in palette:
                color = palette[symbol]
                sprite[y, x] = (*color[:3], color[3] if len(color) == 4 else 255)
    return sprite


def layer_over(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    upper_alpha, lower_alpha = upper[..., 3:] / 255.0, lower[..., 3:] / 255.0
    alpha = upper_alpha + lower_alpha * (1.0 - upper_alpha)
    color = upper[..., :3] * upper_alpha + lower[..., :3] * lower_alpha * (1.0 - upper_alpha)
    color = np.where(alpha > 0, color / np.maximum(alpha, 1e-6), 0.0)
    return np.concatenate([color, alpha * 255.0], axis=-1)


def wanderer_sprites() -> np.ndarray:
    shadow = paint_pixels(SHADOW, {"s": SHADOW_COLOR})
    variants = []
    for robe in ROBE_PALETTES:
        palette = {"o": OUTLINE_COLOR, **HOOD_COLORS, **robe}
        east = paint_pixels(FACING_EAST, palette)
        poses = (paint_pixels(FACING_NORTH, palette), east, paint_pixels(FACING_SOUTH, palette), east[:, ::-1])
        variants.append(np.stack([layer_over(shadow, pose) for pose in poses]))
    return np.stack(variants)
