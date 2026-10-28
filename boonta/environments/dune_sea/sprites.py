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


SIZE = 16
INK = (46, 28, 22)


def blank() -> np.ndarray:
    return np.zeros((SIZE, SIZE, 4), np.float32)


def put(canvas: np.ndarray, mask: np.ndarray, color) -> None:
    canvas[mask] = (*color[:3], 255)


def ellipse(canvas, cx, cy, rx, ry, color) -> None:
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    put(canvas, ((xs + 0.5 - cx) / rx) ** 2 + ((ys + 0.5 - cy) / ry) ** 2 <= 1.0, color)


def rect(canvas, x0, y0, x1, y1, color) -> None:
    mask = np.zeros((SIZE, SIZE), bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    put(canvas, mask, color)


def stroke(canvas, points, color, width=1) -> None:
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        for t in np.linspace(0.0, 1.0, 24):
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            rect(canvas, int(round(x)), int(round(y)), int(round(x)) + width - 1, int(round(y)) + width - 1, color)


def dots(canvas, points, color) -> None:
    for x, y in points:
        if 0 <= x < SIZE and 0 <= y < SIZE:
            canvas[y, x] = (*color[:3], 255)


def outlined(canvas: np.ndarray) -> np.ndarray:
    solid = canvas[..., 3] > 0
    grown = np.zeros_like(solid)
    grown[1:] |= solid[:-1]
    grown[:-1] |= solid[1:]
    grown[:, 1:] |= solid[:, :-1]
    grown[:, :-1] |= solid[:, 1:]
    edge = grown & ~solid
    canvas = canvas.copy()
    canvas[edge] = (*INK, 255)
    return canvas


JAWA_COLORS = (
    {"robe": (112, 82, 56), "dark": (74, 52, 36), "light": (142, 108, 76), "face": (10, 7, 7), "eye": (255, 214, 64), "belt": (52, 38, 28)},
    {"robe": (128, 78, 52), "dark": (86, 50, 34), "light": (158, 102, 70), "face": (10, 7, 7), "eye": (255, 190, 50), "belt": (52, 38, 28)},
)
TUSKEN_COLORS = (
    {"wrap": (214, 192, 148), "dark": (160, 136, 96), "cloth": (182, 156, 112), "goggle": (26, 20, 22), "glint": (255, 176, 60), "strap": (82, 56, 40), "staff": (96, 66, 40), "metal": (176, 178, 190)},
    {"wrap": (168, 176, 182), "dark": (118, 126, 134), "cloth": (136, 144, 152), "goggle": (26, 20, 22), "glint": (120, 220, 255), "strap": (70, 56, 50), "staff": (96, 66, 40), "metal": (176, 178, 190)},
)


def jawa(c: dict, facing: str) -> np.ndarray:
    if facing == "west":
        return jawa(c, "east")[:, ::-1].copy()
    k = blank()
    if facing in ("south", "north"):
        ellipse(k, 7.5, 11.4, 4.8, 3.4, c["robe"])
        rect(k, 3, 11, 12, 14, c["robe"])
        rect(k, 3, 14, 12, 14, c["dark"])
        rect(k, 4, 11, 11, 11, c["belt"])
        ellipse(k, 7.5, 7.0, 4.4, 4.2, c["light"])
        dots(k, [(7, 2), (8, 2)], c["light"])
        if facing == "south":
            ellipse(k, 7.5, 7.6, 2.9, 2.7, c["face"])
            dots(k, [(5, 7), (6, 7), (9, 7), (10, 7)], c["eye"])
            dots(k, [(5, 12), (6, 13), (7, 13)], c["dark"])
        else:
            ellipse(k, 7.5, 8.0, 3.2, 3.2, c["robe"])
            rect(k, 7, 5, 8, 11, c["dark"])
    else:
        ellipse(k, 6.4, 11.4, 4.4, 3.4, c["robe"])
        rect(k, 2, 11, 10, 14, c["robe"])
        rect(k, 2, 14, 10, 14, c["dark"])
        rect(k, 3, 11, 10, 11, c["belt"])
        ellipse(k, 8.2, 7.0, 4.4, 4.2, c["light"])
        dots(k, [(7, 2), (8, 2)], c["light"])
        ellipse(k, 10.4, 7.6, 2.4, 2.6, c["face"])
        dots(k, [(11, 7), (12, 7)], c["eye"])
        rect(k, 9, 12, 12, 12, c["dark"])
    return outlined(k)


def tusken(c: dict, facing: str) -> np.ndarray:
    if facing == "west":
        return tusken(c, "east")[:, ::-1].copy()
    k = blank()
    if facing in ("south", "north"):
        rect(k, 13, 1, 13, 15, c["staff"])
        dots(k, [(12, 0), (14, 0), (13, 0), (12, 1), (14, 1)], c["metal"])
        ellipse(k, 7.0, 10.6, 5.0, 4.6, c["cloth"])
        rect(k, 3, 11, 10, 11, c["strap"])
        rect(k, 3, 14, 10, 14, c["dark"])
        stroke(k, [(3, 8), (10, 13)], c["strap"], 1)
        ellipse(k, 7.0, 5.0, 3.8, 3.6, c["wrap"])
        for x in (4, 5, 6, 7, 8, 9):
            dots(k, [(x, 3)], c["dark"])
        if facing == "south":
            rect(k, 4, 4, 5, 5, c["goggle"]); rect(k, 8, 4, 9, 5, c["goggle"])
            dots(k, [(4, 4), (8, 4)], c["glint"])
            rect(k, 6, 7, 7, 8, c["strap"])
        else:
            rect(k, 4, 6, 9, 6, c["dark"])
    else:
        stroke(k, [(8, 9), (15, 9)], c["staff"], 1)
        dots(k, [(14, 7), (14, 8), (15, 8), (14, 10), (15, 10), (15, 9)], c["metal"])
        ellipse(k, 6.0, 10.6, 4.8, 4.6, c["cloth"])
        rect(k, 2, 11, 9, 11, c["strap"])
        rect(k, 2, 14, 9, 14, c["dark"])
        stroke(k, [(2, 8), (9, 13)], c["strap"], 1)
        ellipse(k, 7.6, 5.0, 3.8, 3.6, c["wrap"])
        for x in (5, 6, 7, 8, 9):
            dots(k, [(x, 3)], c["dark"])
        rect(k, 9, 4, 10, 5, c["goggle"]); dots(k, [(9, 4)], c["glint"])
        rect(k, 10, 6, 11, 7, c["strap"])
    return outlined(k)


def creature_sprites(draw, colors) -> np.ndarray:
    shadow = paint_pixels(SHADOW, {"s": SHADOW_COLOR})
    variants = []
    for palette in colors:
        poses = tuple(draw(palette, facing) for facing in ("north", "east", "south", "west"))
        variants.append(np.stack([layer_over(shadow, pose) for pose in poses]))
    return np.stack(variants)


def team_sprites() -> np.ndarray:
    return np.concatenate(
        [creature_sprites(jawa, JAWA_COLORS), creature_sprites(tusken, TUSKEN_COLORS)]
    )
