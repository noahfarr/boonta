from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp
import lox

from boonta.utils import Array, Key

from ...wrappers.wrapper import Wrapper
from .. import byte, core, flag
from .flags import FLAGS
from .maps import DIMS, NAMES

MAP = 0xD35E
X = 0xD362
Y = 0xD361
BADGES = 0xD356
PARTY_SIZE = 0xD163
HEALTH = (0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248)
SPECIES = 0xD164
LEVELS = (0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268)
MAX_HEALTH = tuple(address + 1 for address in LEVELS)
EXPERIENCE = tuple(address - 19 for address in LEVELS)
EVENTS = (0xD747, 0xD87E)
OWNED = (0xD2F7, 0xD309)
TILEMAP = 0xC3A0
BORDER_CORNER, BORDER_EDGE = 0x79, 0x7A
LEADER_LEVELS = (14, 21, 24, 29, 43, 43, 47, 50, 65)
STARTER_LEVEL = 5
KEYS = (
    "badges",
    "levels",
    "level",
    "events",
    "progress",
    "starter",
    "menu",
    "health",
    "maps",
    "tiles",
    "deaths",
    "healed",
    "experience",
    "caught",
)

NUM_MAPS = len(NAMES)
WIDTHS = jnp.asarray([DIMS[name][0] for name in NAMES], jnp.int32)
HEIGHTS = jnp.asarray([DIMS[name][1] for name in NAMES], jnp.int32)
OFFSETS = jnp.cumsum(WIDTHS * HEIGHTS) - WIDTHS * HEIGHTS
NUM_CELLS = int(jnp.sum(WIDTHS * HEIGHTS))


def map_id(ram):
    return byte(ram, MAP)


def x_position(ram):
    return byte(ram, X)


def y_position(ram):
    return byte(ram, Y)


def badges(ram):
    held = byte(ram, BADGES)
    return sum((held >> bit) & 1 for bit in range(8))


def party_size(ram):
    return byte(ram, PARTY_SIZE)


def starter(ram):
    return byte(ram, SPECIES)


def levels(ram):
    return sum(byte(ram, address) for address in LEVELS)


def level(ram):
    held = jnp.stack([byte(ram, address) for address in LEVELS], axis=-1)
    filled = jnp.arange(len(LEVELS)) < party_size(ram)[..., None]
    return jnp.max(jnp.where(filled, held, 0), axis=-1)


def cap(ram):
    return jnp.asarray(LEADER_LEVELS, jnp.int32)[
        jnp.clip(badges(ram), 0, len(LEADER_LEVELS) - 1)
    ]


def leveling(ram):
    held = jnp.stack([byte(ram, address) for address in LEVELS], axis=-1)
    filled = jnp.arange(len(LEVELS)) < party_size(ram)[..., None]
    raised = jnp.sum(jnp.where(filled, jnp.minimum(held, cap(ram)[..., None]), 0), axis=-1)
    return jnp.maximum(raised - STARTER_LEVEL, 0)


def earned(ram):
    held = jnp.stack(
        [
            byte(ram, address) * 65536 + byte(ram, address + 1) * 256 + byte(ram, address + 2)
            for address in EXPERIENCE
        ],
        axis=-1,
    )
    filled = jnp.arange(len(EXPERIENCE)) < party_size(ram)[..., None]
    ceiling = (cap(ram) ** 3)[..., None]
    return jnp.sum(jnp.where(filled, jnp.minimum(held, ceiling), 0), axis=-1).astype(
        jnp.float32
    )


def tending(health, ailing, steady, sunk, downed):
    mended = (health > ailing) & steady & ~downed
    return mended, jnp.where(health > 0.01, False, sunk | downed)


def caught(ram):
    lo, hi = OWNED
    held = ram[..., lo - 0xC000 : hi - 0xC000 + 1].astype(jnp.int32)
    owned = jnp.sum(sum((held >> bit) & 1 for bit in range(8)), axis=-1)
    return jnp.where(party_size(ram) > 0, owned, 0)


def health(ram, addresses):
    held = jnp.stack([byte(ram, a) * 256 + byte(ram, a + 1) for a in addresses], axis=-1)
    filled = jnp.arange(len(addresses)) < party_size(ram)[..., None]
    return jnp.where(filled, held, 0)


def capacity(ram):
    return jnp.sum(health(ram, MAX_HEALTH), axis=-1)


def vitality(ram):
    held = jnp.sum(health(ram, HEALTH), axis=-1)
    full = capacity(ram)
    return jnp.where(full > 0, held / jnp.maximum(full, 1), 0.0)


def fainted(ram):
    filled = jnp.arange(len(HEALTH)) < party_size(ram)[..., None]
    empty = jnp.all(jnp.where(filled, health(ram, HEALTH) == 0, True), axis=-1)
    return (party_size(ram) > 0) & empty


def lit(ram):
    lo, hi = EVENTS
    held = ram[..., lo - 0xC000 : hi - 0xC000 + 1].astype(jnp.uint32)
    held = held.reshape(*held.shape[:-1], -1, 4)
    return held[..., 0] | (held[..., 1] << 8) | (held[..., 2] << 16) | (held[..., 3] << 24)


def bits(ram):
    held = lit(ram)
    return ((held[..., None] >> jnp.arange(32, dtype=jnp.uint32)) & 1).reshape(
        *held.shape[:-1], -1
    ).astype(jnp.uint8)


def events(ram):
    lo, hi = EVENTS
    held = ram[..., lo - 0xC000 : hi - 0xC000 + 1].astype(jnp.int32)
    return jnp.sum(sum((held >> bit) & 1 for bit in range(8)), axis=-1)


def progress(ram):
    return levels(ram) + events(ram) + 10 * badges(ram)


def vitals(ram):
    worn = ((byte(ram, BADGES) >> jnp.arange(8)) & 1).astype(jnp.float32)
    return jnp.concatenate(
        [
            jnp.atleast_1d(vitality(ram)),
            jnp.atleast_1d(level(ram) / 100.0).astype(jnp.float32),
            jnp.atleast_1d(party_size(ram) / 6.0).astype(jnp.float32),
            worn,
        ],
        axis=-1,
    )


def place(ram):
    return jnp.clip(map_id(ram), 0, NUM_MAPS - 1)


def extent(ram):
    here = place(ram)
    return (WIDTHS[here] * HEIGHTS[here]).astype(jnp.float32) / 360.0


def walk(ram):
    here = place(ram)
    x = jnp.clip(x_position(ram), 0, WIDTHS[here] - 1)
    y = jnp.clip(y_position(ram), 0, HEIGHTS[here] - 1)
    return OFFSETS[here] + y * WIDTHS[here] + x


def crop(seen, ram, height, width):
    here = place(ram)
    wide, tall = WIDTHS[here], HEIGHTS[here]
    down = y_position(ram) + (jnp.arange(height) - height // 2)[:, None]
    across = x_position(ram) + (jnp.arange(width) - width // 2)[None, :]
    inside = (across >= 0) & (across < wide) & (down >= 0) & (down < tall)
    index = OFFSETS[here] + jnp.clip(down, 0, tall - 1) * wide + jnp.clip(across, 0, wide - 1)
    word = seen[index // 32]
    bit = jnp.uint32(1) << (index % 32).astype(jnp.uint32)
    return jnp.where(inside & ((word & bit) > 0), 255, 0).astype(jnp.uint8)


def menu(ram):
    corner = byte(ram, TILEMAP + 10) == BORDER_CORNER
    edge = byte(ram, TILEMAP + 11) == BORDER_EDGE
    return (corner & edge).astype(jnp.int32)


def annotate(ram):
    return {
        "badges": badges(ram),
        "levels": levels(ram),
        "level": level(ram),
        "events": events(ram),
        "progress": progress(ram),
        "starter": starter(ram),
        "menu": menu(ram),
        "health": vitality(ram),
    }


class LogFlags(Wrapper):
    def __init__(self, env, tags: Iterable[str] = ("training",)):
        super().__init__(env)
        self.tags = tags

    def step(self, key: Key, state: Any, action: Array) -> tuple[Any, Any]:
        state, timestep = super().step(key, state, action)
        ram = core(state).ram
        lox.log(
            {
                f"flags/{name}": jnp.mean(flag(ram, EVENTS[0], bit).astype(jnp.float32))
                for name, bit in FLAGS
            },
            tags=self.tags,
        )
        return state, timestep
