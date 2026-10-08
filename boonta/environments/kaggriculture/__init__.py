import ctypes
from functools import partial, reduce
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from boonta.utils import Key, Timestep

from .. import build, load as load_library, register_targets
from ..environment import Environment
from ..spaces import Space

PRODUCTS = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
]
ANIMAL_NAMES = ["GOOSE", "COW", "SHEEP"]
CROP_NAMES = PRODUCTS[:5]
ITEMS = PRODUCTS + ANIMAL_NAMES

NUM_PRODUCTS = len(PRODUCTS)
NUM_CROPS = len(CROP_NAMES)
NUM_ANIMALS = len(ANIMAL_NAMES)
NUM_ITEMS = len(ITEMS)

WHEAT, FERTILIZER = 0, 8
ANIMAL_ITEM_BASE = NUM_PRODUCTS

CROP_SEED = np.array([10, 20, 50, 100, 80], np.int32)
CROP_FIRST_YIELD_DAY = np.array([2, 2, 8, 10, 10], np.int32)
CROP_MAX_YIELD_DAY = np.array([4, 3, 8, 10, 12], np.int32)
CROP_INTERVAL = np.array([0, 0, 1, 2, 0], np.int32)
CROP_MAX_YIELD = np.array([6, 4, 4, 4, 6], np.int32)
CROP_ONGOING = np.array([False, False, True, True, False])

ANIMAL_COST = np.array([300, 400, 500], np.int32)
ANIMAL_FIRST_YIELD_DAY = np.array([4, 8, 6], np.int32)
ANIMAL_INTERVAL = np.array([1, 2, 3], np.int32)
ANIMAL_MAX_HELD = np.array([4, 6, 6], np.int32)
ANIMAL_PRODUCT = np.array([5, 6, 7], np.int32)

EMPTY, LOCKED, WEED, PLANT, COOP, PASTURE = 0, 1, 2, 3, 4, 5
ANIMAL_STRUCTURE = np.array([COOP, PASTURE, PASTURE], np.int32)

LAND_PRICES = np.array([1000, 2000, 4000], np.int32)

SHOP_NAMES = [
    "BAKERY",
    "PIZZA_SHOP",
    "BRUNCH_SPOT",
    "YARN_STORE",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "SMOOTHIE_SHOP",
    "FARMERS_MARKET",
]
NUM_SHOPS = len(SHOP_NAMES)

MARKET_I0 = 10000

OP_PASS, OP_NORTH, OP_SOUTH, OP_EAST, OP_WEST = 0, 1, 2, 3, 4
OP_PICKUP, OP_DROP, OP_PLANT, OP_WATER, OP_HARVEST = 5, 6, 7, 8, 9
OP_FERTILIZE, OP_DIG, OP_BUILD_COOP, OP_BUILD_PASTURE = 10, 11, 12, 13
OP_PLACE, OP_FEED, OP_COLLECT_FERTILIZER, OP_CARE = 14, 15, 16, 17
NUM_UNIT_OPS = 18

UNIT_OP_NAMES = [
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PICKUP",
    "DROP",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "DIG",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "PLACE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
]

MK_NONE, MK_BUY_SEED, MK_BUY_PRODUCT = 0, 1, 2
MK_BUY_ANIMAL, MK_SELL, MK_HIRE, MK_BUY_LAND = 3, 4, 5, 6
NUM_MARKET_OPS = 7
MARKET_OP_NAMES = [
    "NONE",
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
    "HIRE",
    "BUY_LAND",
]

FIB = np.array(
    reduce(lambda run, _: [*run, run[-1] + run[-2]], range(62), [1, 1]), np.int64
)

TILE_FIELDS = (
    "kind",
    "crop",
    "animal",
    "day",
    "watered",
    "unwatered",
    "units",
    "lifespan",
    "fert_until",
    "fed",
    "unfed",
    "cared",
    "fert_ready",
    "care_bonus",
)
OBS_TILE_FIELDS = tuple(field for field in TILE_FIELDS if field != "lifespan")
NUM_OBS_TILE_FIELDS = len(OBS_TILE_FIELDS)

DIRECTORY = Path(__file__).parent / "ffi"
LIBRARY = "libkaggriculture.so"
SOURCES = ("kaggriculture.c", "kaggriculture.h", "ffi.cc", "build.sh")

BOARD = 10
PLAYERS = 2
OBS_UNITS = 17
MAX_ORDERS = 10
NUM_HEADS = 3 * OBS_UNITS + 2 * MAX_ORDERS
NUM_CATEGORIES = 25
EPISODE_STEPS = 720
PLAYER_KEYS = ("tiles", "lifespan", "pos", "inv", "shed", "seeds", "money", "hands", "hires", "quadrants")


def shed_access_tiles(board_size):
    half = board_size // 2
    return np.array(
        [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)],
        np.int32,
    )


def default_spawn(board_size):
    return shed_access_tiles(board_size)[0]


def is_shed_adjacent(pos, board_size):
    tiles = shed_access_tiles(board_size)
    return jnp.any(jnp.all(pos[None] == jnp.asarray(tiles), axis=-1))


def seat_view(obs, player):
    return {
        name: jnp.roll(leaf, -player, axis=1) if name in PLAYER_KEYS else leaf
        for name, leaf in obs.items()
    }


library = None


def load():
    global library
    if library is None:
        build(DIRECTORY, LIBRARY, SOURCES)
        held = load_library(DIRECTORY, LIBRARY)
        register_targets(
            held,
            {
                "kaggriculture_init": "kaggriculture_ffi_init",
                "kaggriculture_step": "kaggriculture_ffi_step",
                "kaggriculture_close": "kaggriculture_ffi_close",
            },
        )
        library = held
    return library


def quantities():
    return tuple((ctypes.c_int32 * NUM_CATEGORIES).in_dll(load(), "KG_QUANTITIES"))


class Kaggriculture(Environment):
    num_agents = PLAYERS

    def __init__(self, num_envs=4096, threads=16, seed=0, episode_steps=EPISODE_STEPS):
        load()
        self.num_envs = num_envs
        self.threads = threads
        self.seed = seed
        self.episode_steps = int(episode_steps)

        shapes = (
            jax.ShapeDtypeStruct((1,), jnp.int32),
            *self.observation_shapes().values(),
            jax.ShapeDtypeStruct((num_envs, PLAYERS), jnp.float32),
            jax.ShapeDtypeStruct((num_envs, PLAYERS), jnp.int8),
        )
        self.spawn = partial(
            jax.ffi.ffi_call(
                "kaggriculture_init", shapes, has_side_effect=True, vmap_method="sequential"
            ),
            num_envs=num_envs,
            threads=threads,
            seed=seed,
            episode_steps=self.episode_steps,
        )
        self.advance = jax.ffi.ffi_call(
            "kaggriculture_step", shapes, has_side_effect=True, vmap_method="sequential"
        )
        self.release = jax.ffi.ffi_call(
            "kaggriculture_close",
            jax.ShapeDtypeStruct((1,), jnp.int32),
            has_side_effect=True,
            vmap_method="sequential",
        )

    def observation_shapes(self):
        n, p, u = self.num_envs, PLAYERS, OBS_UNITS
        return {
            "tiles": jax.ShapeDtypeStruct((n, p, BOARD, BOARD, NUM_OBS_TILE_FIELDS), jnp.int8),
            "lifespan": jax.ShapeDtypeStruct((n, p, BOARD, BOARD), jnp.int8),
            "pos": jax.ShapeDtypeStruct((n, p, u, 2), jnp.int8),
            "inv": jax.ShapeDtypeStruct((n, p, u, NUM_ITEMS), jnp.int16),
            "shed": jax.ShapeDtypeStruct((n, p, NUM_ITEMS), jnp.int16),
            "seeds": jax.ShapeDtypeStruct((n, p, NUM_CROPS), jnp.int16),
            "money": jax.ShapeDtypeStruct((n, p), jnp.float32),
            "hands": jax.ShapeDtypeStruct((n, p), jnp.int16),
            "hires": jax.ShapeDtypeStruct((n, p), jnp.int16),
            "quadrants": jax.ShapeDtypeStruct((n, p), jnp.int16),
            "market": jax.ShapeDtypeStruct((n, NUM_PRODUCTS), jnp.int32),
            "prices": jax.ShapeDtypeStruct((n, NUM_PRODUCTS), jnp.int16),
            "shops": jax.ShapeDtypeStruct((n, NUM_SHOPS), jnp.int8),
            "day": jax.ShapeDtypeStruct((n,), jnp.int8),
            "hour": jax.ShapeDtypeStruct((n,), jnp.int8),
        }

    def timestep(self, outputs, action):
        handle, *rest = outputs
        names = list(self.observation_shapes())
        obs = dict(zip(names, rest[: len(names)]))
        reward, terminated = rest[len(names)], rest[len(names) + 1]
        terminated = terminated.astype(jnp.bool_)
        return handle, Timestep(
            obs=obs,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=jnp.zeros_like(terminated),
        )

    def init(self, key: Key):
        blank = jnp.zeros((self.num_envs, PLAYERS, NUM_HEADS), jnp.int32)
        return self.timestep(self.spawn(), blank)

    def step(self, key: Key, state, action):
        return self.timestep(self.advance(state, action.astype(jnp.int32)), action)

    def close(self, state):
        return self.release(state)

    def observation_space(self):
        big = jnp.iinfo(jnp.int32).max
        return {
            name: Space(shape=spec.shape[1:], dtype=spec.dtype, low=-big, high=big)
            for name, spec in self.observation_shapes().items()
        }

    def action_space(self):
        return Space(shape=(NUM_HEADS,), dtype=jnp.int32, low=0, high=NUM_CATEGORIES - 1)

    def time_limit(self):
        return self.episode_steps - 1


def make(env_id: str = "kaggriculture", num_envs: int = 32, **kwargs) -> Kaggriculture:
    return Kaggriculture(num_envs=num_envs, **kwargs)
