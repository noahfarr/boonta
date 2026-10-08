import ctypes
import json
import math
import shutil

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

if shutil.which("g++") is None:
    pytest.skip("needs a C++ compiler to build the FFI environment", allow_module_level=True)

from boonta.algorithms.recurrent_bc import RecurrentBC, RecurrentBCConfig
from boonta.datasets import kaggriculture as store
from boonta.environments import kaggriculture as K
from boonta.environments.kaggriculture import actions as A
from boonta.environments.kaggriculture import replays, seat_view
from boonta.networks import RNN, SSM, MinGRUCell, Network
from hangar.kaggriculture.distribution import (SEATS, JobDistribution,
                                               categorical_log_prob,
                                               fold_seats, unfold_seats)
from hangar.kaggriculture.network import (Farmer, Heads, Seated, Trunk,
                                          both_seats)

ENVS, FEATURES = 6, 32
MAX_UNITS = 64
NUM_UNITS = K.OBS_UNITS
MAX_ORDERS = K.MAX_ORDERS
CURVE_KIND = {"linear": 0, "sq": 1, "sqrt": 2, "log": 3, "log10": 4, "hinge": 5}
ITEM_INDEX = {name: index for index, name in enumerate(K.ITEMS)}
CROP_INDEX = {name: index for index, name in enumerate(K.CROP_NAMES)}
ANIMAL_INDEX = {name: index for index, name in enumerate(K.ANIMAL_NAMES)}
PRODUCTS_INDEX = {name: index for index, name in enumerate(K.PRODUCTS)}
KINDS = {"WEED": K.WEED, "PLANT": K.PLANT, "COOP": K.COOP, "PASTURE": K.PASTURE}
UNIT_OPS = {name: index for index, name in enumerate(K.UNIT_OP_NAMES)}
MARKET_OPS = {name: index for index, name in enumerate(K.MARKET_OP_NAMES)}


class CTile(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int8), ("crop", ctypes.c_int8), ("animal", ctypes.c_int8),
        ("watered_today", ctypes.c_int8), ("consecutive_unwatered", ctypes.c_int8),
        ("fed_today", ctypes.c_int8), ("consecutive_unfed", ctypes.c_int8),
        ("cared_today", ctypes.c_int8), ("fertilizer_available", ctypes.c_int8),
        ("pending_care_bonus", ctypes.c_int8), ("yield_units", ctypes.c_int8),
        ("planted_day", ctypes.c_int16), ("fertilized_until_day", ctypes.c_int16),
        ("max_lifespan_step", ctypes.c_int16),
    ]


class CFarm(ctypes.Structure):
    _fields_ = [
        ("money", ctypes.c_float),
        ("tiles", CTile * (K.BOARD * K.BOARD)),
        ("unit_x", ctypes.c_int8 * MAX_UNITS),
        ("unit_y", ctypes.c_int8 * MAX_UNITS),
        ("num_units", ctypes.c_int32),
        ("hires_today", ctypes.c_int32),
        ("quadrants", ctypes.c_int32),
        ("next_decay", ctypes.c_int32),
        ("shed", ctypes.c_int32 * K.NUM_ITEMS),
        ("seeds", ctypes.c_int32 * K.NUM_CROPS),
        ("inv", (ctypes.c_int16 * K.NUM_ITEMS) * MAX_UNITS),
    ]


class CState(ctypes.Structure):
    _fields_ = [
        ("step", ctypes.c_int32),
        ("farms", CFarm * K.PLAYERS),
        ("market", ctypes.c_int32 * K.NUM_PRODUCTS),
        ("prices", ctypes.c_int32 * K.NUM_PRODUCTS),
        ("shops", ctypes.c_int8 * K.NUM_SHOPS),
        ("shop_order", ctypes.c_int8 * K.NUM_SHOPS),
        ("shops_unlocked", ctypes.c_int32),
        ("price_dirty", ctypes.c_uint32),
        ("rng", ctypes.c_uint64),
        ("weed_chance", ctypes.c_float),
        ("done", ctypes.c_int32),
        ("episode_steps", ctypes.c_int32),
        ("reward", ctypes.c_float * K.PLAYERS),
    ]


class CUnitAction(ctypes.Structure):
    _fields_ = [("op", ctypes.c_int8), ("arg", ctypes.c_int8), ("n", ctypes.c_int16)]


class CMarketAction(ctypes.Structure):
    _fields_ = [("op", ctypes.c_int8), ("item", ctypes.c_int8), ("n", ctypes.c_int16)]


class CAction(ctypes.Structure):
    _fields_ = [("units", CUnitAction * MAX_UNITS), ("orders", CMarketAction * MAX_ORDERS)]


class CCurve(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_float), ("i0", ctypes.c_float),
        ("below_kind", ctypes.c_int32), ("above_kind", ctypes.c_int32),
        ("below_amp", ctypes.c_float), ("above_amp", ctypes.c_float),
        ("t", ctypes.c_float),
    ]


class CPrice(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_double),
        ("below_amp", ctypes.c_double), ("above_amp", ctypes.c_double),
        ("below_kind", ctypes.c_int32), ("above_kind", ctypes.c_int32),
        ("t", ctypes.c_double),
    ]


POINTER = {"int8": ctypes.c_int8, "int16": ctypes.c_int16, "int32": ctypes.c_int32, "float32": ctypes.c_float}


class CObs(ctypes.Structure):
    _fields_ = [(name, ctypes.POINTER(POINTER[dtype])) for name, _, dtype in replays.OBS_SPEC]


def bind():
    library = K.load()
    library.kg_init_tables.restype = None
    library.kg_reset.argtypes = [ctypes.POINTER(CState), ctypes.c_uint64]
    library.kg_reset.restype = None
    library.kg_step.argtypes = [ctypes.POINTER(CState), ctypes.POINTER(CAction)]
    library.kg_step.restype = None
    library.kg_market_price.argtypes = [ctypes.c_int, ctypes.c_int32]
    library.kg_market_price.restype = ctypes.c_float
    library.kg_batch_encode.argtypes = [ctypes.POINTER(CState), ctypes.c_int, CObs, ctypes.c_int]
    library.kg_batch_encode.restype = None
    library.kg_legal_jobs.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int8)]
    library.kg_init_tables()
    return library


LIB = bind()
CURVES = (CCurve * K.NUM_PRODUCTS).in_dll(LIB, "KG_CURVES")
PRICES = (CPrice * K.NUM_PRODUCTS).in_dll(LIB, "KG_PRICES")


def shape_of(kind, x, t=None):
    x = max(0.0, x)
    if kind == "sq":
        return x * x
    if kind == "sqrt":
        return math.sqrt(x)
    if kind == "log":
        return math.log(1.0 + x)
    if kind == "log10":
        return math.log10(1.0 + x)
    if kind == "hinge":
        if not t or t <= 0:
            return x
        u = x / t
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


PRICE_FIELDS = [name for name, _ in CPrice._fields_]
LIB.kg_market_price(0, K.MARKET_I0)
DEFAULT_PRICES = [{name: getattr(PRICES[index], name) for name in PRICE_FIELDS} for index in range(K.NUM_PRODUCTS)]


def set_curves(resolved=None):
    if resolved is None:
        LIB.kg_init_tables()
        for index, defaults in enumerate(DEFAULT_PRICES):
            for name, value in defaults.items():
                setattr(PRICES[index], name, value)
        return
    for index, item in enumerate(K.PRODUCTS):
        params = resolved[item]
        t = params["T"]
        below = params["below_target"] * params["base"] / shape_of(params["below_func"], t, t)
        above = params["above_target"] * params["base"] / shape_of(params["above_func"], t, t)
        for table in (PRICES[index], CURVES[index]):
            table.base = float(params["base"])
            table.below_kind = CURVE_KIND[params["below_func"]]
            table.above_kind = CURVE_KIND[params["above_func"]]
            table.below_amp = float(below)
            table.above_amp = float(above)
            table.t = float(t)
        CURVES[index].i0 = float(params["I0"])


def market_price(item, inventory):
    return int(LIB.kg_market_price(int(item), int(inventory)))


class Port:
    def __init__(self, seed=1, episode_steps=720, shop_order=None, starting_money=None,
                 market_params=None, weed_chance=0.0):
        set_curves(market_params)
        self.state = CState()
        LIB.kg_reset(ctypes.byref(self.state), ctypes.c_uint64(seed or 1))
        self.state.weed_chance = float(weed_chance)
        self.state.episode_steps = int(episode_steps)
        if shop_order is not None:
            for index, shop in enumerate(shop_order):
                self.state.shop_order[index] = int(shop)
        if starting_money is not None:
            for player in range(K.PLAYERS):
                self.state.farms[player].money = float(starting_money)
        for index in range(K.NUM_PRODUCTS):
            self.state.prices[index] = market_price(index, self.state.market[index])
        self.state.price_dirty = 0

    def step(self, buffer):
        LIB.kg_step(ctypes.byref(self.state), buffer)


def pack(encoded):
    buffer = (CAction * K.PLAYERS)()
    for player, action in enumerate(encoded):
        for unit in range(NUM_UNITS):
            buffer[player].units[unit].op = int(action["unit_op"][unit])
            buffer[player].units[unit].arg = int(action["unit_arg"][unit])
            buffer[player].units[unit].n = int(action["unit_n"][unit])
        for slot in range(MAX_ORDERS):
            buffer[player].orders[slot].op = int(action["market_op"][slot])
            buffer[player].orders[slot].item = int(action["market_item"][slot])
            buffer[player].orders[slot].n = int(action["market_n"][slot])
    return buffer


def observation_arrays(state):
    buffers = {name: np.zeros((1, *shape), dtype) for name, shape, dtype in replays.OBS_SPEC}
    bundle = CObs(**{
        name: buffers[name].ctypes.data_as(ctypes.POINTER(POINTER[dtype]))
        for name, _, dtype in replays.OBS_SPEC
    })
    LIB.kg_batch_encode(ctypes.byref(state), 1, bundle, 1)
    return buffers


def encode(action, into):
    for unit in range(len(into.units)):
        into.units[unit].op, into.units[unit].arg, into.units[unit].n = K.OP_PASS, 0, 1
    for slot in range(MAX_ORDERS):
        into.orders[slot].op, into.orders[slot].item, into.orders[slot].n = 0, 0, 0

    commands = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
    for index, command in enumerate(commands):
        if index >= len(into.units) or not command:
            break
        name = command[0]
        into.units[index].op = UNIT_OPS.get(name, K.OP_PASS)
        if name in ("PLANT", "PICKUP", "PLACE"):
            names = K.CROP_NAMES if name == "PLANT" else K.ITEMS
            arg = replays.slot_of(names, command[1] if len(command) > 1 else None)
            if arg is None:
                into.units[index].op = K.OP_PASS
                continue
            into.units[index].arg = arg
            if name != "PLANT":
                into.units[index].n = replays.count_of(command, 2)

    for slot, order in enumerate(action.get("market") or []):
        if slot >= MAX_ORDERS or not order:
            break
        name = order[0]
        if name not in MARKET_OPS:
            continue
        if name in ("HIRE", "BUY_LAND"):
            into.orders[slot].op, into.orders[slot].n = MARKET_OPS[name], 1
            continue
        item = order[1] if len(order) > 1 else None
        code = replays.slot_of(K.CROP_NAMES if name == "BUY_SEED" else K.ITEMS, item)
        if code is None:
            continue
        into.orders[slot].op = MARKET_OPS[name]
        into.orders[slot].item = code
        into.orders[slot].n = replays.count_of(order, 2)
    return into


def test_the_struct_mirror_matches_the_header():
    sizes = tuple(ctypes.sizeof(kind) for kind in (CTile, CFarm, CState, CAction))
    assert sizes == (18, 3552, 7240, 296)
    assert [(name, getattr(CState, name).offset) for name, _ in CState._fields_] == [
        ("step", 0), ("farms", 4), ("market", 7108), ("prices", 7144),
        ("shops", 7180), ("shop_order", 7188), ("shops_unlocked", 7196),
        ("price_dirty", 7200), ("rng", 7208), ("weed_chance", 7216),
        ("done", 7220), ("episode_steps", 7224), ("reward", 7228),
    ]
    assert [(name, getattr(CFarm, name).offset) for name, _ in CFarm._fields_] == [
        ("money", 0), ("tiles", 4), ("unit_x", 1804), ("unit_y", 1868),
        ("num_units", 1932), ("hires_today", 1936), ("quadrants", 1940),
        ("next_decay", 1944), ("shed", 1948), ("seeds", 1996), ("inv", 2016),
    ]


def test_the_quantity_buckets_cover_every_category():
    quantities = K.quantities()
    assert len(quantities) == A.NUM_QUANTITIES
    assert list(quantities) == sorted(quantities) and quantities[0] == 0


def test_an_episode_keeps_its_shapes_under_jit_and_rolls_over_each_day():
    environment = K.Kaggriculture(num_envs=8, episode_steps=48, threads=1)
    state, timestep = environment.init(jax.random.key(0))

    assert timestep.reward.shape == (8, K.PLAYERS)
    for name, spec in environment.observation_shapes().items():
        assert timestep.obs[name].shape == spec.shape, name
        assert timestep.obs[name].dtype == spec.dtype, name

    action = jnp.zeros((8, K.PLAYERS, K.NUM_HEADS), jnp.int32)
    step = jax.jit(environment.step)
    state, stepped = step(jax.random.key(0), state, action)
    for name, spec in environment.observation_shapes().items():
        assert stepped.obs[name].shape == spec.shape, name
    assert not bool(stepped.terminated.any())

    for _ in range(23):
        state, stepped = step(jax.random.key(2), state, action)
    assert int(stepped.obs["day"][0]) == 1 and int(stepped.obs["hour"][0]) == 0
    spawn = jnp.asarray(K.default_spawn(K.BOARD), stepped.obs["pos"].dtype)
    assert bool((stepped.obs["pos"] == spawn).all())
    environment.close(state)


def test_an_episode_ends_once_and_pays_out_its_money():
    environment = K.Kaggriculture(num_envs=2, episode_steps=24, threads=1)
    state, timestep = environment.init(jax.random.key(0))
    action = jnp.zeros((2, K.PLAYERS, K.NUM_HEADS), jnp.int32)
    ends = []
    for _ in range(environment.time_limit()):
        state, timestep = environment.step(jax.random.key(0), state, action)
        ends.append(bool(timestep.terminated.all()))
    assert ends[-1] and not any(ends[:-1])
    np.testing.assert_array_equal(timestep.reward, timestep.obs["money"])
    environment.close(state)


def test_weeds_spawn_once_per_empty_tile_at_the_configured_rate():
    trials, chance = 4000, 0.3
    unlocked = (K.BOARD // 2) ** 2
    weeds = np.zeros(trials * K.PLAYERS)
    idle = pack([blank() for _ in range(K.PLAYERS)])
    for trial in range(trials):
        port = Port(shop_order=list(range(K.NUM_SHOPS)), seed=trial + 1)
        port.state.weed_chance = chance
        for _ in range(24):
            port.step(idle)
        for player in range(K.PLAYERS):
            weeds[trial * K.PLAYERS + player] = sum(
                tile.kind == K.WEED for tile in port.state.farms[player].tiles
            )
    mean, spread = unlocked * chance, math.sqrt(unlocked * chance * (1 - chance))
    assert abs(weeds.mean() - mean) < 4 * spread / math.sqrt(weeds.size)
    assert weeds.max() <= unlocked


@pytest.mark.parametrize("n", [1, 2, 3, 8, A.UNITS])
def test_matching_is_optimal(n):
    optimize = pytest.importorskip("scipy.optimize")
    generator = np.random.default_rng(n)
    solve = jax.jit(A.matching)
    for _ in range(40):
        cost = generator.integers(0, 19, size=(n, n)).astype(np.float32)
        got = np.asarray(solve(jnp.asarray(cost)))
        assert sorted(got.tolist()) == list(range(n))
        rows, cols = optimize.linear_sum_assignment(cost)
        assert cost[np.arange(n), got].sum() == pytest.approx(cost[rows, cols].sum())


@pytest.mark.parametrize(
    "build",
    [
        lambda n: np.zeros((n, n)),
        lambda n: np.ones((n, n)),
        lambda n: np.tile(np.arange(n, dtype=float), (n, 1)),
    ],
    ids=["all-zero", "all-equal", "ties-across-rows"],
)
def test_degenerate_costs_still_give_a_permutation(build):
    got = np.asarray(jax.jit(A.matching)(jnp.asarray(build(A.UNITS), jnp.float32)))
    assert sorted(got.tolist()) == list(range(A.UNITS))


def test_blocked_rows_never_take_a_real_job():
    n = A.UNITS
    cost = np.full((n, n), A.FORBIDDEN, np.float32)
    cost[:4, :4] = np.arange(16).reshape(4, 4)
    cost[:, 4:] = 0.0
    got = np.asarray(jax.jit(A.matching)(jnp.asarray(cost)))
    assert sorted(got.tolist()) == list(range(n))
    assert set(got[:4].tolist()) == {0, 1, 2, 3}


def test_place_assigns_every_job_to_a_distinct_live_unit():
    live = 5
    chosen = np.zeros((1, A.UNITS), np.int32)
    for slot in range(live):
        chosen[0, slot] = (slot * 11) * A.NUM_JOBS + A.JOB_WATER
    pos = np.zeros((1, A.UNITS, 2), np.int32)
    pos[0, :live, 0] = np.arange(live)
    tile_of, job_of = A.place(
        jnp.asarray(chosen), jnp.asarray(pos), jnp.asarray([live], jnp.int32),
        jnp.asarray([1.0], jnp.float32),
    )
    tile_of = np.asarray(tile_of)[0]
    assigned = [tile for tile in tile_of[:live] if tile >= 0]
    assert len(assigned) == live and len(set(assigned)) == live
    assert (tile_of[live:] == -1).all()
    assert (np.asarray(job_of)[0][:live][tile_of[:live] >= 0] == A.JOB_WATER).all()


def test_place_beats_greedy_where_the_nearest_unit_strands_the_other_job():
    chosen = np.zeros((1, A.UNITS), np.int32)
    chosen[0, 0] = 1 * A.NUM_JOBS + A.JOB_WATER
    chosen[0, 1] = 4 * A.NUM_JOBS + A.JOB_WATER
    pos = np.zeros((1, A.UNITS, 2), np.int32)
    pos[0, 1] = (2, 0)
    tile_of, _ = A.place(
        jnp.asarray(chosen), jnp.asarray(pos), jnp.asarray([2], jnp.int32),
        jnp.asarray([1.0], jnp.float32),
    )
    tile_of = np.asarray(tile_of)[0]
    travel = sum(
        abs(int(pos[0, unit, 0]) - int(tile_of[unit]) % A.BOARD)
        + abs(int(pos[0, unit, 1]) - int(tile_of[unit]) // A.BOARD)
        for unit in range(2)
    )
    assert travel == 3


TILE = 44
FIELDS = ("kind", "crop", "animal", "planted_day", "watered_today", "yield_units", "fertilized_until_day")


def probe(setup, job):
    port = Port()
    setup(port.state)
    port.state.farms[0].unit_x[0] = int(A.TILE_X[TILE])
    port.state.farms[0].unit_y[0] = int(A.TILE_Y[TILE])
    obs = {name: jnp.asarray(leaf) for name, leaf in observation_arrays(port.state).items()}
    legal = bool(np.asarray(A.legal_jobs(obs, 0))[0, TILE, job])
    before = tuple(getattr(port.state.farms[0].tiles[TILE], field) for field in FIELDS)
    buffer = (CAction * K.PLAYERS)()
    buffer[0].units[0].op = int(A.JOB_OP[job])
    buffer[0].units[0].arg = int(A.JOB_ARG[job])
    buffer[0].units[0].n = 5
    port.step(buffer)
    after = tuple(getattr(port.state.farms[0].tiles[TILE], field) for field in FIELDS)
    return legal, after != before


def fresh(state):
    farm = state.farms[0]
    farm.seeds[0] = 5
    tile = farm.tiles[TILE]
    tile.kind, tile.crop = K.PLANT, 0
    tile.planted_day, tile.yield_units = 0, 1


def ripe(state):
    fresh(state)
    state.step = 5 * 24


def fertilised(state):
    fresh(state)
    state.farms[0].tiles[TILE].fertilized_until_day = 30
    state.farms[0].inv[0][A.FERT_ITEM] = 3


def carrying(state):
    fresh(state)
    state.farms[0].inv[0][A.FERT_ITEM] = 3


@pytest.mark.parametrize(
    "setup, job, want",
    [
        (fresh, A.JOB_HARVEST, False),
        (ripe, A.JOB_HARVEST, True),
        (fertilised, A.JOB_FERTILIZE, False),
        (carrying, A.JOB_FERTILIZE, True),
        (fresh, A.JOB_FERTILIZE, False),
        (fresh, A.JOB_DIG, True),
    ],
    ids=["unripe-harvest", "ripe-harvest", "fertilised-again", "fertilise", "fertilise-empty-handed", "dig"],
)
def test_the_job_mask_allows_what_the_engine_carries_out(setup, job, want):
    legal, moved = probe(setup, job)
    assert legal == want
    if want:
        assert moved


def test_the_job_mask_matches_the_engines_own():
    assert list((ctypes.c_int8 * A.NUM_JOBS).in_dll(LIB, "KG_JOB_OP")) == list(A.JOB_OP)
    assert list((ctypes.c_int8 * A.NUM_JOBS).in_dll(LIB, "KG_JOB_ARG")) == list(A.JOB_ARG)

    def engine(state, player):
        out = np.zeros(A.TILES * A.NUM_JOBS, np.int8)
        LIB.kg_legal_jobs(ctypes.byref(state), player, out.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)))
        return out.reshape(A.TILES, A.NUM_JOBS).astype(bool)

    generator = np.random.default_rng(0)
    checked = 0
    for seed in range(3):
        port = Port(seed=seed + 1, weed_chance=0.01)
        buffer = (CAction * K.PLAYERS)()
        for step in range(120):
            for player in range(K.PLAYERS):
                for unit in range(A.UNITS):
                    buffer[player].units[unit].op = int(generator.integers(0, 18))
                    buffer[player].units[unit].arg = int(generator.integers(0, 12))
                    buffer[player].units[unit].n = int(generator.integers(0, 6))
                for slot in range(A.ORDERS):
                    buffer[player].orders[slot].op = int(generator.integers(0, 7))
                    buffer[player].orders[slot].item = int(generator.integers(0, 12))
                    buffer[player].orders[slot].n = int(generator.integers(0, 4))
            port.step(buffer)
            if step % 12:
                continue
            obs = {name: jnp.asarray(leaf) for name, leaf in observation_arrays(port.state).items()}
            for player in range(K.PLAYERS):
                np.testing.assert_array_equal(np.asarray(A.legal_jobs(obs, player))[0], engine(port.state, player))
                checked += 1
    assert checked >= 40


def collide(jobs):
    tiles = np.zeros((1, A.BOARD, A.BOARD, len(A.F)), np.int8)
    tiles[..., A.F["crop"]] = -1
    tiles[0, 4, 4, A.F["kind"]] = K.PLANT
    tiles[0, 4, 4, A.F["crop"]] = 0
    tiles[0, 4, 4, A.F["units"]] = 2
    tile_of = jnp.full((1, A.UNITS), -1, jnp.int32)
    job_of = jnp.zeros((1, A.UNITS), jnp.int32)
    arrived = jnp.zeros((1, A.UNITS), bool)
    for slot, job in enumerate(jobs):
        tile_of = tile_of.at[0, slot].set(TILE)
        job_of = job_of.at[0, slot].set(job)
        arrived = arrived.at[0, slot].set(True)
    out = A.settle(jnp.asarray(tiles), tile_of, job_of, arrived, 1)
    return [int(out[0, slot]) for slot in range(len(jobs))]


def test_settle_drops_only_the_destructive_partner():
    assert collide([A.JOB_PLANT[0], A.JOB_DIG]) == [A.JOB_PLANT[0], A.JOB_PASS]
    assert collide([A.JOB_DIG, A.JOB_HARVEST]) == [A.JOB_PASS, A.JOB_HARVEST]
    assert collide([A.JOB_FERTILIZE, A.JOB_HARVEST]) == [A.JOB_PASS, A.JOB_HARVEST]
    assert collide([A.JOB_HARVEST, A.JOB_DIG]) == [A.JOB_HARVEST, A.JOB_DIG]
    assert collide([A.JOB_WATER, A.JOB_HARVEST]) == [A.JOB_WATER, A.JOB_HARVEST]
    assert collide([A.JOB_DIG]) == [A.JOB_DIG]


def test_a_dig_behind_a_plant_would_have_wasted_the_seed():
    def play(jobs):
        port = Port()
        farm = port.state.farms[0]
        farm.num_units, farm.seeds[0] = 2, 3
        for unit in (0, 1):
            farm.unit_x[unit], farm.unit_y[unit] = 4, 4
        buffer = (CAction * K.PLAYERS)()
        for unit, job in enumerate(jobs):
            buffer[0].units[unit].op = int(A.JOB_OP[job])
            buffer[0].units[unit].arg = int(A.JOB_ARG[job])
            buffer[0].units[unit].n = 1
        port.step(buffer)
        return port.state.farms[0].tiles[TILE].kind

    assert play([A.JOB_PLANT[0], A.JOB_DIG]) == K.EMPTY
    assert play([A.JOB_PLANT[0], A.JOB_PASS]) == K.PLANT


@pytest.fixture(scope="module")
def observation():
    environment = K.Kaggriculture(num_envs=ENVS, threads=1)
    state, timestep = environment.init(jax.random.key(0))
    obs = jax.tree.map(np.asarray, timestep.obs)
    environment.close(state)
    return jax.tree.map(jnp.asarray, obs)


def test_folding_and_unfolding_the_seats_is_the_identity():
    x = jnp.arange(ENVS * K.PLAYERS * 3).reshape(ENVS, K.PLAYERS, 3)
    assert jnp.array_equal(unfold_seats(fold_seats(x)), x)


def test_each_folded_row_holds_its_own_seats_view(observation):
    folded = both_seats(observation)
    for player in range(K.PLAYERS):
        view = seat_view(observation, player)
        rows = np.arange(ENVS) * K.PLAYERS + player
        for name in ("tiles", "money", "shed", "seeds", "pos", "hands"):
            assert jnp.array_equal(folded[name][rows], view[name]), name
    money = np.asarray(observation["money"])
    for env in range(ENVS):
        for player in range(K.PLAYERS):
            assert float(folded["money"][env * K.PLAYERS + player, 0]) == money[env, player]


def test_each_seat_acts_and_is_valued_from_its_own_view(observation):
    network = Farmer(features=16, blocks=1)
    params = network.init(jax.random.key(0), observation, temperature=1.0)
    dist, value = network.apply(params, observation, temperature=1.0)
    assert value.shape == (ENVS, K.PLAYERS, 1)

    action, log_prob = dist.sample_and_log_prob(jax.random.key(1))
    assert A.route(observation, action).shape == (ENVS, K.PLAYERS, K.NUM_HEADS)
    assert log_prob.shape == (ENVS, K.PLAYERS)
    np.testing.assert_allclose(np.asarray(dist.log_prob(action)), np.asarray(log_prob), rtol=1e-5, atol=1e-5)

    bumped = dict(observation)
    money = np.asarray(observation["money"]).copy()
    money[:, 1] += 5000.0
    bumped["money"] = jnp.asarray(money)
    _, moved = network.apply(params, bumped, temperature=1.0)
    assert not np.allclose(np.asarray(value)[:, 1], np.asarray(moved)[:, 1])


def test_a_cold_policy_acts_on_its_mode(observation):
    network = Farmer(features=16, blocks=1)
    params = network.init(jax.random.key(0), observation, temperature=1.0)
    dist, _ = network.apply(params, observation, temperature=0.0)
    first, _ = dist.sample_and_log_prob(jax.random.key(1))
    second, _ = dist.sample_and_log_prob(jax.random.key(2))
    jax.tree.map(np.testing.assert_array_equal, first, dist.mode())
    jax.tree.map(np.testing.assert_array_equal, first, second)


def recurrent(torso):
    return Network(
        feature_extractor=Trunk(features=16, blocks=1, dtype=jnp.float32),
        torso=Seated(torso=torso),
        head=Heads(dtype=jnp.float32),
    )


@pytest.fixture(scope="module")
def memory():
    network = recurrent(RNN(cell=nn.GRUCell(features=FEATURES)))
    environment = K.Kaggriculture(num_envs=4, episode_steps=48, threads=1)
    state, timestep = environment.init(jax.random.key(0))
    environment.close(state)
    sequence = timestep.to_sequence()
    carry = network.initialize_carry(jax.random.key(0), (*timestep.reward.shape, 1))
    done = jnp.zeros((4, 1, SEATS), bool)
    params = network.init(jax.random.key(1), sequence.obs, done=done, carry=carry, temperature=1.0)
    return network, params, sequence, carry, done


def advance(memory, carry, done=None):
    network, params, sequence, _, live = memory
    return network.apply(
        params, sequence.obs, done=live if done is None else done, carry=carry, temperature=1.0
    )


def test_the_carry_is_env_major_with_a_seat_axis(memory):
    _, _, _, carry, _ = memory
    assert jax.tree.all(jax.tree.map(lambda leaf: leaf.shape[:2] == (4, SEATS), carry))


def test_actions_come_back_per_environment_step_and_seat(memory):
    _, _, _, carry, _ = memory
    updated, dist = advance(memory, carry)
    assert jax.tree.all(jax.tree.map(lambda a, b: a.shape == b.shape, carry, updated))
    action, log_prob = dist.sample_and_log_prob(jax.random.key(2))
    assert action["chosen"].shape[:3] == (4, 1, SEATS)
    assert log_prob.shape == (4, 1, SEATS)


def test_a_seat_does_not_read_the_other_seats_memory(memory):
    _, _, _, carry, _ = memory
    disturbed = jax.tree.map(lambda leaf: leaf.at[:, 1].add(1.0), carry)
    (before, _), (after, _) = advance(memory, carry), advance(memory, disturbed)
    assert jax.tree.all(jax.tree.map(lambda a, b: jnp.allclose(a[:, 0], b[:, 0]), before, after))
    assert jax.tree.all(jax.tree.map(lambda a, b: not jnp.allclose(a[:, 1], b[:, 1]), before, after))


def test_an_environment_does_not_read_anothers_memory(memory):
    _, _, _, carry, _ = memory
    disturbed = jax.tree.map(lambda leaf: leaf.at[0].add(1.0), carry)
    (before, _), (after, _) = advance(memory, carry), advance(memory, disturbed)
    assert jax.tree.all(jax.tree.map(lambda a, b: jnp.allclose(a[1:], b[1:]), before, after))


def test_an_episode_end_resets_the_carry(memory):
    _, _, _, carry, _ = memory
    ended = jnp.ones((4, 1, SEATS), bool)
    disturbed = jax.tree.map(lambda leaf: leaf + 1.0, carry)
    kept, _ = advance(memory, disturbed)
    reset, _ = advance(memory, disturbed, ended)
    zeroed, _ = advance(memory, carry, ended)
    assert jax.tree.all(jax.tree.map(lambda a, b: not jnp.allclose(a, b), kept, reset))
    assert jax.tree.all(jax.tree.map(lambda a, b: jnp.allclose(a, b), reset, zeroed))


def test_a_carry_request_without_a_seat_axis_still_gets_one():
    network = recurrent(SSM(cell=MinGRUCell(features=FEATURES, dtype=jnp.float32)))
    blind = network.initialize_carry(jax.random.key(0), (4, 1))
    seated = network.initialize_carry(jax.random.key(0), (4, SEATS, 1))
    assert jax.tree.all(
        jax.tree.map(lambda a, b: a.shape == b.shape == (4, SEATS, FEATURES), blind, seated)
    )


def test_scoring_a_sequence_matches_scoring_each_step():
    envs, steps = 3, 4
    rows = envs * SEATS
    key = jax.random.key(0)
    jobs = jax.random.normal(key, (rows, steps, A.TILES * A.NUM_JOBS))
    logits = {
        "jobs_masked": jobs.at[..., 400:].set(jnp.finfo(jobs.dtype).min),
        "market": jax.random.normal(key, (rows, steps, A.ORDERS, A.NUM_MARKET_ACTIONS)),
        "quantity": jax.random.normal(key, (rows, steps, A.NUM_MARKET_ACTIONS, A.NUM_QUANTITIES)),
        "travel": jnp.zeros((rows, steps)),
        "flow": jnp.zeros((rows, steps, K.NUM_PRODUCTS)),
        "live": jnp.full((rows, steps), 3, jnp.int32),
    }
    timed = JobDistribution(logits)
    action, _ = timed.sample_and_log_prob(jax.random.key(1))
    assert action["chosen"].shape == (envs, steps, SEATS, A.UNITS)
    scored = timed.log_prob(action)
    assert scored.shape == (envs, steps, SEATS)
    for step in range(steps):
        flat = JobDistribution(jax.tree.map(lambda leaf: leaf[:, step], logits))
        each = flat.log_prob(jax.tree.map(lambda leaf: leaf[:, step], action))
        assert jnp.allclose(each, scored[:, step], atol=1e-5), step


def synthetic(rows=96, episodes=2):
    environment = K.Kaggriculture(num_envs=1, episode_steps=24, threads=1)
    state, timestep = environment.init(jax.random.key(0))
    environment.close(state)
    obs = {name: np.repeat(np.asarray(leaf[:1]), rows, axis=0) for name, leaf in timestep.obs.items()}
    terminated = np.zeros(rows, bool)
    terminated[rows // episodes - 1::rows // episodes] = True
    action = {
        "chosen": np.zeros((rows, A.UNITS), np.int32),
        "market": np.zeros((rows, A.ORDERS), np.int32),
        "qty": np.zeros((rows, A.ORDERS), np.int32),
        "filled": np.ones(rows, np.int32),
        "allowed": np.ones((rows, A.ORDERS), bool),
        "value": np.arange(rows, dtype=np.float32),
    }
    return obs, action, terminated


def test_windows_never_cross_an_episode_end(tmp_path):
    parts = synthetic()
    store.save(tmp_path / "expert.npz", parts)
    dataset = store.make("expert", directory=tmp_path, horizon=32)
    _, _, terminated = parts
    assert len(dataset) == terminated.size
    assert dataset.starts.shape[0] == terminated.size - int(terminated.sum()) * 31

    batch = dataset.sample(dataset.init(), jax.random.key(0), (64,))
    assert batch.first.obs["tiles"].shape == (64, 32, K.PLAYERS, K.BOARD, K.BOARD, K.NUM_OBS_TILE_FIELDS)
    assert not bool(batch.first.done[:, :-1].any())
    assert sorted(batch.second.action) == ["allowed", "chosen", "filled", "market", "qty"]
    np.testing.assert_array_equal(batch.aux["weight"][..., 1], 0.0)


def test_a_horizon_longer_than_an_episode_is_refused(tmp_path):
    store.save(tmp_path / "expert.npz", synthetic())
    with pytest.raises(ValueError):
        store.make("expert", directory=tmp_path, horizon=10_000)


def test_pooling_concatenates_players(tmp_path):
    for name in ("alpha", "beta"):
        store.save(tmp_path / f"{name}.npz", synthetic())
    pooled = store.build(store.pool([tmp_path / "alpha.npz", tmp_path / "beta.npz"]), 16)
    single = store.build(store.load(tmp_path / "alpha.npz"), 16)
    assert len(pooled) == 2 * len(single)
    assert pooled.starts.shape[0] == 2 * single.starts.shape[0]


def blank(live=1):
    return {
        "unit_op": np.zeros(NUM_UNITS, np.int32),
        "unit_arg": np.zeros(NUM_UNITS, np.int32),
        "unit_n": np.ones(NUM_UNITS, np.int32),
        "market_op": np.zeros(MAX_ORDERS, np.int32),
        "market_item": np.zeros(MAX_ORDERS, np.int32),
        "market_n": np.zeros(MAX_ORDERS, np.int32),
        "live": live,
    }


def unit(action, index, op, arg=0, n=1):
    action["unit_op"][index] = op
    action["unit_arg"][index] = arg
    action["unit_n"][index] = n
    return action


def order(action, slot, op, item=0, n=0):
    action["market_op"][slot] = op
    action["market_item"][slot] = item
    action["market_n"][slot] = n
    return action


def shed_access(size):
    half = size // 2
    return half - 1, half - 1


def toward(x, y, tx, ty):
    if tx != x:
        return K.OP_EAST if tx > x else K.OP_WEST
    if ty != y:
        return K.OP_SOUTH if ty > y else K.OP_NORTH
    return K.OP_PASS


def find(tiles, predicate):
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if predicate(tile):
                return x, y
    return None


def reference():
    return pytest.importorskip("kaggle_environments.envs.kaggriculture.kaggriculture")


def interpreter(configuration=None):
    kaggle = pytest.importorskip("kaggle_environments")
    return kaggle.make("kaggriculture", configuration=dict(configuration or {}), debug=True)


def sample_action(rng, obs):
    ref = reference()
    farm = obs["farms"][obs["player"]]
    private = obs["private"]
    tiles = farm["tiles"]
    money = farm["money"]
    shed = private["shed"]
    seeds = private["seeds"]
    live = 1 + len(farm["hands"])
    size = len(tiles)
    day = obs.get("day", 0)
    action = blank(live)

    for index in range(live):
        x, y = (farm["farmer"] if index == 0 else farm["hands"][index - 1])[:2]
        held = private["inventories"][index] if index < len(private["inventories"]) else {}
        tile = tiles[y][x]
        op, arg, n = K.OP_PASS, 0, 1
        chaos = rng.random()
        animal = next((name for name in K.ANIMAL_NAMES if held.get(name, 0) > 0), None)
        housing = isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") and "animal" not in tile

        if chaos < 0.18:
            op = int(rng.integers(0, K.NUM_UNIT_OPS))
            arg = int(rng.integers(0, K.NUM_ITEMS))
            n = int(rng.integers(0, 6))
        elif animal is not None:
            want = ref.ANIMALS[animal]["structure"]
            if housing and tile["kind"] == want:
                op, arg = K.OP_PLACE, ITEM_INDEX[animal]
            else:
                spot = find(tiles, lambda t: isinstance(t, dict) and t.get("kind") == want and "animal" not in t)
                if spot is not None:
                    op = toward(x, y, *spot)
                elif tile is None:
                    op = K.OP_BUILD_COOP if want == "COOP" else K.OP_BUILD_PASTURE
                else:
                    op = int(rng.integers(K.OP_NORTH, K.OP_WEST + 1))
        elif isinstance(tile, dict) and "animal" in tile:
            choices = [K.OP_CARE, K.OP_COLLECT_FERTILIZER, K.OP_HARVEST, K.OP_PASS]
            weights = [0.2, 0.2, 0.3, 0.3]
            if held.get("WHEAT", 0) > 0 and rng.random() < 0.75:
                choices, weights = [K.OP_FEED], [1.0]
            op = int(rng.choice(choices, p=weights))
        elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop = ref.CROPS[tile["crop"]]
            grown = day - tile["planted_day"] >= crop["first_yield_day"]
            if grown and tile["yield_units"] > 0 and rng.random() < 0.5:
                op = K.OP_HARVEST
            elif not tile["watered_today"] and rng.random() < 0.8:
                op = K.OP_WATER
            elif held.get("FERTILIZER", 0) > 0 and rng.random() < 0.5:
                op = K.OP_FERTILIZE
        elif isinstance(tile, dict) and tile.get("kind") == "WEED":
            op = K.OP_DIG
        elif tile is None:
            plantable = [name for name in K.CROP_NAMES if seeds.get(name, 0) > 0]
            roll = rng.random()
            if plantable and roll < 0.6:
                op, arg = K.OP_PLANT, CROP_INDEX[str(rng.choice(plantable))]
            elif roll < 0.72:
                op = K.OP_BUILD_COOP
            elif roll < 0.84:
                op = K.OP_BUILD_PASTURE
            else:
                op = int(rng.integers(K.OP_NORTH, K.OP_WEST + 1))
        elif ref._is_shed_adjacent((x, y), size):
            stocked = [name for name in K.ITEMS if shed.get(name, 0) > 0]
            if stocked and rng.random() < 0.6:
                op, arg = K.OP_PICKUP, ITEM_INDEX[str(rng.choice(stocked))]
                n = int(rng.integers(1, 4))
            else:
                op = K.OP_DROP
        else:
            op = int(rng.integers(K.OP_NORTH, K.OP_WEST + 1))

        if op == K.OP_PASS and sum(held.values()) > 0 and rng.random() < 0.4:
            op = toward(x, y, *ref._shed_access_tiles(size)[0])
        unit(action, index, op, arg, n)

    slot = 0

    def push(kind, item=0, count=0):
        nonlocal slot
        if slot < MAX_ORDERS:
            order(action, slot, kind, item, count)
            slot += 1

    stock = sum(shed.get(name, 0) for name in K.ITEMS)
    if stock > 45 or rng.random() < 0.12:
        for name in K.PRODUCTS:
            if shed.get(name, 0) > 0 and rng.random() < 0.7:
                push(K.MK_SELL, PRODUCTS_INDEX[name], int(shed[name]))
    if sum(seeds.values()) < 6 and money > 150 and rng.random() < 0.5:
        crop = str(rng.choice(K.CROP_NAMES))
        push(K.MK_BUY_SEED, CROP_INDEX[crop], int(rng.integers(1, 3)))
    if stock < 60 and money > 150 and rng.random() < 0.25:
        push(K.MK_BUY_PRODUCT, int(rng.choice([K.WHEAT, K.FERTILIZER])), int(rng.integers(1, 4)))
    if money > 900 and rng.random() < 0.3:
        push(K.MK_BUY_ANIMAL, K.ANIMAL_ITEM_BASE + int(rng.integers(0, K.NUM_ANIMALS)), 1)
    if money > 400 and rng.random() < 0.25:
        for _ in range(int(rng.integers(1, 4))):
            push(K.MK_HIRE)
    if money > 3000 and rng.random() < 0.1:
        push(K.MK_BUY_LAND)
    if rng.random() < 0.15:
        push(int(rng.integers(1, K.NUM_MARKET_OPS)), int(rng.integers(0, K.NUM_ITEMS)), int(rng.integers(0, 4)))
    return action


def to_reference(action):
    def command(index):
        op = int(action["unit_op"][index])
        name = K.UNIT_OP_NAMES[op]
        arg = int(action["unit_arg"][index])
        if op == K.OP_PLANT:
            return [name, K.ITEMS[arg]]
        if op in (K.OP_PICKUP, K.OP_PLACE):
            return [name, K.ITEMS[arg], int(action["unit_n"][index])]
        return [name]

    market = []
    for slot in range(action["market_op"].size):
        kind = int(action["market_op"][slot])
        if kind == K.MK_NONE:
            market.append([])
        elif kind in (K.MK_HIRE, K.MK_BUY_LAND):
            market.append([K.MARKET_OP_NAMES[kind]])
        else:
            market.append([
                K.MARKET_OP_NAMES[kind],
                K.ITEMS[int(action["market_item"][slot])],
                int(action["market_n"][slot]),
            ])
    while market and not market[-1]:
        market.pop()
    return {
        "farmer": command(0),
        "hands": [command(index) for index in range(1, action["live"])],
        "market": market,
    }


def scenario_animals(rng, obs, player):
    farm = obs["farms"][player]
    private = obs["private"]
    tiles = farm["tiles"]
    shed, held = private["shed"], private["inventories"][0]
    x, y = farm["farmer"]
    tile = tiles[y][x]
    day = obs.get("day", 0)
    action = blank(1 + len(farm["hands"]))

    slot = 0
    if farm["money"] > 700 and sum(shed.get(name, 0) for name in K.ANIMAL_NAMES) == 0:
        order(action, slot, K.MK_BUY_ANIMAL, ITEM_INDEX[K.ANIMAL_NAMES[day % 3]], 1)
        slot += 1
    if shed.get("WHEAT", 0) < 4 and farm["money"] > 200:
        order(action, slot, K.MK_BUY_PRODUCT, K.WHEAT, 4)
        slot += 1

    animal = next((name for name in K.ANIMAL_NAMES if held.get(name, 0) > 0), None)
    adjacent = (x, y) in {(4, 4), (5, 4), (4, 5), (5, 5)}

    if animal is not None:
        want = K.ANIMAL_NAMES.index(animal)
        kind = ["COOP", "PASTURE", "PASTURE"][want]
        if isinstance(tile, dict) and tile.get("kind") == kind and "animal" not in tile:
            return unit(action, 0, K.OP_PLACE, ITEM_INDEX[animal])
        spot = find(tiles, lambda t: isinstance(t, dict) and t.get("kind") == kind and "animal" not in t)
        if spot is None:
            if tile is None:
                return unit(action, 0, K.OP_BUILD_COOP if want == 0 else K.OP_BUILD_PASTURE)
            return unit(action, 0, K.OP_EAST if x < 4 else K.OP_WEST)
        tx, ty = spot
        if tx != x:
            return unit(action, 0, K.OP_EAST if tx > x else K.OP_WEST)
        return unit(action, 0, K.OP_SOUTH if ty > y else K.OP_NORTH)

    if isinstance(tile, dict) and "animal" in tile:
        starving = (x, y) == (0, 0)
        if tile["yield_units"] > 0:
            return unit(action, 0, K.OP_HARVEST)
        if not starving and held.get("WHEAT", 0) > 0 and not tile["fed_today"]:
            return unit(action, 0, K.OP_FEED)
        if not tile["cared_today"]:
            return unit(action, 0, K.OP_CARE)
        if tile["fertilizer_available"]:
            return unit(action, 0, K.OP_COLLECT_FERTILIZER)

    if adjacent:
        want = next((name for name in K.ANIMAL_NAMES if shed.get(name, 0) > 0), None)
        if want is not None:
            return unit(action, 0, K.OP_PICKUP, ITEM_INDEX[want], 1)
        if shed.get("WHEAT", 0) > 0 and held.get("WHEAT", 0) < 3:
            return unit(action, 0, K.OP_PICKUP, K.WHEAT, 3)

    target = find(tiles, lambda t: isinstance(t, dict) and "animal" in t and not t.get("fed_today"))
    tx, ty = target if target is not None else shed_access(len(tiles))
    if tx != x:
        return unit(action, 0, K.OP_EAST if tx > x else K.OP_WEST)
    if ty != y:
        return unit(action, 0, K.OP_SOUTH if ty > y else K.OP_NORTH)
    return unit(action, 0, K.OP_BUILD_COOP if tile is None else K.OP_PASS)


def scenario_glut(rng, obs, player):
    farm = obs["farms"][player]
    private = obs["private"]
    action = blank(1 + len(farm["hands"]))
    shed = private["shed"]
    hour = obs.get("hour", 0)

    slot = 0
    for name in ("STRAWBERRY", "MELON", "WOOL", "MILK", "WHEAT", "CARROT"):
        if shed.get(name, 0) > 0 and slot < MAX_ORDERS - 2:
            order(action, slot, K.MK_SELL, PRODUCTS_INDEX[name], int(shed[name]))
            slot += 1
    if farm["money"] > 300 and sum(private["seeds"].values()) < 4 and slot < MAX_ORDERS:
        order(action, slot, K.MK_BUY_SEED, CROP_INDEX["STRAWBERRY" if hour % 2 else "MELON"], 2)

    x, y = farm["farmer"]
    tile = farm["tiles"][y][x]
    day = obs.get("day", 0)
    if tile is None and sum(private["seeds"].values()) > 0:
        crop = next(name for name in K.CROP_NAMES if private["seeds"].get(name, 0) > 0)
        return unit(action, 0, K.OP_PLANT, CROP_INDEX[crop])
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop = K.CROP_NAMES.index(tile["crop"])
        if day - tile["planted_day"] >= K.CROP_FIRST_YIELD_DAY[crop] and tile["yield_units"] > 0:
            return unit(action, 0, K.OP_HARVEST)
        if not tile["watered_today"]:
            return unit(action, 0, K.OP_WATER)
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return unit(action, 0, K.OP_DIG)
    return unit(action, 0, [K.OP_NORTH, K.OP_SOUTH, K.OP_EAST, K.OP_WEST][(x + y + hour) % 4])


def scenario_overflow(rng, obs, player):
    farm = obs["farms"][player]
    private = obs["private"]
    action = blank(1 + len(farm["hands"]))
    x, y = farm["farmer"]
    tiles = farm["tiles"]
    tile = tiles[y][x]
    shed = private["shed"]
    held = private["inventories"][0]
    stock = sum(shed.get(name, 0) for name in K.ITEMS)
    hour = obs.get("hour", 0)

    slot = 0
    if farm["money"] > 400 and stock < 120:
        order(action, slot, K.MK_BUY_PRODUCT, K.WHEAT, 9)
        order(action, slot + 1, K.MK_BUY_PRODUCT, K.FERTILIZER, 3)
        slot += 2
    if farm["money"] > 800 and stock < 90 and hour % 5 == 0:
        order(action, slot, K.MK_BUY_ANIMAL, ITEM_INDEX["GOOSE"], 1)
        slot += 1
    if farm["money"] > 200 and sum(private["seeds"].values()) < 3:
        order(action, slot, K.MK_BUY_SEED, CROP_INDEX["WHEAT"], 2)

    adjacent = (x, y) in {(4, 4), (5, 4), (4, 5), (5, 5)}
    carried = sum(held.values())
    if adjacent and carried > 6:
        return unit(action, 0, K.OP_DROP)
    if adjacent:
        stocked = [name for name in K.ITEMS if shed.get(name, 0) > 0]
        if stocked:
            return unit(action, 0, K.OP_PICKUP, ITEM_INDEX[stocked[hour % len(stocked)]], 4)
    if tile is None and private["seeds"].get("WHEAT", 0) > 0:
        return unit(action, 0, K.OP_PLANT, CROP_INDEX["WHEAT"])
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        if tile["yield_units"] > 0 and obs.get("day", 0) - tile["planted_day"] >= 2:
            return unit(action, 0, K.OP_HARVEST)
        if not tile["watered_today"]:
            return unit(action, 0, K.OP_WATER)
    tx, ty = shed_access(len(tiles))
    if tx != x:
        return unit(action, 0, K.OP_EAST if tx > x else K.OP_WEST)
    if ty != y:
        return unit(action, 0, K.OP_SOUTH if ty > y else K.OP_NORTH)
    return unit(action, 0, K.OP_PASS)


def scenario_crowd(rng, obs, player):
    farm = obs["farms"][player]
    private = obs["private"]
    hands = len(farm["hands"])
    action = blank(1 + hands)
    seeds = private["seeds"]

    slot = 0
    if farm["money"] > 300 and hands < NUM_UNITS - 4:
        for _ in range(3):
            order(action, slot, K.MK_HIRE)
            slot += 1
    if farm["money"] > 200 and sum(seeds.values()) < 3:
        order(action, slot, K.MK_BUY_SEED, CROP_INDEX["WHEAT"], 1)
        slot += 1
    for name in K.PRODUCTS:
        if private["shed"].get(name, 0) > 0 and slot < MAX_ORDERS:
            order(action, slot, K.MK_SELL, PRODUCTS_INDEX[name], int(private["shed"][name]))
            slot += 1

    for index in range(1 + hands):
        x, y = (farm["farmer"] if index == 0 else farm["hands"][index - 1])[:2]
        tile = farm["tiles"][y][x]
        if tile is None:
            unit(action, index, K.OP_PLANT, CROP_INDEX["WHEAT"])
        elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
            unit(action, index, K.OP_HARVEST if tile["yield_units"] > 0 else K.OP_WATER)
        elif isinstance(tile, dict) and tile.get("kind") == "WEED":
            unit(action, index, K.OP_DIG)
        else:
            unit(action, index, [K.OP_NORTH, K.OP_SOUTH, K.OP_EAST, K.OP_WEST][(index + x + y) % 4])
    return action


def scenario_seed_contention(rng, obs, player):
    farm = obs["farms"][player]
    private = obs["private"]
    hands = len(farm["hands"])
    action = blank(1 + hands)
    positions = [farm["farmer"], *farm["hands"]]
    crop = "MELON"
    quadrant = len(farm["tiles"]) // 2

    planters, movers = [], []
    for index in range(1 + hands):
        x, y = positions[index][0], positions[index][1]
        tx, ty = index % quadrant, (index // quadrant) % quadrant
        if (x, y) != (tx, ty):
            movers.append((index, toward(x, y, tx, ty)))
        elif farm["tiles"][y][x] is None:
            planters.append(index)

    slot = 0
    if hands < 6 and farm["money"] > 1000:
        order(action, slot, K.MK_HIRE)
        slot += 1
    want = 1 + hands
    held = private["seeds"].get(crop, 0)
    if held < want and farm["money"] > 500:
        order(action, slot, K.MK_BUY_SEED, CROP_INDEX[crop], want - held)

    for index, move in movers:
        unit(action, index, move)
    for index in planters:
        unit(action, index, K.OP_PLANT, CROP_INDEX[crop])
    moving = {index for index, _ in movers}
    for index in range(1 + hands):
        x, y = positions[index][0], positions[index][1]
        tile = farm["tiles"][y][x]
        if index in planters or index in moving:
            continue
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            unit(action, index, K.OP_HARVEST if tile["yield_units"] > 0 else K.OP_DIG)
    return action


def scenario_roundtrip(rng, obs, player):
    farm = obs["farms"][player]
    shed = obs["private"]["shed"]
    action = blank(1 + len(farm["hands"]))
    item = K.WHEAT if obs.get("hour", 0) % 2 == 0 else K.FERTILIZER
    order(action, 0, K.MK_BUY_PRODUCT, item, 5)
    if shed.get(K.PRODUCTS[item], 0) > 0:
        order(action, 1, K.MK_SELL, item, int(shed[K.PRODUCTS[item]]))
    if player == 1:
        for name in ("market_op", "market_item", "market_n"):
            action[name][[0, 1]] = action[name][[1, 0]]
    return unit(action, 0, K.OP_PASS)


def scenario_decay(rng, obs, player):
    farm = obs["farms"][player]
    private = obs["private"]
    action = blank(1 + len(farm["hands"]))
    x, y = farm["farmer"]
    tile = farm["tiles"][y][x]
    day = obs.get("day", 0)

    if farm["money"] > 200 and sum(private["seeds"].values()) < 4:
        order(action, 0, K.MK_BUY_SEED, CROP_INDEX[K.CROP_NAMES[day % 5]], 2)
    if tile is None and sum(private["seeds"].values()) > 0:
        crop = next(name for name in K.CROP_NAMES if private["seeds"].get(name, 0) > 0)
        return unit(action, 0, K.OP_PLANT, CROP_INDEX[crop])
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop = K.CROP_NAMES.index(tile["crop"])
        if day - tile["planted_day"] <= K.CROP_MAX_YIELD_DAY[crop] and not tile["watered_today"]:
            return unit(action, 0, K.OP_WATER)
        return unit(action, 0, K.OP_PASS)
    return unit(action, 0, [K.OP_NORTH, K.OP_SOUTH, K.OP_EAST, K.OP_WEST][(x + y + obs.get("hour", 0)) % 4])


def farm_unit(action, index, obs, farm, private, crops, harvest=True):
    x, y = (farm["farmer"] if index == 0 else farm["hands"][index - 1])[:2]
    tile = farm["tiles"][y][x]
    day = obs.get("day", 0)
    if tile is None:
        ready = [name for name in crops if private["seeds"].get(name, 0) > 0]
        if ready:
            return unit(action, index, K.OP_PLANT, CROP_INDEX[ready[(x + y) % len(ready)]])
        return unit(action, index, [K.OP_NORTH, K.OP_SOUTH, K.OP_EAST, K.OP_WEST][(x + y) % 4])
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop = K.CROP_NAMES.index(tile["crop"])
        mature = day - tile["planted_day"] >= K.CROP_FIRST_YIELD_DAY[crop]
        if harvest and mature and tile["yield_units"] > 0:
            return unit(action, index, K.OP_HARVEST)
        if not tile["watered_today"]:
            return unit(action, index, K.OP_WATER)
        return unit(action, index, K.OP_PASS)
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return unit(action, index, K.OP_DIG)
    return unit(action, index, [K.OP_NORTH, K.OP_SOUTH, K.OP_EAST, K.OP_WEST][(x + y + day) % 4])


def scenario_fragile(rng, obs, player):
    farm = obs["farms"][player]
    private = obs["private"]
    action = blank(1 + len(farm["hands"]))
    shed = private["shed"]
    slot = 0
    for name in ("CARROT", "WHEAT", "FERTILIZER"):
        if shed.get(name, 0) > 0 and slot < MAX_ORDERS - 2:
            order(action, slot, K.MK_SELL, PRODUCTS_INDEX[name], int(shed[name]))
            slot += 1
    if private["seeds"].get("CARROT", 0) < 30 and farm["money"] > 2000:
        order(action, slot, K.MK_BUY_SEED, CROP_INDEX["CARROT"], 30)
        slot += 1
    if farm["money"] > 500 and slot < MAX_ORDERS and len(farm["hands"]) < NUM_UNITS - 3:
        order(action, slot, K.MK_HIRE)
    for index in range(1 + len(farm["hands"])):
        farm_unit(action, index, obs, farm, private, ["CARROT", "WHEAT"])
    return action


def scenario_overflow_hard(rng, obs, player):
    farm = obs["farms"][player]
    private = obs["private"]
    action = blank(1 + len(farm["hands"]))
    stock = sum(private["shed"].get(name, 0) for name in K.ITEMS)
    hour = obs.get("hour", 0)

    slot = 0
    if stock < 97 and farm["money"] > 5000:
        order(action, slot, K.MK_BUY_PRODUCT, K.WHEAT, max(1, min(6, 97 - stock)))
        order(action, slot + 1, K.MK_BUY_PRODUCT, K.FERTILIZER, max(1, min(3, 97 - stock)))
        slot += 2
    if sum(private["seeds"].values()) < 30 and farm["money"] > 500:
        for crop in ("WHEAT", "CARROT"):
            if slot < MAX_ORDERS:
                order(action, slot, K.MK_BUY_SEED, CROP_INDEX[crop], 15)
                slot += 1
    if farm["money"] > 500 and hour % 6 == 0 and slot < MAX_ORDERS and len(farm["hands"]) < NUM_UNITS - 3:
        order(action, slot, K.MK_HIRE)
    for index in range(1 + len(farm["hands"])):
        farm_unit(action, index, obs, farm, private, ["WHEAT", "CARROT", "MELON"])
    return action


def scenario_menagerie(rng, obs, player):
    farm = obs["farms"][player]
    private = obs["private"]
    tiles = farm["tiles"]
    action = blank(1 + len(farm["hands"]))
    x, y = farm["farmer"]
    tile = tiles[y][x]
    shed, held = private["shed"], private["inventories"][0]
    day = obs.get("day", 0)

    slot = 0
    for index, name in enumerate(K.ANIMAL_NAMES):
        owned = shed.get(name, 0) + held.get(name, 0)
        placed = sum(1 for row in tiles for t in row if isinstance(t, dict) and t.get("animal") == name)
        if owned + placed < 2 and farm["money"] > 2000 and slot < MAX_ORDERS:
            order(action, slot, K.MK_BUY_ANIMAL, K.ANIMAL_ITEM_BASE + index, 1)
            slot += 1
    if shed.get("WHEAT", 0) < 8 and farm["money"] > 2000 and slot < MAX_ORDERS:
        order(action, slot, K.MK_BUY_PRODUCT, K.WHEAT, 8)

    animal = next((name for name in K.ANIMAL_NAMES if held.get(name, 0) > 0), None)
    if animal is not None:
        kind = ["COOP", "PASTURE", "PASTURE"][K.ANIMAL_NAMES.index(animal)]
        spot = find(tiles, lambda t: isinstance(t, dict) and t.get("kind") == kind and "animal" not in t)
        if spot is not None:
            tx, ty = spot
            if (tx, ty) == (x, y):
                return unit(action, 0, K.OP_PLACE, ITEM_INDEX[animal])
            if tx != x:
                return unit(action, 0, K.OP_EAST if tx > x else K.OP_WEST)
            return unit(action, 0, K.OP_SOUTH if ty > y else K.OP_NORTH)
        if tile is None:
            return unit(action, 0, K.OP_BUILD_COOP if kind == "COOP" else K.OP_BUILD_PASTURE)
        return unit(action, 0, [K.OP_NORTH, K.OP_SOUTH, K.OP_EAST, K.OP_WEST][(x + y) % 4])

    if isinstance(tile, dict) and "animal" in tile:
        if tile["yield_units"] > 0:
            return unit(action, 0, K.OP_HARVEST)
        if day % 3 != 2 and held.get("WHEAT", 0) > 0 and not tile["fed_today"]:
            return unit(action, 0, K.OP_FEED)
        if not tile["cared_today"]:
            return unit(action, 0, K.OP_CARE)
        if tile["fertilizer_available"]:
            return unit(action, 0, K.OP_COLLECT_FERTILIZER)

    if (x, y) in {(4, 4), (5, 4), (4, 5), (5, 5)}:
        want = next((name for name in K.ANIMAL_NAMES if shed.get(name, 0) > 0), None)
        if want is not None:
            return unit(action, 0, K.OP_PICKUP, ITEM_INDEX[want], 1)
        if shed.get("WHEAT", 0) > 0 and held.get("WHEAT", 0) < 4:
            return unit(action, 0, K.OP_PICKUP, K.WHEAT, 4)

    for ty, row in enumerate(tiles):
        for tx, t in enumerate(row):
            if isinstance(t, dict) and "animal" in t and not t.get("fed_today") and (tx, ty) != (x, y):
                return unit(action, 0, toward(x, y, tx, ty))
    tx, ty = shed_access(len(tiles))
    if (tx, ty) != (x, y):
        return unit(action, 0, toward(x, y, tx, ty))
    if tile is None:
        pastures = sum(1 for row in tiles for t in row if isinstance(t, dict) and t.get("kind") == "PASTURE")
        return unit(action, 0, K.OP_BUILD_PASTURE if pastures < 3 else K.OP_BUILD_COOP)
    return unit(action, 0, K.OP_PASS)


CONFIGS = {
    "fragile": {
        "startingMoney": 200000,
        "marketParams": {
            "WHEAT": {"T": 1, "above_target": 5.0, "above_func": "sq"},
            "FERTILIZER": {"T": 2, "above_target": 4.0, "above_func": "linear"},
            "CARROT": {"T": 1, "above_target": 6.0, "above_func": "sqrt"},
        },
    },
    "overflow_hard": {"startingMoney": 200000},
    "menagerie": {"startingMoney": 200000},
    "seed_contention": {"startingMoney": 200000},
}

SCENARIOS = {
    "animals": scenario_animals,
    "glut": scenario_glut,
    "overflow": scenario_overflow,
    "crowd": scenario_crowd,
    "seed_contention": scenario_seed_contention,
    "roundtrip": scenario_roundtrip,
    "decay": scenario_decay,
    "fragile": scenario_fragile,
    "overflow_hard": scenario_overflow_hard,
    "menagerie": scenario_menagerie,
}

BLANK_TILE = dict(
    kind=K.EMPTY, crop=-1, animal=-1, day=0, watered=0, unwatered=0, units=0,
    lifespan=-1, fert_until=-1, fed=0, unfed=0, cared=0, fert_ready=0, care_bonus=0,
)
CTILE_FIELDS = (
    "kind", "crop", "animal", "planted_day", "watered_today",
    "consecutive_unwatered", "yield_units", "max_lifespan_step",
    "fertilized_until_day", "fed_today", "consecutive_unfed", "cared_today",
    "fertilizer_available", "pending_care_bonus",
)


def reference_tile(tile):
    out = dict(BLANK_TILE)
    if tile is None:
        return out
    if tile == "LOCKED":
        out["kind"] = K.LOCKED
        return out
    kind = tile.get("kind")
    if "animal" in tile:
        out.update(
            kind=KINDS[kind],
            animal=ANIMAL_INDEX[tile["animal"]],
            day=tile["placed_day"],
            units=tile["yield_units"],
            fed=int(tile["fed_today"]),
            unfed=tile["consecutive_unfed"],
            cared=int(tile["cared_today"]),
            fert_ready=int(tile["fertilizer_available"]),
            care_bonus=tile.get("pending_care_bonus", 0),
        )
        return out
    if kind == "PLANT":
        out.update(
            kind=K.PLANT,
            crop=CROP_INDEX[tile["crop"]],
            day=tile["planted_day"],
            watered=int(tile["watered_today"]),
            unwatered=tile["consecutive_unwatered"],
            units=tile["yield_units"],
            lifespan=tile["max_lifespan_step"],
            fert_until=tile["fertilized_until_day"],
        )
        return out
    out["kind"] = KINDS[kind]
    return out


def reference_snapshot(environment):
    public = environment.state[0].observation
    size = len(public.farms[0]["tiles"])
    tiles = np.zeros((2, size, size, len(BLANK_TILE)), np.int64)
    for player, farm in enumerate(public.farms):
        for y in range(size):
            for x in range(size):
                tile = reference_tile(farm["tiles"][y][x])
                tiles[player, y, x] = [tile[field] for field in K.TILE_FIELDS]
    privates = [environment.state[player].observation.private for player in range(2)]
    return {
        "tiles": tiles,
        "money": np.array([farm["money"] for farm in public.farms], np.float64),
        "hands": np.array([len(farm["hands"]) for farm in public.farms], np.int64),
        "hires": np.array([farm["hires_today"] for farm in public.farms], np.int64),
        "quadrants": np.array([len(farm["unlocked_quadrants"]) - 1 for farm in public.farms], np.int64),
        "pos": [[list(farm["farmer"])] + [list(hand) for hand in farm["hands"]] for farm in public.farms],
        "shed": np.array([[private["shed"].get(name, 0) for name in K.ITEMS] for private in privates], np.int64),
        "seeds": np.array([[private["seeds"].get(name, 0) for name in K.CROP_NAMES] for private in privates], np.int64),
        "inv": [[[held.get(name, 0) for name in K.ITEMS] for held in private["inventories"]] for private in privates],
        "market": np.array([public.market["inventory"][name] for name in K.PRODUCTS], np.int64),
        "prices": np.array([public.market["prices"][name] for name in K.PRODUCTS], np.int64),
        "shops": list(public.town["unlocked_shops"]),
        "done": np.array([state.status == "DONE" for state in environment.state], np.int64),
        "reward": np.array([float(state.reward or 0.0) for state in environment.state], np.float64),
    }


def port_snapshot(port):
    state = port.state
    tiles = np.zeros((K.PLAYERS, K.BOARD, K.BOARD, len(CTILE_FIELDS)), np.int64)
    farms = [state.farms[player] for player in range(K.PLAYERS)]
    for player, farm in enumerate(farms):
        for y in range(K.BOARD):
            for x in range(K.BOARD):
                tile = farm.tiles[y * K.BOARD + x]
                tiles[player, y, x] = [getattr(tile, field) for field in CTILE_FIELDS]
    return {
        "tiles": tiles,
        "money": np.array([farm.money for farm in farms], np.float64),
        "hands": np.array([farm.num_units - 1 for farm in farms], np.int64),
        "hires": np.array([farm.hires_today for farm in farms], np.int64),
        "quadrants": np.array([farm.quadrants for farm in farms], np.int64),
        "pos": [[[farm.unit_x[index], farm.unit_y[index]] for index in range(farm.num_units)] for farm in farms],
        "shed": np.array([[farm.shed[index] for index in range(K.NUM_ITEMS)] for farm in farms], np.int64),
        "seeds": np.array([[farm.seeds[index] for index in range(K.NUM_CROPS)] for farm in farms], np.int64),
        "inv": [
            [[farm.inv[index][item] for item in range(K.NUM_ITEMS)] for index in range(farm.num_units)]
            for farm in farms
        ],
        "market": np.array([state.market[index] for index in range(K.NUM_PRODUCTS)], np.int64),
        "prices": np.array([market_price(index, state.market[index]) for index in range(K.NUM_PRODUCTS)], np.int64),
        "shops": [K.SHOP_NAMES[state.shop_order[index]] for index in range(state.shops_unlocked)],
        "done": np.array([state.done] * K.PLAYERS, np.int64),
        "reward": np.array([state.reward[player] for player in range(K.PLAYERS)], np.float64),
    }


def diff(want, got):
    problems = []
    for name in want:
        if name in ("pos", "inv", "shops"):
            if want[name] != got[name]:
                problems.append(f"{name}: reference={want[name]} port={got[name]}")
            continue
        if not np.array_equal(want[name], got[name]):
            where = np.argwhere(np.asarray(want[name]) != np.asarray(got[name]))
            first = tuple(where[0])
            problems.append(
                f"{name}: {len(where)} cells differ, first {first} "
                f"reference={np.asarray(want[name])[first]} port={np.asarray(got[name])[first]}"
            )
    return problems


def reference_shop_order(policy, steps, configuration):
    ref = reference()
    environment = interpreter(configuration)
    environment.reset(2)
    rng = np.random.default_rng(configuration["seed"])
    for _ in range(steps - 1):
        environment.step([to_reference(policy(rng, environment.state[p].observation, p)) for p in range(2)])
    unlocked = environment.state[0].observation.town["unlocked_shops"]
    shops = [K.SHOP_NAMES.index(shop) for shop in unlocked]
    return shops + [0] * (ref.MAX_SHOP_INSTANCES - len(shops))


def race(policy, steps, seed, extra=None):
    ref = reference()
    extra = extra or {}
    configuration = {"episodeSteps": steps, "weedSpawnChance": 0.0, "seed": seed, **extra}
    shops = reference_shop_order(policy, steps, configuration)
    environment = interpreter(configuration)
    environment.reset(2)
    params = extra.get("marketParams")
    port = Port(
        shop_order=shops,
        episode_steps=steps,
        starting_money=extra.get("startingMoney", 3000),
        market_params=ref._resolve_market_params(params) if params else None,
    )
    rng = np.random.default_rng(seed)
    try:
        for step in range(steps - 1):
            encoded = [policy(rng, environment.state[p].observation, p) for p in range(2)]
            environment.step([to_reference(action) for action in encoded])
            port.step(pack(encoded))
            problems = diff(reference_snapshot(environment), port_snapshot(port))
            if problems:
                return step, problems
        return None, []
    finally:
        set_curves(None)


def test_the_tables_match_the_reference():
    ref = reference()
    assert list(ref.CROPS) == K.CROP_NAMES
    assert ref.PRODUCTS == K.PRODUCTS
    assert list(ref.ANIMALS) == K.ANIMAL_NAMES
    for index, name in enumerate(K.CROP_NAMES):
        crop = ref.CROPS[name]
        assert (
            int(K.CROP_SEED[index]), int(K.CROP_FIRST_YIELD_DAY[index]), int(K.CROP_MAX_YIELD_DAY[index]),
            int(K.CROP_INTERVAL[index]), int(K.CROP_MAX_YIELD[index]), bool(K.CROP_ONGOING[index]),
        ) == (
            crop["seed"], crop["first_yield_day"], crop["max_yield_day"],
            crop["interval"], crop["max_yield"], crop["ongoing"],
        )
    for index, name in enumerate(K.ANIMAL_NAMES):
        animal = ref.ANIMALS[name]
        assert (
            int(K.ANIMAL_COST[index]), int(K.ANIMAL_FIRST_YIELD_DAY[index]), int(K.ANIMAL_INTERVAL[index]),
            int(K.ANIMAL_MAX_HELD[index]), int(K.ANIMAL_PRODUCT[index]), int(K.ANIMAL_STRUCTURE[index]),
        ) == (
            animal["cost"], animal["first_yield_day"], animal["interval"], animal["max_held"],
            K.PRODUCTS.index(animal["product"]), KINDS[animal["structure"]],
        )


@pytest.mark.parametrize(
    "overrides",
    [
        None,
        {
            "WHEAT": {"T": 1, "above_target": 5.0, "above_func": "sq"},
            "MELON": {"base": 999, "T": 7, "below_func": "log10", "below_target": 1.5,
                      "above_func": "sqrt", "above_target": 0.05},
        },
    ],
    ids=["defaults", "overrides"],
)
def test_the_engine_quotes_the_reference_prices(overrides):
    ref = reference()
    resolved = ref._resolve_market_params(overrides)
    set_curves(resolved if overrides else None)
    try:
        for index, item in enumerate(K.PRODUCTS):
            for inventory in np.unique(np.concatenate([np.arange(9_700, 10_300), np.arange(0, 20_000, 53)])):
                assert ref.market_price(item, int(inventory), resolved) == market_price(index, inventory), (
                    item, int(inventory)
                )
    finally:
        set_curves(None)


def test_hires_cost_the_reference_fibonacci_run():
    ref = reference()
    for hires in range(40):
        assert ref._hire_cost(hires, 1) == int(K.FIB[hires])


@pytest.mark.parametrize("size", [10, 6, 4])
def test_the_shed_sits_where_the_reference_puts_it(size):
    ref = reference()
    assert [tuple(tile) for tile in K.shed_access_tiles(size)] == [tuple(tile) for tile in ref._shed_access_tiles(size)]
    assert tuple(K.default_spawn(size)) == tuple(ref._default_spawn(size))
    for x in range(size):
        for y in range(size):
            assert bool(K.is_shed_adjacent(jnp.array([x, y]), size)) == ref._is_shed_adjacent((x, y), size)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scripted_scenarios_follow_the_reference(name):
    step, problems = race(SCENARIOS[name], 240, 0, CONFIGS.get(name))
    assert step is None, f"{name} diverged at step {step}: {problems[:4]}"


@pytest.mark.parametrize("seed", range(6))
def test_random_episodes_follow_the_reference(seed):
    step, problems = race(lambda rng, obs, player: sample_action(rng, obs), 720, seed)
    assert step is None, f"diverged at step {step}: {problems[:4]}"


LIVESTOCK = [
    {"farmer": ["PASS"], "hands": [], "market": [["BUY_ANIMAL", "GOOSE", 1]]},
    {"farmer": ["PICKUP", "GOOSE", 1], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 4]]},
    {"farmer": ["BUILD_COOP"], "hands": [], "market": []},
    {"farmer": ["PLACE", "GOOSE", 1], "hands": [], "market": [["BUY_LAND"]]},
    {"farmer": ["PICKUP", "WHEAT", 2], "hands": [], "market": []},
    {"farmer": ["FEED"], "hands": [], "market": []},
    {"farmer": ["CARE"], "hands": [], "market": []},
    {"farmer": ["COLLECT_FERTILIZER"], "hands": [], "market": []},
]


def livestock(obs, configuration):
    step = obs["step"]
    if step < len(LIVESTOCK):
        return LIVESTOCK[step]
    return {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}


def guided():
    rng = np.random.default_rng(0)

    def agent(obs, configuration):
        return to_reference(sample_action(rng, obs))

    return agent


def episode(policy, seed, steps):
    agent = {"random": "random", "guided": guided(), "livestock": livestock}[policy]
    environment = interpreter({"episodeSteps": steps, "weedSpawnChance": 0.0, "seed": seed})
    environment.run([agent, agent])
    return environment.toJSON()


def walk(replay):
    unlocked = replay["steps"][-1][0]["observation"]["town"]["unlocked_shops"]
    shops = [K.SHOP_NAMES.index(shop) for shop in unlocked]
    port = Port(episode_steps=len(replay["steps"]), shop_order=shops + [0] * (K.NUM_SHOPS - len(shops)))
    buffer = (CAction * K.PLAYERS)()
    for index, frame in enumerate(replay["steps"][:-1]):
        yield index, replays.arrays(frame), observation_arrays(port.state)
        for player in range(K.PLAYERS):
            encode(replay["steps"][index + 1][player]["action"], buffer[player])
        port.step(buffer)


@pytest.mark.parametrize("policy, steps", [("random", 96), ("guided", 240), ("livestock", 60)])
def test_a_replay_reads_back_the_observation_the_engine_produces(policy, steps):
    for index, got, want in walk(episode(policy, 0, steps)):
        for name in replays.OBS_KEYS:
            np.testing.assert_array_equal(got[name], want[name], err_msg=f"{index} {name}")


def test_reading_replays_populates_every_field():
    seen = {name: 0 for name in replays.OBS_KEYS}
    tiles = np.zeros(K.NUM_OBS_TILE_FIELDS, np.int64)
    empty = np.array([0, -1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0])
    for policy, steps in (("guided", 240), ("livestock", 60)):
        for _, got, _ in walk(episode(policy, 0, steps)):
            for name in replays.OBS_KEYS:
                seen[name] = max(seen[name], int(np.count_nonzero(got[name])))
            tiles = np.maximum(tiles, np.count_nonzero(got["tiles"][0] != empty, axis=(0, 1, 2)))
    assert not [name for name, count in seen.items() if count == 0]
    assert not [K.OBS_TILE_FIELDS[index] for index, count in enumerate(tiles) if count == 0]


def test_actions_pair_with_the_observation_they_were_chosen_from():
    moves = ["NORTH", "EAST", "SOUTH", "WEST"]
    seen = []

    def scripted(obs, configuration):
        seen.append(tuple(obs["farms"][obs["player"]]["farmer"]))
        return {"farmer": [moves[(len(seen) - 1) % len(moves)]], "hands": [], "market": []}

    environment = interpreter({"episodeSteps": 8, "weedSpawnChance": 0.0, "seed": 3})
    environment.run([scripted, "random"])
    frames = replays.frames(environment.toJSON(), seat=0)
    for index, frame in enumerate(frames[: len(moves)]):
        assert frame["commands"][0] == [moves[index % len(moves)]]
        assert frame["pos"][0] == seen[index]


def test_frames_drop_the_trailing_observation():
    replay = episode("random", 4, 96)
    assert len(replays.frames(replay, seat=0)) == len(replay["steps"]) - 1


@pytest.mark.parametrize("seat", [0, 1])
def test_private_state_comes_from_its_own_seat(seat):
    replay = episode("guided", 5, 240)
    frame = next(
        f for f in replay["steps"]
        if any(f[player]["observation"]["private"]["shed"].values() for player in range(K.PLAYERS))
    )
    got = replays.arrays(frame)
    private = frame[seat]["observation"]["private"]
    want = np.array([private["shed"].get(item, 0) for item in K.ITEMS], np.int16)
    np.testing.assert_array_equal(got["shed"][0, seat], want)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    directory = tmp_path_factory.mktemp("replays")
    paths = []
    for seed in (0, 1):
        path = directory / f"episode-{seed}-replay.json"
        path.write_text(json.dumps(episode("guided", seed, 240)))
        paths.append(path)
    return paths


def test_labels_land_inside_the_job_space(corpus):
    _, action, _ = replays.dataset(corpus)
    assert action["chosen"].min() >= 0
    assert action["chosen"].max() < A.TILES * A.NUM_JOBS


def test_both_seats_are_read(corpus):
    both = replays.dataset(corpus)[1]["chosen"].shape[0]
    one = replays.dataset(corpus, seat=0)[1]["chosen"].shape[0]
    assert both == 2 * one


def test_the_version_filter_rejects_a_foreign_recording(corpus):
    assert replays.version(replays.load(corpus[0])) is not None
    with pytest.raises(ValueError):
        replays.dataset(corpus, module_version="0.0.0")


def test_each_seat_is_attributed_to_its_own_team(corpus):
    replay = replays.load(corpus[0])
    replay["info"] = {"TeamNames": ["alpha", "beta"]}
    path = corpus[0].parent / "named.json"
    path.write_text(json.dumps(replay))
    parts = replays.by_player([path])
    assert sorted(parts) == ["alpha", "beta"]
    for team in parts:
        assert parts[team][1]["chosen"].shape[0] == len(replay["steps"]) - 1
        assert parts[team][2].sum() == 1


def test_the_masks_in_a_label_reproduce_the_hand_rolled_likelihood(corpus):
    obs, action, _ = replays.dataset(corpus, seat=0)
    obs, action = (jax.tree.map(lambda leaf: jnp.asarray(leaf[:16]), tree) for tree in (obs, action))
    action.pop("value")
    network = Farmer(features=16, blocks=1, dtype=jnp.float32)
    params = network.init(jax.random.key(0), obs, temperature=1.0)
    dist, _ = network.apply(params, obs, temperature=1.0)

    seat = slice(0, None, SEATS)
    job, _ = A.score_jobs(dist.logits["jobs_masked"][seat], action["filled"], action["chosen"])
    table = dist.logits["quantity"][seat]
    quantity = table[jnp.arange(table.shape[0])[:, None], action["market"]]
    want = (
        job
        + categorical_log_prob(dist.logits["market"][seat], action["market"], action["allowed"])
        + categorical_log_prob(quantity, action["qty"], action["allowed"])
    )
    got = dist.log_prob(jax.tree.map(lambda leaf: jnp.stack([leaf, leaf], 1), action))
    assert jnp.allclose(got[:, 0], want, atol=1e-5)


def test_a_clone_learns_to_name_the_experts_choices(corpus, tmp_path):
    store.save(tmp_path / "expert.npz", replays.dataset(corpus, seat=0))
    data = store.make("expert", directory=tmp_path, horizon=8)
    network = recurrent(SSM(cell=MinGRUCell(features=FEATURES, dtype=jnp.float32)))
    algorithm = RecurrentBC(
        cfg=RecurrentBCConfig(batch_size=4),
        network=network,
        optimizer=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(1e-3)),
    )
    batch = data.sample(data.init(), jax.random.key(0), (4,))
    environment = K.Kaggriculture(num_envs=4, threads=1)
    environment_state, timestep = environment.init(jax.random.key(0))
    environment.close(environment_state)
    state = algorithm.init(jax.random.key(0), timestep)

    def score(state):
        carry = network.initialize_carry(jax.random.key(0), (4, 1))
        _, dist = network.apply(
            state.params, batch.first.obs, done=batch.first.done, carry=carry, temperature=1.0
        )
        return float(dist.log_prob(batch.second.action)[..., 0].mean())

    before = score(state)
    update = jax.jit(algorithm.update)
    key = jax.random.key(1)
    for _ in range(20):
        key, draw, step = jax.random.split(key, 3)
        state = update(state, step, data.sample(data.init(), draw, (4,)))
    after = score(state)
    assert np.isfinite(before) and np.isfinite(after)
    assert after > before
