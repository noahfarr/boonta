import jax
import jax.numpy as jnp
import numpy as np

from . import (ANIMAL_ITEM_BASE, ANIMAL_COST, ANIMAL_STRUCTURE, BOARD, COOP,
               CROP_FIRST_YIELD_DAY, CROP_NAMES, CROP_ONGOING, CROP_SEED, EMPTY,
               FIB, LAND_PRICES as LAND_COSTS, LOCKED, MAX_ORDERS, NUM_ANIMALS,
               NUM_CATEGORIES, NUM_ITEMS, NUM_PRODUCTS, OBS_TILE_FIELDS,
               OBS_UNITS, OP_EAST, OP_NORTH, OP_PASS, OP_SOUTH, OP_WEST,
               PASTURE, PLANT, PLAYERS, PRODUCTS, UNIT_OP_NAMES, WEED,
               is_shed_adjacent, seat_view)

TILES = BOARD * BOARD
UNITS = OBS_UNITS
ORDERS = MAX_ORDERS
NUM_CROPS = len(CROP_NAMES)

F = {name: index for index, name in enumerate(OBS_TILE_FIELDS)}

WHEAT_ITEM = PRODUCTS.index("WHEAT")
FERT_ITEM = PRODUCTS.index("FERTILIZER")

JOBS = [("PASS", 0)]
JOBS += [("PLANT", crop) for crop in range(NUM_CROPS)]
JOBS += [("WATER", 0), ("HARVEST", 0), ("FERTILIZE", 0), ("DIG", 0),
         ("BUILD_COOP", 0), ("BUILD_PASTURE", 0)]
JOBS += [("PLACE", ANIMAL_ITEM_BASE + animal) for animal in range(NUM_ANIMALS)]
JOBS += [("FEED", 0), ("COLLECT_FERTILIZER", 0), ("CARE", 0)]
JOBS += [("PICKUP", WHEAT_ITEM), ("PICKUP", FERT_ITEM)]
JOBS += [("PICKUP", ANIMAL_ITEM_BASE + animal) for animal in range(NUM_ANIMALS)]
NUM_JOBS = len(JOBS)

JOB_OP = np.array([UNIT_OP_NAMES.index(op) for op, _ in JOBS], np.int32)
JOB_ARG = np.array([arg for _, arg in JOBS], np.int32)

JOB_PASS = 0
JOB_PLANT = np.array([JOBS.index(("PLANT", crop)) for crop in range(NUM_CROPS)])
JOB_WATER = JOBS.index(("WATER", 0))
JOB_HARVEST = JOBS.index(("HARVEST", 0))
JOB_FERTILIZE = JOBS.index(("FERTILIZE", 0))
JOB_DIG = JOBS.index(("DIG", 0))
JOB_BUILD = np.array([JOBS.index(("BUILD_COOP", 0)), JOBS.index(("BUILD_PASTURE", 0))])
JOB_PLACE = np.array(
    [JOBS.index(("PLACE", ANIMAL_ITEM_BASE + animal)) for animal in range(NUM_ANIMALS)]
)
JOB_FEED = JOBS.index(("FEED", 0))
JOB_COLLECT = JOBS.index(("COLLECT_FERTILIZER", 0))
JOB_CARE = JOBS.index(("CARE", 0))
JOB_PICKUP = np.array(
    [
        JOBS.index(("PICKUP", WHEAT_ITEM)),
        JOBS.index(("PICKUP", FERT_ITEM)),
        *[JOBS.index(("PICKUP", ANIMAL_ITEM_BASE + animal)) for animal in range(NUM_ANIMALS)],
    ]
)
PICKUP_ITEM = np.array(
    [WHEAT_ITEM, FERT_ITEM, *[ANIMAL_ITEM_BASE + animal for animal in range(NUM_ANIMALS)]]
)

columns, rows = np.meshgrid(np.arange(BOARD), np.arange(BOARD), indexing="xy")
TILE_X = jnp.asarray(columns.reshape(-1), jnp.int32)
TILE_Y = jnp.asarray(rows.reshape(-1), jnp.int32)
SHED_MASK = jnp.asarray(
    np.array(
        [
            is_shed_adjacent(np.array([x, y]), BOARD)
            for x, y in zip(columns.reshape(-1), rows.reshape(-1))
        ],
        bool,
    )
)
ONGOING = jnp.asarray(np.asarray(CROP_ONGOING, bool))

MARKET_NONE = 0
MARKET_BUY_SEED = np.arange(1, 1 + NUM_CROPS)
MARKET_BUY_WHEAT, MARKET_BUY_FERT = 6, 7
MARKET_BUY_ANIMAL = np.arange(8, 8 + NUM_ANIMALS)
MARKET_SELL = np.arange(11, 11 + NUM_ITEMS)
MARKET_HIRE, MARKET_LAND = 23, 24
NUM_MARKET_ACTIONS = 25
NUM_QUANTITIES = NUM_CATEGORIES

MARKET_ACTIONS = (
    [("NONE", 0)]
    + [("BUY_SEED", crop) for crop in range(NUM_CROPS)]
    + [("BUY_PRODUCT", WHEAT_ITEM), ("BUY_PRODUCT", FERT_ITEM)]
    + [("BUY_ANIMAL", ANIMAL_ITEM_BASE + animal) for animal in range(NUM_ANIMALS)]
    + [("SELL", item) for item in range(NUM_ITEMS)]
    + [("HIRE", 0), ("BUY_LAND", 0)]
)
MARKET_INDEX = {pair: index for index, pair in enumerate(MARKET_ACTIONS)}

LAND_PRICES = jnp.asarray(LAND_COSTS, jnp.float32)
HIRE_PRICES = jnp.asarray(FIB[:64], jnp.float32)
FORBIDDEN = 1.0e6


def legal_jobs(obs, player):
    tiles = obs["tiles"][:, player]
    seeds = obs["seeds"][:, player].astype(jnp.int32)
    shed = obs["shed"][:, player].astype(jnp.int32)
    day = obs["day"].astype(jnp.int32)[:, None]
    held = obs["inv"][:, player].astype(jnp.int32).sum(axis=1)

    batch, *_ = tiles.shape
    kind = tiles[..., F["kind"]].reshape(batch, TILES)
    crop = tiles[..., F["crop"]].reshape(batch, TILES)
    animal = tiles[..., F["animal"]].reshape(batch, TILES)
    watered = tiles[..., F["watered"]].reshape(batch, TILES)
    fed = tiles[..., F["fed"]].reshape(batch, TILES)
    cared = tiles[..., F["cared"]].reshape(batch, TILES)
    units_held = tiles[..., F["units"]].reshape(batch, TILES)
    fert_ready = tiles[..., F["fert_ready"]].reshape(batch, TILES)
    planted = tiles[..., F["day"]].reshape(batch, TILES).astype(jnp.int32)
    fert_until = tiles[..., F["fert_until"]].reshape(batch, TILES).astype(jnp.int32)

    empty = kind == EMPTY
    plant = kind == PLANT
    weed = kind == WEED
    structure = (kind == COOP) | (kind == PASTURE)
    occupied = structure & (animal >= 0)
    vacant = structure & (animal < 0)

    mask = jnp.zeros((batch, TILES, NUM_JOBS), bool)
    mask = mask.at[..., JOB_PASS].set(True)

    for index in range(NUM_CROPS):
        mask = mask.at[..., JOB_PLANT[index]].set(empty & (seeds[:, None, index] > 0))

    mask = mask.at[..., JOB_WATER].set(plant & (watered == 0))

    ripe = jnp.take(
        jnp.asarray(CROP_FIRST_YIELD_DAY), jnp.clip(crop, 0, NUM_CROPS - 1).astype(jnp.int32)
    )
    grown = (day - planted) >= ripe
    mask = mask.at[..., JOB_HARVEST].set((units_held > 0) & ((plant & grown) | occupied))
    mask = mask.at[..., JOB_FERTILIZE].set(
        plant & (fert_until < day + 2) & (held[:, None, FERT_ITEM] > 0)
    )
    mask = mask.at[..., JOB_DIG].set(weed | plant | vacant)
    mask = mask.at[..., JOB_BUILD[0]].set(empty)
    mask = mask.at[..., JOB_BUILD[1]].set(empty)

    for index in range(NUM_ANIMALS):
        housing = COOP if ANIMAL_STRUCTURE[index] == COOP else PASTURE
        mask = mask.at[..., JOB_PLACE[index]].set(
            vacant & (kind == housing) & (shed[:, None, ANIMAL_ITEM_BASE + index] > 0)
        )

    mask = mask.at[..., JOB_FEED].set(occupied & (fed == 0))
    mask = mask.at[..., JOB_COLLECT].set(occupied & (fert_ready > 0))
    mask = mask.at[..., JOB_CARE].set(occupied & (cared == 0))

    for index, item in enumerate(PICKUP_ITEM):
        mask = mask.at[..., JOB_PICKUP[index]].set(
            SHED_MASK[None, :] & (shed[:, None, int(item)] > 0)
        )

    return mask


def legal_market(obs, player):
    money = obs["money"][:, player].astype(jnp.float32)
    shed = obs["shed"][:, player].astype(jnp.int32)
    prices = obs["prices"].astype(jnp.float32)
    quadrants = obs["quadrants"][:, player].astype(jnp.int32)
    hires = obs["hires"][:, player].astype(jnp.int32)
    batch, *_ = money.shape

    mask = jnp.zeros((batch, NUM_MARKET_ACTIONS), bool).at[:, MARKET_NONE].set(True)
    for index in range(NUM_CROPS):
        mask = mask.at[:, MARKET_BUY_SEED[index]].set(money >= float(CROP_SEED[index]))
    mask = mask.at[:, MARKET_BUY_WHEAT].set(money >= prices[:, WHEAT_ITEM])
    mask = mask.at[:, MARKET_BUY_FERT].set(money >= prices[:, FERT_ITEM])
    for index in range(NUM_ANIMALS):
        mask = mask.at[:, MARKET_BUY_ANIMAL[index]].set(money >= float(ANIMAL_COST[index]))
    for item in range(NUM_ITEMS):
        sellable = item < NUM_PRODUCTS
        mask = mask.at[:, MARKET_SELL[item]].set(
            (shed[:, item] > 0) if sellable else jnp.zeros((batch,), bool)
        )
    mask = mask.at[:, MARKET_HIRE].set(
        money >= HIRE_PRICES[jnp.clip(hires, 0, HIRE_PRICES.size - 1)]
    )
    mask = mask.at[:, MARKET_LAND].set(
        (quadrants < 3) & (money >= LAND_PRICES[jnp.clip(quadrants, 0, 2)])
    )
    return mask


def unit_distance(obs, player=0):
    pos = obs["pos"][:, player].astype(jnp.int32)
    live = obs["hands"][:, player].astype(jnp.int32) + 1
    x, y = pos[..., 0], pos[..., 1]
    alive = jnp.arange(UNITS)[None, :] < live[:, None]
    distance = jnp.abs(x[:, None, :] - TILE_X[None, :, None]) + jnp.abs(
        y[:, None, :] - TILE_Y[None, :, None]
    )
    distance = jnp.where(alive[:, None, :], distance, BOARD * 2)
    return jnp.min(distance, axis=-1).astype(jnp.float32)


def select(logits, live, draw):
    batch, *_ = logits.shape
    neg = jnp.finfo(logits.dtype).min

    def pick(avail, slot):
        masked = jnp.where(avail, logits, neg)
        logz = jax.nn.logsumexp(masked, axis=-1)
        index = draw(masked, slot)
        rows = jnp.arange(batch)
        log_prob = masked[rows, index] - logz
        probs = jax.nn.softmax(masked, axis=-1)
        entropy = -jnp.sum(jnp.where(avail, probs * jnp.log(probs + 1e-12), 0.0), -1)
        active = (slot < live) & (log_prob > -jnp.inf)
        avail = avail.at[rows, index].set(False)
        return avail, (
            index,
            jnp.where(active, log_prob, 0.0),
            jnp.where(active, entropy, 0.0),
        )

    _, (index, log_prob, entropy) = jax.lax.scan(pick, logits > neg, jnp.arange(UNITS))
    return index.T, log_prob.sum(0), entropy.sum(0)


def gumbel(key):
    def draw(masked, slot):
        noise = jax.random.gumbel(jax.random.fold_in(key, slot), masked.shape, masked.dtype)
        return jnp.argmax(masked + noise, axis=-1)

    return draw


def stored(chosen):
    def draw(masked, slot):
        return chosen[:, slot]

    return draw


def greedy(masked, slot):
    return jnp.argmax(masked, axis=-1)


def sample_jobs(logits, live, key):
    return select(logits, live, gumbel(key))


def score_jobs(logits, live, chosen):
    _, log_prob, entropy = select(logits, live, stored(chosen))
    return log_prob, entropy


def greedy_jobs(logits, live):
    chosen, _, _ = select(logits, live, greedy)
    return chosen


def matching(cost):
    n = cost.shape[0]
    rows = jnp.arange(n)

    def augment(carry, current):
        u, v, row4col, col4row = carry
        start = (
            jnp.float32(0.0),
            jnp.int32(current),
            jnp.int32(-1),
            jnp.zeros(n, bool),
            jnp.zeros(n, bool),
            jnp.full(n, jnp.inf, jnp.float32),
            jnp.full(n, -1, jnp.int32),
        )

        def grow(state, _):
            low, row, sink, seen_rows, seen_cols, shortest, path = state
            seen_rows = seen_rows.at[row].set(True)
            reduced = low + cost[row] - u[row] - v
            better = (reduced < shortest) & ~seen_cols & (sink < 0)
            path = jnp.where(better, row, path)
            shortest = jnp.where(better, reduced, shortest)
            candidate = jnp.where(seen_cols, jnp.inf, shortest)
            column = jnp.argmin(candidate).astype(jnp.int32)
            live = sink < 0
            low = jnp.where(live, candidate[column], low)
            seen_cols = jnp.where(live, seen_cols.at[column].set(True), seen_cols)
            sink = jnp.where(live & (row4col[column] < 0), column, sink)
            row = jnp.where(live & (row4col[column] >= 0), row4col[column], row)
            return (low, row, sink, seen_rows, seen_cols, shortest, path), None

        (low, _, sink, seen_rows, seen_cols, shortest, path), _ = jax.lax.scan(
            grow, start, jnp.arange(n)
        )

        u = u.at[current].add(low)
        shifted = seen_rows & (rows != current) & (col4row >= 0)
        u = jnp.where(shifted, u + low - shortest[jnp.clip(col4row, 0)], u)
        v = jnp.where(seen_cols, v - (low - shortest), v)

        def walk(state, _):
            sink, row4col, col4row, live = state
            row = path[jnp.clip(sink, 0)]
            row4col = jnp.where(live, row4col.at[jnp.clip(sink, 0)].set(row), row4col)
            following = col4row[jnp.clip(row, 0)]
            col4row = jnp.where(live, col4row.at[jnp.clip(row, 0)].set(sink), col4row)
            return (following, row4col, col4row, live & (row != current)), None

        (_, row4col, col4row, _), _ = jax.lax.scan(
            walk, (sink, row4col, col4row, jnp.bool_(True)), jnp.arange(n)
        )
        return (u, v, row4col, col4row), None

    state = (
        jnp.zeros(n, jnp.float32),
        jnp.zeros(n, jnp.float32),
        jnp.full(n, -1, jnp.int32),
        jnp.full(n, -1, jnp.int32),
    )
    (_, _, _, col4row), _ = jax.lax.scan(augment, state, jnp.arange(n))
    return col4row


def place(chosen, pos, live, travel_weight):
    job_tile = chosen // NUM_JOBS
    job_kind = chosen % NUM_JOBS
    unit_x = pos[..., 0].astype(jnp.int32)
    unit_y = pos[..., 1].astype(jnp.int32)

    alive = jnp.arange(UNITS)[None, :] < live[:, None]

    distance = (
        jnp.abs(unit_x[:, :, None] - TILE_X[job_tile][:, None, :])
        + jnp.abs(unit_y[:, :, None] - TILE_Y[job_tile][:, None, :])
    ).astype(jnp.float32) * travel_weight[:, None, None]

    cost = jnp.where(
        alive[:, :, None],
        jnp.where(alive[:, None, :], distance, 0.0),
        jnp.where(alive[:, None, :], FORBIDDEN, 0.0),
    )

    taken = jax.vmap(matching)(cost)
    ok = alive & jnp.take_along_axis(alive, taken, axis=1)
    tile_of = jnp.where(ok, jnp.take_along_axis(job_tile, taken, axis=1), -1)
    job_of = jnp.where(ok, jnp.take_along_axis(job_kind, taken, axis=1), 0)
    return tile_of.astype(jnp.int32), job_of.astype(jnp.int32)


def step_toward(x, y, tx, ty, unlocked):
    dx = jnp.sign(tx - x)
    dy = jnp.sign(ty - y)
    prefer_x = jnp.abs(tx - x) >= jnp.abs(ty - y)

    def open_at(nx, ny):
        inside = (nx >= 0) & (nx < BOARD) & (ny >= 0) & (ny < BOARD)
        index = jnp.clip(ny, 0, BOARD - 1) * BOARD + jnp.clip(nx, 0, BOARD - 1)
        return inside & jnp.take_along_axis(unlocked, index, axis=-1)

    x_ok = (dx != 0) & open_at(x + dx, y)
    y_ok = (dy != 0) & open_at(x, y + dy)
    use_x = jnp.where(prefer_x, x_ok, ~y_ok & x_ok)
    op_x = jnp.where(dx > 0, OP_EAST, OP_WEST)
    op_y = jnp.where(dy > 0, OP_SOUTH, OP_NORTH)
    return jnp.where(use_x, op_x, jnp.where(y_ok, op_y, OP_PASS))


def settle(tiles, tile_of, job_of, arrived, batch):
    same = (
        (tile_of[:, :, None] == tile_of[:, None, :])
        & arrived[:, :, None]
        & arrived[:, None, :]
        & (tile_of[:, :, None] >= 0)
    )
    index = jnp.arange(UNITS)
    before = same & (index[None, :, None] > index[None, None, :])
    after = same & (index[None, :, None] < index[None, None, :])
    other = same & (index[None, :, None] != index[None, None, :])

    crop = tiles[..., F["crop"]].reshape(batch, TILES)
    units_held = tiles[..., F["units"]].reshape(batch, TILES)

    def at(field):
        return jnp.take_along_axis(field, jnp.clip(tile_of, 0, TILES - 1), axis=1)

    clearing = at(ONGOING[jnp.clip(crop, 0, NUM_CROPS - 1)] == 0)
    holding = at(units_held) > 0

    makes = jnp.isin(job_of, jnp.concatenate([JOB_PLANT, JOB_BUILD, JOB_PLACE]))
    harvests = job_of == JOB_HARVEST

    dig = job_of == JOB_DIG
    undoes = (before & makes[:, None, :]).any(-1)
    robs = (after & harvests[:, None, :]).any(-1) & holding
    wasted = (other & harvests[:, None, :]).any(-1) & clearing

    drop = (dig & (undoes | robs)) | ((job_of == JOB_FERTILIZE) & wasted)
    return jnp.where(drop, JOB_PASS, job_of)


def emit(tiles, pos, tile_of, job_of, market_index, market_qty, batch):
    kind = tiles[..., F["kind"]].reshape(batch, TILES)
    unlocked = kind != LOCKED
    x, y = pos[..., 0], pos[..., 1]
    tx = jnp.where(tile_of >= 0, TILE_X[jnp.maximum(tile_of, 0)], x)
    ty = jnp.where(tile_of >= 0, TILE_Y[jnp.maximum(tile_of, 0)], y)
    arrived = (x == tx) & (y == ty) & (tile_of >= 0)

    job_of = settle(tiles, tile_of, job_of, arrived, batch)

    act_op = jnp.asarray(JOB_OP)[job_of]
    act_arg = jnp.asarray(JOB_ARG)[job_of]
    move_op = step_toward(x, y, tx, ty, unlocked)
    op = jnp.where(tile_of < 0, OP_PASS, jnp.where(arrived, act_op, move_op))
    arg = jnp.where(arrived, act_arg, 0)
    qty = jnp.where(jnp.isin(job_of, jnp.asarray(JOB_PICKUP)) & arrived, 5, 1)
    return jnp.concatenate(
        [
            op.astype(jnp.int32),
            arg.astype(jnp.int32),
            qty.astype(jnp.int32),
            market_index.astype(jnp.int32),
            market_qty.astype(jnp.int32),
        ],
        axis=-1,
    )


def heads_from_selection(obs, player, chosen, market_index, market_qty, travel_weight):
    tiles = obs["tiles"][:, player]
    batch, *_ = tiles.shape
    pos = obs["pos"][:, player].astype(jnp.int32)
    live = obs["hands"][:, player].astype(jnp.int32) + 1
    tile_of, job_of = place(chosen, pos, live, travel_weight)
    return emit(tiles, pos, tile_of, job_of, market_index, market_qty, batch)


def route(obs, action):
    return jnp.stack(
        [
            heads_from_selection(
                seat_view(obs, seat),
                0,
                action["chosen"][:, seat],
                action["market"][:, seat],
                action["qty"][:, seat],
                action["travel"][:, seat],
            )
            for seat in range(PLAYERS)
        ],
        axis=1,
    )
