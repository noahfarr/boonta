import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from . import (ANIMAL_NAMES, BOARD, COOP, CROP_NAMES, EMPTY, ITEMS, LOCKED,
               NUM_CROPS, NUM_ITEMS, NUM_OBS_TILE_FIELDS, NUM_PRODUCTS,
               NUM_SHOPS, OBS_TILE_FIELDS, OBS_UNITS, PASTURE, PLANT, PLAYERS,
               PRODUCTS, SHOP_NAMES, UNIT_OP_NAMES, WEED, quantities)
from . import actions as A

OBS_SPEC = (
    ("tiles", (PLAYERS, BOARD, BOARD, NUM_OBS_TILE_FIELDS), "int8"),
    ("lifespan", (PLAYERS, BOARD, BOARD), "int8"),
    ("pos", (PLAYERS, OBS_UNITS, 2), "int8"),
    ("inv", (PLAYERS, OBS_UNITS, NUM_ITEMS), "int16"),
    ("shed", (PLAYERS, NUM_ITEMS), "int16"),
    ("seeds", (PLAYERS, NUM_CROPS), "int16"),
    ("money", (PLAYERS,), "float32"),
    ("hands", (PLAYERS,), "int16"),
    ("hires", (PLAYERS,), "int16"),
    ("quadrants", (PLAYERS,), "int16"),
    ("market", (NUM_PRODUCTS,), "int32"),
    ("prices", (NUM_PRODUCTS,), "int16"),
    ("shops", (NUM_SHOPS,), "int8"),
    ("day", (), "int8"),
    ("hour", (), "int8"),
    ("reward", (PLAYERS,), "float32"),
    ("terminated", (PLAYERS,), "int8"),
)
OBS_KEYS = tuple(name for name, _, _ in OBS_SPEC if name not in ("reward", "terminated"))
FIELD = {name: index for index, name in enumerate(OBS_TILE_FIELDS)}
STRUCTURE_KIND = {"COOP": COOP, "PASTURE": PASTURE}
EMPTY_TILE = {"crop": -1, "animal": -1, "fert_until": -1}
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
JOB_INDEX = {(UNIT_OP_NAMES.index(op), arg): index for index, (op, arg) in enumerate(A.JOBS)}


def slot_of(names, value):
    try:
        return names.index(value)
    except (ValueError, TypeError):
        return None


def count_of(command, index, default=1):
    try:
        return int(command[index]) if len(command) > index else default
    except (TypeError, ValueError):
        return default


def quantity_index(n):
    usable = [index for index, quantity in enumerate(quantities()) if quantity <= n]
    return usable[-1] if usable else 0


def blank_tiles():
    tiles = np.zeros((BOARD, BOARD, NUM_OBS_TILE_FIELDS), np.int8)
    for name, value in EMPTY_TILE.items():
        tiles[..., FIELD[name]] = value
    return tiles


def write_tile(row, tile, turns_per_day=24):
    if tile == "LOCKED":
        row[FIELD["kind"]] = LOCKED
        return -1
    if tile is None:
        row[FIELD["kind"]] = EMPTY
        return -1
    kind = tile["kind"]
    if kind == "WEED":
        row[FIELD["kind"]] = WEED
        return -1
    if kind == "PLANT":
        row[FIELD["kind"]] = PLANT
        row[FIELD["crop"]] = CROP_NAMES.index(tile["crop"])
        row[FIELD["day"]] = tile["planted_day"]
        row[FIELD["watered"]] = int(tile["watered_today"])
        row[FIELD["unwatered"]] = tile["consecutive_unwatered"]
        row[FIELD["units"]] = tile["yield_units"]
        row[FIELD["fert_until"]] = tile["fertilized_until_day"]
        lifespan = tile["max_lifespan_step"]
        return -1 if lifespan < 0 else lifespan // turns_per_day
    row[FIELD["kind"]] = STRUCTURE_KIND[kind]
    if "animal" not in tile:
        return -1
    row[FIELD["animal"]] = ANIMAL_NAMES.index(tile["animal"])
    row[FIELD["day"]] = tile["placed_day"]
    row[FIELD["units"]] = tile["yield_units"]
    row[FIELD["unfed"]] = tile["consecutive_unfed"]
    row[FIELD["fed"]] = int(tile["fed_today"])
    row[FIELD["cared"]] = int(tile["cared_today"])
    row[FIELD["fert_ready"]] = int(tile["fertilizer_available"])
    row[FIELD["care_bonus"]] = tile["pending_care_bonus"]
    return -1


def arrays(records, turns_per_day=24):
    out = {name: np.zeros((1, *shape), dtype) for name, shape, dtype in OBS_SPEC}
    public = records[0]["observation"]

    for player in range(PLAYERS):
        farm = public["farms"][player]
        tiles = blank_tiles()
        lifespan = np.full((BOARD, BOARD), -1, np.int8)
        for y in range(BOARD):
            for x in range(BOARD):
                lifespan[y, x] = write_tile(tiles[y, x], farm["tiles"][y][x], turns_per_day)
        out["tiles"][0, player] = tiles
        out["lifespan"][0, player] = lifespan

        units = [farm["farmer"], *farm["hands"]]
        for unit, (x, y) in enumerate(units[:OBS_UNITS]):
            out["pos"][0, player, unit] = (x, y)
        half = BOARD // 2
        for unit in range(len(units), OBS_UNITS):
            out["pos"][0, player, unit] = (half - 1, half - 1)

        out["money"][0, player] = farm["money"]
        out["hands"][0, player] = len(units) - 1
        out["hires"][0, player] = farm["hires_today"]
        out["quadrants"][0, player] = len(farm["unlocked_quadrants"]) - 1

        private = records[player]["observation"]["private"]
        for index, item in enumerate(ITEMS):
            out["shed"][0, player, index] = private["shed"].get(item, 0)
        for index, crop in enumerate(CROP_NAMES):
            out["seeds"][0, player, index] = private["seeds"].get(crop, 0)
        for unit, holding in enumerate(private["inventories"][:OBS_UNITS]):
            for index, item in enumerate(ITEMS):
                out["inv"][0, player, unit, index] = holding.get(item, 0)

    market = public["market"]
    for index, product in enumerate(PRODUCTS):
        out["market"][0, index] = market["inventory"][product]
        out["prices"][0, index] = market["prices"][product]
    for shop in public["town"]["unlocked_shops"]:
        out["shops"][0, SHOP_NAMES.index(shop)] += 1
    out["day"][0] = public["day"]
    out["hour"][0] = public["hour"]
    return out


def load(path):
    return json.loads(Path(path).read_text())


def version(replay):
    return replay.get("module_version")


def frames(replay, seat=0, turns_per_day=24):
    out = []
    steps = replay["steps"]
    final = steps[-1][0]["observation"]["farms"][seat]["money"]
    for index in range(len(steps) - 1):
        records = steps[index]
        if len(records) < PLAYERS:
            break
        action = steps[index + 1][seat]["action"]
        if not action:
            continue
        observation = arrays(records, turns_per_day)
        live = int(observation["hands"][0, seat]) + 1
        out.append(
            {
                "obs": observation,
                "action": action,
                "live": live,
                "value": float(final - observation["money"][0, seat]),
                "pos": [
                    (int(observation["pos"][0, seat, unit, 0]), int(observation["pos"][0, seat, unit, 1]))
                    for unit in range(live)
                ],
                "commands": [action.get("farmer") or ["PASS"], *(action.get("hands") or [])],
            }
        )
    return out


def frames_for(replay, seat, turns_per_day=24):
    out = frames(replay, seat, turns_per_day)
    if seat == 0:
        return out
    for frame in out:
        frame["obs"] = {
            name: np.roll(leaf, -seat, axis=1) if leaf.ndim > 1 and leaf.shape[1] == PLAYERS else leaf
            for name, leaf in frame["obs"].items()
        }
    return out


def teams(replay):
    names = (replay.get("info") or {}).get("TeamNames") or []
    return [names[player] if player < len(names) else f"seat{player}" for player in range(PLAYERS)]


def job_of(command):
    if not command:
        return None
    name = command[0]
    if name == "PASS" or name in MOVES:
        return None
    op = slot_of(UNIT_OP_NAMES, name)
    if op is None:
        return None
    if name == "PLANT":
        arg = slot_of(CROP_NAMES, command[1] if len(command) > 1 else None)
    elif name in ("PICKUP", "PLACE"):
        arg = slot_of(ITEMS, command[1] if len(command) > 1 else None)
    else:
        arg = 0
    if arg is None:
        return None
    return JOB_INDEX.get((op, arg))


def market_of(action):
    index = np.zeros(A.ORDERS, np.int32)
    quantity = np.zeros(A.ORDERS, np.int32)
    for slot, order in enumerate((action.get("market") or [])[: A.ORDERS]):
        if not order:
            continue
        name = order[0]
        if name in ("HIRE", "BUY_LAND"):
            index[slot] = A.MARKET_INDEX[(name, 0)]
            quantity[slot] = quantity_index(1)
            continue
        item = order[1] if len(order) > 1 else None
        code = slot_of(CROP_NAMES if name == "BUY_SEED" else ITEMS, item)
        if code is None or (name, code) not in A.MARKET_INDEX:
            continue
        index[slot] = A.MARKET_INDEX[(name, code)]
        quantity[slot] = quantity_index(count_of(order, 2))
    return index, quantity


def targets(frames):
    for frame in frames:
        frame["target"] = [None] * frame["live"]

    for index, frame in enumerate(frames):
        for unit in range(frame["live"]):
            command = frame["commands"][unit] if unit < len(frame["commands"]) else None
            job = job_of(command)
            if job is None:
                continue
            x, y = frame["pos"][unit]
            frame["target"][unit] = (y * A.BOARD + x, job)
            back = index - 1
            while back >= 0 and unit < frames[back]["live"]:
                earlier = frames[back]["commands"][unit] if unit < len(frames[back]["commands"]) else None
                if not earlier or earlier[0] not in MOVES:
                    break
                if frames[back]["target"][unit] is not None:
                    break
                frames[back]["target"][unit] = (y * A.BOARD + x, job)
                back -= 1

    for frame in frames:
        for unit in range(frame["live"]):
            if frame["target"][unit] is not None:
                continue
            command = frame["commands"][unit] if unit < len(frame["commands"]) else None
            if command and command[0] == "PASS":
                x, y = frame["pos"][unit]
                frame["target"][unit] = (y * A.BOARD + x, A.JOB_PASS)
    return frames


def label(frames, seat=0):
    observations = {name: [] for name in OBS_KEYS}
    chosen, market, quantity, filled, allowed, value = [], [], [], [], [], []

    if frames:
        stacked = {
            name: jnp.asarray(np.concatenate([frame["obs"][name] for frame in frames]))
            for name in frames[0]["obs"]
        }
        legal_jobs = np.asarray(A.legal_jobs(stacked, seat))
        legal_market = np.asarray(A.legal_market(stacked, seat))

    for index, frame in enumerate(frames):
        legal = legal_jobs[index].reshape(-1)
        picked = np.zeros(A.UNITS, np.int32)
        seen = set()
        slot = 0
        for unit, target in enumerate(frame["target"]):
            if target is None or slot >= A.UNITS:
                continue
            tile, job = target
            if not legal[tile * A.NUM_JOBS + job] and unit < len(frame["pos"]):
                x, y = frame["pos"][unit]
                if tile == y * A.BOARD + x:
                    job = A.JOB_PASS
            flat = tile * A.NUM_JOBS + job
            if flat in seen or not legal[flat]:
                continue
            seen.add(flat)
            picked[slot] = flat
            slot += 1

        orders, amounts = market_of(frame["action"])
        for name in OBS_KEYS:
            observations[name].append(frame["obs"][name][0])
        chosen.append(picked)
        market.append(orders)
        quantity.append(amounts)
        filled.append(slot)
        allowed.append(legal_market[index][orders])
        value.append(frame.get("value"))

    if any(entry is None for entry in value):
        value = None
    return observations, chosen, market, quantity, filled, allowed, value


def stack(parts, lengths=()):
    observations, chosen, market, quantity, filled, allowed, value = parts
    terminated = np.zeros(len(filled), bool)
    if len(lengths):
        terminated[np.cumsum(lengths) - 1] = True
    action = {
        "chosen": np.stack(chosen),
        "market": np.stack(market),
        "qty": np.stack(quantity),
        "filled": np.asarray(filled, np.int32),
        "allowed": np.stack(allowed),
    }
    if value is not None:
        action["value"] = np.asarray(value, np.float32)
    return {name: np.stack(leaf) for name, leaf in observations.items()}, action, terminated


def dataset(paths, seat=None, module_version=None):
    labelled, lengths = [], []
    for path in paths:
        replay = load(path)
        if module_version and version(replay) != module_version:
            continue
        seats = range(PLAYERS) if seat is None else (seat,)
        for side in seats:
            block = targets(frames_for(replay, side))
            labelled.extend(block)
            lengths.append(len(block))
    if not labelled:
        raise ValueError("no replays matched")
    return stack(label(labelled, 0), lengths)


def by_player(paths, module_version=None):
    labelled, lengths = {}, {}
    for path in paths:
        replay = load(path)
        if module_version and version(replay) != module_version:
            continue
        for seat, team in enumerate(teams(replay)):
            block = targets(frames_for(replay, seat))
            labelled.setdefault(team, []).extend(block)
            lengths.setdefault(team, []).append(len(block))
    return {team: stack(label(frames, 0), lengths[team]) for team, frames in labelled.items()}
