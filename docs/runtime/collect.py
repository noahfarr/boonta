import json
import os
import pathlib
from functools import partial

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import lox
import numpy as np

from boonta.podracers import anakin

import pilots
from menu import DATASETS, HYPERPARAMETERS

DATA = pathlib.Path(__file__).resolve().parent / "data"


def train(pilot, track, num_steps, seed=0):
    racer = pilots.anakin(pilot, track, HYPERPARAMETERS[pilot, track])
    podracer = anakin.make(racer.config, racer.algorithm, racer.environment)
    state = podracer.init(jax.random.key(seed))
    updates = num_steps // podracer.batch_size
    chunk = max(1, updates // 10)
    for index in range(0, updates, chunk):
        state, logs = podracer.train(state, jax.random.key(seed + index + 1), chunk)
        returns = np.asarray(logs["episode_statistics/episode_return"])
        print(f"  {pilot} {track} update {index + chunk}: return {np.nanmean(returns):.2f}", flush=True)
    return racer, state


def record(racer, state, size, seed):
    steps = -(-size // racer.config.num_envs)

    @jax.jit
    @lox.spool
    def play(state, key):
        return jax.lax.scan(
            partial(racer.rollout, temperature=1.0), state, jax.random.split(key, steps)
        )

    (_, transitions), logs = play(state, jax.random.key(seed))
    flat = jax.tree.map(
        lambda leaf: np.asarray(leaf).reshape(-1, *leaf.shape[2:])[:size], transitions
    )
    returns = np.asarray(logs["episode_statistics/episode_return"])
    return flat, float(np.nanmean(returns)), int(np.sum(~np.isnan(returns)))


def pack(array):
    return np.packbits(array.astype(np.uint8).reshape(-1), bitorder="little").tobytes()


def binary(array):
    return set(np.unique(array).tolist()) <= {0, 1}


def write(track, flat, meta):
    obs, next_obs = np.asarray(flat.first.obs), np.asarray(flat.second.obs)
    packed = binary(obs) and binary(next_obs) and obs.ndim > 2
    fields = {
        "obs": obs.astype(np.uint8) if packed else obs.astype(np.float32),
        "next_obs": next_obs.astype(np.uint8) if packed else next_obs.astype(np.float32),
        "action": np.asarray(flat.second.action).astype(np.int32),
        "reward": np.asarray(flat.second.reward).astype(np.float32),
        "terminated": np.asarray(flat.second.terminated).astype(np.bool_),
        "truncated": np.asarray(flat.second.truncated).astype(np.bool_),
    }
    header = {"order": list(fields), "fields": {}, **meta}
    chunks, offset = [], 0
    for name, array in fields.items():
        as_bits = packed and name in ("obs", "next_obs")
        data = pack(array) if as_bits else array.tobytes()
        header["fields"][name] = {
            "offset": offset,
            "bytes": len(data),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "packed": as_bits,
        }
        chunks.append(data)
        offset += len(data)
        padding = (-offset) % 8
        chunks.append(b"\0" * padding)
        offset += padding
    DATA.mkdir(exist_ok=True)
    (DATA / f"{track}.bin").write_bytes(b"".join(chunks))
    (DATA / f"{track}.json").write_text(json.dumps(header, indent=1))
    print(f"  wrote {track}: {offset / 1e6:.2f} MB", flush=True)


def collect(track):
    spec = DATASETS[track]
    racer, state = train(spec["pilot"], track, spec["train_steps"])
    flat, episode_return, episodes = record(racer, state, spec["size"], seed=12345)
    meta = {
        "policy": spec["pilot"],
        "train_steps": spec["train_steps"],
        "size": spec["size"],
        "episodes": episodes,
        "episode_return": episode_return,
    }
    write(track, flat, meta)
