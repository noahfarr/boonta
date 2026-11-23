import argparse
import sys
import time
from pathlib import Path

import jax
import numpy as np

SHARED = {
    "UPDATE_EPOCHS": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "DEBUG": False,
}
SETTINGS = {
    "minatar": {
        **SHARED,
        "LR": 5e-3,
        "NUM_ENVS": 64,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e7,
        "NUM_MINIBATCHES": 8,
        "ENT_COEF": 0.01,
        "ACTIVATION": "relu",
        "ENV_NAME": "SpaceInvaders-MinAtar",
        "ANNEAL_LR": True,
    },
    "brax": {
        **SHARED,
        "LR": 3e-4,
        "NUM_ENVS": 2048,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 5e7,
        "NUM_MINIBATCHES": 32,
        "ENT_COEF": 0.0,
        "ACTIVATION": "tanh",
        "ENV_NAME": "hopper",
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
    },
}
EPOCHS = 10


def episode_returns(metrics):
    returns = np.asarray(metrics["returned_episode_returns"], dtype=float)
    finished = np.asarray(metrics["returned_episode"], dtype=bool)
    per_update = np.where(finished, returns, 0.0).reshape(len(returns), -1).sum(axis=1)
    counts = finished.reshape(len(finished), -1).sum(axis=1)
    return per_update, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=sorted(SETTINGS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--total-timesteps", type=float)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--purejaxrl", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.purejaxrl / "purejaxrl"))
    import ppo
    import ppo_continuous_action

    config = dict(SETTINGS[args.suite])
    if args.num_envs:
        config["NUM_ENVS"] = args.num_envs
    if args.total_timesteps:
        config["TOTAL_TIMESTEPS"] = args.total_timesteps
    make_train = (ppo if args.suite == "minatar" else ppo_continuous_action).make_train
    train = jax.jit(make_train(config))
    key = jax.random.PRNGKey(args.seed)

    start = time.perf_counter()
    compiled = train.lower(key).compile()
    compile_time = time.perf_counter() - start
    start = time.perf_counter()
    out = jax.block_until_ready(compiled(key))
    seconds = time.perf_counter() - start

    batch = config["NUM_ENVS"] * config["NUM_STEPS"]
    updates = config["NUM_UPDATES"]
    sps = updates * batch / seconds
    totals, counts = episode_returns(jax.device_get(out["metrics"]))
    bounds = np.linspace(0, updates, EPOCHS + 1).astype(int)
    epochs = [
        totals[low:high].sum() / max(counts[low:high].sum(), 1) if counts[low:high].sum() else np.nan
        for low, high in zip(bounds[:-1], bounds[1:])
    ]
    steps = bounds * batch
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out / "metrics.npz",
        steps=steps,
        **{
            "training/episode_return": np.array([np.nan, *epochs]),
            "monitor/SPS": np.full(len(steps), sps),
            "monitor/compilation_time": np.full(len(steps), compile_time),
        },
    )
    print(f"{args.suite} seed {args.seed} envs {config['NUM_ENVS']} steps {updates * batch} sps {sps:.0f} compile {compile_time:.1f}s final return {epochs[-1]:.2f}")


if __name__ == "__main__":
    main()
