"""Trains a few updates from a hangar config and saves every leaf of the final algorithm state, for bitwise comparison.

Run once with this tree and once with `main` on PYTHONPATH (both resolve `boonta` and `hangar` to the tree in front):

    PYTHONPATH=<main worktree> python experiments/bb8/bitwise_check.py save main.npz algorithm=ppo
    python experiments/bb8/bitwise_check.py save bb8.npz algorithm=ppo
    python experiments/bb8/bitwise_check.py compare main.npz bb8.npz
"""

import sys
from pathlib import Path

import numpy as np


def save(path, overrides, num_updates=12):
    import jax
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    import boonta
    import hangar
    from hangar import recipes, resolvers  # noqa: F401

    config = Path(hangar.__file__).parent / "config"
    print("boonta from", Path(boonta.__file__).parent, "config", config, flush=True)
    with initialize_config_dir(config_dir=str(config), version_base=None):
        cfg = compose("train", overrides=["logger=[]", "artisan=[]", "seed=0", *overrides], return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
    podracer = recipes.make(cfg)
    state = podracer.init(jax.random.key(0))
    state, logs = podracer.train(state, jax.random.key(1), num_updates)
    algorithm_state = state.algorithm_state
    leaves = jax.tree_util.tree_leaves_with_path(
        {"params": algorithm_state.params, "optimizer_state": algorithm_state.optimizer_state, "step": algorithm_state.step}
    )
    arrays = {jax.tree_util.keystr(name): np.asarray(jax.device_get(leaf)) for name, leaf in leaves}
    np.savez(path, **arrays)
    probe = sorted(name for name in logs if name.startswith("bb8/"))
    print(f"saved {len(arrays)} leaves to {path}; {len(probe)} bb8 logs", flush=True)
    for name in probe:
        if name.endswith("noise_scale_envs"):
            print(f"  {name}: {np.nanmean(np.asarray(logs[name])):.4g}")


def compare(a, b):
    a, b = np.load(a), np.load(b)
    assert sorted(a.files) == sorted(b.files), set(a.files) ^ set(b.files)
    different = [name for name in a.files if not np.array_equal(a[name], b[name])]
    print(f"{len(a.files)} leaves, {len(different)} differ")
    for name in different:
        print("  ", name, float(np.max(np.abs(a[name].astype(float) - b[name].astype(float)))))
    return 1 if different else 0


if __name__ == "__main__":
    if sys.argv[1] == "save":
        save(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(compare(sys.argv[2], sys.argv[3]))
