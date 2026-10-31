"""Does the probe's noise scale depend on the batch it is measured at? From one checkpoint, the fresh-update noise
scale measured on B columns (64 groups of B/64), averaged over many disjoint batches (noise and signal averaged
separately, like the probe's moving average), with the optimizer as trained (gradient clipping per group, then Adam)
and with the clipping removed (Adam alone, same state).

    python experiments/bb8/probe_bias.py <run dir> --step 7864320 --output out.json
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import lox
import numpy as np
import optax
import orbax.checkpoint as ocp

from boonta.algorithms import bb8
from boonta.algorithms.bb8 import BB8Config

import one_update


def main(args):
    cfg, podracer = one_update.build(args.run)
    algorithm = podracer.algorithm
    T, num_envs = int(cfg.podracer.config.num_steps), int(cfg.environment.num_envs)
    names = ("actor", "critic", "total") if hasattr(algorithm.cfg, "clip_coefficient") else ("td",)
    template = podracer.init(jax.random.key(0))
    path = os.path.abspath(Path(args.run) / "checkpoints" / "latest" / str(args.step) / "algorithm_state")
    state = one_update.restore(path, template.algorithm_state)
    rollout = jax.jit(lambda s, keys: jax.lax.scan(lambda c, k: podracer.rollout(c, k, 1.0), s, keys))
    advance = jax.jit(lambda s, keys: jax.lax.scan(lambda c, k: (podracer.rollout(c, k, 1.0)[0], None), s, keys)[0])
    pool = advance(template.replace(algorithm_state=state), jax.random.split(jax.random.key(1), args.warmup))
    clipped = algorithm.optimizer
    adam_only = optax.adam(float(cfg.optimizer.lr))
    adam_state = state.optimizer_state[1]  # chain(clip_by_global_norm, adam): (EmptyState, adam state)

    def probe_with(optimizer, optimizer_state):
        @jax.jit
        def run(algorithm_state, batch):
            probed = dataclasses.replace(
                algorithm, optimizer=optimizer,
                cfg=algorithm.cfg.replace(num_minibatches=1, update_epochs=1,
                                          bb8=BB8Config(noise_scale_probe=True, num_groups=64)))
            algorithm_state = algorithm_state.replace(optimizer_state=optimizer_state, bb8=bb8.init(names))
            _, logs = lox.spool(probed.update)(algorithm_state, jax.random.key(0), batch)
            return {k: v for k, v in logs.items() if k.startswith("bb8/") and k.endswith(("/noise", "/signal"))}
        return run

    variants = {"clipped": probe_with(clipped, state.optimizer_state), "unclipped": probe_with(adam_only, adam_state)}
    sums = {}
    for r, key in enumerate(jax.random.split(jax.random.key(2), args.rollouts)):
        pool, batch = rollout(pool, jax.random.split(key, T))
        for size in args.sizes:
            for start in range(0, num_envs, size):
                sub = jax.tree.map(lambda leaf: leaf[:, start:start + size], batch)
                for variant, run in variants.items():
                    for k, v in run(state, sub).items():
                        sums.setdefault((variant, size, k), []).append(float(np.mean(v)))
        print(f"rollout {r} done", flush=True)
    result = {}
    for variant in variants:
        for size in args.sizes:
            for name in names:
                for space in ("gradient", "fresh", "difference", "update"):
                    noise = np.mean(sums[(variant, size, f"bb8/{name}/{space}/noise")])
                    signal = np.mean(sums[(variant, size, f"bb8/{name}/{space}/signal")])
                    count = len(sums[(variant, size, f"bb8/{name}/{space}/noise")])
                    result[f"{variant}/{size}/{name}/{space}"] = {"noise_scale_envs": noise / signal, "batches": count}
    Path(args.output).write_text(json.dumps(result, indent=1))
    for name in names:
        for space in ("gradient", "fresh", "difference"):
            print(name, space, "  ".join(f"{v} B={s}: {result[f'{v}/{s}/{name}/{space}']['noise_scale_envs']:.0f}"
                                         for v in variants for s in args.sizes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 8192])
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    main(parser.parse_args())
