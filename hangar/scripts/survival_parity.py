"""Survival parity: do restored snapshots survive as long as the trajectories they were taken from?

At a checkpoint of a shadow-PPO run (policy pi, frozen):

1. 1024 environments play pi from the true start for BUILD steps through the training wrapper,
   so the archive fills exactly as in training, and PSR's ledger (k = 4) is updated from the
   same rollouts. The archive and the ledger are then frozen and pi plays FOLLOW more steps, so
   every snapshot's original continuation is recorded.
2. Every snapshot is traced to the trajectory that wrote it. The archive overwrites a cell on
   every from-start visit, the lowest-index environment winning a tie, and stamps the write with
   its clock (one tick per step), so the writer is the lowest-index environment in that cell at
   that tick. Its survival after the snapshot is the number of steps to its episode's end.
3. Three sets of slots: the ERD sampler's picks (uniform over eligible cells), the PSR sampler's
   picks (the selector's water-filling), and slots drawn in proportion to from-start visits
   (the reference). Up to NUM_SNAPSHOTS unique slots per set.
4. Each snapshot is restored COPIES times exactly as the archive restores it (take, then the
   refresh function with a fresh key, so new ghost randomness) and pi plays it for up to HORIZON
   steps.

Survival is capped at min(HORIZON, steps left on the original's time limit); restored copies get
the same cap. Per snapshot, the rank statistic u is the share of copies surviving less than the
original (ties count a half): 0.5 under parity, near 1 when restored copies die sooner.

Usage: uv run --no-sync python hangar/scripts/survival_parity.py RUN_DIR --epochs 5 81 [--out DIR]
"""

import argparse
import json
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

NUM_ENVS = 1024
ROLLOUT = 64
BUILD = 25 * ROLLOUT
FOLLOW = 4 * ROLLOUT
HORIZON = 200
TIME_LIMIT = 1000
NUM_SNAPSHOTS = 500
COPIES = 8


def compose():
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    import hangar.resolvers  # noqa: F401

    config_dir = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return hydra_compose(
            "train",
            overrides=[
                "algorithm=ppo",
                "environment=jumanji/pac_man",
                "curriculum=psr",
                f"environment.num_envs={NUM_ENVS}",
                "logger=[]",
                "artisan=[]",
            ],
        )


def run_checkpoint(checkpoint: str, seed: int) -> dict:
    from boonta.curricula.psr import archive_auto_reset_state, restore_policy
    from boonta.environments.wrappers.archive_auto_reset import pick, unwrap
    from boonta.utils import Transition
    from hangar.recipes import register

    cfg = compose()
    components = register[("ppo", "jumanji", "pac_man")](cfg)
    algorithm, environment, pit = components["algorithm"], components["environment"], components["pit"]
    wrapper = unwrap(environment)
    key = jax.random.key(seed)
    init_key, ledger_key, key = jax.random.split(key, 3)
    algorithm_state, env_state, _, timestep = restore_policy(algorithm, environment, checkpoint, init_key)
    ledger = pit.init(ledger_key, types.SimpleNamespace(algorithm_state=algorithm_state))

    @jax.jit
    def rollout(env_state, timestep, key):
        def step(carry, key):
            env_state, timestep = carry
            act_key, env_key = jax.random.split(key)
            _, _, action, aux = algorithm.step(algorithm_state, None, act_key, timestep, 1.0)
            env_state, next_timestep = environment.step(env_key, env_state, action)
            record = (
                next_timestep.info["cell"],
                next_timestep.info["age"],
                next_timestep.terminated,
                next_timestep.truncated,
            )
            return (env_state, next_timestep), (Transition(first=timestep, second=next_timestep, aux=aux, carry=None), record)

        (env_state, timestep), (transitions, record) = jax.lax.scan(
            step, (env_state, timestep), jax.random.split(key, ROLLOUT)
        )
        return env_state, timestep, transitions, record

    update = jax.jit(lambda ledger, key, transitions: pit.update(ledger, key, transitions, algorithm_state))

    first_cells = np.asarray(timestep.info["cell"])
    first_age = np.asarray(timestep.info["age"])
    records = []
    for index in range(BUILD // ROLLOUT):
        rollout_key, update_key, key = jax.random.split(key, 3)
        env_state, timestep, transitions, record = rollout(env_state, timestep, rollout_key)
        ledger = update(ledger, update_key, transitions)
        records.append(jax.device_get(record))
    node = archive_auto_reset_state(env_state)
    archive_state = jax.tree.map(lambda leaf: leaf, node.archive_state)
    frozen_node = node
    for index in range(FOLLOW // ROLLOUT):
        rollout_key, key = jax.random.split(key)
        env_state, timestep, _, record = rollout(env_state, timestep, rollout_key)
        records.append(jax.device_get(record))
    cells = np.concatenate([first_cells[None]] + [r[0] for r in records])  # row 0 is the initial state
    ages = np.concatenate([first_age[None]] + [r[1] for r in records])
    terminated = np.concatenate([np.zeros_like(first_cells[None], bool)] + [np.asarray(r[2], bool) for r in records])
    truncated = np.concatenate([np.zeros_like(first_cells[None], bool)] + [np.asarray(r[3], bool) for r in records])
    ended = terminated | truncated
    total = cells.shape[0]

    # The three samplers.
    select_key, erd_key, visit_key, key = jax.random.split(key, 4)
    eligible = np.asarray(wrapper.archive.eligible(archive_state))
    select = jax.jit(
        lambda ledger, key: pit.selector.select(
            ledger, wrapper, frozen_node, key, algorithm=algorithm, algorithm_state=algorithm_state
        )
    )
    num_placed = pit.selector.num_placed(NUM_ENVS)
    picks = {
        "psr": np.concatenate([np.asarray(select(ledger, k)) for k in jax.random.split(select_key, 8)]),
        "erd": np.concatenate(
            [
                np.asarray(pick(k, jnp.where(jnp.asarray(eligible), 0.0, -jnp.inf), (num_placed,)))
                for k in jax.random.split(erd_key, 8)
            ]
        ),
        "from_start": np.asarray(
            pick(visit_key, jnp.log(jnp.maximum(ledger.reach, 1e-30)) + jnp.where(jnp.asarray(eligible), 0.0, -jnp.inf), (8 * num_placed,))
        ),
    }
    clocks = np.asarray(archive_state.cell_states.time)

    def original(slot):
        row = int(clocks[slot])  # the write at clock c is row c: row 0 is init, row j the state after step j
        writers = np.flatnonzero(cells[row] == slot)
        if row >= total or len(writers) == 0:
            return None
        env = int(writers[0])
        future = np.flatnonzero(ended[row + 1 :, env])
        cap = min(HORIZON, TIME_LIMIT - int(ages[row, env]))
        if len(future) == 0:
            survival, died = total - 1 - row, False
        else:
            survival, died = int(future[0]) + 1, bool(terminated[row + 1 + future[0], env])
        if survival < cap and not died:
            return None  # cut by the time limit before the cap: no outcome to compare
        return {"slot": int(slot), "row": row, "env": env, "cap": cap, "survival": min(survival, cap), "died": died and survival <= cap}

    inner = wrapper._env

    @jax.jit
    def restore(slots, caps, key):
        snapshots = wrapper.archive.take(archive_state, slots)

        def copy(key):
            refresh_key, play_key = jax.random.split(key)
            states = wrapper.refresh_fn(snapshots, refresh_key)

            def step(carry, key):
                states, alive, survival = carry
                act_key, env_key = jax.random.split(key)
                obs = jax.vmap(inner.observe)(states)
                dist, _ = algorithm.network.apply(algorithm_state.params, obs, temperature=1.0)
                action = dist.sample(seed=act_key)
                states, timestep = jax.vmap(inner.step)(jax.random.split(env_key, slots.shape[0]), states, action)
                dies = alive & timestep.terminated
                survival = survival + alive.astype(jnp.int32)
                return (states, alive & ~dies, survival), dies

            (_, alive, survival), _ = jax.lax.scan(
                step,
                (states, jnp.ones(slots.shape, bool), jnp.zeros(slots.shape, jnp.int32)),
                jax.random.split(play_key, HORIZON),
            )
            return jnp.minimum(survival, caps), ~alive & (survival <= caps)

        return jax.vmap(copy)(jax.random.split(key, COPIES))

    rng = np.random.default_rng(seed)
    result = {"checkpoint": checkpoint, "groups": {}}
    for name, slots in picks.items():
        unique = np.unique(slots[eligible[slots]])
        traced = [o for o in (original(s) for s in unique) if o is not None]
        if len(traced) > NUM_SNAPSHOTS:
            traced = [traced[i] for i in sorted(rng.choice(len(traced), NUM_SNAPSHOTS, replace=False))]
        restore_key, key = jax.random.split(key)
        slot_array = jnp.asarray([o["slot"] for o in traced], jnp.int32)
        caps = jnp.asarray([o["cap"] for o in traced], jnp.int32)
        survived, died = jax.device_get(restore(slot_array, caps, restore_key))
        orig = np.asarray([o["survival"] for o in traced])
        below = (survived < orig[None]).mean(0) + 0.5 * (survived == orig[None]).mean(0)
        counts = np.bincount(np.searchsorted(np.unique(slots), slots), minlength=len(np.unique(slots)))
        result["groups"][name] = {
            "picked_draws": int(len(slots)),
            "unique_picked": int(len(unique)),
            "traced": len(traced),
            "top_slot_share": float(counts.max() / counts.sum()),
            "median_original": float(np.median(orig)),
            "median_restored": float(np.median(survived)),
            "mean_original": float(orig.mean()),
            "mean_restored": float(survived.mean()),
            "die_within_30_original": float(np.mean([o["died"] and o["survival"] <= 30 for o in traced])),
            "die_within_30_restored": float(np.mean(died & (survived <= 30))),
            "rank_u_mean": float(below.mean()),
            "rank_u_se": float(below.std(ddof=1) / np.sqrt(len(below))),
            "caps_median": float(np.median(np.asarray(caps))),
            "pairs": [[int(a), float(b)] for a, b in zip(orig, survived.mean(0))],
        }
        g = result["groups"][name]
        print(
            f"  {name:10s} traced {g['traced']:3d}/{g['unique_picked']:3d} unique | median survival original {g['median_original']:5.0f} "
            f"restored {g['median_restored']:5.0f} | die<=30 original {g['die_within_30_original']:.2f} restored "
            f"{g['die_within_30_restored']:.2f} | u {g['rank_u_mean']:.3f} +- {g['rank_u_se']:.3f}",
            flush=True,
        )
    return result


def plot(results: list, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"from_start": "#7a7a7a", "erd": "#2a6fdb", "psr": "#d9822b"}
    figure, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.8), squeeze=False)
    for axis, result in zip(axes[0], results):
        for name, group in result["groups"].items():
            pairs = np.asarray(group["pairs"])
            jitter = np.random.default_rng(0).uniform(-1.5, 1.5, len(pairs))
            axis.scatter(pairs[:, 0] + jitter, pairs[:, 1], s=8, alpha=0.5, color=colors[name], label=f"{name} (u {group['rank_u_mean']:.2f})")
        axis.plot([0, HORIZON], [0, HORIZON], color="#333333", linewidth=1)
        axis.set_xlabel("original survival after the snapshot (steps, capped)")
        axis.set_ylabel(f"mean survival of {COPIES} restored copies")
        axis.set_title(f"epoch {result['epoch']} (step {result['step'] / 1e6:.0f}M)")
        axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--epochs", type=int, nargs="+", default=[5, 81])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run = Path(args.run)
    out = Path(args.out or run / "survival_parity")
    out.mkdir(parents=True, exist_ok=True)
    saved = run / "checkpoints" / "latest"
    steps = sorted(int(p.name) for p in saved.iterdir() if p.name.isdigit())
    results = []
    for epoch in args.epochs:
        step = steps[min(epoch, len(steps)) - 1]
        print(f"checkpoint at epoch {epoch} (step {step})", flush=True)
        result = run_checkpoint(str(saved / str(step)), seed=epoch)
        result.update(epoch=epoch, step=step)
        results.append(result)
        (out / f"epoch_{epoch:03d}.json").write_text(json.dumps(result, indent=1))
    plot(results, out / "survival_parity.png")


if __name__ == "__main__":
    main()
