"""Round 2, Part 1 (and Part 2, same layout): per arm, steps to threshold among non-stalled seeds, stall rate, final
return and entropy, KL per update and per environment step, gradient steps per environment step.

    python experiments/bb8/analyze_part1.py <dir> --out experiments/bb8/r2p1 [--thresholds thresholds.json]

<dir>/<game>/<arm>/<seed>/ (PQN references under pqn_<game>/<arm>/<seed>). Definitions:
- curve: training return per epoch, episode-weighted (sum/count for pool runs); x-axis: environment steps taken.
- threshold: fixed per game at 0.95 × the mean final return of the reference arm (A_8192 for PPO, `default` for PQN),
  final = mean of the last 10% of epochs; written to thresholds.json the first time and read from it afterwards.
- stall: never reaches the threshold (trailing mean over 5 epochs), or stopped for entropy collapse.
- steps to threshold among non-stalled seeds: IQM with a 95% bootstrap interval over those seeds; all seeds (censored at the
  budget) reported beside it.
- final entropy: `actor/entropy` of the last logged epoch. KL per update: mean `bb8/update_kl` over epochs. KL per
  environment step: KL per update / (B · T). Gradient steps per environment step: epochs × minibatches / (B · T),
  from the resolved config.
- runs that stop short of their budget (by more than one batch) without an entropy stop are refused.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import trim_mean

T = 16


def iqm(values, axis=-1):
    return trim_mean(values, 0.25, axis=axis)


def bootstrap(values, resamples=10_000, seed=0):
    values = np.asarray(values, float)
    if len(values) == 0:
        return None
    draws = np.random.default_rng(seed).integers(0, len(values), (resamples, len(values)))
    return [float(iqm(values)), float(np.percentile(iqm(values[draws]), 2.5)), float(np.percentile(iqm(values[draws]), 97.5))]


def trailing(curve, window=5):
    out = np.full_like(curve, np.nan)
    for i in range(len(curve)):
        chunk = curve[max(0, i - window + 1): i + 1]
        if np.any(np.isfinite(chunk)):
            out[i] = np.nanmean(chunk)
    return out


def load(run):
    data = np.load(run / "metrics.npz")
    n = len(data["steps"])
    get = lambda k: np.nanmean(data[k].reshape(n, -1), -1) if k in data.files else None
    if "training/episode_return_sum" in data.files:
        curve = get("training/episode_return_sum") / get("training/episode_return_count")
        steps = get("pool/env_steps_total")
        steps[0] = 0.0
    else:
        curve, steps = get("training/episode_return"), data["steps"].astype(float)
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    algorithm = cfg.algorithm
    if "podracer" in cfg and "fixed_bucket" in cfg.podracer.config:
        envs = int(cfg.podracer.config.fixed_bucket)
        minibatches = max(1, envs // int(cfg.podracer.config.minibatch_envs))
    else:
        envs, minibatches = int(cfg.environment.num_envs), int(algorithm.num_minibatches)
    return {"steps": steps, "curve": curve, "entropy": get("actor/entropy"), "kl": get("bb8/update_kl"),
            "budget": int(cfg.num_steps), "envs": envs, "minibatches": minibatches,
            "epochs": int(algorithm.update_epochs), "lr": float(cfg.optimizer.lr), "stopped": (run / "stalled.json").exists()}


def final_of(curve):
    return float(np.nanmean(curve[-max(1, round(0.1 * (len(curve) - 1))):]))


def main(args):
    root, out = Path(args.directory), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs, refused = {}, []
    for run in sorted(root.glob("*/*/*")):
        if not (run / "metrics.npz").exists() or not (run / ".hydra" / "config.yaml").exists():
            refused.append(f"{run}: no metrics or config")
            continue
        r = load(run)
        if r["steps"][-1] < r["budget"] - r["envs"] * T and not r["stopped"]:
            refused.append(f"{run}: stops at {r['steps'][-1]:,.0f} of {r['budget']:,}")
            continue
        runs.setdefault((run.parts[-3], run.parts[-2]), []).append(r)
    thresholds_path = Path(args.thresholds)
    thresholds = json.loads(thresholds_path.read_text()) if thresholds_path.exists() else {}
    for game in sorted({g for g, _ in runs}):
        reference = "default" if game.startswith("pqn_") else "A_8192"
        if game not in thresholds and (game, reference) in runs:
            finals = [final_of(r["curve"]) for r in runs[(game, reference)]]
            thresholds[game] = {"threshold": 0.95 * float(np.mean(finals)), "reference": reference, "seeds": len(finals),
                                "reference_final_mean": float(np.mean(finals))}
    if args.write_thresholds:
        thresholds_path.write_text(json.dumps(thresholds, indent=1))
    rows, table = [], []
    for (game, arm), rs in sorted(runs.items()):
        if game not in thresholds:
            continue
        threshold = thresholds[game]["threshold"]
        per = []
        for r in rs:
            hit = np.flatnonzero(np.nan_to_num(trailing(r["curve"]), nan=-np.inf) >= threshold)
            reached = bool(len(hit)) and not (r["stopped"] and not len(hit))
            batch = r["envs"] * T
            kl = float(np.nanmean(r["kl"])) if r["kl"] is not None else float("nan")
            entropy = r["entropy"][np.isfinite(r["entropy"])] if r["entropy"] is not None else np.array([np.nan])
            per.append({"game": game, "arm": arm, "reached": reached, "stopped": r["stopped"],
                        "steps_to_threshold": float(r["steps"][hit[0]]) if len(hit) else float(r["budget"]),
                        "final": final_of(r["curve"]), "final_entropy": float(entropy[-1]) if len(entropy) else float("nan"),
                        "kl_per_update": kl, "kl_per_env_step": kl / batch,
                        "grad_steps_per_env_step": r["epochs"] * r["minibatches"] / batch, "envs": r["envs"],
                        "minibatches": r["minibatches"], "lr": r["lr"]})
        rows += per
        stalls = [p for p in per if not p["reached"] or p["stopped"]]
        good = [p["steps_to_threshold"] for p in per if p["reached"] and not p["stopped"]]
        entry = {"game": game, "arm": arm, "seeds": len(per), "stalls": len(stalls), "stopped": sum(p["stopped"] for p in per),
                 "steps_non_stalled": bootstrap(good), "steps_all": bootstrap([p["steps_to_threshold"] for p in per]),
                 "final": bootstrap([p["final"] for p in per]),
                 "final_entropy": float(np.median([p["final_entropy"] for p in per])),
                 "kl_per_update": float(np.median([p["kl_per_update"] for p in per])),
                 "kl_per_env_step": float(np.median([p["kl_per_env_step"] for p in per])),
                 "grad_steps_per_env_step": per[0]["grad_steps_per_env_step"], "lr": per[0]["lr"]}
        table.append(entry)
    with open(out / "results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps({"thresholds": thresholds, "arms": table}, indent=1))
    fmt = lambda t, s=1e6: "n/a" if t is None else f"{t[0] / s:.2f} [{t[1] / s:.2f}, {t[2] / s:.2f}]"
    lines = [f"Thresholds: " + ", ".join(f"{g} {v['threshold']:.2f} (0.95 × {v['reference']} mean final {v['reference_final_mean']:.2f}, "
                                         f"{v['seeds']} seeds)" for g, v in thresholds.items()), "",
             "| Game | Arm | Seeds | Stalls (entropy-stopped) | Steps to threshold, non-stalled (M) | All seeds (M) | Final return | "
             "Final entropy | KL / update | KL / env step | Grad steps / env step | lr |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in table:
        lines.append(f"| {e['game']} | {e['arm']} | {e['seeds']} | {e['stalls']} ({e['stopped']}) | {fmt(e['steps_non_stalled'])} | "
                     f"{fmt(e['steps_all'])} | {fmt(e['final'], 1)} | {e['final_entropy']:.3f} | {e['kl_per_update']:.4f} | "
                     f"{e['kl_per_env_step']:.2e} | {e['grad_steps_per_env_step']:.2e} | {e['lr']:.2e} |")
    if refused:
        lines += ["", "Refused: " + "; ".join(refused)]
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--thresholds", default="experiments/bb8/thresholds.json")
    parser.add_argument("--write-thresholds", action="store_true", help="fix the thresholds (once, from Part 1)")
    main(parser.parse_args())
