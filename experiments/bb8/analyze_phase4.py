"""Phase 4: steps to threshold, AUC, final return, stall rate and wall-clock per arm; IQM with 95% stratified bootstrap
intervals; the gate.

    python experiments/bb8/analyze_phase4.py <phase4 dir> --out experiments/bb8/phase4

<phase4 dir>/<algorithm>/<game>/<arm>/<seed>/ hold metrics.npz and train.log. Definitions (from analyze.py, fixed
before any Phase 4 run finished, plus what the pool needs):
- curve: training return per logged epoch, weighted by episodes: training/episode_return_sum / _count where the run
  logs them (pool), else training/episode_return (Anakin, already episode-weighted). x-axis: environment steps actually
  taken (pool/env_steps_total for pool runs, the logged step otherwise).
- a run is refused if it stops short of its budget by more than one largest batch (8192 × 16 steps; the pool can leave
  less than one batch of credit unused) or has no metrics, unless it was stopped for entropy collapse (stalled.json):
  then its curve is extended flat at its last value to the budget, and it counts as a stall.
- final: mean of the curve over the last 10% of epochs. threshold: 95% of the best fixed arm's mean final (fixed arms:
  the tuned/default fixed arm and the fixed_N grid), per algorithm and game.
- steps to threshold: first step where the trailing mean over 5 epochs reaches the threshold; censored at the budget.
  stall: a run that never reaches it (entropy-stopped runs included; reported separately as "stopped").
- wall-clock to threshold: cumulative seconds over epochs (epoch steps / logged SPS, compilation included) at the
  first epoch that reaches the threshold; censored at the run's total.
- frontier: per algorithm and game, IQM steps to threshold against IQM wall-clock to threshold, one point per arm;
  the Pareto frontier is drawn over arms where most seeds reach the threshold.
- AUC: trapezoidal area under the curve over steps / budget.
- oracle fixed: the fixed_N arm with the lowest IQM steps to threshold (ties broken by the higher IQM AUC), per
  algorithm and game.
- IQM over seeds (trim 25%); intervals: 10,000 bootstrap resamples of seeds within each arm (and within each game for
  the aggregate over games: stratified bootstrap), percentile 95%.
- gate (per algorithm): passes if, on most games, BB-8's steps-to-threshold interval overlaps or is below oracle
  fixed's, and BB-8's IQM steps to threshold is below naive's. Strong: BB-8's interval lies entirely below oracle's.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import trim_mean

BUDGETS = {"ppo": 19_922_944, "pqn": 79_953_920}
SLACK = 8192 * 16


def iqm(values, axis=-1):
    return trim_mean(values, 0.25, axis=axis)


def bootstrap(values, resamples=10_000, seed=0):
    values = np.asarray(values, float)
    draws = np.random.default_rng(seed).integers(0, len(values), (resamples, len(values)))
    stats = iqm(values[draws])
    return [float(iqm(values)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def stratified(groups, resamples=10_000, seed=0):
    """IQM over the union of groups (games), resampling seeds within each group."""
    rng = np.random.default_rng(seed)
    stacked = [np.asarray(g, float) for g in groups]
    stats = []
    for _ in range(resamples):
        stats.append(iqm(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in stacked])))
    return [float(iqm(np.concatenate(stacked))), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def trailing(curve, window=5):
    out = np.full_like(curve, np.nan)
    for i in range(len(curve)):
        chunk = curve[max(0, i - window + 1): i + 1]
        if np.any(np.isfinite(chunk)):
            out[i] = np.nanmean(chunk)
    return out


def wall_clock(run):
    path = run / "train.log"
    text = path.read_text(errors="replace") if path.exists() else ""
    found = re.findall(r"│ time\s+│ (\d+):(\d+):(\d+)", text)
    if not found:
        return float("nan")
    h, m, s = map(int, found[-1])
    return h * 3600.0 + m * 60.0 + s


def load(run):
    data = np.load(run / "metrics.npz")
    n = len(data["steps"])
    get = lambda k: np.nanmean(data[k].reshape(n, -1), -1) if k in data.files else None
    if "training/episode_return_sum" in data.files:
        curve = get("training/episode_return_sum") / get("training/episode_return_count")
        steps = get("pool/env_steps_total")
        steps[0] = 0.0
    else:
        curve = get("training/episode_return")
        steps = data["steps"].astype(float)
    sps = get("monitor/SPS")
    seconds = np.zeros_like(steps)
    if sps is not None:
        seconds[1:] = np.cumsum(np.diff(steps) / np.maximum(sps[1:], 1e-9))
    extras = {k: get(k) for k in ("pool/bucket", "pool/multiplier", "pool/kl", "pool/noise_scale_envs",
                                  "pool/noise_scale_critic_envs", "pool/step_target", "pool/update_norm") if k in data.files}
    extras["seconds"] = seconds
    return steps, curve, extras


def main(root, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = defaultdict(list)
    problems = []
    for run in sorted(root.glob("*/*/*/*")):
        algorithm, game, arm, seed = run.parts[-4:]
        budget = BUDGETS[algorithm]
        if not (run / "metrics.npz").exists():
            problems.append(f"{run}: no metrics.npz")
            continue
        steps, curve, extras = load(run)
        stopped = (run / "stalled.json").exists()
        if steps[-1] < budget - SLACK and not stopped:
            problems.append(f"{run}: stops at {steps[-1]:,.0f} of {budget:,}")
            continue
        if stopped and steps[-1] < budget:
            # Extend flat at the last value on the epoch grid of the planned run.
            grid = np.linspace(0, budget, int(round(len(steps) * budget / max(steps[-1], 1))))
            last = curve[np.flatnonzero(np.isfinite(curve))[-1]] if np.isfinite(curve).any() else np.nan
            curve = np.concatenate([curve, np.full(max(0, len(grid) - len(steps)), last)])
            steps = np.concatenate([steps, grid[len(steps):]])
            extras["seconds"] = np.concatenate([extras["seconds"], np.full(len(steps) - len(extras["seconds"]), extras["seconds"][-1])])
        runs[(algorithm, game, arm)].append({"seed": int(seed), "run": str(run), "steps": steps, "curve": curve,
                                             "extras": extras, "wall": wall_clock(run), "stopped": stopped})
    if problems:
        (out / "refused.txt").write_text("\n".join(problems) + "\n")
        print(f"refused {len(problems)} runs, see refused.txt")

    rows, summary = [], {}
    for algorithm in sorted({k[0] for k in runs}):
        games = sorted({k[1] for k in runs if k[0] == algorithm})
        summary[algorithm] = {"games": {}}
        for game in games:
            arms = {k[2]: v for k, v in runs.items() if k[0] == algorithm and k[1] == game}
            final_of = lambda r: float(np.nanmean(r["curve"][-max(1, round(0.1 * (len(r["curve"]) - 1))):]))
            fixed = [a for a in arms if a.startswith("fixed_") or a in ("tuned_fixed", "default_fixed")]
            best_fixed = max(fixed, key=lambda a: np.mean([final_of(r) for r in arms[a]]))
            threshold = 0.95 * float(np.mean([final_of(r) for r in arms[best_fixed]]))
            budget = BUDGETS[algorithm]
            per_arm = {}
            for arm, rs in sorted(arms.items()):
                for r in rs:
                    smooth = trailing(r["curve"])
                    hit = np.flatnonzero(np.nan_to_num(smooth, nan=-np.inf) >= threshold)
                    finite = np.isfinite(r["curve"])
                    row = {"algorithm": algorithm, "game": game, "arm": arm, "seed": r["seed"],
                           "final": final_of(r),
                           "auc": float(np.trapezoid(r["curve"][finite], r["steps"][finite]) / budget),
                           "steps_to_threshold": float(r["steps"][hit[0]]) if len(hit) else float(budget),
                           "wall_to_threshold_s": float(r["extras"]["seconds"][hit[0]]) if len(hit) else float(r["extras"]["seconds"][-1]),
                           "reached": bool(len(hit)), "stopped": r["stopped"], "wall_clock_s": r["wall"],
                           "mean_bucket": float(np.nanmean(r["extras"]["pool/bucket"])) if "pool/bucket" in r["extras"] else float("nan")}
                    rows.append(row)
                arm_rows = [row for row in rows if (row["algorithm"], row["game"], row["arm"]) == (algorithm, game, arm)]
                entry = {"seeds": len(arm_rows), "stalls": sum(not row["reached"] for row in arm_rows),
                         "stopped": sum(row["stopped"] for row in arm_rows)}
                for name in ("steps_to_threshold", "wall_to_threshold_s", "auc", "final", "wall_clock_s"):
                    values = [row[name] for row in arm_rows if np.isfinite(row[name])]
                    entry[name] = bootstrap(values) if values else None
                per_arm[arm] = entry
            grid = [a for a in per_arm if a.startswith("fixed_")]
            # Lowest IQM steps to threshold; ties (e.g. every arm censored at the budget) broken by the higher IQM AUC.
            oracle = min(grid, key=lambda a: (per_arm[a]["steps_to_threshold"][0], -per_arm[a]["auc"][0])) if grid else None
            summary[algorithm]["games"][game] = {"threshold": threshold, "best_fixed_by_final": best_fixed,
                                                 "oracle_fixed": oracle, "arms": per_arm}

        # Gate.
        verdicts = {}
        for kappa in ("bb8_k0.25", "bb8_k1"):
            overlaps, beats_oracle, beats_naive, games_seen = 0, 0, 0, 0
            for game, g in summary[algorithm]["games"].items():
                arms = g["arms"]
                if kappa not in arms or g["oracle_fixed"] is None:
                    continue
                games_seen += 1
                b, o = arms[kappa]["steps_to_threshold"], arms[g["oracle_fixed"]]["steps_to_threshold"]
                overlaps += b[1] <= o[2]
                beats_oracle += b[2] < o[1]
                if "naive" in arms:
                    beats_naive += b[0] < arms["naive"]["steps_to_threshold"][0]
            if games_seen:
                verdicts[kappa] = {"games": games_seen, "matches_or_beats_oracle": overlaps, "beats_oracle_outright": beats_oracle,
                                   "beats_naive": beats_naive,
                                   "passes": overlaps > games_seen / 2 and beats_naive > games_seen / 2,
                                   "strong": beats_oracle > games_seen / 2}
        summary[algorithm]["gate"] = verdicts

        # Aggregate over games: steps to threshold / budget, stratified bootstrap.
        aggregate = {}
        for arm in sorted({row["arm"] for row in rows if row["algorithm"] == algorithm}):
            groups = [[row["steps_to_threshold"] / BUDGETS[algorithm] for row in rows
                       if row["algorithm"] == algorithm and row["game"] == game and row["arm"] == arm] for game in games]
            if all(groups):
                aggregate[arm] = stratified(groups)
        oracle_groups = [[row["steps_to_threshold"] / BUDGETS[algorithm] for row in rows if row["algorithm"] == algorithm
                          and row["game"] == game and row["arm"] == summary[algorithm]["games"][game]["oracle_fixed"]]
                         for game in games]
        if all(oracle_groups):
            aggregate["oracle_fixed (per game)"] = stratified(oracle_groups)
        summary[algorithm]["aggregate_steps_to_threshold_over_budget"] = aggregate

    with open(out / "results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    fmt = lambda t, s=1.0, d=2: "n/a" if t is None else f"{t[0] / s:.{d}f} [{t[1] / s:.{d}f}, {t[2] / s:.{d}f}]"
    lines = []
    for algorithm, a in summary.items():
        for game, g in a["games"].items():
            lines += [f"### {algorithm.upper()} {game}: threshold {g['threshold']:.2f} (95% of {g['best_fixed_by_final']}); "
                      f"oracle fixed = {g['oracle_fixed']}", "",
                      "| Arm | Seeds | Stalls (stopped) | Steps to threshold (M) | Wall-clock to threshold (min) | AUC | Final | Wall-clock (min) |",
                      "|---|---|---|---|---|---|---|---|"]
            for arm, e in g["arms"].items():
                lines.append(f"| {arm} | {e['seeds']} | {e['stalls']} ({e['stopped']}) | {fmt(e['steps_to_threshold'], 1e6)} | "
                             f"{fmt(e['wall_to_threshold_s'], 60, 1)} | {fmt(e['auc'])} | "
                             f"{fmt(e['final'])} | {fmt(e['wall_clock_s'], 60, 1)} |")
            lines.append("")
        lines += [f"### {algorithm.upper()} aggregate over games: steps to threshold / budget (IQM, stratified bootstrap)", "",
                  "| Arm | IQM [95% CI] |", "|---|---|"]
        lines += [f"| {arm} | {fmt(v, 1.0, 3)} |" for arm, v in a["aggregate_steps_to_threshold_over_budget"].items()]
        lines += ["", f"Gate ({algorithm.upper()}): " + json.dumps(a["gate"]), ""]
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    for algorithm in summary:
        games = list(summary[algorithm]["games"])
        figure, axes = plt.subplots(2, len(games), figsize=(5.5 * len(games), 8), squeeze=False)
        for column, game in enumerate(games):
            g = summary[algorithm]["games"][game]
            show = [a for a in g["arms"] if not a.startswith("fixed_") or a == g["oracle_fixed"]]
            for arm in show:
                rs = runs[(algorithm, game, arm)]
                length = min(len(r["curve"]) for r in rs)
                stack = np.stack([trailing(r["curve"][:length]) for r in rs])
                steps = np.mean([r["steps"][:length] for r in rs], axis=0)
                axes[0, column].plot(steps / 1e6, iqm(np.nan_to_num(stack, nan=0.0), axis=0), label=arm, lw=1.2)
                if "pool/bucket" in rs[0]["extras"]:
                    short = min(len(r["extras"]["pool/bucket"]) for r in rs)
                    buckets = np.stack([r["extras"]["pool/bucket"][:short] for r in rs])
                    steps = np.mean([r["steps"][:short] for r in rs], axis=0)
                    axes[1, column].plot(steps / 1e6, np.nanmedian(buckets, 0), label=arm, lw=1.2)
            axes[0, column].axhline(g["threshold"], color="grey", ls="--", lw=0.8)
            axes[0, column].set_title(f"{algorithm.upper()} {game}", fontsize=10)
            axes[0, column].set_ylabel("training return (IQM, trailing 5)")
            axes[1, column].set_yscale("log", base=2)
            axes[1, column].set_ylabel("bucket (environments, median over seeds)")
            for axis in axes[:, column]:
                axis.set_xlabel("environment steps (M)")
                axis.grid(alpha=0.3)
                axis.legend(fontsize=6)
        figure.tight_layout()
        figure.savefig(out / f"curves_{algorithm}.png", dpi=120)

        figure, axes = plt.subplots(1, len(games), figsize=(5.5 * len(games), 4.5), squeeze=False)
        for column, game in enumerate(games):
            axis = axes[0, column]
            arms = summary[algorithm]["games"][game]["arms"]
            points = []
            for arm, e in arms.items():
                if e["steps_to_threshold"] is None or e["wall_to_threshold_s"] is None:
                    continue
                x, y = e["wall_to_threshold_s"][0] / 60, e["steps_to_threshold"][0] / 1e6
                points.append((x, y, arm))
                axis.errorbar(x, y, xerr=[[x - e["wall_to_threshold_s"][1] / 60], [e["wall_to_threshold_s"][2] / 60 - x]],
                              yerr=[[y - e["steps_to_threshold"][1] / 1e6], [e["steps_to_threshold"][2] / 1e6 - y]],
                              fmt="o", ms=4, lw=0.6, capsize=2)
                axis.annotate(arm, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
            # Only arms where most seeds reach the threshold: a censored wall-clock (a stalled or stopped run's total
            # time) is not a time to threshold.
            eligible = [p for p in points if arms[p[2]]["stalls"] <= arms[p[2]]["seeds"] / 2]
            frontier = sorted(p for p in eligible if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in eligible))
            if frontier:
                axis.plot([p[0] for p in frontier], [p[1] for p in frontier], "k--", lw=0.8, label="frontier")
            axis.set_xlabel("wall-clock to threshold (min, IQM)")
            axis.set_ylabel("steps to threshold (M, IQM)")
            axis.set_title(f"{algorithm.upper()} {game}", fontsize=10)
            axis.grid(alpha=0.3)
            axis.legend(fontsize=7)
            summary[algorithm]["games"][game]["frontier"] = [p[2] for p in frontier]
        figure.tight_layout()
        figure.savefig(out / f"frontier_{algorithm}.png", dpi=120)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.root, args.out)
