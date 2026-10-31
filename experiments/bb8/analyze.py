"""Ported from exp/psr-breakout (commit 5a0ce3d1, experiments/psr_breakout/analyze.py) and adapted to main.

Aggregates BB-8 runs into per-run metrics, interquartile means with bootstrap intervals, and learning curves.

    python experiments/bb8/analyze.py <manifest.json> --out experiments/bb8/<phase>

The manifest maps each arm to a list of run directories (each holding metrics.npz and train.log) and names the
reference arm whose final performance sets the threshold:

    {"reference": "default", "budget": 19988480, "arms": {"default": ["<run dir>", ...], "bb8": [...]}}

A run is rejected if its last logged step is below `budget` (SLURM and Determined can report a crashed run as finished).

Definitions (from the PSR study, unchanged):
- curve: `training/episode_return` per logged epoch, mean over devices. Epochs without a finished rho episode are NaN.
- smoothed: trailing mean over the last `window` logged epochs (no look-ahead), ignoring NaN.
- final: mean of the raw curve over the last 10% of epochs.
- threshold: `fraction` (0.95) of the mean over reference seeds of their final.
- steps to threshold: the first logged step at which the smoothed curve reaches the threshold. A run that never
  reaches it is censored at the budget; the share that reached it is reported beside it.
- AUC: trapezoidal area under the raw curve over steps, divided by the budget, so it is the mean return over training.
- greedy return: mean `evaluation/episode_return` of the final checkpoint over 1024 episodes from rho (evaluate_final.sh).
- wall-clock: the training `time` the run printed, which excludes setup but includes compilation.
- IQM: mean of the middle 50% of seeds (scipy trim_mean 0.25). Intervals: 95% percentile bootstrap over seeds,
  resampled within each arm (the stratified bootstrap with one task), 10,000 resamples.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import trim_mean

METRIC = "training/episode_return"


def iqm(values, axis=-1):
    return trim_mean(values, 0.25, axis=axis)


def bootstrap(values, statistic=iqm, resamples=10_000, seed=0):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), (resamples, len(values)))
    stats = statistic(values[draws], axis=-1)
    return float(statistic(values)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_bootstrap(arm, reference, combine, resamples=10_000, seed=0):
    """Interval of combine(IQM(arm), IQM(reference)), resampling each arm's seeds independently."""
    arm, reference = np.asarray(arm, float), np.asarray(reference, float)
    rng = np.random.default_rng(seed)
    a = iqm(arm[rng.integers(0, len(arm), (resamples, len(arm)))])
    r = iqm(reference[rng.integers(0, len(reference), (resamples, len(reference)))])
    stats = combine(a, r)
    return float(combine(iqm(arm), iqm(reference))), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def trailing(curve, window):
    out = np.full_like(curve, np.nan)
    for index in range(len(curve)):
        chunk = curve[max(0, index - window + 1) : index + 1]
        if np.any(np.isfinite(chunk)):
            out[index] = np.nanmean(chunk)
    return out


def wall_clock(run: Path) -> float:
    """The `time` row of the run's final table, from the train.log the job file writes into the run directory."""
    text = (run / "train.log").read_text(errors="replace") if (run / "train.log").exists() else ""
    match = re.findall(r"│ time\s+│ (\d+):(\d+):(\d+)", text)
    if not match:
        return float("nan")
    hours, minutes, seconds = map(int, match[-1])
    return hours * 3600.0 + minutes * 60.0 + seconds


def load(run: Path, metric=METRIC):
    data = np.load(run / "metrics.npz")
    steps = data["steps"].astype(float)
    curve = np.nanmean(data[metric].reshape(len(steps), -1), axis=-1) if metric in data.files else np.full(len(steps), np.nan)
    return steps, curve


def greedy(run: Path) -> float:
    path = run / "evaluation" / "metrics.npz"
    if not path.exists():
        return float("nan")
    return float(np.nanmean(np.load(path)["evaluation/episode_return"]))


def per_run(run, steps, curve, threshold, window):
    finite = np.isfinite(curve)
    budget = steps[-1]
    tail = max(1, int(round(0.1 * (len(curve) - 1))))
    smoothed = trailing(curve, window)
    reached = np.flatnonzero(np.nan_to_num(smoothed, nan=-np.inf) >= threshold)
    area = np.trapezoid(curve[finite], steps[finite]) / budget if finite.sum() > 1 else np.nan
    return {
        "run": str(run),
        "budget": budget,
        "final": float(np.nanmean(curve[-tail:])),
        "auc": float(area),
        "steps_to_threshold": float(steps[reached[0]]) if len(reached) else float(budget),
        "reached": bool(len(reached)),
        "greedy_return": greedy(run),
        "wall_clock_s": wall_clock(run),
    }


def main(manifest, out, fraction, window, metric):
    manifest = json.loads(Path(manifest).read_text())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reference = manifest["reference"]
    curves = {arm: [(Path(run), *load(Path(run), metric)) for run in runs] for arm, runs in manifest["arms"].items()}
    budget = manifest.get("budget") or max(steps[-1] for runs in curves.values() for _, steps, _ in runs)
    short = [str(run) for runs in curves.values() for run, steps, _ in runs if steps[-1] < budget]
    if short:
        raise SystemExit(f"runs that stop before {budget:,.0f} steps (crashed or still running): {short}")

    def final_of(curve):
        tail = max(1, int(round(0.1 * (len(curve) - 1))))
        return np.nanmean(curve[-tail:])

    threshold = fraction * float(np.mean([final_of(curve) for _, _, curve in curves[reference]]))
    rows = []
    for arm, runs in curves.items():
        for seed, (run, steps, curve) in enumerate(runs):
            rows.append({"arm": arm, "seed": seed, **per_run(run, steps, curve, threshold, window)})

    with open(out / "results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in curves}
    summary = {"threshold": threshold, "fraction": fraction, "window": window, "metric": metric, "arms": {}}
    reference_rows = by_arm[reference]
    for arm, arm_rows in by_arm.items():
        entry = {"seeds": len(arm_rows), "reached": sum(row["reached"] for row in arm_rows)}
        for name in ("steps_to_threshold", "auc", "final", "greedy_return", "wall_clock_s"):
            values = [row[name] for row in arm_rows]
            entry[name] = bootstrap([v for v in values if np.isfinite(v)]) if np.isfinite(values).any() else None
        reference_steps = [row["steps_to_threshold"] for row in reference_rows]
        arm_steps = [row["steps_to_threshold"] for row in arm_rows]
        entry["speedup_vs_reference"] = paired_bootstrap(arm_steps, reference_steps, lambda a, r: r / a)
        for name in ("auc", "final", "greedy_return"):
            entry[f"{name}_minus_reference"] = paired_bootstrap(
                [row[name] for row in arm_rows], [row[name] for row in reference_rows], lambda a, r: a - r
            )
        summary["arms"][arm] = entry
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    def fmt(triple, scale=1.0, digits=2):
        if triple is None:
            return "n/a"
        value, low, high = (x / scale for x in triple)
        return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"

    lines = [
        f"Threshold: {threshold:.3f} ({fraction:.0%} of the {reference} arm's final `{metric}`), smoothing window {window}.",
        "",
        "| Arm | Seeds | Reached | Steps to threshold (M) | Speed-up vs " + reference + " | AUC | Final return | AUC − " + reference + " | Final − " + reference + " | Greedy return | Greedy − " + reference + " | Wall-clock (min) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, entry in summary["arms"].items():
        lines.append(
            f"| {arm} | {entry['seeds']} | {entry['reached']}/{entry['seeds']} | {fmt(entry['steps_to_threshold'], 1e6)} | "
            f"{fmt(entry['speedup_vs_reference'])} | {fmt(entry['auc'])} | {fmt(entry['final'])} | "
            f"{fmt(entry['auc_minus_reference'])} | {fmt(entry['final_minus_reference'])} | "
            f"{fmt(entry['greedy_return'])} | {fmt(entry['greedy_return_minus_reference'])} | {fmt(entry['wall_clock_s'], 60, 1)} |"
        )
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    for arm, runs in curves.items():
        steps = runs[0][1]
        stack = np.stack([trailing(curve, window) for _, _, curve in runs])
        rng = np.random.default_rng(0)
        draws = rng.integers(0, len(stack), (2000, len(stack)))
        centre = trim_mean(np.nan_to_num(stack, nan=0.0), 0.25, axis=0)
        boots = trim_mean(np.nan_to_num(stack, nan=0.0)[draws], 0.25, axis=1)
        line, = axis.plot(steps / 1e6, centre, label=f"{arm} (n={len(runs)})")
        axis.fill_between(steps / 1e6, *np.percentile(boots, [2.5, 97.5], axis=0), color=line.get_color(), alpha=0.2)
    axis.axhline(threshold, color="grey", linestyle="--", linewidth=1, label="threshold")
    axis.set_xlabel("environment steps (M)")
    axis.set_ylabel(f"{metric} (IQM over seeds, smoothed)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / "learning_curves.png", dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fraction", type=float, default=0.95)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--metric", default=METRIC)
    args = parser.parse_args()
    main(args.manifest, args.out, args.fraction, args.window, args.metric)
