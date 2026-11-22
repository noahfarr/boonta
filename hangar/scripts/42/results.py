"""Summarises the paper's runs and draws the scaling and podracer figures.

    python hangar/scripts/42/results.py runs/paper --pull 42:/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/runs/paper
    python hangar/scripts/42/results.py runs/paper --figures paper/figures

--pull copies only the metrics.npz files with rsync. Set RSYNC_RSH, for example to
"ssh -J ias", when the cluster is only reachable through another host. Runs are
read from <runs>/<name>/metrics.npz, with names as written by experiments.py.
"""

import argparse
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

# The first categorical slots of the validated palette, in fixed order.
COLORS = {"ppo-breakout": "#2a78d6", "ppo-craftax": "#eb6834", "dqn-breakout": "#1baf7a"}
LABELS = {"ppo-breakout": "PPO, MinAtar Breakout", "ppo-craftax": "PPO, Craftax-Classic", "dqn-breakout": "DQN, MinAtar Breakout"}
SHORT = {"ppo-breakout": "PPO Breakout", "ppo-craftax": "PPO Craftax", "dqn-breakout": "DQN Breakout"}
PODRACERS = {"anakin": ("Anakin", "#2a78d6"), "sebulba": ("Sebulba", "#eb6834")}
INK, MUTED, GRID, SURFACE = "#1a1a19", "#6b6b66", "#e4e4e1", "#ffffff"

# Returns that count as solved for the time-to-target column. Adjust once the
# single-device runs show what each workload reaches.
TARGETS = {"ppo-breakout": 40.0, "ppo-craftax": 10.0, "dqn-breakout": 20.0}


def pull(remote, local):
    local.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-az", "--include=*/", "--include=metrics.npz", "--exclude=*", f"{remote.rstrip('/')}/", f"{local}/"]
    if os.environ.get("RSYNC_RSH"):
        command[1:1] = ["-e", os.environ["RSYNC_RSH"]]
    subprocess.run(command, check=True)


def load(path):
    with np.load(path) as data:
        series = {key: np.asarray(data[key], dtype=float) for key in data.files}
    series = {key: value.reshape(len(value), -1).mean(axis=1) if value.ndim > 1 else value for key, value in series.items()}
    steps = series.pop("steps")
    sps = series.get("monitor/SPS", np.full_like(steps, np.nan))
    seconds = np.concatenate([[0.0], np.cumsum(np.diff(steps) / sps[1:])])
    return {
        "steps": steps,
        "seconds": seconds,
        "returns": series.get("training/episode_return", np.full_like(steps, np.nan)),
        "sps": float(np.nanmedian(sps[2:])) if len(sps) > 2 else float("nan"),
        "compile": float(np.nanmax(series.get("monitor/compilation_time", [np.nan]))),
        "memory": float(np.nanmax(series.get("gpu/memory_consumption", [np.nan]))),
    }


def time_to(run, target):
    reached = np.flatnonzero(run["returns"] >= target)
    return float(run["seconds"][reached[0]]) if len(reached) else float("nan")


def collect(root):
    runs = {}
    for path in sorted(root.glob("*/metrics.npz")):
        runs[path.parent.name] = load(path)
    return runs


def scaling(runs, kind):
    table = defaultdict(lambda: defaultdict(list))
    pattern = re.compile(rf"^{kind}-(.+)-(\d+)gpu(?:-s\d+)?$")
    for name, run in runs.items():
        match = pattern.match(name)
        if match:
            table[match.group(1)][int(match.group(2))].append(run)
    return table


def print_scaling(table, kind):
    print(f"\n{kind} scaling")
    print("| Workload | GPUs | Steps per second | Speedup | Efficiency | GPU memory (GiB) | Time to target (s) | Final return |")
    print("|---|---|---|---|---|---|---|---|")
    for workload, by_devices in table.items():
        base = np.mean([r["sps"] for r in by_devices.get(1, [])]) if 1 in by_devices else np.nan
        for devices in sorted(by_devices):
            group = by_devices[devices]
            sps = np.mean([r["sps"] for r in group])
            target = np.nanmean([time_to(r, TARGETS.get(workload, np.inf)) for r in group])
            final = np.nanmean([r["returns"][-1] for r in group])
            memory = np.mean([r["memory"] for r in group])
            print(
                f"| {LABELS.get(workload, workload)} | {devices} | {sps:,.0f} | {sps / base:.2f} | "
                f"{sps / base / devices:.2f} | {memory:.1f} | {target:.0f} | {final:.1f} |"
            )


def print_others(runs):
    print("\nother runs")
    print("| Run | Steps per second | Compilation (s) | Final return |")
    print("|---|---|---|---|")
    for name, run in runs.items():
        if name.startswith(("weak-", "strong-")):
            continue
        print(f"| {name} | {run['sps']:,.0f} | {run['compile']:.0f} | {run['returns'][-1]:.1f} |")


def compact(value, _=None):
    if abs(value) >= 1e9:
        return f"{value / 1e9:g}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:g}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:g}K"
    return f"{value:g}"


def style(axis):
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.grid(True, color=GRID, linewidth=0.75, linestyle="-")
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, labelsize=8, length=0)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)


def spread(positions, gap):
    """Moves sorted label positions apart until neighbours are at least gap apart."""
    order = np.argsort(positions)
    placed = np.array(positions, dtype=float)[order]
    for _ in range(100):
        moved = False
        for i in range(1, len(placed)):
            overlap = gap - (placed[i] - placed[i - 1])
            if overlap > 1e-9:
                placed[i - 1] -= overlap / 2
                placed[i] += overlap / 2
                moved = True
        if not moved:
            break
    result = np.empty_like(placed)
    result[order] = placed
    return result


def draw_scaling(weak, strong, path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), constrained_layout=True)
    for axis, table, title in ((axes[0], weak, "Weak scaling"), (axes[1], strong, "Strong scaling")):
        style(axis)
        devices = sorted({d for by_devices in table.values() for d in by_devices}) or [1]
        axis.plot(devices, devices, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        labels = [("Linear", devices[-1], MUTED)]
        for workload in COLORS:
            by_devices = table.get(workload)
            if not by_devices or 1 not in by_devices:
                continue
            base = np.mean([r["sps"] for r in by_devices[1]])
            xs = sorted(by_devices)
            ys = [np.mean([r["sps"] for r in by_devices[d]]) / base for d in xs]
            axis.plot(xs, ys, color=COLORS[workload], linewidth=1.5, solid_capstyle="round", marker="o", markersize=5,
                      markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[workload], zorder=3)
            labels.append((SHORT[workload], ys[-1], INK))
        # Direct labels sit at the end of their lines, spread apart on the log axis so they never overlap.
        heights = spread([np.log2(y) for _, y, _ in labels], gap=0.3)
        for (text, _, color), height in zip(labels, heights):
            axis.annotate(text, (devices[-1], 2**height), xytext=(5, 0), textcoords="offset points", color=color, fontsize=7, va="center", annotation_clip=False)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xticks(devices, [str(d) for d in devices])
        axis.set_yticks(devices, [f"{d}x" for d in devices])
        axis.set_xlabel("GPUs", fontsize=8)
        axis.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("Speedup over one GPU", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    figure.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


def draw_podracers(runs, path):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(3.3, 2.3), constrained_layout=True)
    style(axis)
    for key, (label, color) in PODRACERS.items():
        group = [run for name, run in runs.items() if name.startswith(f"podracers-{key}-")]
        if not group:
            continue
        steps = group[0]["steps"][1:]
        returns = np.stack([run["returns"][1 : len(steps) + 1] for run in group])
        mean = returns.mean(axis=0)
        resampled = returns[np.random.default_rng(0).integers(0, len(returns), (2000, len(returns)))].mean(axis=1)
        low, high = np.percentile(resampled, [2.5, 97.5], axis=0)
        axis.fill_between(steps, low, high, color=color, alpha=0.15, linewidth=0)
        seeds = f"{len(group)} seed" + ("s" if len(group) > 1 else "")
        axis.plot(steps, mean, color=color, linewidth=1.5, solid_capstyle="round", label=f"{label} ({seeds})")
        axis.annotate(label, (steps[-1], mean[-1]), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.xaxis.set_major_formatter(FuncFormatter(compact))
    axis.set_xlabel("Environment steps", fontsize=8)
    axis.set_ylabel("Episode return", fontsize=8)
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    figure.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, help="directory with one subdirectory per run")
    parser.add_argument("--pull", help="rsync source such as 42:/pfss/.../runs/paper")
    parser.add_argument("--figures", type=Path, help="directory for scaling.pdf and podracers.pdf")
    args = parser.parse_args()

    if args.pull:
        pull(args.pull, args.runs)
    runs = collect(args.runs)
    print(f"{len(runs)} runs in {args.runs}")
    weak, strong = scaling(runs, "weak"), scaling(runs, "strong")
    print_scaling(weak, "weak")
    print_scaling(strong, "strong")
    print_others(runs)
    if args.figures:
        args.figures.mkdir(parents=True, exist_ok=True)
        draw_scaling(weak, strong, args.figures / "scaling.pdf")
        draw_podracers(runs, args.figures / "podracers.pdf")


if __name__ == "__main__":
    main()
