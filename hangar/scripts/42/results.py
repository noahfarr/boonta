import argparse
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

STYLE = Path(__file__).parent / "paper.mplstyle"
TEXT_WIDTH = 5.5
GOLDEN_RATIO = 1.618
SET1 = ("#e41a1c", "#377eb8", "#4daf4a")
COLORS = {"ppo-breakout": SET1[1], "ppo-craftax": SET1[0], "dqn-breakout": SET1[2]}
LABELS = {"ppo-breakout": "PPO, Breakout-MinAtar", "ppo-craftax": "PPO, Craftax-Classic", "dqn-breakout": "DQN, Breakout-MinAtar"}
PODRACERS = {"anakin": ("Anakin (1 GPU)", SET1[1]), "sebulba": ("Sebulba (2 GPUs)", SET1[0])}
LINEAR = {"color": "0.45", "linewidth": 0.8, "linestyle": (0, (4, 2))}
GAMES = {"breakout": "Breakout", "asterix": "Asterix", "freeway": "Freeway", "space_invaders": "Space Invaders"}
ALGORITHMS = {"ppo": ("PPO", SET1[1]), "dqn": ("DQN", SET1[0]), "pqn": ("PQN", SET1[2])}

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


def collect_curves(root):
    curves = {}
    for run in sorted(root.glob("curve-*")):
        points = []
        for path in run.glob("*/metrics.npz"):
            with np.load(path) as data:
                points.append((float(data["steps"].ravel()[-1]), float(np.nanmean(data["evaluation/episode_return"]))))
        if points:
            steps, returns = (np.array(values) for values in zip(*sorted(points)))
            curves[run.name] = {"steps": steps, "returns": returns}
    return curves


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


def style(axis):
    import matplotlib.pyplot as plt

    axis.grid(True, alpha=0.2)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_linewidth(plt.rcParams["axes.linewidth"])


def draw_scaling(weak, strong, path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    plt.style.use(STYLE)
    height = TEXT_WIDTH / 2 / GOLDEN_RATIO * 1.2
    figure, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, height), constrained_layout=True)
    figure.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.06)
    drawn = []
    for axis, table, title in ((axes[0], weak, "Weak scaling"), (axes[1], strong, "Strong scaling")):
        style(axis)
        devices = sorted({d for by_devices in table.values() for d in by_devices}) or [1]
        axis.plot(devices, devices, zorder=1, **LINEAR)
        for workload, color in COLORS.items():
            by_devices = table.get(workload)
            if not by_devices or 1 not in by_devices:
                continue
            base = np.mean([r["sps"] for r in by_devices[1]])
            xs = sorted(by_devices)
            ys = [np.mean([r["sps"] for r in by_devices[d]]) / base for d in xs]
            axis.plot(xs, ys, color=color, marker="o", markersize=3, zorder=3)
            drawn.append(workload)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xticks(devices, [str(d) for d in devices])
        axis.set_yticks(devices, [str(d) for d in devices])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_minor_locator(NullLocator())
        axis.set_title(title)
        axis.set_xlabel("GPUs")
    axes[0].set_ylabel("Speedup over one GPU")
    handles = [plt.Line2D([], [], color=COLORS[w], marker="o", markersize=3, label=LABELS[w]) for w in COLORS if w in drawn]
    handles.append(plt.Line2D([], [], label="Linear", **LINEAR))
    figure.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path}")


def interval(returns):
    resampled = returns[np.random.default_rng(0).integers(0, len(returns), (2000, len(returns)))].mean(axis=1)
    return np.percentile(resampled, [2.5, 97.5], axis=0)


def draw_minatar(curves, path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    rows = [a for a in ALGORITHMS if any(name.startswith(f"curve-{a}-") for name in curves)]
    if not rows:
        return
    plt.style.use(STYLE)
    height = TEXT_WIDTH / 4 / GOLDEN_RATIO * 1.2 * len(rows) + 0.35
    figure, axes = plt.subplots(len(rows), len(GAMES), figsize=(TEXT_WIDTH, height), constrained_layout=True, squeeze=False)
    figure.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.03, hspace=0.04)
    for row, algorithm in enumerate(rows):
        label, color = ALGORITHMS[algorithm]
        for column, (game, title) in enumerate(GAMES.items()):
            axis = axes[row, column]
            style(axis)
            group = [curve for name, curve in curves.items() if re.fullmatch(rf"curve-{algorithm}-{game}-s\d+", name)]
            if group:
                steps = group[0]["steps"] / 1e6
                returns = np.stack([curve["returns"] for curve in group])
                low, high = interval(returns)
                axis.fill_between(steps, low, high, color=color, alpha=0.2, linewidth=0)
                axis.plot(steps, returns.mean(axis=0), color=color)
                axis.set_xlim(0, steps[-1])
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(label, rotation=0, labelpad=12)
    figure.supxlabel("Environment steps (millions)", fontsize=plt.rcParams["axes.labelsize"])
    figure.supylabel("Episode return", fontsize=plt.rcParams["axes.labelsize"])
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path}")


def draw_podracers(curves, path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if not any(name.startswith("curve-podracers-") for name in curves):
        return
    plt.style.use(STYLE)
    width = TEXT_WIDTH / 2
    figure, axis = plt.subplots(figsize=(width, width / GOLDEN_RATIO * 1.25), constrained_layout=True)
    figure.get_layout_engine().set(w_pad=0.01, h_pad=0.01)
    style(axis)
    for key, (label, color) in PODRACERS.items():
        group = [curve for name, curve in curves.items() if name.startswith(f"curve-podracers-{key}-")]
        if not group:
            continue
        steps = group[0]["steps"]
        returns = np.stack([curve["returns"] for curve in group])
        mean = returns.mean(axis=0)
        low, high = interval(returns)
        axis.fill_between(steps / 1e6, low, high, color=color, alpha=0.2, linewidth=0)
        axis.plot(steps / 1e6, mean, color=color, label=label, zorder=2)
    axis.set_xlim(0, 20)
    axis.set_xticks([0, 5, 10, 15, 20])
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.set_xlabel("Environment steps (millions)")
    axis.set_ylabel("Episode return")
    figure.legend(loc="outside upper center", ncol=2, frameon=False)
    figure.savefig(path)
    plt.close(figure)
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
        curves = collect_curves(args.runs)
        draw_podracers(curves, args.figures / "podracers.pdf")
        draw_minatar(curves, args.figures / "minatar.pdf")


if __name__ == "__main__":
    main()
