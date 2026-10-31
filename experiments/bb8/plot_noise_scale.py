"""E2: the noise scale over training, from the probe's logs in each run's metrics.npz.

    python experiments/bb8/plot_noise_scale.py <e2 dir> --out experiments/bb8/e2 [--budget-ppo N --budget-pqn N]

<e2 dir> holds <algorithm>_<game>/<seed>/ run directories. Per epoch, the noise scale is the epoch's mean noise over
its mean signal (signal and noise averaged separately, like the probe's moving average). Runs that stop short of their
budget are refused. Writes noise_scale.png (one panel per algorithm and game, PPO actor and critic and PQN TD, raw
gradient against update) and growth.csv / growth.md: for each series, the median over seeds of the noise scale in the
first and last 20% of training (epoch 1 excluded: its first update comes before any learning) and their ratio, with
the number of seeds whose ratio is above 1.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

BUDGETS = {"ppo": 19_988_480, "pqn": 79_953_920}
LOSSES = {"ppo": ("actor", "critic", "total"), "pqn": ("td",)}
SPACES = ("gradient", "update", "fresh")


def epoch_series(data, name, space):
    steps = len(data["steps"])
    noise = data[f"bb8/{name}/{space}/noise"].reshape(steps, -1).mean(-1)
    signal = data[f"bb8/{name}/{space}/signal"].reshape(steps, -1).mean(-1)
    smoothed = data[f"bb8/{name}/{space}/noise_scale_envs"].reshape(steps, -1).mean(-1)
    return noise / signal, smoothed


def load(e2, budgets):
    runs = {}
    for directory in sorted(Path(e2).glob("*_*/*")):
        algorithm, game = directory.parent.name.split("_", 1)
        path = directory / "metrics.npz"
        if not path.exists():
            raise SystemExit(f"{directory}: no metrics.npz (crashed or still running)")
        data = dict(np.load(path))
        if int(data["steps"][-1]) < budgets[algorithm]:
            raise SystemExit(f"{directory}: stops at {int(data['steps'][-1]):,} of {budgets[algorithm]:,} steps")
        runs.setdefault((algorithm, game), []).append((directory, data))
    return runs


def main(e2, out, budgets):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load(e2, budgets)
    keys = sorted(runs, key=lambda key: (key[0] != "ppo", key[1]))
    figure, axes = plt.subplots(2, len(keys), figsize=(3.2 * len(keys), 6.4), squeeze=False)
    rows = []
    for column, (algorithm, game) in enumerate(keys):
        for name in LOSSES[algorithm]:
            for space in SPACES:
                ratios = []
                for _, data in runs[(algorithm, game)]:
                    steps = data["steps"].astype(float)
                    scale, _ = epoch_series(data, name, space)
                    ratios.append((steps, scale))
                steps = ratios[0][0]
                stack = np.stack([scale for _, scale in ratios])
                count = stack.shape[1]
                early = slice(2, 2 + max(1, int(0.2 * (count - 2))))
                late = slice(count - max(1, int(0.2 * (count - 2))), count)
                first = np.nanmedian(stack[:, early], axis=1)
                last = np.nanmedian(stack[:, late], axis=1)
                rows.append({
                    "algorithm": algorithm, "game": game, "loss": name, "space": space, "seeds": len(stack),
                    "early_envs": float(np.median(first)), "late_envs": float(np.median(last)),
                    "growth": float(np.median(last / first)), "seeds_growing": int(np.sum(last > first)),
                    "early_transitions": float(np.median(first)) * 16, "late_transitions": float(np.median(last)) * 16,
                })
                if space == "fresh" or name == "total":
                    continue
                axis = axes[0 if space == "gradient" else 1, column]
                median = np.nanmedian(np.where(stack > 0, stack, np.nan), axis=0)
                line, = axis.plot(steps / 1e6, median, label=name)
                for seed in stack:
                    axis.plot(steps / 1e6, np.where(seed > 0, seed, np.nan), color=line.get_color(), alpha=0.25, lw=0.7)
        for row, space in enumerate(("gradient", "update")):
            axis = axes[row, column]
            axis.set_yscale("log")
            axis.set_title(f"{algorithm.upper()} {game}: {space}", fontsize=9)
            axis.set_xlabel("environment steps (M)")
            axis.grid(alpha=0.3, which="both")
            axis.legend(fontsize=7)
        axes[0, column].set_ylabel("B_simple (environments of T = 16)")
        axes[1, column].set_ylabel("B_simple (environments of T = 16)")
    figure.tight_layout()
    figure.savefig(out / "noise_scale.png", dpi=130)

    with open(out / "growth.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Algorithm | Game | Loss | Space | Early (envs) | Late (envs) | Late / early | Seeds growing |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['algorithm']} | {row['game']} | {row['loss']} | {row['space']} | {row['early_envs']:.1f} | "
                     f"{row['late_envs']:.1f} | {row['growth']:.2f} | {row['seeds_growing']}/{row['seeds']} |")
    (out / "growth.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("e2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget-ppo", type=int, default=BUDGETS["ppo"])
    parser.add_argument("--budget-pqn", type=int, default=BUDGETS["pqn"])
    args = parser.parse_args()
    main(args.e2, args.out, {"ppo": args.budget_ppo, "pqn": args.budget_pqn})
