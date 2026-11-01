"""E1 v2 extras: proxy vs episode-return ΔJ (correlation over conditions), and the best step against training stage.

    python experiments/bb8/e1_stage.py <phase2v2 dir> --out experiments/bb8/e1v2
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def main(directory, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    rows, best = [], defaultdict(list)
    for path in sorted(Path(directory).glob("ppo_adam_*.json")):
        game, seed = path.stem.split("_")[2], int(path.stem.split("_")[-1])
        report = json.loads(path.read_text())
        for step, entry in report["checkpoints"].items():
            cells = defaultdict(list)
            for c in entry["conditions"]:
                if c["test"] == "grid":
                    cells[(c["columns"], c["multiplier"])].append((c["delta_j"], c["proxy"], c["kl"]))
            keys = sorted(cells)
            dj = np.array([np.mean([v[0] for v in cells[k]]) for k in keys])
            px = np.array([np.mean([v[1] for v in cells[k]]) for k in keys])
            rows.append({"game": game, "seed": seed, "step": int(step), "pearson": float(pearsonr(dj, px)[0]),
                         "spearman": float(spearmanr(dj, px)[0]), "conditions": len(keys)})
            for columns in sorted({k[0] for k in keys}):
                ms = [k for k in keys if k[0] == columns]
                for measure, values in (("episode", dj), ("proxy", px)):
                    chosen = max(ms, key=lambda k: values[keys.index(k)])
                    best[(game, measure, columns)].append((int(step), seed, chosen[1], entry["j_before"]))
    (out / "proxy_validation.json").write_text(json.dumps(rows, indent=1))
    print("proxy vs episode ΔJ over the 50 (B, multiplier) cells, mean over update seeds:")
    for r in rows:
        print(f"  {r['game']:9} seed {r['seed']} {r['step'] / 1e6:5.1f}M  pearson {r['pearson']:+.2f}  spearman {r['spearman']:+.2f}")

    games = sorted({k[0] for k in best})
    figure, axes = plt.subplots(1, len(games), figsize=(5.5 * len(games), 4), squeeze=False)
    summary = {}
    for axis, game in zip(axes[0], games):
        for measure, marker in (("episode", "o"), ("proxy", "s")):
            for columns, colour in ((1024, "tab:blue"), (8192, "tab:red")):
                points = best[(game, measure, columns)]
                steps = np.array([p[0] for p in points]) / 1e6
                mults = np.array([p[2] for p in points])
                jitter = (np.array([p[1] for p in points]) - 1) * 0.25 + (0.12 if measure == "proxy" else -0.12)
                axis.scatter(steps + jitter, mults, marker=marker, color=colour, alpha=0.7, s=22,
                             label=f"B={columns}, {measure}")
                for step in sorted(set(steps)):
                    summary.setdefault(f"{game}/{measure}/B{columns}", {})[f"{step:.1f}M"] = float(
                        np.exp(np.median(np.log(mults[steps == step]))))
        axis.set_yscale("log", base=2)
        axis.axhline(1.0, color="grey", lw=0.8, ls="--")
        axis.set_xlabel("checkpoint (M environment steps; seeds jittered)")
        axis.set_ylabel("best learning-rate multiplier (× tuned 2.4e-3)")
        axis.set_title(f"PPO {game}: best step over training", fontsize=10)
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(out / "best_step_vs_stage.png", dpi=130)
    (out / "best_step_vs_stage.json").write_text(json.dumps(summary, indent=1))
    print("median best multiplier (geometric) over seeds:")
    for k, v in summary.items():
        print(f"  {k:28} " + "  ".join(f"{s}: {m:.3g}" for s, m in v.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.directory, args.out)
