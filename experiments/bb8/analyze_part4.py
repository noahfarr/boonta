"""Round 2, Part 4: the held-out ratio r = held-out surrogate improvement / in-batch surrogate improvement, from a
deterministic rerun of E1 v2's updates (no training): does r ≈ B / (B + N)?

    python experiments/bb8/analyze_part4.py <r2p4 dir> --e1 <phase2v2 dir> --fits experiments/bb8/e1v2/fits.csv \
        --out experiments/bb8/r2p4

Definitions (fixed before looking at any r):
- Surrogate improvement of an update: E[(π'(a|s)/π(a|s) − 1) A], unnormalized GAE of the pre-update critic. Held-out: on 16
  rollouts of the pre-update policy collected after the update's batch; in-batch: on the update's own columns.
- r per condition = held-out / in-batch. Summary per (run, checkpoint, B): the mean of r over the 4 update seeds at
  the multiplier `--multiplier` (default 1, the tuned step), with a bootstrap 95% interval over update seeds; conditions
  whose in-batch improvement is not positive are excluded and counted.
- Fit per (run, checkpoint): r(B) = B / (B + N_r), least squares in r over the five B, N_r on a log grid 1..1e7.
  Residuals: r − fit. Compared with the ΔJ-fitted N (proxy fits from E1 v2) where that was identified (finite CI).
- Reproduction check: every rerun condition's held-out proxy must equal E1 v2's stored value (same keys, same update).
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

GRID = np.exp(np.linspace(0.0, np.log(1e7), 1400))


def fit(b, r):
    model = b[None, :] / (b[None, :] + GRID[:, None])
    errors = np.sum((r[None, :] - model) ** 2, axis=1)
    i = int(np.argmin(errors))
    return float(GRID[i]), model[i]


def main(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fitted = {}
    for row in csv.DictReader(open(args.fits)):
        if row["algorithm"] == "ppo" and row["optimizer"] == "adam" and row["policy"] == "proxy":
            identified = float(row["fit_N_high"]) < 1e6 and float(row["fit_N"]) > 1
            fitted[(row["game"], int(row["seed"]), int(row["step"]))] = float(row["fit_N"]) if identified else None
    mismatches, rows, curves = 0, [], {}
    for path in sorted(Path(args.directory).glob("ppo_adam_*.json")):
        rerun = json.loads(path.read_text())
        original = json.loads((Path(args.e1) / path.name).read_text())
        game, seed = path.stem.split("_")[2], int(path.stem.split("_")[-1])
        for step, entry in rerun["checkpoints"].items():
            stored = [c for c in original["checkpoints"][step]["conditions"] if c["test"] == "grid"]
            mine = [c for c in entry["conditions"] if c["test"] == "grid"]
            mismatches += sum(a["proxy"] != b["proxy"] for a, b in zip(stored, mine)) + abs(len(stored) - len(mine))
            ratios = defaultdict(list)
            excluded = 0
            for c in mine:
                if c["multiplier"] != args.multiplier:
                    continue
                if c["in_batch"] <= 0:
                    excluded += 1
                    continue
                ratios[c["columns"]].append(c["proxy"] / c["in_batch"])
            b = np.array(sorted(ratios), float)
            r = np.array([np.mean(ratios[int(x)]) for x in b])
            rng = np.random.default_rng(0)
            ci = [np.percentile([np.mean(rng.choice(ratios[int(x)], len(ratios[int(x)]))) for _ in range(2000)], [2.5, 97.5])
                  for x in b]
            n_r, model = fit(b, r)
            n_j = fitted.get((game, seed, int(step)))
            curves[(game, seed, int(step))] = (b, r, ci, n_r)
            rows.append({"game": game, "seed": seed, "step": int(step), "multiplier": args.multiplier,
                         "B": " ".join(f"{x:.0f}" for x in b), "r": " ".join(f"{x:.3f}" for x in r),
                         "residuals": " ".join(f"{x:+.3f}" for x in r - model),
                         "rms_residual": float(np.sqrt(np.mean((r - model) ** 2))), "N_r": n_r,
                         "N_deltaJ": n_j if n_j is not None else float("nan"),
                         "N_r_over_N_deltaJ": n_r / n_j if n_j else float("nan"), "excluded": excluded})
    with open(out / "part4.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"Reproduction: {mismatches} rerun conditions differ from E1 v2's stored held-out proxy.", "",
             f"r = held-out / in-batch surrogate improvement at multiplier {args.multiplier}, mean over 4 update seeds.", "",
             "| Game | Seed | Checkpoint | r at B = 64 / 256 / 1024 / 2560 / 8192 | Fit N_r | RMS residual | ΔJ-fitted N | N_r / N |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['game']} | {row['seed']} | {row['step'] / 1e6:.1f}M | {row['r']} | {row['N_r']:.0f} | "
                     f"{row['rms_residual']:.3f} | {row['N_deltaJ']:.0f} | {row['N_r_over_N_deltaJ']:.2f} |")
    (out / "part4.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    figure, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    for column, step in enumerate(sorted({k[2] for k in curves})):
        axis = axes[0, column]
        for (game, seed, s), (b, r, ci, n_r) in sorted(curves.items()):
            if s != step:
                continue
            line, = axis.plot(b, r, "o-" if game == "breakout" else "s--", ms=4, label=f"{game} {seed} (N_r {n_r:.0f})")
            fine = np.geomspace(b.min(), b.max(), 50)
            axis.plot(fine, fine / (fine + n_r), color=line.get_color(), lw=0.6, alpha=0.6)
        axis.set_xscale("log")
        axis.set_xlabel("B (environments)")
        axis.set_ylabel("r = held-out / in-batch")
        axis.set_title(f"checkpoint {step / 1e6:.1f}M", fontsize=10)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(out / "part4_ratio.png", dpi=130)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--e1", required=True)
    parser.add_argument("--fits", required=True)
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
