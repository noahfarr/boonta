"""E1 analysis: fit ΔJ*(B) = ΔJ_max / (1 + N / B) to the best-step curve of every run and checkpoint, and compare N
with the large-probe noise scales measured at the same checkpoint.

    python experiments/bb8/analyze_e1.py <phase2 dir> --out experiments/bb8/e1

Definitions (fixed before looking at any E1 result):
- B is in environments (T = 16 segments): the number of columns of the update's batch.
- ΔJ of a condition: mean over episodes of the per-episode difference (common random numbers), averaged over update
  seeds. Best step at a fraction: the multiplier with the highest ΔJ there.
- Fit: weighted least squares (weights 1 / SE², SE over update seeds and episodes) over N on a log grid from 1 to
  1e7 environments, with ΔJ_max solved in closed form for each N.
- Intervals: 300 bootstrap resamples of episodes and update seeds together. The best multiplier is re-selected in
  every resample, so the selection effect is part of the interval.
- Agreement: fitted N / probe noise scale, for raw, fresh and full, for each loss.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

GRID = np.exp(np.linspace(0.0, np.log(1e7), 1400))
SPACES = ("gradient", "fresh", "update")


def fit(b, y, se):
    w = 1.0 / np.maximum(se, 1e-12) ** 2
    g = 1.0 / (1.0 + GRID[:, None] / b[None, :])
    a = (g * w * y).sum(1) / (g * g * w).sum(1)
    chi = ((y[None] - a[:, None] * g) ** 2 * w).sum(1)
    i = int(np.argmin(chi))
    return float(GRID[i]), float(a[i]), float(chi[i]), a[i] * g[i]


def curves(cube, multipliers):
    """cube[fraction, multiplier, seed, episode] -> best-step ΔJ, its SE, the best multiplier index, per fraction."""
    means = cube.mean(axis=(2, 3))
    best = means.argmax(axis=1)
    chosen = cube[np.arange(cube.shape[0]), best]  # (fraction, seed, episode)
    per_seed = chosen.mean(axis=2)
    se = np.sqrt(chosen.var(axis=2).mean(axis=1) / cube.shape[3] / cube.shape[2] + per_seed.var(axis=1) / cube.shape[2])
    return means[np.arange(cube.shape[0]), best], se, best


def load(directory):
    for path in sorted(Path(directory).glob("*.json")):
        if path.name.startswith("tune_"):
            continue
        report = json.loads(path.read_text())
        arrays = np.load(path.with_suffix(".npz"))
        algorithm, optimizer, *rest = path.stem.split("_")
        if algorithm == "pqn":
            game, seed = "_".join([optimizer, *rest[:-1]]), rest[-1]
            optimizer = "adam"
        else:
            game, seed = "_".join(rest[:-1]), rest[-1]
        yield path.stem, algorithm, optimizer, game, int(seed), report, arrays


def analyze_checkpoint(stem, step, entry, arrays, key):
    grid = [(i, c) for i, c in enumerate(entry["conditions"]) if c["test"] == "grid"]
    fractions = sorted({c["fraction"] for _, c in grid})
    multipliers = sorted({c["multiplier"] for _, c in grid})
    seeds = sorted({c["seed"] for _, c in grid})
    columns = {c["fraction"]: c["columns"] for _, c in grid}
    episodes = len(arrays[f"{step}/{grid[0][0]}"])
    cube = np.zeros((len(fractions), len(multipliers), len(seeds), episodes))
    size = np.zeros((len(fractions), len(multipliers), len(seeds)))
    size_key = "kl" if "kl" in grid[0][1] else "update_norm"
    for index, c in grid:
        f, m, s = fractions.index(c["fraction"]), multipliers.index(c["multiplier"]), seeds.index(c["seed"])
        cube[f, m, s] = arrays[f"{step}/{index}{key}"]
        size[f, m, s] = c[size_key]
    b = np.asarray([columns[f] for f in fractions], float)
    y, se, best = curves(cube, multipliers)
    n, a, chi, predicted = fit(b, y, se)
    rng = np.random.default_rng(0)
    boot_n, boot_y = [], []
    for _ in range(300):
        e = rng.integers(0, episodes, episodes)
        s = rng.integers(0, len(seeds), len(seeds))
        yb, seb, _ = curves(cube[:, :, s][:, :, :, e], multipliers)
        boot_n.append(fit(b, yb, seb)[0])
        boot_y.append(yb)
    boot_y = np.asarray(boot_y)
    low, high = np.percentile(boot_y, [2.5, 97.5], axis=0)
    unadapted = cube[:, multipliers.index(1.0)].mean(axis=(1, 2))
    best_size = size[np.arange(len(fractions)), best].mean(axis=1)
    ratio = b / (b + n)
    slope = float(np.polyfit(np.log(ratio), np.log(np.maximum(best_size, 1e-12)), 1)[0]) if np.all(best_size > 0) else float("nan")
    lr_slope = float(np.polyfit(np.log(ratio), np.log(np.asarray(multipliers)[best]), 1)[0])
    matched = {}
    for index, c in enumerate(entry["conditions"]):
        if c["test"] == "matched_kl":
            matched.setdefault(c["fraction"], []).append(c["delta_j"])
    probe = {}
    for name in ("actor", "critic", "total", "td"):
        for space in SPACES:
            k = f"bb8/{name}/{space}/noise_scale_envs"
            if k in entry["noise_big"]:
                probe[f"{name}/{space}"] = entry["noise_big"][k]
                probe[f"{name}/{space}/single"] = entry["noise_single"][k]
    return {
        "B": b.tolist(), "best_delta_j": y.tolist(), "se": se.tolist(), "ci_low": low.tolist(), "ci_high": high.tolist(),
        "best_multiplier": [multipliers[i] for i in best], "unadapted_delta_j": unadapted.tolist(),
        "best_size": best_size.tolist(), "size_key": size_key, "fit_N": n, "fit_N_ci": np.percentile(boot_n, [2.5, 97.5]).tolist(),
        "fit_delta_j_max": a, "chi2": chi, "predicted": predicted.tolist(),
        "inside_ci": int(np.sum((predicted >= low) & (predicted <= high))),
        "size_slope": slope, "multiplier_slope": lr_slope,
        "matched_kl_delta_j": {str(k): float(np.mean(v)) for k, v in sorted(matched.items())},
        "j_before": entry["j_before"], "probe": probe,
    }


def main(directory, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, results = [], {}
    for stem, algorithm, optimizer, game, seed, report, arrays in load(directory):
        for step, entry in report["checkpoints"].items():
            keys = [""] + (["/greedy"] if algorithm == "pqn" else [])
            for key in keys:
                r = analyze_checkpoint(stem, step, entry, arrays, key)
                policy = "greedy" if key else ("behaviour" if algorithm == "pqn" else "stochastic")
                results[f"{stem}/{step}/{policy}"] = r
                row = {"algorithm": algorithm, "optimizer": optimizer, "game": game, "seed": seed, "step": int(step),
                       "policy": policy, "j_before": r["j_before"], "fit_N": r["fit_N"],
                       "fit_N_low": r["fit_N_ci"][0], "fit_N_high": r["fit_N_ci"][1], "delta_j_max": r["fit_delta_j_max"],
                       "inside_ci": r["inside_ci"], "size_slope": r["size_slope"], "multiplier_slope": r["multiplier_slope"],
                       "best_multipliers": " ".join(f"{m:g}" for m in r["best_multiplier"]),
                       "unadapted_over_best_small": float(np.mean(np.asarray(r["unadapted_delta_j"][:2]) / np.maximum(np.asarray(r["best_delta_j"][:2]), 1e-9)))}
                for name, value in r["probe"].items():
                    if not name.endswith("single"):
                        row[f"probe_{name}"] = value
                        row[f"ratio_{name}"] = r["fit_N"] / value if value and value > 0 else float("nan")
                rows.append(row)
    fields = sorted({k for row in rows for k in row}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(out / "fits.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "fits.json").write_text(json.dumps(results, indent=1))

    groups = defaultdict(list)
    for name, r in results.items():
        stem, step, policy = name.rsplit("/", 2)
        groups[(stem.rsplit("_", 1)[0], step, policy)].append(r)
    keys = sorted(groups)
    columns = 3
    rows_n = (len(keys) + columns - 1) // columns
    figure, axes = plt.subplots(rows_n, columns, figsize=(5 * columns, 3.6 * rows_n), squeeze=False)
    for axis, key in zip(axes.flat, keys):
        for r in groups[key]:
            b = np.asarray(r["B"])
            line, = axis.plot(b, r["best_delta_j"], "o", ms=4)
            axis.fill_between(b, r["ci_low"], r["ci_high"], color=line.get_color(), alpha=0.15)
            fine = np.geomspace(b.min(), b.max(), 50)
            axis.plot(fine, r["fit_delta_j_max"] / (1 + r["fit_N"] / fine), color=line.get_color(), lw=1,
                      label=f"N={r['fit_N']:.0f}")
            axis.plot(b, r["unadapted_delta_j"], "x", color=line.get_color(), alpha=0.6)
        axis.set_xscale("log")
        axis.axhline(0, color="grey", lw=0.5)
        axis.set_title(f"{key[0]} @ {int(key[1]) / 1e6:.1f}M ({key[2]})", fontsize=9)
        axis.set_xlabel("B (environments)")
        axis.set_ylabel("ΔJ (best step o, 1x x)")
        axis.legend(fontsize=7)
    for axis in list(axes.flat)[len(keys):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(out / "law.png", dpi=120)
    print(f"wrote {out}: {len(rows)} fits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.directory, args.out)
