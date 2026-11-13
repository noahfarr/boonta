"""Part B analysis: batch-size probes per checkpoint (seeds 5 and 6), fits, T2 and T3.

Usage: uv run --no-sync python reports/pacman_signal/analyze_partB.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARTB = ROOT / "outputs" / "pacman_signal" / "partB"


def fit(ns, rho_sq):
    from scipy.optimize import curve_fit

    ns, rho_sq = np.asarray(ns), np.asarray(rho_sq)

    def model(n, a, b):
        return a * n / (n + b)

    try:
        (a, b), _ = curve_fit(model, ns, rho_sq, p0=(max(rho_sq.max(), 1e-6), 1.0), bounds=([0, 1e-6], [np.inf, 1e6]))
        residual = rho_sq - model(ns, a, b)
        total = np.sum((rho_sq - rho_sq.mean()) ** 2)
        return {"a": float(a), "B_star": float(b), "r2": float(1 - np.sum(residual**2) / total) if total > 0 else float("nan")}
    except (RuntimeError, ValueError) as error:
        return {"error": str(error)}


def positive_tail_fit(ns, values):
    """The fit restricted to the largest N from which rho stays positive, with at least 3 points."""
    values = np.asarray(values)
    start = len(values)
    while start > 0 and values[start - 1] > 0:
        start -= 1
    if len(values) - start < 3:
        return {"skipped": f"only {len(values) - start} positive points at the large-N end"}
    result = fit(list(ns)[start:], values[start:])
    result["from_N"] = float(ns[start])
    return result


def main():
    results = {}
    for seed_dir in sorted(PARTB.glob("seed*")):
        for path in sorted(seed_dir.glob("epoch_*.json")):
            r = json.loads(path.read_text())
            results[(seed_dir.name, r["epoch"])] = r
    out = {"checkpoints": {}}
    for (seed, epoch), r in sorted(results.items()):
        key = f"{seed}/epoch{epoch}"
        entry = {"S_raw": r["s_raw"], "S_cell": r["s_cell"], "S_look": r["s_look"], "S_look_se": r["s_look_se"], "S_look_single": r["s_look_single"], "schemes": {}}
        print(f"== {key} (step {r['step']/1e6:.1f}M): S_raw {r['s_raw']:.2f}, S_cell {r['s_cell']:.3f}, S_look {r['s_look']:.3f} +- {r['s_look_se']:.3f}")
        for scheme, block in r["schemes"].items():
            rows = block["rows"]
            ns = [row["N"] for row in rows]
            per = {name: [[b[name] for b in row["per_batch"]] for row in rows] for name in ("c_out", "c_in", "delta", "rho_look", "rho_high")}
            look_sq = [float(np.mean(np.square(v))) for v in per["rho_look"]]
            look_sq_se = [float(np.std(np.square(v), ddof=1) / np.sqrt(len(v))) for v in per["rho_look"]]
            # Signed rho^2 = rho |rho| with rho = mean c_out / sqrt(2 mean delta S): the population
            # quantity, not the mean of noisy squares, and negative when the update hurts held-out play.
            look_pop = [
                float(np.sign(np.mean(c)) * np.mean(c) ** 2 / (2 * np.mean(d) * r["s_look"])) if r["s_look"] > 0 else float("nan")
                for c, d in zip(per["c_out"], per["delta"])
            ]
            rho_look = [float(np.mean(c) / np.sqrt(2 * np.mean(d) * r["s_look"])) if r["s_look"] > 0 else float("nan") for c, d in zip(per["c_out"], per["delta"])]
            holdout = [float(np.mean(c) / np.mean(i)) if np.mean(i) else float("nan") for c, i in zip(per["c_out"], per["c_in"])]
            deltas = [float(np.mean(d)) for d in per["delta"]]
            entry["schemes"][scheme] = {
                "N": ns,
                "c_out": [float(np.mean(v)) for v in per["c_out"]],
                "c_out_se": [float(np.std(v, ddof=1) / np.sqrt(len(v))) for v in per["c_out"]],
                "c_out_per_sample": [row["c_out_per_sample"] for row in rows],
                "c_in": [float(np.mean(v)) for v in per["c_in"]],
                "delta": deltas,
                "holdout": holdout,
                "rho_look_sq": look_sq,
                "rho_look_sq_se": look_sq_se,
                "rho_look_sq_pop": look_pop,
                "rho_look": rho_look,
                # The saturating fit only makes sense where every update helps (rho > 0 at every N).
                "fit_look_pop": fit(ns, look_pop) if min(look_pop) > 0 else {"skipped": "rho <= 0 at some N"},
                "fit_look_positive_tail": positive_tail_fit(ns, look_pop),
                "fit_high": fit(ns, [row["rho_high_sq"] for row in rows]),
                "delta_max_over_min": float(max(deltas) / min(deltas)),
                "rho_sq_rises": bool(np.all(np.diff(look_pop) > 0)),
                "holdout_rises": bool(np.all(np.diff(holdout) > 0)),
            }
            e = entry["schemes"][scheme]
            print(f"  {scheme}: delta max/min over N {e['delta_max_over_min']:.2f}; rho^2 rises {e['rho_sq_rises']}; hold-out rises {e['holdout_rises']}; fit {e['fit_look_pop']}; fit over the positive tail {e['fit_look_positive_tail']}")
            for i, n in enumerate(ns):
                print(
                    f"    N={n:<6.4g} c_out {e['c_out'][i]:+.5f} +- {e['c_out_se'][i]:.5f} per sample {e['c_out_per_sample'][i]:+.2e} | c_in {e['c_in'][i]:+.5f} | hold-out {e['holdout'][i]:+.2f} | "
                    f"delta {e['delta'][i]:.5f} | rho_look {e['rho_look'][i]:+.3f} (signed rho^2 {e['rho_look_sq_pop'][i]:+.4f})"
                )
        out["checkpoints"][key] = entry
    (HERE / "partB.json").write_text(json.dumps(out, indent=1))
    plot(out)


def plot(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(out["checkpoints"])
    figure, axes = plt.subplots(2, 4, figsize=(18, 8))
    cmap = plt.get_cmap("viridis")
    for scheme_row, scheme in enumerate(("fixed_count", "fixed_size")):
        for index, key in enumerate(keys):
            e = out["checkpoints"][key]["schemes"].get(scheme)
            if e is None:
                continue
            color = cmap(index / max(len(keys) - 1, 1))
            style = "-" if key.startswith("seed5") else "--"
            ns = np.asarray(e["N"])
            axes[scheme_row, 0].errorbar(ns, e["c_out"], yerr=e["c_out_se"], color=color, linestyle=style, label=key, marker=".")
            axes[scheme_row, 1].plot(ns, e["delta"], color=color, linestyle=style, marker=".")
            axes[scheme_row, 2].plot(ns, e["rho_look"], color=color, linestyle=style, marker=".")
            axes[scheme_row, 2].axhline(0, color="#999999", linewidth=0.8)
            axes[scheme_row, 3].plot(ns, np.clip(e["holdout"], -2, 2), color=color, linestyle=style, marker=".")
        for axis, title in zip(axes[scheme_row], ("c_out(N) per update", "delta(N)", "rho_look(N), signed", "hold-out ratio c_out / c_in")):
            axis.set_xscale("log", base=2)
            axis.set_title(f"{title} [{scheme}]", fontsize=10)
            axis.set_xlabel("N (x 1024 envs, 64 steps)")
        axes[scheme_row, 1].set_yscale("log")
    axes[0, 0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(HERE / "partB_curves.png", dpi=120)


if __name__ == "__main__":
    main()
