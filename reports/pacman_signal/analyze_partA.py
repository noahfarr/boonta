"""Part A analysis: confirmation table with bootstrap CIs, signal-accounting tests T1, T2, T4,
T5, T6 and my P1-P3 and P7-P8, curves, feature rank and restart diagnostics.

Each log row is one epoch: the mean over its 15 updates (983,040 steps). Sums over updates
multiply an epoch's mean by 15. Writes partA.json and the figures next to this file.

Usage: uv run --no-sync python reports/pacman_signal/analyze_partA.py
"""

import glob
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SIGNAL = ROOT / "outputs" / "pacman_signal"
UPDATES_PER_LOG = 15
ARMS = {
    "ppo_shadow": ("plain PPO (shadow)", ["ppo_shadow_ckpt", "ppo_shadow"]),
    "erd": ("ERD", ["erd"]),
    "psr": ("PSR v1", ["psr"]),
    "forecast_gain": ("forecast gain", ["forecast_gain"]),
    "psr_signal": ("PSR, debiased spread", ["psr_signal"]),
}
EARLIER = {"ppo_shadow": "ppo", "erd": "erd", "psr": "psr", "forecast_gain": "forecast_gain"}
rng = np.random.default_rng(0)


def load(directory: Path) -> list:
    runs = []
    for job in sorted(directory.glob("[0-9]*")):
        path = job / "metrics.npz"
        if not path.exists():
            continue
        data = np.load(path)
        if data["steps"][-1] < 98e6:
            continue
        seed = int(OmegaConf.load(job / ".hydra" / "config.yaml").seed)
        runs.append({"seed": seed, "data": {k: data[k][:, 0] if data[k].ndim == 2 else data[k] for k in data.files}})
    return runs


def summary(run: dict) -> dict:
    d = run["data"]
    steps, ret = d["steps"], d["training/episode_return"]
    hit = np.flatnonzero(np.nan_to_num(ret) >= 300)
    return {
        "seed": run["seed"],
        "final": float(np.nanmean(ret[-10:])),
        "to_300": float(steps[hit[0]] / 1e6) if len(hit) else np.inf,
        "peak": float(np.nanmax(ret)),
        "length": float(np.nanmean(d["training/episode_length"][-10:])),
        "sps": float(np.nanmedian(d["monitor/SPS"][2:])),
    }


def bootstrap_diff(a, b, reps=20000):
    a, b = np.asarray(a), np.asarray(b)
    draws = rng.choice(a, (reps, len(a))).mean(1) - rng.choice(b, (reps, len(b))).mean(1)
    return float(a.mean() - b.mean()), [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def median_steps(values):
    values = np.asarray(values)
    m = np.median(values)
    return float(m)


def window_sums(run, width=10):
    """Per 10M-step window: sum of held-out C_gamma over its updates and the change in J_gamma.
    J at a boundary is the mean of the two epochs around it."""
    d = run["data"]
    c = np.nan_to_num(d["signal/c_gamma"]) * UPDATES_PER_LOG
    j = d["training/discounted_episode_return"]
    rows = []
    for start in range(1, len(c) - width + 1, width):
        end = start + width
        j0 = np.nanmean(j[max(start - 1, 1) : start + 1])
        j1 = np.nanmean(j[end - 1 : min(end + 1, len(j))])
        rows.append((start, float(c[start:end].sum()), float(j1 - j0)))
    return rows


def main():
    arms = {name: [r for directory in dirs for r in load(SIGNAL / directory)] for name, (_, dirs) in ARMS.items()}
    out = {"arms": {}}
    print("== Part A: final from-start return, seeds 5-19 (primary)")
    for name, runs in arms.items():
        rows = [summary(r) for r in runs]
        finals = [r["final"] for r in rows]
        out["arms"][name] = {
            "label": ARMS[name][0],
            "n": len(rows),
            "final_mean": float(np.mean(finals)),
            "final_se": float(np.std(finals, ddof=1) / np.sqrt(len(finals))),
            "finals": finals,
            "to_300_median": median_steps([r["to_300"] for r in rows]),
            "reached_300": int(np.sum(np.isfinite([r["to_300"] for r in rows]))),
            "peak_mean": float(np.mean([r["peak"] for r in rows])),
            "length_mean": float(np.mean([r["length"] for r in rows])),
            "sps_mean": float(np.mean([r["sps"] for r in rows])),
            "seeds": [r["seed"] for r in rows],
        }
        a = out["arms"][name]
        print(
            f"  {a['label']:22s} n={a['n']:2d} final {a['final_mean']:6.1f} +- {a['final_se']:4.1f} | to 300 median "
            f"{a['to_300_median']:5.1f}M ({a['reached_300']}/{a['n']}) | peak {a['peak_mean']:5.0f} | len {a['length_mean']:4.0f} | SPS {a['sps_mean']/1e6:.2f}M"
        )

    print("== Differences in final return (bootstrap 95% CI over seeds)")
    pairs = [
        ("forecast_gain", "erd"),
        ("forecast_gain", "psr"),
        ("forecast_gain", "ppo_shadow"),
        ("forecast_gain", "psr_signal"),
        ("erd", "ppo_shadow"),
        ("psr", "ppo_shadow"),
        ("psr_signal", "psr"),
        ("erd", "psr"),
    ]
    out["differences"] = {}
    for a, b in pairs:
        diff, ci = bootstrap_diff(out["arms"][a]["finals"], out["arms"][b]["finals"])
        out["differences"][f"{a}-{b}"] = {"diff": diff, "ci": ci}
        print(f"  {ARMS[a][0]:20s} - {ARMS[b][0]:22s} {diff:+6.1f}  [{ci[0]:+6.1f}, {ci[1]:+6.1f}]")

    print("== Pooled 20 seeds (secondary; seeds 0-4 from the first batch, where plain PPO was not shadow mode)")
    out["pooled"] = {}
    for name, earlier in EARLIER.items():
        old = [summary(r)["final"] for r in load(ROOT / "outputs" / "pacman_batch" / earlier)]
        pooled = out["arms"][name]["finals"] + old
        out["pooled"][name] = {"n": len(pooled), "mean": float(np.mean(pooled)), "se": float(np.std(pooled, ddof=1) / np.sqrt(len(pooled))), "finals": pooled}
        print(f"  {ARMS[name][0]:22s} n={len(pooled)} {np.mean(pooled):6.1f} +- {np.std(pooled, ddof=1)/np.sqrt(len(pooled)):4.1f}")
    for a, b in [("forecast_gain", "erd"), ("forecast_gain", "psr")]:
        diff, ci = bootstrap_diff(out["pooled"][a]["finals"], out["pooled"][b]["finals"])
        out["pooled"][f"{a}-{b}"] = {"diff": diff, "ci": ci}
        print(f"  pooled {ARMS[a][0]} - {ARMS[b][0]}: {diff:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]")

    # Shadow mode against the plain-PPO seeds 5-14 run before this round.
    plain = [summary(r)["final"] for r in load(ROOT / "outputs" / "pacman_confirm" / "ppo")]
    shadow = out["arms"]["ppo_shadow"]["finals"]
    diff, ci = bootstrap_diff(shadow, plain)
    out["shadow_check"] = {"plain_n": len(plain), "plain_mean": float(np.mean(plain)), "diff": diff, "ci": ci}
    print(f"== Shadow check: shadow {np.mean(shadow):.1f} (n={len(shadow)}) vs plain {np.mean(plain):.1f} (n={len(plain)}): {diff:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]")

    print("== T1 accounting: 10M-step windows, sum of held-out C_gamma against change in J_gamma")
    xs, ys, labels = [], [], []
    out["T1"] = {"per_arm": {}}
    for name, runs in arms.items():
        ratios = []
        for run in runs:
            for start, c, dj in window_sums(run):
                xs.append(c)
                ys.append(dj)
                labels.append((name, start))
            d = run["data"]
            total_c = float(np.nansum(d["signal/c_gamma"]) * UPDATES_PER_LOG)
            j = d["training/discounted_episode_return"]
            total_dj = float(np.nanmean(j[-2:]) - j[1])
            ratios.append(total_c / total_dj if total_dj else np.nan)
        out["T1"]["per_arm"][name] = {"ratio_median": float(np.nanmedian(ratios)), "ratio_range": [float(np.nanmin(ratios)), float(np.nanmax(ratios))], "ratios": ratios}
        print(f"  {ARMS[name][0]:22s} whole-run sum C_gamma / delta J_gamma: median {np.nanmedian(ratios):.2f}, range [{np.nanmin(ratios):.2f}, {np.nanmax(ratios):.2f}]")
    xs, ys = np.asarray(xs), np.asarray(ys)
    r = float(np.corrcoef(xs, ys)[0, 1])
    out["T1"]["pearson_r"] = r
    from scipy.stats import spearmanr

    out["T1"]["spearman_r"] = float(spearmanr(xs, ys).statistic)
    keep = xs > -50  # drops the few plain-PPO windows holding single large negative updates
    out["T1"]["pearson_r_without_outliers"] = float(np.corrcoef(xs[keep], ys[keep])[0, 1])
    out["T1"]["outliers"] = int((~keep).sum())
    print(f"  Spearman {out['T1']['spearman_r']:.3f}; Pearson without the {int((~keep).sum())} windows below -50: {out['T1']['pearson_r_without_outliers']:.3f}")
    out["T1"]["windows"] = len(xs)
    phase = {}
    for lo, hi in [(1, 11), (11, 31), (31, 61), (61, 101)]:
        m = np.asarray([lo <= s < hi for _, s in labels])
        phase[f"{lo-1}-{hi-1}M"] = {
            "r": float(np.corrcoef(xs[m], ys[m])[0, 1]) if m.sum() > 2 else np.nan,
            "sum_c_mean": float(xs[m].mean()),
            "delta_j_mean": float(ys[m].mean()),
        }
    out["T1"]["by_phase"] = phase
    print(f"  pooled Pearson r over {len(xs)} windows: {r:.3f}")
    for key, value in phase.items():
        print(f"    windows starting {key}: r {value['r']:.3f}, mean sum C {value['sum_c_mean']:.2f}, mean delta J {value['delta_j_mean']:.2f}")
    t1 = r >= 0.8 and all(0.5 <= v["ratio_median"] <= 2 for v in out["T1"]["per_arm"].values())
    out["T1"]["holds"] = bool(t1)
    print(f"  T1 holds: {t1}")

    print("== T2 budget: mean delta per update by phase")
    out["T2"] = {}
    for lo, hi in [(1, 21), (21, 61), (61, 101)]:
        means = {name: float(np.nanmean([np.nanmean(r["data"]["signal/delta"][lo:hi]) for r in runs])) for name, runs in arms.items()}
        spread = max(means.values()) / min(means.values())
        out["T2"][f"{lo-1}-{hi-1}M"] = {"means": means, "max_over_min": spread}
        print(f"  {lo-1:3d}-{hi-1:3d}M: " + ", ".join(f"{ARMS[n][0].split()[0]} {v:.5f}" for n, v in means.items()) + f" | max/min {spread:.2f}")

    print("== T4: mean c_out per update over the first 60M steps; cumulative C_gamma against final return")
    out["T4"] = {}
    for name, runs in arms.items():
        c60 = float(np.nanmean([np.nanmean(r["data"]["signal/c_out"][1:61]) for r in runs]))
        cum = float(np.mean([np.nansum(r["data"]["signal/c_gamma"]) * UPDATES_PER_LOG for r in runs]))
        out["T4"][name] = {"c_out_60M": c60, "cumulative_c_gamma": cum, "final": out["arms"][name]["final_mean"]}
        print(f"  {ARMS[name][0]:22s} c_out(0-60M) {c60:.5f} | cumulative C_gamma {cum:7.1f} | final {out['arms'][name]['final_mean']:.1f}")
    order_c = sorted(arms, key=lambda n: -out["T4"][n]["cumulative_c_gamma"])
    order_f = sorted(arms, key=lambda n: -out["T4"][n]["final"])
    out["T4"]["order_cumulative"] = order_c
    out["T4"]["order_final"] = order_f
    print(f"  order by cumulative C_gamma: {order_c}\n  order by final return:       {order_f}")

    print("== P1 delta against half Var(u); P2 brackets; P3 hold-out ratio")
    out["P"] = {}
    for name, runs in arms.items():
        d = np.concatenate([r["data"]["signal/delta"][1:] for r in runs])
        h = np.concatenate([r["data"]["signal/half_var_u"][1:] for r in runs])
        raw = np.concatenate([r["data"]["signal/s_raw"][1:] for r in runs])
        cell = np.concatenate([r["data"]["signal/s_cell"][1:] for r in runs])
        lo = np.concatenate([r["data"]["signal/rho_low"][1:] for r in runs])
        hi = np.concatenate([r["data"]["signal/rho_high"][1:] for r in runs])
        hold = np.stack([r["data"]["signal/holdout_ratio_smoothed"] for r in runs])
        out["P"][name] = {
            "delta_over_half_var": [float(np.nanpercentile(d / h, q)) for q in (5, 50, 95)],
            "s_raw_over_s_cell": [float(np.nanpercentile(raw / cell, q)) for q in (5, 50, 95)],
            "rho_low": [float(np.nanpercentile(lo, q)) for q in (5, 50, 95)],
            "rho_high": [float(np.nanpercentile(hi, q)) for q in (5, 50, 95)],
            "holdout_by_phase": {f"{a}-{b}M": float(np.nanmedian(hold[:, a + 1 : b + 1])) for a, b in [(0, 5), (5, 20), (20, 60), (60, 99)]},
        }
        p = out["P"][name]
        print(
            f"  {ARMS[name][0]:22s} delta/halfvar {p['delta_over_half_var'][1]:.3f} [{p['delta_over_half_var'][0]:.3f},{p['delta_over_half_var'][2]:.3f}] | "
            f"S_raw/S_cell {p['s_raw_over_s_cell'][1]:.0f} [{p['s_raw_over_s_cell'][0]:.0f},{p['s_raw_over_s_cell'][2]:.0f}] | rho_low {p['rho_low'][1]:.3f} | rho_high {p['rho_high'][1]:.3f} "
            f"[{p['rho_high'][0]:.3f},{p['rho_high'][2]:.3f}] | holdout " + " ".join(f"{k} {v:.2f}" for k, v in p["holdout_by_phase"].items())
        )

    print("== Feature rank (penultimate, from-start states), by phase")
    out["rank"] = {}
    for name, runs in arms.items():
        n99 = np.stack([r["data"]["rank/penultimate/n99"] for r in runs])
        ent = np.stack([r["data"]["rank/penultimate/entropy"] for r in runs])
        out["rank"][name] = {
            f"{a}-{b}M": {"n99": float(np.nanmean(n99[:, a + 1 : b + 1])), "entropy": float(np.nanmean(ent[:, a + 1 : b + 1]))}
            for a, b in [(0, 10), (10, 30), (30, 60), (60, 99)]
        }
        print(f"  {ARMS[name][0]:22s} " + " | ".join(f"{k} n99 {v['n99']:5.1f} ent {v['entropy']:4.2f}" for k, v in out["rank"][name].items()))
    runs = arms["ppo_shadow"]
    finals = np.asarray([summary(r)["final"] for r in runs])
    weak = finals <= np.sort(finals)[2]
    n99 = np.stack([r["data"]["rank/penultimate/n99"] for r in runs])
    out["rank"]["weakest_ppo"] = {
        "seeds": [runs[i]["seed"] for i in np.flatnonzero(weak)],
        "finals": finals[weak].tolist(),
        "n99_weak_by_phase": {f"{a}-{b}M": float(np.nanmean(n99[weak, a + 1 : b + 1])) for a, b in [(0, 10), (10, 30), (30, 60), (60, 99)]},
        "n99_rest_by_phase": {f"{a}-{b}M": float(np.nanmean(n99[~weak, a + 1 : b + 1])) for a, b in [(0, 10), (10, 30), (30, 60), (60, 99)]},
    }
    w = out["rank"]["weakest_ppo"]
    print(f"  weakest 3 shadow-PPO seeds {w['seeds']} (final {np.round(w['finals'])}): n99 " + " ".join(f"{k} {v:.1f}" for k, v in w["n99_weak_by_phase"].items()))
    print("  the other 12:                                     n99 " + " ".join(f"{k} {v:.1f}" for k, v in w["n99_rest_by_phase"].items()))

    print("== Restart diagnostics (restart arms), by phase")
    out["restarts"] = {}
    for name in ("erd", "psr", "forecast_gain", "psr_signal"):
        runs = arms[name]
        entry = {}
        for a, b in [(0, 10), (10, 40), (40, 99)]:
            sl = slice(a + 1, b + 1)
            buckets = [float(np.nanmean([np.nanmean(r["data"][f"restart/bucket_{k}"][sl]) for r in runs])) for k in range(7)]
            entry[f"{a}-{b}M"] = {
                "pellet_buckets": buckets,
                "placement_length": float(np.nanmean([np.nanmean(r["data"]["restart/placement_length"][sl]) for r in runs])),
                "placement_died": float(np.nanmean([np.nanmean(r["data"]["restart/placement_died"][sl]) for r in runs])),
                "placement_died_within_30": float(np.nanmean([np.nanmean(r["data"]["restart/placement_died_within_30"][sl]) for r in runs])),
                "episode_length_old_metric": float(np.nanmean([np.nanmean(r["data"]["restart/episode_length"][sl]) for r in runs])),
            }
        out["restarts"][name] = entry
        for k, v in entry.items():
            print(
                f"  {ARMS[name][0]:22s} {k:7s} pellets buckets (0-49 ... 300+) "
                + " ".join(f"{x:.2f}" for x in v["pellet_buckets"])
                + f" | placement length {v['placement_length']:5.1f}, died {v['placement_died']:.2f}, died<=30 {v['placement_died_within_30']:.2f} | old metric {v['episode_length_old_metric']:.1f}"
            )

    (HERE / "partA.json").write_text(json.dumps(out, indent=1, default=float))
    plots(arms, xs, ys, labels)


def plots(arms, xs, ys, labels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"ppo_shadow": "#7a7a7a", "erd": "#2a6fdb", "psr": "#3b9b5d", "forecast_gain": "#d9822b", "psr_signal": "#8e5cc2"}
    keys = [
        ("signal/c_out", "c_out (held-out captured signal per sample)"),
        ("signal/delta", "delta = KL(pi_t || pi_t+1)"),
        ("signal/rho_low", "rho_low (S_raw)"),
        ("signal/rho_high", "rho_high (S_cell)"),
        ("signal/holdout_ratio_smoothed", "hold-out ratio c_out / c_in (smoothed)"),
        ("training/episode_return", "from-start return"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, (key, title) in zip(axes.ravel(), keys):
        for name, runs in arms.items():
            curves = np.stack([r["data"][key] for r in runs])
            steps = runs[0]["data"]["steps"] / 1e6
            axis.plot(steps, np.nanmedian(curves, 0), color=colors[name], label=ARMS[name][0], linewidth=1.5)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("steps (M)")
        if key == "signal/delta":
            axis.set_yscale("log")
        if key == "signal/c_out":
            axis.set_yscale("symlog", linthresh=1e-4)  # c_out turns negative for many updates after ~25M
            axis.axhline(0, color="#999999", linewidth=0.8)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(HERE / "partA_signal_curves.png", dpi=120)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, runs in arms.items():
        cum = np.stack([np.cumsum(np.nan_to_num(r["data"]["signal/c_gamma"])) * UPDATES_PER_LOG for r in runs])
        j = np.stack([r["data"]["training/discounted_episode_return"] for r in runs])
        steps = runs[0]["data"]["steps"] / 1e6
        axes[0].plot(steps, np.nanmedian(cum, 0), color=colors[name], label=f"{ARMS[name][0]}: cumulative C_gamma")
        axes[0].plot(steps, np.nanmedian(j - j[:, 1:2], 0), color=colors[name], linestyle="--")
    axes[0].set_title("cumulative C_gamma (solid) vs J_gamma - J_gamma(1M) (dashed)", fontsize=10)
    axes[0].set_xlabel("steps (M)")
    axes[0].legend(fontsize=8)
    for name in arms:
        m = np.asarray([n == name for n, _ in labels])
        axes[1].scatter(xs[m], ys[m], s=10, alpha=0.6, color=colors[name], label=ARMS[name][0])
    lim = [min(xs.min(), ys.min()), max(xs.max(), ys.max())]
    axes[1].plot(lim, lim, color="#333333", linewidth=1)
    axes[1].set_xlabel("sum of held-out C_gamma over a 10M-step window")
    axes[1].set_ylabel("change in J_gamma over the window")
    axes[1].set_title("T1: windows pooled across arms and seeds")
    figure.tight_layout()
    figure.savefig(HERE / "partA_accounting.png", dpi=120)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, runs in arms.items():
        steps = runs[0]["data"]["steps"] / 1e6
        for axis, key in zip(axes, ("rank/penultimate/n99", "rank/penultimate/entropy")):
            curves = np.stack([r["data"][key] for r in runs])
            good = np.isfinite(np.nanmedian(curves, 0))
            axis.plot(steps[good], np.nanmedian(curves, 0)[good], color=colors[name], label=ARMS[name][0], marker=".")
    axes[0].set_title("penultimate n99 (directions for 99% of variance), median over seeds")
    axes[1].set_title("penultimate entropy rank, median over seeds")
    for axis in axes:
        axis.set_xlabel("steps (M)")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(HERE / "partA_rank.png", dpi=120)


if __name__ == "__main__":
    main()
