"""Within-arm placement: does a seed's share of late-game restarts predict its final return?

Late game = restarts with fewer than 150 pellets left (buckets 0-2 of 0-49 ... 300+). Shares are
averaged over a phase of training per seed; final return is the mean of the last 10 logs.
"""
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
ARMS = {"erd": "ERD", "psr": "PSR v1", "forecast_gain": "forecast gain", "psr_signal": "debiased spread"}
PHASES = {"0-40M": (1, 41), "40-100M": (41, 101), "whole run": (1, 101)}


def seeds(arm):
    for job in sorted((ROOT / "outputs" / "pacman_signal" / arm).glob("[0-9]*")):
        d = np.load(job / "metrics.npz")
        yield {k: d[k][:, 0] if d[k].ndim == 2 else d[k] for k in d.files}


out = {}
pooled = {phase: ([], []) for phase in PHASES}
for arm, label in ARMS.items():
    runs = list(seeds(arm))
    final = np.array([np.nanmean(r["training/episode_return"][-10:]) for r in runs])
    out[arm] = {}
    for phase, (a, b) in PHASES.items():
        late = np.array([np.nanmean(sum(r[f"restart/bucket_{k}"][a:b] for k in range(3))) for r in runs])
        frontier = np.array([
            np.nanmean(sum(k * r[f"restart/bucket_{k}"][a:b] for k in range(7))) for r in runs
        ])  # mean bucket index of restarts: lower = later in the game
        rs, ps = spearmanr(late, final)
        rp, pp = pearsonr(late, final)
        out[arm][phase] = {"late_share_mean": float(late.mean()), "late_share_range": [float(late.min()), float(late.max())],
                           "spearman": float(rs), "spearman_p": float(ps), "pearson": float(rp), "pearson_p": float(pp),
                           "mean_bucket_spearman": float(spearmanr(frontier, final).statistic)}
        pooled[phase][0].extend(late - late.mean())
        pooled[phase][1].extend(final - final.mean())
        print(f"{label:16s} {phase:9s} late share {late.mean():.2f} [{late.min():.2f}, {late.max():.2f}] | "
              f"Spearman with final {rs:+.2f} (p {ps:.2f}), Pearson {rp:+.2f} (p {pp:.2f}) | mean bucket vs final {out[arm][phase]['mean_bucket_spearman']:+.2f}")
for phase, (x, y) in pooled.items():
    r, p = pearsonr(x, y)
    out[f"pooled_within_arm_{phase}"] = {"pearson": float(r), "p": float(p), "n": len(x)}
    print(f"pooled within arms (arm means removed), {phase}: Pearson {r:+.2f} (p {p:.3f}, n {len(x)})")
(Path(__file__).parent / "placement.json").write_text(json.dumps(out, indent=1))
