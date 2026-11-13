"""The forecast gain's per-placement pellet-bucket schedule, from the reruns with per-placement
logging (outputs/pacman_signal/forecast_gain_placement). Each epoch's shares are the mean over
seeds of restart/placement_bucket_k; the first log row, before any restart, copies the second.
Writes forecast_bucket_schedule_placement.json next to this file and prints it beside the
per-event schedule from Part A.
"""
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
runs = sorted(glob.glob(str(HERE.parents[1] / "outputs/pacman_signal/forecast_gain_placement/[0-9]*/metrics.npz")))
rows = []
for f in runs:
    d = np.load(f)
    rows.append(np.stack([d[f"restart/placement_bucket_{k}"][:, 0] for k in range(7)], 1))
shares = np.nanmean(np.stack(rows), 0)
shares[0] = shares[1]
shares = np.where(np.isfinite(shares), shares, 1 / 7)
shares = shares / shares.sum(1, keepdims=True)
(HERE / "forecast_bucket_schedule_placement.json").write_text(
    json.dumps({"epoch_steps": 983040, "bucket_pellets": 50, "seeds": len(runs), "shares": shares.round(5).tolist()})
)
old = np.asarray(json.loads((HERE / "forecast_bucket_schedule.json").read_text())["shares"])
print(f"{len(runs)} seeds")
for e in (1, 5, 10, 20, 40, 60, 80, 99):
    print(f"epoch {e:3d} per placement {np.round(shares[e], 2)} | per event (Part A) {np.round(old[e], 2)}")
