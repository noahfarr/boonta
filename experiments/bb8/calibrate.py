"""Phase 4 calibration, from runs that finished before Phase 4: writes experiments/bb8/calibration.env.

    python experiments/bb8/calibrate.py --e2-tuned <dir> --e2 <dir> --e1 <phase2 dir> --step <lr|lr_sqrt|kl>

- N0 (PPO, per game): median over seeds of the median over epochs (epoch 1 left out) of the actor's fresh-update noise
  scale in E2 tuned (in-training probe at 8192 environments: the same estimator the pool uses online).
- m_max = (8192 + N0) / 8192, so that m = m_max B / (B + N0) is 1 at the tuned batch and the typical noise scale.
- KL_med (PPO, per game): median KL of the full-batch, 1× update over E1 checkpoints and seeds (adam runs; the
  `tune` test at multiplier 1 for games without a grid). KL_max = KL_med ((8192 + N0) / 8192)².
- PQN: N0 from E2 (main defaults, 4096 environments) on the TD loss; m_max = (4096 + N0) / 4096.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def n0(directory, prefix, name):
    values = []
    for path in glob.glob(f"{directory}/{prefix}/*/metrics.npz"):
        data = np.load(path)
        n = len(data["steps"])
        noise = np.nanmean(data[f"bb8/{name}/fresh/noise"].reshape(n, -1), -1)[2:]
        signal = np.nanmean(data[f"bb8/{name}/fresh/signal"].reshape(n, -1), -1)[2:]
        values.append(float(np.nanmedian(np.where(signal > 0, noise / signal, np.nan))))
    return float(np.median(values)), len(values)


def kl_med(e1, game):
    kls = []
    for path in glob.glob(f"{e1}/ppo_adam_{game}_*.json") + glob.glob(f"{e1}/klcal_{game}_*.json"):
        report = json.loads(Path(path).read_text())
        for entry in report["checkpoints"].values():
            kls += [c["kl"] for c in entry["conditions"]
                    if c["test"] in ("grid", "tune") and c["fraction"] == 1.0 and c["multiplier"] == 1.0]
    return float(np.median(kls)) if kls else float("nan"), len(kls)


def main(args):
    lines = [f"# Written by calibrate.py. STEP chosen from Phase 2: {args.step}.", f"STEP={args.step}"]
    report = {}
    for game in ("breakout", "asterix", "space_invaders"):
        noise, seeds = n0(args.e2_tuned, f"ppo_{game}", "actor")
        kl, count = kl_med(args.e1, game)
        m_max = (8192 + noise) / 8192
        lines += [f"MMAX_{game}={m_max:.4f}", f"KLMED_{game}={kl:.6f}", f"KLMAX_{game}={kl * m_max**2:.6f}"]
        report[f"ppo/{game}"] = {"N0_envs": noise, "seeds": seeds, "m_max": m_max, "kl_med": kl, "kl_updates": count,
                                 "kl_max": kl * m_max**2}
    for game in ("breakout", "asterix"):
        noise, seeds = n0(args.e2, f"pqn_{game}", "td")
        m_max = (4096 + noise) / 4096
        lines.append(f"PQN_MMAX_{game}={m_max:.4f}")
        report[f"pqn/{game}"] = {"N0_envs": noise, "seeds": seeds, "m_max": m_max}
    Path(args.output).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e2-tuned", required=True)
    parser.add_argument("--e2", required=True)
    parser.add_argument("--e1", required=True)
    parser.add_argument("--step", required=True, choices=["lr", "lr_sqrt", "kl"])
    parser.add_argument("--output", default="experiments/bb8/calibration.env")
    main(parser.parse_args())
