#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=outputs/smoke_spsr
rm -rf "$OUT"

echo "=== 1. Craftax Classic Symbolic (SPSR, 10M steps) ==="
.venv/bin/boonta train \
    algorithm=ppo \
    environment=craftax/craftax_classic/symbolic \
    curriculum=spsr \
    num_steps=10000000 \
    num_epochs=10 \
    seed=0 \
    "logger=[file]" \
    "artisan=[]" \
    "hydra.run.dir=$OUT/craftax_spsr"

echo "=== 2. Jumanji Pac-Man (SPSR + Probe, 10M steps) ==="
.venv/bin/boonta train \
    algorithm=ppo \
    environment=jumanji/pac_man \
    curriculum=spsr \
    probe=true \
    num_steps=10000000 \
    num_epochs=10 \
    seed=0 \
    "logger=[file]" \
    "artisan=[]" \
    "hydra.run.dir=$OUT/pacman_spsr"

echo "=== 3. Jumanji Pac-Man (Cell PSR + TransitionForecast + rho_gate, 10M steps) ==="
.venv/bin/boonta train \
    algorithm=ppo \
    environment=jumanji/pac_man \
    curriculum=psr \
    curriculum/psr/gain=transition_forecast \
    +curriculum.selector.allocation.rho_gate=true \
    probe=true \
    num_steps=10000000 \
    num_epochs=10 \
    seed=0 \
    "logger=[file]" \
    "artisan=[]" \
    "hydra.run.dir=$OUT/pacman_tforecast"

echo "=== 4. Jumanji Sokoban (SPSR, 10M steps) ==="
.venv/bin/boonta train \
    algorithm=ppo \
    environment=jumanji/sokoban \
    curriculum=spsr \
    num_steps=10000000 \
    num_epochs=10 \
    seed=0 \
    "logger=[file]" \
    "artisan=[]" \
    "hydra.run.dir=$OUT/sokoban_spsr"

echo "=== 5. Gymnax Continuous Control: MountainCarContinuous-v0 (SPSR, 5M steps) ==="
.venv/bin/boonta train \
    algorithm=ppo \
    environment=gymnax/continuous_control/mountain_car_continuous \
    curriculum=spsr \
    num_steps=5000000 \
    num_epochs=10 \
    seed=0 \
    "logger=[file]" \
    "artisan=[]" \
    "hydra.run.dir=$OUT/mcc_spsr"

.venv/bin/python - << 'EOF'
import numpy as np
from pathlib import Path

root = Path("outputs/smoke_spsr")
for d in sorted(root.iterdir()):
    p = d / "metrics.npz"
    if not p.exists():
        print(f"MISSING: {d}")
        continue
    m = np.load(p)
    print(f"\n--- {d.name} (keys={len(m.files)}) ---")
    for k in [
        "steps", "monitor/SPS", "training/episode_return", "rho/episode_return",
        "spsr/admitted_share", "spsr/v_rho", "spsr/v_max", "spsr/water_level",
        "spsr/noise_scale", "spsr/mean_multiplicity", "mu/rho_gated_share",
        "archive/placed", "archive/num_eligible", "restart/placement_length",
    ]:
        if k in m.files:
            arr = m[k].reshape(-1)
            print(f"  {k:28s}: {np.round(arr[1:], 4)}")
EOF
