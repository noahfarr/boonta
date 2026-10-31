#!/bin/bash
# Bitwise check against main: probe off must equal main exactly; probe on is also compared (it must not touch training).
# Usage: bitwise_check.sh <main worktree> <scratch dir> [extra overrides]
set -uo pipefail
cd "$(dirname "$0")/../.."
MAIN=$1; OUT=$2; shift 2
for alg in ppo pqn; do
  O=(algorithm=$alg environment=gymnax/minatar/breakout environment.num_envs=256 "$@")
  PYTHONPATH=$MAIN .venv/bin/python experiments/bb8/bitwise_check.py save $OUT/main_$alg.npz "${O[@]}" 2>&1 | grep -E "^boonta|^saved"
  .venv/bin/python experiments/bb8/bitwise_check.py save $OUT/off_$alg.npz "${O[@]}" 2>&1 | grep -E "^boonta|^saved"
  .venv/bin/python experiments/bb8/bitwise_check.py save $OUT/on_$alg.npz "${O[@]}" algorithm.bb8.noise_scale_probe=true 2>&1 | grep -vE "Warning|warn"
  echo "== $alg: main vs probe off"; .venv/bin/python experiments/bb8/bitwise_check.py compare $OUT/main_$alg.npz $OUT/off_$alg.npz
  echo "== $alg: main vs probe on";  .venv/bin/python experiments/bb8/bitwise_check.py compare $OUT/main_$alg.npz $OUT/on_$alg.npz
done
