#!/bin/bash
# Phase 3 check: the pool with BB-8 off and the bucket fixed at the default N equals plain Anakin, PPO (tuned) and PQN.
# Usage: bitwise_pool.sh <scratch dir> [num_envs=256]
cd "$(dirname "$0")/../.."
OUT=$1; N=${2:-256}
for alg in ppo pqn; do
  O=(algorithm=$alg environment=gymnax/minatar/breakout environment.num_envs=$N)
  [ $alg = ppo ] && O+=(hyperparameters=ppo/gymnax/minatar/bb8_tuned environment.num_envs=$N)
  P=(podracer=bb8_pool "podracer.config.buckets=[$((N/4)),$((N/2)),$N]" podracer.config.fixed_bucket=$N)
  .venv/bin/python experiments/bb8/bitwise_check.py save $OUT/anakin_$alg.npz "${O[@]}" 2>&1 | grep -E "^saved"
  .venv/bin/python experiments/bb8/bitwise_check.py save $OUT/pool_$alg.npz "${O[@]}" "${P[@]}" 2>&1 | grep -E "^saved|Error"
  echo "== $alg: anakin vs pool (fixed default bucket, BB-8 off)"
  .venv/bin/python experiments/bb8/bitwise_check.py compare $OUT/anakin_$alg.npz $OUT/pool_$alg.npz
done
