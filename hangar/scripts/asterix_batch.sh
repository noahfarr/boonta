#!/bin/bash
# MinAtar Asterix: plain PPO (shadow mode), ERD, PSR v1 and forecast gain, 15 seeds each, 100M steps,
# with the signal probe on. PPO settings are those of the earlier Asterix runs (Breakout's tuned
# settings); see reports/asterix/PREDICTIONS.md.
#
# Usage: hangar/scripts/asterix_batch.sh [arm ...]   (no arms: all four)
#   OUT (outputs/asterix), SEED_START (0), SEEDS (15).
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=${OUT:-outputs/asterix}
SEED_START=${SEED_START:-0}
SEEDS=${SEEDS:-15}
SEED_RANGE="range($SEED_START,$((SEED_START + SEEDS)))"

arms=("$@")
((${#arms[@]})) || arms=(ppo_shadow erd psr forecast_gain)

TUNED=(
    algorithm.num_minibatches=16 algorithm.update_epochs=4 algorithm.clip_coefficient=0.14
    algorithm.clip_value_loss=True algorithm.entropy_coefficient=0.003 algorithm.value_coefficient=0.5
    algorithm.gamma=0.989 algorithm.gae_lambda=0.77 algorithm.normalize_advantage=True
    podracer.config.num_steps=16 environment.num_envs=8192 optimizer.lr=2.4e-3 optimizer.max_grad_norm=1.5
)
base=(
    algorithm=ppo
    environment=gymnax/minatar/asterix
    num_steps=100000000
    num_epochs=50
    +probe=true
    "seed=$SEED_RANGE"
    "logger=[file]"
    "artisan=[]"
    "${TUNED[@]}"
)

run() {
    local arm=$1
    shift
    echo "=== $arm: seeds $SEED_RANGE -> $OUT/$arm"
    .venv/bin/boonta train -m "${base[@]}" "$@" "hydra.sweep.dir=$OUT/$arm"
}

for arm in "${arms[@]}"; do
    case $arm in
        ppo_shadow) run ppo_shadow curriculum=psr curriculum.selector.k=1.0 ;;
        erd) run erd curriculum=erd ;;
        psr) run psr curriculum=psr ;;
        forecast_gain) run forecast_gain curriculum=psr curriculum/psr/gain=forecast ;;
        *) echo "unknown arm: $arm (ppo_shadow erd psr forecast_gain)" >&2; exit 1 ;;
    esac
done
