#!/bin/bash
# Part A of the signal-accounting round: five arms on fresh seeds 5-19, 100M steps each, with the
# signal probe (boonta/podracers/probe.py) logging every update. Configs as in the 5-seed batch.
#
# Arms: ppo_shadow (PSR with k=1: nothing placed, the cell ledger exists), erd, psr (v1:
# counted relevance x advantage gain), forecast_gain (kappa 1, horizon 10), psr_signal (gain =
# Advantage.signal, everything else as psr). Shadow-PPO seeds 5 and 6 also save a full checkpoint
# (parameters, Adam state, critic) every epoch for the Part B batch-size probes.
#
# Usage: hangar/scripts/pacman_signal.sh [arm ...]   (no arms: all five, in the order above)
#   OUT (outputs/pacman_signal), SEED_START (5), SEEDS (15).
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=${OUT:-outputs/pacman_signal}
SEED_START=${SEED_START:-5}
SEEDS=${SEEDS:-15}
LAST=$((SEED_START + SEEDS))

arms=("$@")
((${#arms[@]})) || arms=(ppo_shadow erd psr forecast_gain psr_signal)

base=(
    algorithm=ppo
    environment=jumanji/pac_man
    num_steps=100000000
    num_epochs=100
    probe=true
)

run() {
    local dir=$1 seeds=$2
    shift 2
    echo "=== $dir: seeds $seeds"
    .venv/bin/boonta train -m "${base[@]}" "seed=$seeds" "$@" "hydra.sweep.dir=$OUT/$dir"
}

for arm in "${arms[@]}"; do
    case $arm in
        ppo_shadow)
            # Seeds 5 and 6 keep every epoch's checkpoint for Part B; the rest log metrics only.
            run ppo_shadow_ckpt "range($SEED_START,$((SEED_START + 2)))" curriculum=psr curriculum.selector.k=1.0 \
                "logger=[file,orbax]" "artisan=[checkpoint]"
            run ppo_shadow "range($((SEED_START + 2)),$LAST)" curriculum=psr curriculum.selector.k=1.0 \
                "logger=[file]" "artisan=[]" ;;
        erd) run erd "range($SEED_START,$LAST)" curriculum=erd "logger=[file]" "artisan=[]" ;;
        psr) run psr "range($SEED_START,$LAST)" curriculum=psr "logger=[file]" "artisan=[]" ;;
        forecast_gain)
            run forecast_gain "range($SEED_START,$LAST)" curriculum=psr curriculum/psr/gain=forecast \
                "logger=[file]" "artisan=[]" ;;
        psr_signal)
            run psr_signal "range($SEED_START,$LAST)" curriculum=psr curriculum/psr/gain=signal \
                "logger=[file]" "artisan=[]" ;;
        *) echo "unknown arm: $arm (ppo_shadow erd psr forecast_gain psr_signal)" >&2; exit 1 ;;
    esac
done
