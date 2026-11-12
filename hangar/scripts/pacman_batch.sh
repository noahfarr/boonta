#!/bin/bash
# PacMan restart comparison: 5 seeds per arm, 100M steps each, logged every 1M steps.
#
# Arms: plain PPO, ERD, PSR, PSR with target relevance only, PSR with target gain only, and the
# oracle (target relevance and target gain). The target arms all take their checkpoint from the
# PSR run that reached 381. Not run by default: greedy (counted relevance x greedy gain, max_a A)
# and uniform_greedy (uniform relevance x greedy gain). Each arm is one multirun under $OUT/<arm>/<seed>/ with a metrics.npz.
#
# Usage: hangar/scripts/pacman_batch.sh [arm ...] [-- extra hydra overrides]
#   SEEDS (5) seeds from SEED_START (0); RANK (4096) states per batch for the feature-rank log.
#   With no arms, runs all six in order. Example: hangar/scripts/pacman_batch.sh psr oracle
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT=${CKPT:-/home/farr/boonta-psr/sweeps/ppo/jumanji/pac_man/2026-11-12_13-56-34/2}
OUT=${OUT:-outputs/pacman_batch}
SEEDS=${SEEDS:-5}
SEED_START=${SEED_START:-0}
RANK=${RANK:-4096}  # states per batch for the feature-rank log (0 turns it off)

arms=()
while (($#)) && [[ $1 != -- ]]; do arms+=("$1"); shift; done
[[ ${1:-} == -- ]] && shift
extra=("$@")
((${#arms[@]})) || arms=(ppo erd psr relevance_only gain_only oracle)

base=(
    algorithm=ppo
    environment=jumanji/pac_man
    num_steps=100000000
    num_epochs=100
    "seed=range($SEED_START,$((SEED_START + SEEDS)))"
    "algorithm.feature_rank=$RANK"
    "logger=[file,wandb]"
    "artisan=[]"
)

run() {
    local arm=$1
    shift
    echo "=== $arm: seeds $SEED_START..$((SEED_START + SEEDS - 1)) -> $OUT/$arm"
    .venv/bin/boonta train -m "${base[@]}" "$@" "${extra[@]}" "hydra.sweep.dir=$OUT/$arm"
}

for arm in "${arms[@]}"; do
    case $arm in
        ppo) run ppo curriculum=default ;;
        erd) run erd curriculum=erd ;;
        psr) run psr curriculum=psr ;;
        relevance_only)
            run relevance_only curriculum=psr curriculum/psr/relevance=target \
                "curriculum.selector.relevance.checkpoint=$CKPT" ;;
        gain_only)
            run gain_only curriculum=psr curriculum/psr/gain=target \
                "curriculum.selector.gain.checkpoint=$CKPT" ;;
        oracle)
            run oracle curriculum=psr curriculum/psr/relevance=target curriculum/psr/gain=target \
                "curriculum.selector.relevance.checkpoint=$CKPT" ;;
        greedy) run greedy curriculum=psr curriculum/psr/gain=greedy_advantage ;;
        uniform_greedy)
            run uniform_greedy curriculum=psr curriculum/psr/relevance=uniform \
                curriculum/psr/gain=greedy_advantage ;;
        double_greedy)
            run double_greedy curriculum=psr curriculum/psr/gain=double_greedy_advantage \
                "curriculum.selector.gain.checkpoint=$CKPT" ;;
        search_gain) run search_gain curriculum=psr curriculum/psr/gain=search ;;
        search_full)
            run search_full curriculum=psr curriculum/psr/relevance=search curriculum/psr/gain=search ;;
        hindsight) run hindsight curriculum=psr curriculum/psr/gain=hindsight ;;
        expectile) run expectile curriculum=psr curriculum/psr/gain=expectile ;;
        forecast_gain) run forecast_gain curriculum=psr curriculum/psr/gain=forecast ;;
        forecast_full)
            run forecast_full curriculum=psr curriculum/psr/relevance=forecast curriculum/psr/gain=forecast ;;
        ppo_ln) run ppo_ln curriculum=default network.layer_norm=true ;;
        erd_ln) run erd_ln curriculum=erd network.layer_norm=true ;;
        forecast_gain_ln) run forecast_gain_ln curriculum=psr curriculum/psr/gain=forecast network.layer_norm=true ;;
        screen)
            run screen curriculum=psr curriculum/psr/gain=fair_screen \
                "curriculum.selector.gain.checkpoint=$CKPT" ;;
        *) echo "unknown arm: $arm (ppo erd psr relevance_only gain_only oracle greedy uniform_greedy double_greedy search_gain search_full hindsight expectile forecast_gain forecast_full ppo_ln erd_ln forecast_gain_ln screen)" >&2; exit 1 ;;
    esac
done
