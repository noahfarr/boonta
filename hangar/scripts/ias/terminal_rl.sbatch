#!/usr/bin/env bash
#SBATCH --job-name=terminal-grpo
#SBATCH --partition=main
#SBATCH --qos=student
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-12:00:00
#SBATCH --no-requeue
#SBATCH --output=terminal-grpo-%j.log
set -euo pipefail
: "${BOONTA_ROOT:?Set checkout}"
: "${QWEN3_TOKENIZER:?Set tokenizer JSON}"
: "${PRETRAINING_CHECKPOINT:?Set one explicit Boonta checkpoint}"
condition="${1:?clm or blm}"; seed="${2:-42}"
cd "$BOONTA_ROOT"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_ORDER=PCI_BUS_ID
.venv/bin/python -m hangar.scripts.run_terminal_rl --checkpoint "$PRETRAINING_CHECKPOINT" \
 --tokenizer "$QWEN3_TOKENIZER" --condition "$condition" --seed "$seed" \
 --budget "${RL_BUDGET:-100000}" --evaluation-interval "${RL_EVAL_INTERVAL:-5000}" \
 --run-dir "outputs/rl-${condition}-seed${seed}-${SLURM_JOB_ID}" --execute
