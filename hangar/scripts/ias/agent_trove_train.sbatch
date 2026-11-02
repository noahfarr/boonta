#!/usr/bin/env bash
#SBATCH --job-name=agent-trove-pair
#SBATCH --partition=main
#SBATCH --qos=student
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-12:00:00
#SBATCH --no-requeue
#SBATCH --output=agent-trove-%j.log
set -euo pipefail
: "${BOONTA_ROOT:?Set your isolated checkout}"
: "${AGENT_TROVE_CACHE:?Set cache path}"
: "${QWEN3_TOKENIZER:?Set tokenizer JSON path}"
: "${AGENT_TROVE_SPLIT:?Set split.json path}"
: "${PREFLIGHT_RUN:?Set the completed matching preflight run directory}"
stage="${1:?pilot or main}"; objective="${2:?clm or blm}"; seed="${3:-42}"
case "$stage" in pilot|main) ;; *) exit 2 ;; esac
case "$objective" in clm|blm) ;; *) exit 2 ;; esac
cd "$BOONTA_ROOT"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_ORDER=PCI_BUS_ID
args=(--cache "$AGENT_TROVE_CACHE" --tokenizer "$QWEN3_TOKENIZER" --split-manifest "$AGENT_TROVE_SPLIT"
      --stage "$stage" --objective "$objective" --seed "$seed" --preflight "$PREFLIGHT_RUN"
      --run-dir "outputs/${stage}-${objective}-seed${seed}-${SLURM_JOB_ID}")
# Explicit operator switches. No job submits, automatic chaining or requeueing.
if [[ "${ENGINEERING_ONLY:-0}" == 1 ]]; then args+=(--engineering-only); fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then args+=(--resume "$RESUME_CHECKPOINT"); fi
if [[ -n "${PILOT_REPORT:-}" ]]; then args+=(--pilot-report "$PILOT_REPORT"); fi
if [[ -n "${RL_READINESS:-}" ]]; then args+=(--rl-readiness "$RL_READINESS"); fi
.venv/bin/python -m hangar.scripts.run_agent_trove "${args[@]}" --execute
