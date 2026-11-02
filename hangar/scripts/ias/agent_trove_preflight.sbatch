#!/usr/bin/env bash
#SBATCH --job-name=blm-preflight
#SBATCH --partition=main
#SBATCH --qos=student
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=blm-preflight-%j.log
set -euo pipefail
: "${BOONTA_ROOT:?Set your isolated experiment checkout}"
: "${AGENT_TROVE_CACHE:?Set the complete token cache path}"
: "${QWEN3_TOKENIZER:?Set the tokenizer JSON path}"
: "${AGENT_TROVE_SPLIT:?Set the grouped split.json path}"
objective="${1:?Choose clm or blm}"
case "$objective" in
  clm) assistant_only=false ;;
  blm) assistant_only=true ;;
  *) exit 2 ;;
esac
cd "$BOONTA_ROOT"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_ORDER=PCI_BUS_ID
run_dir="outputs/preflight-${objective}-${SLURM_JOB_ID}"
nvidia-smi
.venv/bin/boonta train \
  algorithm=recurrent_bc environment=agent_trove \
  hyperparameters=recurrent_bc/agent_trove_preflight \
  scoring=final logger=file seed=42 \
  dataset.kwargs.cache_dir="$AGENT_TROVE_CACHE" \
  dataset.kwargs.tokenizer_id="$QWEN3_TOKENIZER" \
  dataset.kwargs.split_manifest="$AGENT_TROVE_SPLIT" \
  dataset.kwargs.assistant_only="$assistant_only" \
  hydra.run.dir="$run_dir"
