#!/bin/bash
# Runs inside a Determined container: every line of a job file is one command, run PARALLEL at a time on the GPU(s).
# Usage: run_job.sh <job file relative to the repo> <parallel>
set -x
R=/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca
export HOME=$R XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_deterministic_ops=true
export JAX_COMPILATION_CACHE_DIR=$R/jax_compilation_cache PATH=$R/bb8/.venv/bin:/usr/bin:/bin
cd $R/bb8 || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv
git log --oneline -1 2>/dev/null; cat REVISION 2>/dev/null
grep -vE '^\s*(#|$)' "$1" | xargs -P "${2:-1}" -I{} bash -c '{}'
echo "job file $1 finished"
