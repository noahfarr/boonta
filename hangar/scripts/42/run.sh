#!/bin/bash
set -euo pipefail

R=/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca
CONTEXT=$(pwd)
export HOME=$R
export PATH=$R/bin:/usr/bin:/bin
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$R/jax_compilation_cache}

cd "$R/boonta"
echo "commit $(git -c safe.directory='*' rev-parse --short HEAD) run $RUN_NAME"
echo "command ${BOONTA_COMMAND:-train} args $BOONTA_ARGS"
echo "XLA_FLAGS ${XLA_FLAGS:-}"
nvidia-smi --query-gpu=index,name --format=csv,noheader || true
if [ "${LOCAL_RANK:-0}" = 0 ]; then nvidia-smi topo -m || true; fi

if [ "${WORLD_SIZE:-1}" -gt 1 ]; then
  export SLURM_JOB_ID=${DET_EXPERIMENT_ID:-4242}
  export SLURM_NTASKS=$WORLD_SIZE SLURM_STEP_NUM_TASKS=$WORLD_SIZE
  export SLURM_PROCID=$RANK SLURM_LOCALID=$LOCAL_RANK
  export SLURM_STEP_NODELIST=$MASTER_ADDR
  BOONTA_ARGS="$BOONTA_ARGS num_processes=$WORLD_SIZE"
  echo "process $RANK of $WORLD_SIZE, local $LOCAL_RANK, coordinator $MASTER_ADDR"
fi

if [ -n "${BASELINE:-}" ]; then
  # shellcheck disable=SC2086
  exec env PYTHONPATH="$R/purejaxrl-deps" ./.venv/bin/python "$CONTEXT/purejaxrl.py" $BASELINE \
    --out "$R/runs/paper/$RUN_NAME" --purejaxrl "$R/purejaxrl"
fi

# shellcheck disable=SC2086
exec ./.venv/bin/boonta "${BOONTA_COMMAND:-train}" $BOONTA_ARGS "logger=${BOONTA_LOGGER:-file}" \
  "hydra.run.dir=$R/runs/paper/$RUN_NAME" "hydra.sweep.dir=$R/runs/paper/$RUN_NAME"
