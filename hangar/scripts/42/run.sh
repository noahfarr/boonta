#!/bin/bash
set -euo pipefail

R=/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca
export HOME=$R
export PATH=$R/bin:/usr/bin:/bin
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$R/jax_compilation_cache}

cd "$R/boonta"
echo "commit $(git -c safe.directory='*' rev-parse --short HEAD) run $RUN_NAME"
echo "command ${BOONTA_COMMAND:-train} args $BOONTA_ARGS"
echo "XLA_FLAGS ${XLA_FLAGS:-}"
nvidia-smi --query-gpu=index,name --format=csv,noheader || true

# shellcheck disable=SC2086
exec ./.venv/bin/boonta "${BOONTA_COMMAND:-train}" $BOONTA_ARGS "logger=${BOONTA_LOGGER:-file}" \
  "hydra.run.dir=$R/runs/paper/$RUN_NAME" "hydra.sweep.dir=$R/runs/paper/$RUN_NAME"
