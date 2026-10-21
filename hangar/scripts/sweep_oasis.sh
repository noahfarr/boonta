#!/usr/bin/env bash
set -euo pipefail

trials=${1:-150}
steps=${2:-1_000_000_000}

cd "$(dirname "$0")/../.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exec "${VENV:-.venv}/bin/boonta" train -m \
    hydra/sweeper=carbs hydra/launcher=submitit_local \
    algorithm=recurrent_ippo environment=dune_sea/oasis environment.auto_reset.ratio=32 \
    num_steps="$steps" scoring=final logger=file artisan=[] \
    hydra.sweeper.n_trials="$trials" hydra.sweeper.n_jobs=1 \
    hydra.sweep.dir=sweeps/oasis
