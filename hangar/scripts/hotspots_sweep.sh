#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
unset $(compgen -e | grep "^SLURM_") || true
.venv/bin/boonta train -m \
    hydra/launcher=submitit/ias \
    hydra.launcher.array_parallelism=8 \
    hydra.launcher.timeout_min=60 \
    "hydra.sweep.dir=sweeps/hotspots_concentration/\${now:%Y-%m-%d_%H-%M-%S}" \
    algorithm=ppo \
    environment=dune_sea/hotspots \
    curriculum=default,prd,prd_untargeted \
    environment.kwargs.nearest=2,5,8,11 \
    "environment.kwargs.farthest='\${eval:\"\${environment.kwargs.nearest} + 3\"}'" \
    environment.kwargs.num_hotspots=8,32 \
    "seed=range(10)" \
    num_epochs=50 \
    "artisan=[hotspots,concentration]" \
    logger=file \
    "$@"
