#!/bin/bash
# Submits one Determined experiment that runs a job file from the /pfss checkout. Run from the Mac.
# Usage: submit.sh <name> <job file relative to the repo> [parallel=4] [slots=1]
set -euo pipefail
name=$1; job=$2; parallel=${3:-4}; slots=${4:-1}
SCRATCH=${SCRATCH:-$(mktemp -d)}
ctx=$SCRATCH/det_ctx_$name; mkdir -p $ctx
cat > $ctx/run.sh <<RUN
exec bash /pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/bb8/experiments/bb8/det/run_job.sh $job $parallel
RUN
cat > $SCRATCH/det_exp_$name.yaml <<YAML
name: bb8-$name
workspace: RAI_WS
project: recurrent-streaming-rl
max_restarts: 0
resources: {resource_pool: 42_Compute, slots_per_trial: $slots}
searcher: {name: single, metric: episode_return, smaller_is_better: false, max_length: {batches: 1}}
bind_mounts:
  - {host_path: /pfss/mlde/workspaces/mlde_wsp_RAI_WS, container_path: /pfss/mlde/workspaces/mlde_wsp_RAI_WS, propagation: rprivate, read_only: false}
entrypoint: bash run.sh
YAML
det experiment create --project_id 338 $SCRATCH/det_exp_$name.yaml $ctx
