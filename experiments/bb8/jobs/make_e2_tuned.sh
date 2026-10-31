#!/bin/bash
# Writes jobs/e2_tuned.txt: E2 for PPO under the tuned config. Run with bash.
cd "$(dirname "$0")/.." && source budgets.sh
O=/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/bb8_runs/e2_tuned
{
echo "# E2 tuned: PPO tuned config, probe on, checkpoints every epoch, seeds 0-3."
for g in $GAMES4; do for s in 0 1 2 3; do d=$O/ppo_$g/$s; echo "mkdir -p $d && boonta train algorithm=ppo environment=gymnax/minatar/$g $TUNED $TUNED_BUDGET seed=$s logger=[file,orbax] algorithm.bb8.noise_scale_probe=true hydra.run.dir=$d > $d/train.log 2>&1"; done; done
} > jobs/e2_tuned.txt
