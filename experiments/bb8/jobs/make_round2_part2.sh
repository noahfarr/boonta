#!/bin/bash
# Round 2, Part 2: are batch size and learning rate interchangeable? 8 seeds (0-7). Writes jobs/r2p2.txt. Run with bash.
# Convention from Part 1 (arm A): plain Anakin, num_envs = B, num_minibatches fixed at 16 (minibatches shrink with B),
# batch advantage normalization, tuned epochs; PPO with the entropy stop and update-KL logging.
# lr = lr_ref * factor; rules: sqrt (factor = sqrt(B / B_ref)), linear (B / B_ref), unscaled (1).
# Reused from Part 1 (identical settings, seeds 0-7, symlinked): PPO A_1024 = B1024_unscaled, A_8192 = B8192_ref,
# PQN default = B4096_ref.
cd "$(dirname "$0")/.." && source budgets.sh
R=/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca; O=$R/bb8_runs/r2p2; P1=$R/bb8_runs/r2p1
PPO="algorithm=ppo $TUNED $TUNED_BUDGET +stop_entropy_fraction=0.01 algorithm.bb8.log_update_kl=true logger=file"
PQN="algorithm=pqn $PQN_BUDGET logger=file"
factor() { /usr/bin/python3 -c "import math; b,r,rule=$1,$2,'$3'; print({'sqrt': math.sqrt(b/r), 'linear': b/r, 'unscaled': 1.0}[rule])"; }
line() { # game arm seed overrides...
  local game=$1 arm=$2 seed=$3; shift 3
  local d=$O/$game/$arm/$seed
  echo "mkdir -p $d && { grep -q 'time  ' $d/train.log 2>/dev/null || [ -f $d/stalled.json ] || boonta train environment=gymnax/minatar/${game#pqn_} seed=$seed $* hydra.run.dir=$d > $d/train.log 2>&1; }"
}
{
echo "# Round 2 Part 2 (8 seeds). Reused Part 1 runs are symlinked first."
for s in $(seq 0 7); do
  for g in breakout asterix; do
    echo "mkdir -p $O/$g/B1024_unscaled $O/$g/B8192_ref && ln -sfn $P1/$g/A_1024/$s $O/$g/B1024_unscaled/$s && ln -sfn $P1/$g/A_8192/$s $O/$g/B8192_ref/$s"
  done
  echo "mkdir -p $O/pqn_breakout/B4096_ref && ln -sfn $P1/pqn_breakout/default/$s $O/pqn_breakout/B4096_ref/$s"
done
for s in $(seq 0 7); do
  for g in breakout asterix; do
    for b in 1024 2048 4096; do for rule in sqrt linear unscaled; do
      [ $b = 1024 ] && [ $rule = unscaled ] && continue
      f=$(factor $b 8192 $rule)
      line $g B${b}_$rule $s $PPO environment.num_envs=$b optimizer.lr=$(/usr/bin/python3 -c "print(2.4e-3*$f)")
    done; done
  done
  for b in 1024 2048 8192; do for rule in sqrt linear unscaled; do
    f=$(factor $b 4096 $rule)
    line pqn_breakout B${b}_$rule $s $PQN environment.num_envs=$b optimizer.lr=$(/usr/bin/python3 -c "print(5e-4*$f)")
  done; done
done
} > jobs/r2p2.txt
