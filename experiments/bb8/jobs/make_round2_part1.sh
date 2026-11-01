#!/bin/bash
# Round 2, Part 1: reconcile with prd Part D. Writes jobs/r2p1_{a,b}.txt (two GPUs). Run with bash.
# A: plain Anakin, num_minibatches 16 (minibatches shrink with the batch), batch advantage normalization, tuned epochs.
# B: as A, fixed minibatch size (8192 transitions: 2 minibatches at 1024; B@8192 is A@8192).
# C: as A, running advantage normalization.
# D: pool podracer (8192 states, round-robin), fixed bucket 1024, 16 minibatches, batch normalization.
# Also the threshold references for Part 3: tuned fixed on Space Invaders, PQN default on Breakout and Asterix.
cd "$(dirname "$0")/.." && source budgets.sh
R=/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca; O=$R/bb8_runs/r2p1
COMMON="algorithm=ppo $TUNED $TUNED_BUDGET +stop_entropy_fraction=0.01 algorithm.bb8.log_update_kl=true logger=file"
line() { # game arm seed overrides...
  local game=$1 arm=$2 seed=$3; shift 3
  local d=$O/$game/$arm/$seed
  echo "mkdir -p $d && { grep -q 'time  ' $d/train.log 2>/dev/null || [ -f $d/stalled.json ] || boonta train environment=gymnax/minatar/$game seed=$seed $* hydra.run.dir=$d > $d/train.log 2>&1; }"
}
ppo_arms() { # game seed
  line $1 A_1024 $2 $COMMON environment.num_envs=1024
  line $1 A_8192 $2 $COMMON environment.num_envs=8192
  line $1 B_1024 $2 $COMMON environment.num_envs=1024 algorithm.num_minibatches=2
  line $1 C_1024 $2 $COMMON environment.num_envs=1024 algorithm.bb8.running_advantage_normalization=true
  line $1 C_8192 $2 $COMMON environment.num_envs=8192 algorithm.bb8.running_advantage_normalization=true
  line $1 D_1024 $2 $COMMON environment.num_envs=8192 podracer=bb8_pool podracer.config.schedule=fixed podracer.config.fixed_bucket=1024 podracer.config.minibatch_envs=64 podracer.config.step=none podracer.config.running_normalization_below=0
}
{ echo "# Round 2 Part 1, GPU a: PPO Breakout (6 arms x 16 seeds), PPO Space Invaders reference (16 seeds)."
  for s in $(seq 0 15); do ppo_arms breakout $s; done
  for s in $(seq 0 15); do line space_invaders A_8192 $s $COMMON environment.num_envs=8192; done
} > jobs/r2p1_a.txt
{ echo "# Round 2 Part 1, GPU b: PPO Asterix (6 arms x 16 seeds), PQN default references on Breakout and Asterix (16 seeds)."
  for s in $(seq 0 15); do ppo_arms asterix $s; done
  for g in breakout asterix; do for s in $(seq 0 15); do
    d=$O/pqn_$g/default/$s
    echo "mkdir -p $d && { grep -q 'time  ' $d/train.log 2>/dev/null || boonta train algorithm=pqn environment=gymnax/minatar/$g seed=$s $PQN_BUDGET logger=file hydra.run.dir=$d > $d/train.log 2>&1; }"
  done; done
} > jobs/r2p1_b.txt
