#!/bin/bash
# Writes jobs/phase2_<part>.txt for E1. Run with bash. Parts: a (PPO Breakout + PQN), b (PPO Asterix + optimizer
# tuning on Space Invaders), c (SGD and Lion on Breakout and Asterix; needs the tuned base learning rates as $2 $3).
cd "$(dirname "$0")/.."
R=/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca; O=$R/bb8_runs/phase2; S="python experiments/bb8/one_update.py"
PPO_STEPS="2097152 7864320 15990784"   # epochs 8, 30, 61 of 76: 10.5%, 39.5%, 80.3%
PQN_STEP=31457280                      # epoch 24 of 61: 39.3%
ppo() { echo "mkdir -p $O && $S $R/bb8_runs/e2_tuned/ppo_$1/$2 --steps $PPO_STEPS --output $O/ppo_adam_$1_$2.json > $O/ppo_adam_$1_$2.log 2>&1"; }
case $1 in
a) { for s in 0 1 2; do ppo breakout $s; done
     for g in breakout asterix; do for s in 0 1 2; do echo "mkdir -p $O && $S $R/bb8_runs/e2/pqn_$g/$s --steps $PQN_STEP --tests grid --output $O/pqn_${g}_$s.json > $O/pqn_${g}_$s.log 2>&1"; done; done; } > jobs/phase2_a.txt ;;
b) { for s in 0 1 2; do ppo asterix $s; done
     echo "mkdir -p $O && $S $R/bb8_runs/e2_tuned/ppo_space_invaders/0 --steps 7864320 --tests tune --seeds 2 --optimizer sgd --base-lr 0.1 --tune-multipliers 0.01 0.03 0.1 0.3 1 3 10 --output $O/tune_sgd.json > $O/tune_sgd.log 2>&1"
     echo "mkdir -p $O && $S $R/bb8_runs/e2_tuned/ppo_space_invaders/0 --steps 7864320 --tests tune --seeds 2 --optimizer lion --base-lr 1e-4 --tune-multipliers 0.03 0.1 0.3 1 3 10 30 --output $O/tune_lion.json > $O/tune_lion.log 2>&1"
   } > jobs/phase2_b.txt ;;
c) { for g in breakout asterix; do for s in 0 1 2; do
       echo "mkdir -p $O && $S $R/bb8_runs/e2_tuned/ppo_$g/$s --steps 7864320 --tests grid --optimizer sgd --base-lr $2 --output $O/ppo_sgd_${g}_$s.json > $O/ppo_sgd_${g}_$s.log 2>&1"
       echo "mkdir -p $O && $S $R/bb8_runs/e2_tuned/ppo_$g/$s --steps 7864320 --tests grid --optimizer lion --base-lr $3 --output $O/ppo_lion_${g}_$s.json > $O/ppo_lion_${g}_$s.log 2>&1"
     done; done
     # Grid extension below 0.125x where the best step sat on the grid's lower edge (PPO 40% and 80%, PQN 40%).
     for g in breakout asterix; do for s in 0 1 2; do
       echo "mkdir -p $O && $S $R/bb8_runs/e2_tuned/ppo_$g/$s --steps 7864320 15990784 --tests grid --multipliers 0.015625 0.03125 0.0625 --output $O/ppo_adam_${g}_$s.ext.json > $O/ppo_adam_${g}_$s.ext.log 2>&1"
       echo "mkdir -p $O && $S $R/bb8_runs/e2/pqn_$g/$s --steps $PQN_STEP --tests grid --multipliers 0.015625 0.03125 0.0625 --output $O/pqn_${g}_$s.ext.json > $O/pqn_${g}_$s.ext.log 2>&1"
     done; done
     # KL of the full-batch 1x update on Space Invaders (calibration of KL_max; Breakout and Asterix have the grid).
     for s in 0 1 2; do
       echo "mkdir -p $O && $S $R/bb8_runs/e2_tuned/ppo_space_invaders/$s --steps $PPO_STEPS --tests tune --tune-multipliers 1 --seeds 2 --output $O/klcal_space_invaders_$s.json > $O/klcal_space_invaders_$s.log 2>&1"
     done; } > jobs/phase2_c.txt ;;
esac
