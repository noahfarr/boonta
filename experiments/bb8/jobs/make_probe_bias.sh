#!/bin/bash
# Writes jobs/probe_bias.txt. Run with bash (zsh does not word-split).
cd "$(dirname "$0")/.."
R=/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca; O=$R/bb8_runs/probe_bias
P="mkdir -p $O && cd experiments/bb8 && python probe_bias.py"
{
echo "# Probe bias v2 (JVP fresh, trust region): warm-up sensitivity, then B-dependence with desynchronized environments."
for wn in "300 0" "1000 0" "3000 0" "300 1000" "1000 1000"; do
  set -- $wn
  echo "$P $R/bb8_runs/e2_tuned/ppo_breakout/0 --step 7864320 --rollouts 16 --warmup $1 --noops $2 --sizes 256 1024 8192 --output $O/breakout_0_7864320_w$1_n$2.json > $O/breakout_0_7864320_w$1_n$2.log 2>&1"
done
for spec in "breakout 0 2097152" "asterix 0 7864320" "space_invaders 0 7864320"; do
  set -- $spec
  echo "$P $R/bb8_runs/e2_tuned/ppo_$1/$2 --step $3 --rollouts 16 --warmup 1000 --noops 1000 --sizes 256 512 1024 2048 8192 --output $O/$1_$2_$3_w1000_n1000.json > $O/$1_$2_$3_w1000_n1000.log 2>&1"
done
} > jobs/probe_bias.txt
