# Paper experiments on 42

These scripts run the paper's experiments on the hessian.AI 42 cluster through Determined and turn the results into the paper's tables and figures.

- [`experiments.py`](experiments.py) lists the runs, writes one Determined config per run, and submits them.
- [`run.sh`](run.sh) is the only file in the Determined context. It runs `boonta train` from the checkout on `/pfss` and writes `metrics.npz` to `/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/runs/paper/<name>`.
- [`results.py`](results.py) copies the `metrics.npz` files back, prints the scaling tables, and draws `scaling.pdf` and `podracers.pdf`.

## Runs

| Group | Runs | What it measures |
| --- | --- | --- |
| `check` | PPO and MinGRU recurrent PPO on 2 GPUs, PPO with Sebulba | that the current code trains on GPUs |
| `weak` | PPO and DQN on MinAtar Breakout, PPO on Craftax-Classic, on 1, 2, 4 and 8 GPUs | throughput with a fixed number of environments per GPU |
| `strong` | the same workloads with a fixed global batch, three seeds | speedup and wall-clock time to a target return with unchanged learning dynamics |
| `podracers` | the same PPO on Breakout with Anakin and with Sebulba, three seeds | learning curves when only the podracer changes |

Scaling runs request a whole node (8 slots) and use the first N GPUs, so no other job shares the node. The workspace limit is 24 slots, so submit one group at a time.

## Steps

1. Update the checkout on `/pfss` to `main` from any shell, since only the containers have `git`.
   ```bash
   det shell open <id> 'bash -c "cd /pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/boonta && git pull --ff-only"'
   ```
2. Submit a group, starting with `check`.
   ```bash
   python hangar/scripts/42/experiments.py submit --only check --dry-run   # print the det commands
   python hangar/scripts/42/experiments.py submit --only check
   ```
   `--seeds 0` limits the strong-scaling and podracer runs to one seed. Set `DET` to another command, such as a wrapper script, if `det` needs extra settings.
3. Copy the results back and draw the figures.
   ```bash
   RSYNC_RSH="ssh -J ias" python hangar/scripts/42/results.py runs/paper \
     --pull 42:/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/runs/paper --figures paper/figures
   ```
   Set the target returns in `TARGETS` in `results.py` once the one-GPU runs show what each workload reaches.

## Reaching the cluster from outside the TU network

The Determined master only accepts connections from the TU network. Without the VPN, forward its port through a host inside the network, such as `ias`, and point `det` at the forward.

```bash
ssh -f -N -L 18080:login01.ai.tu-darmstadt.de:8080 ias
export DET_MASTER=https://localhost:18080 DET_MASTER_CERT_FILE=noverify DET_USER=ar44ruca
export DET_USER_TOKEN=<the token for login01 in ~/Library/Application Support/determined/auth.json>
```

The login node is reachable with `ssh -J ias 42` for reading files on `/pfss`.
