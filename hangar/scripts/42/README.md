# Paper experiments on 42

These scripts run the paper's experiments on the hessian.AI 42 cluster through Determined and turn the results into the paper's tables and figures.

- [`experiments.py`](experiments.py) lists the runs, writes one Determined config per run, and submits them.
- [`run.sh`](run.sh) is the only file in the Determined context. It runs `boonta train` from the checkout on `/pfss` and writes `metrics.npz` to `/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/runs/paper/<name>`.
- [`results.py`](results.py) copies the `metrics.npz` files back, prints the scaling tables, and draws `scaling.pdf` and `podracers.pdf`.

## Runs

| Group | Paper | Runs | Seeds | GPUs per run |
| --- | --- | --- | --- | --- |
| `check` | none | PPO and MinGRU recurrent PPO on 2 GPUs, PPO with Sebulba | 1 | 2 |
| `podracers` | Figure 2 | the same PPO on MinAtar Breakout with Anakin and with Sebulba | 10 | 1 and 2 |
| `weak` | Figure 3, left | PPO on Breakout, PPO on Craftax-Classic and DQN on Breakout on 1, 2, 4 and 8 GPUs, with fixed work per GPU | 1 | 8 (whole node) |
| `strong` | Figure 3, right | the same workloads with a fixed global batch | 3 | 8 (whole node) |
| `autoreset` | Appendix C | PPO on Breakout with same-step, next-step and optimistic auto-reset | 10 | 1 |
| `reproduction` | Table 5 | PPO, DQN and PQN on five MinAtar games, SAC on five Brax MuJoCo tasks, BC and IQL on three Minari expert datasets, IPPO and MAPPO on four SMAX scenarios | 10 on MinAtar, 5 otherwise | 1 |
| `curricula` | Section 5, curricula | recurrent PPO on JaxUED mazes with PLR and with domain randomization, and PLR with Sebulba | 10, 5 with Sebulba | 1 and 2 |

That is 350 runs and 704 slot-runs. PPO on Breakout uses the settings of `quickstart.ipynb`, since hangar's defaults learn Breakout far more slowly. All other combinations use hangar's settings. Scaling runs take a whole node and use its first N GPUs, so that no other job shares the node. The workspace limit is 24 slots, so submit one group at a time.

### Before submitting

- The environment on `/pfss` needs the suites the runs use. Install them with `uv sync --frozen --no-dev --extra hangar --extra gymnax --extra jaxued --extra brax --extra jaxmarl --extra craftax --extra minari` from a shell in the checkout.
- `TARGETS` in `results.py` sets the returns for time to target. Set them from the one-GPU strong-scaling runs.

### Not configured yet

- Two-host runs. `hangar/train.py` starts `jax.distributed` from SLURM, MPI or JAX variables, which Determined does not set, so `run.sh` needs to translate Determined's distributed launcher first.
- Sebulba throughput as a function of the number of actor GPUs, which is only meaningful with an environment outside JAX. PPO has no ALE recipe, and the ALE adapter needs its native library built.
- The baselines (PureJaxRL, Stoix, Cleanba and Brax's multi-device PPO). Each needs its own environment on `/pfss` at a pinned commit and a runner script.
- Evaluation on JaxUED's held-out mazes, which its reference numbers use. boonta evaluates on random mazes only.
- The league, which has no hangar configuration, and a second algorithm with PLR.

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
