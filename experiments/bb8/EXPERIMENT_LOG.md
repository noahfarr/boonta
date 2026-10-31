# BB-8 (Batch-Balanced on-policy RL): experiment log

Every command, run directory, seed range, wall-clock time and decision, in order. Times are UTC.

## 2026-10-31 Setup

- Branch `exp/bb8` made from `origin/main` at `cf09ddde` (#116), in the worktree `~/boonta-BB-8`. The local `main` ref is stale (it points at #113, `59addd0c`). `origin/main` is the real main and is what every comparison below means by `main`. Nothing is pushed.
- Compute. The task asked for the `ias` SLURM cluster (A5000 only, at most 3 GPUs). At the start, the account's 3-GPU QOS on `ias` was fully used by another experiment (`psr_cross`, jobs 186980–187047, about 100 array tasks pending on `QOSMaxGRESPerUser`). Submitting there would have queued behind it and slowed it down. The user offered their Determined shells on the 42 cluster, so this study runs on **42 (hessian.AI SuperPOD), A100-SXM4-80GB only**, with at most 3 GPUs in use at once (the user's own 1-slot shell `genuinely-rare-penguin` counts towards that). Every comparison inside this study uses the same hardware. Results are not hardware-identical to the A5000 numbers of the PSR study.
- Access. The Mac is not on the TU network and has no TUDA VPN, so 42 is reached through `ias`. SSH: `ssh -J ias 42`. The det CLI: an SSH SOCKS tunnel (`ssh -D 18888 ias`) with `HTTPS_PROXY=socks5h://127.0.0.1:18888` and PySocks on `PYTHONPATH` from a scratch directory. The user's det tool install is not modified. All of this is in `det/env.sh`.
- Cluster layout. The code is rsynced to `/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/bb8`, with its own venv (`uv sync --frozen --extra hangar --extra gymnax`). The user's `…/ar44ruca/boonta` checkout is a different commit with local edits, and it is left alone. `main` for the GPU bitwise check is rsynced to `…/ar44ruca/bb8_main`. Runs write to `…/ar44ruca/bb8_runs/<phase>/…`. `REVISION` in the cluster checkout records the commit and the uncommitted files at each sync, and every job prints it.
- Launching. `det/submit.sh <name> <job file> <parallel> <slots>` creates one Determined experiment (`max_restarts: 0`, project 338) whose context is a single `run.sh`. That script runs `det/run_job.sh`, which executes every line of the job file, `<parallel>` at a time, with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, `XLA_PYTHON_CLIENT_PREALLOCATE=false` and the shared `JAX_COMPILATION_CACHE_DIR`. Each run writes `train.log` into its own run directory. The repo's `hydra_determined_launcher` was not used: it needs `det` on the machine that runs the sweeper, and the cluster has none.
- Install. `uv sync --extra hangar --extra gymnax --extra plot` fails on `main`: there is no `plot` extra. matplotlib already comes in through another dependency, so no extra is needed and `uv.lock` is unchanged.

## Phase 0

### Facts about `main`, checked

- PPO, `boonta/algorithms/ppo.py`. Advantages are normalized over the whole batch, before minibatching, only when `normalize_advantage=True` (default False). Minibatches are random transitions after flattening (T, num_envs). The optimizer is `optax.chain(clip_by_global_norm, adam)`, built in `hangar/recipes/ppo_minatar.py`. All confirmed.
- PQN is on `main`: `boonta/algorithms/pqn.py`, recipe `hangar/recipes/pqn_minatar.py` (also `chain(clip_by_global_norm, adam)`, LayerNorm torso), and `hangar/config/hyperparameters/pqn/gymnax/minatar.yaml`.
- Default PPO (`algorithm/ppo.yaml`): 4096 envs, T 16, 16 minibatches, 2 epochs, lr 2.5e-4, clip 0.2, max grad norm 0.5. Confirmed. MinAtar has only `hyperparameters/ppo/gymnax/minatar.yaml` (`num_steps: 20_000_000`). There is no Breakout-specific file. Confirmed.
- Default PQN resolved on MinAtar: 4096 envs, T 16, 16 minibatches, 1 epoch, lr 5e-4, max grad norm 10, γ 0.99, λ 0.65, ε from 1.0 to 0.01 over the first 20% of `num_steps`, 80M steps. `algorithm/pqn.yaml` alone says 128 envs and 4 minibatches. The MinAtar hyperparameter file overrides both.
- Anakin fixes `environment.num_envs` and `podracer.config.num_steps` at compile time. `batch_size = num_envs × num_steps × num_agents` (`boonta/podracers/anakin.py`). Every environment steps every iteration, and `algorithm_state.step` counts `num_envs × num_agents` per step (`anakin.py`, `rollout`). `hangar/train.py` floors `num_steps / (batch_size × num_epochs)` to whole updates per epoch.
- Environments on `main`: gymnax (minatar, bsuite, classic_control), craftax, kinetix, xland_minigrid, mujoco_playground, brax, jumanji, plus others (ale, jaxtari, dune_sea, …). There is no NAVIX. Confirmed.
- Ported from the PSR branch: `experiments/bb8/analyze.py`, from `exp/psr-breakout` at `5a0ce3d1` (`experiments/psr_breakout/analyze.py`). The adaptations: the metric is `training/episode_return` (main has no `rho/` metrics), wall-clock is read from each run's `train.log`, and the manifest takes an explicit `budget`, below which a run is refused. `hangar/scripts/one_update.py` from that branch was read, not ported yet. It is for Phase 2, and it keeps `num_minibatches` fixed when it shrinks the batch, which is exactly what this task forbids, so it needs changing before use.

### Step budgets (`budgets.sh`)

One update is 4096 × 16 = 65,536 steps for both algorithms.
- PPO: `num_steps=19988480 num_epochs=61`, so 305 updates, 5 per epoch, and 61 logged points. This is the largest whole budget at or below main's 20M.
- PQN: `num_steps=79953920 num_epochs=61`, so 1220 updates, 20 per epoch, and 61 logged points (main: 80M). The ε schedule is built from `num_steps`, so it ends at 15,990,784 steps.
- Checkpoints (`orbax` keeps every epoch): the epochs nearest 10/40/80% of training are 6, 24 and 49 of 61, which is 9.8%, 39.3% and 80.3%.

### Resolved configs

`boonta train --cfg job algorithm=<ppo|pqn> environment=gymnax/minatar/<game> <budget> logger=file`, saved as `configs/<alg>_<game>.yaml` for breakout, asterix, space_invaders, freeway and seaquest. Within an algorithm, the five files differ only in `env_id`.

### Smoke runs and throughput (experiment 55179, `jobs/phase0.txt`, 2026-10-31 12:10)

2M steps (`num_steps=1966080 num_epochs=6`, 30 updates), Breakout, seed 0, one run at a time on one A100. Runs are in `bb8_runs/phase0/<alg>_probe_<on|off>`. `monitor/SPS` is the mean over epochs 2–6, so it leaves out compilation:

| Algorithm | Probe off (SPS) | Probe on (SPS) | Probe overhead (wall time per step) | Total time off / on, compile included |
|---|---|---|---|---|
| PPO | 1.80M | 1.07M | +68% | 16 s / 22 s |
| PQN | 2.55M | 1.38M | +85% | (see log) |

The probe costs nearly one extra pass over the batch: 64 vmapped group gradients (two backward passes for PPO, actor and critic separately), plus 3 frozen optimizer updates per loss. That is acceptable for measurement runs. For the Phase 3 schedule, it argues for probing every k-th update. I've noted that and not done it.

## Phase 1

### Implementation

- `boonta/algorithms/bb8.py`. `BB8Config(noise_scale_probe=False, num_groups=64, decay=0.9)`, the estimator `noise_and_signal`, `group_columns` (contiguous environment columns, whole T-step segments), `frozen_updates` (vmapped `optimizer.update` with the state held fixed), and `probe` (moving averages of noise and signal separately, bias-corrected, then the ratio).
- PPO and PQN get `cfg.bb8` (a static field, default off) and `state.bb8` (None when off, so the state pytree and checkpoints are unchanged). When on, `probe_noise_scale` runs once per update, at the pre-update parameters, after advantages and targets are computed (and normalized, if that is on), and before minibatching. It reuses the update's own `loss_fn`, so the math isn't duplicated. PPO probes `actor` (surrogate − c_ent·entropy), `critic` (c_v·value loss), and `total` (their sum, which is what training applies). PQN probes `td`. Auxiliary losses are not probed. None are used here.
- Configs: an `algorithm.bb8` block in `algorithm/ppo.yaml` and `algorithm/pqn.yaml`, all off.
- Logged per loss and space: `bb8/<loss>/<space>/{noise, signal, noise_scale_envs_raw, noise_scale_envs, noise_scale_transitions}` and `bb8/<loss>/group_gradient_norm`. The spaces:
  - `gradient`: the raw group gradients;
  - `update`: `optimizer.update(g_i, frozen state)`, with gradient clipping per group and Adam;
  - `fresh`: `update(g_i) − update(0)`.
  Decision: `update` contains the part of Adam's step that comes from its frozen first moment. That part is identical across groups, so it inflates the signal and not the noise. `fresh` removes it. Both are logged, because the task's definition ("map each group's gradient through `optimizer.update`") is `update`, but `fresh` is the part the new batch is responsible for. A second caveat, to keep in mind when reading: the group updates are updates for a batch of B/m environments, so with clipping or Adam's nonlinearity they are not the full-batch update. Group size: 64 environments × 16 steps = 1024 transitions, against 4096 transitions per training minibatch.

### Unit test of the estimator (`tests/test_bb8.py`, 5 tests, all pass on CPU)

The synthetic quadratic L(θ) = ½‖A(θ − θ*)‖² has a known mean gradient and a diagonal covariance. B = 4096 environments in m = 64 groups, true noise scale 100 environments:
- independent samples (T = 1): the noise and signal pooled over 64 batches give 101.2 (+1.2%); a single batch gives a median of 102, with 10–90% at [95, 109];
- correlated segments (T = 16, a shared draw per segment plus step noise): 100.6 (+0.6%); a single batch [95, 107];
- the smoothed probe with SGD: `gradient` within 10%, and `update` and `fresh` equal to `gradient` (to 1e-4), since SGD is linear;
- frozen Adam with a late state (ν spread over 8 orders of magnitude): `fresh` matches the noise scale of the preconditioned gradient within 10%;
- groups are whole environment columns.

### Bitwise identity with `main` (`bitwise_check.sh`)

12 updates of Breakout from the hangar config, seed 0. Every leaf of params, optimizer state and step is compared with `np.array_equal`.

| Device | Config | PPO: probe off vs main | PPO: probe on vs main | PQN: probe off vs main | PQN: probe on vs main |
|---|---|---|---|---|---|
| CPU (M3) | 256 envs | identical (26/26) | identical | identical (32/32) | identical |
| A100 (55179) | 256 envs | identical | 24 of 26 leaves differ | identical | 30 of 32 differ |
| A100 (55179) | 4096 envs | identical | identical | identical | 30 of 32 differ |

With the probe off, training is bitwise identical to `main` everywhere. That was the requirement. With the probe on, training is unchanged on CPU but not always on GPU: the probe uses no randomness and writes nothing to the training path, yet the parameters drift by about 1e-2 after 12 updates. My working explanation is that adding the probe to the graph changes XLA's kernel or fusion choices for the shared computation, and training amplifies the last-bit difference. The 2M smoke runs agree: PPO's returns with the probe on equal those with it off epoch by epoch, while PQN's differ from epoch 2. Consequence: E2's probed runs are statistically equivalent to default training but not bitwise copies of it. The diagnostic `--xla_gpu_autotune_level=0` is queued with E2.

### E2: noise scale over training (experiment 55181, `jobs/e2.txt` from `jobs/make_e2.sh`, submitted 2026-10-31 12:17)

- PPO: 5 games × seeds 0–3, `PPO_BUDGET`, probe on, `logger=[file,orbax]` (checkpoint every epoch).
- PQN: Breakout and Asterix × seeds 0–3, `PQN_BUDGET`, same.
- Run dirs: `bb8_runs/e2/<alg>_<game>/<seed>`. 4 runs at a time on 1 A100.
- An earlier submission, 55181's predecessor 55180, was killed before it started: zsh did not word-split `$GAMES` while I generated the job file, so the PPO lines held one invalid game name. Nothing was written. The job file is now generated by a bash script.

### E2 results (55181 COMPLETED 2026-10-31 ~12:50; wall-clock per run with 4 sharing one A100: PPO 1:22–2:33, PQN 4:26–6:19)

- **Seaquest fails on `main`** (all 4 seeds, `ValueError: Seaquest-MinAtar is not in registered gymnax environments`). gymnax raises `NotImplementedError("Seaquest is not yet supported.")` for it (`gymnax/registration.py:84-86`) and leaves it out of `registered_envs`, but `hangar/config/environment/gymnax/minatar/seaquest.yaml:3` points at it. Bug on `main`. Proposed fix: delete that config, or point it at a working MinAtar Seaquest. Not fixed here. **Decision: the study uses the 4 MinAtar games gymnax has** (Breakout, Asterix, Space Invaders, Freeway). The failed runs were moved to `bb8_runs/e2_failed/`.
- Autotune diagnostic: `--xla_gpu_autotune_level=0` leaves the probe-on drift in place (PPO 24/26 and PQN 30/32 leaves differ at 256 envs), so autotuning is not the cause. The cause is still unexplained, perhaps fusion or reduction order. Probe off remains bitwise equal to main.
- All 24 runs reach their budget. Final `training/episode_return` (last 10% of epochs): PPO Asterix 46–54, Breakout 9–21, Space Invaders 110–129, Freeway NaN; PQN Asterix 63–65, Breakout 22–58. Freeway's NaN is a logging effect: episodes last 2,500 steps, and an epoch is 80 steps per environment, so most epochs finish no episode. A return metric for Freeway needs a window spanning epochs.
- Figure `e2/noise_scale.png`, table `e2/growth.md` (`plot_noise_scale.py`). Early is the first 20% of epochs, after epoch 1; late is the last 20%. Medians over 4 seeds, in environments (× 16 for transitions):
  - **Raw gradient: it grows.** Critic: Breakout 82 → 495, Asterix 106 → 2264, Freeway 24 → 6188, Space Invaders 4 → 3639 (6–820×, 4/4 seeds in every game). Actor: 1.1–3.7×, but it starts at 1.6k–4.7k environments, the same order as the batch (4096), where the estimate is poor (signal is a small difference of large numbers). Total (what training applies): 6–650×, 4/4 seeds everywhere. PQN TD: Breakout 119 → 293, Asterix 121 → 715.
  - **`update` (Adam with its frozen first moment) does not grow much**, and it is small: 30–80 environments throughout for PPO actor and total, and 40–70 for PQN. Late/early is 0.7–2.3, and 6.0 only for Space Invaders total. The frozen momentum is the same for every group, so it dominates the signal and pins the ratio.
  - **`fresh` (the gradient's own contribution to the update) grows** like the raw gradient. PPO total: 3.8–64×. PQN: Asterix 5.5×, Breakout 1.0× (flat). The actor's `fresh` noise scale is larger than its raw one (3k–30k environments, above the batch), because Adam's preconditioning boosts low-variance-per-ν coordinates.
- Reading: the prediction "the noise scale grows over training" holds for the raw gradient and for `fresh`, in every PPO game and in PQN Asterix (4/4 seeds), and weakly for PQN Breakout. It does not hold for the noise scale of the full Adam update with momentum. Which of these the law in Phase 2 should use is the open question. It is put to the user before PREDICTIONS.md is written.
- Comparison with the PSR-branch evidence (12.5k samples early, 60–76k later, which is about 780 → 4,000–4,750 environments): Breakout total-loss raw gradient 111 → 681 environments (1.8k → 11k transitions). That is the same growth but 5–7× smaller. The configs differ: advantage normalization is off here, and lr, clip, γ, λ and epochs are untuned defaults, not the `prd` Breakout file.
