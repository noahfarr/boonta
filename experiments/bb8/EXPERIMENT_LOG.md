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

## 2026-10-31 (afternoon) Continuation: answers and fixes before Phase 2

The user answered the three decisions. (1) The rest of the plan arrived. (2) "On the update" means the **fresh** update; raw and full are diagnostics. (3) Stay on 42 A100s, record the GPU type per run, never mix hardware within a comparison. Every run in this study ran on A100-SXM4-80GB; each job prints `nvidia-smi` at its start.

### Tuned PPO config
- `hangar/config/hyperparameters/ppo/gymnax/minatar/bb8_tuned.yaml`, selected with `hyperparameters=ppo/gymnax/minatar/bb8_tuned`. It holds prd's Breakout values: 8192 envs, lr 2.4e-3, γ 0.989, λ 0.77, clip 0.14, 4 epochs, 16 minibatches, entropy 0.003, max grad norm 1.5, advantage normalization on. prd's stale `rollout.num_steps` is not copied, and the resolved `podracer.config.num_steps` is 16 (checked for Breakout and Asterix).
- Surprise: `hyperparameters/ppo/gymnax/minatar/` did not exist (`minatar.yaml` is a file). The first override pointed at a missing file and Hydra silently resolved it as "no hyperparameters", because the defaults entry is `optional`. Creating the directory fixed it, and the default cascade still finds `minatar.yaml` (checked).
- Budget: 8192 × 16 = 131,072 steps per update. `num_steps=19922944 num_epochs=76` gives 152 updates, 2 per epoch, and 76 logged points. E1 checkpoints: epochs 8, 30 and 61 (2,097,152 / 7,864,320 / 15,990,784 steps = 10.5% / 39.5% / 80.3%).

### Freeway
With 8192 environments each environment takes 2,432 steps in the whole 20M budget, less than one Freeway episode (2,500 steps; all environments start together). No training episode finishes, so neither longer epochs nor carrying statistics across windows can give a training return. **Freeway is dropped from Phase 4** and is kept in E2 for the noise scale only.

### E2 tuned (experiment 55186, `jobs/e2_tuned.txt`, A100; 4 at a time; 4:00–4:20 per run)
All 16 runs reach 19,922,944 steps. Final training return (last ~10% of epochs): Breakout 26/54/55/84, Asterix 68–69, Space Invaders 157–165, Freeway NaN (see above). Table `e2_tuned/growth.md`, figure `e2_tuned/noise_scale.png`. Fresh noise scale, early → late (median of 4 seeds, environments), and the late/early ratio:
- actor: Breakout 1.5k → 15k (9.7×), Asterix 19k → 141k (7.7×), Space Invaders 2.8k → 107k (41×), Freeway 2.6k → 16k (6.4×). 4/4 seeds grow in every game.
- critic: 12 → 442 (33×), 47 → 3.4k (75×), 14 → 664 (42×), 79 → 2.6k (31×). 4/4 everywhere.
- total: 5.4×, 16×, 21×, 7.5×. 4/4 everywhere.
- full update (frozen momentum): actor 1.3–2.6×, total 1.9–3.9×, critic 0.65–14×. Far flatter than fresh, but growing. See below for why it stays flat.
- The actor's fresh noise scale is mostly above the batch (8192 environments), already from early training on. A single-batch estimate there is a small difference of large numbers; the large-probe E1 values below are the reliable ones.

Why the full update is flat. With the optimizer state frozen, update(g_i) = −lr · (β1 m + (1 − β1) g_i) / (√ν̂ + ε) up to bias corrections. The momentum term β1 m is the same for every group: it adds the constant ‖β1 m / √ν̂‖² to the signal ‖E[U]‖² and nothing to the noise tr Cov[U]. With β1 = 0.9 it is about 9× the gradient's own contribution in amplitude, so the ratio is pinned near noise / momentum², which moves little over training. The fresh update removes the constant term, and its noise scale is that of the gradient in Adam's metric.

## Phase 2 (E1)

### Script: `one_update.py` (adapted from exp/psr-breakout's `hangar/scripts/one_update.py`)
- The update is the algorithm's own `update`, with `num_minibatches = max(1, columns / minibatch envs)`: the minibatch size stays fixed, and the number of epochs (PPO 4, PQN 1) stays fixed. The optimizer is rebuilt with a traced learning rate. Check: at the full batch and multiplier 1, the result is **bitwise equal** to the algorithm's update (CPU), even with the traced learning rate.
- Fractions: 64, 256, 1024, 2560 and 8192 environments (0.78 / 3.1 / 12.5 / 31.25 / 100%), so that the 512-environment minibatch divides the batch. **Deviation:** the two smallest batches are smaller than one minibatch, so they run as a single minibatch of 1,024 or 4,096 transitions; the minibatch size cannot be kept there. PQN: 32, 128, 512, 1280 and 4096 of 4096 (minibatch 256 environments; the two smallest are one smaller minibatch).
- Multipliers 0.125–8× (7 points), 4 update seeds (minibatch shuffles and column subsets), ΔJ over 4096 episodes from reset with common random numbers (the same reset keys, Gumbel noise and environment keys before and after). PPO: stochastic policy. PQN: ε-greedy at the checkpoint's ε (0.01 at 40%), and greedy separately.
- Matched KL (PPO): bisection (8 steps) on the multiplier until the KL equals the median KL of the full-batch 1× update.
- Large-probe noise scale: 8 consecutive rollouts (65,536 columns, environment-major so a probe group holds whole environments across rollouts), through the Phase 1 probe. Raw, fresh and full for actor, critic and total. The single-batch estimate is recorded too.
- First submission (55187, 3 processes on one GPU): 7 of 9 processes ran out of memory in the large probe (19–32 GB each). **Decision:** E1 runs one process per GPU. Retried as 55188, which also runs the GPU bitwise checks of Phase 3.
- PQN greedy ΔJ is degenerate on Breakout: gymnax MinAtar is close to deterministic under a greedy policy, so the 4096 greedy episodes are near copies, and ΔJ takes a few discrete values (0, −1, −23…). The "best" greedy step is then just one small enough to leave every greedy action unchanged. It is reported as degenerate, not fitted.
- Space Invaders is the held-out game for tuning the SGD and Lion base learning rates (55189, together with PPO Asterix).

## Phase 3 (implemented while E1 ran)

- `boonta/podracers/pool.py` (`podracer=bb8_pool`, `hangar/config/podracer/bb8_pool.yaml`): pool of `max(buckets)` environment states, and one compiled rollout-and-update branch per bucket (a `lax.switch`). It gathers N environments round-robin, steps only those, and scatters them back.
  - Accounting by slots of `unit` = 256 environments × T: an update of bucket N runs when the credit reaches N, and the bucket only changes right after an update, so every step taken is budgeted. The exact count is in `algorithm_state.step`, which `train.py` now also writes as `pool/env_steps_total` for pool runs (a guarded addition, which changes nothing for Anakin).
  - Per bucket the algorithm gets `num_minibatches = max(1, N / minibatch_envs)`, and the optimizer is `chain(original, scale(m))`. For Adam that is exactly lr · m, and the recipes are unchanged.
- Schedules: `fixed`, `doubling` (equal fractions of `budget`), and `noise`. `noise` uses the target κ · B_noise, clipped to [256, 8192], snapped in log scale, with hysteresis: it moves only when the target crosses the geometric boundary next to the current bucket by a factor of 1.5.
- Step options: `none`, `lr` (m = m_max B/(B + B_noise)), `lr_sqrt`, `kl` (m adapts multiplicatively, at most ×2 per update, so the realized KL tracks KL_max (B/(B + B_noise))²), and `kl_constant`.
- The probe runs every `probe_every` = 8 updates (a `lax.cond`), with the moving average in between. B_noise is the fresh actor (PPO) or TD (PQN) noise scale, and the critic's is logged as well.
- Running advantage normalization (`algorithm.bb8.running_advantage_normalization`): moments weighted by transitions and forgotten at exp(−n/524,288). The pool uses it whenever B < `running_normalization_below` (8192 for PPO).
- Logs per update, reduced to scalars inside the branch: `pool/{bucket, noise_scale_envs, noise_scale_critic_envs, step_target, multiplier, kl, update_norm, env_steps}`. Episode statistics are logged as mean, sum and count, so per-epoch returns can be weighted by episodes (`training/episode_return_sum / _count`).
- Bugs found while building it: `copy.copy` of a `Wrapper` recurses through `Wrapper.__getattr__` (`boonta/environments/wrappers/wrapper.py:24-25`) because `_env` is not set on the new object yet. Worked around in `pool.with_num_envs` by copying `__dict__`. Proposed fix on main: `__getattr__` should raise `AttributeError` for `_env` (and dunder names) instead of recursing.
- Checks (CPU): every mode runs (noise/lr, noise/kl, doubling/lr_sqrt, fixed/kl_constant, PQN noise/lr). Steps taken 9,984 of the 10,240 budgeted in the smoke test; the difference is the credit still waiting (less than one batch). **With BB-8 off and the bucket fixed at the default N, the pool is bitwise identical to Anakin** for PPO (tuned, 26 leaves) and PQN (32 leaves). The GPU check is in 55188.

## 2026-10-31 (evening) After the intermediate report

### Phase 4 pilot v1 (55190 with a scan of lax.cond slots; 55193 with the host loop), 1 seed per arm, Breakout: findings
- GPU throughput. A `lax.cond` per 256-environment slot made XLA read a predicate to the host each slot: the pool ran
  about 7× slower than Anakin on a shared A100. The fix is a host loop with one jitted step per (bucket, probe) and one
  sync per update. Results are identical (the CPU smoke test reproduces the scan version exactly). Cost: about
  0.9–1.2 GPU-minutes per PPO pool run, with 6 runs sharing a GPU.
- Entropy collapse. BB-8 κ = 0.25 with KL adaptation sat at bucket 512 because its online noise scale read 175–500
  environments. The multiplier went to its cap of 16, entropy fell from 0.23 to 0.002, and the return stalled at 6.5.
  The doubling arm showed the same low readings at small buckets. That pointed to a measurement problem (below).

### Measurement fixes (addendum item 1)
- **The online fresh noise scale was biased low at small B.** Offline on one checkpoint (Breakout seed 0, 40%), with the
  old fresh (= u(g) − u(0)): actor 536 / 741 / 2310 environments at B = 256 / 1024 / 8192. The raw gradient was flat
  (5062 / 3020 / 2965). Clipping was not involved (clipped and unclipped were identical; the clip never fired). The
  cause: `optimizer.update(g_i)` also updates Adam's second moment with the group's own g², which damps the noisy
  small groups.
- **Fix: `fresh` is now the JVP of the update map at g = 0 with the whole optimizer state frozen** (`bb8.frozen_updates`).
  Clipping is the identity at 0, so it is excluded by construction. JVP at 0 is NaN only where Adam's ν is exactly 0
  (parameters that never had a gradient, so g is 0 there as well); those tangents are set to 0 and counted
  (`bb8/<loss>/fresh_undefined`). The old quantity is kept as `difference`. Lion's fresh is undefined (sign has a zero
  derivative at 0); Lion uses `difference`. With the JVP fresh, the same offline test is flat in B: actor 742 vs 849,
  total 148 vs 151, critic 8 vs 8 (short warm-up, CPU smoke test; GPU job 55214 has the full test).
- **One estimator.** `bb8.noise_and_signal(group_means, num_segments, metric)` is the only implementation. Units: the
  sample is a segment (one environment's T steps in one rollout); B counts segments, so B = N for one rollout and
  B = R · N for R rollouts stacked environment-major by `bb8.stack_rollouts`; noise scales are in segments
  ("environments") and × T in transitions. The online probe (PPO/PQN `probe_noise_scale`), the E1 large probe and the
  bias test all go through `bb8.probe`. Test `test_every_caller_recovers_the_noise_scale_in_segments`: synthetic
  gradients with a known noise scale of 100 segments, through the data path of each caller (online, column subsets,
  R = 1 and 4 rollouts stacked) × m ∈ {16, 64} × B ∈ {256, 8192} × T ∈ {1, 16}: 32 cases, all within 12%.
- **Root cause of the 290 vs 18k gap.** Not units: the invariance test rules out T × R = 64 and m = 64. Two causes:
  (1) warm-up and synchronization. After 300 steps from a synchronized reset, the actor's raw noise scale was 272;
  after 1000 steps it was 2965, the same order as E1. (2) The old nonlinear fresh, as above.
- **Desynchronization.** Random no-op starts (`--noops K`: environment e takes U{0..K} no-ops after reset, then the policy
  warm-up) in `one_update.desynchronize`, used by E1 and the bias test. Warm-up sensitivity (300 / 1000 / 3000, with
  and without 1000 no-ops) is GPU job 55214. From now on every offline measurement uses 1000 no-ops + 1000 warm-up.
- **Trust-region noise scale (PPO).** Space `trust_region`: the fresh updates in the policy's Fisher metric,
  ‖d‖²_F = E_s[Var_{a~π}((J_logits d)_a)] (twice the KL to second order). It is computed with a JVP of the logits
  along d on 1024 states of the batch (a fixed stride). Noise = b Σ ‖u_i − ū‖²_F / (m − 1), signal = ‖ū‖²_F − noise/B,
  through the same estimator.
- Restoring checkpoints written before these spaces existed: `one_update.restore` retries with the old probe layout
  and replaces the probe statistics by zeros (E1 does not use them).

### E1 v2 (addendum item 2)
- Held-out surrogate proxy: 16 more rollouts (16 × 8192 segments) from the pre-update policy, after the probe rollouts
  and independent of the update's batch, with GAE from the pre-update critic, unnormalized. Proxy =
  E[(π′(a|s)/π(a|s) − 1) A], kept in 1024 blocks of 128 columns for the bootstrap. Every condition uses the same
  held-out batch.
- Rerun (`jobs/phase2v2_adam.txt`, 55213; `jobs/phase2v2_other.txt` next) with: 10 multipliers 1/64–8×, 1000 no-ops +
  1000 warm-up, JVP fresh, trust region, proxy, episode ΔJ at every checkpoint, and matched KL. The old SGD/Lion job
  (55192) was killed, since it had neither the new noise scales nor the proxy; SGD and Lion are in the rerun.
- First reading (Breakout seed 0): the actor's JVP-fresh noise scale on the large batch is 96k environments at 10% and
  negative (signal estimate below 0) at 40%. In Adam's metric the actor's mean update is not resolved at B = 65,536.

### BB-8 fixes (addendum item 3)
- `PoolConfig.multiplier_cap` = 1 (never above the tuned learning rate); `early_multiplier_cap` (1, or 2 if E1 supports
  it) in the first `early_fraction` = 10% of the budget.
- KL modes (`kl`, `kl_constant`, `kl_decay`): the update trains on 3/4 of the batch's columns, and the controller's KL is
  measured on the held-out quarter. All N environments are stepped and counted, so these arms pay 25% of their samples
  for the measurement. Disclosed in the report.
- `train.py`: optional `stop_entropy_fraction` (off by default). When the epoch's `actor/entropy` falls below that
  fraction of the first epoch's, the run writes `stalled.json` and stops. The Phase 4 analysis counts it as a stall and
  extends its curve flat at its last value. Checked with a fraction above 1, which stops at epoch 0.
- `PoolConfig.noise_space` chooses which smoothed noise scale drives the schedule and the step.
- New step modes for the baselines: `anneal` (linear to 0 over the budget), `kl_decay` (KL target decaying linearly to 0).

### Bug fix (addendum item 5)
- `Wrapper.__getattr__` and `WrapperState.__getattr__` (`boonta/environments/wrappers/wrapper.py`) now raise
  `AttributeError` for the wrapped attribute itself (`_env`, `env_state`) and for dunder names. Before, `copy.copy`,
  `copy.deepcopy` and `pickle` recursed without end, because the object being built had no `_env` yet. Test
  `test_a_wrapper_and_its_state_can_be_copied_and_pickled` fails without the fix and passes with it. Environment,
  podracer, bb8 and PPO/PQN algorithm tests: 264 passed, 41 skipped. The pool's workaround is removed.

### Phase 4 additions (addendum item 4, prepared)
- `jobs/make_phase4.sh`: PPO arms tuned_fixed, tuned_anneal, kl_alone, kl_decay, fixed_{256..8192} with step
  adaptation, fixed_{256,512}_anneal, doubling, naive, bb8_k0.25, bb8_k1. Every PPO arm has the entropy stop. PQN arms:
  default_fixed, default_anneal, fixed_{256..8192} with lr adaptation, naive, bb8_k0.25.
- `analyze_phase4.py`: wall-clock to threshold (cumulative epoch steps / SPS, compilation included), stalls including
  entropy stops, and the frontier plot (IQM steps to threshold against IQM wall-clock to threshold, Pareto frontier
  marked).
- Pilot: 4 seeds per arm, Breakout and Asterix, after E1 v2 picks the noise scale, the step option and the early cap.

## 2026-11-01 Results of the measurement fixes and E1 v2; Phase 4 pilot

### Probe bias v2 (55214; Breakout seed 0 at 40% unless noted; 16 rollouts; noise scale in environments)
- Warm-up sensitivity, actor raw / actor trust region / total trust region at B = 8192:
  - warm-up 300: 883 / 945 / 144
  - warm-up 1000: 3739 / 4711 / 992
  - warm-up 3000: 5839 / 8156 / 1024
  - 300 + 1000 no-ops: 2388 / 3104 / 498
  - 1000 + 1000 no-ops: 6402 / 7713 / 869
  
  A synchronized start lowers the estimate several-fold. 1000 no-ops + 1000 warm-up reaches the 3000-step level. It is the setting used from here on.
- B-dependence (desynchronized): raw, trust region and total-loss fresh are roughly flat from B = 256 to 8192. Examples: Space Invaders actor trust 1226–1427, total trust 492–536; Breakout 10% actor trust 148–153. Asterix 40% rises 2156 → 3476 (1.6×) and Breakout 40% total trust is noisy (869–2214).
- The actor's JVP-fresh noise scale is unusable. It swings over 2k–430k and is often negative, even at B = 65,536: Adam's 1/√ν̂ makes it heavy-tailed. Total-loss fresh is stable.

### E1 v2 (55213 PPO Adam; 55215/55216 SGD, Lion, PQN). Analysis in `e1v2/` (`analyze_e1.py`, `e1_stage.py`)
- **Proxy validation**: held-out surrogate against episode-return ΔJ over the 50 (B, multiplier) cells. At 10%: Pearson 0.66–0.99, Spearman 0.85–0.97. At 40%: Pearson 0.89–1.00, Spearman 0.60–0.93. At 80%: Pearson 0.62–0.94 but Spearman −0.26 to 0.78. Late in training the proxy prefers steps larger than the episode returns support (first-order; it ignores the state-distribution shift). `e1v2/proxy_validation.json`.
- **Fitted N against the large-probe noise scales** (`e1v2/n_vs_probe.txt`, identified fits only):
  - 10% (6/6 identified with either measure). Proxy fits: total trust region N/probe geometric mean 0.95, 5/6 within 3×. Actor raw 0.67 and actor trust 0.67, 4/6. Actor fresh 0.014, 0/6. Total fresh 0.05, 1/6. Critic 100–300×. Episode-ΔJ fits: total trust 0.60 (3/6), actor raw/trust 0.43 (2/6).
  - 40% (4/6 identified with the proxy, 2/6 with episodes): the fitted N (250–1300) does not grow, while every probe noise scale grows 3–20×. No candidate is within 3× in a majority, except the flat full update (3/4) and critic fresh (4/4), which I read as coincidences.
  - 80%: 1–2 of 6 identified.
  
  **Conclusion: no noise scale predicts the fitted N across training within 3×.** The best candidate, the total-loss trust-region noise scale, works at 10% only. The fresh-update noise scale (the hypothesis) fails by about 70× at 10%.
- **Best step against training stage** (headline; `e1v2/best_step_vs_stage.png`). The best multiplier at B = 8192 (episode ΔJ, geometric median over seeds) falls 2 → 0.5 → 0.125 (Breakout) and 1 → 0.25 → 0.0625 (Asterix) from 10% to 40% to 80%: 16× in both games. The KL at that best step varies only 2.1× (Breakout) and 6.7× (Asterix). Within a stage, the best-step KL is roughly the same at every B.
- **Step option decision: `kl`.** The best step is far more constant in KL than in learning rate.
- **Early cap**: at 10% the best multiplier at B = 8192 is 2 (Breakout) and 1 (Asterix), so EARLY_CAP = 2 is supported for Breakout. The KL controller can still take smaller steps.
- **SGD with momentum / Lion** (40%): the best step is mostly at 8×, the top of the grid. The base learning rates (3e-3 / 1e-5) from the noisy held-out tuning were too low, so ε*(B) is censored from above. SGD fits are within 3× of its own noise scale in 2/6 cells; Lion's are 2–25× off. Lion's fresh and trust-region noise scales are undefined (sign). Inconclusive.
- **PQN** (40%): N is mostly unidentified (greedy ΔJ is degenerate on Breakout as before), or 2–20× above the TD fresh noise scale (24–143 environments). The best multiplier falls as B grows (0.5–1× at small B, 1/64–1/8× at the full batch): with a fixed minibatch size, a bigger batch is also more Adam steps.
- **Consequence (addendum rule):** none predicts N within 3×. The main Phase 4 comparison is therefore the step-only arms (tuned_anneal, kl_alone, kl_decay) and the fixed-small-batch arms (fixed_N with step adaptation, fixed_{256,512}_anneal). BB-8 uses the best available candidate, the total-loss trust-region noise scale (PPO) and TD fresh (PQN), with this caveat.

### Calibration (`calibrate.py` → `calibration.env`)
STEP=kl, NOISE_SOURCE=total, NOISE_SPACE=trust_region, PQN_NOISE_SPACE=fresh, EARLY_CAP=2. N0 is the median E1 v2 large-probe value: Breakout 887, Asterix 4184, Space Invaders 536 (from the bias test; no E1 v2 large probe there), PQN Breakout 135, Asterix 26. KL_med is the median full-batch 1× KL in E1: 0.0094 / 0.0097 / 0.0173. KL_max = KL_med ((8192 + N0)/8192)²: 0.0115 / 0.0221 / 0.0196. PQN m_max = 1.03 / 1.006. With PQN's N0 of 26–135 environments, κ = 0.25 targets far below 256, so PQN BB-8 will mostly sit at the 256 bucket.

### Pool fix before the pilot
- An unresolved noise scale (smoothed signal ≤ 0) was read as the cap (32768), which drove the KL target, and so the step, to zero. Now the pool keeps the last resolved estimate (`PoolState.scale`; N = B before the first) and logs `pool/noise_scale_resolved`. Pool-off is still bitwise equal to Anakin (CPU).

### Phase 4 pilot (55277: PPO Breakout + PQN Asterix; 55278: PPO Asterix + PQN Breakout)
`jobs/phase4_pilot2_{a,b}.txt` from `make_phase4.sh` with `calibration.env`. 4 seeds (0–3) per arm, 16 PPO arms, 10 PQN arms, 208 runs, 6 per A100. Run dirs: `bb8_runs/phase4_pilot2/<alg>/<game>/<arm>/<seed>`.

### Phase 4 pilot results (55277, 55278 COMPLETED 13:25 UTC; 208/208 runs; 38 entropy stops, all PPO Breakout)
See `PILOT_REPORT.md`. The gate fails for PPO and for PQN; the step-only arms at the tuned batch are the best. Analysis fixes made on the way: plot crash on stopped runs, oracle tie-break by AUC, frontier restricted to arms where most seeds reach the threshold. Stopped before the full Phase 4, as instructed.
