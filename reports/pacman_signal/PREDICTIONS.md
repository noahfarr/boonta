# Predictions: PSR on Pac-Man, forecast-gain confirmation and signal accounting

Written before any Part A or Part B run was launched. Scored in `REPORT_signal.md`.

## Quantities (per update π_t → π_{t+1}, u = log π_{t+1} − log π_t, from-start samples only)

- captured signal C = E_s[Cov_π(u, A)]; held-out c_out on rollout t+1, in-batch c_in on rollout t
- movement δ = E_s[KL(π_t ‖ π_{t+1})] ≈ ½ E_s[Var_π(u)]
- available signal S, bracketed by S_raw = mean Â² and S_cell = Σ_c reach(c)·signal(c)⁺
- efficiency ρ = C / sqrt(2δS); ρ_low uses S_raw, ρ_high uses S_cell
- C_γ = Σ_i γ^{τ_i} c_i / (n_from-start / L); J_γ the γ = 0.99 discounted from-start return

## Hypotheses from the brief

- **T1 Accounting.** Over 10M-step windows pooled across arms, Σ C_γ against ΔJ_γ has Pearson
  r ≥ 0.8, and Σ C_γ / ΔJ_γ over a whole run lies in [0.5, 2] for every arm.
- **T2 Budget.** Mean δ per update differs by less than 2× across the five arms in the same
  training phase, and by less than 2× across N in Part B (fixed number of minibatches).
- **T3 Batch size.** ρ²(N) rises with N at every checkpoint, the fit ρ² = a·N/(N + B*) has
  R² ≥ 0.9, and the hold-out ratio rises with N. B* grows over training.
- **T4 Restarts.** Mean c_out per update over the first 60M steps ranks forecast gain > ERD >
  plain PPO, and the arms' order by cumulative C_γ matches their order by final return.
- **T5 Control.** Debiased-spread PSR reaches 300 later than forecast gain and has lower
  cumulative C_γ.
- **T6 Confirmation.** Forecast gain's final return beats ERD's on seeds 5–19, with a 95% CI on
  the difference that excludes 0.

Decision rules: method confirmed iff T6; framework stands iff T1, and only then are T2–T5 read
as tests of it; if T1 fails, report how C_γ and J_γ diverge and stop using the framework; if T5
fails, the win is noise removal, not captured signal.

## My own predictions (before launch)

From the 5M-step smoke runs with the probe (shadow PPO and forecast gain, one seed each):

- **P1 Second-order check.** δ and ½·mean Var_π(u) agree within 10% in every arm and phase
  (smoke: 0.0150 vs 0.0153).
- **P2 The brackets are far apart.** S_raw exceeds S_cell by more than 30× throughout, so
  ρ_low < 0.1 and ρ_high lies in [0.05, 0.6] (smoke: S_raw 25–100, S_cell 0.2–1.4). ρ_low is
  then mostly a measure of critic error, not of efficiency.
- **P3 Hold-out ratio.** c_out / c_in (smoothed) stays below 1 and falls over training, from
  about 0.95 early to under 0.8 by mid-training (smoke: 0.96 → 0.73 in 5M steps).
- **P4 T1.** I give T1 about 40%. Early on, Σ C_γ tracks ΔJ_γ (smoke: per-update C_γ ≈ 1.3
  against ΔJ_γ ≈ 2.5 per update in the first 1M steps, a ratio near 0.5). Late in training ΔJ_γ
  per 10M steps is small and noisy and the restart arms measure C from only 25% of samples, so I
  expect r < 0.8 when the windows are pooled, even if the whole-run ratios land in [0.5, 2].
- **P5 T2.** Holds across arms (PPO's clip sets δ more than the restart rule does). Fails or
  is marginal across N in Part B: with the number of minibatches fixed, a smaller batch takes
  the same number of steps on noisier gradients, which moves the policy more, not less.
- **P6 T6.** Forecast gain beats ERD on seeds 5–19 by 10–30 points; a bootstrap CI excluding 0
  at about 60%, given the 5-seed gap of 28 ± 13.
- **P7 Restart diagnostics.** Forecast-gain restarts sit at the frontier: most restarts land
  in one or two pellet buckets just below the bucket the from-start policy usually reaches,
  and over half of restarted episodes end within 30 steps (smoke: 69–74% late).
- **P8 Feature rank.** n99 of the penultimate layer is within 30% across arms in the same
  phase, and plain PPO's weakest seeds do not lose rank before the others, so the
  feature-conditioning account of the stabilization fails.

## Deviations planned before launch

- **Bitwise reproduction.** The original forecast-gain seed 0 cannot be reproduced bitwise by
  any code: two runs of identical code on this GPU already diverge at the first log (atomic
  scatter-adds are nondeterministic). The check is done in XLA's deterministic mode instead
  (`--xla_gpu_deterministic_ops=true`, bitwise reproducible, about 3× slower): forecast-gain seed
  0, 100M steps, logging off against logging on.
- **Logging granularity.** Every quantity is computed every update; the file logger stores the
  mean over each epoch (about 15 updates per 1M steps). Sums over windows multiply by the
  updates per epoch. Feature rank is measured every 50 updates.
- **From-start mask** also drops truncated steps, whose GAE is cut off.
- **S_cell** comes from the probe's own cell ledger (the same Advantage statistics and reach
  update as PSR, rate 0.2) in every arm, so ERD and shadow PPO are measured the same way as PSR.
- **Shadow PPO** (PSR with k = 1, nothing placed) uses the archive wrapper, which draws its
  randomness differently from plain PPO's auto-reset, so it cannot match plain PPO bitwise. It
  is checked statistically against the 10 plain-PPO seeds (5–14) finished earlier.
- **No cluster.** Runs are local; per-run metrics stay in `outputs/pacman_signal/<arm>/<job>/`.
