# PSR on Pac-Man: forecast-gain confirmation and the signal-accounting framework

Branch `psr-signal` (local commits only). Predictions were fixed beforehand in `PREDICTIONS.md`.
Analysis scripts: `analyze_partA.py`, `analyze_partB.py`; their full output is in
`partA_output.txt`, `partB_output.txt`, `partA.json` and `partB.json`; figures are in this folder.
Per-run metrics are in `outputs/pacman_signal/<arm>/<job>/metrics.npz`, and the Part B results in
`outputs/pacman_signal/partB/seed{5,6}/`.

## Outcome

| Decision rule | Result |
|---|---|
| Method confirmed iff T6 | **Not confirmed.** Forecast gain − ERD on seeds 5–19: +14.2, 95% CI [−6.5, +34.9]. |
| Framework stands iff T1 | **T1 fails** (pooled r = 0.751 < 0.8; plain PPO's whole-run ratio 0.38 is outside [0.5, 2]). The framework is not used beyond describing the divergence below; T2–T5 are reported descriptively only. |
| If T5 fails because debiased spread matches forecast gain | Does not apply: debiased spread loses to forecast gain by 46 points [+19.9, +71.9]. T5 fails on its second clause (cumulative C_γ), which says nothing once T1 has failed. |

The forecast gain is still the best arm on every summary (final return, steps to 300, peak,
episode length), and pooled over 20 seeds it beats ERD with a CI just clear of zero
(+17.5 [+0.4, +34.5], secondary). But the confirmation that was designed to settle it, fresh
seeds 5–19 alone, does not exclude zero.

## Reproducibility and deviations

- **Runs are not bitwise reproducible on this GPU.** Two runs of identical code diverge at the
  first log (step 1M), first in `actor/advantage`, because GPU scatter-adds use atomics. The
  bitwise rule was replaced, as agreed, by a check in XLA's deterministic mode
  (`--xla_gpu_deterministic_ops=true`): forecast-gain seed 0 over 100M steps is bitwise identical
  on all 48 shared metrics with the probe on and off. Part A then ran in normal mode.
- A bug found on the way: last round's rank logging consumed a random-key split in PPO's update
  even with logging off. It now takes a folded key, so logging no longer shifts training
  randomness. Runs from before this round are unaffected; the stopped confirmation batch
  (`outputs/pacman_confirm`) was affected and is used only for the shadow-mode check.
- Every probe quantity is computed every update; the file logger stores each epoch's mean over its
  15 updates. Sums over updates multiply by 15. Feature rank is measured every 50 updates.
- The from-start mask also drops truncated steps.
- S_cell comes from the probe's own cell ledger (PSR's Advantage statistics and reach update,
  rate 0.2), so every arm is measured the same way.
- Shadow PPO (PSR with k = 1, nothing placed) cannot match plain PPO bitwise because the archive
  wrapper draws randomness differently. Statistically: shadow 268.3 (n = 15) against plain PPO
  seeds 5–14 run earlier 276.5 (n = 10), difference −8.2 [−65.6, +58.5]. Plain PPO's spread is
  twice as large because a few of its seeds collapse.
- Part B evaluation and training batches start with a 512-step burn-in from the true start, so
  their states cover every episode age rather than only the first 64 steps of an episode.
- Part B's fixed-minibatch-size scheme cannot go below one minibatch: at N = 1/16 it uses a single
  minibatch of 4,096 instead of 8,192.
- Part B fits: ρ² = a·N/(N + B*) cannot be fitted where ρ < 0, which happens at small N at every
  checkpoint. ρ is reported signed (ρ·|ρ| for "ρ²"), and the fit is restricted to the run of N at
  the large end where ρ stays positive (at least 3 points), and labelled so.
- No cluster: all runs are local.

## Pre-launch: survival parity

Script `hangar/scripts/survival_parity.py`, run on a shadow-PPO policy (seed 0, outside the Part A
seeds) at 5M and 80M steps. Pi plays from the true start for 1,600 steps through the training
wrapper, with 1024 environments, filling the archive and PSR's ledger exactly as in training. Both
are then frozen and pi plays 256 more steps to record each snapshot's real continuation. Each
snapshot is traced to the trajectory that wrote it and restored 8 times through the archive's own
take-and-reseed path. Survival is capped at min(200, steps left on the original's clock). u is the
share of copies dying before the original; 0.5 is parity.

| checkpoint | sampler | snapshots | median survival, original | median survival, restored | die within 30, original | die within 30, restored | u |
|---|---|---|---|---|---|---|---|
| 5M | from-start | 255 | 33 | 36 | 0.48 | 0.45 | 0.48 ± 0.02 |
| 5M | ERD | 500 | 29 | 30 | 0.52 | 0.50 | 0.48 ± 0.01 |
| 5M | PSR | 70 | 25 | 34 | 0.53 | 0.43 | 0.41 ± 0.04 |
| 80M | from-start | 199 | 37 | 37 | 0.45 | 0.43 | 0.49 ± 0.02 |
| 80M | ERD | 500 | 31 | 32 | 0.49 | 0.48 | 0.45 ± 0.01 |
| 80M | PSR | 71 | 17 | 20 | 0.63 | 0.59 | 0.43 ± 0.03 |

Restored copies never die sooner than the originals, so restore and snapshot selection are not
broken. ERD's picks are no more doomed than from-start states. Scatter plot:
`outputs/pacman_signal/parity_policy/survival_parity/survival_parity.png`. The policy is a weak
seed (return 142 at 80M), which doesn't matter for a test of restore correctness.

**The old restart-length metric is biased.** `restart/episode_length` is measured the same way in
every arm, by the shared archive wrapper. Cuts at the 64-step rollout boundary are not counted as
episode ends, and a placed environment that dies restarts from the same snapshot for the rest of
the rollout. Doomed snapshots therefore add many short episodes, and survivors add none. Simulating
that counting on the parity data gives 6–9 steps even for from-start snapshots, whose median
survival is 33–37. ERD's "3 steps" was this artefact, and so were my earlier remarks that the
hindsight, expectile and forecast arms restart into short, doomed episodes. The probe now counts
each placement once (`restart/placement_*`).

## Part A: confirmation (seeds 5–19, 100M steps)

Final from-start return is the mean of the last 10 logs. Steps to 300 is a median over 15 seeds,
counting seeds that never reach 300 as more than 100M.

| arm | final ± s.e. | steps to 300 (median) | seeds that reached 300 | peak | episode length | SPS |
|---|---|---|---|---|---|---|
| plain PPO (shadow) | 268.3 ± 12.7 | >100M | 4/15 | 277 | 229 | 0.63M |
| ERD | 309.8 ± 6.3 | 73.7M | 13/15 | 319 | 284 | 0.98M |
| PSR v1 | 303.6 ± 7.0 | 96.3M | 10/15 | 311 | 283 | 0.64M |
| **forecast gain** | **324.0 ± 9.0** | **60.9M** | 12/15 | 333 | 331 | 0.59M |
| PSR, debiased spread | 278.0 ± 10.5 | >100M | 5/15 | 284 | 242 | 0.63M |

Differences in final return (bootstrap 95% CI over seeds, 20,000 resamples):

| comparison | difference | 95% CI |
|---|---|---|
| forecast gain − ERD | +14.2 | [−6.5, +34.9] |
| forecast gain − PSR v1 | +20.4 | [−1.8, +41.2] |
| forecast gain − plain PPO | +55.7 | [+26.2, +85.2] |
| forecast gain − debiased spread | +46.0 | [+19.9, +71.9] |
| ERD − plain PPO | +41.5 | [+15.2, +68.1] |
| PSR v1 − plain PPO | +35.3 | [+7.7, +62.2] |
| debiased spread − PSR v1 | −25.6 | [−50.0, −2.0] |
| ERD − PSR v1 | +6.2 | [−12.1, +23.4] |

**Pooled over 20 seeds (secondary).** Seeds 0–4 come from the first batch, where "plain PPO" was
plain PPO, not shadow mode, and the debiased-spread arm did not exist.

| arm | n | final ± s.e. |
|---|---|---|
| plain PPO | 20 | 252.2 ± 14.9 |
| ERD | 20 | 311.5 ± 4.9 |
| PSR v1 | 20 | 304.3 ± 5.4 |
| forecast gain | 20 | 329.0 ± 7.5 |

Pooled: forecast gain − ERD +17.5 [+0.4, +34.5]; forecast gain − PSR v1 +24.7 [+6.6, +42.1].

Curves (median over seeds per arm): `partA_signal_curves.png` (c_out, δ, ρ_low, ρ_high, hold-out
ratio, return), `partA_accounting.png` (cumulative C_γ against J_γ, and the T1 windows) and
`partA_rank.png`.

Return over training: plain PPO and ERD rise fastest until about 30M; the three PSR arms trail
until about 40M; the forecast gain then overtakes everything from about 45M. Plain PPO stalls near
265 from 30M.

## How C_γ and J_γ diverge (T1 failed)

| windows starting | Pearson r | mean ΣC_γ per 10M window | mean ΔJ_γ per window |
|---|---|---|---|
| 0–10M | 0.25 | 64.3 | 71.1 |
| 10–30M | 0.56 | 4.8 | 10.5 |
| 30–60M | 0.14 | −0.1 | 2.8 |
| 60–100M | 0.63 | −4.0 | 1.3 |
| all 750 windows | 0.751 | | |

Whole-run ΣC_γ / ΔJ_γ (median and range over seeds): plain PPO 0.38 [−3.22, 1.11], ERD 0.64
[0.41, 1.03], PSR v1 0.76 [0.11, 1.20], forecast gain 0.72 [0.37, 0.98], debiased spread 0.77
[0.38, 1.55].

- **By phase.** The account works in size only at the start. In the first 10M steps, ΣC_γ (64)
  and ΔJ_γ (71) agree to about 10%, and that early contrast carries most of the pooled correlation.
  From 10M to 30M, C_γ captures about half of the change in J_γ. After 30M, the held-out captured
  signal is zero to slightly negative on average, while J_γ keeps rising by about 1–3 per window.
  On fresh data, late updates show no first-order improvement, yet return still creeps up.
- **The hold-out ratio** (smoothed c_out / c_in) falls from about 0.9 in the first 5M steps to
  0.34–0.49 at 5–20M, about 0.0–0.17 at 20–60M and 0.0–0.05 after 60M, in every arm. Most of what
  an update gains on its own batch after about 20M does not carry over to the next batch.
- **By arm.** Plain PPO is where the account fails worst: eight 10M windows contain single updates
  with large negative held-out captured signal (window sums down to −440) while J_γ barely moves,
  and the seeds with collapses give negative whole-run ratios. Without those 8 windows the pooled
  Pearson r would be 0.89, and the Spearman correlation over all windows is 0.65. These numbers are
  descriptive; under the decision rule T1 fails as stated, and the framework is not used to
  interpret T2–T5.

## Part A descriptive results (not framework tests)

- **T2, movement per update (δ), by arm.** 0–20M: 0.00128 (ERD) to 0.00205 (PSR v1), 1.6×.
  20–60M: 2.4×. 60–100M: 3.4×. ERD moves least after 20M; plain PPO moves most late (0.00166).
- **T4.** Mean c_out per update over 0–60M: PSR v1 0.00246, debiased spread 0.00246, forecast
  gain 0.00241, plain PPO 0.00220, ERD 0.00214. Plain PPO is above ERD, so the predicted order
  fails. Cumulative C_γ orders the arms debiased spread (84.0) > forecast gain (76.0) > PSR v1
  (75.8) > ERD (69.0) > plain PPO (−16.6); final return orders them forecast gain > ERD > PSR v1 >
  debiased spread > plain PPO. These do not match.
- **T5.** Debiased spread reaches 300 later than the forecast gain (5/15 seeds ever, against
  12/15) but has the highest cumulative C_γ (84.0 against 76.0). So T5 fails on its second clause.
- **δ against ½·Var(u).** Median ratio 0.99 in every arm (5–95%: 0.95–1.02, plain PPO 0.79–1.03).
- **The brackets on S.** S_raw / S_cell has a median of 37–127 across arms (5th percentile 11–23).
  ρ_low has a median of 0.001–0.004, ρ_high a median of 0.017–0.025 (5–95%: about −0.07 to 0.22),
  both mostly set by the first few million steps.

## Feature rank (penultimate layer, from-start states)

The egocentric encoder's actor and critic share one trunk, each a single Dense on the penultimate
features, so there is one feature set to measure. n99 is the number of eigen-directions for 99% of
the variance; "entropy" is the entropy effective rank. Means over seeds and phase:

| arm | 0–10M n99 / entropy | 10–30M | 30–60M | 60–100M |
|---|---|---|---|---|
| plain PPO (shadow) | 36.1 / 5.7 | 11.7 / 1.8 | 9.0 / 1.8 | 8.1 / 1.7 |
| ERD | 55.9 / 6.7 | 23.5 / 2.2 | 14.5 / 1.8 | 11.6 / 1.6 |
| PSR v1 | 36.0 / 5.7 | 16.0 / 2.0 | 12.6 / 1.8 | 9.4 / 1.7 |
| forecast gain | 37.1 / 5.8 | 14.3 / 1.9 | 10.3 / 1.8 | 8.7 / 1.7 |
| debiased spread | 35.4 / 5.7 | 12.3 / 1.9 | 11.1 / 1.9 | 9.8 / 1.8 |

- Every arm collapses to roughly 8–12 directions by 60M. ERD keeps up to twice the directions
  through 30M (it restarts from uniformly chosen cells and so trains on the most varied states).
- **Plain PPO's weakest seeds do not lose rank first.** The three weakest shadow-PPO seeds (19,
  12 and 14, final 209, 217 and 197) have *higher* n99 than the other 12 in every phase: 38.1 / 14.2 /
  10.3 / 9.4 against 35.6 / 11.0 / 8.7 / 7.8. The account "restarts stabilize PPO by keeping its
  features well conditioned" is not supported.
- The forecast gain, the best arm, has slightly lower rank than PSR v1 and ERD throughout. Rank
  does not track return here.

## Restart diagnostics

Restart share per pellets-left bucket (bucket 0 is 0–49 pellets left, bucket 6 is 300 or more of
the 318), with the per-placement survival: the first restored segment per environment per rollout,
followed to its first end or the 64-step cut.

| arm | phase | pellet buckets 0 … 6 | placement length | placements that die | die within 30 |
|---|---|---|---|---|---|
| ERD | 0–10M | .06 .19 .18 .19 .15 .19 .04 | 25.6 | 0.81 | 0.64 |
| ERD | 10–40M | .24 .22 .17 .15 .11 .11 .00 | 35.9 | 0.64 | 0.46 |
| ERD | 40–100M | .16 .19 .17 .17 .14 .16 .01 | 47.9 | 0.40 | 0.25 |
| PSR v1 | 0–10M | .00 .00 .00 .02 .17 .43 .38 | 49.7 | 0.45 | 0.21 |
| PSR v1 | 10–40M | .00 .00 .01 .14 .33 .41 .11 | 51.6 | 0.34 | 0.19 |
| PSR v1 | 40–100M | .01 .07 .21 .20 .18 .27 .06 | 52.5 | 0.31 | 0.17 |
| **forecast gain** | 0–10M | .00 .00 .00 .07 .24 .41 .28 | 46.6 | 0.55 | 0.25 |
| **forecast gain** | 10–40M | .00 .03 .14 .26 .22 .23 .13 | 46.7 | 0.49 | 0.26 |
| **forecast gain** | 40–100M | .09 .20 .24 .20 .10 .11 .06 | 48.9 | 0.41 | 0.23 |
| debiased spread | 0–10M | .00 .00 .00 .02 .11 .29 .58 | 53.5 | 0.32 | 0.15 |
| debiased spread | 10–40M | .00 .00 .02 .07 .16 .32 .44 | 56.1 | 0.19 | 0.12 |
| debiased spread | 40–100M | .01 .05 .08 .11 .08 .27 .40 | 57.0 | 0.19 | 0.10 |

- **Where the forecast gain restarts.** It follows the frontier: early in training most restarts
  have 200–300+ pellets left (early game), and by 40–100M the mode is 100–149 pellets left, with
  29% of restarts at fewer than 100. It shifts later into the game than PSR v1, which still puts 27%
  at 250–299 pellets late.
- **Why its restarts looked short.** They aren't short. A forecast-gain placement survives about
  47–49 of its at most 64 steps; 41–55% die before the cut and only 23–26% die within 30 steps. The
  short numbers reported earlier came from the biased metric (16–27 steps for this arm).
- **The debiased spread stays at the start of the board.** It keeps 40–58% of its restarts at 300+
  pellets throughout, its placements rarely die (10–15% within 30 steps), and it ends 46 points
  behind the forecast gain. Its priority picks states with spread in the advantages, which here
  means the early board. It also has the highest cumulative C_γ, so per-update captured signal does
  not predict final return in this comparison.
- **ERD** spreads restarts evenly over the board, as uniform-over-cells sampling should; its
  placements die often early on (81%) and less as the policy improves (40%).

## Part B: batch-size probes (plain PPO seeds 5 and 6; checkpoints at 10, 30, 60, 90%)

Each checkpoint took 5–6 minutes, so both seeds ran. The evaluation set is 8 default-size rollouts
(524,288 states). Each N gets 16 fresh batches, each updated once from a copy of the checkpoint
under both schemes (the number of minibatches fixed at 8, and the minibatch size fixed at 8,192).
S_look is the lookahead estimate on 4,096 evaluation states: for every action, two independent
estimates of its advantage, each averaging 4 rollouts of 64 steps with a critic bootstrap.

**Available signal by checkpoint.**

| checkpoint | S_raw | S_cell | S_look ± s.e. | S_look, single estimate |
|---|---|---|---|---|
| seed 5, 10M | 75.3 | 0.603 | 5.33 ± 0.38 | 8.08 |
| seed 5, 30M | 9.56 | 0.147 | 0.250 ± 0.019 | 0.295 |
| seed 5, 60M | 3.90 | 0.039 | 0.067 ± 0.022 | 0.061 |
| seed 5, 90M | 2.34 | 0.001 | 0.0021 ± 0.0002 | 0.0023 |
| seed 6, 10M | 48.9 | 0.692 | 4.00 ± 0.36 | — |
| seed 6, 30M | 25.6 | 0.219 | 1.58 ± 0.34 | — |
| seed 6, 60M | 3.09 | 0.012 | 0.051 ± 0.012 | — |
| seed 6, 90M | 1.88 | 0.015 | 0.014 ± 0.003 | — |

S_look always sits between the two brackets, much closer to S_cell (1.5–9× S_cell, against
S_raw at 12–2,000× S_cell). The available signal falls by about 3 orders of magnitude over training.

**c_out(N), number of minibatches fixed** (mean ± s.e. over 16 batches; per training sample in
brackets):

| N | seed 5, 10M | seed 6, 10M | seed 5, 30M | seed 5, 60M | seed 5, 90M |
|---|---|---|---|---|---|
| 1/16 | −0.00048 ± 0.00042 [−1.2e-7] | −0.00219 ± 0.00068 | −0.00249 ± 0.00214 | −0.00514 ± 0.00081 | −0.00009 |
| 1/8 | +0.00009 ± 0.00021 [+1.2e-8] | −0.00060 ± 0.00045 | −0.00035 ± 0.00063 | −0.01204 ± 0.00744 | −0.00009 |
| 1/4 | +0.00033 ± 0.00017 [+2.0e-8] | +0.00037 ± 0.00031 | +0.00031 ± 0.00020 | −0.00559 ± 0.00315 | −0.00008 |
| 1/2 | +0.00065 ± 0.00011 [+2.0e-8] | +0.00063 ± 0.00021 | +0.00023 ± 0.00025 | −0.00359 ± 0.00169 | −0.00009 |
| 1 | +0.00073 ± 0.00008 [+1.1e-8] | +0.00087 ± 0.00020 | +0.00082 ± 0.00013 | +0.00024 ± 0.00017 | −0.00009 |
| 2 | +0.00079 ± 0.00007 [+6.1e-9] | +0.00099 ± 0.00019 | +0.00046 ± 0.00009 | +0.00052 ± 0.00009 | −0.00009 |
| 4 | +0.00093 ± 0.00003 [+3.6e-9] | +0.00096 ± 0.00012 | +0.00041 ± 0.00010 | +0.00071 ± 0.00004 | −0.00007 |

All checkpoints, both schemes and the hold-out ratios are in `partB_output.txt` and
`partB_curves.png`.

- **At 10M the batch-size picture is clean** (both seeds). Captured signal per update rises with N
  and levels off by N = 2–4. Per training sample it peaks at N = 1/4–1/2 and then falls, so bigger
  batches buy less per sample. The hold-out ratio rises from about 0 at N = 1/16 to 0.63–0.66 at
  N = 4. The saturating fit on the positive range gives R² 0.97 (seed 5, B* ≈ 3.0 default batches,
  from N = 1/8) and 0.88 (seed 6, B* ≈ 1.7, from N = 1/4).
- **At 30M and later it breaks down.** Small batches (N ≤ 1/2) give negative held-out captured
  signal at 60M in both seeds: the update makes the policy worse on fresh data. At 90M, c_out is
  slightly negative at every N under both schemes. At seed 6's 30M checkpoint, even the in-batch
  captured signal is negative for N ≥ 1 (fixed count), meaning the update does not improve the
  policy on its own batch. One possible reason, not tested: PPO's entropy bonus (0.01) outweighs the
  remaining advantage signal.
- **Movement depends strongly on N** with the number of minibatches fixed: δ varies 6–36× across N
  at 7 of the 8 checkpoints (1.3× at seed 5's 90M), usually largest at the smallest batches, which
  take the same number of gradient steps on noisier gradients.
- **Fixed minibatch size** moves less at small N (fewer gradient steps) and at 30M gives the
  *largest* held-out captured signal at the smallest batches (seed 6: +0.0048 at N = 1/16 against
  −0.0017 at N = 1). The batch-size dependence is therefore not separable from the number of
  optimizer steps.

## Predictions scored

| prediction | result | note |
|---|---|---|
| T1 accounting | **fails** | r = 0.751; plain PPO ratio 0.38 |
| T2 budget, across arms | **fails** | 1.6× early, 2.4× and 3.4× later |
| T2 budget, across N (fixed count) | **fails** | 6–36× at 7 of 8 checkpoints |
| T3 batch size | **fails** | holds at 10M only (rises with N, R² 0.97 and 0.88, hold-out rises); later ρ < 0 at small N, no rise, no fit; B* growth cannot be assessed |
| T4 restarts | **fails** | plain PPO > ERD on c_out; order by C_γ ≠ order by return |
| T5 control | **fails** (second clause) | reaches 300 later ✓; cumulative C_γ higher ✗ |
| T6 confirmation | **fails** | +14.2 [−6.5, +34.9] |
| P1 δ ≈ ½Var(u) within 10% | holds | median 0.99; only plain PPO's 5th percentile (0.79) falls outside |
| P2 S_raw ≫ 30× S_cell; ρ_low < 0.1; ρ_high in [0.05, 0.6] | partly | median ratio 37–127, but the 5th percentile is 11–23; ρ_low < 0.1 ✓; ρ_high's median is 0.02, below the range |
| P3 hold-out ratio < 1 and falling | holds, more strongly than predicted | from about 0.9 to about 0.0–0.05 |
| P4 T1 at about 40% | consistent | early windows match in size, late ones don't |
| P5 T2 holds across arms, fails across N | half right | fails across arms too |
| P6 forecast beats ERD by 10–30, CI excludes 0 at about 60% | size ✓, CI ✗ | +14.2, CI includes 0 |
| P7 restarts at the frontier, over half end within 30 steps | half right | follows the frontier ✓; 23–26% within 30 steps ✗ (the old metric misled me) |
| P8 n99 within 30% across arms; weak PPO seeds don't lose rank first | half right | ERD up to 2× higher ✗; weak seeds have higher rank ✓ |

## What this leaves

1. The forecast gain is the best arm on Pac-Man by every measure, but on fresh seeds its margin
   over ERD is not established at 95%. The pooled 20-seed result is suggestive, not confirmatory.
2. The first-order accounting describes the first 10–20M steps, where most of the return is
   gained. After that, held-out captured signal is about zero while return keeps rising slowly. It
   cannot rank the restart rules, and the arm with the most captured signal (debiased spread) has
   the second-worst return.
3. Restarts do not keep PPO's features well conditioned in any way that tracks return, and the
   weakest plain-PPO seeds keep more rank than the rest.
4. The batch-size probes show a clean saturating dependence only at 10M. Later, small batches and
   even default batches produce updates that hurt held-out play, and with a fixed number of
   minibatches the movement depends strongly on N.
