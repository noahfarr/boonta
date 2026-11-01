# BB-8 Phase 4 pilot: report (stop before the full 16-seed Phase 4)

2026-11-01. Experiments 55277 and 55278, run 11:02–13:25 UTC on 2 A100-SXM4-80GB. 4 seeds (0–3) per arm, Breakout and
Asterix. 208 runs: 128 PPO, 80 PQN. All 208 reached their budget or were stopped for entropy collapse; no crash.
Tables `phase4_pilot/tables.md`, per-run `phase4_pilot/results.csv`, curves `phase4_pilot/curves_{ppo,pqn}.png`,
frontier `phase4_pilot/frontier_{ppo,pqn}.png`, pool traces `phase4_pilot/traces.txt`. Calibration
`calibration.env` (STEP=kl, total-loss trust-region noise scale, EARLY_CAP=2), chosen from E1 v2 before the pilot.

## Answer first

- **Gate, PPO: fails, clearly.** BB-8 (κ = 0.25 and κ = 1) reaches the threshold in 0 of 8 runs per game.
  - Breakout: 7 of 8 BB-8 runs collapse (entropy below 1% of its start) and plateau at a return of about 6.
  - Asterix: final return 16–20 against 68 for the tuned arm.
  - The gate's "matches oracle fixed" count is 2/2 only because oracle fixed (the best fixed_N with step adaptation) also never reaches the threshold on Breakout and only once on Asterix; that is a weak bar, not a pass. BB-8 does not beat naive noise-scale batching on either game.
- **Gate, PQN: fails.** BB-8 κ = 0.25 overlaps oracle fixed on Breakout only (1 of 2 games) and beats naive on 1 of 2.
  - Aggregate steps to threshold / budget: BB-8 0.95 [0.84, 1.00] against the default 0.62 [0.58, 0.74] and the per-game oracle 0.57 [0.55, 0.69].
- **Best arms in the pilot are the step-only ones at the tuned batch**, as the addendum anticipated. PPO aggregate steps to threshold / budget:
  - tuned + linear lr anneal 0.82 [0.74, 0.87]
  - KL-adaptive alone 0.86 [0.80, 0.94]
  - tuned fixed 0.86 [0.79, 0.93]
  - KL target decaying 0.95 [0.85, 1.00]
  
  Every small-batch arm: 1.00 (never reaches it). On Breakout, tuned_anneal reaches the threshold at 12.7M [11.8, 16.8] against 17.3M [13.1, 19.9] for tuned_fixed, with 0/4 stalls against 1/4. That is suggestive, with overlapping intervals at 4 seeds.
- **Frontier (steps against wall-clock to threshold, arms where most seeds reach it):**
  - PPO Breakout: kl_alone, tuned_anneal
  - PPO Asterix: fixed_8192, tuned_fixed
  - PQN Breakout: default_fixed, fixed_1024
  - PQN Asterix: fixed_8192, default_fixed, naive
  
  BB-8 is on none. Small buckets also cost wall-clock: PQN fixed_256 takes 56 min per run against 3–4 for the default, because of per-update overhead at small batches.

## What happened (mechanisms, from the traces)

1. **Small batches collapse PPO on Breakout, with or without step adaptation.** Every arm that spends early training at
   256–2048 environments ends with entropy around 0.003–0.008 and a return near 6:
   - fixed_{256, 512, 1024, 2048}, with KL adaptation or with the lr anneal;
   - doubling, naive, BB-8.
   
   The 8192-environment arms end near 0.14–0.18. The realized KL per update at small B is small (≈ 0.001), but with
   8192/B more updates per environment step the policy moves 2–5× further per sample than at the full batch. On Asterix the same arms do not collapse but learn about 3× slower. All 38 entropy stops are PPO Breakout runs; none on Asterix or PQN (tables: "Stalls (stopped)").
2. **BB-8's step is throttled where its batch is large.** On Asterix the online total-loss trust-region noise scale reads
   about 25k environments, against N0 = 4184 from the E1 large probe used for calibration, so KL_max (B/(B+N))² is about
   10× below the calibrated value and the multiplier sits at 0.15–0.23. Its buckets (2k–8k) are reasonable; its steps are not.
3. **The online noise scale is resolved in 58–98% of probes** (higher at larger buckets). When unresolved, the pool keeps its
   last estimate (fix made before the pilot).
4. **PQN's online TD noise scale depends on the bucket**: Asterix 12.7k at 256, 14k at 512, 1.6k at 8192; Breakout 1.1k
   at 256 against 0.25k at 8192. These are different runs, so bias and trajectory are confounded, but the direction
   (higher at small B) is the opposite of the old PPO bias. The bias test covered PPO only.
5. **The KL controller's held-out quarter costs 25% of the batch.** fixed_8192 (KL step on 3/4 of the batch) ends at 37
   [17, 67] on Breakout against 55 [26, 84] for tuned_fixed, and level on Asterix (68.0 against 68.5). The intervals
   overlap; at 4 seeds this is not resolved.

## Relation to the predictions (PREDICTIONS.md, written before Phase 2; final scoring in REPORT.md)

- 14 (oracle fixed below 8192 on ≥ 2/3 games): wrong in the pilot. With step adaptation, no fixed batch below 4096 reaches the threshold.
- 15 (BB-8 passes the gate): wrong. 16 (naive worse than BB-8 and stalls): wrong on the first part (naive is not worse);
  stalls yes (Breakout 4/4).
- 18 (KL-adaptive alone no better than tuned fixed): right so far (overlapping).
- 19 (PQN BB-8 passes on ≥ 1 game): wrong (1 of 2 on the oracle criterion, but slower than default).
- 20 (probe overhead ≤ 15%): not measurable from these arms alone. Pool runs are slower than Anakin for reasons other than the probe (small buckets); measured separately in REPORT.md.

## Problems found in the pilot, and their state

- `analyze_phase4.py`: the curve plot crashed on entropy-stopped runs (shorter logs); the oracle choice tied when every
  fixed arm was censored; the frontier included collapsed arms (a censored wall-clock is not a time to threshold). All three are fixed; the tables above are from the fixed version.
- The Phase 4 job files ran PQN before PPO (the concatenation order). No effect on results.
- Nothing ran overnight before the pilot: after E1 v2 finished on 2026-10-31, nothing was queued until the user's
  "Continue" this morning. That was my scheduling, not the cluster.

## Decision needed before the full Phase 4

Running the full 16-seed gate as specified would very likely confirm the failure at higher cost (about 2× this pilot's
208 runs × 4, plus Space Invaders): roughly 10–14 GPU-hours. Options:

1. **Run it as specified** (all arms, 16 seeds, 3 PPO games and 2 PQN games), for the record.
2. **Run the 16-seed comparison only for the arms that are still in question**: tuned_fixed, tuned_anneal, kl_alone,
   kl_decay, fixed_4096/8192 (PPO), and default_fixed, default_anneal, fixed_2048–8192 (PQN), plus BB-8 and naive for
   the record. This tests the step-only result the pilot points to, at about a third of the cost.
3. **Stop here** and write REPORT.md from Phases 0–3 and this pilot.

Recommendation: option 2. The pilot already separates BB-8 and the small-batch arms from the rest by large margins (they
do not reach the threshold at all), while the step-only differences are small and need the 16 seeds.
