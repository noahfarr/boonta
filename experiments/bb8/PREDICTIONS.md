# BB-8 predictions

Written 2026-10-31, after Phase 1 (E2 under main's defaults) and before any Phase 2 run and before E2-tuned had
finished. Not to be edited afterwards; results are compared against it in REPORT.md.

Definitions used below.
- "Noise scale" without qualifier means the **fresh-update** noise scale: update(g) − update(0) with the optimizer
  state frozen, in environments (T = 16 segments). "Raw" is the gradient's, "full" is update(g) with frozen momentum.
- Early / late: first / last 20% of logged epochs (epoch 1 excluded).
- E1 fractions: 0.78%, 3.1%, 12.5%, 31.25%, 100% of the 8192 columns (64, 256, 1024, 2560, 8192 environments; chosen
  so the minibatch of 512 environments divides the batch; the two smallest are one minibatch smaller than 512).
- "Best step": the best of the 7 learning-rate multipliers 0.125×–8× at that fraction.

## E2 under the tuned PPO config

1. The total-loss fresh noise scale grows from early to late by at least 2× (median over seeds) in all 4 games, with
   at least 3 of 4 seeds growing in each game. *Confidence: high.*
2. The actor's fresh noise scale grows in at least 3 of 4 games (median ratio > 1), but by less than the critic's in
   every game. *Confidence: medium.*
3. The full-update noise scale stays within 0.5–2.5× from early to late in at least 3 of 4 games: the frozen
   momentum keeps it flat, as under the defaults. *Confidence: medium-high.*
4. With advantage normalization on, the actor's noise scale in environments at mid-training is within 3× of the
   PSR-branch estimate (Breakout ~780 environments early, 4,000–4,750 late). *Confidence: low-medium.*

## E1, the law for one update (PPO, Breakout and Asterix, 10/40/80% checkpoints)

5. The best-step ΔJ(B) is increasing in B and fits ΔJ_max / (1 + N/B) with the fitted curve inside the bootstrap CI
   of at least 4 of 5 fractions, at at least 4 of the 6 game × checkpoint cells. *Confidence: medium.*
6. The fitted N is within a factor of 3 of the large-probe fresh noise scale of the **actor** (total loss as a second
   candidate), in at least 4 of the 6 cells; it is closer to the actor's than to the critic's in at least 4 of 6.
   Raw and full are further from N than fresh in the majority of cells. *Confidence: medium (fresh vs actor);
   low (exact factor-3 agreement).*
7. The fitted N increases from the 10% to the 80% checkpoint in both games. *Confidence: medium-high.*
8. Without step adaptation (multiplier 1×), at the two smallest fractions ΔJ is less than half the best-step ΔJ in
   at least 3 of the 4 cells (2 games × 2 smallest fractions) at the 40% checkpoint, and the best multiplier there
   is below 1. At 100% the best multiplier is within 0.5–2×. *Confidence: medium-high.*
9. The KL of the best-step update rises with B; a log-log fit of KL against B/(B+N) (N fitted) has slope between
   1 and 3 (prediction 2). *Confidence: low-medium (the multiplier grid is coarse, factor 2).*
10. Matched-KL variant: holding KL fixed across fractions gives a ΔJ(B) that rises less steeply at small B than the
    best-step curve's prediction would allow if KL had to shrink; i.e. the best-step KL at 0.78% is below the matched
    target. *Confidence: low.*

## E1, other optimizers (PPO, 40% checkpoint)

11. SGD with momentum: the law holds with N within 3× of its own fresh noise scale on at least one of the two games
    (fresh = raw for SGD with zero momentum state). *Confidence: medium.*
12. Lion: the best-step curve is flatter at small B than the law predicts (sign updates have a B-dependent mean), so
    either the fit is poor or the fitted N disagrees with Lion's fresh noise scale by more than 3×.
    *Confidence: low-medium.*

## E1, PQN (40% checkpoint, Breakout and Asterix)

13. The law holds for PQN's behaviour-policy ΔJ with N within 3× of the TD fresh noise scale in at least one game;
    the best-step ‖Δθ‖ rises with B. Greedy-policy ΔJ is noisier and is not predicted. *Confidence: low-medium.*

## Phase 4, pilot gate

14. Under the tuned config the best fixed batch ("oracle fixed", with step adaptation) is smaller than 8192
    environments on at least 2 of 3 PPO games (Breakout, Asterix, Space Invaders) in steps to threshold.
    *Confidence: medium.*
15. PPO: BB-8 (better of κ = 0.25, 1) **passes** the gate (CI overlapping or better than oracle fixed in steps to
    threshold, and better than naive noise-scale batching) on at least 2 of 3 games. It does **not** give the strong
    result (beating oracle fixed outright) on a majority of games. *Confidence: medium for passing, medium-high for
    not strong.*
16. Naive noise-scale batching without step adaptation is worse than BB-8 in steps to threshold or AUC on at least 2
    of 3 PPO games, and stalls (never reaches the threshold) in at least 2 of 16 seeds on at least one game.
    *Confidence: medium.*
17. The doubling schedule with step adaptation has a CI overlapping BB-8's on at least 2 of 3 PPO games.
    *Confidence: medium.*
18. KL-adaptive learning rate alone at the tuned fixed batch does not reach the threshold sooner than the tuned
    fixed arm (overlapping CIs) on any game. *Confidence: medium.*
19. PQN: BB-8 (κ = 0.25) passes the gate on at least 1 of 2 games; not strong. *Confidence: low.*
20. With the probe every K = 8 updates, BB-8's wall-clock overhead from the probe is at most 15% of the run.
    *Confidence: medium.*
