# BB-8 round 2: predictions

Written 2026-11-01, before any round-2 run, after the Phase 4 pilot. Not to be edited afterwards; scored in
REPORT_2.md. B is in environments (T = 16). "Within the CI" means the arm's IQM lies inside the reference's 95% interval.

What I know when writing: prd Part D (same tuned config, archive auto-reset without restarts, `rho/episode_return`,
16 seeds) found default PPO on Breakout at 1024 environments 2.6× faster than at 8192 among non-stalled seeds (5.6M vs
14.9M), with 2/16 stalls at 1024 and 3/16 at 8192, and num_minibatches fixed at 16. My pilot's small fixed batches
(pool, fixed minibatch size of 512 environments, KL controller with a held-out quarter, running advantage
normalization) never reached the threshold, and collapsed on Breakout.

## Part 1: reconcile with prd Part D (PPO, Breakout and Asterix, 16 seeds)

1. A (plain Anakin, 16 minibatches, batch normalization) reproduces Part D on Breakout: A@1024 reaches the threshold
   faster than A@8192 among non-stalled seeds, by at least 1.5×, with at most 4/16 stalls at 1024. *Confidence: medium.*
2. On Asterix, A@1024 is also faster than A@8192 among non-stalled seeds (≥ 1.3×). *Confidence: low-medium.*
3. B (fixed minibatch size, so 2 minibatches at 1024 and 8 Adam steps per update) loses most of A's advantage: B@1024 is
   slower than A@1024 on both games (IQM outside A's CI) and no faster than A@8192 on Breakout. The main cause of the
   pilot's failure is the 8× fewer gradient steps per environment step, together with the KL controller.
   *Confidence: medium.*
4. C (running advantage normalization) at 1024 is within A@1024's CI on steps to threshold, with no more stalls. At 8192
   it is within A@8192's CI. *Confidence: medium.*
5. D (round-robin pool, idle environments paused, otherwise A) at 1024 is within A@1024's CI. *Confidence: medium.*
6. KL per environment step is higher at 1024 than at 8192 in A (≥ 2×); stalled seeds have the highest KL per environment
   step and the lowest final entropy. *Confidence: medium.*
7. If A@1024 does not beat A@8192 on Breakout, the archive wrapper (vs same-step auto-reset) is the difference.
   *Confidence: low.*

## Part 2: are batch size and learning rate interchangeable? (8 seeds)

8. PPO, Breakout and Asterix: no rule is interchangeable in the strict sense (within the B_ref CI at every B with no
   extra stalls). With the unscaled lr, small batches are faster on samples and stall more. *Confidence: medium.*
9. The square-root rule gives the fewest stalls among the three rules at B = 1024 and 2048, and is still at least as fast
   on samples as B_ref. *Confidence: low-medium.*
10. The linear rule (lr / 8 at 1024) is slowest at small B: outside B_ref's CI on the slow side on both games.
    *Confidence: medium.*
11. PQN Breakout: small batches with the unscaled lr are at least as sample-efficient as the default 4096, and the
    square-root rule is within the default's CI at every B. *Confidence: low.*
12. Part 2 finds a working small-batch setup (at least one B < 8192 and one rule within or better than B_ref's CI with
    no extra stalls) for PPO, so PPO-EWMA is not needed. *Confidence: medium.*

## Part 3: schedules against the progress rule (16 seeds; run only if Part 2 finds a working small batch)

13. (b) The anneal tuned once on Breakout transfers: within the CI of the best fixed setup ((a) or (d)) on at least 2 of
    3 PPO games and on both PQN games. *Confidence: medium.*
14. (c) KL-adaptive alone is within (a)'s CI on every game. *Confidence: medium.*
15. (d) The best fixed small batch beats (a) on samples on Breakout (outside the CI), not on every game.
    *Confidence: low-medium.*
16. (f) matches (b) on samples (within the CI) on at least 2 of 3 PPO games, but needs **more** updates than (b), not
    fewer, because it starts small. Its wall-clock to threshold is within a factor 1.5 of (b)'s. *Confidence: medium.*
17. (f) does not reduce stalls relative to (e) by a resolvable margin, and the two have overlapping CIs on steps to
    threshold on every game. Tying the rule to progress instead of time adds nothing measurable at 16 seeds.
    *Confidence: medium.*
18. PQN (f) with the plateau trigger is within (a)'s CI on both games. *Confidence: low.*

## Part 4: the held-out ratio (analysis of E1 v2)

19. r = held-out surrogate improvement / in-batch surrogate improvement rises with B at every checkpoint, and r < 1 at
    every fraction below 100%. *Confidence: medium-high.*
20. At 10%, r ≈ B/(B + N) fits with N within 3× of the ΔJ-fitted N (proxy fits) in at least 4 of 6 runs.
    *Confidence: low-medium.*
21. At 40% and 80% the fit is poor (residuals larger than at 10%) or N is not identified, mirroring E1.
    *Confidence: medium.*
