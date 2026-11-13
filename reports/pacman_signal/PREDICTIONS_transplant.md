# Transplant arm: reading fixed before launch

Written before the transplant run. `PREDICTIONS.md` is unchanged.

## Arm

Uniform restarts over eligible archive cells (as ERD), with each restart's pellets-left bucket
(50 pellets per bucket, 7 buckets) drawn from a fixed per-epoch schedule: the forecast gain's
per-placement bucket shares, averaged over 5 forecast-gain seeds rerun with per-placement logging
(seeds 5–9, `outputs/pacman_signal/forecast_gain_placement`). Within a bucket, restarts are
uniform over cells; a bucket with no eligible cell yet passes its share to the buckets that have
one. Seeds 5–19, 100M steps, k = 4, restart at rollout, the signal probe on. Everything else as
Part A.

## Reading

- Primary: final return, the mean of the last 10 logs.
- Reported: transplant − ERD and transplant − forecast gain with bootstrap 95% CIs over seeds
  (against the Part A arms, seeds 5–19), steps to 300 (median, never-reached censored at more
  than 100M), and mean return over 60–100M.
- 15 seeds per arm resolve only gaps above about 25 points (the Part A spreads give a 95% CI
  half-width of about 22–25 on a difference of means).

Decision, on the means:

- Transplant mean final ≥ forecast gain mean − 5: extend ERD, forecast gain and transplant to 70
  seeds each, and report that second stage separately.
- Transplant mean final ≤ ERD mean + 5: stop. A fixed average schedule is not the same as
  following each seed's own frontier, so this does not rule out progress as the mechanism.
- In between: inconclusive. Don't extend; move on to Asterix.
