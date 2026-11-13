# Predictions: PSR on MinAtar Asterix

Written before any of these runs was launched. Scored in the Asterix report.

## Arms and settings

Four arms, seeds 0–14, 100M steps (50 logs of 1.97M steps; 98.3M steps in all):

- plain PPO, in shadow mode (PSR with k = 1: nothing placed, the archive and the cell ledger exist
  so the probe can run; checked against plain PPO on Pac-Man, −8.2 [−65.6, +58.5]);
- ERD (uniform over eligible archive cells, k = 4, restart at rollout);
- PSR v1: today's defaults (counted relevance × advantage gain, successor footprint, diagonal
  score, water-filling allocation, k = 4, restart at rollout, no importance weights);
- forecast gain: PSR v1 with the forecast gain, κ = 1, horizon 10.

PPO settings are those of the earlier Asterix runs (Breakout's tuned settings): 8192
environments, 16-step rollouts, 16 minibatches, 4 epochs, clip 0.14, entropy 0.003, γ 0.989,
λ 0.77, lr 2.4e-3, max grad norm 1.5, advantage normalisation. Cells: `gymnax.keyed`, bucket 2.
No tuning. The signal probe logs on every arm.

## The earlier result, for reference

The earlier 12M-vs-20M result (plain PPO and PSR, 3 seeds each) came from PSR v1 with *Marginal*
allocation, the default then, and predates the MinAtar time-limit fix, so restored snapshots
carried the game's own episode clock. ERD was not run. From the from-start returns: plain PPO
reached 95% of its mean final return (67.3) at 20M on all three seeds and PSR at 12M on all three;
final returns were equal (plain 70.3/71.0/71.1, PSR 70.6/69.8/71.5).

## Metrics (fixed)

- **Final return**: the mean of the last 10 logs of the from-start return (about the last 20M
  steps).
- **Steps to 95% of plain PPO's mean final return**: the first log at which a seed's from-start
  return reaches 0.95 × (plain PPO's mean final return over its 15 seeds). Median over all 15
  seeds per arm; a seed that never reaches it counts as censored (more than 98.3M), and the median
  is reported as censored if more than half are.
- **Bootstrap 95% CIs** (over seeds, 20,000 resamples) for forecast gain − ERD and for each
  restart arm − plain PPO, on both metrics. Steps-to-95% differences use censored seeds at 98.3M,
  which understates gaps involving them, and say so.
- 15 seeds per arm resolve only gaps of roughly the size the Pac-Man spreads allowed (about 25
  points there); Asterix's spread will set its own resolution, reported with the results.

## My predictions

- **A1** Final return: all four arms within 3 points of plain PPO's mean, and the forecast gain −
  ERD CI includes 0. Asterix under these settings levels off near 70 for every arm.
- **A2** Steps to 95%: PSR v1 reaches it earlier than plain PPO, but by less than the earlier
  8M, and with the CI on the difference including 0 (about 55%). The earlier gap came with
  Marginal allocation and without the clock fix.
- **A3** ERD is the fastest restart arm to 95% (it was on Pac-Man up to 40M), and the forecast gain
  is not faster than ERD.
- **A4** Survival parity holds on Asterix (rank statistic within 0.4–0.6 for every sampler).
