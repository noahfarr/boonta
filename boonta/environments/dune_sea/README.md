# Dune Sea

## Boxoban tower

`BoxobanTower` wraps **Jumanji's Sokoban environment**. Jumanji handles movement,
box pushing, dense rewards, level completion, and terminal puzzle timeouts. The
tower samples an ordered list of local Boxoban levels once at construction and
advances to the next floor after each solution. Episodes and parallel environment
copies share that fixed tower.

Install the optional dependency into the active environment and fetch the levels:

```bash
uv pip install jumanji
git clone --depth 1 https://github.com/google-deepmind/boxoban-levels.git data/boxoban-levels
boonta train algorithm=ppo environment=dune_sea/boxoban_tower
```

Alternatively use the project's `jumanji` extra when managing dependencies with
`uv sync` or `uv run` (alongside the other extras needed by your command).
Add `curriculum=psr` to train with prioritized state restarts.

The default is five floors sampled without replacement from `unfiltered/train`,
with tower seed zero, Jumanji's **120 actions per floor**, and a separate overall
budget of 2,000 actions. Adjust the floor limit with
`environment.kwargs.floor_horizon=200`. A floor timeout ends the whole episode;
it does not retry the same floor inside an episode. The ordinary auto-reset
wrapper then restarts at floor zero; PSR can restart from an assigned archive
snapshot. Both a puzzle timeout and the tower budget are terminal failures
(`terminated=True`, `truncated=False`). Solving an intermediate floor, including
on its last allowed action, advances with a fresh floor clock and no episode
boundary. Completing the summit ends the episode successfully.

Actions are Jumanji's four movements: `0=up`, `1=right`, `2=down`, `3=left`.
There is no manual reset action. Rewards come directly from Jumanji's default
`DenseReward`: +1 per box placed on a target, -1 per box removed, +10 for solving a
floor, and -0.1 per action. The final placement and completion bonuses stack.
Calls after an episode has ended leave the state unchanged and earn zero.

Observations are float32 arrays of shape `(10, 10, 6)`: walls, targets, boxes,
player, a constant `floor / num_floors` channel, and a normalized Jumanji puzzle
clock (`step_count / floor_horizon`). Rendering remains a JIT-compatible RGB
uint8 image via `env.render(state, tile=32)`. Floors are zero based; successful
tower completion has `floor == num_floors`. State retains the native Jumanji
puzzle in `state.puzzle`, with convenience properties `player`, `boxes`, and
`floor_steps`.

This replaces the earlier custom implementation. Old five-action/five-channel
checkpoints are incompatible: start a fresh training run. The custom reward
arguments (`reward_mode`, `placement_reward`, `completion_reward`, `step_reward`)
have been removed in favor of Jumanji's native rewards.

The source can be a single Boxoban text file or a directory searched recursively.
Use a specific difficulty and split, rather than the dataset root, to keep
training and evaluation separate. Every selected level must be 10 by 10 with
four boxes and four targets, as required by Jumanji. For small tests, supply
`levels=[ascii_level, ...]` instead of `dataset_path`.

`environment.kwargs.seed` selects the tower independently of the learner's
training `seed`. A new seed does not guarantee disjoint puzzles; use held-out
splits for held-out evaluation. Floors remain randomly ordered, with no automatic
difficulty ranking. Follow the [Boxoban dataset's license and citation
instructions](https://github.com/google-deepmind/boxoban-levels).

PSR archive keys contain the floor, player coordinates, and box occupancy. Keys
exclude time; snapshots retain their native Jumanji clock and total tower steps.
Evaluate complete climbs from floor zero separately from archive restarts.

## Maze

`Maze` is one fixed maze of `size` by `size` rooms on a `2 * size + 1` tile grid,
carved by a seeded randomized depth-first search. It is a perfect maze: exactly
one route joins any two cells. The agent starts in the top-left room and the goal
is the bottom-right room. Every episode and every parallel copy shares the maze;
`environment.kwargs.seed` picks it, independently of the training `seed`.

```bash
boonta train algorithm=ppo environment=dune_sea/maze
boonta train algorithm=ppo environment=dune_sea/maze curriculum=psr
```

Actions are `0=up`, `1=right`, `2=down`, `3=left`; a move into a wall leaves the
agent where it is. Each step pays for the cell it lands on:

```
reward = -(straight-line distance to the goal) / (farthest such distance in the maze)
```

so the reward is -1 at the start, rises towards 0 near the goal, and is exactly 0
on the step that arrives. The distance ignores the walls, which makes the reward
dense but deceptive: climbing it leads into dead ends. Reaching the goal
terminates the episode; `horizon` steps without reaching it truncate it.

Observations are a one-hot of the agent's cell over the `2 * size**2 - 1` open
cells. PSR archive keys are that cell index. `env.render(state, tile=32)` draws
walls grey, the goal green and the agent red.

## Chain

`Chain` is the smallest testbed for restarts: `length` cells in a row, each its
own bandit. A step answers the current cell, paying 1 for that cell's secret
action and 0 otherwise. Then the episode ends with probability `hazard`, and
otherwise moves on to the next cell; it also ends after the last cell.
`environment.kwargs.seed` draws the secrets, independently of the training `seed`.

```bash
boonta train algorithm=ppo environment=dune_sea/chain "artisan=[checkpoint,chain,concentration]"
boonta train algorithm=ppo environment=dune_sea/chain curriculum=psr "artisan=[checkpoint,chain,concentration]"
```

No action changes where the agent goes, so cell `d` is reached from the start
with probability `(1 - hazard) ** d` whatever the policy does, and never sooner
than step `d`. The answers are independent, so the advantage of a cell depends
only on that cell. Shallow cells get the most data and are learned first; what
is left to learn then sits in the deep cells, which the start rarely reaches and
a restart reaches at once. Raise `length` or `hazard` to make the deep cells
rarer. `hazard=0` keeps every cell certain and leaves only the walk to skip.

Observations are a one-hot of the agent's cell, and PSR archive keys are that
cell index. The `chain` artisan logs `chain/correct_<d>`, the probability the
policy gives each cell's secret, their plain mean `chain/correct`, and
`chain/return`, the exact expected return of the current policy against
`chain/best_return`. The return weighs each cell by its reach, so it is settled
by the shallow cells: watch `chain/correct` to see the deep ones.

`Chain` also carries the tables the restart scripts read: `expected_return`,
`advantages`, `transitions`, `place` and `cell_grid`.

## Combination lock

`CombinationLock` is `Chain` where a wrong answer ends the episode. Each of the
`length` cells pays 1 for its own secret action, drawn by
`environment.kwargs.seed`; any other action pays nothing and ends the episode.
An optional `hazard` (0 by default) also ends it at random after every answer.

```bash
boonta train algorithm=ppo environment=dune_sea/combination_lock "artisan=[checkpoint,combination_lock,concentration]"
boonta train algorithm=ppo environment=dune_sea/combination_lock curriculum=psr "artisan=[checkpoint,combination_lock,concentration]"
```

An episode from the start reaches cell `d` only by answering every cell before
it, with probability `prod(pi(secret | c) for c < d)` times `(1 - hazard) ** d`.
Unlike the chain, where the hazard decides the reach whatever the agent does,
the reach here depends on the policy: deep cells stay rare while the shallow
ones are still uncertain, and a restart reaches them at once.

`render` draws a padlock: a steel shackle over a brass body holding one number
wheel per cell, read left to right and top to bottom. Opened wheels show the
digit of their secret action, the wheel being answered is framed red, and wheels
ahead are dark and blank (digits are drawn for up to ten actions). The shackle
sits in empty rows at the top of `cell_grid`, so overlays line up with the
picture. `Chain` renders the same way.

Set the number of wheels with `environment.kwargs.length`, the actions per wheel
with `environment.kwargs.num_actions` and the combination with
`environment.kwargs.seed`.

The `combination_lock` artisan logs the same quantities as `chain`, under
`combination_lock/`. The tables (`expected_return`, `advantages`, `transitions`,
`place`, `cell_grid`) take the lock into account; `reached(policy)` gives the
reach of every cell under a policy.
