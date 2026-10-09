# hangar

`hangar` turns a Hydra config into a training run. `main.py` asks `recipes.make` for a podracer built from the chosen algorithm and environment, trains it, and logs the results.

## Run

`hangar` needs Hydra and OmegaConf, which come with the `hangar` extra (`pip install "boonta[hangar]"`). In a checkout, `uv sync --extra hangar` installs them.

```bash
uv run boonta algorithm=ppo environment=gymnax/minatar/breakout
```

`boonta` is the console command for `hangar/main.py`. Hydra resolves the config relative to `main.py`, so it works from any directory, and an installed package gets the same command. A run trains one seed. Each of its `training.num_epochs` epochs runs the same whole number of updates, as many as fit in `total_timesteps` (an update is the podracer's `batch_size` steps), and the logged step count is exactly what ran. It writes to `outputs/<date>/<time>-<array id>/`, or to `outputs/<job id>-<array id>/` inside a SLURM job.

| Override | Effect |
| --- | --- |
| `seed=3` | Seed of the run: 0 by default, the trial number in a multirun |
| `total_timesteps=100_000_000` | Environment steps in total |
| `training.num_epochs=50` | Train and log cycles the steps are split into |
| `evaluation.num_steps=1000` | Evaluate for this many steps after every epoch (0 turns it off) |
| `checkpoint=<path>` | Start from a run directory (its latest checkpoint) or from a checkpoint step directory, continuing at the checkpoint's epoch |
| `early_stopping=epochs early_stopping.at=4` | End the run after 4 epochs of its plan |

### Several machines

`num_processes` above 1 runs one training across that many processes. Set it to the total count, machines times tasks per machine, and launch one task per GPU, for example `srun --nodes=2 --ntasks-per-node=4 uv run boonta num_processes=8`. `main.py` then calls `jax.distributed.initialize()`, which finds each task's rank and the coordinator through SLURM, Open MPI or JAX's coordinator variables. A count that does not match what the launcher started fails or hangs until it times out. `anakin` and `quadinaros` shard over every device of every task. All tasks write to one directory, `outputs/<job id>-<array id>/`. Process 0 alone prints the brief and runs the file, dashboard and wandb loggers. Every process takes part in an orbax checkpoint save. `sebulba` still runs on one machine.

## Config groups

| Group | Choices | Default |
| --- | --- | --- |
| `algorithm` | the files in `config/algorithm/` | `ppo` |
| `environment` | paths under `config/environment/`, for example `gymnax/minatar/breakout` | `gymnax/minatar/breakout` |
| `podracer` | `anakin` (on-policy), `sebulba` (actor/learner split), `quadinaros` (offline) | `anakin` |
| `logger` | `dashboard`, `file`, `wandb`, `orbax` | `dashboard` |
| `artisan` | `checkpointer`, `render` (needs the environment wrapped in `LogEnvState`, which no recipe on main does yet), `transcript` | none |
| `torso`, `stack`, `cell` | recurrent network parts | set by the recurrent algorithms |
| `buffer` | `transition`, `trajectory`, `episode`, `prioritised_episode` (the last two need `buffer.sample_sequence_length`) | set by the replay algorithms |
| `curriculum` | `default` | `default` |
| `dataset` | offline datasets (`minari/mujoco/expert`, `kinetix/offline_m`, `kinetix/offline_s`); a Minari dataset with `dataset.kwargs.pool_size` above 0 streams random whole episodes in pools of that many transitions | none |
| `scoring` | `best`, `final`, `mean`: how a run's returns become its score for a sweeper | `best` |
| `early_stopping` | `default` (never), `epochs` (`at` epochs), `steps` (`at` steps), `plateau` (`patience` epochs without a better `score`), checked after every epoch | `default` |
| `search_space` | the sweepers' search spaces, `<algorithm>/<environment>` | picked like `hyperparameters` |

Loggers and artisans combine as lists, for example `logger=[file,wandb] +artisan=[checkpointer]`. `artisan` is not in the defaults list, so it takes a leading `+`. Saving checkpoints to disk takes both the `checkpointer` artisan and the `orbax` logger.

`hyperparameters/<algorithm>/<environment>.yaml` is applied automatically, after the algorithm, environment and curriculum. When there is no file for the exact environment, the `cascade` resolver walks up the environment path until it finds one, and falls back to `hyperparameters/<algorithm>.yaml`. So `hyperparameters/ppo/gymnax/minatar.yaml` covers every MinAtar game. At each step it first tries `<environment>/<curriculum>.yaml`, which holds what a curriculum needs on that environment, such as the level sampler `hyperparameters/ppo/kinetix/plr.yaml` gives PLR.

## Recipes

`recipes.make(cfg)` picks the recipe registered for `(algorithm, environment namespace, suite)` in `recipes/__init__.py`. The algorithm name is the group choice (`algorithm=...`), not a field of the config. A recipe returns the algorithm, the environment and any extra podracer arguments, and `make` hands them to the podracer. [`recipes/README.md`](recipes/README.md) shows how to write one.

## Sweeps

`hydra/sweeper=carbs` turns a multirun into a CARBS search in which every trial is one run, seeded with its trial number unless you set `seed`. A grid multirun is unaffected:

```bash
uv run boonta -m hydra/sweeper=carbs algorithm=ippo environment=connectx/connectx
uv run boonta -m algorithm=ippo environment=connectx/connectx seed=0,1,2
```

The search space is `config/search_space/<algorithm>/<environment>.yaml`, looked up like the hyperparameters above, so `search_space/ippo/connectx.yaml` holds the ConnectX space. Each file starts with `# @package search_space` and holds only the parameters. Each parameter is a config key with a `distribution` (`uniform`, `int_uniform`, `uniform_pow2`, `log_normal`, `logit_normal`), a `min` and a `max`, and optionally a `center`, a `scale` and a `rounding_factor`. An algorithm-wide space goes in `search_space/<algorithm>.yaml`, and an environment's space builds on it by listing `/search_space/<algorithm>` before `_self_` in its defaults. Every run carries its space as plain data in the top-level `search_space` key, which only the CARBS sweeper reads: its `params` default to `${oc.select:search_space,null}`. Keys set under `hydra.sweeper.params` in a config are merged on top, and `++hydra.sweeper.params={...}` replaces the space.

The sweeper is the `hydra_carbs_sweeper` plugin in `hydra_plugins/` at the repository root. It needs `carbs`, which, with the `submitit` launcher, comes from the `sweep` dependency group; the `slurmpilot` launcher comes from the `slurm` extra (`uv sync --extra hangar --extra slurm`). Its settings live under `hydra.sweeper`: `n_trials` (default 100), `n_jobs` (trials in flight at once, 1), `num_random_samples` (4), `resample_frequency` (5), `max_failure_rate` (1.0), `max_suggestion_cost`, `seed` (0), and `warm_start_from`, an earlier sweep directory or CARBS checkpoint to start from. It runs trials in worker threads. The default `dashboard` logger works there, but with `n_jobs` above 1 several dashboards draw over one terminal, so pick `logger=file` then.

Pick a launcher as usual, for example `hydra/launcher=submitit_local` or `hydra/launcher=submitit/ias` (in `config/hydra/launcher/`). The ConnectX sweep is:

```bash
uv sync --extra hangar --group sweep
uv run boonta -m hydra/sweeper=carbs algorithm=ippo environment=connectx/connectx \
    hydra.sweeper.n_trials=1024 hydra.sweeper.num_random_samples=16 hydra.sweeper.resample_frequency=16 \
    hydra.sweeper.max_failure_rate=0.5 hydra.sweeper.max_suggestion_cost=900 \
    logger=[file,orbax] loggers.orbax.max_to_keep=1 ++loggers.orbax.best=false \
    +artisan=[checkpointer] scoring=final \
    hydra/launcher=submitit_local hydra.launcher.gpus_per_node=1 hydra.launcher.timeout_min=30
```

Results go to `sweeps/<algorithm>/<environment>/<time>/`.

Finished sweeps are recorded in `hangar/sweeps/` under the same path: copy `multirun.yaml`, `optimization_results.yaml` and the latest `carbs/carbs_experiment/carbs_<N>obs.pt` from the sweep dir.

### Population-based training

`hydra/sweeper=pbt` runs [population-based training](https://arxiv.org/abs/1711.09846) over the same search space, one generation at a time. It needs nothing beyond `hangar`:

```bash
uv run boonta -m hydra/sweeper=pbt algorithm=ppo environment=gymnax/minatar/breakout \
    logger=file evaluation.num_steps=1000
```

Each member starts from values drawn from the space: uniformly, on a log scale for `log_normal`, on a logit scale for `logit_normal`, and over the powers of two for `uniform_pow2`. `center` and `scale` are not used. A generation is `training.num_epochs // generations` epochs of the full planned run, so `generations` must divide `training.num_epochs`. Each generation launches every member's `seeds` runs through the launcher with `early_stopping=epochs`, `early_stopping.at=<the generation's last epoch>` and `scoring=final`. Every run writes to `<sweep dir>/generation_<g>/member_<m>/seed_<s>`. The sweeper adds the `checkpointer` artisan and an `orbax` logger to whatever loggers the run already has. The logger keeps only the latest checkpoint (`max_to_keep=1`, no `best`), which is the generation's last epoch. The next generation passes that run dir as `checkpoint=`, which resolves to `checkpoints/latest`. The checkpointer saves only `algorithm_state` by default, so a resumed run starts its environments afresh. On `anakin` with JAX environments, `++artisans.checkpointer.fields=[algorithm_state,environment_state,timestep]` also carries the environments. With those fields, a run stopped and resumed at the same seed matches an uninterrupted run bit for bit. Sweep jobs are seeded by job number, so a member's generations still differ in their random keys. A member's fitness is the mean score of its seeds. The worst `fraction` of members, rounded up and at most half, each pick one of as many best members. A loser copies its pick when the gap in mean exceeds `threshold` pooled standard errors. A copier takes the winner's checkpoints seed by seed. It redraws each parameter from the space with probability `resample_probability`. Otherwise it multiplies a continuous or integer parameter by one of `factors` and clips it to `[min, max]`, and it moves a `uniform_pow2` parameter to the neighbouring power of two. Members that copy nothing keep their parameters. A member whose runs all scored nothing copies regardless of the gap. Selection and exploration follow Ray Tune's `PopulationBasedTraining` in synchronous mode. The seed-averaged gate, the clipping, rounding integers to the nearest value and the 1.25 factor are deliberate departures.

Because members load each other's checkpoints, PBT can only change parameters that keep the shapes of the algorithm state: learning rates and loss coefficients, not `environment.num_envs`, `rollout.num_steps`, `replay.capacity` or anything under `network`, `cell`, `torso` or `stack`. The sweeper refuses those keys before launching anything. It also refuses `total_timesteps`, because every member must plan the same run for generations to end on the same epochs. A resumed run continues from its checkpoint's step, so learning-rate schedules continue too.

Settings live under `hydra.sweeper`:

| Setting | Default | Source |
|---|---|---|
| `members` | 4 | a small population; with `fraction` 0.25 one member copies per generation |
| `seeds` | 4 | enough runs per member for the gate's pooled standard error |
| `generations` | 5 | a few exploit steps per run; must divide `training.num_epochs` (10 by default) |
| `fraction` | 0.25 | Ray Tune's `quantile_fraction`; Jaderberg et al. (2017) used 20% |
| `threshold` | 2.0 | about two standard errors, in the spirit of the paper's t-test selection |
| `factors` | [0.8, 1.25] | Jaderberg et al. (2017) use 0.8 and 1.2; 1.25 undoes 0.8 |
| `resample_probability` | 0.25 | Ray Tune's PBT default |
| `seed` | 0 | the sweeper's own random draws |

Seeds follow the job number unless you set `seed`. Setting it gives every run the same seed. The sweep dir gets `population.csv`, with one row per member and generation holding its scores, its parameters and the member it resumed from (`parent`). It also gets `optimization_results.yaml`, which holds the best final member, its checkpoint and its parameter schedule traced back through its parents.
