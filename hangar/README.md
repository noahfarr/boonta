# hangar

`hangar` turns a Hydra config into a training run. `main.py` asks `recipes.make` for a podracer built from the chosen algorithm and environment, trains it, and logs the results.

## Run

From the repository root:

```bash
python hangar/main.py algorithm=ppo environment=gymnax/minatar/breakout
```

Hydra resolves the config relative to `main.py`, so the command works from any directory, and `python -m hangar.main` runs the same thing from an installed package. A run trains one seed. Each of its `training.num_epochs` epochs runs the same whole number of updates, as many as fit in `total_timesteps` (an update is the podracer's `batch_size` steps), and the logged step count is exactly what ran. It writes to `outputs/<date>/<time>-<array id>/`.

| Override | Effect |
| --- | --- |
| `seed=3` | Seed of the run |
| `total_timesteps=100_000_000` | Environment steps in total |
| `training.num_epochs=50` | Train and log cycles the steps are split into |
| `evaluation.num_steps=1000` | Evaluate for this many steps after every epoch (0 turns it off) |
| `+resume=<path>` | Resume from a run directory (its latest checkpoint) or from a checkpoint step directory |

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
| `dataset` | offline datasets (`minari`) | none |

Loggers and artisans combine as lists, for example `logger=[file,wandb] +artisan=[checkpointer]`. `artisan` is not in the defaults list, so it takes a leading `+`. Saving checkpoints to disk takes both the `checkpointer` artisan and the `orbax` logger.

`hyperparameters/<algorithm>/<environment>.yaml` is applied automatically. When there is no file for the exact environment, the `cascading_fallback` resolver walks up the environment path until it finds one.

## Kaggriculture

`environment=kaggriculture/kaggriculture` runs a C port of the two-player Kaggriculture farming game, built on first use. `algorithm=ppo` and `algorithm=recurrent_pupo` train in self-play, both seats sharing one network. `algorithm=recurrent_bc` clones players from recorded games and needs a dataset that is not included. To build one, download episode replays of the Kaggriculture competition from Kaggle (each episode's replay JSON, for example with the `kaggle` CLI), then run

```bash
python hangar/scripts/build_kaggriculture_dataset.py --replays <replay dir> --out <dataset dir>
```

It writes one file per player under `<dataset dir>/experts` and pools the top players by mean episode return into `<dataset dir>/pooled/board.npz`, holding a few out in `holdout.npz`. Train with `dataset.kwargs.directory=<dataset dir>/pooled`. A clone can start a self-play run with `network.pretrained=<checkpoint>`, the `algorithm_state` item of a saved `recurrent_bc` checkpoint.

## Recipes

`recipes.make(cfg)` picks the recipe registered for `(algorithm, environment namespace, suite)` in `recipes/__init__.py`. The algorithm name is the group choice (`algorithm=...`), not a field of the config. A recipe returns the algorithm, the environment and any extra podracer arguments, and `make` hands them to the podracer.

## Sweeps

`config/sweep.yaml` runs a CARBS search in which every trial is one run with its own seed. The sweeper and the `submitit` and Determined launchers come from the `sweep` dependency group; the `slurmpilot` launcher comes from the `slurm` extra (`uv sync --extra slurm`):

```bash
uv sync --group sweep
python hangar/main.py -cn sweep +sweep=<name> -m
```

A file `config/sweep/<name>.yaml` sets the search space, the number of trials and the launcher. Trials run on Slurm through the `submitit` or `slurmpilot` launchers, or on Determined through `determined/42`, all in `config/hydra/launcher/`. `determined/42` reads your home directory on the cluster from the `DETERMINED_HOME` environment variable and expects the venv at `$DETERMINED_HOME/relax/.venv`. Results go to `sweeps/<algorithm>/<environment>/<time>/`.

Finished sweeps are recorded in `hangar/sweeps/` under the same path: copy `multirun.yaml`, `optimization_results.yaml` and the latest `carbs/carbs_experiment/carbs_<N>obs.pt` from the sweep dir.
