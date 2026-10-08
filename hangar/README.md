# hangar

`hangar` turns a Hydra config into a training run. `main.py` asks `recipes.make` for a podracer built from the chosen algorithm and environment, trains it, and logs the results.

## Run

From the repository root:

```bash
python hangar/main.py algorithm=ppo environment=gymnax/minatar/breakout
```

The command works from any directory, and `python -m hangar.main` runs the same thing from an installed package. A run trains one seed. Each of its `training.num_epochs` epochs runs the same whole number of updates, as many as fit in `total_timesteps` (an update is the podracer's `batch_size` steps), and the logged step count is exactly what ran. It writes to `outputs/<date>/<time>-<array id>/`.

| Override | Effect |
| --- | --- |
| `seed=3` | Seed of the run. It defaults to 0, and in a multirun to the trial number |
| `total_timesteps=100_000_000` | Environment steps in total |
| `training.num_epochs=50` | Train and log cycles the steps are split into |
| `evaluation.num_steps=1000` | Evaluate for this many steps after every epoch (0 turns it off) |
| `+resume=<path>` | Resume from a run directory (its latest checkpoint) or from a checkpoint step directory |

A multirun (`-m seed=0,1,2`) writes to `sweeps/<algorithm>/<environment>/<time>/`.

## Config

The config is typed Python, built with [hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) and stored in Hydra's config store, so the command line, multiruns, sweepers and launchers work as they do with YAML. It lives in `config/`:

| File | Holds |
| --- | --- |
| `config/__init__.py` | the root config and its defaults list |
| `config/sections.py` | typed sections the recipes read: `training`, `evaluation`, `rollout`, `replay`, `exploration`, `optimizer`, `network`, `environment`, `dataset` |
| `config/algorithms/` | one module per algorithm, mirroring `boonta/algorithms/`: its `algorithm` config, its settings on particular environments, and its search spaces |
| `config/environments/` | the `environment` group, one module per namespace (`gymnax.py`, `craftax.py`, `brax.py`, ...) |
| `config/networks.py` | the `cell`, `torso` and `stack` groups |
| `config/podracers.py` | the `podracer`, `curriculum` and `scoring` groups |
| `config/data.py` | the `buffer` and `dataset` groups |
| `config/outputs.py` | the `logger` and `artisan` groups |
| `config/sweeps.py` | the `sweep` and `search_space` groups |

Each module in `config/algorithms/` holds everything about one algorithm. Its config is built from the algorithm's own config class with its full signature, so every field can be set without a `+`, and a key the class does not have fails when the config is composed. `config/algorithms/pqn.py` is the whole of PQN's configuration:

```python
pqn = algorithm(
    dict(
        algorithm=fbuilds(PQNConfig, num_minibatches=4, update_epochs=1, gamma=0.99, q_lambda=0.65),
        rollout=Rollout(num_steps=16),
        environment=dict(num_envs=128),
        optimizer=Optimizer(lr=5e-4, max_grad_norm=10.0),
        exploration=epsilon,
    ),
    name="pqn",
)

pqn.hyperparameters(
    "gymnax/minatar",
    total_timesteps=80_000_000,
    algorithm=dict(num_minibatches=16),
    environment=dict(num_envs=4096),
)
```

`algorithm(node, name=...)` registers the node in the `algorithm` group and returns a handle for that algorithm. The handle keeps the node as `.node`, so `config/algorithms/mappo.py` reuses IPPO's with `algorithm(ippo.node, name="mappo")`. `config/algorithms/__init__.py` holds what the modules share (`recurrent`, `offline`, `epsilon`), `algorithm`, and imports every module. A new algorithm is a new module there and one line in that import.

| Group | Choices | Default |
| --- | --- | --- |
| `algorithm` | the modules in `config/algorithms/` | `ppo` |
| `environment` | the paths in `config/environments/`, for example `gymnax/minatar/breakout` | `gymnax/minatar/breakout` |
| `podracer` | `anakin` (on-policy), `sebulba` (actor/learner split), `quadinaros` (offline) | `anakin` |
| `logger` | `dashboard`, `file`, `wandb`, `orbax` | `dashboard` |
| `artisan` | `checkpointer`, `render` (needs the environment wrapped in `LogEnvState`, which no recipe on main does yet), `transcript` | none |
| `torso`, `stack`, `cell` | recurrent network parts | set by the recurrent algorithms |
| `buffer` | `transition`, `trajectory`, `episode`, `prioritised_episode` (the last two need `buffer.sample_sequence_length`) | set by the replay algorithms |
| `curriculum` | `default` | `default` |
| `dataset` | offline datasets (`minari`, `disk`, `kinetix`) | none |
| `scoring` | `best`, `final`, `mean`: how a run's returns become its score for a sweeper | `best` |
| `sweep` | `carbs` | none |

Loggers and artisans combine as lists, for example `logger=[file,wandb] +artisan=[checkpointer]`. `artisan` is not in the defaults list, so it takes a leading `+`. Saving checkpoints to disk takes both the `checkpointer` artisan and the `orbax` logger.

`.hyperparameters(environment path, **settings)` on the handle registers the settings for that algorithm on an environment in the `hyperparameters` group. They are applied after the algorithm and environment and before the curriculum. When there is no entry for the exact environment, `cascade` walks up the environment path until it finds one, so this line in `config/algorithms/ppo.py` covers every MinAtar game:

```python
ppo.hyperparameters("gymnax/minatar", total_timesteps=20_000_000)
```

## Recipes

`recipes.make(cfg)` picks the recipe registered for `(algorithm, environment namespace, suite)` in `recipes/__init__.py`. The algorithm name is the group choice (`algorithm=...`), not a field of the config. A recipe returns the algorithm, the environment and any extra podracer arguments, and `make` hands them to the podracer.

## Sweeps

`+sweep=carbs` runs a CARBS search in which every trial is one run, seeded with its trial number unless you set `seed`. It switches to multirun by itself, so it needs no `-m`. The search space comes from `.search_space(environment path, **space)` on the algorithm's handle, looked up like the settings above, on top of an algorithm-wide space, `.search_space(**space)`, if there is one. The ConnectX space is `ippo.search_space("connectx", ...)` in `config/algorithms/ippo.py`. Plain runs and grid multiruns never see a space. The sweeper and the `submitit` launcher come from the `sweep` dependency group; the `slurmpilot` launcher comes from the `slurm` extra (`uv sync --extra slurm`):

```bash
uv sync --group sweep
python hangar/main.py +sweep=carbs algorithm=ippo environment=connectx/connectx
```

Pick a launcher as usual, for example `hydra/launcher=submitit_local` or `hydra/launcher=submitit/ias` (in `config/hydra/launcher/`). The ConnectX sweep that used to be `config/sweep/connectx.yaml` is:

```bash
python hangar/main.py +sweep=carbs algorithm=ippo environment=connectx/connectx \
    logger=[file,orbax] loggers.orbax.max_to_keep=1 loggers.orbax.best=false \
    +artisan=[checkpointer] scoring=final \
    hydra/launcher=submitit_local hydra.launcher.gpus_per_node=1 hydra.launcher.timeout_min=30
```

Results go to `sweeps/<algorithm>/<environment>/<time>/`. Finished sweeps are recorded in `hangar/sweeps/` under the same path: copy `multirun.yaml`, `optimization_results.yaml` and the latest `carbs/carbs_experiment/carbs_<N>obs.pt` from the sweep dir.
