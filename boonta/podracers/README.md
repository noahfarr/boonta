# podracers

A podracer is the training loop. It owns the devices and the sharding, calls the [algorithm](../algorithms/README.md) and the [environment](../environments/README.md), and nothing else. All three share one interface, from [`podracer.py`](podracer.py).

| Method | Does |
| --- | --- |
| `init(key)` | builds the environment and algorithm states |
| `train(state, key, num_updates)` | runs exactly `num_updates` updates and returns `(state, logs)` |
| `evaluate(state, key, num_steps)` | restarts the environment and plays `num_steps` greedy steps |
| `close(state)` | frees the environment |
| `batch_size` | the steps one update consumes |

## Which one to use

| Podracer | In plain words | Pick it when |
| --- | --- | --- |
| [`anakin`](anakin.py) | acting, stepping and learning compile into one program, split across every device | the environment is written in JAX |
| [`sebulba`](sebulba.py) | actor threads collect rollouts while a learner trains, and they swap parameters through queues | the environment is slow or not JAX, such as Gymnasium |
| [`quadinaros`](quadinaros.py) | trains on batches from a fixed dataset and only plays the environment to evaluate | offline RL and behaviour cloning |

Pick one with `podracer=sebulba` on the command line. Offline algorithms such as `bc` choose `quadinaros` in their own config.

## Steps and updates

`batch_size` is the number of steps one update trains on.

- `anakin`: `num_envs × num_steps × num_agents`.
- `sebulba`: the same, times the number of actor devices.
- `quadinaros`: the product of `batch_shape`.

`hangar/main.py` splits `total_timesteps` into `training.num_epochs` epochs of the same whole number of updates, and logs `updates × batch_size`. It never asks a podracer to round a step budget.

The podracer counts steps into `algorithm_state.step`. The online podracers add `num_envs × num_agents` after every environment step. `quadinaros` adds `batch_size` after every update.

## Sharding

Each podracer builds its shardings from the state's dataclass fields. A field marked `metadata={"axis": "data"}` is split along the mesh's `data` axis. On `anakin` and `quadinaros` the environment states and timesteps are always split this way, as are the carries and replay buffers the algorithm marks. Everything else, such as parameters, is replicated on every device. On `sebulba` each actor device keeps its own environments whole, and only the learner's state is sharded by these marks. The mesh comes from `boonta.utils.mesh(count, start)`, and the configs in [`hangar/config/podracer/`](../../hangar/config/podracer) build it over every device by default.

## pit and lap

All three podracers take two hooks. Both default to identity and take and return the podracer's state.

- `pit` fires once per update: at `init`, after every `algorithm.update`, and after `evaluate` restarts the environment. Use it when the algorithm's state decides part of the environment, such as a curriculum choosing the next level or a league seating an opponent.
- `lap` fires after every environment step, and at `init` and `evaluate` just before `pit`. Use it for state the environment must see every step.

Each hook must fire in all those places. `evaluate` restarts the environment at its default settings, so a `pit` skipped there would evaluate a game training never played. A recipe passes the hooks by returning `pit` or `lap` in its dict. A [curriculum](../curricula/curriculum.py) supplies both: it takes `(algorithm, environment, **kwargs)` and returns `(algorithm, environment, pit, lap)`. [`tests/test_podracers.py`](../../tests/test_podracers.py) counts the calls on every podracer.

## anakin

`init`, `train` and `evaluate` are each one `jax.jit` call. `train` scans over updates, and each update scans `num_steps` environment steps and then calls `algorithm.update`. Parameters are replicated and the environments are split across the mesh.

## sebulba

Each actor device owns its own environments and steps them in its own thread. `num_envs` is per actor, so an update trains on `actors × num_envs` environments. The learner takes one rollout from every actor, in actor order, and joins them along the environment axis.

The lag is fixed at one update. After every update the learner puts its `algorithm_state` in every actor's queue, and each actor takes from its queue before every rollout except the second. So rollout *k* is collected with the parameters of update *k*−2 while the learner trains update *k*−1. Nothing depends on thread timing, and a run is reproducible at a fixed seed. The queues are rebuilt at every `train` call, so changing `training.num_epochs` changes results.

The state is `SebulbaState(actors, algorithm_state)`. `algorithm_state` exists once, as the learner's. Each `ActorState` holds only the actor's timestep, environment state and recurrent carry. `pit` and `lap` run on the actor, and what they write into `algorithm_state` is dropped after the rollout, except the step count. `evaluate` runs on the first actor. `sebulba` rejects `Ensemble` and `PSRO`.

The default config puts the learner and one actor on device 0. To give them separate GPUs, move the actor mesh:

```bash
uv run boonta podracer=sebulba podracer.config.actor.start=1
```

## quadinaros

`quadinaros` trains from a [dataset](../datasets/README.md). `train` first calls `dataset.update` on the host, then runs `fit`, a jitted scan in which every update calls `dataset.sample` for a batch of `batch_shape`. The dataset state rides in the podracer state and is split across devices. A `pit` that rewrites `dataset_state` changes what the next update samples. Training takes no environment steps, so `lap` fires only at `init` and in `evaluate`. `close` also closes the dataset.

## Several machines

Set `num_processes` to the total number of processes and start one per GPU, for example with `srun --nodes=2 --ntasks-per-node=4 uv run boonta num_processes=8`. `main.py` then calls `jax.distributed.initialize`, and the default meshes of `anakin` and `quadinaros` span every device of every process. `sebulba` runs on one machine. See [`hangar/README.md`](../../hangar/README.md).

## Add a podracer

A new podracer implements the methods above and a `make(config, algorithm, environment, ..., pit, lap)` function, and gets a config file in `hangar/config/podracer/`. Add a builder like `on_anakin` to [`tests/test_podracers.py`](../../tests/test_podracers.py) and put it in `EVERY`, so the shared contract tests run on it.
