# podracers

A podracer is the training loop. It owns the devices and the sharding, calls the [algorithm](../algorithms/README.md) and the [environment](../environments/README.md), and nothing else. All three share one interface, from [`podracer.py`](podracer.py).

| Method | Does |
| --- | --- |
| `init(key)` | builds the environment and algorithm states |
| `train(state, key, num_updates)` | runs exactly `num_updates` updates |
| `evaluate(state, key, num_steps)` | restarts the environment and plays `num_steps` greedy steps |
| `close(state)` | frees the environment |
| `batch_size` | the steps one update consumes |

None of these methods return metrics. Components log with `lox.log`, and each podracer wraps `train` and `evaluate` with `lox.spool`, so both give back `(state, logs)`.

Each compiled method is an ordinary `jax.jit`, with the podracer as a static argument (`@partial(jax.jit, static_argnums=0)`, which is why podracers are `@dataclass(eq=False)`), and `train` and `evaluate` stack `lox.spool` and `lox.strip` under it. No method passes `in_shardings` or `out_shardings`: inputs arrive already placed, and every output takes its layout from the `pin` it ends with. On `anakin` that covers `init`, `train`, `evaluate` and `close`. `quadinaros` jits `fit` and `evaluate`, and its `train` updates the dataset on the host before calling `fit`. `sebulba` jits `unroll`, `update` and `assess`, which its actors' threads and its learner call. `rollout` stays a plain method on every podracer, for an artisan that drives the loop itself, and `type(podracer).train.lower(podracer, ...)` shows a compiled program.

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

`hangar/main.py` splits `num_steps` into `num_epochs` epochs of the same whole number of updates, and logs `updates × batch_size`. It never asks a podracer to round a step budget.

The podracer counts steps into `algorithm_state.step`. The online podracers add `num_envs × num_agents` after every environment step. `quadinaros` adds `batch_size` after every update.

## Sharding

The mesh is `boonta.utils.mesh(count, start, model=1)`, with axes `("data", "model")` built by `jax.make_mesh`. `model` takes `model` devices for tensor parallelism and `data` takes the rest. The configs in [`hangar/config/podracer/`](../../hangar/config/podracer) build it over every device, and `podracer.config.mesh.model=2` reshapes it.

Each podracer resolves every leaf of its state to a `PartitionSpec` in one place, `boonta.utils.partition`:

- A field marked `metadata={"axis": "batch"}` holds one row per environment and splits its leading dimension over `data`. On `anakin` and `quadinaros` the environment states, timesteps and recurrent carry are always split this way, as are the replay buffers the algorithm marks. On `sebulba` each actor keeps its environments whole, and only the learner's state is sharded.
- Everything else is replicated, unless the recipe passes `partition_rules`: ordered `(regex, PartitionSpec)` pairs matched against each leaf's path, first match winning, the way EasyLM shards. A rule names both axes of a weight, for example `P("data", "model")` for a column-parallel kernel and `P("model", "data")` for the row-parallel one after it. `data` is FSDP and `model` is tensor parallelism, so one table gives both, and an architecture with rules is sharded over every `data` device by default. Optimizer moments, target and reference parameters live under the same paths and follow the same rules. On a leaf marked `batch`, the mark keeps the leading dimension and the rule fills in the rest, so a catch-all `(".*", P())` never replicates an environment and the KV cache rule splits heads over `model` while its batch stays split. A rule may write `"batch"` for a batch dimension.
- Every architecture exports its rules next to its definition, such as `boonta.networks.architectures.qwen3.partition_rules`, and `boonta.networks.blocks.lora.partition_rules` covers LoRA's adapters. A `model` axis above one refuses to run without rules, a leaf no rule matches is an error instead of a silent replica, and a rule that cannot split a dimension evenly names the leaf.
- Specs drop `model` when it has size one, so a recipe without rules compiles to the same program as before.

Each podracer pins its state with `self.pin`, which resolves the shardings from the mesh and the partition rules at trace time and constrains every leaf to them with `boonta.utils.pin`: at the start of every update, so each loop carries exactly the layout the rules give, and on what `train`, `evaluate` and `init` return. `boonta.utils.put` does the same with `jax.device_put` outside `jit`, for `quadinaros`'s eager `init` and `sebulba`'s hand-over to the learner. A missing or uneven partition rule therefore fails at the first trace, in `init`. Constrained only at its boundary, XLA kept the parameters sharded through the loops but chose its own layout inside them, sharding leaves the rules replicate and gathering them again at the end. Pinned at the start of every update, nothing is gathered outside the loops. XLA gathers what the rollout needs once per update, before the rollout scan, so acting never gathers per environment step, and during the rollout each device holds those weights whole along `data`.

## pit and lap

All three podracers take two hooks, `pit` and `lap`. Both default to doing nothing, and `None` means the same as `Pit()` or `Lap()`.

A `pit` is a [`Pit`](podracer.py) with three methods, like an algorithm that acts on the environment instead of the agent:

- `init(key, state)` returns the pit's own state, built from the podracer's starting state. The podracer keeps it in `state.pit_state`, replicated. The default returns `None`.
- `update(pit_state, key, transitions, algorithm_state)` learns from the rollout and returns the new pit state. It fires after every `algorithm.update`, with the transitions that update trained on and the updated algorithm state. It never touches the environment. On `quadinaros` the transitions are the sampled batch.
- `pit(key, state)` acts: it reads `state.pit_state` and rewrites the environment, such as a curriculum choosing the next level or a league seating an opponent. It fires once per update, at `init`, after every `update`, and after `evaluate` restarts the environment. It never sees transitions. It may also rewrite `state.algorithm_state`, as the league does to restart a learner it just admitted. Since `evaluate` fires it again without an update in between, the call must give the same result when repeated.

A `lap` is a [`Lap`](podracer.py): a pit without `update`. `init(key, state)` returns its state, which the podracer keeps in `state.lap_state`, and `lap(key, state)` fires after every environment step, and at `init` and `evaluate` just before `pit`. Use it for state the environment must see every step. It needs no `update` because it learns and acts in the same place: the call can read the latest timestep and rewrite `state.lap_state` itself. Both states are replicated unless their own dataclass fields mark an axis, so a lap that keeps one value per environment marks that field `metadata={"axis": "batch"}`.

Each hook must fire in all those places. `evaluate` restarts the environment at its default settings, so a `pit` skipped there would evaluate a game training never played. A recipe passes the hooks by returning `pit` or `lap` in its dict. A [curriculum](../curricula/curriculum.py) supplies both: it takes `(algorithm, environment, **kwargs)` and returns `(algorithm, environment, pit, lap)`. [`tests/test_podracers.py`](../../tests/test_podracers.py) counts the calls on every podracer.

## anakin

`init`, `train` and `evaluate` are each one `jax.jit` call. `train` scans over updates, and each update scans `num_steps` environment steps and then calls `algorithm.update`. The environments are split across the mesh, and parameters are replicated unless partition rules shard them.

## sebulba

Each actor device owns its own environments and steps them in its own thread. `num_envs` is per actor, so an update trains on `actors × num_envs` environments. The learner takes one rollout from every actor, in actor order, and joins them along the environment axis.

The lag is fixed at one update. After every update the learner puts its `algorithm_state` in every actor's queue, and each actor takes from its queue before every rollout except the second. So rollout *k* is collected with the parameters of update *k*−2 while the learner trains update *k*−1. Nothing depends on thread timing, and a run is reproducible at a fixed seed. The queues are rebuilt at every `train` call, so changing `num_epochs` changes results.

The state is `SebulbaState(actors, algorithm_state, pit_state)`. `algorithm_state` and `pit_state` exist once, as the learner's. Each `ActorState` holds only the actor's timestep, environment state, recurrent carry and `lap_state`, and hands the learner the carry it started and ended each rollout on, so every actor keeps its own lap state across rollouts. `pit.update` runs on the learner after every update, and the learner sends the new `pit_state` to the actors along with `algorithm_state`, with the same lag. The `pit` call and `lap` run on the actor, and what they write into `algorithm_state` or `pit_state` is dropped after the rollout, except the step count. `evaluate` runs on the first actor. `sebulba` rejects `Population`, and so the league's `Roster`.

The default config puts the learner and one actor on device 0. To give them separate GPUs, move the actor mesh:

```bash
uv run boonta podracer=sebulba podracer.config.actor.start=1
```

## quadinaros

`quadinaros` trains from a [dataset](../datasets/README.md). `train` first calls `dataset.update` on the host, then runs `fit`, a jitted scan in which every update calls `dataset.sample` for a batch of `batch_shape`. The dataset state rides in the podracer state and is split across devices. A `pit` that rewrites `dataset_state` changes what the next update samples. Training takes no environment steps, so `lap` fires only at `init` and in `evaluate`. `close` also closes the dataset.

## Several machines

Set `num_processes` to the total number of processes and start one per GPU, for example with `srun --nodes=2 --ntasks-per-node=4 uv run boonta num_processes=8`. `main.py` then calls `jax.distributed.initialize`, and the default meshes of `anakin` and `quadinaros` span every device of every process. `sebulba` runs on one machine. See [`hangar/README.md`](../../hangar/README.md).

## Add a podracer

A new podracer is a `@dataclass(eq=False)` taking `(config, algorithm, environment, ..., pit, lap, partition_rules)` that implements the methods above, jits the ones that compile with the podracer as a static argument, and pins its state where its loops begin and its compiled methods return. It gets a config file in `hangar/config/podracer/` whose `_target_` is the class.
