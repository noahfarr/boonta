# datasets

A dataset is a fixed set of recorded transitions that the [`quadinaros`](../podracers/README.md#quadinaros) podracer trains from. It stands where the environment stands in online training. The interface is in [`dataset.py`](dataset.py).

| Method | Runs | Does |
| --- | --- | --- |
| `init()` | once, on the host | returns the dataset state |
| `update(state, key, sharding)` | on the host, once per `train` call | returns the state for the next epoch |
| `sample(state, key, batch_shape)` | inside the jitted training scan | returns a batch as a `Transition` |
| `close()` | at the end of the run | stops any background work |

The state rides inside the podracer state and is split across devices along `data`. So mark the fields of a state dataclass with `metadata={"axis": "data"}`. `sample` must be pure JAX, because it runs inside `jax.jit`. `update` runs outside it, just before each `train` call's jitted scan, so one state lasts one epoch. Raising `training.num_epochs` refreshes it more often.

A batch is a `Transition(first, second)`, the same shape online algorithms see. `first` is the timestep acted on, with `first.terminated` set at episode starts. `second` holds the action taken, the reward and how the step ended.

## Minari

[`minari.py`](minari.py) reads a [Minari](https://minari.farama.org) dataset. Pick one with the `dataset` config group:

```bash
python hangar/main.py algorithm=bc environment=brax/mujoco/hopper
```

`bc` and `iql` load `dataset: minari/mujoco/expert` by default, which reads `mujoco/<env_id>/expert-v0`. Its config is [`hangar/config/dataset/minari/mujoco/expert.yaml`](../../hangar/config/dataset/minari/mujoco/expert.yaml):

```yaml
namespace: minari
dataset_id: mujoco/${environment.env_id}/expert-v0
kwargs:
  pool_size: 0
  num_devices: ${podracer.config.mesh.count}
```

`pool_size` decides how much is in memory.

- `0`, or at least the dataset's size, loads every transition once. `update` returns the state unchanged.
- A smaller number streams. Each `update` returns a pool of that many transitions, made of whole random episodes with only the last one cut short. A background thread stages the next pool while the current epoch trains. Pools depend only on the keys `update` receives.

Both trim the rows to a multiple of `num_devices`, so the state splits evenly. `sample` draws single transitions uniformly, so `batch_shape` must be `(batch,)`. `update` replaces the whole state, so a `pit` that writes into a streamed pool loses the write at the next `train` call.

## Kinetix

[`kinetix.py`](kinetix.py) reads Kinetix trajectory shards through Kinetix's own `TrajectoryDatasetManager`. Its state is empty. Each `sample` pulls the manager's next batch onto the device through `io_callback` and renders observations from the stored environment states. Batches are whole trajectories, so `batch_shape` must be `(batch_size, length)` as the manager yields them. It is used by the `recurrent_bc` recipe for Kinetix with `dataset=kinetix/offline_m` or `dataset=kinetix/offline_s`, which need `dataset.dataset_id` set to the shard folder.

## Add a dataset

1. Write a class with the four methods. The `Episodes` dummy in [`tests/dummies.py`](../../tests/dummies.py) is a complete in-memory example:

   ```python
   @struct.dataclass(frozen=True)
   class EpisodesState:
       transitions: Transition = struct.field(metadata={"axis": "data"})


   class Episodes:
       def __init__(self, transitions: Transition):
           self.transitions = transitions

       def init(self) -> EpisodesState:
           return EpisodesState(self.transitions)

       def update(self, state, key, sharding) -> EpisodesState:
           return state

       def sample(self, state, key, batch_shape) -> Transition:
           batch_size, *_ = batch_shape
           count, *_ = state.transitions.second.reward.shape
           index = jax.random.randint(key, (batch_size,), 0, count)
           return jax.tree.map(lambda leaf: leaf[index], state.transitions)

       def close(self) -> None:
           pass
   ```

   A dataset that refreshes its data should place the new state with the `sharding` it is given, as `Minari.stage` does with `jax.device_put`.

2. Register a constructor in `registry` in [`__init__.py`](__init__.py). `datasets.make(namespace, dataset_id, kwargs=...)` calls it as `registry[namespace](dataset_id, **kwargs)`.

3. Add a config file under [`hangar/config/dataset/`](../../hangar/config/dataset) with `namespace`, `dataset_id` and `kwargs`. A recipe builds it with `datasets.make(**cfg.dataset)` and returns it under the `dataset` key, as [`bc_mujoco.py`](../../hangar/recipes/bc_mujoco.py) does.

4. Test it in [`tests/test_datasets.py`](../../tests/test_datasets.py). Its tests write a small Minari dataset with `publish` from `tests/dummies.py` and run each check on a whole and a streamed source. The podracer side is in [`tests/test_podracers.py`](../../tests/test_podracers.py).
