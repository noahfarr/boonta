# datasets

A dataset is a fixed set of recorded transitions that the [`quadinaros`](../podracers/README.md#quadinaros) podracer trains from. It stands where the environment stands in online training. The interface is in [`dataset.py`](dataset.py).

| Method | Runs | Does |
| --- | --- | --- |
| `init()` | once, on the host | returns the dataset state |
| `update(state, key, sharding)` | on the host, once per `train` call | returns the state for the next epoch |
| `sample(state, key, batch_shape)` | inside the jitted training scan | returns a batch as a `Transition` |
| `close()` | at the end of the run | stops any background work |

The state rides inside the podracer state and is split across devices along `data`. So mark the fields of a state dataclass with `metadata={"axis": "data"}`. `sample` must be pure JAX, because it runs inside `jax.jit`. `update` runs outside it, just before each `train` call's jitted scan, so one state lasts one epoch. Raising `num_epochs` refreshes it more often.

A batch is a `Transition(first, second)`, the same shape online algorithms see. `first` is the timestep acted on, with `first.terminated` set at episode starts. `second` holds the action taken, the reward and how the step ended.

## Minari

[`minari.py`](minari.py) reads a [Minari](https://minari.farama.org) dataset. Minari comes with the `minari` extra, and the module imports it only when a dataset is built. Pick one with the `dataset` config group:

```bash
uv run boonta algorithm=bc environment=brax/mujoco/hopper
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

## AgentTrove

See the [measured role, token-budget, and trajectory-quality audit](../../docs/agent_trove_audit.md)
before treating the merged data as clean, command-only demonstrations. The
reproducible sampling tool is `hangar/scripts/audit_agent_trove.py`.
The [complete-trajectory readiness results](../../docs/agent_trove_readiness.md)
describe the current adapter and the retained action budget. The
[pretraining preparation report](../../docs/agent_trove_pretraining.md) documents
full-corpus preprocessing, cached sampling, experimental controls, and memory limits.
The [full-corpus comparison](../../docs/agent_trove_full_corpus.md) reports all
1,696,847 rows and motivates `hyperparameters=recurrent_bc/agent_trove_192m`: a
191.8M native Qwen3 at 16K with explicit cache/tokenizer paths. No training was
launched; target-GPU memory must still be checked.

[`agent_trove.py`](agent_trove.py) streams the Hugging Face
[AgentTrove](https://huggingface.co/datasets/open-thoughts/AgentTrove) training split
into a bounded token pool. Install the `agent-trove` and `hangar` extras, then run:

```bash
uv run --extra agent-trove --extra hangar boonta algorithm=recurrent_bc environment=agent_trove
uv run --extra agent-trove --extra hangar boonta algorithm=recurrent_bc environment=agent_trove hyperparameters=recurrent_bc/agent_trove_0_6b
```

The default `hyperparameters/recurrent_bc/agent_trove.yaml` and optional
`hyperparameters/recurrent_bc/agent_trove_0_6b.yaml` implement the
12-layer/768-wide and 28-layer/1024-wide Qwen3 configurations respectively,
with a 151,936-token vocabulary and 4,096-token context.
The environment selects `cell=qwen3` and `stack=qwen3`; the hyperparameters set
the attention dimensions, FFN width, layer count, vocabulary and compute dtype.
The recipe builds Boonta's native `Qwen3` torso with `instantiate(cfg.stack)`,
using causal grouped-query attention, Q/K normalization, RoPE, RMSNorm, and SwiGLU.
Both presets initialize weights randomly; only the Qwen tokenizer is downloaded. Embedding and output
weights are separate, so the preset names do not imply exact parameter counts.
Parameters remain float32 with bfloat16 computation. The RoPE wavelength is
1,000,000; these use the library's default Flax initializers.
The default batch contains one sequence per device. Any overridden batch size
must be divisible by the device count.

Rows may contain `messages` with `role`/`content` or `conversations` with
`from`/`value`. Turns are serialized as Qwen ChatML, preserving tool responses.
Null or empty trajectories are skipped; nullable message fields fall back to
their ShareGPT equivalents. Unknown schemas and malformed messages still raise.
By default only assistant bodies and their end markers receive loss; set
`dataset.kwargs.assistant_only=false` for CLM on all nonpadding next-token targets.
Both modes use the same trajectory selection, serialized inputs, targets,
padding and resets; only the loss mask changes.

The `complete_trajectory` policy keeps only whole recorded interactions whose
serialized length is at most the context length. The first non-system message
must be a nonempty user task and the last non-system message must be assistant.
At least one assistant target must exist. This means complete recorded history,
not independently verified task success. Unknown-outcome episodes remain eligible.
Long records are excluded whole, never split or truncated. Each accepted record
gets its own padded window with one reset at the beginning, preserving every
message preceding every assistant target. No unrelated histories are joined.
Prefix-plus-history compression is not implemented in this baseline.
This uses ChatML directly rather than an instruction model's thinking-mode chat template.

`dataset.kwargs.pool_size=1024` bounds the number of tokenized windows held in
memory; it must be divisible by the device count. Each training epoch refreshes
the pool on the host. Legacy updates sample uniformly with replacement; the
192M/pilot presets instead consume deterministic permutation pools once each.
With `cache_dir`, metadata is indexed once on the host; seeded pool refresh
samples eligible index entries uniformly before reading selected token spans. The raw stream uses a bounded shuffle and its first pool
is not a uniform sample of a source-ordered full corpus. This is sampled-pool training, not a full pass over the split.
Increase `num_epochs` to refresh more often. `num_steps` counts token positions
including masked positions. `dataset.kwargs.revision` can pin a dataset revision.
The adapter also accepts a local `tokenizer.json` path as `tokenizer_id`.


For corpus preparation, use `hangar.scripts.preprocess_agent_trove` with a pinned
revision, local Qwen3 tokenizer JSON, and expected row count. It emits exact
4K/8K/16K/32K accounting plus a ragged binary cache. `--accounting-only` omits
binaries; `--resume` continues an interrupted run. Point
`dataset.kwargs.cache_dir` to a completed cache and use `split=train` or
`split=validation`. Incomplete caches, mismatched tokenizer fingerprints, and
contexts above the cached limit are rejected. Identical conversations share a
deterministic split assignment; related but different traces can cross splits.

Before training, run the offline data-readiness check against the cached audit
sample to measure length distributions, retained observation/action/assistant
token budgets by source and outcome, context coverage `q`, and CLM/BLM parity:

```bash
.venv/bin/python -m hangar.scripts.check_agent_trove_readiness \
  --sample outputs/agent_trove_audit/sample \
  --tokenizer /path/to/Qwen3-0.6B/tokenizer.json
```

The check writes `readiness.json` and `readiness_records.json`. Its exact
deduplication is within the cached sample; it is not a full-corpus deduplication
pass. A 4,096-token complete-only baseline retains a small and source-biased
fraction of the sampled action budget. Inspect those results before selecting
the model size or launching a long run.

This recipe uses AdamW and logs `actor/likelihood` (masked training negative log
likelihood; lower is better). It has an initialization-only environment for the
offline podracer. The run score uses `actor/log_likelihood` (higher is better),
the negative of that loss. The recipe does not support interactive evaluation
or held-out metrics.
Model and optimizer checkpoints use the usual checkpoint artisan; the current
pool and permutation cursor are included in dataset state. The paired presets
use a seeded permutation without replacement; the legacy replacement mode
remains available. Resume uses ordinary epoch-boundary checkpoints.

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

       def sample(self, state, key, batch_shape, *, step=None) -> Transition:
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
