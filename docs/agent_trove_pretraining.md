# AgentTrove CLM/BLM preparation

This branch prepares a controlled native-Boonta pretraining comparison. It does
not implement downstream RL or held-out generation evaluation. No full training
has been launched. GPU memory fit and throughput remain unmeasured.

The broader implementation is preserved on local branch
`boonta-blm-evaluation-rl-backup-20261103` at commit `28abcb1f`. Evaluation and RL
can be reviewed separately; that prototype is not certified ready for training.
Paper files are user work and remain untouched by this cleanup.

## Setup and preprocessing

Run on Linux/WSL2 or macOS, preferably where training data will be stored:

```bash
uv sync --frozen --python 3.12 --extra hangar --extra agent-trove
uv run --no-sync python -m hangar.scripts.download_agent_trove_tokenizer
uv run --no-sync python -m hangar.scripts.preprocess_agent_trove \
  --output data/agent_trove_full --tokenizer data/tokenizer.json \
  --revision b395a4307a2bc9950a90dc899438f149e115fc60 \
  --expected-rows 1696847 --executor processes --workers 8
uv run --no-sync python -m hangar.scripts.group_agent_trove \
  --cache data/agent_trove_full --output data/agent_trove_grouped \
  --tokenizer data/tokenizer.json --seed 42 --max-length 16384
```

The tokenizer downloader fetches only a pinned tokenizer and verifies its SHA256.
No pretrained weights are loaded. Preprocessing streams the pinned source,
processes every row, and writes ordered results independently of worker count.
Use a fresh output directory, or `--resume` for an interrupted preprocessing run.
Resuming replays source rows to the committed offset; it is not an instant seek.
Do not run preprocessing on a shared cluster login node; allocate CPU resources
and run the same command there. Slurm is resource allocation, not another data
pipeline. The grouped-split command runs after preprocessing completes.

The cache consists of little-endian int32 `tokens.bin`, uint8 `mask.bin`, and
`records.jsonl` indexing separate trajectories. Physical adjacency in the binary
file does not join trajectories in attention. `manifest.json` records provenance
and completion; `statistics.json` and `report.md` contain corpus accounting.
The 32K cache contains 16,059,601,054 tokens and 80,298,005,270 binary bytes.

Verify a completed cache without training:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run --no-sync python -m hangar.scripts.check_agent_trove_readiness \
  --cache data/agent_trove_full --tokenizer data/tokenizer.json
```

See [full-corpus measurements](agent_trove_full_corpus.md) and
[machine-readable statistics](agent_trove_statistics.json). Earlier audit and
readiness reports describe samples and are historical evidence, not corpus totals.

## Native dataset and loss

`AgentTrove` implements `init/update/sample/close`. Quadinaros passes the allocated training
step through the optional keyword argument `sample(..., step=...)` for indexed sampling. Host-side refresh reads selected
trajectories through memory maps and places fixed-shape pools using existing
sharding; minibatch selection remains JAX-native.

Messages retain their order and content. Qwen ChatML framing is explicit. Whole
trajectories must start with a nonempty user task after optional system messages,
end with an assistant message, contain assistant targets, and fit the context.
Oversized records are excluded, not truncated or split. No success filtering is
applied. Missing outcomes stay unknown; failures, timeouts, and recognized
infrastructure errors remain distinct. Structural completeness does not certify
successful tasks or upstream episode completeness.

Inputs and targets are shifted by one token; padding has zero weight. Each batch
row contains one trajectory with an initial reset. There is no trajectory packing
or cross-record causal attention. Assistant loss includes body tokens, ending
markers and trailing newline, but excludes role headers. User and tool messages
remain visible to both models. `assistant_only=false` selects CLM and `true`
selects BLM; only this loss-mask flag changes between paired runs.

RecurrentBC already supported weighted likelihood. Its normalization remains
`sum(weight * NLL) / max(sum(weight), 1)`: CLM averages valid targets, BLM averages
assistant targets. Target counts are logged by the dataset. There is no new
algorithm, objective, or head parameterization.

The paired presets use a seeded global permutation without replacement. Each
refresh consumes its pool exactly once. The dataset cursor is checkpointed.
Paired runs use the same epoch and update keys. Resume is supported at ordinary
epoch boundaries. The legacy replacement mode is unchanged. A framework epoch here means a
pool refresh, not a full pass through the corpus.

The grouped split links detected task copies, prompts and repository identities.
It is deterministic, but incomplete identities mean task/repository-disjoint
validation is NOT certified. The original hash split in the cache is retained for
provenance; the supplied grouped manifest overrides assignments during training.

## Model and manual smoke command

Native Qwen3: 12 layers, width 512, FFN 1536, 8 query heads / 2 KV groups,
head dimension 64, vocabulary 151936, untied embeddings: 191,772,672 parameters.
The input/output embeddings account for 155,582,464 parameters. All parameters
are initialized randomly. CLM and BLM use the same seed and configuration.

The optional chunked categorical computation preserves Dense parameter names and
values, but projects small position tiles instead of materializing all vocabulary
logits simultaneously. Dense/chunked values and gradients are tested for equality.
It does not solve the attention/KV-cache memory problem or establish GPU fit.
At 16K, one float32 full-vocabulary logit tensor alone is about 9.27 GiB per
sequence. Actual full-step GPU memory and throughput must be measured.

After preparing the data, the two-update smoke check uses the normal entry point:

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/boonta train \
  algorithm=recurrent_bc environment=agent_trove \
  hyperparameters=recurrent_bc/agent_trove_preflight \
  scoring=final logger=file seed=42 \
  dataset.kwargs.cache_dir=data/agent_trove_full \
  dataset.kwargs.tokenizer_id=data/tokenizer.json \
  dataset.kwargs.split_manifest=data/agent_trove_grouped/split.json \
  dataset.kwargs.assistant_only=false
```

For the paired smoke check change only `assistant_only=true`. The optional IAS
preflight sbatch file requests resources and invokes this same training entry
point. No Python runner or readiness approval wrapper is required.

The 8K pilot preset allocates 2^28 positions; the 16K main preset allocates 2^34.
These include padding and are matched between objectives, not normalized to equal
supervised targets. Main budgets require X64 for the global step counter. The
common optimizer is AdamW, LR 3e-4 with cosine decay, weight decay .1, clipping 1,
and zero entropy coefficient. The `scoring=final` override is explicit because
the framework default is best, not final. It does not change the loss.

`artisan=agent_trove_pretraining` now selects only ordinary epoch-boundary model,
optimizer and dataset-state checkpoints, retaining the last two. The standard
training loop is unchanged. Custom 0/10/25/50/75/100% checkpoint scheduling,
held-out token-aggregated action/command/observation NLL, valid tool-call evaluation,
and downstream RL are follow-up work. Do not treat the main preset as approval
to launch a scientific comparison before those prerequisites are ready.

Conventional dense-CLM token/parameter scaling heuristics are planning references,
not established scaling laws for masked behavioral cloning. Source selection,
padding overhead, attention cost, GPU fit and trustworthy validation remain
material constraints on the experiment.
