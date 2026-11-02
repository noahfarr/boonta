# AgentTrove CLM/BLM preparation

No pretraining has been launched. Full preprocessing verified all **1,696,847**
rows at `open-thoughts/AgentTrove` revision
`b395a4307a2bc9950a90dc899438f149e115fc60`. The corpus contains **17,978,954,637
serialized tokens** and **6,855,111,869 assistant targets** before filtering.
The source repository has 38 Parquet shards totaling about 19.55 GB (metadata).
The eligible cache through 32K contains 80,298,005,270 binary bytes, plus metadata.

See the [full-corpus comparison and recommendation](agent_trove_full_corpus.md)
and [machine-readable statistics](agent_trove_statistics.json). Full histograms
are in `outputs/agent_trove_full/statistics.json`. Every row and the entire stored
BLM mask were reconciled; paired sampling passed on four logical CPU devices in
both splits at all four contexts. Target-GPU memory remains unverified.

## Reproduce accounting and preprocessing

Use the existing `agent-trove` optional dependencies and Boonta environment:

```bash
.venv/bin/python -m hangar.scripts.preprocess_agent_trove \
  --output outputs/agent_trove_full \
  --tokenizer /path/to/Qwen3-0.6B/tokenizer.json \
  --revision b395a4307a2bc9950a90dc899438f149e115fc60 \
  --expected-rows 1696847 --executor processes --workers 8 --cache-max-length 32768
```

The output directory must be new, or use `--resume` to continue an incomplete run
with matching dataset/tokenizer provenance. Resume reconstructs accounting from
committed row metadata and trims only uncommitted binary/JSONL tails. `--accounting-only` omits token binaries;
`--limit N` explicitly produces a partial smoke result. Accounting processes every
untruncated source row, validates roles, and aborts on unsupported data rather than
silently dropping it. Full runs stream Parquet through Hugging Face Datasets;
there is no dependency on the truncated dataset viewer. Tokenizer-level
truncation and automatic padding are rejected to prevent silent message changes. No model weights load.
Worker results remain in source order, with only 32 rows in flight. Threads and
spawned tokenizer processes produce the same ordered output.

Outputs are `statistics.json`, `report.md`, `manifest.json`, `records.jsonl`,
`tokens.bin` (little-endian int32), and `mask.bin` (uint8). Binaries contain only
eligible complete records up to `--cache-max-length`; statistics cover every row.
Verify a completed cache before any run:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 .venv/bin/python \
  -m hangar.scripts.check_agent_trove_readiness \
  --cache outputs/agent_trove_full --tokenizer /path/to/Qwen3-0.6B/tokenizer.json
```

This reads all metadata and the full mask binary, reconciles aggregate counts,
and tests paired sampling at all four contexts using JIT and named sharding.
It writes `cache_readiness.json`; it does not initialize or train a model.

The manifest is incomplete until EOF and expected-row-count verification. Partial
and accounting-only artifacts cannot be used as training caches. Local JSONL
input is also supported, with a recorded file hash; its revision is supplied by
the caller and must not be treated as independently verified provenance.

Exact length histograms and retained histograms, source/outcome/split budgets,
metadata availability, raw result values, serialized token retention, recognized
command retention, padding, and supervision densities are machine readable.
No outcome filtering or deduplication occurs. Unknown outcomes stay unknown;
timeouts and recognized infrastructure errors remain separate from failure.
Conflicting explicit labels remain conflicts. Command accounting is a heuristic
for final `commands` lists, not a universal parser for arbitrary tool-call formats.

## Full-corpus composition

At 4K/8K/16K/32K the retained BLM target budgets are 420.0M / 1.429B / 3.708B /
6.223B, and recognized command retention is 3.81% / 16.29% / 51.56% / 89.29%.
Padding consumes 33.10% / 47.36% / 56.10% / 67.88% of allocated positions.
These are exhaustive measurements, not extrapolations from the earlier sample.

Length selection remains a research limitation: at 16K just 188/4,077 SWEGym,
10,447/36,533 r2egym, and 33,292/78,246 swesmith records survive. At 32K SWEGym
still retains only 1,841 records. Full source-specific counts and retained
length distributions are in the linked statistics. The maximum serialized
record is 890,996 tokens; the nonempty median is 8,780 and the 95th percentile
is 28,833.

There are 128,435 rows without supported embedded messages. `exp_rle` and
`exp_rpt` account for 118,845 of these; neither cohort supplies trainable
messages through the existing adapter. No history is fabricated from other
metadata. The earlier sample audit remains documented separately in
[agent_trove_readiness.md](agent_trove_readiness.md).

Only 218,706 rows have a non-null `result`. Explicit labels produce 18,159
reported successes, 1,328 failures, 181,447 timeouts, 17,186 recognized Daytona
infrastructure errors, and 1,478,727 unknown outcomes. Unknown includes other
unclassified exception types, preserved verbatim in metadata. Missing results
never imply failure. Source labels are not independently verified task outcomes.

## Boonta integration and experimental controls

The existing `AgentTrove` Dataset protocol remains `init/update/sample/close`.
Set `dataset.kwargs.cache_dir` to the completed cache, retain the matching local
`tokenizer_id`, and set `revision` to the pinned SHA. The adapter checks tokenizer
and dataset revision agreement, rejects contexts above the cache limit, and
continues to return fixed-shape `Transition` pools to Quadinaros. Host processing
stays outside JIT; pool placement and JIT sampling retain existing sharding.

Cache initialization scans metadata into a compact host index (20 bytes per
input row of capacity). Seeded pool refresh samples eligible index entries
uniformly without replacement, then reads selected token spans through memory
maps. It does not repeatedly parse the entire metadata file at each refresh. Taking the first pool
from a bounded shuffle would bias sampling toward the start of a source-ordered
corpus. Refresh is deterministic given the same seed/key, split, pool size, and
context. Within a pool, JAX samples with replacement. Set `num_epochs` deliberately:
the framework default of ten refreshes exposes at most ten pools, regardless
of the requested token budget. Report this restriction and the repetition rate. A refresh is not an exact
training epoch; report actual sampled positions and examples, not nominal epochs.
The original raw-stream path remains available but should not be used to claim
uniform full-corpus sampling.

Both objectives use the same serialization, complete-record eligibility, ordered
pool construction, sampled indices, shifted targets, initial reset, and padding.
Only `dataset.kwargs.assistant_only` changes. CLM weights every valid next-token
target; BLM weights assistant body tokens, including ChatML end/handoff markers
and trailing newline. Assistant role headers are fixed framing, not BLM targets.
User/tool tokens remain visible in both objectives. The final serialized token
is a target, not an input; one padding position remains when serialized length
equals context. “Complete” means the recorded interaction is preserved, not that
task success or upstream episode completeness was verified.

A SHA256 hash of the normalized conversation assigns approximately 1% to
validation. Identical conversations cannot cross splits, regardless of source or
metadata. Related tasks with different traces can cross splits; task-grouped
validation would need a trustworthy cross-source task identifier. No evidence
supports calling this a task-disjoint split.

`RecurrentBC.update` already implements `sum(weight * NLL) / max(sum(weight), 1)`.
CLM therefore averages over valid targets; BLM averages over assistant targets.
Averaging BLM over all valid positions would rescale its gradient by assistant
density and is a different convention. All-zero masks produce zero weighted
loss. Use entropy coefficient zero, the same model and optimizer initialization
key, data keys, context, split, batch size, pool size, schedule, and update count.
The recipe's schedule counts allocated token positions, including padding; do
not independently retune either schedule using its supervised-token count.

## Model candidates and compute estimates

The recipe now instantiates Boonta's native `Qwen3` directly. No Hugging Face
Transformers model, pretrained weight loading, separate head, or role embedding
was introduced. Shape-only parameter counts for the actual untied input embedding
and categorical output projection, vocabulary 151,936, 12 layers, head dimension
64, are:

| Width | Heads / KV groups | FFN width | Parameters | Input + output embeddings |
| --- | --- | ---: | ---: | ---: |
| 384 | 6 / 2 | 1,024 | 135,572,352 | 116,686,848 |
| 512 | 8 / 2 | 1,536 | 191,772,672 | 155,582,464 |
| 768 (existing preset) | 12 / 4 | 2,048 | 308,891,904 | 233,373,696 |

Recommend the **191,772,672-parameter, 16K** configuration in
`hyperparameters/recurrent_bc/agent_trove_192m.yaml`, conditional on a target-GPU
memory check. It balances the requested size range against the actual **3.671B
BLM / 8.494B CLM targets in the 16K train split**. Its untied embeddings account
for 155.6M parameters; 36.2M remain in the transformer. The 135.6M alternative
reduces parameter memory but does not eliminate vocabulary-logit costs. This is
an initial research choice, not evidence of BLM-optimal scaling. Use 8K for a
separate memory-constrained pilot and label its reduced source coverage.

The new preset requires explicit `dataset.kwargs.cache_dir` and matching local
`dataset.kwargs.tokenizer_id`; it pins the corpus revision. It sets seed 42,
2^34 allocated positions (2^20 trajectory draws), 4,096 pool refreshes, a 1,024-row
pool, one sequence per device, and the inherited AdamW/cosine schedule (LR 3e-4,
weight decay 0.1, clip 1.0, entropy coefficient zero). CLM overrides only
`dataset.kwargs.assistant_only=false`. Use the common override `scoring=final` and compare final or matched-step
checkpoints, instead of choosing each objective's best training loss. The fully resolved training configurations
are tested for equality after removing that one flag; the actual recipe factory
is shape-checked at 191,772,672 parameters.

There are 256 global trajectory draws per refresh. For power-of-two device counts
up to 256 the preset's update budget divides exactly. At larger device counts,
adjust the common budget/refresh count to avoid zero or rounded update cycles.
This samples about 0.888 nominal passes over the 16K train trajectories and is
not exhaustive coverage: JAX samples with replacement within finite pools.
Expected targets are approximately 3.260B BLM and 7.542B CLM under this schedule;
actual sampled counts must be logged by a future training run. Both objectives
must use identical keys and the complete same schedule.

The conventional dense-CLM heuristic of about 20 training tokens per parameter
([Hoffmann et al., 2022](https://arxiv.org/pdf/2203.15556), Table 3)
would imply about 3.84B tokens for 191.8M parameters. It is not an established
scaling law for masked behavioral cloning: neither counting only action targets
nor counting all conditioning tokens establishes a BLM-optimal size.

The raw accounting JSON uses `6 * 200M * allocated_positions` for a hypothetical
dense pass. The committed summary additionally estimates the chosen 191.8M
configuration and includes attention: `24 * layers * width * context * positions`
assumes training costs 3× forward with the current doubled key axis. The planned
2^34-position run is about **1.98e19 dense + 4.15e19 attention FLOPs**, before
additional rematerialization, optimizer, and communication overhead. These are
analytic estimates, not measured GPU throughput or GPU-hour promises.

Both objectives execute the same dense graph; masking loss does not
proportionately reduce compute. Token storage is 5 bytes per serialized token
plus row metadata. Float32 parameters, Adam moments, and gradients for 191.8M
parameters occupy roughly 3.07 GB before activations and temporary buffers.

## Attention, padding, and remaining checks

SelfAttention appends the full existing KV cache to each fresh sequence even
though the initial cache is invalid. A length-L update with context L attends to
2L key slots and constructs/broadcasts an explicit `[B, heads, L, 2L]` mask.
For 8 heads and batch 1, its logical bool mask sizes at 4K/8K/16K/32K are
256 MiB / 1 GiB / 4 GiB / 16 GiB. Actual materialization depends on compilation;
XLA attention can also need much larger float32 score buffers. At 32K the
vocabulary projection output is about 9.27 GiB in bfloat16 per sequence.
Boonta's categorical distribution casts these logits to float32, making its
logical logits array about 18.55 GiB, before normalization/entropy temporaries. BF16 KV
carry for the 512-wide candidate is about 192 MiB at 32K per sequence.

NVIDIA capability and shape checks may select cuDNN attention; CPU falls back to
float32 XLA attention. No GPU memory or distributed-throughput measurement has
been performed. Long-context feasibility is not certified by a tiny CPU test.
Qwen3 rematerializes layers, but that does not remove vocabulary-logit or
explicit-mask costs. Reset segments isolate attention; positional offsets do
not reset within a packed segment. No packing is introduced, so unrelated
trajectories never share a sequence and each starts at position zero with fresh
carry.

Power-of-two length buckets would reduce full-corpus allocated positions from
49,997,381,632 to 22,624,621,568 at 32K, or from 19,545,210,880 to
11,718,251,520 at 16K. This is an estimate, not implemented variable-shape
batching. Quadinaros compiles a fixed batch shape, and algorithm schedules count
that shape. A bucket scheduler would need separate compiled shapes and identical
paired ordering without accidentally giving the long bucket a different context
contract. Do not concatenate trajectories under ordinary causal attention.

Tests cover exact cached/raw serialization and shifting, masks, padding/reset,
paired objective inputs, deterministic sampling, full-range indexed pool selection,
cache certification/fingerprints, accounting conservation, JIT, existing named
sharding, native Qwen3 causality and batch isolation, and weighted RecurrentBC.
Named sharding was also tested with two and four logical CPU devices.
Multi-host/GPU execution and target-GPU long-context memory remain unverified.
The offline recipe logs training likelihood; a held-out likelihood evaluation
loop remains necessary before reporting validation results.

Evaluate both models with both all-token and assistant-token NLL on shared
validation histories. Their training NLLs use different denominators and should
not be compared directly. Recorded role fidelity, source annotation errors,
external payload recovery for message-less rows, exact deduplication, and
task-disjoint validation remain outside the completed structural checks.
