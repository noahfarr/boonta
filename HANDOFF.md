# AgentTrove pretraining handoff

## User intent and working rules

Prepare a controlled from-scratch CLM/BLM experiment inside native Boonta.
The user wants to understand and manually launch every job. Do not submit Slurm
jobs, allocate GPUs, or launch pilot/main training. Tiny synthetic CPU unit tests
are acceptable. Prefer normal `boonta train` and portable preprocessing commands;
do not add runners, preflight approval gates, or a second training framework.
Keep changes small and explain why any framework interface must change.

The user is concerned about PR size and unnecessary infrastructure. Detailed
held-out evaluation, bespoke checkpoint scheduling, language-policy adaptation,
and terminal RL were removed from this branch and deferred. Do not reintroduce
them without discussion. User-owned `paper/` and `COLM_2026_Template.zip` were
preserved; do not edit them as part of framework cleanup.

## Branches and current edits

- Working branch: `boonta-blm`.
- Last pushed commit before this handoff: `09d2cf90`, scope cleanup.
- Remote backup: `boonta-blm-evaluation-rl-backup-20261103`, preserving the broad
  evaluation/RL prototype at `28abcb1f`. That prototype had unresolved integration
  issues and is not a ready RL pipeline.
- Current follow-up edits remove two `hasattr` shortcuts and add this handoff.
  Inspect `git status` to establish whether they have since been committed.
- RecurrentBC now uses ordinary `dist.log_prob(targets)` and `dist.entropy()`;
  no chunked-head capability check. The chunked distribution implements those
  methods. Separate calls can mean extra projection work; no measured performance
  claim. Do not silently reintroduce a special-case branch to optimize it.
- Dataset protocol explicitly accepts `sample(..., *, step=None)`. Quadinaros
  always passes allocated training positions. AgentTrove permutation mode uses
  them for indexed batches; random samplers ignore them. Kinetix, Minari, test
  datasets and protocol documentation were updated. `sample_at` is removed.

## Retained implementation

- `boonta/datasets/agent_trove.py`: native Dataset, complete-trajectory ChatML
  serialization, shifted targets, masks, ragged-cache reader, deterministic
  sampling without replacement, checkpointed pool cursor.
- `boonta/networks/chunked_categorical.py`: same Dense parameters, position-tiled
  vocabulary likelihood and entropy. No extra model heads or parameters.
- `hangar/recipes/recurrent_bc_agent_trove.py`: native random Qwen3 + existing
  RecurrentBC + AdamW + Quadinaros; initialization-only offline environment.
- `hangar/scripts/download_agent_trove_tokenizer.py`: pinned tokenizer only,
  checksum verification; no pretrained weights.
- `hangar/scripts/preprocess_agent_trove.py`: full streamed source accounting and
  reusable cache; resumable, ordered worker results, host preprocessing.
- `hangar/scripts/group_agent_trove.py`: separate split overlay, no token copies.
- Existing audit and readiness scripts/tests retained.
- `artisan=agent_trove_pretraining` now only saves ordinary epoch-boundary
  algorithm/dataset state checkpoints via existing artisans/loggers (last two).
- `hangar/train.py`, GRPO, checkpointer, Orbax and system monitor were restored to
  their pre-feature implementations. No custom milestone scheduler remains.

## Data and experimental semantics

Source: `open-thoughts/AgentTrove`, pinned revision
`b395a4307a2bc9950a90dc899438f149e115fc60`, 1,696,847 rows.
Qwen tokenizer: `Qwen/Qwen3-0.6B`, revision
`c1899de289a04d12100db370d81485cdf75e47ca`; tokenizer.json SHA256
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.

Raw records contain `conversations` or `messages` plus source/task/model/result
metadata. Observations can be recorded as `user`, including terminal output and
parser errors. Assistant messages can contain reasoning, plans and JSON command
payloads. BLM supervises the entire assistant body, not commands alone.

Preserve message contents and order. Eligibility requires initial nonempty user
message after optional system messages, final assistant message, assistant
prediction targets, and whole serialized length within context. No truncation,
windowing, packing, success filtering, or deduplication. Structural completeness
is not verified task success. Unknown outcomes remain unknown.

Both objectives use identical histories, shifted targets, order, seeds, splits,
optimizer, schedule and allocated positions. Only `assistant_only` changes:
false = CLM, true = BLM. Assistant body/end marker/newline have BLM weight;
role headers do not. Padding has zero weight. RecurrentBC normalizes by the sum
of supervised weights, separately for each objective.

The binary cache stores int32 token IDs and uint8 assistant masks, plus JSONL
record offsets/metadata, manifest and accounting. Separate file spans remain
independent training sequences. Through 32K: 16,059,601,054 cached tokens,
80,298,005,270 binary bytes, plus metadata. Full source: 17,978,954,637 serialized
tokens. Reports: `docs/agent_trove_full_corpus.md`, `docs/agent_trove_statistics.json`.
Earlier sample reports are historical, not full-corpus measurements.

Context: eligible trajectories / BLM targets / CLM targets, before split:
- 4K: 374,361 / 420,001,511 / 1,025,796,151.
- 8K: 744,939 / 1,428,647,439 / 3,212,592,581.
- 16K: 1,192,945 / 3,708,150,333 / 8,580,584,674.
- 32K: 1,525,799 / 6,222,812,257 / 16,058,075,255.

Grouped 16K train: 1,182,693 trajectories, 3,676,006,457 BLM targets,
8,500,636,866 CLM targets. Original hash-split counts in older reports differ.
Grouped validation is NOT certified task/repository disjoint: identities are
incomplete. This blocks scientific conclusions from a main comparison, not
engineering smoke tests.

## Requested experiment and unresolved work

Native Qwen3 191,772,672 parameters: 12 layers, width 512, FFN 1536, 8 Q heads,
2 KV groups, head dimension 64, vocabulary 151936, untied embeddings. Random
initialization, one paired seed initially. AdamW LR 3e-4 cosine, decay .1,
clipping 1, entropy 0. Explicit `scoring=final` (framework default is best).

Pilot preset: 8K, 2^28 allocated positions. Main preset: 16K, 2^34 allocated
positions, requires JAX_ENABLE_X64=true for step counter. These are allocated
positions including padding, NOT equalized supervised-token budgets. Do not
launch either. GPU memory fit and actual throughput remain unmeasured.

Deferred requirements: 0/10/25/50/75/100% checkpoints, held-out token-total NLL
for assistant/command/observation targets, common tool-call validity evaluation,
certified leakage-resistant validation, and downstream existing-Boonta GRPO
comparison. Terminal RL is a follow-up, not part of this pretraining integration.
No claim that an RTX 5080 can fit the 8K/16K optimizer step yet. Attention/KV
storage remains substantial despite tiled vocabulary projection. The conventional
Chinchilla heuristic is not a validated masked-BC scaling law.

## Machines and portable commands

IAS alias `ias` logs in as stud_farr; checkout `/home/stud_farr/boonta-blm` was
created from the local branch. User ran uv sync there. Three-GPU student quota;
availability changes. Do not use other existing Boonta checkouts. User also has
an RTX 5080 machine; OS/driver/GPU functionality has not been verified.

An 81 GB Mac-to-IAS upload suffered repeated timeouts and token-file checksum
failure. Do NOT assume that uploaded cache is valid. User chose to preprocess
directly where the data will be used. Source files/caches under `outputs/` are
not in Git. Do not overwrite/delete partial remote files without discussing it.

Portable setup, on Linux/WSL2 or macOS:

```bash
uv sync --frozen --python 3.12 --extra hangar --extra agent-trove
uv run --no-sync python -m hangar.scripts.download_agent_trove_tokenizer
uv run --no-sync python -m hangar.scripts.preprocess_agent_trove \
  --output data/agent_trove_full --tokenizer data/tokenizer.json \
  --revision b395a4307a2bc9950a90dc899438f149e115fc60 \
  --expected-rows 1696847 --workers 8 --executor processes
uv run --no-sync python -m hangar.scripts.group_agent_trove \
  --cache data/agent_trove_full --output data/agent_trove_grouped \
  --tokenizer data/tokenizer.json --seed 42 --max-length 16384
```

Allocate CPUs before running on IAS; avoid heavy work on login nodes. Add
`--resume` only to incomplete preprocessing caches. Full reproducibility,
readiness and manual native-training commands: `docs/agent_trove_pretraining.md`.
The optional IAS preflight sbatch invokes the ordinary training CLI, but only
the user should submit it. No custom Python training launcher remains.

## Validation

Before the interface follow-up: 67 focused tests passed, 1 skipped; 16 dataset
tests passed on four logical CPU devices; Hydra preflight composition with the
checkpoint artisan passed. Tests use synthetic tiny CPU models, not full runs.

For current interface changes rerun:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
  tests/test_agent_trove.py tests/test_agent_trove_audit.py \
  tests/test_agent_trove_preprocessing.py tests/test_blm_experiment.py \
  tests/test_recipes.py tests/test_algorithms.py tests/test_podracers.py \
  -k 'agent_trove or chunked or without_replacement or grouped_split or large_main_budget or every_recipe_has or qwen3_cell or recurrent_bc or quadinaros' -q
```

Known local environment warnings: TFP deprecation and an unrelated incomplete
Hydra determined/submitit launcher plugin. Do not change unrelated dependencies
to suppress those warnings.
