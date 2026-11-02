# AgentTrove audit — 2026-11-02

The initially audited adapter preserves speaker separation in the sample, but the data is
not a set of uniformly successful, command-only demonstrations. Do not equate
the advertised trajectory count with effective BC supervision.

**Historical windowing baseline:** the chunking counts below describe the initial
fixed-token-chunk adapter. Following this audit, the adapter was changed to
`complete_trajectory` windows. Use `check_agent_trove_readiness` and
`outputs/agent_trove_audit/sample/readiness.json` for the current selection policy
and retained budgets; do not treat the old chunking density as the new policy's density.
The measured follow-up is in [AgentTrove data readiness](agent_trove_readiness.md).

## Scope and reproducibility

- Dataset: [open-thoughts/AgentTrove](https://huggingface.co/datasets/open-thoughts/AgentTrove), `default/train`, 1,696,847 rows reported by the viewer.
- Sample: 1,024 rows, 16 contiguous rows at a seeded random offset in each of 64 equally sized row-index strata. Seed 42; no failed blocks or truncated cells.
- Tokenizer: Qwen3-0.6B tokenizer snapshot `c1899de289a04d12100db370d81485cdf75e47ca`; exact tokenizer SHA-256 is recorded in the JSON report.
- Serialization/mask: the actual `boonta.datasets.agent_trove.tokenize`, assistant-only, context 4096.
- Sample spans 17 `original_source` labels, including `unknown`. These are source families, not all 219 upstream repositories listed in the card.
- Artifacts: `outputs/agent_trove_audit/sample/{report,records}.json` and cached `rows_*.json`. These are ignored by Git. The report includes row indices, hashes, flags, source and outcome breakdowns, and the bootstrap intervals.
- The viewer did not return a dataset commit revision. This is a cached, dated sample of the live viewer, not a revision-pinned full-corpus scan. It cannot certify corpus-wide duplicate rates or exact token totals.
- The viewer's precomputed statistics endpoint reports only 180,240 examples and `partial=true`; its null and result counts must not be presented as statistics for all 1.7M rows. Its response is cached as `corpus_statistics.json` for inspection.

Reproduce from the repository root (the tokenizer may also be another local `tokenizer.json`):

```bash
.venv/bin/python -m hangar.scripts.audit_agent_trove \
  --tokenizer /path/to/Qwen3-0.6B/tokenizer.json \
  --blocks 64 --rows-per-block 16 --seed 42 \
  --output outputs/agent_trove_audit/sample
```

## 1. Role correctness

| Measured quantity | Sample count |
| --- | ---: |
| User messages | 9,618 |
| Assistant messages | 9,608 |
| Explicit tool/system messages | 0 |
| Messages beginning with recognized terminal/parser-feedback prefixes | 8,666 |
| Those feedback messages labeled user | 8,666 |
| Those feedback messages labeled assistant | 0 |

The format is primarily **user/assistant**, not a formal assistant-tool-call /
tool-result schema. Environment output and harness feedback arrive as `user`
turns. Initial task instructions also occupy user turns. Thus `role=user` alone
does not distinguish a human task from environment feedback; position and harness
markers do. No explicit tool-call IDs are available for reliable call/result joins.

The adapter masks all user turns, including terminal feedback. It does not
relabel user turns as assistant or supervise their text. Explicit `tool` roles
would also be masked, though none appeared here. The source convention is
consistent in the recognized feedback subset; this does not prove correctness
for every unrecognized message or every upstream repository.

A broad phrase search found 174 assistant messages mentioning terminal output or
state. These are **review candidates, not 174 mislabeled outputs**. Spot checks
showed reasoning discussing the terminal, quotes, and plans. No assistant message
began with the high-specificity feedback prefixes. Blanket keyword-based role
rewriting would corrupt valid assistant reasoning.

Recommendation: preserve the existing speaker mask. Add an independent semantic
turn type (`task`, `environment_feedback`, `assistant`, `other`) during preprocessing
for analysis, and match the training role convention at inference. Do not convert
all user turns into tool turns.

## 2. Effective supervision

| Quantity | Sample measurement |
| --- | ---: |
| Rows with trainable tokens | 928 / 1,024 (90.6%) |
| Serialized tokens, including role framing | 11,579,302 |
| Supervised next-token targets | 4,263,426 |
| Supervised / serialized tokens | **36.8%** |
| Retained windows | 3,277 |
| Allocated positions after window selection and padding | 13,422,592 |
| Supervised / allocated positions | **31.8%** |
| Recognized final command-list payload tokens | 2,007,159 |
| Command payload / supervised tokens | **47.1%** |
| Command payload / serialized tokens | **17.3%** |

The assistant mask covers reasoning, analysis, plans, command JSON, completion
claims, and end-of-message markers. It is not an action-only mask. Command-payload
tokens count the final recognized `commands` list value, including JSON and
durations; they are not just keystroke strings. Closed `<think>` draft blocks are
excluded from that *action metric*, but remain supervised by the actual adapter.
Malformed or unrecognized action formats are not counted by this heuristic.

Resampling whole sample blocks 2,000 times gives an approximate 95% interval of
**34.4–39.2%** for the supervised/serialized ratio, and **41.3–51.9%** for the
recognized command/supervised ratio. These describe sampling variability, not
undetected schema problems, missed source families, or exact corpus-wide totals.

The 96 empty rows all have `task_binary` payloads. One was inspected without
executing or extracting files: it is a compressed task archive containing
instructions, environment configuration, and tests, not a recorded trajectory.
The adapter cannot turn these into demonstrations and correctly skips them.
All sampled `exp_rle` and `exp_rpt` rows were of this type; this is a sample finding.

Training's `num_steps` counts allocated positions, including masked tokens.
Furthermore, bounded shuffled pools and uniform window sampling are not a full
uniform corpus pass. The above ratios do not imply that a particular training
run sees every source in these proportions. Log supervised target counts and
source exposure separately from nominal token positions.

## 3. Trajectory quality

### Duplicates

No exact normalized-conversation duplicates or identical first-user prompts were
found within this sample. Conversation hashes ignore metadata but preserve role
and exact content. This does not rule out corpus duplicates, whitespace variants,
near duplicates, shared prefixes, repeated task attempts, or evaluation leakage.
Repeated sampling from a training pool is a separate source of repetition.

Before a major run, hash all normalized conversations and group train/validation
splits by stable task identity. Keep distinct attempts separate from exact copied
trajectories; use prompt similarity where task IDs are missing.

### Broken or incomplete interaction

- **180 trajectories (17.6% of all sampled rows; 19.4% of nonempty ones)** contain
  explicit harness parser-error feedback: 346 such feedback turns in total. These
  are stronger evidence than a heuristic JSON scan. Recovery later in a trajectory
  does not make the earlier rejected action a clean action target.
- **79 trajectories** have at least one command-list decoding candidate that fails
  the audit's simple parser. These flags need harness-compatible parsing/review;
  they are not an exhaustive count of invalid JSON responses.
- **469 trajectories** have a nonempty command list with no immediately following
  user/tool observation. Of these, **197** also contain a completion claim and
  **272** do not. A final completion action can legitimately terminate a trace;
  missing terminal feedback alone is not sufficient to mark an episode failed.
- **337 trajectories** end in an assistant turn without any recognized completion
  claim. This is a truncation/format review candidate, not an outcome label.
- **396 trajectories** contain common terminal-error text. A failed intermediate
  command can be part of successful recovery, so this is not a whole-episode filter.

There is also a training-induced continuity problem: **2,361 / 3,277 retained
windows (72.0%)** are later windows without the original task prefix, and **887
(27.1%)** start inside an assistant body. The initially audited adapter resets carry for
every window. It can therefore supervise continuations of reasoning or JSON
without the task or action prefix that generated them. Token coverage is retained,
but full trajectory conditioning is not.

Recommended handling: mask rejected assistant responses based on the next harness
parser-error turn, preserve useful recovery context, and analyze unfinished tails
separately. Prefer assistant-turn-aligned, prompt-preserving windows or correctly
managed cross-window state before treating this as long-horizon action BC.

### Outcomes

| Metadata category | Rows | Share |
| --- | ---: | ---: |
| Explicit success (`success` or `1.0`) | 42 | 4.1% |
| Timeout (`AgentTimeoutError` or `timeout`) | 80 | 7.8% |
| Infrastructure error (`DaytonaError` / `DaytonaNotFoundError`) | 13 | 1.3% |
| Unknown / null | 889 | 86.8% |

No reward field was populated in this sample. The source card's reward/schema
description cannot substitute for outcome labels in the rows actually served.
Assistant `task_complete=true` is a self-report and is not verified success.

Keep success, unknown, timeout, and infrastructure-error cohorts separate. Exclude
infrastructure failures from a clean success-oriented BC baseline; treat timeout
prefixes as a separate experiment. Do not silently interpret null as success or
failure. A success-only filter here would retain only 42 rows and 212,025
supervised tokens (about 5.0% of sampled supervision), with strong source-selection
bias. Most usable trajectories require better verifier metadata to assess quality.

## Recommended next steps

1. Keep speaker masking; add semantic feedback/action annotations and per-source
   token accounting. Do not apply text-keyword role rewrites.
2. Build deterministic preprocessing that records exact supervised and action
   token budgets, deduplicates the full corpus, and groups splits by task.
3. Fix window conditioning and mask rejected action turns; measure these separately
   from successful recovery episodes.
4. Compare explicit-success, unknown-outcome, and timeout cohorts rather than
   training the merged corpus as uniformly high-quality demonstrations.

This audit adds instrumentation and recommendations; it does not silently change
the training dataset, masks, or inclusion policy.
