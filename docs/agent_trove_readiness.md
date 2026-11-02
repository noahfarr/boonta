# AgentTrove data-readiness baseline — 2026-11-02

The adapter now uses **complete recorded trajectories only**. It never splits a
message, removes preceding history, or joins unrelated histories. CLM and BLM
receive the same records, serialized inputs, targets, padding and resets; only
their loss masks differ. The default policy is `complete_trajectory`.

Eligibility requires a nonempty initial user task (after optional system turns),
a final assistant turn, some assistant supervision, and a complete serialized
record no longer than the context. Unknown outcome metadata remains unknown and
does not exclude a record. “Complete” means the entire recorded interaction is
preserved; it does not mean verified task success or an independently verified
untruncated source episode.

## Context quality

The operational context contract is: the first non-system user turn is the
original task; every supervised assistant token sees that turn and **all**
preceding serialized messages, including the latest environment interaction,
prior action, and the beginning of its current assistant message. This tests
structural context preservation, not whether an upstream task description was
semantically sufficient or an emitted command was correct.

For recognized final `commands` payloads:

\[
q = \frac{\text{supervised command tokens meeting this context contract}}
         {\text{all supervised command tokens in the selected windows}}.
\]

On the cached 1,024-row audit sample, **legacy 4K chunking has q = 28.95%**.
The complete-only policy has **q = 100% for retained command tokens** at every
tested context length with any retained actions. Every retained BLM assistant
target also keeps its full prefix. At 1K the sample has no eligible trajectory,
so q is undefined, not 100%.

These denominators differ: improving q by excluding data does not recover the
discarded action-token budget. Both quantities must be reported together.

## Retained budgets

This reuses the same cached sample and Qwen3 tokenizer as the
[initial audit](agent_trove_audit.md). It contains 928 nonempty records, 4,263,426
assistant-body/end-marker targets, and 2,007,159 recognized command-payload
tokens before context filtering. Exact duplicate conversations were removed
within the sample; none were found. No success/timeout/parser-error cohort was
silently removed. Role validation and whole-record selection are shared by both
objectives.

| Context | Retained records | BLM targets | Command-payload tokens | Original action budget retained | q |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0 | 0 | 0 | 0% | undefined |
| 2,048 | 55 | 30,714 | 7,337 | 0.37% | 100% |
| 4,096 | 171 | 185,523 | 54,377 | 2.71% | 100% |
| 8,192 | 411 | 839,227 | 333,367 | 16.61% | 100% |
| 16,384 | 699 | 2,259,317 | 998,748 | 49.76% | 100% |
| 32,768 | 884 | 3,609,850 | 1,646,784 | 82.05% | 100% |

At 4K, the same 171 windows contain **259,298 user/tool body tokens**, 185,523
BLM targets, and **447,542 CLM targets**. The difference includes role framing:
CLM supervises all nonpadding next-token targets, while BLM supervises assistant
bodies/end markers. Recognized command payloads are a subset of BLM targets;
reasoning, plans and completion text remain part of the BLM objective. Allocated
positions total 700,416, giving 26.49% BLM supervision density after padding.

The complete-record length distribution among nonempty sampled records is:

| Quantile | Serialized tokens |
| --- | ---: |
| 10th | 2,881 |
| 25th | 4,909 |
| Median | 9,288 |
| 75th | 16,255 |
| 90th | 25,767 |
| 95th | 32,648 |
| 99th | 56,035 |
| Maximum | 119,131 |

Length filtering changes source composition. At 4K, only **1 of 128 sampled
`swesmith` records** survives; **39 of 96 `taskmaster` records** survive. No
sampled `SWEGym` record survives. A complete-only 4K baseline therefore tests a
short-trajectory subpopulation; it is not representative of the merged corpus.

## Checks and reproducibility

The offline checker asserts input/target equality between objectives, exact
preservation of the original token sequence, correct shifted targets, one reset
per trajectory, and zero loss on padding. It performs these checks on each
retained record at each tested context. No prefix omission occurs, so no omission
marker or synthetic adjacency is needed. Prefix-plus-recent-history windows are
deliberately outside this first baseline.

```bash
.venv/bin/python -m hangar.scripts.check_agent_trove_readiness \
  --sample outputs/agent_trove_audit/sample \
  --tokenizer /path/to/Qwen3-0.6B/tokenizer.json
```

Artifacts are `readiness.json` and `readiness_records.json` in that sample folder,
including budgets by source and outcome. Results are a sample-level structural
readiness test, not a full-corpus deduplication or token-budget certificate.

Keep AgentTrove provisionally. A small 4K complete-only baseline is now a useful
conditioning-controlled experiment, but it retains only 2.7% of sampled action
tokens. Run full preprocessing/token accounting before sizing a 150M model or
committing to its training budget. Then compare 8K/16K full-trajectory baselines
or explicitly specified message-boundary context policies on paired CLM/BLM data.
No model training was launched as part of this readiness assessment.
