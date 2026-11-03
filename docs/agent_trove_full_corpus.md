# AgentTrove accounting

Scope: complete input. Rows: 1,696,847/1,696,847.
Revision: `b395a4307a2bc9950a90dc899438f149e115fc60`. Input: `huggingface_stream`.

| Context | Eligible | Retained tokens | Discarded tokens | BLM targets | CLM targets | Commands | Token retention | Padding |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 374,361 | 1,026,170,512 | 16,952,784,125 | 420,001,511 | 1,025,796,151 | 124,646,304 | 5.71% | 33.10% |
| 8192 | 744,939 | 3,213,337,520 | 14,765,617,117 | 1,428,647,439 | 3,212,592,581 | 532,355,077 | 17.87% | 47.36% |
| 16384 | 1,192,945 | 8,581,777,619 | 9,397,177,018 | 3,708,150,333 | 8,580,584,674 | 1,685,125,764 | 47.73% | 56.10% |
| 32768 | 1,525,799 | 16,059,601,054 | 1,919,353,583 | 6,222,812,257 | 16,058,075,255 | 2,917,851,115 | 89.32% | 67.88% |

## Retained source composition

| Source | Input rows | 4096 | 8192 | 16384 | 32768 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Glaive Code Assistant | 9647 | 1853 | 4799 | 7903 | 9433 |
| Inferred Bugs | 26478 | 132 | 5960 | 21844 | 26132 |
| MagiCoder Evol Instruct | 100 | 54 | 81 | 92 | 99 |
| Nemotron Prism | 6931 | 3647 | 5945 | 6752 | 6916 |
| Qasper | 16305 | 496 | 5496 | 13144 | 16049 |
| SWEGym | 4077 | 7 | 60 | 188 | 1841 |
| Self-Instruct Naive | 9626 | 3573 | 7329 | 9429 | 9624 |
| StackExchange Overflow | 23404 | 10555 | 15241 | 20549 | 23056 |
| WikiTable Format Conversion | 1927 | 469 | 1902 | 1927 | 1927 |
| bash textbook | 9998 | 2808 | 5559 | 8916 | 9987 |
| code_contests | 45923 | 14531 | 28369 | 40696 | 45285 |
| codeforces | 39263 | 16476 | 30935 | 38462 | 39251 |
| defects 4j | 15806 | 207 | 441 | 5503 | 15254 |
| exp_rle | 60000 | 0 | 0 | 0 | 0 |
| exp_rpt | 58845 | 0 | 0 | 0 | 0 |
| exp_tas | 375168 | 84644 | 220222 | 297939 | 372606 |
| freelancer | 281269 | 15215 | 56421 | 187660 | 265782 |
| gsm8k | 50 | 34 | 48 | 50 | 50 |
| math | 9994 | 9299 | 9855 | 9983 | 9993 |
| multifile composition | 11266 | 23 | 1113 | 9604 | 11253 |
| neulab | 55200 | 8407 | 19632 | 39576 | 52329 |
| nl2bash | 101012 | 81299 | 100296 | 100997 | 101012 |
| puzzles | 47285 | 37673 | 42738 | 45581 | 47106 |
| r2egym | 36533 | 5220 | 7317 | 10447 | 30034 |
| repo scaffold | 11596 | 1324 | 2479 | 8658 | 11574 |
| stack_exchange | 127554 | 16543 | 52237 | 98069 | 124410 |
| staqc | 20298 | 5650 | 12732 | 18847 | 20249 |
| swesmith | 78246 | 2016 | 8844 | 33292 | 72698 |
| taskmaster | 121610 | 43003 | 71185 | 100395 | 120379 |
| unknown | 91436 | 9203 | 27703 | 56442 | 81470 |

Outcomes (all rows): failure=1328, infrastructure_error=17186, success=18159, timeout=181447, unknown=1478727

JSON contains exact length histograms, outcome/source/split budgets, metadata availability, command retention, supervision density, and padding estimates.
Compute estimates use 6 × 200M × allocated positions per pass; they exclude quadratic attention and are not measured hardware throughput.
Power-of-two bucket positions are a potential batching estimate, not implemented packing or measured training speed.

## Outcome retention

| Outcome | Input rows | 4K | 8K | 16K | 32K |
| --- | ---: | ---: | ---: | ---: | ---: |
| failure | 1328 | 14 | 65 | 240 | 1327 |
| infrastructure_error | 17186 | 5986 | 10409 | 13481 | 16590 |
| success | 18159 | 148 | 1289 | 7783 | 17863 |
| timeout | 181447 | 18288 | 35424 | 72270 | 171772 |
| unknown | 1478727 | 349925 | 697752 | 1099171 | 1318247 |

## Supervision and length distribution

| Context | Commands retained | BLM / valid targets | BLM / allocated positions |
| --- | ---: | ---: | ---: |
| 4096 | 3.81% | 40.94% | 27.39% |
| 8192 | 16.29% | 44.47% | 23.41% |
| 16384 | 51.56% | 43.22% | 18.97% |
| 32768 | 89.29% | 38.75% | 12.45% |

Nonempty serialized-length quantiles: 0=74, 0.1=2,606, 0.25=4,225, 0.5=8,780, 0.75=15,993, 0.9=24,160, 0.95=28,833, 0.99=44,917, 1=890,996.

Empty rows: 128,435. The `exp_rle` and `exp_rpt` cohorts contain no supported embedded messages; they contribute no tokens. No outcome-based filtering was applied.

## Recommendation and compute estimates

Use the native **191,772,672-parameter Qwen3** preset, 12 layers, width 512, FFN 1,536, 8 query heads / 2 KV groups, head dimension 64, vocabulary 151,936. Start the planned comparison at **16K**, subject to a target-GPU memory check; use 8K for a separate memory-constrained pilot. The original conversation-hash 16K train split has **1,180,895 trajectories, 3,670,910,974 BLM targets, and 8,493,991,202 CLM targets**. These historical split counts differ from the later grouped overlay (1,182,693 train trajectories, 3,676,006,457 BLM targets, 8,500,636,866 CLM targets). Corpus-wide counts above are unaffected.

This remains a length-selected population: at 16K only 188/4,077 SWEGym and 33,292/78,246 swesmith records survive; even 32K retains just 1,841/4,077 SWEGym records.

The preset specifies 2^34 allocated positions (2^20 trajectory draws), 4,096 pool refreshes, pool size 256, one sequence per device, seed 42, and the inherited AdamW/cosine schedule (LR 3e-4, weight decay 0.1, clip 1.0, entropy coefficient zero). It requires explicit cache and tokenizer paths. Use `scoring=final` for both runs and compare matched-step checkpoints. Toggle only `dataset.kwargs.assistant_only` for the paired objectives.

Expected supervised targets under this sampling budget are 3.260B BLM and 7.542B CLM; these are sampling expectations, not additional corpus measurements.

| Context | Train-pass dense FLOPs | Additional attention FLOPs | Total estimate |
| --- | ---: | ---: | ---: |
| 4096 | 1.746e+18 | 9.166e+17 | 2.663e+18 |
| 8192 | 6.951e+18 | 7.297e+18 | 1.425e+19 |
| 16384 | 2.226e+19 | 4.674e+19 | 6.900e+19 |
| 32768 | 5.695e+19 | 2.391e+20 | 2.961e+20 |

Estimates use 6NT for dense work and 24 × 12 × 512 × context × allocated positions for attention with the existing doubled key axis and a 3× forward-to-training factor. They omit additional rematerialization, optimizer, and communication overhead and are not measured GPU throughput. One pass means the hypothetical sum over every retained record; the paired presets now sample a deterministic permutation without replacement.

The conventional dense-CLM 20-token-per-parameter planning heuristic would suggest about 3.84B tokens for this model; it does not establish a scaling law or an optimal size for action-masked behavioral cloning. See [Hoffmann et al.](https://arxiv.org/pdf/2203.15556).

## Verification and remaining limits

Full-cache verification passed for all 1,696,847 rows, the entire stored BLM mask, split assignments, and paired CLM/BLM JIT sampling on four logical CPU devices at every context in both splits. No pretraining was launched.

An unchunked categorical head casts vocabulary logits to float32 (18.55 GiB logical logits per sequence at 32K). The paired presets tile the vocabulary projection over positions to bound those intermediates; attention also includes a full invalid initial KV cache and an explicit mask. GPU memory and multi-host execution remain unverified. No packing was introduced. Length buckets could reduce padding, but need a scheduler compatible with Quadinaros' fixed compiled shapes. Held-out likelihood evaluation still needs wiring before reporting model comparisons.

Outcomes are explicit source labels, not independently verified task results. Only 218,706 rows have non-null `result`; recognized infrastructure labels are Daytona errors. Other unclassified exceptions remain unknown with their raw types preserved. Missing metadata never means failure. Recorded assistant-role provenance and task-disjoint validation are not certified.

[Compact machine-readable statistics](agent_trove_statistics.json) include source/outcome/split budgets, exact quantiles and model estimates. Full histograms and per-row metadata remain in `outputs/agent_trove_full/`. [Implementation and experimental controls](agent_trove_pretraining.md) describe reproduction, normalization, sampling and memory limits.
