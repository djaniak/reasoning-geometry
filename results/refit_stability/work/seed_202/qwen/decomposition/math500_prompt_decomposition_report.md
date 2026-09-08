# qwen math500 prompt decomposition

Data: 500 complete prompts with N=8; partial_data=false.

| Layer | Method | Pooled AUC | Centered AUC | Within macro | Within pair | ICC | Spearman difficulty |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 21 | entropy | 0.571 | 0.559 | 0.599 | 0.595 | 0.738 | 0.151 |
| 21 | logprob | 0.575 | 0.559 | 0.595 | 0.589 | 0.669 | 0.153 |
| 21 | length | 0.737 | 0.563 | 0.581 | 0.582 | 0.867 | 0.478 |
| 21 | activation_norm | 0.279 | 0.441 | 0.431 | 0.442 | 0.907 | -0.449 |
| 21 | centroid | 0.321 | 0.471 | 0.434 | 0.455 | 0.871 | -0.372 |
| 21 | raw | 0.322 | 0.466 | 0.427 | 0.447 | 0.901 | -0.360 |
| 21 | rmd | 0.787 | 0.592 | 0.611 | 0.597 | 0.959 | 0.547 |
| 21 | rmd_high_entropy_q20 | 0.788 | 0.627 | 0.665 | 0.653 | 0.953 | 0.545 |
| 21 | rmd_tail_q20 | 0.840 | 0.644 | 0.680 | 0.673 | 0.878 | 0.655 |
| 21 | rmd_random_q20 | 0.784 | 0.583 | 0.591 | 0.584 | 0.942 | 0.547 |
| 21 | entropy_he | 0.426 | 0.442 | 0.403 | 0.407 | 0.744 | -0.154 |
| 21 | logprob_he | 0.576 | 0.559 | 0.597 | 0.592 | 0.666 | 0.156 |
| 21 | prompt_local_rmd | 0.319 | 0.525 | 0.511 | 0.531 | 0.939 | -0.371 |
| 21 | contrast_tail_q20 | 0.722 | 0.572 | 0.588 | 0.584 | 0.749 | 0.472 |

## Supervised cross-fitted incremental readouts

These logistic readouts are supervised and cross-fitted by the existing prompt fold. Training uses only parseable mixed prompts; features are prompt-centered, standardized on training rows, and weighted equally by prompt and class. They appear only in parseable-only metrics.

| Method | Features |
|:---|:---|
| probe_outputs | logprob, entropy, length |
| probe_outputs_plus_rmd_high_entropy_q20 | logprob, entropy, length, rmd_high_entropy_q20 |
| probe_outputs_plus_contrast_high_entropy_q20 | logprob, entropy, length, contrast_high_entropy_q20 |
| probe_b0 | length, entropy, logprob |
| probe_b1 | length, entropy, logprob, entropy_he, logprob_he |
| probe_g_he | length, entropy, logprob, entropy_he, logprob_he, rmd_high_entropy_q20 |
| probe_g_random | length, entropy, logprob, entropy_he, logprob_he, rmd_random_q20 |

## Primary contrast: RMD − length

Trace length is the strong baseline for correctness (wrong/hard/truncated traces ramble), so RMD's contribution is its margin OVER length, not over entropy. Point estimate with 95% prompt-bootstrap CI and two-sided p; a contribution requires the CI to exclude zero.

| Layer | RMD−length pooled AUC | RMD−length centered AUC | RMD−length within macro |
|---:|:---|:---|:---|
| 21 | 0.050 [0.020, 0.079] p=0.010 | 0.029 [-0.019, 0.074] p=0.280 | 0.030 [-0.037, 0.093] p=0.380 |

## Truncation / parseability diagnostic

Unparsed (no final answer): 328/4000 (0.082); length-capped at 1024: 338 (0.085); unparsed share of the incorrect class: 0.185.

Unparsed traces are auto-labeled incorrect upstream and are usually truncated, not wrong-answer. The within-prompt metrics below restrict to traces that emitted a parseable answer; a large drop in mixed-prompt count or in the RMD-minus-entropy gap means the headline within-prompt signal was a truncation detector.

| Layer | Method | Parseable within macro | Parseable centered AUC | Mixed prompts |
|---:|:---|---:|---:|---:|
| 21 | entropy | 0.660 | 0.611 | 117 |
| 21 | logprob | 0.649 | 0.609 | 117 |
| 21 | length | 0.478 | 0.474 | 117 |
| 21 | activation_norm | 0.483 | 0.485 | 117 |
| 21 | centroid | 0.408 | 0.453 | 117 |
| 21 | raw | 0.422 | 0.472 | 117 |
| 21 | rmd | 0.568 | 0.549 | 117 |
| 21 | rmd_high_entropy_q20 | 0.639 | 0.603 | 117 |
| 21 | rmd_tail_q20 | 0.624 | 0.577 | 117 |
| 21 | rmd_random_q20 | 0.554 | 0.546 | 117 |
| 21 | entropy_he | 0.345 | 0.388 | 117 |
| 21 | logprob_he | 0.652 | 0.608 | 117 |
| 21 | prompt_local_rmd | 0.482 | 0.501 | 117 |
| 21 | contrast_tail_q20 | 0.610 | 0.593 | 117 |
| 21 | probe_outputs | 0.648 | 0.603 | 117 |
| 21 | probe_outputs_plus_rmd_high_entropy_q20 | 0.691 | 0.619 | 117 |
| 21 | probe_b0 | 0.648 | 0.603 | 117 |
| 21 | probe_b1 | 0.636 | 0.603 | 117 |
| 21 | probe_g_he | 0.692 | 0.619 | 117 |
| 21 | probe_g_random | 0.650 | 0.603 | 117 |
| 21 | probe_hidden_tail_q20 | 0.560 | 0.558 | 117 |

## Prespecified parseable score contrasts

Point estimates, raw 95% prompt-bootstrap intervals, and raw two-sided p-values are reported without post-hoc layer selection or multiplicity-adjusted claims.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | rmd_high_entropy_q20_minus_rmd | 0.054 [0.019, 0.090] p=0.000 | 0.072 [0.024, 0.122] p=0.000 |
| 21 | rmd_tail_q20_minus_rmd | 0.029 [0.000, 0.058] p=0.050 | 0.056 [-0.003, 0.100] p=0.070 |
| 21 | contrast_tail_q20_minus_rmd_tail_q20 | 0.016 [-0.021, 0.058] p=0.440 | -0.013 [-0.060, 0.038] p=0.740 |
| 21 | rmd_high_entropy_q20_minus_rmd_random_q20 | 0.057 [0.025, 0.097] p=0.000 | 0.086 [0.032, 0.135] p=0.000 |
| 21 | rmd_high_entropy_q20_minus_logprob | -0.006 [-0.054, 0.040] p=0.690 | -0.010 [-0.080, 0.049] p=0.720 |
| 21 | rmd_tail_q20_minus_logprob | -0.031 [-0.096, 0.026] p=0.240 | -0.026 [-0.096, 0.038] p=0.500 |
| 21 | contrast_tail_q20_minus_logprob | -0.016 [-0.064, 0.028] p=0.550 | -0.039 [-0.099, 0.017] p=0.200 |
| 21 | probe_outputs_plus_rmd_high_entropy_q20_minus_probe_outputs | 0.016 [-0.006, 0.040] p=0.180 | 0.043 [0.005, 0.072] p=0.030 |
| 21 | contrast_tail_q20_minus_rmd | 0.044 [0.005, 0.088] p=0.030 | 0.043 [-0.017, 0.097] p=0.210 |
| 21 | rmd_random_q20_minus_rmd | -0.003 [-0.022, 0.016] p=0.710 | -0.014 [-0.042, 0.011] p=0.240 |
| 21 | rmd_random_q20_minus_logprob | -0.062 [-0.130, -0.004] p=0.040 | -0.096 [-0.163, -0.024] p=0.010 |

## Supervised hidden-state probe (exploratory)

LDA fit on PCA-projected region means, cross-fitted by prompt fold on pooled labels over parseable traces. This bounds how much of the geometry signal supervision on the same activations recovers. **Post-hoc, added 2026-07-29 -- not part of the pre-registered contrast set.**

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | probe_hidden_tail_q20_minus_rmd_tail_q20 | -0.019 [-0.061, 0.024] p=0.390 | -0.064 [-0.121, 0.000] p=0.060 |
| 21 | probe_hidden_tail_q20_minus_length | 0.084 [0.025, 0.133] p=0.000 | 0.083 [0.006, 0.163] p=0.040 |

Rank correlation of each score against trace length (parseable only). A scorer that merely rediscovers "long traces are wrong" shows |rho| near 1.

| Layer | Score | Spearman vs length | Pearson vs length | n |
|---:|:---|---:|---:|---:|
| 21 | probe_hidden_tail_q20 | 0.432 | 0.421 | 3672 |
| 21 | rmd | 0.675 | 0.674 | 3672 |
| 21 | rmd_tail_q20 | 0.694 | 0.683 | 3672 |
| 21 | rmd_high_entropy_q20 | 0.632 | 0.604 | 3672 |
| 21 | entropy | -0.163 | -0.145 | 3672 |
| 21 | logprob | -0.134 | -0.131 | 3672 |

## E2 same-token output autopsy

Fixed cross-fitted probes: B0=global outputs, B1=global plus same high-entropy-token outputs, G_he=B1 plus high-entropy RMD, and G_random=B1 plus matched random-20% RMD. Only the two pre-specified geometry contrasts are shown here.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | probe_g_he_minus_probe_b1 | 0.016 [-0.003, 0.039] p=0.140 | 0.056 [0.023, 0.090] p=0.000 |
| 21 | probe_g_he_minus_probe_g_random | 0.017 [-0.001, 0.032] p=0.060 | 0.042 [0.012, 0.075] p=0.010 |

## Prompt-contrastive direction diagnostics

Directions are fit out-of-fold from parseable mixed training prompts. Each prompt contributes one normalized difference vector; alignment nulls shuffle labels within prompts while preserving class counts.

| Layer | Region | Prompt vectors | Observed alignment | Pairwise cosine | Null mean | Null 95% interval | p |
|---:|:---|---:|---:|---:|---:|:---|---:|
| 21 | tail_q20 | 85 | 0.125 | 0.004 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 93 | 0.156 | 0.014 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 101 | 0.148 | 0.012 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 95 | 0.169 | 0.018 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 94 | 0.149 | 0.012 | NA | [NA, NA] | NA |

## Parseable paired contrasts

Contrastive score minus baseline, using prompt-cluster bootstrap intervals.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | rmd_high_entropy_q20_minus_rmd | 0.054 [0.019, 0.090] p=0.000 | 0.072 [0.024, 0.122] p=0.000 |
| 21 | rmd_tail_q20_minus_rmd | 0.029 [0.000, 0.058] p=0.050 | 0.056 [-0.003, 0.100] p=0.070 |
| 21 | contrast_tail_q20_minus_rmd_tail_q20 | 0.016 [-0.021, 0.058] p=0.440 | -0.013 [-0.060, 0.038] p=0.740 |
| 21 | rmd_high_entropy_q20_minus_logprob | -0.006 [-0.054, 0.040] p=0.690 | -0.010 [-0.080, 0.049] p=0.720 |
| 21 | rmd_tail_q20_minus_logprob | -0.031 [-0.096, 0.026] p=0.240 | -0.026 [-0.096, 0.038] p=0.500 |
| 21 | contrast_tail_q20_minus_logprob | -0.016 [-0.064, 0.028] p=0.550 | -0.039 [-0.099, 0.017] p=0.200 |
| 21 | contrast_tail_q20_minus_rmd | 0.044 [0.005, 0.088] p=0.030 | 0.043 [-0.017, 0.097] p=0.210 |
| 21 | rmd_random_q20_minus_rmd | -0.003 [-0.022, 0.016] p=0.710 | -0.014 [-0.042, 0.011] p=0.240 |
| 21 | rmd_random_q20_minus_logprob | -0.062 [-0.130, -0.004] p=0.040 | -0.096 [-0.163, -0.024] p=0.010 |

No layer was selected after observing these results.

Confidence intervals use a prompt-cluster bootstrap over fixed out-of-fold predictions; reference fitting is not repeated.
