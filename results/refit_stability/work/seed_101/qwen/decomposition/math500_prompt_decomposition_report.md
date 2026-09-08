# qwen math500 prompt decomposition

Data: 500 complete prompts with N=8; partial_data=false.

| Layer | Method | Pooled AUC | Centered AUC | Within macro | Within pair | ICC | Spearman difficulty |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 21 | entropy | 0.571 | 0.559 | 0.599 | 0.595 | 0.738 | 0.151 |
| 21 | logprob | 0.575 | 0.559 | 0.595 | 0.589 | 0.669 | 0.153 |
| 21 | length | 0.737 | 0.563 | 0.581 | 0.582 | 0.867 | 0.478 |
| 21 | activation_norm | 0.279 | 0.441 | 0.431 | 0.442 | 0.907 | -0.449 |
| 21 | centroid | 0.322 | 0.472 | 0.437 | 0.458 | 0.871 | -0.368 |
| 21 | raw | 0.322 | 0.468 | 0.424 | 0.443 | 0.901 | -0.360 |
| 21 | rmd | 0.784 | 0.593 | 0.617 | 0.603 | 0.959 | 0.541 |
| 21 | rmd_high_entropy_q20 | 0.780 | 0.627 | 0.668 | 0.659 | 0.955 | 0.530 |
| 21 | rmd_tail_q20 | 0.836 | 0.641 | 0.672 | 0.663 | 0.879 | 0.647 |
| 21 | rmd_random_q20 | 0.781 | 0.586 | 0.593 | 0.586 | 0.942 | 0.542 |
| 21 | entropy_he | 0.426 | 0.442 | 0.403 | 0.407 | 0.744 | -0.154 |
| 21 | logprob_he | 0.576 | 0.559 | 0.597 | 0.592 | 0.666 | 0.156 |
| 21 | prompt_local_rmd | 0.319 | 0.528 | 0.513 | 0.530 | 0.940 | -0.370 |
| 21 | contrast_tail_q20 | 0.708 | 0.557 | 0.562 | 0.562 | 0.742 | 0.446 |

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
| 21 | 0.047 [0.015, 0.075] p=0.000 | 0.030 [-0.019, 0.068] p=0.220 | 0.037 [-0.034, 0.087] p=0.340 |

## Truncation / parseability diagnostic

Unparsed (no final answer): 328/4000 (0.082); length-capped at 1024: 338 (0.085); unparsed share of the incorrect class: 0.185.

Unparsed traces are auto-labeled incorrect upstream and are usually truncated, not wrong-answer. The within-prompt metrics below restrict to traces that emitted a parseable answer; a large drop in mixed-prompt count or in the RMD-minus-entropy gap means the headline within-prompt signal was a truncation detector.

| Layer | Method | Parseable within macro | Parseable centered AUC | Mixed prompts |
|---:|:---|---:|---:|---:|
| 21 | entropy | 0.660 | 0.611 | 117 |
| 21 | logprob | 0.649 | 0.609 | 117 |
| 21 | length | 0.478 | 0.474 | 117 |
| 21 | activation_norm | 0.483 | 0.485 | 117 |
| 21 | centroid | 0.414 | 0.454 | 117 |
| 21 | raw | 0.427 | 0.476 | 117 |
| 21 | rmd | 0.571 | 0.549 | 117 |
| 21 | rmd_high_entropy_q20 | 0.647 | 0.604 | 117 |
| 21 | rmd_tail_q20 | 0.612 | 0.577 | 117 |
| 21 | rmd_random_q20 | 0.557 | 0.549 | 117 |
| 21 | entropy_he | 0.345 | 0.388 | 117 |
| 21 | logprob_he | 0.652 | 0.608 | 117 |
| 21 | prompt_local_rmd | 0.488 | 0.504 | 117 |
| 21 | contrast_tail_q20 | 0.584 | 0.579 | 117 |
| 21 | probe_outputs | 0.639 | 0.601 | 117 |
| 21 | probe_outputs_plus_rmd_high_entropy_q20 | 0.703 | 0.622 | 117 |
| 21 | probe_b0 | 0.639 | 0.601 | 117 |
| 21 | probe_b1 | 0.637 | 0.597 | 117 |
| 21 | probe_g_he | 0.687 | 0.619 | 117 |
| 21 | probe_g_random | 0.638 | 0.593 | 117 |
| 21 | probe_hidden_tail_q20 | 0.556 | 0.546 | 117 |

## Prespecified parseable score contrasts

Point estimates, raw 95% prompt-bootstrap intervals, and raw two-sided p-values are reported without post-hoc layer selection or multiplicity-adjusted claims.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | rmd_high_entropy_q20_minus_rmd | 0.055 [0.020, 0.089] p=0.000 | 0.075 [0.027, 0.123] p=0.000 |
| 21 | rmd_tail_q20_minus_rmd | 0.027 [-0.009, 0.062] p=0.140 | 0.041 [-0.015, 0.086] p=0.180 |
| 21 | contrast_tail_q20_minus_rmd_tail_q20 | 0.002 [-0.042, 0.043] p=0.900 | -0.028 [-0.077, 0.027] p=0.380 |
| 21 | rmd_high_entropy_q20_minus_rmd_random_q20 | 0.055 [0.019, 0.092] p=0.000 | 0.089 [0.030, 0.151] p=0.000 |
| 21 | rmd_high_entropy_q20_minus_logprob | -0.005 [-0.053, 0.049] p=0.860 | -0.003 [-0.063, 0.067] p=1.000 |
| 21 | rmd_tail_q20_minus_logprob | -0.032 [-0.096, 0.030] p=0.320 | -0.037 [-0.115, 0.036] p=0.360 |
| 21 | contrast_tail_q20_minus_logprob | -0.030 [-0.077, 0.016] p=0.250 | -0.065 [-0.145, -0.004] p=0.030 |
| 21 | probe_outputs_plus_rmd_high_entropy_q20_minus_probe_outputs | 0.021 [-0.003, 0.048] p=0.060 | 0.063 [0.024, 0.108] p=0.000 |
| 21 | contrast_tail_q20_minus_rmd | 0.030 [-0.018, 0.068] p=0.180 | 0.013 [-0.045, 0.069] p=0.750 |
| 21 | rmd_random_q20_minus_rmd | -0.000 [-0.016, 0.018] p=0.990 | -0.014 [-0.043, 0.012] p=0.280 |
| 21 | rmd_random_q20_minus_logprob | -0.059 [-0.114, -0.000] p=0.050 | -0.092 [-0.160, -0.011] p=0.010 |

## Supervised hidden-state probe (exploratory)

LDA fit on PCA-projected region means, cross-fitted by prompt fold on pooled labels over parseable traces. This bounds how much of the geometry signal supervision on the same activations recovers. **Post-hoc, added 2026-07-29 -- not part of the pre-registered contrast set.**

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | probe_hidden_tail_q20_minus_rmd_tail_q20 | -0.031 [-0.075, 0.010] p=0.180 | -0.056 [-0.114, -0.001] p=0.050 |
| 21 | probe_hidden_tail_q20_minus_length | 0.071 [0.006, 0.128] p=0.030 | 0.079 [0.000, 0.150] p=0.050 |

Rank correlation of each score against trace length (parseable only). A scorer that merely rediscovers "long traces are wrong" shows |rho| near 1.

| Layer | Score | Spearman vs length | Pearson vs length | n |
|---:|:---|---:|---:|---:|
| 21 | probe_hidden_tail_q20 | 0.421 | 0.413 | 3672 |
| 21 | rmd | 0.655 | 0.658 | 3672 |
| 21 | rmd_tail_q20 | 0.674 | 0.668 | 3672 |
| 21 | rmd_high_entropy_q20 | 0.606 | 0.591 | 3672 |
| 21 | entropy | -0.163 | -0.145 | 3672 |
| 21 | logprob | -0.134 | -0.131 | 3672 |

## E2 same-token output autopsy

Fixed cross-fitted probes: B0=global outputs, B1=global plus same high-entropy-token outputs, G_he=B1 plus high-entropy RMD, and G_random=B1 plus matched random-20% RMD. Only the two pre-specified geometry contrasts are shown here.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | probe_g_he_minus_probe_b1 | 0.022 [-0.001, 0.045] p=0.070 | 0.049 [0.014, 0.088] p=0.010 |
| 21 | probe_g_he_minus_probe_g_random | 0.026 [0.006, 0.048] p=0.020 | 0.049 [0.016, 0.083] p=0.000 |

## Prompt-contrastive direction diagnostics

Directions are fit out-of-fold from parseable mixed training prompts. Each prompt contributes one normalized difference vector; alignment nulls shuffle labels within prompts while preserving class counts.

| Layer | Region | Prompt vectors | Observed alignment | Pairwise cosine | Null mean | Null 95% interval | p |
|---:|:---|---:|---:|---:|---:|:---|---:|
| 21 | tail_q20 | 93 | 0.169 | 0.018 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 98 | 0.144 | 0.011 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 92 | 0.158 | 0.014 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 90 | 0.145 | 0.010 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 95 | 0.139 | 0.009 | NA | [NA, NA] | NA |

## Parseable paired contrasts

Contrastive score minus baseline, using prompt-cluster bootstrap intervals.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | rmd_high_entropy_q20_minus_rmd | 0.055 [0.020, 0.089] p=0.000 | 0.075 [0.027, 0.123] p=0.000 |
| 21 | rmd_tail_q20_minus_rmd | 0.027 [-0.009, 0.062] p=0.140 | 0.041 [-0.015, 0.086] p=0.180 |
| 21 | contrast_tail_q20_minus_rmd_tail_q20 | 0.002 [-0.042, 0.043] p=0.900 | -0.028 [-0.077, 0.027] p=0.380 |
| 21 | rmd_high_entropy_q20_minus_logprob | -0.005 [-0.053, 0.049] p=0.860 | -0.003 [-0.063, 0.067] p=1.000 |
| 21 | rmd_tail_q20_minus_logprob | -0.032 [-0.096, 0.030] p=0.320 | -0.037 [-0.115, 0.036] p=0.360 |
| 21 | contrast_tail_q20_minus_logprob | -0.030 [-0.077, 0.016] p=0.250 | -0.065 [-0.145, -0.004] p=0.030 |
| 21 | contrast_tail_q20_minus_rmd | 0.030 [-0.018, 0.068] p=0.180 | 0.013 [-0.045, 0.069] p=0.750 |
| 21 | rmd_random_q20_minus_rmd | -0.000 [-0.016, 0.018] p=0.990 | -0.014 [-0.043, 0.012] p=0.280 |
| 21 | rmd_random_q20_minus_logprob | -0.059 [-0.114, -0.000] p=0.050 | -0.092 [-0.160, -0.011] p=0.010 |

No layer was selected after observing these results.

Confidence intervals use a prompt-cluster bootstrap over fixed out-of-fold predictions; reference fitting is not repeated.
