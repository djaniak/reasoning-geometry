# qwen math500 prompt decomposition

Data: 500 complete prompts with N=8; partial_data=false.

| Layer | Method | Pooled AUC | Centered AUC | Within macro | Within pair | ICC | Spearman difficulty |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 21 | entropy | 0.571 | 0.559 | 0.599 | 0.595 | 0.738 | 0.151 |
| 21 | logprob | 0.575 | 0.559 | 0.595 | 0.589 | 0.669 | 0.153 |
| 21 | length | 0.737 | 0.563 | 0.581 | 0.582 | 0.867 | 0.478 |
| 21 | activation_norm | 0.279 | 0.441 | 0.431 | 0.442 | 0.907 | -0.449 |
| 21 | centroid | 0.320 | 0.473 | 0.437 | 0.456 | 0.871 | -0.372 |
| 21 | raw | 0.323 | 0.468 | 0.422 | 0.442 | 0.901 | -0.361 |
| 21 | rmd | 0.786 | 0.584 | 0.610 | 0.597 | 0.960 | 0.545 |
| 21 | rmd_high_entropy_q20 | 0.785 | 0.626 | 0.657 | 0.645 | 0.956 | 0.539 |
| 21 | rmd_tail_q20 | 0.839 | 0.639 | 0.673 | 0.664 | 0.878 | 0.650 |
| 21 | rmd_random_q20 | 0.784 | 0.578 | 0.580 | 0.573 | 0.943 | 0.546 |
| 21 | entropy_he | 0.426 | 0.442 | 0.403 | 0.407 | 0.744 | -0.154 |
| 21 | logprob_he | 0.576 | 0.559 | 0.597 | 0.592 | 0.666 | 0.156 |
| 21 | prompt_local_rmd | 0.319 | 0.527 | 0.515 | 0.535 | 0.940 | -0.369 |
| 21 | contrast_tail_q20 | 0.715 | 0.555 | 0.577 | 0.568 | 0.728 | 0.471 |

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
| 21 | 0.049 [0.013, 0.079] p=0.000 | 0.021 [-0.020, 0.063] p=0.360 | 0.029 [-0.029, 0.084] p=0.380 |

## Truncation / parseability diagnostic

Unparsed (no final answer): 328/4000 (0.082); length-capped at 1024: 338 (0.085); unparsed share of the incorrect class: 0.185.

Unparsed traces are auto-labeled incorrect upstream and are usually truncated, not wrong-answer. The within-prompt metrics below restrict to traces that emitted a parseable answer; a large drop in mixed-prompt count or in the RMD-minus-entropy gap means the headline within-prompt signal was a truncation detector.

| Layer | Method | Parseable within macro | Parseable centered AUC | Mixed prompts |
|---:|:---|---:|---:|---:|
| 21 | entropy | 0.660 | 0.611 | 117 |
| 21 | logprob | 0.649 | 0.609 | 117 |
| 21 | length | 0.478 | 0.474 | 117 |
| 21 | activation_norm | 0.483 | 0.485 | 117 |
| 21 | centroid | 0.412 | 0.456 | 117 |
| 21 | raw | 0.418 | 0.475 | 117 |
| 21 | rmd | 0.568 | 0.540 | 117 |
| 21 | rmd_high_entropy_q20 | 0.638 | 0.602 | 117 |
| 21 | rmd_tail_q20 | 0.618 | 0.578 | 117 |
| 21 | rmd_random_q20 | 0.549 | 0.541 | 117 |
| 21 | entropy_he | 0.345 | 0.388 | 117 |
| 21 | logprob_he | 0.652 | 0.608 | 117 |
| 21 | prompt_local_rmd | 0.491 | 0.504 | 117 |
| 21 | contrast_tail_q20 | 0.597 | 0.582 | 117 |
| 21 | probe_outputs | 0.633 | 0.602 | 117 |
| 21 | probe_outputs_plus_rmd_high_entropy_q20 | 0.696 | 0.620 | 117 |
| 21 | probe_b0 | 0.633 | 0.602 | 117 |
| 21 | probe_b1 | 0.632 | 0.601 | 117 |
| 21 | probe_g_he | 0.694 | 0.618 | 117 |
| 21 | probe_g_random | 0.648 | 0.604 | 117 |
| 21 | probe_hidden_tail_q20 | 0.528 | 0.539 | 117 |

## Prespecified parseable score contrasts

Point estimates, raw 95% prompt-bootstrap intervals, and raw two-sided p-values are reported without post-hoc layer selection or multiplicity-adjusted claims.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | rmd_high_entropy_q20_minus_rmd | 0.062 [0.026, 0.097] p=0.000 | 0.069 [0.025, 0.123] p=0.000 |
| 21 | rmd_tail_q20_minus_rmd | 0.038 [0.005, 0.065] p=0.040 | 0.049 [0.001, 0.088] p=0.050 |
| 21 | contrast_tail_q20_minus_rmd_tail_q20 | 0.004 [-0.038, 0.045] p=0.850 | -0.020 [-0.087, 0.044] p=0.410 |
| 21 | rmd_high_entropy_q20_minus_rmd_random_q20 | 0.062 [0.023, 0.096] p=0.000 | 0.089 [0.032, 0.141] p=0.000 |
| 21 | rmd_high_entropy_q20_minus_logprob | -0.006 [-0.056, 0.045] p=0.890 | -0.012 [-0.074, 0.070] p=0.800 |
| 21 | rmd_tail_q20_minus_logprob | -0.031 [-0.093, 0.033] p=0.380 | -0.032 [-0.105, 0.055] p=0.450 |
| 21 | contrast_tail_q20_minus_logprob | -0.027 [-0.077, 0.025] p=0.290 | -0.052 [-0.128, 0.020] p=0.200 |
| 21 | probe_outputs_plus_rmd_high_entropy_q20_minus_probe_outputs | 0.018 [-0.008, 0.041] p=0.130 | 0.063 [0.017, 0.113] p=0.000 |
| 21 | contrast_tail_q20_minus_rmd | 0.042 [-0.001, 0.083] p=0.070 | 0.029 [-0.047, 0.085] p=0.340 |
| 21 | rmd_random_q20_minus_rmd | 0.000 [-0.018, 0.020] p=0.990 | -0.019 [-0.046, 0.006] p=0.220 |
| 21 | rmd_random_q20_minus_logprob | -0.068 [-0.126, -0.008] p=0.010 | -0.100 [-0.181, -0.010] p=0.030 |

## Supervised hidden-state probe (exploratory)

LDA fit on PCA-projected region means, cross-fitted by prompt fold on pooled labels over parseable traces. This bounds how much of the geometry signal supervision on the same activations recovers. **Post-hoc, added 2026-07-29 -- not part of the pre-registered contrast set.**

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | probe_hidden_tail_q20_minus_rmd_tail_q20 | -0.039 [-0.076, -0.001] p=0.050 | -0.089 [-0.142, -0.029] p=0.000 |
| 21 | probe_hidden_tail_q20_minus_length | 0.064 [0.009, 0.123] p=0.020 | 0.051 [-0.025, 0.133] p=0.220 |

Rank correlation of each score against trace length (parseable only). A scorer that merely rediscovers "long traces are wrong" shows |rho| near 1.

| Layer | Score | Spearman vs length | Pearson vs length | n |
|---:|:---|---:|---:|---:|
| 21 | probe_hidden_tail_q20 | 0.425 | 0.419 | 3672 |
| 21 | rmd | 0.660 | 0.661 | 3672 |
| 21 | rmd_tail_q20 | 0.677 | 0.670 | 3672 |
| 21 | rmd_high_entropy_q20 | 0.619 | 0.597 | 3672 |
| 21 | entropy | -0.163 | -0.145 | 3672 |
| 21 | logprob | -0.134 | -0.131 | 3672 |

## E2 same-token output autopsy

Fixed cross-fitted probes: B0=global outputs, B1=global plus same high-entropy-token outputs, G_he=B1 plus high-entropy RMD, and G_random=B1 plus matched random-20% RMD. Only the two pre-specified geometry contrasts are shown here.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | probe_g_he_minus_probe_b1 | 0.017 [-0.005, 0.041] p=0.190 | 0.062 [0.022, 0.103] p=0.000 |
| 21 | probe_g_he_minus_probe_g_random | 0.014 [-0.004, 0.031] p=0.150 | 0.046 [0.008, 0.087] p=0.030 |

## Prompt-contrastive direction diagnostics

Directions are fit out-of-fold from parseable mixed training prompts. Each prompt contributes one normalized difference vector; alignment nulls shuffle labels within prompts while preserving class counts.

| Layer | Region | Prompt vectors | Observed alignment | Pairwise cosine | Null mean | Null 95% interval | p |
|---:|:---|---:|---:|---:|---:|:---|---:|
| 21 | tail_q20 | 87 | 0.154 | 0.012 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 94 | 0.155 | 0.014 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 98 | 0.161 | 0.016 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 96 | 0.134 | 0.008 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 93 | 0.148 | 0.011 | NA | [NA, NA] | NA |

## Parseable paired contrasts

Contrastive score minus baseline, using prompt-cluster bootstrap intervals.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | rmd_high_entropy_q20_minus_rmd | 0.062 [0.026, 0.097] p=0.000 | 0.069 [0.025, 0.123] p=0.000 |
| 21 | rmd_tail_q20_minus_rmd | 0.038 [0.005, 0.065] p=0.040 | 0.049 [0.001, 0.088] p=0.050 |
| 21 | contrast_tail_q20_minus_rmd_tail_q20 | 0.004 [-0.038, 0.045] p=0.850 | -0.020 [-0.087, 0.044] p=0.410 |
| 21 | rmd_high_entropy_q20_minus_logprob | -0.006 [-0.056, 0.045] p=0.890 | -0.012 [-0.074, 0.070] p=0.800 |
| 21 | rmd_tail_q20_minus_logprob | -0.031 [-0.093, 0.033] p=0.380 | -0.032 [-0.105, 0.055] p=0.450 |
| 21 | contrast_tail_q20_minus_logprob | -0.027 [-0.077, 0.025] p=0.290 | -0.052 [-0.128, 0.020] p=0.200 |
| 21 | contrast_tail_q20_minus_rmd | 0.042 [-0.001, 0.083] p=0.070 | 0.029 [-0.047, 0.085] p=0.340 |
| 21 | rmd_random_q20_minus_rmd | 0.000 [-0.018, 0.020] p=0.990 | -0.019 [-0.046, 0.006] p=0.220 |
| 21 | rmd_random_q20_minus_logprob | -0.068 [-0.126, -0.008] p=0.010 | -0.100 [-0.181, -0.010] p=0.030 |

No layer was selected after observing these results.

Confidence intervals use a prompt-cluster bootstrap over fixed out-of-fold predictions; reference fitting is not repeated.
