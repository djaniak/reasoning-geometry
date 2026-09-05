# deepseek math500 prompt decomposition

Data: 500 complete prompts with N=8; partial_data=false.

| Layer | Method | Pooled AUC | Centered AUC | Within macro | Within pair | ICC | Spearman difficulty |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 21 | entropy | 0.609 | 0.561 | 0.593 | 0.587 | 0.668 | 0.255 |
| 21 | logprob | 0.611 | 0.559 | 0.596 | 0.592 | 0.660 | 0.261 |
| 21 | length | 0.701 | 0.640 | 0.731 | 0.732 | 0.807 | 0.410 |
| 21 | activation_norm | 0.343 | 0.388 | 0.385 | 0.386 | 0.812 | -0.341 |
| 21 | centroid | 0.325 | 0.513 | 0.530 | 0.532 | 0.869 | -0.389 |
| 21 | raw | 0.344 | 0.585 | 0.617 | 0.617 | 0.895 | -0.351 |
| 21 | rmd | 0.757 | 0.676 | 0.741 | 0.743 | 0.898 | 0.503 |
| 21 | rmd_high_entropy_q20 | 0.730 | 0.654 | 0.729 | 0.731 | 0.886 | 0.463 |
| 21 | rmd_tail_q20 | 0.762 | 0.742 | 0.761 | 0.764 | 0.833 | 0.518 |
| 21 | rmd_random_q20 | 0.757 | 0.677 | 0.740 | 0.741 | 0.895 | 0.505 |
| 21 | entropy_he | 0.368 | 0.431 | 0.389 | 0.393 | 0.618 | -0.309 |
| 21 | logprob_he | 0.629 | 0.563 | 0.608 | 0.602 | 0.604 | 0.305 |
| 21 | prompt_local_rmd | 0.339 | 0.562 | 0.545 | 0.565 | 0.886 | -0.358 |
| 21 | contrast_tail_q20 | 0.469 | 0.436 | 0.388 | 0.382 | 0.588 | -0.064 |

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
| 21 | 0.055 [0.032, 0.082] p=0.000 | 0.036 [0.015, 0.057] p=0.000 | 0.010 [-0.016, 0.036] p=0.500 |

## Truncation / parseability diagnostic

Unparsed (no final answer): 351/4000 (0.088); length-capped at 8192: 374 (0.093); unparsed share of the incorrect class: 0.293.

Unparsed traces are auto-labeled incorrect upstream and are usually truncated, not wrong-answer. The within-prompt metrics below restrict to traces that emitted a parseable answer; a large drop in mixed-prompt count or in the RMD-minus-entropy gap means the headline within-prompt signal was a truncation detector.

| Layer | Method | Parseable within macro | Parseable centered AUC | Mixed prompts |
|---:|:---|---:|---:|---:|
| 21 | entropy | 0.468 | 0.459 | 49 |
| 21 | logprob | 0.478 | 0.462 | 49 |
| 21 | length | 0.437 | 0.441 | 49 |
| 21 | activation_norm | 0.474 | 0.423 | 49 |
| 21 | centroid | 0.442 | 0.429 | 49 |
| 21 | raw | 0.563 | 0.542 | 49 |
| 21 | rmd | 0.473 | 0.447 | 49 |
| 21 | rmd_high_entropy_q20 | 0.507 | 0.451 | 49 |
| 21 | rmd_tail_q20 | 0.461 | 0.467 | 49 |
| 21 | rmd_random_q20 | 0.456 | 0.450 | 49 |
| 21 | entropy_he | 0.528 | 0.539 | 49 |
| 21 | logprob_he | 0.484 | 0.460 | 49 |
| 21 | prompt_local_rmd | 0.505 | 0.509 | 49 |
| 21 | contrast_tail_q20 | 0.477 | 0.536 | 49 |
| 21 | probe_outputs | 0.474 | 0.484 | 49 |
| 21 | probe_outputs_plus_rmd_high_entropy_q20 | 0.451 | 0.474 | 49 |
| 21 | probe_b0 | 0.474 | 0.484 | 49 |
| 21 | probe_b1 | 0.456 | 0.474 | 49 |
| 21 | probe_g_he | 0.432 | 0.463 | 49 |
| 21 | probe_g_random | 0.423 | 0.465 | 49 |
| 21 | probe_hidden_tail_q20 | 0.571 | 0.541 | 49 |

## Prespecified parseable score contrasts

Point estimates, raw 95% prompt-bootstrap intervals, and raw two-sided p-values are reported without post-hoc layer selection or multiplicity-adjusted claims.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | rmd_high_entropy_q20_minus_rmd | 0.004 [-0.019, 0.026] p=0.730 | 0.034 [-0.004, 0.077] p=0.090 |
| 21 | rmd_tail_q20_minus_rmd | 0.020 [-0.016, 0.059] p=0.310 | -0.012 [-0.069, 0.045] p=0.550 |
| 21 | contrast_tail_q20_minus_rmd_tail_q20 | 0.069 [-0.080, 0.219] p=0.420 | 0.016 [-0.149, 0.161] p=0.940 |
| 21 | rmd_high_entropy_q20_minus_rmd_random_q20 | 0.001 [-0.024, 0.024] p=0.980 | 0.051 [0.010, 0.092] p=0.010 |
| 21 | rmd_high_entropy_q20_minus_logprob | -0.010 [-0.049, 0.041] p=0.680 | 0.029 [-0.026, 0.089] p=0.320 |
| 21 | rmd_tail_q20_minus_logprob | 0.005 [-0.049, 0.067] p=0.790 | -0.017 [-0.101, 0.075] p=0.710 |
| 21 | contrast_tail_q20_minus_logprob | 0.075 [-0.084, 0.236] p=0.410 | -0.001 [-0.180, 0.169] p=0.940 |
| 21 | probe_outputs_plus_rmd_high_entropy_q20_minus_probe_outputs | -0.010 [-0.028, 0.005] p=0.210 | -0.024 [-0.057, 0.008] p=0.180 |
| 21 | contrast_tail_q20_minus_rmd | 0.089 [-0.083, 0.262] p=0.370 | 0.004 [-0.179, 0.164] p=0.960 |
| 21 | rmd_random_q20_minus_rmd | 0.003 [-0.007, 0.015] p=0.580 | -0.017 [-0.038, 0.006] p=0.140 |
| 21 | rmd_random_q20_minus_logprob | -0.011 [-0.052, 0.037] p=0.740 | -0.022 [-0.084, 0.045] p=0.600 |

## Supervised hidden-state probe (exploratory)

LDA fit on PCA-projected region means, cross-fitted by prompt fold on pooled labels over parseable traces. This bounds how much of the geometry signal supervision on the same activations recovers. **Post-hoc, added 2026-07-29 -- not part of the pre-registered contrast set.**

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | probe_hidden_tail_q20_minus_rmd_tail_q20 | 0.074 [0.008, 0.146] p=0.030 | 0.110 [-0.003, 0.210] p=0.070 |
| 21 | probe_hidden_tail_q20_minus_length | 0.100 [0.027, 0.182] p=0.000 | 0.135 [0.028, 0.249] p=0.010 |

Rank correlation of each score against trace length (parseable only). A scorer that merely rediscovers "long traces are wrong" shows |rho| near 1.

| Layer | Score | Spearman vs length | Pearson vs length | n |
|---:|:---|---:|---:|---:|
| 21 | probe_hidden_tail_q20 | 0.223 | 0.187 | 3649 |
| 21 | rmd | 0.820 | 0.840 | 3649 |
| 21 | rmd_tail_q20 | 0.805 | 0.827 | 3649 |
| 21 | rmd_high_entropy_q20 | 0.808 | 0.823 | 3649 |
| 21 | entropy | 0.350 | 0.400 | 3649 |
| 21 | logprob | 0.369 | 0.420 | 3649 |

## E2 same-token output autopsy

Fixed cross-fitted probes: B0=global outputs, B1=global plus same high-entropy-token outputs, G_he=B1 plus high-entropy RMD, and G_random=B1 plus matched random-20% RMD. Only the two pre-specified geometry contrasts are shown here.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | probe_g_he_minus_probe_b1 | -0.011 [-0.026, 0.009] p=0.240 | -0.024 [-0.066, 0.009] p=0.200 |
| 21 | probe_g_he_minus_probe_g_random | -0.002 [-0.017, 0.014] p=0.860 | 0.009 [-0.039, 0.043] p=0.690 |

## Prompt-contrastive direction diagnostics

Directions are fit out-of-fold from parseable mixed training prompts. Each prompt contributes one normalized difference vector; alignment nulls shuffle labels within prompts while preserving class counts.

| Layer | Region | Prompt vectors | Observed alignment | Pairwise cosine | Null mean | Null 95% interval | p |
|---:|:---|---:|---:|---:|---:|:---|---:|
| 21 | tail_q20 | 40 | 0.180 | 0.008 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 41 | 0.143 | -0.004 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 41 | 0.155 | -0.000 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 41 | 0.175 | 0.007 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 33 | 0.255 | 0.036 | NA | [NA, NA] | NA |

## Parseable paired contrasts

Contrastive score minus baseline, using prompt-cluster bootstrap intervals.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | rmd_high_entropy_q20_minus_rmd | 0.004 [-0.019, 0.026] p=0.730 | 0.034 [-0.004, 0.077] p=0.090 |
| 21 | rmd_tail_q20_minus_rmd | 0.020 [-0.016, 0.059] p=0.310 | -0.012 [-0.069, 0.045] p=0.550 |
| 21 | contrast_tail_q20_minus_rmd_tail_q20 | 0.069 [-0.080, 0.219] p=0.420 | 0.016 [-0.149, 0.161] p=0.940 |
| 21 | rmd_high_entropy_q20_minus_logprob | -0.010 [-0.049, 0.041] p=0.680 | 0.029 [-0.026, 0.089] p=0.320 |
| 21 | rmd_tail_q20_minus_logprob | 0.005 [-0.049, 0.067] p=0.790 | -0.017 [-0.101, 0.075] p=0.710 |
| 21 | contrast_tail_q20_minus_logprob | 0.075 [-0.084, 0.236] p=0.410 | -0.001 [-0.180, 0.169] p=0.940 |
| 21 | contrast_tail_q20_minus_rmd | 0.089 [-0.083, 0.262] p=0.370 | 0.004 [-0.179, 0.164] p=0.960 |
| 21 | rmd_random_q20_minus_rmd | 0.003 [-0.007, 0.015] p=0.580 | -0.017 [-0.038, 0.006] p=0.140 |
| 21 | rmd_random_q20_minus_logprob | -0.011 [-0.052, 0.037] p=0.740 | -0.022 [-0.084, 0.045] p=0.600 |

No layer was selected after observing these results.

Confidence intervals use a prompt-cluster bootstrap over fixed out-of-fold predictions; reference fitting is not repeated.
