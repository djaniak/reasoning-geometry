# deepseek math500 prompt decomposition

Data: 500 complete prompts with N=8; partial_data=false.

| Layer | Method | Pooled AUC | Centered AUC | Within macro | Within pair | ICC | Spearman difficulty |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 21 | entropy | 0.609 | 0.561 | 0.593 | 0.587 | 0.668 | 0.255 |
| 21 | logprob | 0.611 | 0.559 | 0.596 | 0.592 | 0.660 | 0.261 |
| 21 | length | 0.701 | 0.640 | 0.731 | 0.732 | 0.807 | 0.410 |
| 21 | activation_norm | 0.343 | 0.388 | 0.385 | 0.386 | 0.812 | -0.341 |
| 21 | centroid | 0.323 | 0.512 | 0.527 | 0.529 | 0.870 | -0.392 |
| 21 | raw | 0.344 | 0.587 | 0.626 | 0.626 | 0.895 | -0.352 |
| 21 | rmd | 0.761 | 0.675 | 0.735 | 0.739 | 0.899 | 0.513 |
| 21 | rmd_high_entropy_q20 | 0.734 | 0.652 | 0.720 | 0.728 | 0.885 | 0.472 |
| 21 | rmd_tail_q20 | 0.768 | 0.741 | 0.761 | 0.764 | 0.831 | 0.532 |
| 21 | rmd_random_q20 | 0.762 | 0.675 | 0.736 | 0.739 | 0.895 | 0.515 |
| 21 | entropy_he | 0.368 | 0.431 | 0.389 | 0.393 | 0.618 | -0.309 |
| 21 | logprob_he | 0.629 | 0.563 | 0.608 | 0.602 | 0.604 | 0.305 |
| 21 | prompt_local_rmd | 0.339 | 0.564 | 0.556 | 0.574 | 0.886 | -0.359 |
| 21 | contrast_tail_q20 | 0.471 | 0.427 | 0.376 | 0.371 | 0.575 | -0.110 |

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
| 21 | 0.060 [0.032, 0.088] p=0.000 | 0.035 [0.016, 0.059] p=0.010 | 0.004 [-0.024, 0.033] p=0.650 |

## Truncation / parseability diagnostic

Unparsed (no final answer): 351/4000 (0.088); length-capped at 8192: 374 (0.093); unparsed share of the incorrect class: 0.293.

Unparsed traces are auto-labeled incorrect upstream and are usually truncated, not wrong-answer. The within-prompt metrics below restrict to traces that emitted a parseable answer; a large drop in mixed-prompt count or in the RMD-minus-entropy gap means the headline within-prompt signal was a truncation detector.

| Layer | Method | Parseable within macro | Parseable centered AUC | Mixed prompts |
|---:|:---|---:|---:|---:|
| 21 | entropy | 0.468 | 0.459 | 49 |
| 21 | logprob | 0.478 | 0.462 | 49 |
| 21 | length | 0.437 | 0.441 | 49 |
| 21 | activation_norm | 0.474 | 0.423 | 49 |
| 21 | centroid | 0.440 | 0.432 | 49 |
| 21 | raw | 0.568 | 0.546 | 49 |
| 21 | rmd | 0.456 | 0.446 | 49 |
| 21 | rmd_high_entropy_q20 | 0.498 | 0.447 | 49 |
| 21 | rmd_tail_q20 | 0.462 | 0.464 | 49 |
| 21 | rmd_random_q20 | 0.457 | 0.448 | 49 |
| 21 | entropy_he | 0.528 | 0.539 | 49 |
| 21 | logprob_he | 0.484 | 0.460 | 49 |
| 21 | prompt_local_rmd | 0.525 | 0.515 | 49 |
| 21 | contrast_tail_q20 | 0.523 | 0.570 | 49 |
| 21 | probe_outputs | 0.545 | 0.525 | 49 |
| 21 | probe_outputs_plus_rmd_high_entropy_q20 | 0.506 | 0.498 | 49 |
| 21 | probe_b0 | 0.545 | 0.525 | 49 |
| 21 | probe_b1 | 0.514 | 0.500 | 49 |
| 21 | probe_g_he | 0.477 | 0.475 | 49 |
| 21 | probe_g_random | 0.490 | 0.492 | 49 |
| 21 | probe_hidden_tail_q20 | 0.586 | 0.546 | 49 |

## Prespecified parseable score contrasts

Point estimates, raw 95% prompt-bootstrap intervals, and raw two-sided p-values are reported without post-hoc layer selection or multiplicity-adjusted claims.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | rmd_high_entropy_q20_minus_rmd | 0.001 [-0.016, 0.022] p=0.920 | 0.042 [0.010, 0.083] p=0.000 |
| 21 | rmd_tail_q20_minus_rmd | 0.018 [-0.024, 0.057] p=0.450 | 0.006 [-0.055, 0.066] p=0.930 |
| 21 | contrast_tail_q20_minus_rmd_tail_q20 | 0.106 [-0.056, 0.250] p=0.260 | 0.062 [-0.138, 0.233] p=0.530 |
| 21 | rmd_high_entropy_q20_minus_rmd_random_q20 | -0.001 [-0.022, 0.021] p=0.890 | 0.041 [0.012, 0.074] p=0.010 |
| 21 | rmd_high_entropy_q20_minus_logprob | -0.015 [-0.069, 0.026] p=0.510 | 0.020 [-0.034, 0.080] p=0.630 |
| 21 | rmd_tail_q20_minus_logprob | 0.003 [-0.066, 0.060] p=0.920 | -0.017 [-0.111, 0.070] p=0.600 |
| 21 | contrast_tail_q20_minus_logprob | 0.109 [-0.081, 0.265] p=0.270 | 0.045 [-0.176, 0.234] p=0.590 |
| 21 | probe_outputs_plus_rmd_high_entropy_q20_minus_probe_outputs | -0.026 [-0.052, 0.001] p=0.070 | -0.039 [-0.079, -0.002] p=0.030 |
| 21 | contrast_tail_q20_minus_rmd | 0.125 [-0.074, 0.298] p=0.240 | 0.067 [-0.119, 0.238] p=0.510 |
| 21 | rmd_random_q20_minus_rmd | 0.002 [-0.008, 0.012] p=0.600 | 0.001 [-0.015, 0.021] p=0.790 |
| 21 | rmd_random_q20_minus_logprob | -0.014 [-0.072, 0.029] p=0.540 | -0.021 [-0.093, 0.047] p=0.510 |

## Supervised hidden-state probe (exploratory)

LDA fit on PCA-projected region means, cross-fitted by prompt fold on pooled labels over parseable traces. This bounds how much of the geometry signal supervision on the same activations recovers. **Post-hoc, added 2026-07-29 -- not part of the pre-registered contrast set.**

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | probe_hidden_tail_q20_minus_rmd_tail_q20 | 0.082 [-0.013, 0.168] p=0.080 | 0.124 [-0.009, 0.247] p=0.060 |
| 21 | probe_hidden_tail_q20_minus_length | 0.105 [-0.001, 0.208] p=0.060 | 0.150 [0.020, 0.274] p=0.040 |

Rank correlation of each score against trace length (parseable only). A scorer that merely rediscovers "long traces are wrong" shows |rho| near 1.

| Layer | Score | Spearman vs length | Pearson vs length | n |
|---:|:---|---:|---:|---:|
| 21 | probe_hidden_tail_q20 | 0.231 | 0.198 | 3649 |
| 21 | rmd | 0.815 | 0.836 | 3649 |
| 21 | rmd_tail_q20 | 0.804 | 0.824 | 3649 |
| 21 | rmd_high_entropy_q20 | 0.798 | 0.816 | 3649 |
| 21 | entropy | 0.350 | 0.400 | 3649 |
| 21 | logprob | 0.369 | 0.420 | 3649 |

## E2 same-token output autopsy

Fixed cross-fitted probes: B0=global outputs, B1=global plus same high-entropy-token outputs, G_he=B1 plus high-entropy RMD, and G_random=B1 plus matched random-20% RMD. Only the two pre-specified geometry contrasts are shown here.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | probe_g_he_minus_probe_b1 | -0.024 [-0.052, -0.001] p=0.040 | -0.037 [-0.082, 0.005] p=0.150 |
| 21 | probe_g_he_minus_probe_g_random | -0.017 [-0.048, 0.018] p=0.340 | -0.013 [-0.058, 0.021] p=0.570 |

## Prompt-contrastive direction diagnostics

Directions are fit out-of-fold from parseable mixed training prompts. Each prompt contributes one normalized difference vector; alignment nulls shuffle labels within prompts while preserving class counts.

| Layer | Region | Prompt vectors | Observed alignment | Pairwise cosine | Null mean | Null 95% interval | p |
|---:|:---|---:|---:|---:|---:|:---|---:|
| 21 | tail_q20 | 41 | 0.165 | 0.003 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 37 | 0.151 | -0.004 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 37 | 0.207 | 0.016 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 39 | 0.176 | 0.005 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 42 | 0.165 | 0.004 | NA | [NA, NA] | NA |

## Parseable paired contrasts

Contrastive score minus baseline, using prompt-cluster bootstrap intervals.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | rmd_high_entropy_q20_minus_rmd | 0.001 [-0.016, 0.022] p=0.920 | 0.042 [0.010, 0.083] p=0.000 |
| 21 | rmd_tail_q20_minus_rmd | 0.018 [-0.024, 0.057] p=0.450 | 0.006 [-0.055, 0.066] p=0.930 |
| 21 | contrast_tail_q20_minus_rmd_tail_q20 | 0.106 [-0.056, 0.250] p=0.260 | 0.062 [-0.138, 0.233] p=0.530 |
| 21 | rmd_high_entropy_q20_minus_logprob | -0.015 [-0.069, 0.026] p=0.510 | 0.020 [-0.034, 0.080] p=0.630 |
| 21 | rmd_tail_q20_minus_logprob | 0.003 [-0.066, 0.060] p=0.920 | -0.017 [-0.111, 0.070] p=0.600 |
| 21 | contrast_tail_q20_minus_logprob | 0.109 [-0.081, 0.265] p=0.270 | 0.045 [-0.176, 0.234] p=0.590 |
| 21 | contrast_tail_q20_minus_rmd | 0.125 [-0.074, 0.298] p=0.240 | 0.067 [-0.119, 0.238] p=0.510 |
| 21 | rmd_random_q20_minus_rmd | 0.002 [-0.008, 0.012] p=0.600 | 0.001 [-0.015, 0.021] p=0.790 |
| 21 | rmd_random_q20_minus_logprob | -0.014 [-0.072, 0.029] p=0.540 | -0.021 [-0.093, 0.047] p=0.510 |

No layer was selected after observing these results.

Confidence intervals use a prompt-cluster bootstrap over fixed out-of-fold predictions; reference fitting is not repeated.
