# deepseek math500 prompt decomposition

Data: 500 complete prompts with N=8; partial_data=false.

| Layer | Method | Pooled AUC | Centered AUC | Within macro | Within pair | ICC | Spearman difficulty |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 21 | entropy | 0.609 | 0.561 | 0.593 | 0.587 | 0.668 | 0.255 |
| 21 | logprob | 0.611 | 0.559 | 0.596 | 0.592 | 0.660 | 0.261 |
| 21 | length | 0.701 | 0.640 | 0.731 | 0.732 | 0.807 | 0.410 |
| 21 | activation_norm | 0.343 | 0.388 | 0.385 | 0.386 | 0.812 | -0.341 |
| 21 | centroid | 0.324 | 0.511 | 0.528 | 0.529 | 0.868 | -0.389 |
| 21 | raw | 0.345 | 0.586 | 0.627 | 0.628 | 0.894 | -0.348 |
| 21 | rmd | 0.756 | 0.676 | 0.744 | 0.746 | 0.897 | 0.503 |
| 21 | rmd_high_entropy_q20 | 0.729 | 0.655 | 0.722 | 0.729 | 0.883 | 0.461 |
| 21 | rmd_tail_q20 | 0.763 | 0.742 | 0.767 | 0.768 | 0.830 | 0.521 |
| 21 | rmd_random_q20 | 0.757 | 0.678 | 0.744 | 0.744 | 0.893 | 0.505 |
| 21 | entropy_he | 0.368 | 0.431 | 0.389 | 0.393 | 0.618 | -0.309 |
| 21 | logprob_he | 0.629 | 0.563 | 0.608 | 0.602 | 0.604 | 0.305 |
| 21 | prompt_local_rmd | 0.340 | 0.564 | 0.552 | 0.573 | 0.885 | -0.356 |
| 21 | contrast_tail_q20 | 0.463 | 0.429 | 0.391 | 0.387 | 0.509 | -0.126 |

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
| 21 | 0.055 [0.031, 0.080] p=0.000 | 0.036 [0.017, 0.057] p=0.000 | 0.013 [-0.012, 0.040] p=0.330 |

## Truncation / parseability diagnostic

Unparsed (no final answer): 351/4000 (0.088); length-capped at 8192: 374 (0.093); unparsed share of the incorrect class: 0.293.

Unparsed traces are auto-labeled incorrect upstream and are usually truncated, not wrong-answer. The within-prompt metrics below restrict to traces that emitted a parseable answer; a large drop in mixed-prompt count or in the RMD-minus-entropy gap means the headline within-prompt signal was a truncation detector.

| Layer | Method | Parseable within macro | Parseable centered AUC | Mixed prompts |
|---:|:---|---:|---:|---:|
| 21 | entropy | 0.468 | 0.459 | 49 |
| 21 | logprob | 0.478 | 0.462 | 49 |
| 21 | length | 0.437 | 0.441 | 49 |
| 21 | activation_norm | 0.474 | 0.423 | 49 |
| 21 | centroid | 0.434 | 0.428 | 49 |
| 21 | raw | 0.570 | 0.544 | 49 |
| 21 | rmd | 0.472 | 0.449 | 49 |
| 21 | rmd_high_entropy_q20 | 0.497 | 0.451 | 49 |
| 21 | rmd_tail_q20 | 0.473 | 0.468 | 49 |
| 21 | rmd_random_q20 | 0.460 | 0.453 | 49 |
| 21 | entropy_he | 0.528 | 0.539 | 49 |
| 21 | logprob_he | 0.484 | 0.460 | 49 |
| 21 | prompt_local_rmd | 0.515 | 0.513 | 49 |
| 21 | contrast_tail_q20 | 0.554 | 0.586 | 49 |
| 21 | probe_outputs | 0.507 | 0.528 | 49 |
| 21 | probe_outputs_plus_rmd_high_entropy_q20 | 0.501 | 0.526 | 49 |
| 21 | probe_b0 | 0.507 | 0.528 | 49 |
| 21 | probe_b1 | 0.505 | 0.509 | 49 |
| 21 | probe_g_he | 0.504 | 0.508 | 49 |
| 21 | probe_g_random | 0.497 | 0.504 | 49 |
| 21 | probe_hidden_tail_q20 | 0.621 | 0.552 | 49 |

## Prespecified parseable score contrasts

Point estimates, raw 95% prompt-bootstrap intervals, and raw two-sided p-values are reported without post-hoc layer selection or multiplicity-adjusted claims.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | rmd_high_entropy_q20_minus_rmd | 0.002 [-0.017, 0.024] p=0.770 | 0.025 [-0.015, 0.067] p=0.230 |
| 21 | rmd_tail_q20_minus_rmd | 0.019 [-0.022, 0.060] p=0.330 | 0.001 [-0.063, 0.057] p=0.990 |
| 21 | contrast_tail_q20_minus_rmd_tail_q20 | 0.118 [-0.027, 0.296] p=0.140 | 0.081 [-0.093, 0.298] p=0.350 |
| 21 | rmd_high_entropy_q20_minus_rmd_random_q20 | -0.002 [-0.022, 0.019] p=0.830 | 0.037 [0.006, 0.081] p=0.050 |
| 21 | rmd_high_entropy_q20_minus_logprob | -0.011 [-0.058, 0.033] p=0.620 | 0.019 [-0.053, 0.081] p=0.530 |
| 21 | rmd_tail_q20_minus_logprob | 0.006 [-0.051, 0.067] p=0.930 | -0.005 [-0.088, 0.076] p=0.880 |
| 21 | contrast_tail_q20_minus_logprob | 0.124 [-0.037, 0.320] p=0.130 | 0.075 [-0.099, 0.279] p=0.370 |
| 21 | probe_outputs_plus_rmd_high_entropy_q20_minus_probe_outputs | -0.002 [-0.010, 0.004] p=0.450 | -0.006 [-0.026, 0.010] p=0.510 |
| 21 | contrast_tail_q20_minus_rmd | 0.137 [-0.038, 0.352] p=0.110 | 0.082 [-0.099, 0.290] p=0.360 |
| 21 | rmd_random_q20_minus_rmd | 0.005 [-0.005, 0.018] p=0.330 | -0.013 [-0.042, 0.007] p=0.270 |
| 21 | rmd_random_q20_minus_logprob | -0.008 [-0.060, 0.043] p=0.810 | -0.019 [-0.106, 0.051] p=0.660 |

## Supervised hidden-state probe (exploratory)

LDA fit on PCA-projected region means, cross-fitted by prompt fold on pooled labels over parseable traces. This bounds how much of the geometry signal supervision on the same activations recovers. **Post-hoc, added 2026-07-29 -- not part of the pre-registered contrast set.**

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | probe_hidden_tail_q20_minus_rmd_tail_q20 | 0.085 [0.004, 0.165] p=0.030 | 0.148 [0.021, 0.258] p=0.000 |
| 21 | probe_hidden_tail_q20_minus_length | 0.111 [0.011, 0.205] p=0.030 | 0.184 [0.063, 0.303] p=0.000 |

Rank correlation of each score against trace length (parseable only). A scorer that merely rediscovers "long traces are wrong" shows |rho| near 1.

| Layer | Score | Spearman vs length | Pearson vs length | n |
|---:|:---|---:|---:|---:|
| 21 | probe_hidden_tail_q20 | 0.226 | 0.205 | 3649 |
| 21 | rmd | 0.816 | 0.838 | 3649 |
| 21 | rmd_tail_q20 | 0.799 | 0.819 | 3649 |
| 21 | rmd_high_entropy_q20 | 0.806 | 0.823 | 3649 |
| 21 | entropy | 0.350 | 0.400 | 3649 |
| 21 | logprob | 0.369 | 0.420 | 3649 |

## E2 same-token output autopsy

Fixed cross-fitted probes: B0=global outputs, B1=global plus same high-entropy-token outputs, G_he=B1 plus high-entropy RMD, and G_random=B1 plus matched random-20% RMD. Only the two pre-specified geometry contrasts are shown here.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | probe_g_he_minus_probe_b1 | -0.000 [-0.007, 0.005] p=0.910 | -0.001 [-0.023, 0.017] p=0.960 |
| 21 | probe_g_he_minus_probe_g_random | 0.005 [-0.006, 0.019] p=0.420 | 0.007 [-0.036, 0.050] p=0.670 |

## Prompt-contrastive direction diagnostics

Directions are fit out-of-fold from parseable mixed training prompts. Each prompt contributes one normalized difference vector; alignment nulls shuffle labels within prompts while preserving class counts.

| Layer | Region | Prompt vectors | Observed alignment | Pairwise cosine | Null mean | Null 95% interval | p |
|---:|:---|---:|---:|---:|---:|:---|---:|
| 21 | tail_q20 | 39 | 0.160 | 0.000 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 38 | 0.171 | 0.003 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 36 | 0.203 | 0.014 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 38 | 0.164 | 0.000 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 45 | 0.149 | 0.000 | NA | [NA, NA] | NA |

## Parseable paired contrasts

Contrastive score minus baseline, using prompt-cluster bootstrap intervals.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | rmd_high_entropy_q20_minus_rmd | 0.002 [-0.017, 0.024] p=0.770 | 0.025 [-0.015, 0.067] p=0.230 |
| 21 | rmd_tail_q20_minus_rmd | 0.019 [-0.022, 0.060] p=0.330 | 0.001 [-0.063, 0.057] p=0.990 |
| 21 | contrast_tail_q20_minus_rmd_tail_q20 | 0.118 [-0.027, 0.296] p=0.140 | 0.081 [-0.093, 0.298] p=0.350 |
| 21 | rmd_high_entropy_q20_minus_logprob | -0.011 [-0.058, 0.033] p=0.620 | 0.019 [-0.053, 0.081] p=0.530 |
| 21 | rmd_tail_q20_minus_logprob | 0.006 [-0.051, 0.067] p=0.930 | -0.005 [-0.088, 0.076] p=0.880 |
| 21 | contrast_tail_q20_minus_logprob | 0.124 [-0.037, 0.320] p=0.130 | 0.075 [-0.099, 0.279] p=0.370 |
| 21 | contrast_tail_q20_minus_rmd | 0.137 [-0.038, 0.352] p=0.110 | 0.082 [-0.099, 0.290] p=0.360 |
| 21 | rmd_random_q20_minus_rmd | 0.005 [-0.005, 0.018] p=0.330 | -0.013 [-0.042, 0.007] p=0.270 |
| 21 | rmd_random_q20_minus_logprob | -0.008 [-0.060, 0.043] p=0.810 | -0.019 [-0.106, 0.051] p=0.660 |

No layer was selected after observing these results.

Confidence intervals use a prompt-cluster bootstrap over fixed out-of-fold predictions; reference fitting is not repeated.
