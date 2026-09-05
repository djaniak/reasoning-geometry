# deepseek math500 prompt decomposition

Data: 500 complete prompts with N=8; partial_data=false.

| Layer | Method | Pooled AUC | Centered AUC | Within macro | Within pair | ICC | Spearman difficulty |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 21 | entropy | 0.609 | 0.561 | 0.593 | 0.587 | 0.668 | 0.255 |
| 21 | logprob | 0.611 | 0.559 | 0.596 | 0.592 | 0.660 | 0.261 |
| 21 | length | 0.701 | 0.640 | 0.731 | 0.732 | 0.807 | 0.410 |
| 21 | activation_norm | 0.343 | 0.388 | 0.385 | 0.386 | 0.812 | -0.341 |
| 21 | centroid | 0.325 | 0.511 | 0.535 | 0.535 | 0.869 | -0.388 |
| 21 | raw | 0.346 | 0.585 | 0.622 | 0.623 | 0.895 | -0.348 |
| 21 | rmd | 0.758 | 0.676 | 0.735 | 0.739 | 0.898 | 0.506 |
| 21 | rmd_high_entropy_q20 | 0.735 | 0.656 | 0.720 | 0.725 | 0.884 | 0.472 |
| 21 | rmd_tail_q20 | 0.766 | 0.741 | 0.760 | 0.762 | 0.831 | 0.527 |
| 21 | rmd_random_q20 | 0.759 | 0.677 | 0.732 | 0.735 | 0.895 | 0.507 |
| 21 | entropy_he | 0.368 | 0.431 | 0.389 | 0.393 | 0.618 | -0.309 |
| 21 | logprob_he | 0.629 | 0.563 | 0.608 | 0.602 | 0.604 | 0.305 |
| 21 | prompt_local_rmd | 0.341 | 0.565 | 0.561 | 0.581 | 0.885 | -0.355 |
| 21 | contrast_tail_q20 | 0.446 | 0.397 | 0.335 | 0.332 | 0.631 | -0.111 |

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
| 21 | 0.057 [0.032, 0.080] p=0.000 | 0.036 [0.012, 0.053] p=0.010 | 0.004 [-0.029, 0.039] p=0.780 |

## Truncation / parseability diagnostic

Unparsed (no final answer): 351/4000 (0.088); length-capped at 8192: 374 (0.093); unparsed share of the incorrect class: 0.293.

Unparsed traces are auto-labeled incorrect upstream and are usually truncated, not wrong-answer. The within-prompt metrics below restrict to traces that emitted a parseable answer; a large drop in mixed-prompt count or in the RMD-minus-entropy gap means the headline within-prompt signal was a truncation detector.

| Layer | Method | Parseable within macro | Parseable centered AUC | Mixed prompts |
|---:|:---|---:|---:|---:|
| 21 | entropy | 0.468 | 0.459 | 49 |
| 21 | logprob | 0.478 | 0.462 | 49 |
| 21 | length | 0.437 | 0.441 | 49 |
| 21 | activation_norm | 0.474 | 0.423 | 49 |
| 21 | centroid | 0.444 | 0.430 | 49 |
| 21 | raw | 0.579 | 0.548 | 49 |
| 21 | rmd | 0.469 | 0.447 | 49 |
| 21 | rmd_high_entropy_q20 | 0.493 | 0.452 | 49 |
| 21 | rmd_tail_q20 | 0.459 | 0.464 | 49 |
| 21 | rmd_random_q20 | 0.453 | 0.452 | 49 |
| 21 | entropy_he | 0.528 | 0.539 | 49 |
| 21 | logprob_he | 0.484 | 0.460 | 49 |
| 21 | prompt_local_rmd | 0.523 | 0.515 | 49 |
| 21 | contrast_tail_q20 | 0.456 | 0.523 | 49 |
| 21 | probe_outputs | 0.527 | 0.515 | 49 |
| 21 | probe_outputs_plus_rmd_high_entropy_q20 | 0.445 | 0.476 | 49 |
| 21 | probe_b0 | 0.527 | 0.515 | 49 |
| 21 | probe_b1 | 0.499 | 0.503 | 49 |
| 21 | probe_g_he | 0.438 | 0.467 | 49 |
| 21 | probe_g_random | 0.463 | 0.483 | 49 |
| 21 | probe_hidden_tail_q20 | 0.536 | 0.520 | 49 |

## Prespecified parseable score contrasts

Point estimates, raw 95% prompt-bootstrap intervals, and raw two-sided p-values are reported without post-hoc layer selection or multiplicity-adjusted claims.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | rmd_high_entropy_q20_minus_rmd | 0.005 [-0.014, 0.028] p=0.580 | 0.024 [-0.015, 0.063] p=0.210 |
| 21 | rmd_tail_q20_minus_rmd | 0.017 [-0.022, 0.054] p=0.340 | -0.010 [-0.062, 0.044] p=0.790 |
| 21 | contrast_tail_q20_minus_rmd_tail_q20 | 0.059 [-0.112, 0.223] p=0.480 | -0.003 [-0.185, 0.173] p=0.990 |
| 21 | rmd_high_entropy_q20_minus_rmd_random_q20 | 0.001 [-0.019, 0.027] p=0.850 | 0.040 [0.001, 0.084] p=0.040 |
| 21 | rmd_high_entropy_q20_minus_logprob | -0.009 [-0.048, 0.039] p=0.880 | 0.014 [-0.051, 0.081] p=0.550 |
| 21 | rmd_tail_q20_minus_logprob | 0.003 [-0.056, 0.064] p=0.820 | -0.020 [-0.087, 0.070] p=0.710 |
| 21 | contrast_tail_q20_minus_logprob | 0.061 [-0.105, 0.226] p=0.520 | -0.022 [-0.192, 0.159] p=0.890 |
| 21 | probe_outputs_plus_rmd_high_entropy_q20_minus_probe_outputs | -0.039 [-0.070, -0.014] p=0.030 | -0.082 [-0.136, -0.021] p=0.010 |
| 21 | contrast_tail_q20_minus_rmd | 0.075 [-0.115, 0.247] p=0.440 | -0.013 [-0.203, 0.151] p=0.900 |
| 21 | rmd_random_q20_minus_rmd | 0.004 [-0.007, 0.015] p=0.460 | -0.016 [-0.040, 0.002] p=0.080 |
| 21 | rmd_random_q20_minus_logprob | -0.010 [-0.056, 0.039] p=0.860 | -0.026 [-0.093, 0.049] p=0.570 |

## Supervised hidden-state probe (exploratory)

LDA fit on PCA-projected region means, cross-fitted by prompt fold on pooled labels over parseable traces. This bounds how much of the geometry signal supervision on the same activations recovers. **Post-hoc, added 2026-07-29 -- not part of the pre-registered contrast set.**

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | probe_hidden_tail_q20_minus_rmd_tail_q20 | 0.056 [-0.023, 0.135] p=0.130 | 0.078 [-0.042, 0.194] p=0.190 |
| 21 | probe_hidden_tail_q20_minus_length | 0.079 [-0.006, 0.164] p=0.080 | 0.100 [-0.031, 0.224] p=0.150 |

Rank correlation of each score against trace length (parseable only). A scorer that merely rediscovers "long traces are wrong" shows |rho| near 1.

| Layer | Score | Spearman vs length | Pearson vs length | n |
|---:|:---|---:|---:|---:|
| 21 | probe_hidden_tail_q20 | 0.222 | 0.196 | 3649 |
| 21 | rmd | 0.825 | 0.846 | 3649 |
| 21 | rmd_tail_q20 | 0.816 | 0.832 | 3649 |
| 21 | rmd_high_entropy_q20 | 0.812 | 0.828 | 3649 |
| 21 | entropy | 0.350 | 0.400 | 3649 |
| 21 | logprob | 0.369 | 0.420 | 3649 |

## E2 same-token output autopsy

Fixed cross-fitted probes: B0=global outputs, B1=global plus same high-entropy-token outputs, G_he=B1 plus high-entropy RMD, and G_random=B1 plus matched random-20% RMD. Only the two pre-specified geometry contrasts are shown here.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | probe_g_he_minus_probe_b1 | -0.036 [-0.066, -0.009] p=0.010 | -0.061 [-0.130, -0.009] p=0.020 |
| 21 | probe_g_he_minus_probe_g_random | -0.016 [-0.052, 0.016] p=0.350 | -0.025 [-0.106, 0.031] p=0.480 |

## Prompt-contrastive direction diagnostics

Directions are fit out-of-fold from parseable mixed training prompts. Each prompt contributes one normalized difference vector; alignment nulls shuffle labels within prompts while preserving class counts.

| Layer | Region | Prompt vectors | Observed alignment | Pairwise cosine | Null mean | Null 95% interval | p |
|---:|:---|---:|---:|---:|---:|:---|---:|
| 21 | tail_q20 | 40 | 0.173 | 0.005 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 38 | 0.177 | 0.005 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 41 | 0.207 | 0.019 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 36 | 0.170 | 0.001 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 41 | 0.166 | 0.003 | NA | [NA, NA] | NA |

## Parseable paired contrasts

Contrastive score minus baseline, using prompt-cluster bootstrap intervals.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | rmd_high_entropy_q20_minus_rmd | 0.005 [-0.014, 0.028] p=0.580 | 0.024 [-0.015, 0.063] p=0.210 |
| 21 | rmd_tail_q20_minus_rmd | 0.017 [-0.022, 0.054] p=0.340 | -0.010 [-0.062, 0.044] p=0.790 |
| 21 | contrast_tail_q20_minus_rmd_tail_q20 | 0.059 [-0.112, 0.223] p=0.480 | -0.003 [-0.185, 0.173] p=0.990 |
| 21 | rmd_high_entropy_q20_minus_logprob | -0.009 [-0.048, 0.039] p=0.880 | 0.014 [-0.051, 0.081] p=0.550 |
| 21 | rmd_tail_q20_minus_logprob | 0.003 [-0.056, 0.064] p=0.820 | -0.020 [-0.087, 0.070] p=0.710 |
| 21 | contrast_tail_q20_minus_logprob | 0.061 [-0.105, 0.226] p=0.520 | -0.022 [-0.192, 0.159] p=0.890 |
| 21 | contrast_tail_q20_minus_rmd | 0.075 [-0.115, 0.247] p=0.440 | -0.013 [-0.203, 0.151] p=0.900 |
| 21 | rmd_random_q20_minus_rmd | 0.004 [-0.007, 0.015] p=0.460 | -0.016 [-0.040, 0.002] p=0.080 |
| 21 | rmd_random_q20_minus_logprob | -0.010 [-0.056, 0.039] p=0.860 | -0.026 [-0.093, 0.049] p=0.570 |

No layer was selected after observing these results.

Confidence intervals use a prompt-cluster bootstrap over fixed out-of-fold predictions; reference fitting is not repeated.
