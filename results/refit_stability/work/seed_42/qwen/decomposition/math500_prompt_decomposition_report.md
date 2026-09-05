# qwen math500 prompt decomposition

Data: 500 complete prompts with N=8; partial_data=false.

| Layer | Method | Pooled AUC | Centered AUC | Within macro | Within pair | ICC | Spearman difficulty |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 21 | entropy | 0.571 | 0.559 | 0.599 | 0.595 | 0.738 | 0.151 |
| 21 | logprob | 0.575 | 0.559 | 0.595 | 0.589 | 0.669 | 0.153 |
| 21 | length | 0.737 | 0.563 | 0.581 | 0.582 | 0.867 | 0.478 |
| 21 | activation_norm | 0.279 | 0.441 | 0.431 | 0.442 | 0.907 | -0.449 |
| 21 | centroid | 0.322 | 0.472 | 0.435 | 0.456 | 0.870 | -0.367 |
| 21 | raw | 0.324 | 0.467 | 0.423 | 0.441 | 0.901 | -0.357 |
| 21 | rmd | 0.786 | 0.592 | 0.616 | 0.602 | 0.960 | 0.547 |
| 21 | rmd_high_entropy_q20 | 0.787 | 0.630 | 0.672 | 0.660 | 0.955 | 0.543 |
| 21 | rmd_tail_q20 | 0.839 | 0.640 | 0.658 | 0.653 | 0.879 | 0.652 |
| 21 | rmd_random_q20 | 0.784 | 0.586 | 0.596 | 0.589 | 0.942 | 0.548 |
| 21 | entropy_he | 0.426 | 0.442 | 0.403 | 0.407 | 0.744 | -0.154 |
| 21 | logprob_he | 0.576 | 0.559 | 0.597 | 0.592 | 0.666 | 0.156 |
| 21 | prompt_local_rmd | 0.320 | 0.523 | 0.511 | 0.529 | 0.940 | -0.367 |
| 21 | contrast_tail_q20 | 0.727 | 0.581 | 0.595 | 0.593 | 0.744 | 0.482 |

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
| 21 | 0.049 [0.020, 0.080] p=0.000 | 0.029 [-0.016, 0.070] p=0.180 | 0.036 [-0.036, 0.092] p=0.240 |

## Truncation / parseability diagnostic

Unparsed (no final answer): 328/4000 (0.082); length-capped at 1024: 338 (0.085); unparsed share of the incorrect class: 0.185.

Unparsed traces are auto-labeled incorrect upstream and are usually truncated, not wrong-answer. The within-prompt metrics below restrict to traces that emitted a parseable answer; a large drop in mixed-prompt count or in the RMD-minus-entropy gap means the headline within-prompt signal was a truncation detector.

| Layer | Method | Parseable within macro | Parseable centered AUC | Mixed prompts |
|---:|:---|---:|---:|---:|
| 21 | entropy | 0.660 | 0.611 | 117 |
| 21 | logprob | 0.649 | 0.609 | 117 |
| 21 | length | 0.478 | 0.474 | 117 |
| 21 | activation_norm | 0.483 | 0.485 | 117 |
| 21 | centroid | 0.412 | 0.455 | 117 |
| 21 | raw | 0.421 | 0.474 | 117 |
| 21 | rmd | 0.574 | 0.547 | 117 |
| 21 | rmd_high_entropy_q20 | 0.654 | 0.605 | 117 |
| 21 | rmd_tail_q20 | 0.598 | 0.576 | 117 |
| 21 | rmd_random_q20 | 0.564 | 0.548 | 117 |
| 21 | entropy_he | 0.345 | 0.388 | 117 |
| 21 | logprob_he | 0.652 | 0.608 | 117 |
| 21 | prompt_local_rmd | 0.483 | 0.500 | 117 |
| 21 | contrast_tail_q20 | 0.611 | 0.598 | 117 |
| 21 | probe_outputs | 0.634 | 0.590 | 117 |
| 21 | probe_outputs_plus_rmd_high_entropy_q20 | 0.683 | 0.608 | 117 |
| 21 | probe_b0 | 0.634 | 0.590 | 117 |
| 21 | probe_b1 | 0.619 | 0.589 | 117 |
| 21 | probe_g_he | 0.672 | 0.607 | 117 |
| 21 | probe_g_random | 0.641 | 0.593 | 117 |
| 21 | probe_hidden_tail_q20 | 0.546 | 0.536 | 117 |

## Prespecified parseable score contrasts

Point estimates, raw 95% prompt-bootstrap intervals, and raw two-sided p-values are reported without post-hoc layer selection or multiplicity-adjusted claims.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | rmd_high_entropy_q20_minus_rmd | 0.058 [0.020, 0.101] p=0.000 | 0.080 [0.029, 0.132] p=0.000 |
| 21 | rmd_tail_q20_minus_rmd | 0.029 [-0.006, 0.055] p=0.090 | 0.025 [-0.021, 0.068] p=0.360 |
| 21 | contrast_tail_q20_minus_rmd_tail_q20 | 0.022 [-0.012, 0.071] p=0.260 | 0.013 [-0.047, 0.081] p=0.620 |
| 21 | rmd_high_entropy_q20_minus_rmd_random_q20 | 0.057 [0.019, 0.098] p=0.000 | 0.091 [0.038, 0.144] p=0.010 |
| 21 | rmd_high_entropy_q20_minus_logprob | -0.004 [-0.062, 0.046] p=0.780 | 0.005 [-0.075, 0.066] p=0.970 |
| 21 | rmd_tail_q20_minus_logprob | -0.033 [-0.111, 0.031] p=0.280 | -0.051 [-0.151, 0.029] p=0.180 |
| 21 | contrast_tail_q20_minus_logprob | -0.011 [-0.072, 0.036] p=0.660 | -0.038 [-0.116, 0.034] p=0.360 |
| 21 | probe_outputs_plus_rmd_high_entropy_q20_minus_probe_outputs | 0.018 [-0.011, 0.043] p=0.250 | 0.049 [0.001, 0.086] p=0.050 |
| 21 | contrast_tail_q20_minus_rmd | 0.051 [0.008, 0.100] p=0.020 | 0.037 [-0.020, 0.102] p=0.160 |
| 21 | rmd_random_q20_minus_rmd | 0.001 [-0.016, 0.024] p=0.890 | -0.010 [-0.037, 0.030] p=0.640 |
| 21 | rmd_random_q20_minus_logprob | -0.061 [-0.132, -0.000] p=0.040 | -0.086 [-0.174, -0.011] p=0.010 |

## Supervised hidden-state probe (exploratory)

LDA fit on PCA-projected region means, cross-fitted by prompt fold on pooled labels over parseable traces. This bounds how much of the geometry signal supervision on the same activations recovers. **Post-hoc, added 2026-07-29 -- not part of the pre-registered contrast set.**

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | probe_hidden_tail_q20_minus_rmd_tail_q20 | -0.040 [-0.078, 0.001] p=0.070 | -0.052 [-0.106, 0.007] p=0.070 |
| 21 | probe_hidden_tail_q20_minus_length | 0.062 [0.009, 0.113] p=0.030 | 0.068 [-0.004, 0.145] p=0.060 |

Rank correlation of each score against trace length (parseable only). A scorer that merely rediscovers "long traces are wrong" shows |rho| near 1.

| Layer | Score | Spearman vs length | Pearson vs length | n |
|---:|:---|---:|---:|---:|
| 21 | probe_hidden_tail_q20 | 0.425 | 0.413 | 3672 |
| 21 | rmd | 0.658 | 0.661 | 3672 |
| 21 | rmd_tail_q20 | 0.675 | 0.669 | 3672 |
| 21 | rmd_high_entropy_q20 | 0.615 | 0.594 | 3672 |
| 21 | entropy | -0.163 | -0.145 | 3672 |
| 21 | logprob | -0.134 | -0.131 | 3672 |

## E2 same-token output autopsy

Fixed cross-fitted probes: B0=global outputs, B1=global plus same high-entropy-token outputs, G_he=B1 plus high-entropy RMD, and G_random=B1 plus matched random-20% RMD. Only the two pre-specified geometry contrasts are shown here.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | probe_g_he_minus_probe_b1 | 0.018 [-0.006, 0.044] p=0.160 | 0.054 [0.008, 0.097] p=0.010 |
| 21 | probe_g_he_minus_probe_g_random | 0.014 [-0.004, 0.031] p=0.180 | 0.032 [-0.002, 0.070] p=0.080 |

## Prompt-contrastive direction diagnostics

Directions are fit out-of-fold from parseable mixed training prompts. Each prompt contributes one normalized difference vector; alignment nulls shuffle labels within prompts while preserving class counts.

| Layer | Region | Prompt vectors | Observed alignment | Pairwise cosine | Null mean | Null 95% interval | p |
|---:|:---|---:|---:|---:|---:|:---|---:|
| 21 | tail_q20 | 97 | 0.151 | 0.013 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 89 | 0.165 | 0.016 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 95 | 0.147 | 0.011 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 92 | 0.137 | 0.008 | NA | [NA, NA] | NA |
| 21 | tail_q20 | 95 | 0.145 | 0.011 | NA | [NA, NA] | NA |

## Parseable paired contrasts

Contrastive score minus baseline, using prompt-cluster bootstrap intervals.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 21 | rmd_high_entropy_q20_minus_rmd | 0.058 [0.020, 0.101] p=0.000 | 0.080 [0.029, 0.132] p=0.000 |
| 21 | rmd_tail_q20_minus_rmd | 0.029 [-0.006, 0.055] p=0.090 | 0.025 [-0.021, 0.068] p=0.360 |
| 21 | contrast_tail_q20_minus_rmd_tail_q20 | 0.022 [-0.012, 0.071] p=0.260 | 0.013 [-0.047, 0.081] p=0.620 |
| 21 | rmd_high_entropy_q20_minus_logprob | -0.004 [-0.062, 0.046] p=0.780 | 0.005 [-0.075, 0.066] p=0.970 |
| 21 | rmd_tail_q20_minus_logprob | -0.033 [-0.111, 0.031] p=0.280 | -0.051 [-0.151, 0.029] p=0.180 |
| 21 | contrast_tail_q20_minus_logprob | -0.011 [-0.072, 0.036] p=0.660 | -0.038 [-0.116, 0.034] p=0.360 |
| 21 | contrast_tail_q20_minus_rmd | 0.051 [0.008, 0.100] p=0.020 | 0.037 [-0.020, 0.102] p=0.160 |
| 21 | rmd_random_q20_minus_rmd | 0.001 [-0.016, 0.024] p=0.890 | -0.010 [-0.037, 0.030] p=0.640 |
| 21 | rmd_random_q20_minus_logprob | -0.061 [-0.132, -0.000] p=0.040 | -0.086 [-0.174, -0.011] p=0.010 |

No layer was selected after observing these results.

Confidence intervals use a prompt-cluster bootstrap over fixed out-of-fold predictions; reference fitting is not repeated.
