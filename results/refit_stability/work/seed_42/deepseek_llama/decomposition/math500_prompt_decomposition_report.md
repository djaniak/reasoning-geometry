# deepseek_llama math500 prompt decomposition

Data: 500 complete prompts with N=8; partial_data=false.

| Layer | Method | Pooled AUC | Centered AUC | Within macro | Within pair | ICC | Spearman difficulty |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 24 | entropy | 0.546 | 0.470 | 0.478 | 0.476 | 0.640 | 0.130 |
| 24 | logprob | 0.544 | 0.470 | 0.483 | 0.480 | 0.640 | 0.131 |
| 24 | length | 0.581 | 0.506 | 0.525 | 0.530 | 0.834 | 0.199 |
| 24 | activation_norm | 0.452 | 0.383 | 0.389 | 0.379 | 0.641 | -0.047 |
| 24 | centroid | 0.399 | 0.372 | 0.380 | 0.370 | 0.790 | -0.187 |
| 24 | raw | 0.427 | 0.384 | 0.386 | 0.377 | 0.782 | -0.126 |
| 24 | rmd | 0.674 | 0.537 | 0.532 | 0.538 | 0.893 | 0.382 |
| 24 | rmd_high_entropy_q20 | 0.672 | 0.555 | 0.571 | 0.570 | 0.881 | 0.371 |
| 24 | rmd_tail_q20 | 0.729 | 0.620 | 0.642 | 0.638 | 0.799 | 0.475 |
| 24 | rmd_random_q20 | 0.673 | 0.535 | 0.542 | 0.548 | 0.885 | 0.380 |
| 24 | entropy_he | 0.452 | 0.531 | 0.516 | 0.521 | 0.643 | -0.137 |
| 24 | logprob_he | 0.547 | 0.467 | 0.483 | 0.479 | 0.641 | 0.138 |
| 24 | prompt_local_rmd | 0.436 | 0.540 | 0.503 | 0.526 | 0.802 | -0.142 |
| 24 | contrast_tail_q20 | 0.580 | 0.687 | 0.726 | 0.730 | 0.687 | 0.078 |

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
| 24 | 0.094 [0.074, 0.117] p=0.000 | 0.031 [0.014, 0.053] p=0.000 | 0.007 [-0.026, 0.045] p=0.750 |

## Truncation / parseability diagnostic

Unparsed (no final answer): 229/4000 (0.057); length-capped at 12288: 250 (0.062); unparsed share of the incorrect class: 0.128.

Unparsed traces are auto-labeled incorrect upstream and are usually truncated, not wrong-answer. The within-prompt metrics below restrict to traces that emitted a parseable answer; a large drop in mixed-prompt count or in the RMD-minus-entropy gap means the headline within-prompt signal was a truncation detector.

| Layer | Method | Parseable within macro | Parseable centered AUC | Mixed prompts |
|---:|:---|---:|---:|---:|
| 24 | entropy | 0.446 | 0.435 | 158 |
| 24 | logprob | 0.447 | 0.436 | 158 |
| 24 | length | 0.422 | 0.423 | 158 |
| 24 | activation_norm | 0.379 | 0.372 | 158 |
| 24 | centroid | 0.379 | 0.361 | 158 |
| 24 | raw | 0.373 | 0.370 | 158 |
| 24 | rmd | 0.443 | 0.465 | 158 |
| 24 | rmd_high_entropy_q20 | 0.499 | 0.491 | 158 |
| 24 | rmd_tail_q20 | 0.554 | 0.531 | 158 |
| 24 | rmd_random_q20 | 0.456 | 0.465 | 158 |
| 24 | entropy_he | 0.560 | 0.568 | 158 |
| 24 | logprob_he | 0.441 | 0.432 | 158 |
| 24 | prompt_local_rmd | 0.511 | 0.545 | 158 |
| 24 | contrast_tail_q20 | 0.664 | 0.650 | 158 |
| 24 | probe_outputs | 0.564 | 0.576 | 158 |
| 24 | probe_outputs_plus_rmd_high_entropy_q20 | 0.607 | 0.596 | 158 |
| 24 | probe_b0 | 0.564 | 0.576 | 158 |
| 24 | probe_b1 | 0.560 | 0.573 | 158 |
| 24 | probe_g_he | 0.588 | 0.594 | 158 |
| 24 | probe_g_random | 0.566 | 0.579 | 158 |
| 24 | probe_hidden_tail_q20 | 0.615 | 0.593 | 158 |

## Prespecified parseable score contrasts

Point estimates, raw 95% prompt-bootstrap intervals, and raw two-sided p-values are reported without post-hoc layer selection or multiplicity-adjusted claims.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 24 | rmd_high_entropy_q20_minus_rmd | 0.026 [0.007, 0.044] p=0.000 | 0.056 [0.024, 0.086] p=0.000 |
| 24 | rmd_tail_q20_minus_rmd | 0.066 [0.037, 0.093] p=0.000 | 0.111 [0.072, 0.150] p=0.000 |
| 24 | contrast_tail_q20_minus_rmd_tail_q20 | 0.118 [0.068, 0.166] p=0.000 | 0.109 [0.049, 0.159] p=0.000 |
| 24 | rmd_high_entropy_q20_minus_rmd_random_q20 | 0.025 [0.007, 0.050] p=0.000 | 0.044 [-0.003, 0.084] p=0.070 |
| 24 | rmd_high_entropy_q20_minus_logprob | 0.055 [0.028, 0.086] p=0.000 | 0.052 [0.009, 0.103] p=0.050 |
| 24 | rmd_tail_q20_minus_logprob | 0.095 [0.055, 0.143] p=0.000 | 0.107 [0.051, 0.168] p=0.000 |
| 24 | contrast_tail_q20_minus_logprob | 0.214 [0.144, 0.277] p=0.000 | 0.217 [0.139, 0.288] p=0.000 |
| 24 | probe_outputs_plus_rmd_high_entropy_q20_minus_probe_outputs | 0.020 [-0.007, 0.047] p=0.170 | 0.043 [-0.001, 0.081] p=0.060 |
| 24 | contrast_tail_q20_minus_rmd | 0.185 [0.121, 0.247] p=0.000 | 0.221 [0.138, 0.286] p=0.000 |
| 24 | rmd_random_q20_minus_rmd | 0.000 [-0.011, 0.011] p=0.990 | 0.013 [-0.013, 0.039] p=0.430 |
| 24 | rmd_random_q20_minus_logprob | 0.029 [0.001, 0.059] p=0.040 | 0.009 [-0.036, 0.054] p=0.680 |

## Supervised hidden-state probe (exploratory)

LDA fit on PCA-projected region means, cross-fitted by prompt fold on pooled labels over parseable traces. This bounds how much of the geometry signal supervision on the same activations recovers. **Post-hoc, added 2026-07-29 -- not part of the pre-registered contrast set.**

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 24 | probe_hidden_tail_q20_minus_rmd_tail_q20 | 0.062 [0.026, 0.103] p=0.000 | 0.061 [0.012, 0.119] p=0.020 |
| 24 | probe_hidden_tail_q20_minus_length | 0.170 [0.130, 0.225] p=0.000 | 0.193 [0.136, 0.275] p=0.000 |

Rank correlation of each score against trace length (parseable only). A scorer that merely rediscovers "long traces are wrong" shows |rho| near 1.

| Layer | Score | Spearman vs length | Pearson vs length | n |
|---:|:---|---:|---:|---:|
| 24 | probe_hidden_tail_q20 | 0.091 | 0.062 | 3771 |
| 24 | rmd | 0.790 | 0.787 | 3771 |
| 24 | rmd_tail_q20 | 0.599 | 0.639 | 3771 |
| 24 | rmd_high_entropy_q20 | 0.722 | 0.726 | 3771 |
| 24 | entropy | 0.643 | 0.622 | 3771 |
| 24 | logprob | 0.661 | 0.642 | 3771 |

## E2 same-token output autopsy

Fixed cross-fitted probes: B0=global outputs, B1=global plus same high-entropy-token outputs, G_he=B1 plus high-entropy RMD, and G_random=B1 plus matched random-20% RMD. Only the two pre-specified geometry contrasts are shown here.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 24 | probe_g_he_minus_probe_b1 | 0.021 [-0.006, 0.042] p=0.120 | 0.029 [-0.008, 0.064] p=0.150 |
| 24 | probe_g_he_minus_probe_g_random | 0.014 [0.001, 0.028] p=0.030 | 0.023 [-0.008, 0.056] p=0.190 |

## Prompt-contrastive direction diagnostics

Directions are fit out-of-fold from parseable mixed training prompts. Each prompt contributes one normalized difference vector; alignment nulls shuffle labels within prompts while preserving class counts.

| Layer | Region | Prompt vectors | Observed alignment | Pairwise cosine | Null mean | Null 95% interval | p |
|---:|:---|---:|---:|---:|---:|:---|---:|
| 24 | tail_q20 | 129 | 0.264 | 0.062 | NA | [NA, NA] | NA |
| 24 | tail_q20 | 124 | 0.248 | 0.054 | NA | [NA, NA] | NA |
| 24 | tail_q20 | 133 | 0.285 | 0.074 | NA | [NA, NA] | NA |
| 24 | tail_q20 | 119 | 0.262 | 0.061 | NA | [NA, NA] | NA |
| 24 | tail_q20 | 127 | 0.271 | 0.066 | NA | [NA, NA] | NA |

## Parseable paired contrasts

Contrastive score minus baseline, using prompt-cluster bootstrap intervals.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 24 | rmd_high_entropy_q20_minus_rmd | 0.026 [0.007, 0.044] p=0.000 | 0.056 [0.024, 0.086] p=0.000 |
| 24 | rmd_tail_q20_minus_rmd | 0.066 [0.037, 0.093] p=0.000 | 0.111 [0.072, 0.150] p=0.000 |
| 24 | contrast_tail_q20_minus_rmd_tail_q20 | 0.118 [0.068, 0.166] p=0.000 | 0.109 [0.049, 0.159] p=0.000 |
| 24 | rmd_high_entropy_q20_minus_logprob | 0.055 [0.028, 0.086] p=0.000 | 0.052 [0.009, 0.103] p=0.050 |
| 24 | rmd_tail_q20_minus_logprob | 0.095 [0.055, 0.143] p=0.000 | 0.107 [0.051, 0.168] p=0.000 |
| 24 | contrast_tail_q20_minus_logprob | 0.214 [0.144, 0.277] p=0.000 | 0.217 [0.139, 0.288] p=0.000 |
| 24 | contrast_tail_q20_minus_rmd | 0.185 [0.121, 0.247] p=0.000 | 0.221 [0.138, 0.286] p=0.000 |
| 24 | rmd_random_q20_minus_rmd | 0.000 [-0.011, 0.011] p=0.990 | 0.013 [-0.013, 0.039] p=0.430 |
| 24 | rmd_random_q20_minus_logprob | 0.029 [0.001, 0.059] p=0.040 | 0.009 [-0.036, 0.054] p=0.680 |

No layer was selected after observing these results.

Confidence intervals use a prompt-cluster bootstrap over fixed out-of-fold predictions; reference fitting is not repeated.
