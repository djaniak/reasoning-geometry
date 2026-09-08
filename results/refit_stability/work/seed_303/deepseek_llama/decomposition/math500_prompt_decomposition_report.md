# deepseek_llama math500 prompt decomposition

Data: 500 complete prompts with N=8; partial_data=false.

| Layer | Method | Pooled AUC | Centered AUC | Within macro | Within pair | ICC | Spearman difficulty |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 24 | entropy | 0.546 | 0.470 | 0.478 | 0.476 | 0.640 | 0.130 |
| 24 | logprob | 0.544 | 0.470 | 0.483 | 0.480 | 0.640 | 0.131 |
| 24 | length | 0.581 | 0.506 | 0.525 | 0.530 | 0.834 | 0.199 |
| 24 | activation_norm | 0.452 | 0.383 | 0.389 | 0.379 | 0.641 | -0.047 |
| 24 | centroid | 0.399 | 0.373 | 0.375 | 0.362 | 0.789 | -0.186 |
| 24 | raw | 0.427 | 0.385 | 0.394 | 0.385 | 0.780 | -0.125 |
| 24 | rmd | 0.678 | 0.544 | 0.539 | 0.545 | 0.897 | 0.386 |
| 24 | rmd_high_entropy_q20 | 0.676 | 0.562 | 0.576 | 0.579 | 0.886 | 0.379 |
| 24 | rmd_tail_q20 | 0.733 | 0.627 | 0.648 | 0.645 | 0.802 | 0.479 |
| 24 | rmd_random_q20 | 0.677 | 0.541 | 0.544 | 0.549 | 0.890 | 0.385 |
| 24 | entropy_he | 0.452 | 0.531 | 0.516 | 0.521 | 0.643 | -0.137 |
| 24 | logprob_he | 0.547 | 0.467 | 0.483 | 0.479 | 0.641 | 0.138 |
| 24 | prompt_local_rmd | 0.436 | 0.541 | 0.510 | 0.532 | 0.801 | -0.139 |
| 24 | contrast_tail_q20 | 0.577 | 0.685 | 0.725 | 0.732 | 0.690 | 0.078 |

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
| 24 | 0.097 [0.078, 0.123] p=0.000 | 0.039 [0.016, 0.063] p=0.000 | 0.014 [-0.027, 0.046] p=0.540 |

## Truncation / parseability diagnostic

Unparsed (no final answer): 229/4000 (0.057); length-capped at 12288: 250 (0.062); unparsed share of the incorrect class: 0.128.

Unparsed traces are auto-labeled incorrect upstream and are usually truncated, not wrong-answer. The within-prompt metrics below restrict to traces that emitted a parseable answer; a large drop in mixed-prompt count or in the RMD-minus-entropy gap means the headline within-prompt signal was a truncation detector.

| Layer | Method | Parseable within macro | Parseable centered AUC | Mixed prompts |
|---:|:---|---:|---:|---:|
| 24 | entropy | 0.446 | 0.435 | 158 |
| 24 | logprob | 0.447 | 0.436 | 158 |
| 24 | length | 0.422 | 0.423 | 158 |
| 24 | activation_norm | 0.379 | 0.372 | 158 |
| 24 | centroid | 0.376 | 0.363 | 158 |
| 24 | raw | 0.384 | 0.371 | 158 |
| 24 | rmd | 0.452 | 0.472 | 158 |
| 24 | rmd_high_entropy_q20 | 0.504 | 0.497 | 158 |
| 24 | rmd_tail_q20 | 0.561 | 0.540 | 158 |
| 24 | rmd_random_q20 | 0.458 | 0.470 | 158 |
| 24 | entropy_he | 0.560 | 0.568 | 158 |
| 24 | logprob_he | 0.441 | 0.432 | 158 |
| 24 | prompt_local_rmd | 0.517 | 0.545 | 158 |
| 24 | contrast_tail_q20 | 0.658 | 0.646 | 158 |
| 24 | probe_outputs | 0.565 | 0.573 | 158 |
| 24 | probe_outputs_plus_rmd_high_entropy_q20 | 0.612 | 0.594 | 158 |
| 24 | probe_b0 | 0.565 | 0.573 | 158 |
| 24 | probe_b1 | 0.556 | 0.571 | 158 |
| 24 | probe_g_he | 0.597 | 0.594 | 158 |
| 24 | probe_g_random | 0.560 | 0.576 | 158 |
| 24 | probe_hidden_tail_q20 | 0.632 | 0.605 | 158 |

## Prespecified parseable score contrasts

Point estimates, raw 95% prompt-bootstrap intervals, and raw two-sided p-values are reported without post-hoc layer selection or multiplicity-adjusted claims.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 24 | rmd_high_entropy_q20_minus_rmd | 0.025 [0.005, 0.046] p=0.010 | 0.051 [0.015, 0.089] p=0.000 |
| 24 | rmd_tail_q20_minus_rmd | 0.068 [0.044, 0.095] p=0.000 | 0.109 [0.073, 0.150] p=0.000 |
| 24 | contrast_tail_q20_minus_rmd_tail_q20 | 0.106 [0.056, 0.162] p=0.000 | 0.097 [0.044, 0.164] p=0.000 |
| 24 | rmd_high_entropy_q20_minus_rmd_random_q20 | 0.027 [0.004, 0.049] p=0.000 | 0.045 [0.001, 0.095] p=0.050 |
| 24 | rmd_high_entropy_q20_minus_logprob | 0.061 [0.030, 0.100] p=0.000 | 0.056 [0.009, 0.107] p=0.010 |
| 24 | rmd_tail_q20_minus_logprob | 0.104 [0.067, 0.150] p=0.000 | 0.114 [0.061, 0.175] p=0.000 |
| 24 | contrast_tail_q20_minus_logprob | 0.210 [0.153, 0.280] p=0.000 | 0.211 [0.148, 0.281] p=0.000 |
| 24 | probe_outputs_plus_rmd_high_entropy_q20_minus_probe_outputs | 0.021 [-0.009, 0.052] p=0.180 | 0.046 [0.002, 0.085] p=0.040 |
| 24 | contrast_tail_q20_minus_rmd | 0.174 [0.112, 0.242] p=0.000 | 0.206 [0.146, 0.275] p=0.000 |
| 24 | rmd_random_q20_minus_rmd | -0.002 [-0.013, 0.011] p=0.770 | 0.006 [-0.021, 0.032] p=0.660 |
| 24 | rmd_random_q20_minus_logprob | 0.034 [0.004, 0.067] p=0.030 | 0.011 [-0.041, 0.063] p=0.690 |

## Supervised hidden-state probe (exploratory)

LDA fit on PCA-projected region means, cross-fitted by prompt fold on pooled labels over parseable traces. This bounds how much of the geometry signal supervision on the same activations recovers. **Post-hoc, added 2026-07-29 -- not part of the pre-registered contrast set.**

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 24 | probe_hidden_tail_q20_minus_rmd_tail_q20 | 0.065 [0.027, 0.102] p=0.000 | 0.071 [0.022, 0.117] p=0.010 |
| 24 | probe_hidden_tail_q20_minus_length | 0.183 [0.129, 0.238] p=0.000 | 0.210 [0.144, 0.271] p=0.000 |

Rank correlation of each score against trace length (parseable only). A scorer that merely rediscovers "long traces are wrong" shows |rho| near 1.

| Layer | Score | Spearman vs length | Pearson vs length | n |
|---:|:---|---:|---:|---:|
| 24 | probe_hidden_tail_q20 | 0.080 | 0.048 | 3771 |
| 24 | rmd | 0.777 | 0.779 | 3771 |
| 24 | rmd_tail_q20 | 0.594 | 0.636 | 3771 |
| 24 | rmd_high_entropy_q20 | 0.694 | 0.705 | 3771 |
| 24 | entropy | 0.643 | 0.622 | 3771 |
| 24 | logprob | 0.661 | 0.642 | 3771 |

## E2 same-token output autopsy

Fixed cross-fitted probes: B0=global outputs, B1=global plus same high-entropy-token outputs, G_he=B1 plus high-entropy RMD, and G_random=B1 plus matched random-20% RMD. Only the two pre-specified geometry contrasts are shown here.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 24 | probe_g_he_minus_probe_b1 | 0.023 [-0.010, 0.052] p=0.140 | 0.042 [-0.002, 0.084] p=0.090 |
| 24 | probe_g_he_minus_probe_g_random | 0.018 [-0.001, 0.035] p=0.060 | 0.037 [0.007, 0.070] p=0.020 |

## Prompt-contrastive direction diagnostics

Directions are fit out-of-fold from parseable mixed training prompts. Each prompt contributes one normalized difference vector; alignment nulls shuffle labels within prompts while preserving class counts.

| Layer | Region | Prompt vectors | Observed alignment | Pairwise cosine | Null mean | Null 95% interval | p |
|---:|:---|---:|---:|---:|---:|:---|---:|
| 24 | tail_q20 | 130 | 0.260 | 0.060 | NA | [NA, NA] | NA |
| 24 | tail_q20 | 127 | 0.257 | 0.059 | NA | [NA, NA] | NA |
| 24 | tail_q20 | 125 | 0.252 | 0.056 | NA | [NA, NA] | NA |
| 24 | tail_q20 | 122 | 0.295 | 0.079 | NA | [NA, NA] | NA |
| 24 | tail_q20 | 128 | 0.272 | 0.067 | NA | [NA, NA] | NA |

## Parseable paired contrasts

Contrastive score minus baseline, using prompt-cluster bootstrap intervals.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 24 | rmd_high_entropy_q20_minus_rmd | 0.025 [0.005, 0.046] p=0.010 | 0.051 [0.015, 0.089] p=0.000 |
| 24 | rmd_tail_q20_minus_rmd | 0.068 [0.044, 0.095] p=0.000 | 0.109 [0.073, 0.150] p=0.000 |
| 24 | contrast_tail_q20_minus_rmd_tail_q20 | 0.106 [0.056, 0.162] p=0.000 | 0.097 [0.044, 0.164] p=0.000 |
| 24 | rmd_high_entropy_q20_minus_logprob | 0.061 [0.030, 0.100] p=0.000 | 0.056 [0.009, 0.107] p=0.010 |
| 24 | rmd_tail_q20_minus_logprob | 0.104 [0.067, 0.150] p=0.000 | 0.114 [0.061, 0.175] p=0.000 |
| 24 | contrast_tail_q20_minus_logprob | 0.210 [0.153, 0.280] p=0.000 | 0.211 [0.148, 0.281] p=0.000 |
| 24 | contrast_tail_q20_minus_rmd | 0.174 [0.112, 0.242] p=0.000 | 0.206 [0.146, 0.275] p=0.000 |
| 24 | rmd_random_q20_minus_rmd | -0.002 [-0.013, 0.011] p=0.770 | 0.006 [-0.021, 0.032] p=0.660 |
| 24 | rmd_random_q20_minus_logprob | 0.034 [0.004, 0.067] p=0.030 | 0.011 [-0.041, 0.063] p=0.690 |

No layer was selected after observing these results.

Confidence intervals use a prompt-cluster bootstrap over fixed out-of-fold predictions; reference fitting is not repeated.
