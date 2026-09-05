# deepseek_llama math500 prompt decomposition

Data: 500 complete prompts with N=8; partial_data=false.

| Layer | Method | Pooled AUC | Centered AUC | Within macro | Within pair | ICC | Spearman difficulty |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 24 | entropy | 0.546 | 0.470 | 0.478 | 0.476 | 0.640 | 0.130 |
| 24 | logprob | 0.544 | 0.470 | 0.483 | 0.480 | 0.640 | 0.131 |
| 24 | length | 0.581 | 0.506 | 0.525 | 0.530 | 0.834 | 0.199 |
| 24 | activation_norm | 0.452 | 0.383 | 0.389 | 0.379 | 0.641 | -0.047 |
| 24 | centroid | 0.401 | 0.373 | 0.379 | 0.368 | 0.789 | -0.183 |
| 24 | raw | 0.428 | 0.383 | 0.388 | 0.382 | 0.780 | -0.122 |
| 24 | rmd | 0.678 | 0.536 | 0.523 | 0.531 | 0.895 | 0.391 |
| 24 | rmd_high_entropy_q20 | 0.675 | 0.558 | 0.566 | 0.572 | 0.882 | 0.380 |
| 24 | rmd_tail_q20 | 0.731 | 0.625 | 0.643 | 0.643 | 0.805 | 0.482 |
| 24 | rmd_random_q20 | 0.677 | 0.532 | 0.532 | 0.540 | 0.887 | 0.388 |
| 24 | entropy_he | 0.452 | 0.531 | 0.516 | 0.521 | 0.643 | -0.137 |
| 24 | logprob_he | 0.547 | 0.467 | 0.483 | 0.479 | 0.641 | 0.138 |
| 24 | prompt_local_rmd | 0.438 | 0.540 | 0.501 | 0.525 | 0.803 | -0.137 |
| 24 | contrast_tail_q20 | 0.581 | 0.688 | 0.720 | 0.725 | 0.690 | 0.079 |

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
| 24 | 0.098 [0.074, 0.126] p=0.000 | 0.030 [0.007, 0.054] p=0.010 | -0.003 [-0.037, 0.036] p=0.900 |

## Truncation / parseability diagnostic

Unparsed (no final answer): 229/4000 (0.057); length-capped at 12288: 250 (0.062); unparsed share of the incorrect class: 0.128.

Unparsed traces are auto-labeled incorrect upstream and are usually truncated, not wrong-answer. The within-prompt metrics below restrict to traces that emitted a parseable answer; a large drop in mixed-prompt count or in the RMD-minus-entropy gap means the headline within-prompt signal was a truncation detector.

| Layer | Method | Parseable within macro | Parseable centered AUC | Mixed prompts |
|---:|:---|---:|---:|---:|
| 24 | entropy | 0.446 | 0.435 | 158 |
| 24 | logprob | 0.447 | 0.436 | 158 |
| 24 | length | 0.422 | 0.423 | 158 |
| 24 | activation_norm | 0.379 | 0.372 | 158 |
| 24 | centroid | 0.379 | 0.362 | 158 |
| 24 | raw | 0.381 | 0.370 | 158 |
| 24 | rmd | 0.436 | 0.463 | 158 |
| 24 | rmd_high_entropy_q20 | 0.491 | 0.492 | 158 |
| 24 | rmd_tail_q20 | 0.556 | 0.535 | 158 |
| 24 | rmd_random_q20 | 0.448 | 0.461 | 158 |
| 24 | entropy_he | 0.560 | 0.568 | 158 |
| 24 | logprob_he | 0.441 | 0.432 | 158 |
| 24 | prompt_local_rmd | 0.511 | 0.545 | 158 |
| 24 | contrast_tail_q20 | 0.655 | 0.650 | 158 |
| 24 | probe_outputs | 0.561 | 0.577 | 158 |
| 24 | probe_outputs_plus_rmd_high_entropy_q20 | 0.618 | 0.603 | 158 |
| 24 | probe_b0 | 0.561 | 0.577 | 158 |
| 24 | probe_b1 | 0.556 | 0.574 | 158 |
| 24 | probe_g_he | 0.593 | 0.601 | 158 |
| 24 | probe_g_random | 0.558 | 0.578 | 158 |
| 24 | probe_hidden_tail_q20 | 0.625 | 0.606 | 158 |

## Prespecified parseable score contrasts

Point estimates, raw 95% prompt-bootstrap intervals, and raw two-sided p-values are reported without post-hoc layer selection or multiplicity-adjusted claims.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 24 | rmd_high_entropy_q20_minus_rmd | 0.029 [0.011, 0.046] p=0.000 | 0.055 [0.024, 0.089] p=0.000 |
| 24 | rmd_tail_q20_minus_rmd | 0.072 [0.046, 0.101] p=0.000 | 0.120 [0.086, 0.163] p=0.000 |
| 24 | contrast_tail_q20_minus_rmd_tail_q20 | 0.115 [0.068, 0.162] p=0.000 | 0.099 [0.043, 0.154] p=0.000 |
| 24 | rmd_high_entropy_q20_minus_rmd_random_q20 | 0.031 [0.009, 0.048] p=0.000 | 0.042 [-0.001, 0.080] p=0.060 |
| 24 | rmd_high_entropy_q20_minus_logprob | 0.056 [0.024, 0.084] p=0.000 | 0.044 [-0.003, 0.085] p=0.070 |
| 24 | rmd_tail_q20_minus_logprob | 0.099 [0.056, 0.144] p=0.000 | 0.109 [0.055, 0.178] p=0.000 |
| 24 | contrast_tail_q20_minus_logprob | 0.214 [0.158, 0.286] p=0.000 | 0.208 [0.134, 0.278] p=0.000 |
| 24 | probe_outputs_plus_rmd_high_entropy_q20_minus_probe_outputs | 0.027 [0.005, 0.045] p=0.030 | 0.057 [0.018, 0.093] p=0.010 |
| 24 | contrast_tail_q20_minus_rmd | 0.187 [0.130, 0.249] p=0.000 | 0.219 [0.157, 0.279] p=0.000 |
| 24 | rmd_random_q20_minus_rmd | -0.002 [-0.011, 0.010] p=0.720 | 0.013 [-0.008, 0.040] p=0.260 |
| 24 | rmd_random_q20_minus_logprob | 0.025 [-0.006, 0.054] p=0.150 | 0.001 [-0.048, 0.051] p=0.890 |

## Supervised hidden-state probe (exploratory)

LDA fit on PCA-projected region means, cross-fitted by prompt fold on pooled labels over parseable traces. This bounds how much of the geometry signal supervision on the same activations recovers. **Post-hoc, added 2026-07-29 -- not part of the pre-registered contrast set.**

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 24 | probe_hidden_tail_q20_minus_rmd_tail_q20 | 0.071 [0.030, 0.112] p=0.000 | 0.069 [0.016, 0.124] p=0.010 |
| 24 | probe_hidden_tail_q20_minus_length | 0.184 [0.138, 0.234] p=0.000 | 0.203 [0.140, 0.264] p=0.000 |

Rank correlation of each score against trace length (parseable only). A scorer that merely rediscovers "long traces are wrong" shows |rho| near 1.

| Layer | Score | Spearman vs length | Pearson vs length | n |
|---:|:---|---:|---:|---:|
| 24 | probe_hidden_tail_q20 | 0.080 | 0.054 | 3771 |
| 24 | rmd | 0.784 | 0.787 | 3771 |
| 24 | rmd_tail_q20 | 0.576 | 0.622 | 3771 |
| 24 | rmd_high_entropy_q20 | 0.723 | 0.733 | 3771 |
| 24 | entropy | 0.643 | 0.622 | 3771 |
| 24 | logprob | 0.661 | 0.642 | 3771 |

## E2 same-token output autopsy

Fixed cross-fitted probes: B0=global outputs, B1=global plus same high-entropy-token outputs, G_he=B1 plus high-entropy RMD, and G_random=B1 plus matched random-20% RMD. Only the two pre-specified geometry contrasts are shown here.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 24 | probe_g_he_minus_probe_b1 | 0.027 [0.007, 0.047] p=0.010 | 0.037 [0.005, 0.085] p=0.040 |
| 24 | probe_g_he_minus_probe_g_random | 0.022 [0.010, 0.039] p=0.000 | 0.035 [0.009, 0.066] p=0.010 |

## Prompt-contrastive direction diagnostics

Directions are fit out-of-fold from parseable mixed training prompts. Each prompt contributes one normalized difference vector; alignment nulls shuffle labels within prompts while preserving class counts.

| Layer | Region | Prompt vectors | Observed alignment | Pairwise cosine | Null mean | Null 95% interval | p |
|---:|:---|---:|---:|---:|---:|:---|---:|
| 24 | tail_q20 | 132 | 0.255 | 0.058 | NA | [NA, NA] | NA |
| 24 | tail_q20 | 130 | 0.259 | 0.060 | NA | [NA, NA] | NA |
| 24 | tail_q20 | 125 | 0.282 | 0.072 | NA | [NA, NA] | NA |
| 24 | tail_q20 | 121 | 0.262 | 0.061 | NA | [NA, NA] | NA |
| 24 | tail_q20 | 124 | 0.272 | 0.066 | NA | [NA, NA] | NA |

## Parseable paired contrasts

Contrastive score minus baseline, using prompt-cluster bootstrap intervals.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 24 | rmd_high_entropy_q20_minus_rmd | 0.029 [0.011, 0.046] p=0.000 | 0.055 [0.024, 0.089] p=0.000 |
| 24 | rmd_tail_q20_minus_rmd | 0.072 [0.046, 0.101] p=0.000 | 0.120 [0.086, 0.163] p=0.000 |
| 24 | contrast_tail_q20_minus_rmd_tail_q20 | 0.115 [0.068, 0.162] p=0.000 | 0.099 [0.043, 0.154] p=0.000 |
| 24 | rmd_high_entropy_q20_minus_logprob | 0.056 [0.024, 0.084] p=0.000 | 0.044 [-0.003, 0.085] p=0.070 |
| 24 | rmd_tail_q20_minus_logprob | 0.099 [0.056, 0.144] p=0.000 | 0.109 [0.055, 0.178] p=0.000 |
| 24 | contrast_tail_q20_minus_logprob | 0.214 [0.158, 0.286] p=0.000 | 0.208 [0.134, 0.278] p=0.000 |
| 24 | contrast_tail_q20_minus_rmd | 0.187 [0.130, 0.249] p=0.000 | 0.219 [0.157, 0.279] p=0.000 |
| 24 | rmd_random_q20_minus_rmd | -0.002 [-0.011, 0.010] p=0.720 | 0.013 [-0.008, 0.040] p=0.260 |
| 24 | rmd_random_q20_minus_logprob | 0.025 [-0.006, 0.054] p=0.150 | 0.001 [-0.048, 0.051] p=0.890 |

No layer was selected after observing these results.

Confidence intervals use a prompt-cluster bootstrap over fixed out-of-fold predictions; reference fitting is not repeated.
