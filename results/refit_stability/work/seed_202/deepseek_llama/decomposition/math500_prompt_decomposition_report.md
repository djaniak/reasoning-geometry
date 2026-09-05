# deepseek_llama math500 prompt decomposition

Data: 500 complete prompts with N=8; partial_data=false.

| Layer | Method | Pooled AUC | Centered AUC | Within macro | Within pair | ICC | Spearman difficulty |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 24 | entropy | 0.546 | 0.470 | 0.478 | 0.476 | 0.640 | 0.130 |
| 24 | logprob | 0.544 | 0.470 | 0.483 | 0.480 | 0.640 | 0.131 |
| 24 | length | 0.581 | 0.506 | 0.525 | 0.530 | 0.834 | 0.199 |
| 24 | activation_norm | 0.452 | 0.383 | 0.389 | 0.379 | 0.641 | -0.047 |
| 24 | centroid | 0.399 | 0.373 | 0.379 | 0.369 | 0.789 | -0.186 |
| 24 | raw | 0.426 | 0.385 | 0.388 | 0.381 | 0.780 | -0.128 |
| 24 | rmd | 0.672 | 0.539 | 0.525 | 0.532 | 0.900 | 0.371 |
| 24 | rmd_high_entropy_q20 | 0.673 | 0.560 | 0.570 | 0.574 | 0.886 | 0.369 |
| 24 | rmd_tail_q20 | 0.721 | 0.624 | 0.638 | 0.636 | 0.811 | 0.452 |
| 24 | rmd_random_q20 | 0.671 | 0.536 | 0.535 | 0.542 | 0.892 | 0.370 |
| 24 | entropy_he | 0.452 | 0.531 | 0.516 | 0.521 | 0.643 | -0.137 |
| 24 | logprob_he | 0.547 | 0.467 | 0.483 | 0.479 | 0.641 | 0.138 |
| 24 | prompt_local_rmd | 0.435 | 0.539 | 0.503 | 0.527 | 0.802 | -0.141 |
| 24 | contrast_tail_q20 | 0.582 | 0.689 | 0.723 | 0.729 | 0.689 | 0.081 |

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
| 24 | 0.091 [0.066, 0.118] p=0.000 | 0.033 [0.013, 0.052] p=0.000 | -0.001 [-0.035, 0.033] p=0.970 |

## Truncation / parseability diagnostic

Unparsed (no final answer): 229/4000 (0.057); length-capped at 12288: 250 (0.062); unparsed share of the incorrect class: 0.128.

Unparsed traces are auto-labeled incorrect upstream and are usually truncated, not wrong-answer. The within-prompt metrics below restrict to traces that emitted a parseable answer; a large drop in mixed-prompt count or in the RMD-minus-entropy gap means the headline within-prompt signal was a truncation detector.

| Layer | Method | Parseable within macro | Parseable centered AUC | Mixed prompts |
|---:|:---|---:|---:|---:|
| 24 | entropy | 0.446 | 0.435 | 158 |
| 24 | logprob | 0.447 | 0.436 | 158 |
| 24 | length | 0.422 | 0.423 | 158 |
| 24 | activation_norm | 0.379 | 0.372 | 158 |
| 24 | centroid | 0.379 | 0.363 | 158 |
| 24 | raw | 0.379 | 0.371 | 158 |
| 24 | rmd | 0.442 | 0.467 | 158 |
| 24 | rmd_high_entropy_q20 | 0.500 | 0.494 | 158 |
| 24 | rmd_tail_q20 | 0.550 | 0.535 | 158 |
| 24 | rmd_random_q20 | 0.456 | 0.467 | 158 |
| 24 | entropy_he | 0.560 | 0.568 | 158 |
| 24 | logprob_he | 0.441 | 0.432 | 158 |
| 24 | prompt_local_rmd | 0.514 | 0.545 | 158 |
| 24 | contrast_tail_q20 | 0.660 | 0.652 | 158 |
| 24 | probe_outputs | 0.567 | 0.577 | 158 |
| 24 | probe_outputs_plus_rmd_high_entropy_q20 | 0.624 | 0.603 | 158 |
| 24 | probe_b0 | 0.567 | 0.577 | 158 |
| 24 | probe_b1 | 0.562 | 0.576 | 158 |
| 24 | probe_g_he | 0.600 | 0.603 | 158 |
| 24 | probe_g_random | 0.580 | 0.585 | 158 |
| 24 | probe_hidden_tail_q20 | 0.624 | 0.607 | 158 |

## Prespecified parseable score contrasts

Point estimates, raw 95% prompt-bootstrap intervals, and raw two-sided p-values are reported without post-hoc layer selection or multiplicity-adjusted claims.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 24 | rmd_high_entropy_q20_minus_rmd | 0.026 [0.009, 0.045] p=0.000 | 0.058 [0.027, 0.086] p=0.000 |
| 24 | rmd_tail_q20_minus_rmd | 0.068 [0.040, 0.090] p=0.000 | 0.109 [0.068, 0.144] p=0.000 |
| 24 | contrast_tail_q20_minus_rmd_tail_q20 | 0.116 [0.059, 0.164] p=0.000 | 0.109 [0.039, 0.158] p=0.000 |
| 24 | rmd_high_entropy_q20_minus_rmd_random_q20 | 0.027 [0.005, 0.047] p=0.010 | 0.044 [0.003, 0.080] p=0.030 |
| 24 | rmd_high_entropy_q20_minus_logprob | 0.058 [0.030, 0.084] p=0.000 | 0.053 [0.003, 0.091] p=0.030 |
| 24 | rmd_tail_q20_minus_logprob | 0.099 [0.055, 0.133] p=0.000 | 0.103 [0.028, 0.157] p=0.000 |
| 24 | contrast_tail_q20_minus_logprob | 0.216 [0.153, 0.271] p=0.000 | 0.213 [0.130, 0.266] p=0.000 |
| 24 | probe_outputs_plus_rmd_high_entropy_q20_minus_probe_outputs | 0.026 [-0.000, 0.052] p=0.060 | 0.057 [0.010, 0.100] p=0.010 |
| 24 | contrast_tail_q20_minus_rmd | 0.184 [0.116, 0.237] p=0.000 | 0.218 [0.154, 0.275] p=0.000 |
| 24 | rmd_random_q20_minus_rmd | -0.000 [-0.011, 0.010] p=0.910 | 0.014 [-0.008, 0.039] p=0.250 |
| 24 | rmd_random_q20_minus_logprob | 0.031 [0.001, 0.058] p=0.050 | 0.009 [-0.045, 0.058] p=0.730 |

## Supervised hidden-state probe (exploratory)

LDA fit on PCA-projected region means, cross-fitted by prompt fold on pooled labels over parseable traces. This bounds how much of the geometry signal supervision on the same activations recovers. **Post-hoc, added 2026-07-29 -- not part of the pre-registered contrast set.**

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 24 | probe_hidden_tail_q20_minus_rmd_tail_q20 | 0.072 [0.034, 0.111] p=0.000 | 0.073 [0.024, 0.128] p=0.010 |
| 24 | probe_hidden_tail_q20_minus_length | 0.184 [0.137, 0.242] p=0.000 | 0.202 [0.140, 0.275] p=0.000 |

Rank correlation of each score against trace length (parseable only). A scorer that merely rediscovers "long traces are wrong" shows |rho| near 1.

| Layer | Score | Spearman vs length | Pearson vs length | n |
|---:|:---|---:|---:|---:|
| 24 | probe_hidden_tail_q20 | 0.085 | 0.059 | 3771 |
| 24 | rmd | 0.772 | 0.775 | 3771 |
| 24 | rmd_tail_q20 | 0.576 | 0.618 | 3771 |
| 24 | rmd_high_entropy_q20 | 0.709 | 0.718 | 3771 |
| 24 | entropy | 0.643 | 0.622 | 3771 |
| 24 | logprob | 0.661 | 0.642 | 3771 |

## E2 same-token output autopsy

Fixed cross-fitted probes: B0=global outputs, B1=global plus same high-entropy-token outputs, G_he=B1 plus high-entropy RMD, and G_random=B1 plus matched random-20% RMD. Only the two pre-specified geometry contrasts are shown here.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 24 | probe_g_he_minus_probe_b1 | 0.027 [0.000, 0.049] p=0.050 | 0.038 [-0.009, 0.087] p=0.080 |
| 24 | probe_g_he_minus_probe_g_random | 0.018 [-0.001, 0.033] p=0.060 | 0.020 [-0.019, 0.055] p=0.320 |

## Prompt-contrastive direction diagnostics

Directions are fit out-of-fold from parseable mixed training prompts. Each prompt contributes one normalized difference vector; alignment nulls shuffle labels within prompts while preserving class counts.

| Layer | Region | Prompt vectors | Observed alignment | Pairwise cosine | Null mean | Null 95% interval | p |
|---:|:---|---:|---:|---:|---:|:---|---:|
| 24 | tail_q20 | 124 | 0.245 | 0.052 | NA | [NA, NA] | NA |
| 24 | tail_q20 | 125 | 0.285 | 0.074 | NA | [NA, NA] | NA |
| 24 | tail_q20 | 128 | 0.269 | 0.065 | NA | [NA, NA] | NA |
| 24 | tail_q20 | 125 | 0.264 | 0.062 | NA | [NA, NA] | NA |
| 24 | tail_q20 | 130 | 0.266 | 0.064 | NA | [NA, NA] | NA |

## Parseable paired contrasts

Contrastive score minus baseline, using prompt-cluster bootstrap intervals.

| Layer | Contrast | Centered AUC delta | Within macro delta |
|---:|:---|:---|:---|
| 24 | rmd_high_entropy_q20_minus_rmd | 0.026 [0.009, 0.045] p=0.000 | 0.058 [0.027, 0.086] p=0.000 |
| 24 | rmd_tail_q20_minus_rmd | 0.068 [0.040, 0.090] p=0.000 | 0.109 [0.068, 0.144] p=0.000 |
| 24 | contrast_tail_q20_minus_rmd_tail_q20 | 0.116 [0.059, 0.164] p=0.000 | 0.109 [0.039, 0.158] p=0.000 |
| 24 | rmd_high_entropy_q20_minus_logprob | 0.058 [0.030, 0.084] p=0.000 | 0.053 [0.003, 0.091] p=0.030 |
| 24 | rmd_tail_q20_minus_logprob | 0.099 [0.055, 0.133] p=0.000 | 0.103 [0.028, 0.157] p=0.000 |
| 24 | contrast_tail_q20_minus_logprob | 0.216 [0.153, 0.271] p=0.000 | 0.213 [0.130, 0.266] p=0.000 |
| 24 | contrast_tail_q20_minus_rmd | 0.184 [0.116, 0.237] p=0.000 | 0.218 [0.154, 0.275] p=0.000 |
| 24 | rmd_random_q20_minus_rmd | -0.000 [-0.011, 0.010] p=0.910 | 0.014 [-0.008, 0.039] p=0.250 |
| 24 | rmd_random_q20_minus_logprob | 0.031 [0.001, 0.058] p=0.050 | 0.009 [-0.045, 0.058] p=0.730 |

No layer was selected after observing these results.

Confidence intervals use a prompt-cluster bootstrap over fixed out-of-fold predictions; reference fitting is not repeated.
