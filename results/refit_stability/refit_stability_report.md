# Full-refit stability

Built by `controls/refit_stability.py`. Each refit re-runs the pipeline end to end on a different prompt partition: the OOF scores are regenerated, the prompt-level readouts refitted, and the last-token probe refitted including its in-fold layer and penalty choice, and the peer ladder refitted across all models at that seed.

Refits collected: 4 (seeds 42, 101, 202, 303) over 3 of 3 registered models (qwen, deepseek, deepseek_llama).
Seeds carrying every quantity requested of *this* run: 42, 101, 202, 303; incomplete: none. That is completeness relative to this invocation, not to the registered protocol.

**Registered protocol: complete.**

Seed 42 is the frozen-partition reproduction check when it appears among the collected refits.

The quantity is the **spread of point estimates across refits**. The bootstrap intervals inside any single refit cannot see it, which is why the review says more draws are not a substitute.

## 1. Per-refit values

| Seed | Model | `B1 - B0` AURC | Peer residual AURC | Probe pooled | Probe macro | Probe pooled - macro | Layers chosen |
|---:|:--|---:|---:|---:|---:|---:|:--|
| 42 | qwen | -0.0520 | 0.0561 | 0.9013 | 0.6444 | 0.2569 | 7, 21 |
| 42 | deepseek | -0.0284 | 0.0946 | 0.9139 | 0.5823 | 0.3316 | 14, 21 |
| 42 | deepseek_llama | -0.0469 | 0.0295 | 0.9032 | 0.7177 | 0.1855 | 16 |
| 101 | qwen | -0.0520 | 0.0621 | 0.8951 | 0.6209 | 0.2742 | 7, 21 |
| 101 | deepseek | -0.0240 | 0.0869 | 0.9015 | 0.6647 | 0.2368 | 7, 14 |
| 101 | deepseek_llama | -0.0354 | 0.0493 | 0.8936 | 0.7132 | 0.1804 | 8, 16 |
| 202 | qwen | -0.0593 | 0.0505 | 0.9039 | 0.6140 | 0.2899 | 7 |
| 202 | deepseek | -0.0343 | 0.0791 | 0.9201 | 0.6859 | 0.2342 | 7, 14 |
| 202 | deepseek_llama | -0.0294 | 0.0426 | 0.8958 | 0.6824 | 0.2135 | 16 |
| 303 | qwen | -0.0546 | 0.0648 | 0.9057 | 0.5960 | 0.3098 | 7 |
| 303 | deepseek | -0.0407 | 0.0930 | 0.9229 | 0.6567 | 0.2662 | 7, 14 |
| 303 | deepseek_llama | -0.0485 | 0.0338 | 0.9011 | 0.7014 | 0.1997 | 16 |

## 2. Stability across refits

`sign stable` is the review's decision rule: a quantity that changes sign across refits is demoted regardless of how tight its within-refit interval is.

| Model | Quantity | n | Mean | Min | Max | Spread | Sign stable | Max drift from frozen |
|:--|:--|---:|---:|---:|---:|---:|:--|---:|
| qwen | `b1_minus_b0_aurc` | 4 | -0.0545 | -0.0593 | -0.0520 | 0.0073 | yes | 0.0072 |
| qwen | `peer_residual_aurc` | 4 | 0.0584 | 0.0505 | 0.0648 | 0.0142 | yes | 0.0087 |
| qwen | `peer_residual_deployable_aurc` | 4 | 0.0500 | 0.0392 | 0.0660 | 0.0267 | yes | 0.0251 |
| qwen | `probe_pooled_minus_macro` | 4 | 0.2827 | 0.2569 | 0.3098 | 0.0529 | yes | 0.0529 |
| qwen | `probe_pooled` | 4 | 0.9015 | 0.8951 | 0.9057 | 0.0106 | yes | 0.0062 |
| qwen | `probe_macro` | 4 | 0.6188 | 0.5960 | 0.6444 | 0.0484 | yes | 0.0484 |
| qwen | `rmd_pooled_minus_macro` | 4 | 0.1932 | 0.1844 | 0.2085 | 0.0241 | yes | 0.0241 |
| deepseek | `b1_minus_b0_aurc` | 4 | -0.0318 | -0.0407 | -0.0240 | 0.0167 | yes | 0.0124 |
| deepseek | `peer_residual_aurc` | 4 | 0.0884 | 0.0791 | 0.0946 | 0.0155 | yes | 0.0155 |
| deepseek | `peer_residual_deployable_aurc` | 4 | 0.0133 | 0.0071 | 0.0169 | 0.0098 | yes | 0.0098 |
| deepseek | `probe_pooled_minus_macro` | 4 | 0.2672 | 0.2342 | 0.3316 | 0.0974 | yes | 0.0974 |
| deepseek | `probe_pooled` | 4 | 0.9146 | 0.9015 | 0.9229 | 0.0214 | yes | 0.0124 |
| deepseek | `probe_macro` | 4 | 0.6474 | 0.5823 | 0.6859 | 0.1035 | yes | 0.1035 |
| deepseek | `rmd_pooled_minus_macro` | 4 | 0.2159 | 0.2041 | 0.2230 | 0.0189 | yes | 0.0097 |
| deepseek_llama | `b1_minus_b0_aurc` | 4 | -0.0400 | -0.0485 | -0.0294 | 0.0191 | yes | 0.0176 |
| deepseek_llama | `peer_residual_aurc` | 4 | 0.0388 | 0.0295 | 0.0493 | 0.0198 | yes | 0.0198 |
| deepseek_llama | `peer_residual_deployable_aurc` | 4 | 0.0689 | 0.0569 | 0.0779 | 0.0209 | yes | 0.0209 |
| deepseek_llama | `probe_pooled_minus_macro` | 4 | 0.1947 | 0.1804 | 0.2135 | 0.0331 | yes | 0.0280 |
| deepseek_llama | `probe_pooled` | 4 | 0.8984 | 0.8936 | 0.9032 | 0.0096 | yes | 0.0096 |
| deepseek_llama | `probe_macro` | 4 | 0.7037 | 0.6824 | 0.7177 | 0.0353 | yes | 0.0353 |
| deepseek_llama | `rmd_pooled_minus_macro` | 4 | 0.1347 | 0.1309 | 0.1373 | 0.0064 | yes | 0.0052 |

## What this establishes

A quantity whose spread across refits is comparable to or larger than its bootstrap interval was being reported with the wrong uncertainty. A quantity that changes sign across refits does not survive, and the review's instruction for that case is to keep the original increment and demote the residual rather than to average the refits.

The refits share one thing that is not resampled: the collected traces. This measures stability of the fitting path, not of the data collection.
