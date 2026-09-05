# Full-refit stability

Built by `controls/refit_stability.py`. Each refit re-runs the pipeline end to end on a different prompt partition: the OOF scores are regenerated, the prompt-level readouts refitted, the last-token probe refitted including its in-fold layer and penalty choice, and the peer ladder refitted across all models at that seed.

Complete refits collected: 3 (seeds 42, 101, 202). Incomplete seeds: none.
Seed 42 is the frozen-partition reproduction check when it appears among the complete refits.

The quantity is the **spread of point estimates across refits**. The bootstrap intervals inside any single refit cannot see it, which is why the review says more draws are not a substitute.

## 1. Per-refit values

| Seed | Model | `B1 - B0` AURC | Peer residual AURC | Probe pooled | Probe macro | Probe pooled - macro | Layers chosen |
|---:|:--|---:|---:|---:|---:|---:|:--|
| 42 | deepseek | -0.0284 | -- | 0.9139 | 0.5823 | 0.3316 | 14, 21 |
| 42 | deepseek_llama | -0.0469 | -- | 0.9032 | 0.7177 | 0.1855 | 16 |
| 101 | deepseek | -0.0240 | -- | 0.9015 | 0.6647 | 0.2368 | 7, 14 |
| 101 | deepseek_llama | -0.0354 | -- | 0.8936 | 0.7132 | 0.1804 | 8, 16 |
| 202 | deepseek | -0.0343 | -- | 0.9201 | 0.6859 | 0.2342 | 7, 14 |
| 202 | deepseek_llama | -0.0294 | -- | 0.8958 | 0.6824 | 0.2135 | 16 |

## 2. Stability across refits

`sign stable` is the review's decision rule: a quantity that changes sign across refits is demoted regardless of how tight its within-refit interval is.

| Model | Quantity | n | Mean | Min | Max | Spread | Sign stable | Max drift from frozen |
|:--|:--|---:|---:|---:|---:|---:|:--|---:|
| deepseek | `b1_minus_b0_aurc` | 3 | -0.0289 | -0.0343 | -0.0240 | 0.0103 | yes | 0.0059 |
| deepseek | `peer_residual_aurc` | 0 | -- | -- | -- | -- | -- | -- |
| deepseek | `peer_residual_deployable_aurc` | 0 | -- | -- | -- | -- | -- | -- |
| deepseek | `probe_pooled_minus_macro` | 3 | 0.2675 | 0.2342 | 0.3316 | 0.0974 | yes | 0.0974 |
| deepseek | `probe_pooled` | 3 | 0.9118 | 0.9015 | 0.9201 | 0.0186 | yes | 0.0124 |
| deepseek | `probe_macro` | 3 | 0.6443 | 0.5823 | 0.6859 | 0.1035 | yes | 0.1035 |
| deepseek | `rmd_pooled_minus_macro` | 3 | 0.2137 | 0.2041 | 0.2230 | 0.0189 | yes | 0.0097 |
| deepseek_llama | `b1_minus_b0_aurc` | 3 | -0.0372 | -0.0469 | -0.0294 | 0.0176 | yes | 0.0176 |
| deepseek_llama | `peer_residual_aurc` | 0 | -- | -- | -- | -- | -- | -- |
| deepseek_llama | `peer_residual_deployable_aurc` | 0 | -- | -- | -- | -- | -- | -- |
| deepseek_llama | `probe_pooled_minus_macro` | 3 | 0.1931 | 0.1804 | 0.2135 | 0.0331 | yes | 0.0280 |
| deepseek_llama | `probe_pooled` | 3 | 0.8975 | 0.8936 | 0.9032 | 0.0096 | yes | 0.0096 |
| deepseek_llama | `probe_macro` | 3 | 0.7044 | 0.6824 | 0.7177 | 0.0353 | yes | 0.0353 |
| deepseek_llama | `rmd_pooled_minus_macro` | 3 | 0.1348 | 0.1309 | 0.1373 | 0.0064 | yes | 0.0052 |

## What this establishes

A quantity whose spread across refits is comparable to or larger than its bootstrap interval was being reported with the wrong uncertainty. A quantity that changes sign across refits does not survive, and the review's instruction for that case is to keep the original increment and demote the residual rather than to average the refits.

The refits share one thing that is not resampled: the collected traces. This measures stability of the fitting path, not of the data collection.
