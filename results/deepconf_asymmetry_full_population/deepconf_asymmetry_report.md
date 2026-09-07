# DeepConf asymmetry between DeepSeek-Qwen and Llama

Population: `full_population`. Seed 42, 1000 bootstrap draws.

## Base rate first

| model | n | base accuracy |
|---|---:|---:|
| deepseek_qwen | 500 | 0.7500 |
| llama | 500 | 0.6340 |

## Fitted readouts, as excess over the base rate

| readout | deepseek_qwen AUACC | deepseek_qwen excess | llama AUACC | llama excess |
|---|---|---|---|---|
| `B0` | 0.8337 | +0.0837 | 0.7500 | +0.1160 |
| `B1` | 0.8621 | +0.1121 | 0.7970 | +0.1630 |
| `dumb_cap_count` | 0.7659 | +0.0159 | 0.6386 | +0.0046 |
| `dumb_unparsed_count` | 0.7657 | +0.0157 | 0.6382 | +0.0042 |

## Raw features, AUROC (base-rate invariant; 0.5 is chance)

| feature | deepseek_qwen AUROC | deepseek_qwen d | llama AUROC | llama d |
|---|---|---|---|---|
| `length` | 0.653 [0.599, 0.704] | +0.548 | 0.582 [0.531, 0.634] | +0.299 |
| `entropy` | 0.582 [0.525, 0.636] | +0.281 | 0.559 [0.504, 0.608] | +0.208 |
| `logprob` | 0.582 [0.525, 0.637] | +0.289 | 0.559 [0.505, 0.608] | +0.209 |
| `vote_agreement` | 0.611 [0.566, 0.656] | +0.713 | 0.669 [0.624, 0.714] | +0.646 |
| `rmd_tail_q20` | 0.714 [0.665, 0.762] | +0.824 | 0.712 [0.666, 0.758] | +0.756 |
| `deepconf_global` | 0.497 [0.438, 0.554] | -0.001 | 0.496 [0.439, 0.550] | -0.030 |
| `deepconf_tail_q20` | 0.542 [0.484, 0.603] | +0.143 | 0.519 [0.464, 0.572] | +0.075 |
| `bottom10_group_confidence` | 0.485 [0.428, 0.541] | -0.055 | 0.488 [0.432, 0.540] | -0.052 |
| `lowest_group_confidence` | 0.485 [0.428, 0.541] | -0.062 | 0.491 [0.436, 0.544] | -0.045 |

## DeepConf statistics against B0's features (Pearson)

| model | statistic | length | entropy | logprob | vote_agreement | rmd_tail_q20 |
|---|---|---|---|---|---|---|
| deepseek_qwen | `deepconf_global` | -0.256 | +0.659 | +0.635 | +0.143 | +0.032 |
| deepseek_qwen | `deepconf_tail_q20` | +0.104 | +0.588 | +0.577 | +0.290 | +0.363 |
| deepseek_qwen | `bottom10_group_confidence` | -0.306 | +0.630 | +0.606 | +0.089 | -0.045 |
| deepseek_qwen | `lowest_group_confidence` | -0.325 | +0.609 | +0.585 | +0.082 | -0.063 |
| llama | `deepconf_global` | -0.380 | +0.383 | +0.341 | +0.295 | -0.143 |
| llama | `deepconf_tail_q20` | -0.296 | +0.060 | +0.037 | +0.458 | +0.138 |
| llama | `bottom10_group_confidence` | -0.325 | +0.439 | +0.396 | +0.192 | -0.167 |
| llama | `lowest_group_confidence` | -0.346 | +0.416 | +0.372 | +0.201 | -0.170 |

## Tail window size

| model | median trace length | median tail window (tokens) |
|---|---:|---:|
| deepseek_qwen | 2290 | 458 |
| llama | 1654 | 331 |
