# DeepConf used inside the prompt: weighted and filtered voting

Population: `full_population`. Seed 42, 1000 bootstrap draws.

## 1. Vote scores as raw abstention features (AUROC, 0.5 is chance)

| feature | deepseek_qwen | llama |
|---|---|---|
| `vote_agreement` | 0.611 | 0.669 |
| `dcfilter1_bottom10_group_confidence` | 0.529 | 0.572 |
| `dcfilter1_deepconf_global` | 0.524 | 0.559 |
| `dcfilter1_deepconf_tail_q20` | 0.505 | 0.576 |
| `dcfilter1_lowest_group_confidence` | 0.525 | 0.592 |
| `dcfilter2_bottom10_group_confidence` | 0.550 | 0.598 |
| `dcfilter2_deepconf_global` | 0.540 | 0.613 |
| `dcfilter2_deepconf_tail_q20` | 0.543 | 0.602 |
| `dcfilter2_lowest_group_confidence` | 0.548 | 0.591 |
| `dcfilter4_bottom10_group_confidence` | 0.573 | 0.644 |
| `dcfilter4_deepconf_global` | 0.560 | 0.632 |
| `dcfilter4_deepconf_tail_q20` | 0.559 | 0.630 |
| `dcfilter4_lowest_group_confidence` | 0.565 | 0.641 |
| `dcfilter6_bottom10_group_confidence` | 0.581 | 0.664 |
| `dcfilter6_deepconf_global` | 0.587 | 0.651 |
| `dcfilter6_deepconf_tail_q20` | 0.573 | 0.647 |
| `dcfilter6_lowest_group_confidence` | 0.582 | 0.667 |
| `dcvote_bottom10_group_confidence` | 0.589 | 0.666 |
| `dcvote_deepconf_global` | 0.589 | 0.668 |
| `dcvote_deepconf_tail_q20` | 0.588 | 0.669 |
| `dcvote_lowest_group_confidence` | 0.589 | 0.666 |
| `dcvote_own_bottom10_group_confidence` | 0.589 | 0.666 |
| `dcvote_own_deepconf_global` | 0.589 | 0.667 |
| `dcvote_own_deepconf_tail_q20` | 0.588 | 0.668 |
| `dcvote_own_lowest_group_confidence` | 0.589 | 0.666 |

## 2. Readouts (AUACC, and excess over the base rate)

| readout | deepseek_qwen AUACC | deepseek_qwen excess | llama AUACC | llama excess |
|---|---|---|---|---|
| `B0` | 0.8337 | +0.0837 | 0.7500 | +0.1160 |
| `B1` | 0.8621 | +0.1121 | 0.7970 | +0.1630 |
| `B0_dcvote_bottom10_group_confidence` | 0.8339 | +0.0839 | 0.7499 | +0.1159 |
| `B0_plus_dcvote_bottom10_group_confidence` | 0.8328 | +0.0828 | 0.7499 | +0.1159 |
| `B1_dcvote_bottom10_group_confidence` | 0.8626 | +0.1126 | 0.7965 | +0.1625 |
| `B1_plus_dcvote_bottom10_group_confidence` | 0.8615 | +0.1115 | 0.7965 | +0.1625 |
| `B0_dcvote_deepconf_tail_q20` | 0.8332 | +0.0832 | 0.7508 | +0.1168 |
| `B0_plus_dcvote_deepconf_tail_q20` | 0.8330 | +0.0830 | 0.7507 | +0.1167 |
| `B1_dcvote_deepconf_tail_q20` | 0.8624 | +0.1124 | 0.7970 | +0.1630 |
| `B1_plus_dcvote_deepconf_tail_q20` | 0.8615 | +0.1115 | 0.7970 | +0.1630 |

## 3. Paired deltas (AUACC)

| comparison | deepseek_qwen | llama |
|---|---|---|
| `B1_minus_B0` | +0.0284 [+0.006, +0.054] p=0.016 | +0.0469 [+0.017, +0.079] p=0.002 |
| `B1_minus_B0_dcvote_bottom10_group_confidence` | +0.0287 [+0.004, +0.054] p=0.018 | +0.0466 [+0.019, +0.077] p=0.000 |
| `B1_minus_B0_plus_dcvote_bottom10_group_confidence` | +0.0286 [+0.004, +0.052] p=0.020 | +0.0466 [+0.016, +0.077] p=0.000 |
| `B0_dcvote_bottom10_group_confidence_minus_B0` | +0.0002 [-0.005, +0.006] p=0.960 | -0.0001 [-0.003, +0.003] p=0.722 |
| `B0_plus_dcvote_bottom10_group_confidence_minus_B0` | -0.0009 [-0.002, -0.000] p=0.016 | -0.0001 [-0.001, +0.000] p=0.506 |
| `B1_minus_B0_plus_dcvote_bottom10_group_confidence_baseline` | +0.0293 [+0.007, +0.055] p=0.012 | +0.0471 [+0.017, +0.076] p=0.000 |
| `B1_minus_B0_dcvote_deepconf_tail_q20` | +0.0292 [+0.006, +0.054] p=0.014 | +0.0462 [+0.017, +0.078] p=0.002 |
| `B1_minus_B0_plus_dcvote_deepconf_tail_q20` | +0.0285 [+0.003, +0.052] p=0.024 | +0.0463 [+0.017, +0.075] p=0.002 |
| `B0_dcvote_deepconf_tail_q20_minus_B0` | -0.0005 [-0.007, +0.006] p=0.822 | +0.0007 [-0.002, +0.004] p=0.752 |
| `B0_plus_dcvote_deepconf_tail_q20_minus_B0` | -0.0007 [-0.002, -0.000] p=0.048 | +0.0007 [-0.002, +0.004] p=0.836 |
| `B1_minus_B0_plus_dcvote_deepconf_tail_q20_baseline` | +0.0291 [+0.005, +0.054] p=0.016 | +0.0463 [+0.018, +0.076] p=0.000 |

## 4. Accuracy of the answer each rule selects (and how often it departs from plurality)

| rule | deepseek_qwen acc | deepseek_qwen differs | llama acc | llama differs |
|---|---|---|---|---|
| `plurality` | 0.7500 | 0.000 | 0.6340 | 0.000 |
| `weighted_bottom10_group_confidence` | 0.7460 | 0.010 | 0.6360 | 0.024 |
| `top1_bottom10_group_confidence` | 0.7400 | 0.040 | 0.5680 | 0.178 |
| `top2_bottom10_group_confidence` | 0.7400 | 0.040 | 0.5680 | 0.178 |
| `top4_bottom10_group_confidence` | 0.7460 | 0.018 | 0.6060 | 0.086 |
| `top6_bottom10_group_confidence` | 0.7460 | 0.010 | 0.6220 | 0.042 |
| `weighted_deepconf_tail_q20` | 0.7480 | 0.004 | 0.6360 | 0.022 |
| `top1_deepconf_tail_q20` | 0.7380 | 0.018 | 0.6100 | 0.118 |
| `top2_deepconf_tail_q20` | 0.7380 | 0.018 | 0.6100 | 0.118 |
| `top4_deepconf_tail_q20` | 0.7460 | 0.010 | 0.6260 | 0.064 |
| `top6_deepconf_tail_q20` | 0.7460 | 0.004 | 0.6340 | 0.038 |
| `weighted_deepconf_global` | 0.7480 | 0.006 | 0.6360 | 0.016 |
| `top1_deepconf_global` | 0.7460 | 0.022 | 0.5760 | 0.156 |
| `top2_deepconf_global` | 0.7460 | 0.022 | 0.5760 | 0.156 |
| `top4_deepconf_global` | 0.7420 | 0.014 | 0.6080 | 0.082 |
| `top6_deepconf_global` | 0.7480 | 0.006 | 0.6260 | 0.034 |
| `weighted_lowest_group_confidence` | 0.7460 | 0.010 | 0.6420 | 0.030 |
| `top1_lowest_group_confidence` | 0.7400 | 0.038 | 0.5780 | 0.180 |
| `top2_lowest_group_confidence` | 0.7400 | 0.038 | 0.5780 | 0.180 |
| `top4_lowest_group_confidence` | 0.7460 | 0.018 | 0.6100 | 0.086 |
| `top6_lowest_group_confidence` | 0.7480 | 0.012 | 0.6280 | 0.052 |

## 5. How much room the weights have to change anything

| statistic | deepseek_qwen within-prompt CV | deepseek_qwen max/min | llama within-prompt CV | llama max/min |
|---|---|---|---|---|
| `bottom10_group_confidence` | 0.0518 | 1.181 | 0.0561 | 1.198 |
| `deepconf_tail_q20` | 0.0482 | 1.161 | 0.0551 | 1.189 |
| `deepconf_global` | 0.0322 | 1.105 | 0.0331 | 1.110 |
| `lowest_group_confidence` | 0.0550 | 1.192 | 0.0581 | 1.200 |
