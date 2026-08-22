# The two closest cheap baselines (experiments 1a and 1b)

AURC, **lower is better**; a negative delta favours the left-hand readout.
Every readout is the frozen cross-fitted logistic on the frozen prompt folds,
and every interval is the frozen paired prompt bootstrap over fixed OOF
predictions -- it does not propagate reference refitting.

`H` is `neg_answer_entropy`: minus the Shannon entropy (nats) of the
normalized exact-answer histogram over parseable siblings. `rmd_full` is the
whole-trace mean of per-token RMD (Vazhentsev ATRMD); `rmd_tail_q20` is the
same mean restricted to the final 20% of tokens.

## Population: `full_population` (headline)

| model | layer | n | base acc | prompts with no parseable answer |
|---|---:|---:|---:|---:|
| qwen | 21 | 500 | 0.620 | 2 |
| deepseek | 21 | 500 | 0.750 | 7 |
| deepseek_llama | 24 | 500 | 0.634 | 1 |

### Marginal AUROC of each single feature

| model | vote_agreement | H | rmd_tail_q20 | rmd_full |
|---|---|---|---|---|
| qwen | 0.684 [0.635, 0.727] | 0.675 [0.629, 0.719] | 0.819 [0.781, 0.854] | 0.759 [0.715, 0.801] |
| deepseek | 0.611 [0.566, 0.656] | 0.588 [0.544, 0.630] | 0.714 [0.665, 0.762] | 0.714 [0.664, 0.762] |
| deepseek_llama | 0.669 [0.624, 0.714] | 0.662 [0.617, 0.708] | 0.712 [0.666, 0.758] | 0.681 [0.635, 0.727] |

### Redundancy (Pearson / Spearman)

| model | H vs vote | H vs rmd_tail | rmd_full vs rmd_tail |
|---|---|---|---|
| qwen | 0.980 / 0.995 | 0.458 / 0.463 | 0.931 / 0.924 |
| deepseek | 0.967 / 1.000 | 0.204 / 0.184 | 0.957 / 0.966 |
| deepseek_llama | 0.984 / 0.996 | 0.264 / 0.287 | 0.938 / 0.924 |

### Readout AURC

| model | B0 | B1 | B0_plus_H | B0_plus_H_plus_rmd_tail | B0_plus_rmd_full | B0_plus_both_rmd |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 0.2251 | 0.1730 | 0.2248 | 0.1728 | 0.2073 | 0.1609 |
| deepseek | 0.1643 | 0.1359 | 0.1644 | 0.1363 | 0.1355 | 0.1325 |
| deepseek_llama | 0.2480 | 0.2010 | 0.2496 | 0.2034 | 0.2034 | 0.1993 |

### 1a -- does tail RMD survive the answer histogram?

| model | B1 - B0 (reproduction) | H over B0 | **rmd_tail over B0+H** | H over B1 |
|---|---|---|---|---|
| qwen | -0.0520 [-0.0832, -0.0218] p=0.004 | -0.0003 [-0.0016, +0.0009] p=0.598 | **-0.0519 [-0.0829, -0.0216] p=0.004** | -0.0002 [-0.0011, +0.0006] p=0.558 |
| deepseek | -0.0284 [-0.0531, -0.0060] p=0.008 | +0.0001 [-0.0017, +0.0013] p=0.792 | **-0.0281 [-0.0524, -0.0057] p=0.008** | +0.0004 [-0.0010, +0.0016] p=0.490 |
| deepseek_llama | -0.0469 [-0.0773, -0.0178] p=0.006 | +0.0016 [-0.0000, +0.0035] p=0.056 | **-0.0462 [-0.0773, -0.0157] p=0.008** | +0.0023 [-0.0001, +0.0052] p=0.062 |

### 1b -- the tail against the whole trace

| model | rmd_full over B0 | rmd_tail over B0 | **rmd_tail over rmd_full** | rmd_full over rmd_tail |
|---|---|---|---|---|
| qwen | -0.0178 [-0.0435, +0.0105] p=0.204 | -0.0520 [-0.0832, -0.0218] p=0.004 | **-0.0464 [-0.0724, -0.0224] p=0.000** | -0.0122 [-0.0238, -0.0024] p=0.012 |
| deepseek | -0.0287 [-0.0534, -0.0055] p=0.016 | -0.0284 [-0.0531, -0.0060] p=0.008 | **-0.0030 [-0.0109, +0.0035] p=0.436** | -0.0034 [-0.0178, +0.0107] p=0.662 |
| deepseek_llama | -0.0445 [-0.0739, -0.0151] p=0.002 | -0.0469 [-0.0773, -0.0178] p=0.006 | **-0.0041 [-0.0118, +0.0035] p=0.320** | -0.0017 [-0.0079, +0.0045] p=0.596 |

## Population: `cap_free_valid_plurality` (sensitivity)

| model | layer | n | base acc | prompts with no parseable answer |
|---|---:|---:|---:|---:|
| qwen | 21 | 392 | 0.691 | 0 |
| deepseek | 21 | 393 | 0.796 | 0 |
| deepseek_llama | 24 | 408 | 0.674 | 0 |

### Marginal AUROC of each single feature

| model | vote_agreement | H | rmd_tail_q20 | rmd_full |
|---|---|---|---|---|
| qwen | 0.634 [0.580, 0.689] | 0.631 [0.578, 0.687] | 0.806 [0.758, 0.853] | 0.715 [0.656, 0.768] |
| deepseek | 0.587 [0.541, 0.639] | 0.587 [0.540, 0.639] | 0.686 [0.620, 0.750] | 0.682 [0.612, 0.749] |
| deepseek_llama | 0.650 [0.599, 0.706] | 0.649 [0.598, 0.706] | 0.709 [0.660, 0.760] | 0.667 [0.612, 0.721] |

### Redundancy (Pearson / Spearman)

| model | H vs vote | H vs rmd_tail | rmd_full vs rmd_tail |
|---|---|---|---|
| qwen | 0.984 / 0.998 | 0.363 / 0.324 | 0.890 / 0.878 |
| deepseek | 0.974 / 1.000 | 0.106 / 0.096 | 0.950 / 0.944 |
| deepseek_llama | 0.987 / 0.996 | 0.280 / 0.291 | 0.900 / 0.879 |

### Readout AURC

| model | B0 | B1 | B0_plus_H | B0_plus_H_plus_rmd_tail | B0_plus_rmd_full | B0_plus_both_rmd |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 0.1960 | 0.1375 | 0.1954 | 0.1368 | 0.1823 | 0.1240 |
| deepseek | 0.1522 | 0.1167 | 0.1488 | 0.1158 | 0.1187 | 0.1172 |
| deepseek_llama | 0.2369 | 0.1808 | 0.2372 | 0.1816 | 0.1860 | 0.1794 |

### 1a -- does tail RMD survive the answer histogram?

| model | B1 - B0 (reproduction) | H over B0 | **rmd_tail over B0+H** | H over B1 |
|---|---|---|---|---|
| qwen | -0.0585 [-0.0975, -0.0221] p=0.000 | -0.0006 [-0.0022, +0.0009] p=0.398 | **-0.0586 [-0.0974, -0.0223] p=0.000** | -0.0007 [-0.0020, +0.0003] p=0.242 |
| deepseek | -0.0355 [-0.0636, -0.0071] p=0.010 | -0.0035 [-0.0087, +0.0000] p=0.054 | **-0.0330 [-0.0611, -0.0046] p=0.016** | -0.0010 [-0.0045, +0.0014] p=0.542 |
| deepseek_llama | -0.0560 [-0.0917, -0.0183] p=0.002 | +0.0003 [-0.0015, +0.0028] p=0.746 | **-0.0557 [-0.0915, -0.0176] p=0.002** | +0.0007 [-0.0004, +0.0021] p=0.284 |

### 1b -- the tail against the whole trace

| model | rmd_full over B0 | rmd_tail over B0 | **rmd_tail over rmd_full** | rmd_full over rmd_tail |
|---|---|---|---|---|
| qwen | -0.0137 [-0.0493, +0.0213] p=0.428 | -0.0585 [-0.0975, -0.0221] p=0.000 | **-0.0583 [-0.0929, -0.0258] p=0.000** | -0.0134 [-0.0284, +0.0020] p=0.094 |
| deepseek | -0.0335 [-0.0639, -0.0023] p=0.034 | -0.0355 [-0.0636, -0.0071] p=0.010 | **-0.0016 [-0.0091, +0.0068] p=0.750** | +0.0004 [-0.0128, +0.0142] p=0.968 |
| deepseek_llama | -0.0509 [-0.0879, -0.0152] p=0.004 | -0.0560 [-0.0917, -0.0183] p=0.002 | **-0.0066 [-0.0159, +0.0019] p=0.154** | -0.0014 [-0.0088, +0.0050] p=0.650 |

## Population: `cap_free_all_eight_parseable` (sensitivity)

| model | layer | n | base acc | prompts with no parseable answer |
|---|---:|---:|---:|---:|
| qwen | 21 | 391 | 0.691 | 0 |
| deepseek | 21 | 380 | 0.797 | 0 |
| deepseek_llama | 24 | 408 | 0.674 | 0 |

### Marginal AUROC of each single feature

| model | vote_agreement | H | rmd_tail_q20 | rmd_full |
|---|---|---|---|---|
| qwen | 0.635 [0.580, 0.695] | 0.632 [0.579, 0.692] | 0.805 [0.756, 0.850] | 0.715 [0.661, 0.774] |
| deepseek | 0.584 [0.537, 0.635] | 0.584 [0.537, 0.635] | 0.687 [0.627, 0.752] | 0.685 [0.622, 0.747] |
| deepseek_llama | 0.650 [0.599, 0.706] | 0.649 [0.598, 0.706] | 0.709 [0.660, 0.760] | 0.667 [0.612, 0.721] |

### Redundancy (Pearson / Spearman)

| model | H vs vote | H vs rmd_tail | rmd_full vs rmd_tail |
|---|---|---|---|
| qwen | 0.984 / 0.998 | 0.367 / 0.330 | 0.890 / 0.879 |
| deepseek | 0.975 / 1.000 | 0.098 / 0.092 | 0.949 / 0.942 |
| deepseek_llama | 0.987 / 0.996 | 0.280 / 0.291 | 0.900 / 0.879 |

### Readout AURC

| model | B0 | B1 | B0_plus_H | B0_plus_H_plus_rmd_tail | B0_plus_rmd_full | B0_plus_both_rmd |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 0.1966 | 0.1379 | 0.1961 | 0.1375 | 0.1828 | 0.1244 |
| deepseek | 0.1553 | 0.1175 | 0.1536 | 0.1178 | 0.1185 | 0.1153 |
| deepseek_llama | 0.2369 | 0.1808 | 0.2372 | 0.1816 | 0.1860 | 0.1794 |

### 1a -- does tail RMD survive the answer histogram?

| model | B1 - B0 (reproduction) | H over B0 | **rmd_tail over B0+H** | H over B1 |
|---|---|---|---|---|
| qwen | -0.0587 [-0.0996, -0.0213] p=0.002 | -0.0005 [-0.0019, +0.0010] p=0.476 | **-0.0586 [-0.0995, -0.0211] p=0.002** | -0.0004 [-0.0016, +0.0006] p=0.534 |
| deepseek | -0.0378 [-0.0686, -0.0071] p=0.012 | -0.0016 [-0.0067, +0.0017] p=0.578 | **-0.0358 [-0.0671, -0.0056] p=0.016** | +0.0004 [-0.0010, +0.0015] p=0.512 |
| deepseek_llama | -0.0560 [-0.0917, -0.0183] p=0.002 | +0.0003 [-0.0015, +0.0028] p=0.746 | **-0.0557 [-0.0915, -0.0176] p=0.002** | +0.0007 [-0.0004, +0.0021] p=0.284 |

### 1b -- the tail against the whole trace

| model | rmd_full over B0 | rmd_tail over B0 | **rmd_tail over rmd_full** | rmd_full over rmd_tail |
|---|---|---|---|---|
| qwen | -0.0138 [-0.0517, +0.0227] p=0.464 | -0.0587 [-0.0996, -0.0213] p=0.002 | **-0.0584 [-0.0933, -0.0219] p=0.000** | -0.0135 [-0.0289, +0.0023] p=0.084 |
| deepseek | -0.0368 [-0.0682, -0.0068] p=0.020 | -0.0378 [-0.0686, -0.0071] p=0.012 | **-0.0032 [-0.0127, +0.0047] p=0.510** | -0.0022 [-0.0148, +0.0109] p=0.714 |
| deepseek_llama | -0.0509 [-0.0879, -0.0152] p=0.004 | -0.0560 [-0.0917, -0.0183] p=0.002 | **-0.0066 [-0.0159, +0.0019] p=0.154** | -0.0014 [-0.0088, +0.0050] p=0.650 |

## 1b follow-up: is the tail restriction a window-size effect?

Distillation, reasoning training and trace length are collinear *between*
these three models, so no cross-model comparison can separate them. Trace
length varies *within* each model, so the terciles below ask whether the
tail's advantage over the whole-trace mean decays with window size inside
one model, holding everything else fixed. Not a region sweep:
`rmd_tail_q20` keeps its frozen definition and no new region is opened.

Strata are terciles of the sibling-mean tail-window size `ceil(0.20 * trace_length)`, in tokens. A stratum is reported only with at least
25 prompts of each class.

| model | stratum | n | wrong | base acc | window med [min, max] | **rmd_tail over rmd_full** |
|---|---|---:|---:|---:|---|---|
| qwen | window_short | 167 | 35 | 0.790 | 65 [34, 80] | -0.0448 [-0.0993, -0.0044] p=0.020 |
| qwen | window_mid | 166 | 60 | 0.639 | 103 [80, 131] | -0.1083 [-0.1682, -0.0489] p=0.000 |
| qwen | window_long | 167 | 95 | 0.431 | 168 [131, 205] | -0.0491 [-0.0868, -0.0084] p=0.018 |
| qwen | window_below_median | 250 | 62 | 0.752 | 72 [34, 103] | -0.0563 [-0.1038, -0.0163] p=0.004 |
| qwen | window_above_median | 250 | 128 | 0.488 | 151 [103, 205] | -0.0662 [-0.0998, -0.0288] p=0.000 |
| deepseek | window_short | 167 | 23 | 0.862 | 233 [47, 342] | not reported (min class 23) |
| deepseek | window_mid | 166 | 43 | 0.741 | 478 [343, 686] | +0.0039 [-0.0052, +0.0122] p=0.378 |
| deepseek | window_long | 167 | 59 | 0.647 | 1130 [692, 1639] | -0.0187 [-0.0454, +0.0064] p=0.182 |
| deepseek | window_below_median | 250 | 43 | 0.828 | 285 [47, 477] | -0.0062 [-0.0186, +0.0072] p=0.378 |
| deepseek | window_above_median | 250 | 82 | 0.672 | 837 [479, 1639] | -0.0098 [-0.0247, +0.0034] p=0.142 |
| deepseek_llama | window_short | 167 | 51 | 0.695 | 115 [52, 203] | -0.0081 [-0.0193, +0.0012] p=0.100 |
| deepseek_llama | window_mid | 167 | 55 | 0.671 | 359 [206, 605] | -0.0024 [-0.0151, +0.0118] p=0.766 |
| deepseek_llama | window_long | 166 | 77 | 0.536 | 1293 [606, 2458] | -0.0194 [-0.0461, +0.0088] p=0.160 |
| deepseek_llama | window_below_median | 250 | 77 | 0.692 | 148 [52, 359] | -0.0010 [-0.0092, +0.0065] p=0.754 |
| deepseek_llama | window_above_median | 250 | 106 | 0.576 | 860 [359, 2458] | -0.0182 [-0.0354, -0.0026] p=0.022 |

## Pre-declared rules

Evaluated on `full_population`.

**1a** (`rmd_tail_over_B0_plus_H`) -- stop the 'beyond self-consistency' claim if two or more models have an interval overlapping zero. Overlapping: none (0/3). Triggered: **no**.

**1b** (`rmd_tail_over_rmd_full`) -- no region or percentile sweep follows, whichever way this lands. This one has no
trigger; what it decides is whether a tail-specific contribution survives.

| model | branch |
|---|---|
| qwen | `tail_wins` |
| deepseek | `tie_or_full_wins` |
| deepseek_llama | `tie_or_full_wins` |

Tail wins on 1/3 models.

### Multiplicity over the pre-declared family

Holm-Bonferroni over the two pre-declared contrasts across three models. The
five other contrasts per model were exploratory and are not in the family.

| test | raw p | Holm p | significant at 0.05 |
|---|---:|---:|---|
| `qwen:rmd_tail_over_rmd_full` | 0.000 | 0.000 | yes |
| `qwen:rmd_tail_over_B0_plus_H` | 0.004 | 0.020 | yes |
| `deepseek:rmd_tail_over_B0_plus_H` | 0.008 | 0.032 | yes |
| `deepseek_llama:rmd_tail_over_B0_plus_H` | 0.008 | 0.032 | yes |
| `deepseek_llama:rmd_tail_over_rmd_full` | 0.320 | 0.640 | no |
| `deepseek:rmd_tail_over_rmd_full` | 0.436 | 0.640 | no |

Family size 6. The bootstrap resolves p to 1/1000, so any Holm p within a few thousandths of its threshold should be read as borderline rather than as a clean pass.
