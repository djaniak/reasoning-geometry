# Experiment 2 -- the cross-model empirical difficulty control

For each target model, the other two models' eight-sibling pass rates on the
same prompt ids enter `B0` as two features. The question is whether
`rmd_tail_q20` still adds once empirical problem difficulty is controlled by a
signal the target model did not produce.

AURC, **lower is better**; a negative delta favours the left-hand readout.
Readouts, folds, populations and the paired prompt bootstrap are the frozen
ones; the bootstrap runs over fixed OOF predictions and does not propagate
reference refitting. Prompt-id alignment is asserted from stored gold answers,
not assumed.

`B0 + peer` is a **control, not a baseline**: two other models' pass rates are
not available at decision time, so no method here competes with the headline.

## Population: `full_population` (headline)

| model | layer | peers | n | base acc |
|---|---:|---|---:|---:|
| qwen | 21 | deepseek, deepseek_llama | 500 | 0.620 |
| deepseek | 21 | deepseek_llama, qwen | 500 | 0.750 |
| deepseek_llama | 24 | deepseek, qwen | 500 | 0.634 |

### How strong is the control?

Marginal AUROC of each peer pass rate against the target's own outcome,
with the target's own `vote_agreement` and `rmd_tail_q20` for scale.

| model | peer 1 | peer 2 | vote_agreement | rmd_tail_q20 |
|---|---|---|---|---|
| qwen | 0.899 [0.868, 0.929] | 0.815 [0.778, 0.855] | 0.684 [0.635, 0.727] | 0.819 [0.781, 0.854] |
| deepseek | 0.903 [0.879, 0.928] | 0.925 [0.904, 0.944] | 0.611 [0.566, 0.656] | 0.714 [0.665, 0.762] |
| deepseek_llama | 0.831 [0.789, 0.870] | 0.800 [0.759, 0.840] | 0.669 [0.624, 0.714] | 0.712 [0.666, 0.758] |

### What the control is correlated with (Spearman)

| model | peer column | vs outcome | vs length | vs vote | vs rmd_tail |
|---|---|---:|---:|---:|---:|
| qwen | `peer_pass_rate__deepseek` | 0.749 | 0.252 | 0.297 | 0.483 |
| qwen | `peer_pass_rate__deepseek_llama` | 0.549 | 0.183 | 0.229 | 0.330 |
| deepseek | `peer_pass_rate__deepseek_llama` | 0.626 | 0.274 | 0.258 | 0.364 |
| deepseek | `peer_pass_rate__qwen` | 0.674 | 0.536 | 0.272 | 0.588 |
| deepseek_llama | `peer_pass_rate__deepseek` | 0.617 | 0.352 | 0.265 | 0.558 |
| deepseek_llama | `peer_pass_rate__qwen` | 0.529 | 0.505 | 0.237 | 0.598 |

### Readout AURC, and how much risk was still removable

`oracle` is the AURC of a ranker that puts every correct prompt ahead of
every wrong one: AURC does not bottom out at zero, and the floor rises as
base accuracy falls. `headroom` is `B0+peer` minus that floor -- the risk
still available for the tail feature to remove. `share` is the fraction of
that headroom the tail actually removes. When a delta is small, the share
is the number to read: a small delta against a large headroom is weak
evidence, a small delta against no headroom is no evidence at all.

| model | oracle | B0 | B1 | B0+peer | B1+peer | headroom at B0+peer | share removed |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen | 0.0836 | 0.2251 | 0.1730 | 0.1170 | 0.1103 | 0.0334 | 20% |
| deepseek | 0.0342 | 0.1643 | 0.1359 | 0.0414 | 0.0408 | 0.0071 | 8% |
| deepseek_llama | 0.0771 | 0.2480 | 0.2010 | 0.1715 | 0.1690 | 0.0944 | 3% |

### Paired deltas, AURC (pre-declared metric)

| model | B1 - B0 (reproduction) | **B1 - B0 given peer** | peer over B0 | peer over B1 |
|---|---|---|---|---|
| qwen | -0.0520 [-0.0845, -0.0218] p=0.000 | **-0.0067 [-0.0164, -0.0002] p=0.042** | -0.1081 [-0.1428, -0.0770] p=0.000 | -0.0561 [-0.0845, -0.0274] p=0.000 |
| deepseek | -0.0284 [-0.0526, -0.0048] p=0.010 | **-0.0006 [-0.0016, +0.0003] p=0.194** | -0.1229 [-0.1602, -0.0896] p=0.000 | -0.0946 [-0.1259, -0.0669] p=0.000 |
| deepseek_llama | -0.0469 [-0.0743, -0.0162] p=0.004 | **-0.0025 [-0.0069, +0.0026] p=0.280** | -0.0765 [-0.1227, -0.0326] p=0.000 | -0.0295 [-0.0640, +0.0034] p=0.084 |

### Paired deltas, AUACC (secondary)

| model | B1 - B0 (reproduction) | **B1 - B0 given peer** | peer over B0 | peer over B1 |
|---|---|---|---|---|
| qwen | +0.0520 [+0.0187, +0.0830] p=0.002 | **+0.0067 [+0.0001, +0.0157] p=0.046** | +0.1081 [+0.0764, +0.1398] p=0.000 | +0.0561 [+0.0303, +0.0858] p=0.000 |
| deepseek | +0.0284 [+0.0037, +0.0528] p=0.010 | **+0.0006 [-0.0004, +0.0016] p=0.232** | +0.1229 [+0.0872, +0.1614] p=0.000 | +0.0946 [+0.0666, +0.1279] p=0.000 |
| deepseek_llama | +0.0469 [+0.0161, +0.0765] p=0.000 | **+0.0025 [-0.0027, +0.0070] p=0.262** | +0.0765 [+0.0310, +0.1217] p=0.002 | +0.0295 [-0.0072, +0.0662] p=0.102 |

## Population: `cap_free_valid_plurality` (sensitivity)

| model | layer | peers | n | base acc |
|---|---:|---|---:|---:|
| qwen | 21 | deepseek, deepseek_llama | 392 | 0.691 |
| deepseek | 21 | deepseek_llama, qwen | 393 | 0.796 |
| deepseek_llama | 24 | deepseek, qwen | 408 | 0.674 |

### How strong is the control?

Marginal AUROC of each peer pass rate against the target's own outcome,
with the target's own `vote_agreement` and `rmd_tail_q20` for scale.

| model | peer 1 | peer 2 | vote_agreement | rmd_tail_q20 |
|---|---|---|---|---|
| qwen | 0.908 [0.871, 0.944] | 0.839 [0.795, 0.883] | 0.634 [0.580, 0.689] | 0.806 [0.758, 0.853] |
| deepseek | 0.904 [0.872, 0.932] | 0.961 [0.943, 0.975] | 0.587 [0.541, 0.639] | 0.686 [0.620, 0.750] |
| deepseek_llama | 0.813 [0.768, 0.856] | 0.802 [0.751, 0.851] | 0.650 [0.599, 0.706] | 0.709 [0.660, 0.760] |

### What the control is correlated with (Spearman)

| model | peer column | vs outcome | vs length | vs vote | vs rmd_tail |
|---|---|---:|---:|---:|---:|
| qwen | `peer_pass_rate__deepseek` | 0.765 | 0.140 | 0.219 | 0.427 |
| qwen | `peer_pass_rate__deepseek_llama` | 0.563 | 0.178 | 0.246 | 0.355 |
| deepseek | `peer_pass_rate__deepseek_llama` | 0.589 | 0.176 | 0.212 | 0.306 |
| deepseek | `peer_pass_rate__qwen` | 0.696 | 0.401 | 0.191 | 0.486 |
| deepseek_llama | `peer_pass_rate__deepseek` | 0.606 | 0.146 | 0.250 | 0.451 |
| deepseek_llama | `peer_pass_rate__qwen` | 0.529 | 0.373 | 0.208 | 0.497 |

### Readout AURC, and how much risk was still removable

`oracle` is the AURC of a ranker that puts every correct prompt ahead of
every wrong one: AURC does not bottom out at zero, and the floor rises as
base accuracy falls. `headroom` is `B0+peer` minus that floor -- the risk
still available for the tail feature to remove. `share` is the fraction of
that headroom the tail actually removes. When a delta is small, the share
is the number to read: a small delta against a large headroom is weak
evidence, a small delta against no headroom is no evidence at all.

| model | oracle | B0 | B1 | B0+peer | B1+peer | headroom at B0+peer | share removed |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen | 0.0535 | 0.1960 | 0.1375 | 0.0865 | 0.0757 | 0.0330 | 33% |
| deepseek | 0.0223 | 0.1522 | 0.1167 | 0.0268 | 0.0265 | 0.0045 | 8% |
| deepseek_llama | 0.0601 | 0.2369 | 0.1808 | 0.1698 | 0.1573 | 0.1097 | 11% |

### Paired deltas, AURC (pre-declared metric)

| model | B1 - B0 (reproduction) | **B1 - B0 given peer** | peer over B0 | peer over B1 |
|---|---|---|---|---|
| qwen | -0.0585 [-0.1026, -0.0182] p=0.004 | **-0.0108 [-0.0251, -0.0004] p=0.036** | -0.1095 [-0.1486, -0.0689] p=0.000 | -0.0510 [-0.0870, -0.0139] p=0.002 |
| deepseek | -0.0355 [-0.0642, -0.0097] p=0.004 | **-0.0004 [-0.0016, +0.0005] p=0.544** | -0.1254 [-0.1679, -0.0851] p=0.000 | -0.0899 [-0.1228, -0.0610] p=0.000 |
| deepseek_llama | -0.0560 [-0.0910, -0.0232] p=0.000 | **-0.0125 [-0.0230, -0.0026] p=0.004** | -0.0671 [-0.1163, -0.0160] p=0.016 | -0.0110 [-0.0518, +0.0326] p=0.598 |

### Paired deltas, AUACC (secondary)

| model | B1 - B0 (reproduction) | **B1 - B0 given peer** | peer over B0 | peer over B1 |
|---|---|---|---|---|
| qwen | +0.0585 [+0.0234, +0.0963] p=0.002 | **+0.0108 [-0.0003, +0.0259] p=0.054** | +0.1095 [+0.0731, +0.1487] p=0.000 | +0.0510 [+0.0190, +0.0858] p=0.004 |
| deepseek | +0.0355 [+0.0099, +0.0647] p=0.006 | **+0.0004 [-0.0005, +0.0016] p=0.458** | +0.1254 [+0.0857, +0.1690] p=0.000 | +0.0899 [+0.0607, +0.1245] p=0.000 |
| deepseek_llama | +0.0560 [+0.0195, +0.0938] p=0.000 | **+0.0125 [+0.0020, +0.0229] p=0.018** | +0.0671 [+0.0169, +0.1197] p=0.010 | +0.0110 [-0.0324, +0.0581] p=0.550 |

## Pre-declared rule

Evaluated on `full_population`, metric `aurc`, contrast `B1_minus_B0_given_peer`.

Report the increment as substantially a difficulty proxy if two or more models have an interval overlapping zero. Overlapping: deepseek, deepseek_llama (2/3). Triggered: **YES**.

**Near-oracle note** (pre-declared). A peer rate reaches |Spearman| >= 0.60 against the target's own outcome on: qwen (deepseek +0.75); deepseek (deepseek_llama +0.63, qwen +0.67); deepseek_llama (deepseek +0.62).
A control this strong makes a surviving increment mean more and a dying
increment mean less -- a near-oracle can saturate the readout on its own,
which is a third reading, distinct from 'geometry is redundant with
difficulty'.

The flag fires on every model, so on its own it does not separate them.
The headroom column does, and it is the statistic that should have been
pre-declared in its place:

| model | headroom at `B0+peer` | delta | share of headroom removed |
|---|---:|---:|---:|
| qwen | 0.0334 | -0.0067 | 20% |
| deepseek | 0.0071 | -0.0006 | 8% |
| deepseek_llama | 0.0944 | -0.0025 | 3% |

### Multiplicity over the pre-declared family

Holm-Bonferroni over `B1_minus_B0_given_peer` across 3 models.
The three other contrasts per model are harness checks or exploratory and are
not in the family.

| model | raw p | Holm p | significant at 0.05 |
|---|---:|---:|---|
| qwen | 0.042 | 0.126 | no |
| deepseek | 0.194 | 0.388 | no |
| deepseek_llama | 0.280 | 0.388 | no |

The bootstrap resolves p to 1/1000, so a Holm p within a few thousandths
of its threshold is borderline rather than a clean pass.
