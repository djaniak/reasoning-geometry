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

## Population: `cap_free_valid_plurality` (headline)

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

## Population: `cap_free_all_eight_parseable` (sensitivity)

| model | layer | peers | n | base acc |
|---|---:|---|---:|---:|
| qwen | 21 | deepseek, deepseek_llama | 391 | 0.691 |
| deepseek | 21 | deepseek_llama, qwen | 380 | 0.797 |
| deepseek_llama | 24 | deepseek, qwen | 408 | 0.674 |

### How strong is the control?

Marginal AUROC of each peer pass rate against the target's own outcome,
with the target's own `vote_agreement` and `rmd_tail_q20` for scale.

| model | peer 1 | peer 2 | vote_agreement | rmd_tail_q20 |
|---|---|---|---|---|
| qwen | 0.908 [0.871, 0.945] | 0.838 [0.792, 0.880] | 0.635 [0.580, 0.695] | 0.805 [0.756, 0.850] |
| deepseek | 0.907 [0.875, 0.935] | 0.960 [0.941, 0.975] | 0.584 [0.537, 0.635] | 0.687 [0.627, 0.752] |
| deepseek_llama | 0.813 [0.768, 0.856] | 0.802 [0.751, 0.851] | 0.650 [0.599, 0.706] | 0.709 [0.660, 0.760] |

### What the control is correlated with (Spearman)

| model | peer column | vs outcome | vs length | vs vote | vs rmd_tail |
|---|---|---:|---:|---:|---:|
| qwen | `peer_pass_rate__deepseek` | 0.764 | 0.138 | 0.222 | 0.426 |
| qwen | `peer_pass_rate__deepseek_llama` | 0.562 | 0.175 | 0.251 | 0.353 |
| deepseek | `peer_pass_rate__deepseek_llama` | 0.592 | 0.177 | 0.227 | 0.306 |
| deepseek | `peer_pass_rate__qwen` | 0.691 | 0.398 | 0.197 | 0.485 |
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
| qwen | 0.0538 | 0.1966 | 0.1379 | 0.0872 | 0.0760 | 0.0334 | 33% |
| deepseek | 0.0221 | 0.1553 | 0.1175 | 0.0265 | 0.0261 | 0.0044 | 9% |
| deepseek_llama | 0.0601 | 0.2369 | 0.1808 | 0.1698 | 0.1573 | 0.1097 | 11% |

### Paired deltas, AURC (pre-declared metric)

| model | B1 - B0 (reproduction) | **B1 - B0 given peer** | peer over B0 | peer over B1 |
|---|---|---|---|---|
| qwen | -0.0587 [-0.0974, -0.0214] p=0.004 | **-0.0112 [-0.0282, -0.0001] p=0.046** | -0.1094 [-0.1480, -0.0715] p=0.000 | -0.0507 [-0.0856, -0.0160] p=0.002 |
| deepseek | -0.0378 [-0.0689, -0.0079] p=0.016 | **-0.0004 [-0.0017, +0.0005] p=0.492** | -0.1287 [-0.1756, -0.0881] p=0.000 | -0.0909 [-0.1285, -0.0599] p=0.000 |
| deepseek_llama | -0.0560 [-0.0910, -0.0232] p=0.000 | **-0.0125 [-0.0230, -0.0026] p=0.004** | -0.0671 [-0.1163, -0.0160] p=0.016 | -0.0110 [-0.0518, +0.0326] p=0.598 |

### Paired deltas, AUACC (secondary)

| model | B1 - B0 (reproduction) | **B1 - B0 given peer** | peer over B0 | peer over B1 |
|---|---|---|---|---|
| qwen | +0.0587 [+0.0171, +0.0974] p=0.004 | **+0.0112 [+0.0003, +0.0277] p=0.040** | +0.1094 [+0.0684, +0.1497] p=0.000 | +0.0507 [+0.0175, +0.0858] p=0.002 |
| deepseek | +0.0378 [+0.0064, +0.0688] p=0.016 | **+0.0004 [-0.0006, +0.0016] p=0.498** | +0.1287 [+0.0874, +0.1692] p=0.000 | +0.0909 [+0.0618, +0.1243] p=0.000 |
| deepseek_llama | +0.0560 [+0.0195, +0.0938] p=0.000 | **+0.0125 [+0.0020, +0.0229] p=0.018** | +0.0671 [+0.0169, +0.1197] p=0.010 | +0.0110 [-0.0324, +0.0581] p=0.550 |

## Pre-declared rule

Evaluated on `cap_free_valid_plurality`, metric `aurc`, contrast `B1_minus_B0_given_peer`.

Report the increment as substantially a difficulty proxy if two or more models have an interval overlapping zero. Overlapping: deepseek (1/3). Triggered: **no**.

**Near-oracle note** (pre-declared). A peer rate reaches |Spearman| >= 0.60 against the target's own outcome on: qwen (deepseek +0.76); deepseek (qwen +0.70); deepseek_llama (deepseek +0.61).
A control this strong makes a surviving increment mean more and a dying
increment mean less -- a near-oracle can saturate the readout on its own,
which is a third reading, distinct from 'geometry is redundant with
difficulty'.

The flag fires on every model, so on its own it does not separate them.
The headroom column does, and it is the statistic that should have been
pre-declared in its place:

| model | headroom at `B0+peer` | delta | share of headroom removed |
|---|---:|---:|---:|
| qwen | 0.0330 | -0.0108 | 33% |
| deepseek | 0.0045 | -0.0004 | 8% |
| deepseek_llama | 0.1097 | -0.0125 | 11% |

### Multiplicity over the pre-declared family

Holm-Bonferroni over `B1_minus_B0_given_peer` across 3 models.
The three other contrasts per model are harness checks or exploratory and are
not in the family.

| model | raw p | Holm p | significant at 0.05 |
|---|---:|---:|---|
| deepseek_llama | 0.004 | 0.012 | yes |
| qwen | 0.036 | 0.072 | no |
| deepseek | 0.544 | 0.544 | no |

The bootstrap resolves p to 1/1000, so a Holm p within a few thousandths
of its threshold is borderline rather than a clean pass.
