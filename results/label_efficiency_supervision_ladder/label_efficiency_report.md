# Label-efficiency curves: one-class geometry versus a supervised probe

`rmd_tail_q20` fits a Gaussian on correct traces only; `probe_hidden_tail_q20` fits an LDA on both classes over the same PCA-projected tail means. At the full label budget the probe is ahead, so the only deployment claim geometry can carry is that it needs fewer labels. These curves are that claim, or its refutation.

At each budget the PCA basis, the correct-trace Gaussian, the background Gaussian, the LDA, and the logistic readout are all refitted from that budget's prompts alone. Training sets are nested along one permutation per replicate; the evaluation set is the headline-population complement of the largest budget and is held fixed across budgets, so every number in a row is scored on the same prompts.

## qwen

Layer 21, PCA 128, 500 prompts (392 in the `cap_free_valid_plurality` evaluation pool), 10 label draws, 3 inner folds on the training side, 256 reference tokens per trace.

### Feature AUROC against the budget

Prompt-level AUROC of each feature alone, the base-rate-free view. Median over label draws, 10--90 band.

`probe_token_tail_q20` is the pooling-matched probe: LDA per tail token, token scores averaged, which is the order `rmd_tail_q20` uses. `qmd_tail_q20` goes one step further and matches the decision function too -- it is RMD's own quadratic with the unconditional background replaced by an incorrect-trace Gaussian, so `rmd − qmd` is the gap left when only supervision differs.

| labelled prompts | `rmd_tail_q20` | `probe_hidden_tail_q20` | `probe_token_tail_q20` | `qmd_tail_q20` | rmd − probe | rmd − token probe | rmd − qmd |
|---:|---|---|---|---|---|---|---|
| 25 | 0.719 [0.639, 0.796] | 0.692 [0.646, 0.759] | 0.700 [0.605, 0.774] | 0.671 [0.566, 0.758] | 0.028 [-0.060, 0.070] | 0.021 [-0.080, 0.080] | 0.040 [0.028, 0.075] |
| 50 | 0.780 [0.656, 0.809] | 0.765 [0.680, 0.800] | 0.759 [0.714, 0.814] | 0.737 [0.635, 0.784] | 0.028 [-0.091, 0.085] | 0.016 [-0.098, 0.073] | 0.031 [0.020, 0.054] |
| 100 | 0.799 [0.743, 0.834] | 0.786 [0.748, 0.816] | 0.789 [0.761, 0.815] | 0.788 [0.734, 0.828] | 0.023 [-0.051, 0.043] | 0.017 [-0.040, 0.029] | 0.003 [-0.012, 0.037] |

### AURC against the budget

Lower is better. `excess` subtracts the chance AURC `(1 − 1/n) − base`, which is what makes a level comparable at all.

| labelled prompts | eval n | base acc | B0 | B0+rmd | B0+probe | B0+token probe | B0+qmd | excess B0+rmd | excess B0+probe |
|---:|---:|---|---|---|---|---|---|---|---|
| 25 | 314 | 0.689 | 0.230 [0.207, 0.251] | 0.193 [0.158, 0.255] | 0.218 [0.183, 0.249] | 0.216 [0.167, 0.243] | 0.207 [0.175, 0.254] | -0.109 [-0.149, -0.062] | -0.082 [-0.129, -0.060] |
| 50 | 314 | 0.689 | 0.216 [0.191, 0.280] | 0.154 [0.140, 0.201] | 0.172 [0.156, 0.216] | 0.190 [0.141, 0.236] | 0.165 [0.152, 0.211] | -0.151 [-0.166, -0.110] | -0.135 [-0.154, -0.084] |
| 100 | 314 | 0.689 | 0.203 [0.189, 0.218] | 0.139 [0.116, 0.160] | 0.149 [0.129, 0.164] | 0.146 [0.127, 0.184] | 0.142 [0.120, 0.194] | -0.168 [-0.180, -0.145] | -0.162 [-0.176, -0.139] |

### Paired AURC deltas against the budget

Paired inside a replicate: identical evaluation prompts, identical labelled prompts, one logistic each. Negative favours the left readout. `wins` is the share of label draws landing on that side.

| labelled prompts | B0+rmd − B0 | B0+probe − B0 | B0+token probe − B0 | B0+rmd − B0+probe | wins | sign p | B0+rmd − B0+token probe | wins | sign p |
|---:|---|---|---|---|---:|---:|---|---:|---:|
| 25 | -0.037 [-0.061, 0.013] | -0.005 [-0.040, 0.009] | -0.005 [-0.071, 0.012] | -0.005 [-0.044, 0.019] | 0.70 | 0.344 | -0.000 [-0.050, 0.014] | 0.50 | 1.000 |
| 50 | -0.063 [-0.130, -0.002] | -0.047 [-0.071, 0.008] | -0.029 [-0.075, 0.019] | -0.017 [-0.038, 0.010] | 0.80 | 0.109 | -0.033 [-0.044, 0.024] | 0.80 | 0.109 |
| 100 | -0.066 [-0.081, -0.034] | -0.059 [-0.069, -0.041] | -0.053 [-0.076, -0.030] | -0.013 [-0.023, 0.026] | 0.70 | 0.344 | -0.011 [-0.027, 0.011] | 0.70 | 0.344 |

### The supervision ladder

Each rung releases one variable, so a gap that survives every rung is the one attributable to that rung alone. Negative favours the left readout.

| rung | what it releases |
|---|---|
| `B0+rmd − B0+qmd` | whether the negative class was labelled |
| `B0+qmd − B0+token probe` | quadratic against linear decision function |
| `B0+token probe − B0+probe` | score-then-pool against pool-then-score |

| labelled prompts | rmd − qmd | wins | sign p | qmd − token probe | wins | sign p | qmd − B0 |
|---:|---|---:|---:|---|---:|---:|---|
| 25 | -0.009 [-0.027, 0.006] | 0.50 | 1.000 | -0.004 [-0.032, 0.021] | 0.60 | 0.754 | -0.020 [-0.054, 0.011] |
| 50 | -0.009 [-0.022, 0.003] | 0.80 | 0.109 | -0.018 [-0.036, 0.013] | 0.60 | 0.754 | -0.041 [-0.122, 0.003] |
| 100 | -0.003 [-0.016, 0.007] | 0.60 | 0.754 | -0.006 [-0.028, 0.024] | 0.70 | 0.344 | -0.063 [-0.082, -0.025] |

### Crossing

geometry is ahead at every budget tested.

## deepseek

Layer 21, PCA 128, 500 prompts (393 in the `cap_free_valid_plurality` evaluation pool), 10 label draws, 3 inner folds on the training side, 256 reference tokens per trace.

### Feature AUROC against the budget

Prompt-level AUROC of each feature alone, the base-rate-free view. Median over label draws, 10--90 band.

`probe_token_tail_q20` is the pooling-matched probe: LDA per tail token, token scores averaged, which is the order `rmd_tail_q20` uses. `qmd_tail_q20` goes one step further and matches the decision function too -- it is RMD's own quadratic with the unconditional background replaced by an incorrect-trace Gaussian, so `rmd − qmd` is the gap left when only supervision differs.

| labelled prompts | `rmd_tail_q20` | `probe_hidden_tail_q20` | `probe_token_tail_q20` | `qmd_tail_q20` | rmd − probe | rmd − token probe | rmd − qmd |
|---:|---|---|---|---|---|---|---|
| 25 | 0.663 [0.585, 0.789] | 0.664 [0.599, 0.733] | 0.625 [0.546, 0.720] | 0.645 [0.568, 0.741] | 0.003 [-0.037, 0.061] | 0.034 [0.020, 0.075] | 0.014 [0.002, 0.052] |
| 50 | 0.732 [0.640, 0.754] | 0.755 [0.658, 0.767] | 0.680 [0.618, 0.750] | 0.728 [0.652, 0.775] | -0.026 [-0.071, 0.018] | 0.013 [-0.025, 0.058] | -0.012 [-0.025, 0.020] |
| 100 | 0.721 [0.669, 0.744] | 0.768 [0.738, 0.800] | 0.749 [0.669, 0.782] | 0.736 [0.693, 0.791] | -0.034 [-0.111, -0.010] | -0.030 [-0.088, 0.004] | -0.040 [-0.063, 0.033] |

### AURC against the budget

Lower is better. `excess` subtracts the chance AURC `(1 − 1/n) − base`, which is what makes a level comparable at all.

| labelled prompts | eval n | base acc | B0 | B0+rmd | B0+probe | B0+token probe | B0+qmd | excess B0+rmd | excess B0+probe |
|---:|---:|---|---|---|---|---|---|---|---|
| 25 | 314 | 0.796 | 0.162 [0.135, 0.208] | 0.137 [0.086, 0.196] | 0.170 [0.146, 0.200] | 0.161 [0.140, 0.236] | 0.152 [0.110, 0.221] | -0.064 [-0.100, -0.005] | -0.028 [-0.052, -0.005] |
| 50 | 314 | 0.796 | 0.155 [0.128, 0.188] | 0.110 [0.082, 0.157] | 0.140 [0.079, 0.173] | 0.163 [0.095, 0.241] | 0.123 [0.089, 0.203] | -0.089 [-0.118, -0.043] | -0.059 [-0.115, -0.034] |
| 100 | 314 | 0.796 | 0.158 [0.127, 0.167] | 0.125 [0.076, 0.148] | 0.099 [0.069, 0.114] | 0.104 [0.069, 0.177] | 0.104 [0.075, 0.153] | -0.082 [-0.102, -0.050] | -0.102 [-0.119, -0.097] |

### Paired AURC deltas against the budget

Paired inside a replicate: identical evaluation prompts, identical labelled prompts, one logistic each. Negative favours the left readout. `wins` is the share of label draws landing on that side.

| labelled prompts | B0+rmd − B0 | B0+probe − B0 | B0+token probe − B0 | B0+rmd − B0+probe | wins | sign p | B0+rmd − B0+token probe | wins | sign p |
|---:|---|---|---|---|---:|---:|---|---:|---:|
| 25 | -0.013 [-0.086, 0.004] | 0.005 [-0.046, 0.034] | 0.012 [-0.048, 0.037] | -0.021 [-0.053, -0.008] | 0.90 | 0.021 | -0.024 [-0.062, -0.009] | 1.00 | 0.002 |
| 50 | -0.042 [-0.059, -0.014] | -0.014 [-0.067, 0.003] | 0.001 [-0.042, 0.053] | -0.011 [-0.051, 0.008] | 0.70 | 0.344 | -0.043 [-0.086, -0.007] | 0.90 | 0.021 |
| 100 | -0.034 [-0.053, -0.014] | -0.056 [-0.073, -0.046] | -0.051 [-0.066, 0.022] | 0.021 [0.001, 0.043] | 0.10 | 0.021 | 0.008 [-0.056, 0.033] | 0.40 | 0.754 |

### The supervision ladder

Each rung releases one variable, so a gap that survives every rung is the one attributable to that rung alone. Negative favours the left readout.

| rung | what it releases |
|---|---|
| `B0+rmd − B0+qmd` | whether the negative class was labelled |
| `B0+qmd − B0+token probe` | quadratic against linear decision function |
| `B0+token probe − B0+probe` | score-then-pool against pool-then-score |

| labelled prompts | rmd − qmd | wins | sign p | qmd − token probe | wins | sign p | qmd − B0 |
|---:|---|---:|---:|---|---:|---:|---|
| 25 | -0.014 [-0.037, 0.000] | 0.90 | 0.021 | -0.015 [-0.047, 0.002] | 0.80 | 0.109 | 0.002 [-0.074, 0.018] |
| 50 | -0.021 [-0.039, 0.001] | 0.80 | 0.109 | -0.032 [-0.050, -0.000] | 1.00 | 0.002 | -0.020 [-0.055, 0.016] |
| 100 | 0.008 [-0.008, 0.027] | 0.40 | 0.754 | 0.003 [-0.030, 0.009] | 0.30 | 0.344 | -0.052 [-0.068, 0.006] |

### Crossing

Median `B0_rmd − B0_probe` crosses zero at about **63 labelled prompts** (bracketed by [50, 100]); geometry is the better feature below this budget.

## deepseek_llama

Layer 24, PCA 128, 500 prompts (408 in the `cap_free_valid_plurality` evaluation pool), 10 label draws, 3 inner folds on the training side, 256 reference tokens per trace.

### Feature AUROC against the budget

Prompt-level AUROC of each feature alone, the base-rate-free view. Median over label draws, 10--90 band.

`probe_token_tail_q20` is the pooling-matched probe: LDA per tail token, token scores averaged, which is the order `rmd_tail_q20` uses. `qmd_tail_q20` goes one step further and matches the decision function too -- it is RMD's own quadratic with the unconditional background replaced by an incorrect-trace Gaussian, so `rmd − qmd` is the gap left when only supervision differs.

| labelled prompts | `rmd_tail_q20` | `probe_hidden_tail_q20` | `probe_token_tail_q20` | `qmd_tail_q20` | rmd − probe | rmd − token probe | rmd − qmd |
|---:|---|---|---|---|---|---|---|
| 25 | 0.645 [0.538, 0.708] | 0.654 [0.583, 0.680] | 0.618 [0.562, 0.728] | 0.637 [0.543, 0.709] | 0.011 [-0.092, 0.061] | 0.009 [-0.057, 0.051] | 0.007 [-0.006, 0.020] |
| 50 | 0.676 [0.611, 0.736] | 0.695 [0.664, 0.747] | 0.695 [0.607, 0.755] | 0.655 [0.624, 0.737] | -0.014 [-0.074, 0.038] | -0.005 [-0.041, 0.029] | 0.005 [-0.013, 0.035] |
| 100 | 0.719 [0.645, 0.748] | 0.743 [0.708, 0.760] | 0.727 [0.652, 0.785] | 0.718 [0.641, 0.758] | -0.012 [-0.119, 0.016] | -0.020 [-0.085, 0.036] | 0.002 [-0.037, 0.039] |

### AURC against the budget

Lower is better. `excess` subtracts the chance AURC `(1 − 1/n) − base`, which is what makes a level comparable at all.

| labelled prompts | eval n | base acc | B0 | B0+rmd | B0+probe | B0+token probe | B0+qmd | excess B0+rmd | excess B0+probe |
|---:|---:|---|---|---|---|---|---|---|---|
| 25 | 328 | 0.671 | 0.242 [0.216, 0.327] | 0.255 [0.190, 0.320] | 0.238 [0.204, 0.324] | 0.236 [0.220, 0.344] | 0.277 [0.201, 0.340] | -0.079 [-0.121, -0.001] | -0.075 [-0.112, -0.003] |
| 50 | 328 | 0.671 | 0.238 [0.223, 0.283] | 0.184 [0.158, 0.221] | 0.210 [0.176, 0.234] | 0.212 [0.184, 0.342] | 0.199 [0.156, 0.237] | -0.127 [-0.160, -0.109] | -0.112 [-0.152, -0.081] |
| 100 | 328 | 0.671 | 0.229 [0.216, 0.265] | 0.168 [0.151, 0.202] | 0.166 [0.145, 0.201] | 0.170 [0.145, 0.213] | 0.169 [0.153, 0.201] | -0.154 [-0.164, -0.127] | -0.162 [-0.166, -0.119] |

### Paired AURC deltas against the budget

Paired inside a replicate: identical evaluation prompts, identical labelled prompts, one logistic each. Negative favours the left readout. `wins` is the share of label draws landing on that side.

| labelled prompts | B0+rmd − B0 | B0+probe − B0 | B0+token probe − B0 | B0+rmd − B0+probe | wins | sign p | B0+rmd − B0+token probe | wins | sign p |
|---:|---|---|---|---|---:|---:|---|---:|---:|
| 25 | -0.015 [-0.051, 0.044] | -0.010 [-0.022, 0.036] | -0.013 [-0.029, 0.031] | -0.014 [-0.043, 0.063] | 0.60 | 0.754 | 0.007 [-0.040, 0.027] | 0.50 | 1.000 |
| 50 | -0.058 [-0.083, -0.044] | -0.046 [-0.066, -0.012] | -0.026 [-0.058, 0.065] | -0.025 [-0.046, 0.010] | 0.80 | 0.109 | -0.021 [-0.136, -0.003] | 0.90 | 0.021 |
| 100 | -0.062 [-0.085, -0.034] | -0.069 [-0.079, -0.049] | -0.061 [-0.084, -0.029] | 0.002 [-0.010, 0.017] | 0.40 | 0.754 | -0.002 [-0.025, 0.015] | 0.60 | 0.754 |

### The supervision ladder

Each rung releases one variable, so a gap that survives every rung is the one attributable to that rung alone. Negative favours the left readout.

| rung | what it releases |
|---|---|
| `B0+rmd − B0+qmd` | whether the negative class was labelled |
| `B0+qmd − B0+token probe` | quadratic against linear decision function |
| `B0+token probe − B0+probe` | score-then-pool against pool-then-score |

| labelled prompts | rmd − qmd | wins | sign p | qmd − token probe | wins | sign p | qmd − B0 |
|---:|---|---:|---:|---|---:|---:|---|
| 25 | -0.015 [-0.034, 0.018] | 0.70 | 0.344 | 0.008 [-0.028, 0.050] | 0.50 | 1.000 | 0.006 [-0.035, 0.074] |
| 50 | -0.009 [-0.024, 0.001] | 0.80 | 0.109 | -0.008 [-0.118, 0.004] | 0.70 | 0.344 | -0.046 [-0.069, -0.028] |
| 100 | -0.001 [-0.015, 0.007] | 0.50 | 1.000 | 0.004 [-0.031, 0.018] | 0.40 | 0.754 | -0.063 [-0.078, -0.026] |

### Crossing

Median `B0_rmd − B0_probe` crosses zero at about **95 labelled prompts** (bracketed by [50, 100]); geometry is the better feature below this budget.

## Across the models

Every label draw from all 3 models, pooled per budget. Cells read `median delta · draws on that side · sign p`; negative favours the left readout except in the last column, where positive favours geometry. `agree` counts models whose own median lands on the geometry side of `B0+rmd − B0+probe`.

The models are separate datasets so the direction pools, but the draws inside a model share an evaluation set, so the pooled `p` is a summary of consistency and not a test on independent observations. `agree` is the statistic that does not depend on that assumption.

| labelled prompts | `B0+rmd − B0` | `B0+probe − B0` | `B0+rmd − B0+probe` | agree | `B0+rmd − B0+token probe` | agree | `B0+both − B0+rmd` | AUROC rmd − probe |
|---:|---|---|---|---:|---|---:|---|---|
| 25 | -0.017 · 22/30 · p=0.016 | -0.006 · 18/30 · p=0.362 | -0.018 · 22/30 · p=0.016 | 3/3 | -0.012 · 20/30 · p=0.099 | 1/3 | -0.000 · 16/30 · p=0.856 | 0.018 · 18/30 · p=0.362 |
| 50 | -0.057 · 30/30 · p=0.000 | -0.042 · 25/30 · p=0.000 | -0.017 · 23/30 · p=0.005 | 3/3 | -0.033 · 26/30 · p=0.000 | 3/3 | -0.001 · 18/30 · p=0.362 | -0.010 · 12/30 · p=0.362 |
| 100 | -0.058 · 29/30 · p=0.000 | -0.059 · 30/30 · p=0.000 | 0.004 · 12/30 · p=0.362 | 1/3 | -0.002 · 17/30 · p=0.585 | 2/3 | -0.011 · 22/30 · p=0.016 | -0.011 · 10/30 · p=0.099 |

Pooled supervision ladder. `rmd − qmd` holds pooling order *and* the decision function fixed, so it is the only column in this report where supervision moves alone.

| labelled prompts | `B0+rmd − B0+qmd` | agree | `B0+qmd − B0+token probe` | agree | `B0+qmd − B0` | AUROC rmd − qmd |
|---:|---|---:|---|---:|---|---|
| 25 | -0.014 · 21/30 · p=0.043 | 2/3 | -0.006 · 19/30 · p=0.200 | 2/3 | -0.001 · 15/30 · p=1.000 | 0.018 · 24/30 · p=0.001 |
| 50 | -0.011 · 24/30 · p=0.001 | 3/3 | -0.018 · 23/30 · p=0.005 | 3/3 | -0.039 · 26/30 · p=0.000 | 0.014 · 18/30 · p=0.362 |
| 100 | 0.000 · 15/30 · p=1.000 | 1/3 | 0.000 · 14/30 · p=0.856 | 1/3 | -0.058 · 28/30 · p=0.000 | -0.004 · 12/30 · p=0.362 |

## Scope

- Reference fits see a fixed per-trace token subsample (`max_tokens_per_trace`), not the whole sequence. The frozen pipeline caps pooled reference tokens instead; this is the same device moved per trace so the token count still grows with the budget. Applied identically to both features.
- Only the 20% tail block is retained, which is lossless for both features under comparison and for nothing else.
- The PCA solver is pinned to `randomized`. The frozen helper picks it by token count, switching to `full` below 200k pooled tokens -- a threshold that falls inside this sweep, so leaving it alone would put a change of decomposition in the middle of the curve.
- The evaluation set is small by construction: the largest budget takes most of the prompts, and what is left is what can be scored. The spread quoted is over label draws, which is the relevant variation, but it carries that evaluation noise inside it.
- Numbers here are not interchangeable with the frozen artifacts.

