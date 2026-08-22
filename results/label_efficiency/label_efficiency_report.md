# Label-efficiency curves: one-class geometry versus a supervised probe

`rmd_tail_q20` fits a Gaussian on correct traces only; `probe_hidden_tail_q20` fits an LDA on both classes over the same PCA-projected tail means. At the full label budget the probe is ahead, so the only deployment claim geometry can carry is that it needs fewer labels. These curves are that claim, or its refutation.

At each budget the PCA basis, the correct-trace Gaussian, the background Gaussian, the LDA, and the logistic readout are all refitted from that budget's prompts alone. Training sets are nested along one permutation per replicate; the evaluation set is the headline-population complement of the largest budget and is held fixed across budgets, so every number in a row is scored on the same prompts.

## qwen

Layer 21, PCA 128, 500 prompts (392 in the `cap_free_valid_plurality` evaluation pool), 10 label draws, 3 inner folds on the training side, 256 reference tokens per trace.

### Feature AUROC against the budget

Prompt-level AUROC of each feature alone, the base-rate-free view. Median over label draws, 10--90 band.

| labelled prompts | `rmd_tail_q20` | `probe_hidden_tail_q20` | rmd − probe |
|---:|---|---|---|
| 25 | 0.659 [0.605, 0.839] | 0.688 [0.577, 0.776] | 0.026 [-0.049, 0.067] |
| 50 | 0.775 [0.656, 0.822] | 0.746 [0.670, 0.810] | 0.011 [-0.054, 0.083] |
| 100 | 0.787 [0.694, 0.833] | 0.772 [0.714, 0.830] | 0.022 [-0.065, 0.060] |
| 200 | 0.817 [0.717, 0.853] | 0.845 [0.720, 0.883] | -0.018 [-0.080, 0.042] |
| 400 | 0.815 [0.774, 0.840] | 0.861 [0.808, 0.888] | -0.044 [-0.066, -0.006] |

### AURC against the budget

Lower is better. `excess` subtracts the chance AURC `(1 − 1/n) − base`, which is what makes a level comparable at all.

| labelled prompts | eval n | base acc | B0 | B0+rmd | B0+probe | excess B0+rmd | excess B0+probe |
|---:|---:|---|---|---|---|---|---|
| 25 | 76 | 0.690 | 0.229 [0.143, 0.261] | 0.204 [0.101, 0.245] | 0.221 [0.127, 0.294] | -0.109 [-0.152, -0.020] | -0.086 [-0.141, 0.033] |
| 50 | 76 | 0.690 | 0.200 [0.148, 0.266] | 0.169 [0.110, 0.218] | 0.169 [0.122, 0.242] | -0.129 [-0.158, -0.085] | -0.118 [-0.161, -0.048] |
| 100 | 76 | 0.690 | 0.199 [0.142, 0.235] | 0.128 [0.096, 0.200] | 0.142 [0.126, 0.207] | -0.147 [-0.176, -0.109] | -0.138 [-0.169, -0.093] |
| 200 | 76 | 0.690 | 0.198 [0.141, 0.227] | 0.110 [0.096, 0.170] | 0.114 [0.091, 0.170] | -0.161 [-0.194, -0.125] | -0.176 [-0.195, -0.109] |
| 400 | 76 | 0.690 | 0.192 [0.150, 0.230] | 0.106 [0.092, 0.163] | 0.103 [0.092, 0.155] | -0.154 [-0.197, -0.137] | -0.172 [-0.206, -0.146] |

### Paired AURC deltas against the budget

Paired inside a replicate: identical evaluation prompts, identical labelled prompts, one logistic each. Negative favours the left readout. `wins` is the share of label draws landing on that side.

| labelled prompts | B0+rmd − B0 | B0+probe − B0 | B0+rmd − B0+probe | wins | sign p |
|---:|---|---|---|---:|---:|
| 25 | -0.019 [-0.096, 0.009] | -0.004 [-0.046, 0.013] | -0.019 [-0.056, 0.008] | 0.70 | 0.344 |
| 50 | -0.039 [-0.070, -0.004] | -0.021 [-0.072, 0.024] | -0.018 [-0.053, 0.011] | 0.80 | 0.109 |
| 100 | -0.058 [-0.086, -0.003] | -0.046 [-0.067, -0.000] | -0.011 [-0.056, 0.039] | 0.60 | 0.754 |
| 200 | -0.075 [-0.094, -0.021] | -0.069 [-0.092, -0.039] | -0.001 [-0.032, 0.038] | 0.50 | 1.000 |
| 400 | -0.069 [-0.093, -0.048] | -0.084 [-0.110, -0.036] | 0.006 [-0.016, 0.043] | 0.40 | 0.754 |

### Crossing

Median `B0_rmd − B0_probe` crosses zero at about **226 labelled prompts** (bracketed by [200, 400]); geometry is the better feature below this budget.

## deepseek_qwen

Layer 21, PCA 128, 500 prompts (393 in the `cap_free_valid_plurality` evaluation pool), 10 label draws, 3 inner folds on the training side, 256 reference tokens per trace.

### Feature AUROC against the budget

Prompt-level AUROC of each feature alone, the base-rate-free view. Median over label draws, 10--90 band.

| labelled prompts | `rmd_tail_q20` | `probe_hidden_tail_q20` | rmd − probe |
|---:|---|---|---|
| 25 | 0.649 [0.509, 0.805] | 0.641 [0.542, 0.727] | 0.023 [-0.093, 0.101] |
| 50 | 0.689 [0.610, 0.755] | 0.724 [0.622, 0.757] | -0.052 [-0.109, 0.066] |
| 100 | 0.714 [0.629, 0.756] | 0.728 [0.652, 0.789] | 0.016 [-0.149, 0.089] |
| 200 | 0.727 [0.670, 0.769] | 0.777 [0.711, 0.857] | -0.058 [-0.129, 0.025] |
| 400 | 0.731 [0.680, 0.775] | 0.795 [0.731, 0.866] | -0.060 [-0.140, -0.024] |

### AURC against the budget

Lower is better. `excess` subtracts the chance AURC `(1 − 1/n) − base`, which is what makes a level comparable at all.

| labelled prompts | eval n | base acc | B0 | B0+rmd | B0+probe | excess B0+rmd | excess B0+probe |
|---:|---:|---|---|---|---|---|---|
| 25 | 78 | 0.805 | 0.129 [0.115, 0.204] | 0.119 [0.060, 0.205] | 0.162 [0.128, 0.218] | -0.048 [-0.124, 0.020] | -0.018 [-0.062, 0.019] |
| 50 | 78 | 0.805 | 0.137 [0.101, 0.216] | 0.116 [0.074, 0.160] | 0.146 [0.065, 0.192] | -0.070 [-0.117, -0.027] | -0.050 [-0.107, -0.000] |
| 100 | 78 | 0.805 | 0.125 [0.110, 0.168] | 0.122 [0.081, 0.132] | 0.088 [0.070, 0.126] | -0.078 [-0.114, -0.041] | -0.084 [-0.133, -0.063] |
| 200 | 78 | 0.805 | 0.117 [0.107, 0.160] | 0.066 [0.049, 0.143] | 0.066 [0.047, 0.111] | -0.108 [-0.142, -0.091] | -0.105 [-0.162, -0.068] |
| 400 | 78 | 0.805 | 0.121 [0.113, 0.160] | 0.077 [0.056, 0.113] | 0.068 [0.040, 0.101] | -0.109 [-0.136, -0.078] | -0.107 [-0.163, -0.083] |

### Paired AURC deltas against the budget

Paired inside a replicate: identical evaluation prompts, identical labelled prompts, one logistic each. Negative favours the left readout. `wins` is the share of label draws landing on that side.

| labelled prompts | B0+rmd − B0 | B0+probe − B0 | B0+rmd − B0+probe | wins | sign p |
|---:|---|---|---|---:|---:|
| 25 | -0.014 [-0.057, 0.000] | 0.013 [-0.007, 0.031] | -0.022 [-0.088, -0.004] | 0.90 | 0.021 |
| 50 | -0.026 [-0.047, -0.005] | -0.009 [-0.046, 0.028] | -0.006 [-0.069, 0.013] | 0.60 | 0.754 |
| 100 | -0.022 [-0.039, 0.002] | -0.036 [-0.064, -0.024] | 0.015 [-0.007, 0.043] | 0.20 | 0.109 |
| 200 | -0.049 [-0.068, -0.027] | -0.052 [-0.082, -0.001] | 0.002 [-0.027, 0.023] | 0.40 | 0.754 |
| 400 | -0.046 [-0.058, -0.035] | -0.052 [-0.083, -0.039] | 0.003 [-0.011, 0.034] | 0.30 | 0.344 |

### Crossing

Median `B0_rmd − B0_probe` crosses zero at about **60 labelled prompts** (bracketed by [50, 100]); geometry is the better feature below this budget.

## deepseek_llama

Layer 24, PCA 128, 500 prompts (408 in the `cap_free_valid_plurality` evaluation pool), 10 label draws, 3 inner folds on the training side, 256 reference tokens per trace.

### Feature AUROC against the budget

Prompt-level AUROC of each feature alone, the base-rate-free view. Median over label draws, 10--90 band.

| labelled prompts | `rmd_tail_q20` | `probe_hidden_tail_q20` | rmd − probe |
|---:|---|---|---|
| 25 | 0.658 [0.521, 0.694] | 0.625 [0.591, 0.734] | -0.011 [-0.081, 0.062] |
| 50 | 0.666 [0.603, 0.730] | 0.696 [0.627, 0.750] | -0.013 [-0.136, 0.063] |
| 100 | 0.696 [0.605, 0.726] | 0.718 [0.657, 0.773] | -0.020 [-0.150, 0.038] |
| 200 | 0.724 [0.686, 0.760] | 0.777 [0.724, 0.814] | -0.056 [-0.093, 0.004] |
| 400 | 0.754 [0.704, 0.773] | 0.790 [0.764, 0.821] | -0.041 [-0.105, -0.008] |

### AURC against the budget

Lower is better. `excess` subtracts the chance AURC `(1 − 1/n) − base`, which is what makes a level comparable at all.

| labelled prompts | eval n | base acc | B0 | B0+rmd | B0+probe | excess B0+rmd | excess B0+probe |
|---:|---:|---|---|---|---|---|---|
| 25 | 82 | 0.681 | 0.239 [0.159, 0.314] | 0.225 [0.178, 0.379] | 0.234 [0.165, 0.338] | -0.067 [-0.108, 0.028] | -0.066 [-0.117, 0.005] |
| 50 | 82 | 0.681 | 0.219 [0.160, 0.267] | 0.186 [0.113, 0.247] | 0.196 [0.144, 0.260] | -0.125 [-0.144, -0.076] | -0.084 [-0.133, -0.075] |
| 100 | 82 | 0.681 | 0.191 [0.159, 0.253] | 0.169 [0.128, 0.210] | 0.167 [0.135, 0.187] | -0.142 [-0.155, -0.106] | -0.127 [-0.172, -0.115] |
| 200 | 82 | 0.681 | 0.198 [0.152, 0.238] | 0.154 [0.111, 0.195] | 0.139 [0.126, 0.174] | -0.151 [-0.167, -0.136] | -0.150 [-0.200, -0.131] |
| 400 | 82 | 0.681 | 0.203 [0.139, 0.259] | 0.148 [0.120, 0.188] | 0.144 [0.110, 0.158] | -0.153 [-0.176, -0.125] | -0.158 [-0.195, -0.140] |

### Paired AURC deltas against the budget

Paired inside a replicate: identical evaluation prompts, identical labelled prompts, one logistic each. Negative favours the left readout. `wins` is the share of label draws landing on that side.

| labelled prompts | B0+rmd − B0 | B0+probe − B0 | B0+rmd − B0+probe | wins | sign p |
|---:|---|---|---|---:|---:|
| 25 | -0.004 [-0.040, 0.059] | -0.007 [-0.011, 0.016] | -0.011 [-0.030, 0.064] | 0.70 | 0.344 |
| 50 | -0.044 [-0.065, -0.005] | -0.017 [-0.068, 0.012] | -0.012 [-0.058, 0.016] | 0.60 | 0.754 |
| 100 | -0.033 [-0.056, -0.009] | -0.023 [-0.089, 0.001] | -0.005 [-0.029, 0.037] | 0.70 | 0.344 |
| 200 | -0.047 [-0.076, -0.020] | -0.057 [-0.089, -0.004] | 0.011 [-0.028, 0.046] | 0.30 | 0.344 |
| 400 | -0.058 [-0.122, 0.007] | -0.065 [-0.141, -0.016] | 0.009 [-0.008, 0.035] | 0.40 | 0.754 |

### Crossing

Median `B0_rmd − B0_probe` crosses zero at about **123 labelled prompts** (bracketed by [100, 200]); geometry is the better feature below this budget.

## Across the models

Every label draw from all 3 models, pooled per budget. Cells read `median delta · draws on that side · sign p`; negative favours the left readout except in the last column, where positive favours geometry. `agree` counts models whose own median lands on the geometry side of `B0+rmd − B0+probe`.

The models are separate datasets so the direction pools, but the draws inside a model share an evaluation set, so the pooled `p` is a summary of consistency and not a test on independent observations. `agree` is the statistic that does not depend on that assumption.

| labelled prompts | `B0+rmd − B0` | `B0+probe − B0` | `B0+rmd − B0+probe` | agree | `B0+both − B0+rmd` | AUROC rmd − probe |
|---:|---|---|---|---:|---|---|
| 25 | -0.014 · 23/30 · p=0.005 | 0.000 · 15/30 · p=1.000 | -0.019 · 23/30 · p=0.005 | 3/3 | 0.000 · 13/30 · p=0.585 | 0.023 · 17/30 · p=0.585 |
| 50 | -0.040 · 27/30 · p=0.000 | -0.015 · 23/30 · p=0.005 | -0.011 · 20/30 · p=0.099 | 3/3 | -0.001 · 16/30 · p=0.856 | -0.013 · 13/30 · p=0.585 |
| 100 | -0.033 · 26/30 · p=0.000 | -0.038 · 28/30 · p=0.000 | -0.000 · 15/30 · p=1.000 | 2/3 | -0.005 · 19/30 · p=0.200 | 0.012 · 16/30 · p=0.856 |
| 200 | -0.053 · 28/30 · p=0.000 | -0.055 · 28/30 · p=0.000 | 0.002 · 12/30 · p=0.362 | 0/3 | -0.006 · 21/30 · p=0.043 | -0.048 · 9/30 · p=0.043 |
| 400 | -0.052 · 28/30 · p=0.000 | -0.067 · 30/30 · p=0.000 | 0.006 · 11/30 · p=0.200 | 0/3 | -0.007 · 24/30 · p=0.001 | -0.044 · 2/30 · p=0.000 |

## Scope

- Reference fits see a fixed per-trace token subsample (`max_tokens_per_trace`), not the whole sequence. The frozen pipeline caps pooled reference tokens instead; this is the same device moved per trace so the token count still grows with the budget. Applied identically to both features.
- Only the 20% tail block is retained, which is lossless for both features under comparison and for nothing else.
- The PCA solver is pinned to `randomized`. The frozen helper picks it by token count, switching to `full` below 200k pooled tokens -- a threshold that falls inside this sweep, so leaving it alone would put a change of decomposition in the middle of the curve.
- The evaluation set is small by construction: the largest budget takes most of the prompts, and what is left is what can be scored. The spread quoted is over label draws, which is the relevant variation, but it carries that evaluation noise inside it.
- Numbers here are not interchangeable with the frozen artifacts.

