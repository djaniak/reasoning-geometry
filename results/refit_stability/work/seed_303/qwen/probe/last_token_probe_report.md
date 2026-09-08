# The last-token probe, reproduced and decomposed

Built by `controls/last_token_probe.py`. Last-token hidden state, L2 logistic readout, prompt-disjoint outer folds; layer *and* penalty chosen inside each training split by prompt-disjoint inner folds.

Three readouts, reported on each population. `pooled` counts every held-out trace and ignores prompt identity -- it is the published claim shape. `micro` weights every within-prompt (correct, incorrect) pair; `macro` averages per-prompt AUROC with each prompt counting once. Prompts with a single outcome contribute traces to `pooled` but no pairs to `micro` and no term to `macro`, so the three numbers are not defined on the same population. That is the finding, not a caveat.

## 1. What was fitted

| Model | Layers offered | Layers chosen | C chosen | Hidden dim | Traces | Unparsed |
|---|---|---|---|---:|---:|---:|
| qwen | 7, 14, 21 | 7 | 0.003 | 3584 | 4000 | 328 |

## 2. Population `parseable`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 3672 | 498 | 117 | 381 | 1104 | 0.606 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| qwen | `last_token_probe` | 0.9057 | [0.8783, 0.9279] | 0.5879 | 0.5960 | 0.6110 | 0.3098 | [0.2526, 0.3530] |
| qwen | `length` | 0.6775 | [0.6311, 0.7171] | 0.4534 | 0.4776 | 0.4745 | 0.1999 | [0.1355, 0.2682] |
| qwen | `mean_logprob` | 0.5983 | [0.5611, 0.6407] | 0.6540 | 0.6494 | 0.6087 | -0.0511 | [-0.1152, 0.0176] |
| qwen | `mean_entropy` | 0.5951 | [0.5559, 0.6384] | 0.6676 | 0.6602 | 0.6107 | -0.0651 | [-0.1286, 0.0065] |
| qwen | `rmd_tail_q20` | 0.8063 | [0.7700, 0.8400] | 0.5897 | 0.6175 | 0.5778 | 0.1888 | [0.1344, 0.2518] |
| qwen | `probe_hidden_tail_q20` | 0.8429 | [0.8148, 0.8742] | 0.5290 | 0.5284 | 0.5387 | 0.3145 | [0.2515, 0.3918] |

## 3. Population `all_traces`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 4000 | 500 | 131 | 369 | 1451 | 0.557 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| qwen | `last_token_probe` | 0.9193 | [0.9038, 0.9382] | 0.6954 | 0.6975 | 0.6953 | 0.2218 | [0.1682, 0.2732] |
| qwen | `length` | 0.7367 | [0.6992, 0.7738] | 0.5820 | 0.5807 | 0.5629 | 0.1560 | [0.0920, 0.2251] |
| qwen | `mean_logprob` | 0.5747 | [0.5308, 0.6150] | 0.5892 | 0.5947 | 0.5594 | -0.0200 | [-0.0828, 0.0405] |
| qwen | `mean_entropy` | 0.5708 | [0.5298, 0.6118] | 0.5948 | 0.5992 | 0.5588 | -0.0283 | [-0.0911, 0.0378] |
| qwen | `rmd_tail_q20` | 0.8388 | [0.8072, 0.8679] | 0.6644 | 0.6731 | 0.6392 | 0.1657 | [0.1241, 0.2150] |
| qwen | `probe_hidden_tail_q20` | 0.8401 | [0.8152, 0.8663] | 0.5376 | 0.5372 | 0.5435 | 0.3028 | [0.2483, 0.3622] |

## 4. Layer selection inside training data

Mean inner-fold pooled AUROC per layer, per outer fold. Selection never sees the outer test prompts.

| Model | Population | Fold | Layer | C | Best inner pooled AUROC by layer |
|---|---|---:|---:|---:|---|
| qwen | `parseable` | 0 | 7 | 0.003 | L7 0.8929, L14 0.8825, L21 0.8808 |
| qwen | `parseable` | 1 | 7 | 0.003 | L7 0.8974, L14 0.8844, L21 0.8889 |
| qwen | `parseable` | 2 | 7 | 0.003 | L7 0.8923, L14 0.8883, L21 0.8888 |
| qwen | `parseable` | 3 | 7 | 0.003 | L7 0.9062, L14 0.8938, L21 0.8924 |
| qwen | `parseable` | 4 | 7 | 0.003 | L7 0.9044, L14 0.8897, L21 0.8910 |
| qwen | `all_traces` | 0 | 7 | 0.003 | L7 0.9150, L14 0.9066, L21 0.9046 |
| qwen | `all_traces` | 1 | 7 | 0.003 | L7 0.9222, L14 0.9153, L21 0.9128 |
| qwen | `all_traces` | 2 | 7 | 0.003 | L7 0.9055, L14 0.8963, L21 0.8946 |
| qwen | `all_traces` | 3 | 7 | 0.003 | L7 0.9182, L14 0.9121, L21 0.9106 |
| qwen | `all_traces` | 4 | 7 | 0.003 | L7 0.9158, L14 0.9069, L21 0.9067 |

## What this does and does not establish

The probe is fitted at the strength in-fold selection gives it, not at a fixed penalty. That matters: a loose penalty separates the training set perfectly and costs several points of held-out AUROC, which would understate the very claim this is meant to reproduce before decomposing.

It establishes what happens to a published-style pooled trace AUROC when prompt identity is conditioned on, under the same protocol that produced it. It does not establish that any particular published number is wrong: the models, datasets, and training populations differ. The claim is about the claim shape.

The intervals resample prompts with the fit held fixed. They do not carry the uncertainty of fold assignment, layer choice, or coefficients; that is the outer-refit blocker.
