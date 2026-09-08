# The last-token probe, reproduced and decomposed

Built by `controls/last_token_probe.py`. Last-token hidden state, L2 logistic readout, prompt-disjoint outer folds; layer *and* penalty chosen inside each training split by prompt-disjoint inner folds.

Three readouts, reported on each population. `pooled` counts every held-out trace and ignores prompt identity -- it is the published claim shape. `micro` weights every within-prompt (correct, incorrect) pair; `macro` averages per-prompt AUROC with each prompt counting once. Prompts with a single outcome contribute traces to `pooled` but no pairs to `micro` and no term to `macro`, so the three numbers are not defined on the same population. That is the finding, not a caveat.

## 1. What was fitted

| Model | Layers offered | Layers chosen | C chosen | Hidden dim | Traces | Unparsed |
|---|---|---|---|---:|---:|---:|
| qwen | 7, 14, 21 | 7 | 0.003, 0.01 | 3584 | 4000 | 328 |

## 2. Population `parseable`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 3672 | 498 | 117 | 381 | 1104 | 0.606 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| qwen | `last_token_probe` | 0.9039 | [0.8821, 0.9245] | 0.5960 | 0.6140 | 0.6253 | 0.2899 | [0.2261, 0.3477] |
| qwen | `length` | 0.6775 | [0.6300, 0.7151] | 0.4534 | 0.4776 | 0.4745 | 0.1999 | [0.1234, 0.2642] |
| qwen | `mean_logprob` | 0.5983 | [0.5564, 0.6396] | 0.6540 | 0.6494 | 0.6087 | -0.0511 | [-0.1257, 0.0202] |
| qwen | `mean_entropy` | 0.5951 | [0.5538, 0.6407] | 0.6676 | 0.6602 | 0.6107 | -0.0651 | [-0.1389, -0.0018] |
| qwen | `rmd_tail_q20` | 0.8082 | [0.7773, 0.8401] | 0.5942 | 0.6238 | 0.5775 | 0.1844 | [0.1174, 0.2483] |
| qwen | `probe_hidden_tail_q20` | 0.8475 | [0.8190, 0.8743] | 0.5580 | 0.5602 | 0.5584 | 0.2873 | [0.2208, 0.3607] |

## 3. Population `all_traces`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 4000 | 500 | 131 | 369 | 1451 | 0.557 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| qwen | `last_token_probe` | 0.9203 | [0.8952, 0.9363] | 0.6975 | 0.7000 | 0.6941 | 0.2203 | [0.1829, 0.2720] |
| qwen | `length` | 0.7367 | [0.7005, 0.7666] | 0.5820 | 0.5807 | 0.5629 | 0.1560 | [0.0926, 0.2129] |
| qwen | `mean_logprob` | 0.5747 | [0.5294, 0.6116] | 0.5892 | 0.5947 | 0.5594 | -0.0200 | [-0.0871, 0.0449] |
| qwen | `mean_entropy` | 0.5708 | [0.5250, 0.6099] | 0.5948 | 0.5992 | 0.5588 | -0.0283 | [-0.0895, 0.0373] |
| qwen | `rmd_tail_q20` | 0.8402 | [0.8128, 0.8640] | 0.6726 | 0.6799 | 0.6436 | 0.1602 | [0.1044, 0.2146] |
| qwen | `probe_hidden_tail_q20` | 0.8435 | [0.8163, 0.8694] | 0.5637 | 0.5696 | 0.5622 | 0.2739 | [0.2147, 0.3374] |

## 4. Layer selection inside training data

Mean inner-fold pooled AUROC per layer, per outer fold. Selection never sees the outer test prompts.

| Model | Population | Fold | Layer | C | Best inner pooled AUROC by layer |
|---|---|---:|---:|---:|---|
| qwen | `parseable` | 0 | 7 | 0.003 | L7 0.9097, L14 0.9014, L21 0.9044 |
| qwen | `parseable` | 1 | 7 | 0.003 | L7 0.9003, L14 0.8838, L21 0.8883 |
| qwen | `parseable` | 2 | 7 | 0.003 | L7 0.8931, L14 0.8875, L21 0.8831 |
| qwen | `parseable` | 3 | 7 | 0.003 | L7 0.8947, L14 0.8878, L21 0.8924 |
| qwen | `parseable` | 4 | 7 | 0.003 | L7 0.9084, L14 0.8932, L21 0.8905 |
| qwen | `all_traces` | 0 | 7 | 0.003 | L7 0.9153, L14 0.9045, L21 0.9047 |
| qwen | `all_traces` | 1 | 7 | 0.01 | L7 0.9343, L14 0.9256, L21 0.9238 |
| qwen | `all_traces` | 2 | 7 | 0.003 | L7 0.9092, L14 0.8950, L21 0.9037 |
| qwen | `all_traces` | 3 | 7 | 0.003 | L7 0.9101, L14 0.9044, L21 0.9025 |
| qwen | `all_traces` | 4 | 7 | 0.003 | L7 0.9199, L14 0.9084, L21 0.9124 |

## What this does and does not establish

The probe is fitted at the strength in-fold selection gives it, not at a fixed penalty. That matters: a loose penalty separates the training set perfectly and costs several points of held-out AUROC, which would understate the very claim this is meant to reproduce before decomposing.

It establishes what happens to a published-style pooled trace AUROC when prompt identity is conditioned on, under the same protocol that produced it. It does not establish that any particular published number is wrong: the models, datasets, and training populations differ. The claim is about the claim shape.

The intervals resample prompts with the fit held fixed. They do not carry the uncertainty of fold assignment, layer choice, or coefficients; that is the outer-refit blocker.
