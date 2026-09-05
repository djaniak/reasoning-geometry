# The last-token probe, reproduced and decomposed

Built by `controls/last_token_probe.py`. Last-token hidden state, L2 logistic readout, prompt-disjoint outer folds; layer *and* penalty chosen inside each training split by prompt-disjoint inner folds.

Three readouts, reported on each population. `pooled` counts every held-out trace and ignores prompt identity -- it is the published claim shape. `micro` weights every within-prompt (correct, incorrect) pair; `macro` averages per-prompt AUROC with each prompt counting once. Prompts with a single outcome contribute traces to `pooled` but no pairs to `micro` and no term to `macro`, so the three numbers are not defined on the same population. That is the finding, not a caveat.

## 1. What was fitted

| Model | Layers offered | Layers chosen | C chosen | Hidden dim | Traces | Unparsed |
|---|---|---|---|---:|---:|---:|
| qwen | 7, 14, 21 | 7, 21 | 0.003 | 3584 | 4000 | 328 |

## 2. Population `parseable`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 3672 | 498 | 117 | 381 | 1104 | 0.606 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| qwen | `last_token_probe` | 0.9013 | [0.8775, 0.9197] | 0.6214 | 0.6444 | 0.6351 | 0.2569 | [0.2022, 0.3112] |
| qwen | `length` | 0.6775 | [0.6415, 0.7201] | 0.4534 | 0.4776 | 0.4745 | 0.1999 | [0.1377, 0.2651] |
| qwen | `mean_logprob` | 0.5983 | [0.5563, 0.6397] | 0.6540 | 0.6494 | 0.6087 | -0.0511 | [-0.1182, 0.0145] |
| qwen | `mean_entropy` | 0.5951 | [0.5511, 0.6418] | 0.6676 | 0.6602 | 0.6107 | -0.0651 | [-0.1349, 0.0063] |
| qwen | `rmd_tail_q20` | 0.8069 | [0.7753, 0.8353] | 0.5697 | 0.5984 | 0.5761 | 0.2085 | [0.1470, 0.2670] |
| qwen | `probe_hidden_tail_q20` | 0.8474 | [0.8187, 0.8714] | 0.5371 | 0.5460 | 0.5360 | 0.3014 | [0.2275, 0.3607] |

## 3. Population `all_traces`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 4000 | 500 | 131 | 369 | 1451 | 0.557 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| qwen | `last_token_probe` | 0.9211 | [0.9031, 0.9379] | 0.7023 | 0.7068 | 0.6962 | 0.2143 | [0.1712, 0.2780] |
| qwen | `length` | 0.7367 | [0.7009, 0.7733] | 0.5820 | 0.5807 | 0.5629 | 0.1560 | [0.0857, 0.2223] |
| qwen | `mean_logprob` | 0.5747 | [0.5341, 0.6187] | 0.5892 | 0.5947 | 0.5594 | -0.0200 | [-0.0816, 0.0458] |
| qwen | `mean_entropy` | 0.5708 | [0.5308, 0.6141] | 0.5948 | 0.5992 | 0.5588 | -0.0283 | [-0.0927, 0.0325] |
| qwen | `rmd_tail_q20` | 0.8393 | [0.8094, 0.8658] | 0.6527 | 0.6584 | 0.6396 | 0.1809 | [0.1305, 0.2454] |
| qwen | `probe_hidden_tail_q20` | 0.8459 | [0.8192, 0.8706] | 0.5424 | 0.5496 | 0.5433 | 0.2963 | [0.2408, 0.3605] |

## 4. Layer selection inside training data

Mean inner-fold pooled AUROC per layer, per outer fold. Selection never sees the outer test prompts.

| Model | Population | Fold | Layer | C | Best inner pooled AUROC by layer |
|---|---|---:|---:|---:|---|
| qwen | `parseable` | 0 | 21 | 0.003 | L7 0.8927, L14 0.8875, L21 0.8929 |
| qwen | `parseable` | 1 | 7 | 0.003 | L7 0.8949, L14 0.8822, L21 0.8812 |
| qwen | `parseable` | 2 | 7 | 0.003 | L7 0.9003, L14 0.8880, L21 0.8968 |
| qwen | `parseable` | 3 | 7 | 0.003 | L7 0.9053, L14 0.8939, L21 0.8930 |
| qwen | `parseable` | 4 | 7 | 0.003 | L7 0.9058, L14 0.8985, L21 0.8870 |
| qwen | `all_traces` | 0 | 7 | 0.003 | L7 0.9108, L14 0.9023, L21 0.9028 |
| qwen | `all_traces` | 1 | 7 | 0.003 | L7 0.9124, L14 0.9044, L21 0.9001 |
| qwen | `all_traces` | 2 | 7 | 0.003 | L7 0.9207, L14 0.9067, L21 0.9154 |
| qwen | `all_traces` | 3 | 7 | 0.003 | L7 0.9159, L14 0.9111, L21 0.9143 |
| qwen | `all_traces` | 4 | 7 | 0.003 | L7 0.9199, L14 0.9174, L21 0.9114 |

## What this does and does not establish

The probe is fitted at the strength in-fold selection gives it, not at a fixed penalty. That matters: a loose penalty separates the training set perfectly and costs several points of held-out AUROC, which would understate the very claim this is meant to reproduce before decomposing.

It establishes what happens to a published-style pooled trace AUROC when prompt identity is conditioned on, under the same protocol that produced it. It does not establish that any particular published number is wrong: the models, datasets, and training populations differ. The claim is about the claim shape.

The intervals resample prompts with the fit held fixed. They do not carry the uncertainty of fold assignment, layer choice, or coefficients; that is the outer-refit blocker.
