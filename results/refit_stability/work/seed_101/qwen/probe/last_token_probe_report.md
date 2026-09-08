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
| qwen | `last_token_probe` | 0.8951 | [0.8713, 0.9169] | 0.6024 | 0.6209 | 0.6154 | 0.2742 | [0.2140, 0.3362] |
| qwen | `length` | 0.6775 | [0.6344, 0.7109] | 0.4534 | 0.4776 | 0.4745 | 0.1999 | [0.1250, 0.2839] |
| qwen | `mean_logprob` | 0.5983 | [0.5593, 0.6442] | 0.6540 | 0.6494 | 0.6087 | -0.0511 | [-0.1227, 0.0207] |
| qwen | `mean_entropy` | 0.5951 | [0.5536, 0.6415] | 0.6676 | 0.6602 | 0.6107 | -0.0651 | [-0.1355, 0.0099] |
| qwen | `rmd_tail_q20` | 0.8032 | [0.7689, 0.8356] | 0.5797 | 0.6120 | 0.5766 | 0.1913 | [0.1281, 0.2494] |
| qwen | `probe_hidden_tail_q20` | 0.8414 | [0.8067, 0.8736] | 0.5498 | 0.5564 | 0.5460 | 0.2849 | [0.2266, 0.3431] |

## 3. Population `all_traces`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 4000 | 500 | 131 | 369 | 1451 | 0.557 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| qwen | `last_token_probe` | 0.9186 | [0.8994, 0.9350] | 0.6947 | 0.6948 | 0.6925 | 0.2238 | [0.1674, 0.2710] |
| qwen | `length` | 0.7367 | [0.7019, 0.7731] | 0.5820 | 0.5807 | 0.5629 | 0.1560 | [0.0910, 0.2344] |
| qwen | `mean_logprob` | 0.5747 | [0.5371, 0.6132] | 0.5892 | 0.5947 | 0.5594 | -0.0200 | [-0.0762, 0.0427] |
| qwen | `mean_entropy` | 0.5708 | [0.5290, 0.6103] | 0.5948 | 0.5992 | 0.5588 | -0.0283 | [-0.0818, 0.0321] |
| qwen | `rmd_tail_q20` | 0.8362 | [0.8053, 0.8621] | 0.6630 | 0.6716 | 0.6411 | 0.1646 | [0.1082, 0.2132] |
| qwen | `probe_hidden_tail_q20` | 0.8386 | [0.8091, 0.8680] | 0.5541 | 0.5621 | 0.5487 | 0.2765 | [0.2304, 0.3293] |

## 4. Layer selection inside training data

Mean inner-fold pooled AUROC per layer, per outer fold. Selection never sees the outer test prompts.

| Model | Population | Fold | Layer | C | Best inner pooled AUROC by layer |
|---|---|---:|---:|---:|---|
| qwen | `parseable` | 0 | 21 | 0.003 | L7 0.8993, L14 0.8934, L21 0.9010 |
| qwen | `parseable` | 1 | 7 | 0.003 | L7 0.8936, L14 0.8923, L21 0.8921 |
| qwen | `parseable` | 2 | 7 | 0.003 | L7 0.8998, L14 0.8927, L21 0.8894 |
| qwen | `parseable` | 3 | 7 | 0.003 | L7 0.9001, L14 0.8925, L21 0.8942 |
| qwen | `parseable` | 4 | 7 | 0.003 | L7 0.9086, L14 0.8958, L21 0.8989 |
| qwen | `all_traces` | 0 | 7 | 0.003 | L7 0.9104, L14 0.9015, L21 0.9053 |
| qwen | `all_traces` | 1 | 7 | 0.003 | L7 0.9208, L14 0.9113, L21 0.9096 |
| qwen | `all_traces` | 2 | 7 | 0.003 | L7 0.9240, L14 0.9191, L21 0.9210 |
| qwen | `all_traces` | 3 | 7 | 0.003 | L7 0.9216, L14 0.9157, L21 0.9133 |
| qwen | `all_traces` | 4 | 7 | 0.003 | L7 0.9162, L14 0.9062, L21 0.9069 |

## What this does and does not establish

The probe is fitted at the strength in-fold selection gives it, not at a fixed penalty. That matters: a loose penalty separates the training set perfectly and costs several points of held-out AUROC, which would understate the very claim this is meant to reproduce before decomposing.

It establishes what happens to a published-style pooled trace AUROC when prompt identity is conditioned on, under the same protocol that produced it. It does not establish that any particular published number is wrong: the models, datasets, and training populations differ. The claim is about the claim shape.

The intervals resample prompts with the fit held fixed. They do not carry the uncertainty of fold assignment, layer choice, or coefficients; that is the outer-refit blocker.
