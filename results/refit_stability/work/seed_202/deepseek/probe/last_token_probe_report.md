# The last-token probe, reproduced and decomposed

Built by `controls/last_token_probe.py`. Last-token hidden state, L2 logistic readout, prompt-disjoint outer folds; layer *and* penalty chosen inside each training split by prompt-disjoint inner folds.

Three readouts, reported on each population. `pooled` counts every held-out trace and ignores prompt identity -- it is the published claim shape. `micro` weights every within-prompt (correct, incorrect) pair; `macro` averages per-prompt AUROC with each prompt counting once. Prompts with a single outcome contribute traces to `pooled` but no pairs to `micro` and no term to `macro`, so the three numbers are not defined on the same population. That is the finding, not a caveat.

## 1. What was fitted

| Model | Layers offered | Layers chosen | C chosen | Hidden dim | Traces | Unparsed |
|---|---|---|---|---:|---:|---:|
| deepseek | 7, 14, 21 | 7, 14, 21 | 0.001, 0.003, 0.01 | 3584 | 4000 | 351 |

## 2. Population `parseable`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| deepseek | 3649 | 493 | 49 | 444 | 409 | 0.768 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| deepseek | `last_token_probe` | 0.9201 | [0.8928, 0.9467] | 0.6846 | 0.6859 | 0.6629 | 0.2342 | [0.1442, 0.3360] |
| deepseek | `length` | 0.5916 | [0.5450, 0.6476] | 0.4364 | 0.4365 | 0.4410 | 0.1551 | [0.0464, 0.2514] |
| deepseek | `mean_logprob` | 0.5428 | [0.4977, 0.5872] | 0.4768 | 0.4784 | 0.4616 | 0.0644 | [-0.0444, 0.1615] |
| deepseek | `mean_entropy` | 0.5423 | [0.4951, 0.5884] | 0.4621 | 0.4684 | 0.4588 | 0.0739 | [-0.0435, 0.1695] |
| deepseek | `rmd_tail_q20` | 0.6816 | [0.6336, 0.7195] | 0.4597 | 0.4586 | 0.4641 | 0.2230 | [0.1286, 0.3080] |
| deepseek | `probe_hidden_tail_q20` | 0.7870 | [0.7422, 0.8232] | 0.5232 | 0.5362 | 0.5196 | 0.2508 | [0.1551, 0.3446] |

## 3. Population `all_traces`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| deepseek | 4000 | 500 | 102 | 398 | 1060 | 0.701 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| deepseek | `last_token_probe` | 0.9340 | [0.9099, 0.9519] | 0.8642 | 0.8683 | 0.8727 | 0.0657 | [0.0225, 0.1068] |
| deepseek | `length` | 0.7013 | [0.6585, 0.7381] | 0.7321 | 0.7311 | 0.6400 | -0.0298 | [-0.1000, 0.0488] |
| deepseek | `mean_logprob` | 0.6109 | [0.5739, 0.6452] | 0.5925 | 0.5957 | 0.5585 | 0.0151 | [-0.0583, 0.0759] |
| deepseek | `mean_entropy` | 0.6094 | [0.5724, 0.6450] | 0.5868 | 0.5932 | 0.5612 | 0.0163 | [-0.0531, 0.0788] |
| deepseek | `rmd_tail_q20` | 0.7661 | [0.7303, 0.7995] | 0.7623 | 0.7602 | 0.7414 | 0.0059 | [-0.0541, 0.0726] |
| deepseek | `probe_hidden_tail_q20` | 0.7651 | [0.7224, 0.8038] | 0.5472 | 0.5497 | 0.5389 | 0.2154 | [0.1523, 0.2890] |

## 4. Layer selection inside training data

Mean inner-fold pooled AUROC per layer, per outer fold. Selection never sees the outer test prompts.

| Model | Population | Fold | Layer | C | Best inner pooled AUROC by layer |
|---|---|---:|---:|---:|---|
| deepseek | `parseable` | 0 | 14 | 0.001 | L7 0.9129, L14 0.9157, L21 0.9153 |
| deepseek | `parseable` | 1 | 7 | 0.003 | L7 0.9312, L14 0.9226, L21 0.9212 |
| deepseek | `parseable` | 2 | 14 | 0.001 | L7 0.9079, L14 0.9134, L21 0.9108 |
| deepseek | `parseable` | 3 | 7 | 0.001 | L7 0.9036, L14 0.9016, L21 0.9020 |
| deepseek | `parseable` | 4 | 7 | 0.003 | L7 0.9346, L14 0.9310, L21 0.9279 |
| deepseek | `all_traces` | 0 | 14 | 0.003 | L7 0.9308, L14 0.9409, L21 0.9399 |
| deepseek | `all_traces` | 1 | 7 | 0.01 | L7 0.9463, L14 0.9439, L21 0.9416 |
| deepseek | `all_traces` | 2 | 21 | 0.003 | L7 0.9342, L14 0.9302, L21 0.9368 |
| deepseek | `all_traces` | 3 | 21 | 0.003 | L7 0.9354, L14 0.9387, L21 0.9393 |
| deepseek | `all_traces` | 4 | 7 | 0.01 | L7 0.9469, L14 0.9377, L21 0.9395 |

## What this does and does not establish

The probe is fitted at the strength in-fold selection gives it, not at a fixed penalty. That matters: a loose penalty separates the training set perfectly and costs several points of held-out AUROC, which would understate the very claim this is meant to reproduce before decomposing.

It establishes what happens to a published-style pooled trace AUROC when prompt identity is conditioned on, under the same protocol that produced it. It does not establish that any particular published number is wrong: the models, datasets, and training populations differ. The claim is about the claim shape.

The intervals resample prompts with the fit held fixed. They do not carry the uncertainty of fold assignment, layer choice, or coefficients; that is the outer-refit blocker.
