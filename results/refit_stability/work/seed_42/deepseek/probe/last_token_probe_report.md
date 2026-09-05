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
| deepseek | `last_token_probe` | 0.9139 | [0.8904, 0.9408] | 0.6015 | 0.5823 | 0.6188 | 0.3316 | [0.2308, 0.4479] |
| deepseek | `length` | 0.5916 | [0.5386, 0.6393] | 0.4364 | 0.4365 | 0.4410 | 0.1551 | [0.0410, 0.2708] |
| deepseek | `mean_logprob` | 0.5428 | [0.4986, 0.5939] | 0.4768 | 0.4784 | 0.4616 | 0.0644 | [-0.0488, 0.1800] |
| deepseek | `mean_entropy` | 0.5423 | [0.4986, 0.5926] | 0.4621 | 0.4684 | 0.4588 | 0.0739 | [-0.0410, 0.1930] |
| deepseek | `rmd_tail_q20` | 0.6748 | [0.6294, 0.7219] | 0.4621 | 0.4610 | 0.4669 | 0.2139 | [0.1151, 0.3171] |
| deepseek | `probe_hidden_tail_q20` | 0.7807 | [0.7416, 0.8238] | 0.5501 | 0.5714 | 0.5407 | 0.2094 | [0.1216, 0.3257] |

## 3. Population `all_traces`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| deepseek | 4000 | 500 | 102 | 398 | 1060 | 0.701 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| deepseek | `last_token_probe` | 0.9251 | [0.9041, 0.9449] | 0.8425 | 0.8504 | 0.8669 | 0.0747 | [0.0219, 0.1318] |
| deepseek | `length` | 0.7013 | [0.6598, 0.7442] | 0.7321 | 0.7311 | 0.6400 | -0.0298 | [-0.1014, 0.0434] |
| deepseek | `mean_logprob` | 0.6109 | [0.5713, 0.6509] | 0.5925 | 0.5957 | 0.5585 | 0.0151 | [-0.0495, 0.0779] |
| deepseek | `mean_entropy` | 0.6094 | [0.5707, 0.6481] | 0.5868 | 0.5932 | 0.5612 | 0.0163 | [-0.0454, 0.0783] |
| deepseek | `rmd_tail_q20` | 0.7616 | [0.7283, 0.8008] | 0.7642 | 0.7609 | 0.7415 | 0.0007 | [-0.0623, 0.0777] |
| deepseek | `probe_hidden_tail_q20` | 0.7511 | [0.7154, 0.7854] | 0.5425 | 0.5544 | 0.5365 | 0.1966 | [0.1283, 0.2846] |

## 4. Layer selection inside training data

Mean inner-fold pooled AUROC per layer, per outer fold. Selection never sees the outer test prompts.

| Model | Population | Fold | Layer | C | Best inner pooled AUROC by layer |
|---|---|---:|---:|---:|---|
| deepseek | `parseable` | 0 | 14 | 0.001 | L7 0.9083, L14 0.9098, L21 0.9052 |
| deepseek | `parseable` | 1 | 14 | 0.001 | L7 0.9113, L14 0.9212, L21 0.9143 |
| deepseek | `parseable` | 2 | 14 | 0.003 | L7 0.9268, L14 0.9288, L21 0.9220 |
| deepseek | `parseable` | 3 | 14 | 0.001 | L7 0.9040, L14 0.9060, L21 0.9035 |
| deepseek | `parseable` | 4 | 21 | 0.001 | L7 0.9063, L14 0.9097, L21 0.9114 |
| deepseek | `all_traces` | 0 | 14 | 0.001 | L7 0.9289, L14 0.9319, L21 0.9313 |
| deepseek | `all_traces` | 1 | 7 | 0.01 | L7 0.9343, L14 0.9280, L21 0.9329 |
| deepseek | `all_traces` | 2 | 14 | 0.001 | L7 0.9265, L14 0.9316, L21 0.9288 |
| deepseek | `all_traces` | 3 | 14 | 0.003 | L7 0.9366, L14 0.9421, L21 0.9372 |
| deepseek | `all_traces` | 4 | 21 | 0.01 | L7 0.9447, L14 0.9449, L21 0.9486 |

## What this does and does not establish

The probe is fitted at the strength in-fold selection gives it, not at a fixed penalty. That matters: a loose penalty separates the training set perfectly and costs several points of held-out AUROC, which would understate the very claim this is meant to reproduce before decomposing.

It establishes what happens to a published-style pooled trace AUROC when prompt identity is conditioned on, under the same protocol that produced it. It does not establish that any particular published number is wrong: the models, datasets, and training populations differ. The claim is about the claim shape.

The intervals resample prompts with the fit held fixed. They do not carry the uncertainty of fold assignment, layer choice, or coefficients; that is the outer-refit blocker.
