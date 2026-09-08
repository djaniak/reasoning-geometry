# The last-token probe, reproduced and decomposed

Built by `controls/last_token_probe.py`. Last-token hidden state, L2 logistic readout, prompt-disjoint outer folds; layer *and* penalty chosen inside each training split by prompt-disjoint inner folds.

Three readouts, reported on each population. `pooled` counts every held-out trace and ignores prompt identity -- it is the published claim shape. `micro` weights every within-prompt (correct, incorrect) pair; `macro` averages per-prompt AUROC with each prompt counting once. Prompts with a single outcome contribute traces to `pooled` but no pairs to `micro` and no term to `macro`, so the three numbers are not defined on the same population. That is the finding, not a caveat.

## 1. What was fitted

| Model | Layers offered | Layers chosen | C chosen | Hidden dim | Traces | Unparsed |
|---|---|---|---|---:|---:|---:|
| deepseek_llama | 8, 16, 24 | 16 | 0.001, 0.003 | 4096 | 4000 | 229 |

## 2. Population `parseable`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| deepseek_llama | 3771 | 499 | 158 | 341 | 1636 | 0.585 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| deepseek_llama | `last_token_probe` | 0.9011 | [0.8793, 0.9213] | 0.6999 | 0.7014 | 0.6714 | 0.1997 | [0.1577, 0.2443] |
| deepseek_llama | `length` | 0.5196 | [0.4777, 0.5615] | 0.4242 | 0.4219 | 0.4227 | 0.0977 | [0.0252, 0.1587] |
| deepseek_llama | `mean_logprob` | 0.5088 | [0.4684, 0.5474] | 0.4352 | 0.4470 | 0.4360 | 0.0618 | [0.0139, 0.1117] |
| deepseek_llama | `mean_entropy` | 0.5119 | [0.4729, 0.5502] | 0.4346 | 0.4457 | 0.4349 | 0.0662 | [0.0193, 0.1193] |
| deepseek_llama | `rmd_tail_q20` | 0.6957 | [0.6579, 0.7317] | 0.5654 | 0.5613 | 0.5401 | 0.1344 | [0.0750, 0.1906] |
| deepseek_llama | `probe_hidden_tail_q20` | 0.7795 | [0.7532, 0.8084] | 0.6345 | 0.6323 | 0.6054 | 0.1472 | [0.0991, 0.1919] |

## 3. Population `all_traces`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| deepseek_llama | 4000 | 500 | 182 | 318 | 2017 | 0.552 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| deepseek_llama | `last_token_probe` | 0.9123 | [0.8928, 0.9284] | 0.7590 | 0.7622 | 0.7395 | 0.1501 | [0.1117, 0.1862] |
| deepseek_llama | `length` | 0.5808 | [0.5445, 0.6174] | 0.5297 | 0.5253 | 0.5057 | 0.0555 | [-0.0146, 0.1174] |
| deepseek_llama | `mean_logprob` | 0.5445 | [0.5054, 0.5777] | 0.4799 | 0.4830 | 0.4704 | 0.0615 | [-0.0034, 0.1110] |
| deepseek_llama | `mean_entropy` | 0.5456 | [0.5077, 0.5797] | 0.4760 | 0.4782 | 0.4696 | 0.0674 | [-0.0017, 0.1162] |
| deepseek_llama | `rmd_tail_q20` | 0.7333 | [0.7015, 0.7617] | 0.6450 | 0.6480 | 0.6274 | 0.0853 | [0.0312, 0.1227] |
| deepseek_llama | `probe_hidden_tail_q20` | 0.7777 | [0.7490, 0.8028] | 0.6495 | 0.6532 | 0.6173 | 0.1245 | [0.0798, 0.1686] |

## 4. Layer selection inside training data

Mean inner-fold pooled AUROC per layer, per outer fold. Selection never sees the outer test prompts.

| Model | Population | Fold | Layer | C | Best inner pooled AUROC by layer |
|---|---|---:|---:|---:|---|
| deepseek_llama | `parseable` | 0 | 16 | 0.001 | L8 0.8754, L16 0.8810, L24 0.8754 |
| deepseek_llama | `parseable` | 1 | 16 | 0.001 | L8 0.8868, L16 0.8986, L24 0.8916 |
| deepseek_llama | `parseable` | 2 | 16 | 0.001 | L8 0.9084, L16 0.9123, L24 0.9097 |
| deepseek_llama | `parseable` | 3 | 16 | 0.001 | L8 0.8977, L16 0.9032, L24 0.8925 |
| deepseek_llama | `parseable` | 4 | 16 | 0.003 | L8 0.8904, L16 0.8991, L24 0.8897 |
| deepseek_llama | `all_traces` | 0 | 16 | 0.001 | L8 0.9041, L16 0.9075, L24 0.9013 |
| deepseek_llama | `all_traces` | 1 | 16 | 0.003 | L8 0.9017, L16 0.9114, L24 0.9039 |
| deepseek_llama | `all_traces` | 2 | 16 | 0.001 | L8 0.9091, L16 0.9142, L24 0.9134 |
| deepseek_llama | `all_traces` | 3 | 16 | 0.001 | L8 0.9100, L16 0.9122, L24 0.9092 |
| deepseek_llama | `all_traces` | 4 | 16 | 0.001 | L8 0.8990, L16 0.9042, L24 0.8987 |

## What this does and does not establish

The probe is fitted at the strength in-fold selection gives it, not at a fixed penalty. That matters: a loose penalty separates the training set perfectly and costs several points of held-out AUROC, which would understate the very claim this is meant to reproduce before decomposing.

It establishes what happens to a published-style pooled trace AUROC when prompt identity is conditioned on, under the same protocol that produced it. It does not establish that any particular published number is wrong: the models, datasets, and training populations differ. The claim is about the claim shape.

The intervals resample prompts with the fit held fixed. They do not carry the uncertainty of fold assignment, layer choice, or coefficients; that is the outer-refit blocker.
