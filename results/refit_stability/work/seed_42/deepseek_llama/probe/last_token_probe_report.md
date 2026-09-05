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
| deepseek_llama | `last_token_probe` | 0.9032 | [0.8834, 0.9238] | 0.7145 | 0.7177 | 0.6812 | 0.1855 | [0.1443, 0.2309] |
| deepseek_llama | `length` | 0.5196 | [0.4803, 0.5683] | 0.4242 | 0.4219 | 0.4227 | 0.0977 | [0.0339, 0.1674] |
| deepseek_llama | `mean_logprob` | 0.5088 | [0.4728, 0.5451] | 0.4352 | 0.4470 | 0.4360 | 0.0618 | [0.0056, 0.1119] |
| deepseek_llama | `mean_entropy` | 0.5119 | [0.4767, 0.5479] | 0.4346 | 0.4457 | 0.4349 | 0.0662 | [0.0078, 0.1222] |
| deepseek_llama | `rmd_tail_q20` | 0.6902 | [0.6593, 0.7278] | 0.5562 | 0.5541 | 0.5312 | 0.1361 | [0.0810, 0.1875] |
| deepseek_llama | `probe_hidden_tail_q20` | 0.7691 | [0.7395, 0.8036] | 0.6149 | 0.6148 | 0.5927 | 0.1543 | [0.0980, 0.2052] |

## 3. Population `all_traces`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| deepseek_llama | 4000 | 500 | 182 | 318 | 2017 | 0.552 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| deepseek_llama | `last_token_probe` | 0.9103 | [0.8915, 0.9276] | 0.7531 | 0.7580 | 0.7379 | 0.1523 | [0.1146, 0.1912] |
| deepseek_llama | `length` | 0.5808 | [0.5424, 0.6222] | 0.5297 | 0.5253 | 0.5057 | 0.0555 | [-0.0159, 0.1178] |
| deepseek_llama | `mean_logprob` | 0.5445 | [0.5077, 0.5824] | 0.4799 | 0.4830 | 0.4704 | 0.0615 | [0.0216, 0.1118] |
| deepseek_llama | `mean_entropy` | 0.5456 | [0.5090, 0.5830] | 0.4760 | 0.4782 | 0.4696 | 0.0674 | [0.0258, 0.1172] |
| deepseek_llama | `rmd_tail_q20` | 0.7287 | [0.6916, 0.7603] | 0.6376 | 0.6416 | 0.6199 | 0.0871 | [0.0377, 0.1386] |
| deepseek_llama | `probe_hidden_tail_q20` | 0.7708 | [0.7425, 0.7959] | 0.6292 | 0.6338 | 0.6020 | 0.1370 | [0.0897, 0.1810] |

## 4. Layer selection inside training data

Mean inner-fold pooled AUROC per layer, per outer fold. Selection never sees the outer test prompts.

| Model | Population | Fold | Layer | C | Best inner pooled AUROC by layer |
|---|---|---:|---:|---:|---|
| deepseek_llama | `parseable` | 0 | 16 | 0.003 | L8 0.8919, L16 0.9011, L24 0.8903 |
| deepseek_llama | `parseable` | 1 | 16 | 0.001 | L8 0.9044, L16 0.9081, L24 0.9015 |
| deepseek_llama | `parseable` | 2 | 16 | 0.001 | L8 0.8895, L16 0.8947, L24 0.8892 |
| deepseek_llama | `parseable` | 3 | 16 | 0.001 | L8 0.8919, L16 0.8937, L24 0.8937 |
| deepseek_llama | `parseable` | 4 | 16 | 0.001 | L8 0.8863, L16 0.8896, L24 0.8858 |
| deepseek_llama | `all_traces` | 0 | 16 | 0.001 | L8 0.9006, L16 0.9105, L24 0.9047 |
| deepseek_llama | `all_traces` | 1 | 16 | 0.003 | L8 0.9078, L16 0.9119, L24 0.9020 |
| deepseek_llama | `all_traces` | 2 | 16 | 0.001 | L8 0.8933, L16 0.8983, L24 0.8974 |
| deepseek_llama | `all_traces` | 3 | 16 | 0.001 | L8 0.9113, L16 0.9126, L24 0.9107 |
| deepseek_llama | `all_traces` | 4 | 16 | 0.003 | L8 0.9140, L16 0.9173, L24 0.9113 |

## What this does and does not establish

The probe is fitted at the strength in-fold selection gives it, not at a fixed penalty. That matters: a loose penalty separates the training set perfectly and costs several points of held-out AUROC, which would understate the very claim this is meant to reproduce before decomposing.

It establishes what happens to a published-style pooled trace AUROC when prompt identity is conditioned on, under the same protocol that produced it. It does not establish that any particular published number is wrong: the models, datasets, and training populations differ. The claim is about the claim shape.

The intervals resample prompts with the fit held fixed. They do not carry the uncertainty of fold assignment, layer choice, or coefficients; that is the outer-refit blocker.
