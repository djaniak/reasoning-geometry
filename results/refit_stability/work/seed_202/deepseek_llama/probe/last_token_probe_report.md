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
| deepseek_llama | `last_token_probe` | 0.8958 | [0.8706, 0.9147] | 0.6907 | 0.6824 | 0.6651 | 0.2135 | [0.1713, 0.2576] |
| deepseek_llama | `length` | 0.5196 | [0.4853, 0.5595] | 0.4242 | 0.4219 | 0.4227 | 0.0977 | [0.0430, 0.1560] |
| deepseek_llama | `mean_logprob` | 0.5088 | [0.4697, 0.5465] | 0.4352 | 0.4470 | 0.4360 | 0.0618 | [0.0005, 0.1156] |
| deepseek_llama | `mean_entropy` | 0.5119 | [0.4717, 0.5505] | 0.4346 | 0.4457 | 0.4349 | 0.0662 | [0.0076, 0.1210] |
| deepseek_llama | `rmd_tail_q20` | 0.6814 | [0.6478, 0.7159] | 0.5544 | 0.5505 | 0.5351 | 0.1309 | [0.0752, 0.1793] |
| deepseek_llama | `probe_hidden_tail_q20` | 0.7743 | [0.7408, 0.8023] | 0.6351 | 0.6238 | 0.6071 | 0.1504 | [0.0999, 0.2019] |

## 3. Population `all_traces`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| deepseek_llama | 4000 | 500 | 182 | 318 | 2017 | 0.552 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| deepseek_llama | `last_token_probe` | 0.9130 | [0.8916, 0.9297] | 0.7665 | 0.7731 | 0.7467 | 0.1399 | [0.1053, 0.1762] |
| deepseek_llama | `length` | 0.5808 | [0.5389, 0.6150] | 0.5297 | 0.5253 | 0.5057 | 0.0555 | [0.0002, 0.1208] |
| deepseek_llama | `mean_logprob` | 0.5445 | [0.5109, 0.5757] | 0.4799 | 0.4830 | 0.4704 | 0.0615 | [0.0123, 0.1092] |
| deepseek_llama | `mean_entropy` | 0.5456 | [0.5111, 0.5769] | 0.4760 | 0.4782 | 0.4696 | 0.0674 | [0.0163, 0.1137] |
| deepseek_llama | `rmd_tail_q20` | 0.7210 | [0.6863, 0.7547] | 0.6356 | 0.6383 | 0.6240 | 0.0827 | [0.0344, 0.1346] |
| deepseek_llama | `probe_hidden_tail_q20` | 0.7744 | [0.7397, 0.7995] | 0.6470 | 0.6474 | 0.6195 | 0.1270 | [0.0787, 0.1643] |

## 4. Layer selection inside training data

Mean inner-fold pooled AUROC per layer, per outer fold. Selection never sees the outer test prompts.

| Model | Population | Fold | Layer | C | Best inner pooled AUROC by layer |
|---|---|---:|---:|---:|---|
| deepseek_llama | `parseable` | 0 | 16 | 0.001 | L8 0.8969, L16 0.9000, L24 0.8897 |
| deepseek_llama | `parseable` | 1 | 16 | 0.003 | L8 0.9046, L16 0.9102, L24 0.9070 |
| deepseek_llama | `parseable` | 2 | 16 | 0.003 | L8 0.8826, L16 0.8875, L24 0.8791 |
| deepseek_llama | `parseable` | 3 | 16 | 0.001 | L8 0.8877, L16 0.8901, L24 0.8864 |
| deepseek_llama | `parseable` | 4 | 16 | 0.001 | L8 0.9028, L16 0.9111, L24 0.9042 |
| deepseek_llama | `all_traces` | 0 | 16 | 0.001 | L8 0.9032, L16 0.9090, L24 0.9066 |
| deepseek_llama | `all_traces` | 1 | 16 | 0.003 | L8 0.9197, L16 0.9215, L24 0.9185 |
| deepseek_llama | `all_traces` | 2 | 16 | 0.001 | L8 0.8989, L16 0.9018, L24 0.8956 |
| deepseek_llama | `all_traces` | 3 | 16 | 0.001 | L8 0.8992, L16 0.9030, L24 0.8996 |
| deepseek_llama | `all_traces` | 4 | 16 | 0.003 | L8 0.9159, L16 0.9219, L24 0.9163 |

## What this does and does not establish

The probe is fitted at the strength in-fold selection gives it, not at a fixed penalty. That matters: a loose penalty separates the training set perfectly and costs several points of held-out AUROC, which would understate the very claim this is meant to reproduce before decomposing.

It establishes what happens to a published-style pooled trace AUROC when prompt identity is conditioned on, under the same protocol that produced it. It does not establish that any particular published number is wrong: the models, datasets, and training populations differ. The claim is about the claim shape.

The intervals resample prompts with the fit held fixed. They do not carry the uncertainty of fold assignment, layer choice, or coefficients; that is the outer-refit blocker.
