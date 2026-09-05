# The last-token probe, reproduced and decomposed

Built by `controls/last_token_probe.py`. Last-token hidden state, L2 logistic readout, prompt-disjoint outer folds; layer *and* penalty chosen inside each training split by prompt-disjoint inner folds.

Three readouts, reported on each population. `pooled` counts every held-out trace and ignores prompt identity -- it is the published claim shape. `micro` weights every within-prompt (correct, incorrect) pair; `macro` averages per-prompt AUROC with each prompt counting once. Prompts with a single outcome contribute traces to `pooled` but no pairs to `micro` and no term to `macro`, so the three numbers are not defined on the same population. That is the finding, not a caveat.

## 1. What was fitted

| Model | Layers offered | Layers chosen | C chosen | Hidden dim | Traces | Unparsed |
|---|---|---|---|---:|---:|---:|
| deepseek_llama | 8, 16, 24 | 8, 16 | 0.001, 0.003 | 4096 | 4000 | 229 |

## 2. Population `parseable`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| deepseek_llama | 3771 | 499 | 158 | 341 | 1636 | 0.585 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| deepseek_llama | `last_token_probe` | 0.8936 | [0.8743, 0.9131] | 0.7103 | 0.7132 | 0.6679 | 0.1804 | [0.1396, 0.2237] |
| deepseek_llama | `length` | 0.5196 | [0.4787, 0.5587] | 0.4242 | 0.4219 | 0.4227 | 0.0977 | [0.0368, 0.1540] |
| deepseek_llama | `mean_logprob` | 0.5088 | [0.4710, 0.5408] | 0.4352 | 0.4470 | 0.4360 | 0.0618 | [0.0158, 0.1060] |
| deepseek_llama | `mean_entropy` | 0.5119 | [0.4746, 0.5429] | 0.4346 | 0.4457 | 0.4349 | 0.0662 | [0.0190, 0.1174] |
| deepseek_llama | `rmd_tail_q20` | 0.6930 | [0.6574, 0.7242] | 0.5617 | 0.5557 | 0.5354 | 0.1373 | [0.0915, 0.1830] |
| deepseek_llama | `probe_hidden_tail_q20` | 0.7854 | [0.7564, 0.8140] | 0.6326 | 0.6251 | 0.6064 | 0.1603 | [0.1100, 0.2093] |

## 3. Population `all_traces`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| deepseek_llama | 4000 | 500 | 182 | 318 | 2017 | 0.552 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| deepseek_llama | `last_token_probe` | 0.9009 | [0.8781, 0.9200] | 0.7640 | 0.7667 | 0.7409 | 0.1342 | [0.0974, 0.1687] |
| deepseek_llama | `length` | 0.5808 | [0.5448, 0.6162] | 0.5297 | 0.5253 | 0.5057 | 0.0555 | [-0.0041, 0.1090] |
| deepseek_llama | `mean_logprob` | 0.5445 | [0.5074, 0.5723] | 0.4799 | 0.4830 | 0.4704 | 0.0615 | [0.0120, 0.1068] |
| deepseek_llama | `mean_entropy` | 0.5456 | [0.5094, 0.5739] | 0.4760 | 0.4782 | 0.4696 | 0.0674 | [0.0180, 0.1128] |
| deepseek_llama | `rmd_tail_q20` | 0.7311 | [0.6983, 0.7646] | 0.6425 | 0.6431 | 0.6248 | 0.0880 | [0.0453, 0.1409] |
| deepseek_llama | `probe_hidden_tail_q20` | 0.7848 | [0.7602, 0.8089] | 0.6475 | 0.6459 | 0.6173 | 0.1390 | [0.0951, 0.1854] |

## 4. Layer selection inside training data

Mean inner-fold pooled AUROC per layer, per outer fold. Selection never sees the outer test prompts.

| Model | Population | Fold | Layer | C | Best inner pooled AUROC by layer |
|---|---|---:|---:|---:|---|
| deepseek_llama | `parseable` | 0 | 8 | 0.001 | L8 0.9061, L16 0.9033, L24 0.8977 |
| deepseek_llama | `parseable` | 1 | 16 | 0.001 | L8 0.8928, L16 0.8954, L24 0.8896 |
| deepseek_llama | `parseable` | 2 | 8 | 0.001 | L8 0.8925, L16 0.8909, L24 0.8900 |
| deepseek_llama | `parseable` | 3 | 16 | 0.001 | L8 0.8935, L16 0.9074, L24 0.8964 |
| deepseek_llama | `parseable` | 4 | 16 | 0.001 | L8 0.8910, L16 0.8914, L24 0.8820 |
| deepseek_llama | `all_traces` | 0 | 16 | 0.001 | L8 0.9051, L16 0.9135, L24 0.9110 |
| deepseek_llama | `all_traces` | 1 | 16 | 0.003 | L8 0.9051, L16 0.9110, L24 0.9104 |
| deepseek_llama | `all_traces` | 2 | 8 | 0.003 | L8 0.9166, L16 0.9160, L24 0.9110 |
| deepseek_llama | `all_traces` | 3 | 16 | 0.003 | L8 0.9062, L16 0.9161, L24 0.9033 |
| deepseek_llama | `all_traces` | 4 | 16 | 0.001 | L8 0.9093, L16 0.9116, L24 0.9092 |

## What this does and does not establish

The probe is fitted at the strength in-fold selection gives it, not at a fixed penalty. That matters: a loose penalty separates the training set perfectly and costs several points of held-out AUROC, which would understate the very claim this is meant to reproduce before decomposing.

It establishes what happens to a published-style pooled trace AUROC when prompt identity is conditioned on, under the same protocol that produced it. It does not establish that any particular published number is wrong: the models, datasets, and training populations differ. The claim is about the claim shape.

The intervals resample prompts with the fit held fixed. They do not carry the uncertainty of fold assignment, layer choice, or coefficients; that is the outer-refit blocker.
