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
| deepseek | `last_token_probe` | 0.9229 | [0.8959, 0.9482] | 0.6626 | 0.6567 | 0.6747 | 0.2662 | [0.1759, 0.3779] |
| deepseek | `length` | 0.5916 | [0.5477, 0.6415] | 0.4364 | 0.4365 | 0.4410 | 0.1551 | [0.0439, 0.2642] |
| deepseek | `mean_logprob` | 0.5428 | [0.4992, 0.5931] | 0.4768 | 0.4784 | 0.4616 | 0.0644 | [-0.0312, 0.1632] |
| deepseek | `mean_entropy` | 0.5423 | [0.4976, 0.5938] | 0.4621 | 0.4684 | 0.4588 | 0.0739 | [-0.0368, 0.1950] |
| deepseek | `rmd_tail_q20` | 0.6845 | [0.6444, 0.7381] | 0.4621 | 0.4617 | 0.4641 | 0.2228 | [0.1282, 0.3286] |
| deepseek | `probe_hidden_tail_q20` | 0.7921 | [0.7555, 0.8307] | 0.5770 | 0.5861 | 0.5464 | 0.2060 | [0.1158, 0.3061] |

## 3. Population `all_traces`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| deepseek | 4000 | 500 | 102 | 398 | 1060 | 0.701 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| deepseek | `last_token_probe` | 0.9315 | [0.9126, 0.9504] | 0.8557 | 0.8609 | 0.8727 | 0.0706 | [0.0228, 0.1223] |
| deepseek | `length` | 0.7013 | [0.6544, 0.7482] | 0.7321 | 0.7311 | 0.6400 | -0.0298 | [-0.0999, 0.0495] |
| deepseek | `mean_logprob` | 0.6109 | [0.5675, 0.6504] | 0.5925 | 0.5957 | 0.5585 | 0.0151 | [-0.0606, 0.0780] |
| deepseek | `mean_entropy` | 0.6094 | [0.5673, 0.6483] | 0.5868 | 0.5932 | 0.5612 | 0.0163 | [-0.0549, 0.0804] |
| deepseek | `rmd_tail_q20` | 0.7684 | [0.7300, 0.8014] | 0.7642 | 0.7614 | 0.7410 | 0.0070 | [-0.0619, 0.0740] |
| deepseek | `probe_hidden_tail_q20` | 0.7640 | [0.7301, 0.7978] | 0.5415 | 0.5524 | 0.5401 | 0.2116 | [0.1493, 0.2737] |

## 4. Layer selection inside training data

Mean inner-fold pooled AUROC per layer, per outer fold. Selection never sees the outer test prompts.

| Model | Population | Fold | Layer | C | Best inner pooled AUROC by layer |
|---|---|---:|---:|---:|---|
| deepseek | `parseable` | 0 | 14 | 0.001 | L7 0.9242, L14 0.9277, L21 0.9206 |
| deepseek | `parseable` | 1 | 14 | 0.001 | L7 0.8912, L14 0.8947, L21 0.8845 |
| deepseek | `parseable` | 2 | 14 | 0.001 | L7 0.9129, L14 0.9161, L21 0.9138 |
| deepseek | `parseable` | 3 | 7 | 0.003 | L7 0.9163, L14 0.9091, L21 0.9101 |
| deepseek | `parseable` | 4 | 14 | 0.001 | L7 0.9151, L14 0.9188, L21 0.9169 |
| deepseek | `all_traces` | 0 | 7 | 0.003 | L7 0.9437, L14 0.9400, L21 0.9381 |
| deepseek | `all_traces` | 1 | 21 | 0.003 | L7 0.9400, L14 0.9413, L21 0.9414 |
| deepseek | `all_traces` | 2 | 7 | 0.003 | L7 0.9264, L14 0.9183, L21 0.9188 |
| deepseek | `all_traces` | 3 | 21 | 0.01 | L7 0.9463, L14 0.9468, L21 0.9490 |
| deepseek | `all_traces` | 4 | 14 | 0.003 | L7 0.9317, L14 0.9371, L21 0.9328 |

## What this does and does not establish

The probe is fitted at the strength in-fold selection gives it, not at a fixed penalty. That matters: a loose penalty separates the training set perfectly and costs several points of held-out AUROC, which would understate the very claim this is meant to reproduce before decomposing.

It establishes what happens to a published-style pooled trace AUROC when prompt identity is conditioned on, under the same protocol that produced it. It does not establish that any particular published number is wrong: the models, datasets, and training populations differ. The claim is about the claim shape.

The intervals resample prompts with the fit held fixed. They do not carry the uncertainty of fold assignment, layer choice, or coefficients; that is the outer-refit blocker.
