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
| deepseek | `last_token_probe` | 0.9015 | [0.8668, 0.9315] | 0.6455 | 0.6647 | 0.6369 | 0.2368 | [0.1331, 0.3449] |
| deepseek | `length` | 0.5916 | [0.5456, 0.6385] | 0.4364 | 0.4365 | 0.4410 | 0.1551 | [0.0329, 0.2662] |
| deepseek | `mean_logprob` | 0.5428 | [0.4989, 0.5946] | 0.4768 | 0.4784 | 0.4616 | 0.0644 | [-0.0376, 0.1758] |
| deepseek | `mean_entropy` | 0.5423 | [0.4991, 0.5949] | 0.4621 | 0.4684 | 0.4588 | 0.0739 | [-0.0241, 0.1859] |
| deepseek | `rmd_tail_q20` | 0.6771 | [0.6299, 0.7222] | 0.4719 | 0.4730 | 0.4678 | 0.2041 | [0.1073, 0.3071] |
| deepseek | `probe_hidden_tail_q20` | 0.7830 | [0.7439, 0.8156] | 0.5892 | 0.6210 | 0.5524 | 0.1620 | [0.0746, 0.2538] |

## 3. Population `all_traces`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| deepseek | 4000 | 500 | 102 | 398 | 1060 | 0.701 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| deepseek | `last_token_probe` | 0.9336 | [0.9128, 0.9505] | 0.8623 | 0.8667 | 0.8742 | 0.0669 | [0.0195, 0.1104] |
| deepseek | `length` | 0.7013 | [0.6542, 0.7435] | 0.7321 | 0.7311 | 0.6400 | -0.0298 | [-0.1105, 0.0300] |
| deepseek | `mean_logprob` | 0.6109 | [0.5721, 0.6525] | 0.5925 | 0.5957 | 0.5585 | 0.0151 | [-0.0515, 0.0902] |
| deepseek | `mean_entropy` | 0.6094 | [0.5713, 0.6524] | 0.5868 | 0.5932 | 0.5612 | 0.0163 | [-0.0519, 0.0916] |
| deepseek | `rmd_tail_q20` | 0.7631 | [0.7243, 0.8003] | 0.7679 | 0.7669 | 0.7419 | -0.0037 | [-0.0651, 0.0578] |
| deepseek | `probe_hidden_tail_q20` | 0.7574 | [0.7163, 0.7893] | 0.5491 | 0.5631 | 0.5389 | 0.1942 | [0.1265, 0.2795] |

## 4. Layer selection inside training data

Mean inner-fold pooled AUROC per layer, per outer fold. Selection never sees the outer test prompts.

| Model | Population | Fold | Layer | C | Best inner pooled AUROC by layer |
|---|---|---:|---:|---:|---|
| deepseek | `parseable` | 0 | 7 | 0.001 | L7 0.9089, L14 0.9031, L21 0.9069 |
| deepseek | `parseable` | 1 | 7 | 0.003 | L7 0.9150, L14 0.9143, L21 0.9107 |
| deepseek | `parseable` | 2 | 14 | 0.003 | L7 0.9259, L14 0.9291, L21 0.9188 |
| deepseek | `parseable` | 3 | 7 | 0.01 | L7 0.9445, L14 0.9407, L21 0.9340 |
| deepseek | `parseable` | 4 | 14 | 0.001 | L7 0.9136, L14 0.9167, L21 0.9163 |
| deepseek | `all_traces` | 0 | 7 | 0.01 | L7 0.9326, L14 0.9300, L21 0.9270 |
| deepseek | `all_traces` | 1 | 21 | 0.003 | L7 0.9347, L14 0.9385, L21 0.9402 |
| deepseek | `all_traces` | 2 | 7 | 0.01 | L7 0.9529, L14 0.9486, L21 0.9501 |
| deepseek | `all_traces` | 3 | 7 | 0.003 | L7 0.9445, L14 0.9383, L21 0.9412 |
| deepseek | `all_traces` | 4 | 14 | 0.003 | L7 0.9295, L14 0.9363, L21 0.9358 |

## What this does and does not establish

The probe is fitted at the strength in-fold selection gives it, not at a fixed penalty. That matters: a loose penalty separates the training set perfectly and costs several points of held-out AUROC, which would understate the very claim this is meant to reproduce before decomposing.

It establishes what happens to a published-style pooled trace AUROC when prompt identity is conditioned on, under the same protocol that produced it. It does not establish that any particular published number is wrong: the models, datasets, and training populations differ. The claim is about the claim shape.

The intervals resample prompts with the fit held fixed. They do not carry the uncertainty of fold assignment, layer choice, or coefficients; that is the outer-refit blocker.
