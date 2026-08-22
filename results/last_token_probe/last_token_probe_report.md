# The last-token probe, reproduced and decomposed

Built by `controls/last_token_probe.py`. Last-token hidden state, L2 logistic readout, prompt-disjoint outer folds; layer *and* penalty chosen inside each training split by prompt-disjoint inner folds.

Three readouts, reported on each population. `pooled` counts every held-out trace and ignores prompt identity -- it is the published claim shape. `micro` weights every within-prompt (correct, incorrect) pair; `macro` averages per-prompt AUROC with each prompt counting once. Prompts with a single outcome contribute traces to `pooled` but no pairs to `micro` and no term to `macro`, so the three numbers are not defined on the same population. That is the finding, not a caveat.

## 1. What was fitted

| Model | Layers offered | Layers chosen | C chosen | Hidden dim | Traces | Unparsed |
|---|---|---|---|---:|---:|---:|
| qwen | 7, 14, 21 | 7, 21 | 0.003 | 3584 | 4000 | 328 |
| deepseek | 7, 14, 21 | 7, 14, 21 | 0.001, 0.003, 0.01 | 3584 | 4000 | 351 |
| deepseek_llama | 8, 16, 24 | 16 | 0.001, 0.003 | 4096 | 4000 | 229 |

## 2. Population `parseable`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 3672 | 498 | 117 | 381 | 1104 | 0.606 |
| deepseek | 3649 | 493 | 49 | 444 | 409 | 0.768 |
| deepseek_llama | 3771 | 499 | 158 | 341 | 1636 | 0.585 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| qwen | `last_token_probe` | 0.9013 | [0.8776, 0.9211] | 0.6214 | 0.6444 | 0.6351 | 0.2569 | [0.1980, 0.3148] |
| qwen | `length` | 0.6775 | [0.6377, 0.7187] | 0.4534 | 0.4776 | 0.4745 | 0.1999 | [0.1330, 0.2674] |
| qwen | `mean_logprob` | 0.5983 | [0.5539, 0.6409] | 0.6540 | 0.6494 | 0.6087 | -0.0511 | [-0.1250, 0.0179] |
| qwen | `mean_entropy` | 0.5951 | [0.5492, 0.6402] | 0.6676 | 0.6602 | 0.6107 | -0.0651 | [-0.1383, 0.0033] |
| qwen | `rmd_tail_q20` | 0.8069 | [0.7749, 0.8398] | 0.5697 | 0.5984 | 0.5761 | 0.2085 | [0.1441, 0.2693] |
| qwen | `probe_hidden_tail_q20` | 0.8474 | [0.8179, 0.8735] | 0.5371 | 0.5460 | 0.5360 | 0.3014 | [0.2306, 0.3656] |
| deepseek | `last_token_probe` | 0.9139 | [0.8841, 0.9399] | 0.6015 | 0.5823 | 0.6188 | 0.3316 | [0.2260, 0.4499] |
| deepseek | `length` | 0.5916 | [0.5449, 0.6441] | 0.4364 | 0.4365 | 0.4410 | 0.1551 | [0.0418, 0.2730] |
| deepseek | `mean_logprob` | 0.5428 | [0.4961, 0.5931] | 0.4768 | 0.4784 | 0.4616 | 0.0644 | [-0.0406, 0.1710] |
| deepseek | `mean_entropy` | 0.5423 | [0.4945, 0.5929] | 0.4621 | 0.4684 | 0.4588 | 0.0739 | [-0.0363, 0.1819] |
| deepseek | `rmd_tail_q20` | 0.6748 | [0.6294, 0.7228] | 0.4621 | 0.4610 | 0.4669 | 0.2139 | [0.1091, 0.3171] |
| deepseek | `probe_hidden_tail_q20` | 0.7807 | [0.7396, 0.8196] | 0.5501 | 0.5714 | 0.5407 | 0.2094 | [0.1118, 0.3106] |
| deepseek_llama | `last_token_probe` | 0.9032 | [0.8807, 0.9225] | 0.7145 | 0.7177 | 0.6812 | 0.1855 | [0.1410, 0.2325] |
| deepseek_llama | `length` | 0.5196 | [0.4784, 0.5607] | 0.4242 | 0.4219 | 0.4227 | 0.0977 | [0.0326, 0.1611] |
| deepseek_llama | `mean_logprob` | 0.5088 | [0.4704, 0.5441] | 0.4352 | 0.4470 | 0.4360 | 0.0618 | [0.0064, 0.1131] |
| deepseek_llama | `mean_entropy` | 0.5119 | [0.4740, 0.5478] | 0.4346 | 0.4457 | 0.4349 | 0.0662 | [0.0113, 0.1169] |
| deepseek_llama | `rmd_tail_q20` | 0.6902 | [0.6583, 0.7245] | 0.5562 | 0.5541 | 0.5312 | 0.1361 | [0.0842, 0.1884] |
| deepseek_llama | `probe_hidden_tail_q20` | 0.7691 | [0.7395, 0.7981] | 0.6149 | 0.6148 | 0.5927 | 0.1543 | [0.1065, 0.2033] |

## 3. Population `all_traces`

| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 4000 | 500 | 131 | 369 | 1451 | 0.557 |
| deepseek | 4000 | 500 | 102 | 398 | 1060 | 0.701 |
| deepseek_llama | 4000 | 500 | 182 | 318 | 2017 | 0.552 |

### Pooled versus within-prompt

| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | Prompt-centered | Pooled - macro | 95% CI |
|---|---|---:|---|---:|---:|---:|---:|---|
| qwen | `last_token_probe` | 0.9211 | [0.9022, 0.9379] | 0.7023 | 0.7068 | 0.6962 | 0.2143 | [0.1666, 0.2669] |
| qwen | `length` | 0.7367 | [0.7012, 0.7716] | 0.5820 | 0.5807 | 0.5629 | 0.1560 | [0.0877, 0.2228] |
| qwen | `mean_logprob` | 0.5747 | [0.5338, 0.6140] | 0.5892 | 0.5947 | 0.5594 | -0.0200 | [-0.0807, 0.0392] |
| qwen | `mean_entropy` | 0.5708 | [0.5284, 0.6127] | 0.5948 | 0.5992 | 0.5588 | -0.0283 | [-0.0920, 0.0342] |
| qwen | `rmd_tail_q20` | 0.8393 | [0.8095, 0.8655] | 0.6527 | 0.6584 | 0.6396 | 0.1809 | [0.1282, 0.2363] |
| qwen | `probe_hidden_tail_q20` | 0.8459 | [0.8186, 0.8719] | 0.5424 | 0.5496 | 0.5433 | 0.2963 | [0.2359, 0.3554] |
| deepseek | `last_token_probe` | 0.9251 | [0.8982, 0.9464] | 0.8425 | 0.8504 | 0.8669 | 0.0747 | [0.0216, 0.1308] |
| deepseek | `length` | 0.7013 | [0.6574, 0.7417] | 0.7321 | 0.7311 | 0.6400 | -0.0298 | [-0.1017, 0.0442] |
| deepseek | `mean_logprob` | 0.6109 | [0.5713, 0.6505] | 0.5925 | 0.5957 | 0.5585 | 0.0151 | [-0.0546, 0.0783] |
| deepseek | `mean_entropy` | 0.6094 | [0.5702, 0.6489] | 0.5868 | 0.5932 | 0.5612 | 0.0163 | [-0.0559, 0.0803] |
| deepseek | `rmd_tail_q20` | 0.7616 | [0.7243, 0.7972] | 0.7642 | 0.7609 | 0.7415 | 0.0007 | [-0.0654, 0.0696] |
| deepseek | `probe_hidden_tail_q20` | 0.7511 | [0.7151, 0.7849] | 0.5425 | 0.5544 | 0.5365 | 0.1966 | [0.1220, 0.2682] |
| deepseek_llama | `last_token_probe` | 0.9103 | [0.8912, 0.9281] | 0.7531 | 0.7580 | 0.7379 | 0.1523 | [0.1152, 0.1922] |
| deepseek_llama | `length` | 0.5808 | [0.5411, 0.6211] | 0.5297 | 0.5253 | 0.5057 | 0.0555 | [-0.0118, 0.1181] |
| deepseek_llama | `mean_logprob` | 0.5445 | [0.5058, 0.5817] | 0.4799 | 0.4830 | 0.4704 | 0.0615 | [0.0084, 0.1148] |
| deepseek_llama | `mean_entropy` | 0.5456 | [0.5064, 0.5834] | 0.4760 | 0.4782 | 0.4696 | 0.0674 | [0.0141, 0.1186] |
| deepseek_llama | `rmd_tail_q20` | 0.7287 | [0.6973, 0.7618] | 0.6376 | 0.6416 | 0.6199 | 0.0871 | [0.0362, 0.1386] |
| deepseek_llama | `probe_hidden_tail_q20` | 0.7708 | [0.7409, 0.7994] | 0.6292 | 0.6338 | 0.6020 | 0.1370 | [0.0908, 0.1838] |

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
