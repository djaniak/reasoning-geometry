# Is tail RMD a proxy for the vote it is scored against?

Direct answer to Orgad et al. (arXiv:2410.02707): hidden states encode the
resampling agreement structure, so a reviewer will read `rmd_tail_q20` as a
worse-instrumented `vote_agreement`. AURC is reported (lower is better);
AUROC is base-rate invariant and needs no such caveat.

## 1. Redundancy between the two features

| model | n | Pearson | Spearman | AUROC rmd_tail_q20 | AUROC vote_agreement |
|---|---:|---:|---:|---|---|
| DeepSeek-Qwen | 500 | 0.377 | 0.239 | 0.714 [0.665, 0.762] | 0.611 [0.566, 0.656] |
| Llama | 500 | 0.296 | 0.295 | 0.712 [0.666, 0.758] | 0.669 [0.624, 0.714] |
| Qwen | 500 | 0.491 | 0.477 | 0.819 [0.781, 0.854] | 0.684 [0.635, 0.727] |

## 2. Geometry inside a fixed level of agreement

Agreement does not vary within a stratum, so a proxy cannot separate anything
there. The unanimous stratum is where self-consistency has nothing left to say.

| model | stratum | n | base acc | AUROC rmd_tail_q20 |
|---|---|---:|---:|---|
| DeepSeek-Qwen | unanimous | 424 | 0.797 | 0.717 [0.660, 0.770] |
| DeepSeek-Qwen | split | 76 | 0.487 | 0.634 [0.506, 0.749] |
| Llama | unanimous | 253 | 0.771 | 0.739 [0.670, 0.802] |
| Llama | split | 247 | 0.494 | 0.656 [0.586, 0.719] |
| Qwen | unanimous | 302 | 0.732 | 0.827 [0.774, 0.876] |
| Qwen | split | 198 | 0.449 | 0.755 [0.690, 0.824] |

## 3. Orthogonal component (out-of-fold linear residual)

| model | AUROC of rmd_tail_q20 given vote | AUROC of vote given rmd_tail_q20 |
|---|---|---|
| DeepSeek-Qwen | 0.663 [0.611, 0.714] | 0.472 [0.404, 0.541] |
| Llama | 0.660 [0.611, 0.709] | 0.592 [0.538, 0.645] |
| Qwen | 0.739 [0.694, 0.783] | 0.522 [0.460, 0.575] |

## 4. Substitution, both directions (AURC, lower is better)

| model | B1 - B0 | rmd for vote - B0 | B1 - (rmd for vote) | rmd added to voteless |
|---|---|---|---|---|
| DeepSeek-Qwen | -0.0284 [-0.0531, -0.0060] p=0.008 | -0.0207 [-0.0490, +0.0055] p=0.110 | -0.0076 [-0.0207, +0.0046] p=0.234 | -0.0368 [-0.0627, -0.0119] p=0.004 |
| Llama | -0.0469 [-0.0773, -0.0178] p=0.006 | -0.0339 [-0.0726, +0.0024] p=0.076 | -0.0131 [-0.0250, -0.0026] p=0.016 | -0.1188 [-0.1682, -0.0717] p=0.000 |
| Qwen | -0.0520 [-0.0832, -0.0218] p=0.004 | -0.0480 [-0.0787, -0.0159] p=0.006 | -0.0040 [-0.0149, +0.0055] p=0.444 | -0.0550 [-0.0842, -0.0243] p=0.000 |

## 5. Agreement levels present

| model | agreement | n | accuracy |
|---|---:|---:|---:|
| DeepSeek-Qwen | 0.000 | 7 | 0.000 |
| DeepSeek-Qwen | 0.333 | 1 | 1.000 |
| DeepSeek-Qwen | 0.400 | 1 | 0.000 |
| DeepSeek-Qwen | 0.429 | 2 | 0.500 |
| DeepSeek-Qwen | 0.500 | 5 | 0.200 |
| DeepSeek-Qwen | 0.571 | 1 | 0.000 |
| DeepSeek-Qwen | 0.600 | 3 | 0.667 |
| DeepSeek-Qwen | 0.625 | 6 | 0.333 |
| DeepSeek-Qwen | 0.667 | 1 | 1.000 |
| DeepSeek-Qwen | 0.714 | 2 | 0.500 |
| DeepSeek-Qwen | 0.750 | 15 | 0.600 |
| DeepSeek-Qwen | 0.800 | 3 | 0.667 |
| DeepSeek-Qwen | 0.833 | 3 | 0.667 |
| DeepSeek-Qwen | 0.857 | 6 | 0.667 |
| DeepSeek-Qwen | 0.875 | 20 | 0.550 |
| DeepSeek-Qwen | 1.000 | 424 | 0.797 |
| Llama | 0.000 | 1 | 0.000 |
| Llama | 0.125 | 4 | 0.000 |
| Llama | 0.167 | 2 | 0.000 |
| Llama | 0.200 | 1 | 0.000 |
| Llama | 0.250 | 15 | 0.533 |
| Llama | 0.286 | 2 | 0.000 |
| Llama | 0.333 | 1 | 0.000 |
| Llama | 0.375 | 15 | 0.333 |
| Llama | 0.400 | 3 | 0.000 |
| Llama | 0.429 | 6 | 0.167 |
| Llama | 0.500 | 28 | 0.429 |
| Llama | 0.600 | 2 | 1.000 |
| Llama | 0.625 | 34 | 0.735 |
| Llama | 0.667 | 7 | 0.429 |
| Llama | 0.714 | 2 | 1.000 |
| Llama | 0.750 | 47 | 0.511 |
| Llama | 0.800 | 2 | 0.000 |
| Llama | 0.833 | 3 | 1.000 |
| Llama | 0.857 | 8 | 0.250 |
| Llama | 0.875 | 64 | 0.547 |
| Llama | 1.000 | 253 | 0.771 |
| Qwen | 0.000 | 2 | 0.000 |
| Qwen | 0.125 | 2 | 0.000 |
| Qwen | 0.167 | 2 | 0.000 |
| Qwen | 0.200 | 2 | 0.500 |
| Qwen | 0.250 | 10 | 0.000 |
| Qwen | 0.286 | 4 | 0.000 |
| Qwen | 0.333 | 7 | 0.143 |
| Qwen | 0.375 | 13 | 0.308 |
| Qwen | 0.400 | 2 | 0.000 |
| Qwen | 0.429 | 5 | 0.200 |
| Qwen | 0.500 | 32 | 0.406 |
| Qwen | 0.571 | 4 | 0.750 |
| Qwen | 0.600 | 8 | 0.250 |
| Qwen | 0.625 | 25 | 0.440 |
| Qwen | 0.667 | 5 | 0.600 |
| Qwen | 0.714 | 6 | 0.667 |
| Qwen | 0.750 | 22 | 0.636 |
| Qwen | 0.800 | 4 | 0.000 |
| Qwen | 0.833 | 2 | 1.000 |
| Qwen | 0.857 | 7 | 0.286 |
| Qwen | 0.875 | 34 | 0.824 |
| Qwen | 1.000 | 302 | 0.732 |
