# Is tail RMD a proxy for the vote it is scored against?

Direct answer to Orgad et al. (arXiv:2410.02707): hidden states encode the
resampling agreement structure, so a reviewer will read `rmd_tail_q20` as a
worse-instrumented `vote_agreement`. AURC is reported (lower is better);
AUROC is base-rate invariant and needs no such caveat.

## 1. Redundancy between the two features

| model | n | Pearson | Spearman | AUROC rmd_tail_q20 | AUROC vote_agreement |
|---|---:|---:|---:|---|---|
| DeepSeek-Qwen | 393 | 0.105 | 0.098 | 0.686 [0.620, 0.750] | 0.587 [0.541, 0.639] |
| Llama | 408 | 0.274 | 0.287 | 0.709 [0.660, 0.760] | 0.650 [0.599, 0.706] |
| Qwen | 392 | 0.361 | 0.325 | 0.806 [0.758, 0.853] | 0.634 [0.580, 0.689] |

## 2. Geometry inside a fixed level of agreement

Agreement does not vary within a stratum, so a proxy cannot separate anything
there. The unanimous stratum is where self-consistency has nothing left to say.

| model | stratum | n | base acc | AUROC rmd_tail_q20 |
|---|---|---:|---:|---|
| DeepSeek-Qwen | unanimous | 349 | 0.828 | 0.714 [0.648, 0.784] |
| DeepSeek-Qwen | split | 44 | 0.545 | 0.531 [0.369, 0.701] |
| Llama | unanimous | 214 | 0.794 | 0.756 [0.685, 0.830] |
| Llama | split | 194 | 0.541 | 0.636 [0.554, 0.713] |
| Qwen | unanimous | 274 | 0.755 | 0.829 [0.771, 0.884] |
| Qwen | split | 118 | 0.542 | 0.726 [0.634, 0.812] |

## 3. Orthogonal component (out-of-fold linear residual)

| model | AUROC of rmd_tail_q20 given vote | AUROC of vote given rmd_tail_q20 |
|---|---|---|
| DeepSeek-Qwen | 0.660 [0.590, 0.728] | 0.447 [0.373, 0.534] |
| Llama | 0.670 [0.618, 0.723] | 0.562 [0.500, 0.627] |
| Qwen | 0.744 [0.689, 0.801] | 0.480 [0.404, 0.558] |

## 4. Substitution, both directions (AURC, lower is better)

| model | B1 - B0 | rmd for vote - B0 | B1 - (rmd for vote) | rmd added to voteless |
|---|---|---|---|---|
| DeepSeek-Qwen | -0.0355 [-0.0636, -0.0071] p=0.010 | -0.0232 [-0.0563, +0.0113] p=0.208 | -0.0123 [-0.0267, +0.0005] p=0.056 | -0.0448 [-0.0734, -0.0104] p=0.010 |
| Llama | -0.0560 [-0.0917, -0.0183] p=0.002 | -0.0467 [-0.0879, -0.0020] p=0.032 | -0.0093 [-0.0192, -0.0007] p=0.042 | -0.1685 [-0.2194, -0.1171] p=0.000 |
| Qwen | -0.0585 [-0.0975, -0.0221] p=0.000 | -0.0548 [-0.0963, -0.0183] p=0.004 | -0.0037 [-0.0174, +0.0082] p=0.632 | -0.0658 [-0.1058, -0.0300] p=0.000 |

## 5. Agreement levels present

| model | agreement | n | accuracy |
|---|---:|---:|---:|
| DeepSeek-Qwen | 0.429 | 1 | 1.000 |
| DeepSeek-Qwen | 0.500 | 2 | 0.500 |
| DeepSeek-Qwen | 0.571 | 1 | 0.000 |
| DeepSeek-Qwen | 0.625 | 6 | 0.333 |
| DeepSeek-Qwen | 0.750 | 11 | 0.636 |
| DeepSeek-Qwen | 0.800 | 1 | 1.000 |
| DeepSeek-Qwen | 0.857 | 3 | 0.667 |
| DeepSeek-Qwen | 0.875 | 19 | 0.526 |
| DeepSeek-Qwen | 1.000 | 349 | 0.828 |
| Llama | 0.125 | 4 | 0.000 |
| Llama | 0.250 | 12 | 0.667 |
| Llama | 0.375 | 15 | 0.333 |
| Llama | 0.500 | 22 | 0.455 |
| Llama | 0.625 | 34 | 0.735 |
| Llama | 0.750 | 43 | 0.512 |
| Llama | 0.875 | 64 | 0.547 |
| Llama | 1.000 | 214 | 0.794 |
| Qwen | 0.125 | 2 | 0.000 |
| Qwen | 0.250 | 7 | 0.000 |
| Qwen | 0.375 | 13 | 0.308 |
| Qwen | 0.500 | 18 | 0.389 |
| Qwen | 0.625 | 25 | 0.440 |
| Qwen | 0.714 | 1 | 1.000 |
| Qwen | 0.750 | 19 | 0.737 |
| Qwen | 0.875 | 33 | 0.818 |
| Qwen | 1.000 | 274 | 0.755 |
