# Step 2 -- allocation precheck

Does single-trace hidden-state geometry predict the **gain from buying more
samples**, `g(p) = a(p,8) - a(p,1)`? This is not the correctness question every
other rung asks. Gain is non-monotone in difficulty: a prompt solved 0/8 and a
prompt solved 8/8 both gain nothing.

`a(p,k)` is the expected plurality-vote correctness over **all** `C(8,k)` sibling
subsets, using the frozen `_winning_answer` and the frozen automatic-failure rule
for an all-unparsed subset. Stage-1 features are the four that exist at one
sample; `vote_agreement` is excluded because a single sample has no siblings.

Every number below is the median over the **eight choices of stage-1 trace**, with
the full range in brackets. Which trace you happen to draw first is a random
variable, and fixing it at `sample_id == 0` would hide that variance.

## Population and target

| model | layer | n | base acc | mean g | share g = 0 | share g < 0 | rho(pass rate, g) |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen | 21 | 500 | 0.620 | 0.063 | 0.738 | 0.056 | 0.006 |
| deepseek | 21 | 500 | 0.750 | 0.049 | 0.796 | 0.024 | -0.230 |
| deepseek_llama | 24 | 500 | 0.634 | 0.082 | 0.636 | 0.072 | 0.050 |

`rho(pass rate, g)` is the non-monotonicity, measured rather than asserted: were
gain a monotone function of difficulty it would sit near -1 and this precheck
would be redundant with the difficulty results.

Mean expected accuracy along the curve:

| model | a(p,1) | a(p,2) | a(p,4) | a(p,8) |
|---|---:|---:|---:|---:|
| qwen | 0.557 | 0.581 | 0.603 | 0.620 |
| deepseek | 0.701 | 0.724 | 0.740 | 0.750 |
| deepseek_llama | 0.552 | 0.558 | 0.603 | 0.634 |

## The gate

Out-of-fold prediction of `g(p)` on the frozen prompt folds. `R^2` is against a
**cross-fitted constant** -- the training-fold mean -- so the baseline is held out
exactly as the readouts are.

| model | Spearman geometry | Spearman output | Spearman both | R^2 geometry | R^2 output | R^2 both |
|---|---|---|---|---|---|---|
| qwen | 0.056 [0.023, 0.090] | 0.141 [0.118, 0.155] | 0.158 [0.135, 0.173] | 0.019 [0.011, 0.033] | 0.047 [0.040, 0.062] | 0.051 [0.040, 0.060] |
| deepseek | 0.291 [0.265, 0.316] | 0.272 [0.244, 0.307] | 0.285 [0.265, 0.308] | 0.137 [0.119, 0.157] | 0.103 [0.078, 0.112] | 0.139 [0.109, 0.154] |
| deepseek_llama | 0.031 [-0.001, 0.055] | -0.014 [-0.091, 0.028] | 0.007 [-0.034, 0.048] | 0.016 [0.010, 0.019] | -0.004 [-0.016, 0.005] | 0.006 [-0.001, 0.023] |

| model | geometry beats constant | geometry adds over output (paired Spearman) | passes |
|---|---|---|---|
| qwen | yes (R^2 0.019) | yes (0.018 [-0.002, 0.039]) | **PASS** |
| deepseek | yes (R^2 0.137) | yes (0.006 [-0.016, 0.024]) | **PASS** |
| deepseek_llama | yes (R^2 0.016) | yes (0.012 [0.000, 0.065]) | **PASS** |

Pre-declared rule: geometry alone beats the cross-fitted constant (R^2 > 0) and adds over output-alone in out-of-fold Spearman, on at least 2 of the models; medians over the stage-1 draws.

Passing: qwen, deepseek, deepseek_llama (3/3). Gate: **PASS** -- write allocation.py (step 3).

## Diagnostics (not the gate)

These explain a failure; they do not decide one.

`AUROC vs prompt outcome` holds the target fixed at the eight-sibling plurality
outcome and varies only the feature, so it is directly comparable to the
sibling-mean marginal AUROC in the 2026-08-10 table (0.806 / 0.686 / 0.709) and
measures how much `rmd_tail_q20` degrades at n = 1. `AUROC vs own trace` is the
n = 1 decision problem itself.

| model | AUROC vs prompt outcome | AUROC vs own trace | rho(geometry, pass rate) | rho(geometry, g) |
|---|---|---|---|---|
| qwen | 0.808 [0.795, 0.814] | 0.838 [0.829, 0.864] | 0.625 [0.615, 0.630] | -0.064 [-0.095, -0.030] |
| deepseek | 0.705 [0.680, 0.716] | 0.761 [0.747, 0.780] | 0.493 [0.458, 0.505] | -0.293 [-0.318, -0.264] |
| deepseek_llama | 0.700 [0.685, 0.717] | 0.735 [0.696, 0.750] | 0.441 [0.411, 0.461] | -0.060 [-0.077, -0.034] |

Flagged with |rho(geometry, pass rate)| >= 0.20 and |rho(geometry, g)| < 0.10: qwen, deepseek_llama.

On these models **geometry reads difficulty but not marginal gain** -- the
specific failure mode this precheck exists to catch. It is consistent with the
2026-08-10 peer control, which found most of the `rmd_tail_q20` increment is
prompt difficulty: difficulty is exactly the thing that does *not* order
prompts by how much another sample would help.

## Harness checks

| model | a(p,8) == frozen outcome | max |a(p,1) - cached pass rate| | prompts differing |
|---|---|---:|---:|
| qwen | yes | 0.0000 | 0/500 |
| deepseek | yes | 0.0000 | 0/500 |
| deepseek_llama | yes | 0.0000 | 0/500 |

`a(p,8)` reproducing the frozen prompt outcome is an identity -- C(8,8) is the one
subset containing every sibling -- and it is asserted at run time, not merely
reported. `a(p,1)` against the cached `is_correct` column is *not* an identity:
the first recomputes the answer match through the frozen parser, the second is the
collector's stored verdict, and any gap is answer-normalization drift.
