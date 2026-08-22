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
| qwen | 21 | 392 | 0.691 | 0.034 | 0.791 | 0.046 | -0.090 |
| deepseek | 21 | 393 | 0.796 | 0.012 | 0.896 | 0.028 | -0.161 |
| deepseek_llama | 24 | 408 | 0.674 | 0.070 | 0.679 | 0.064 | -0.035 |

`rho(pass rate, g)` is the non-monotonicity, measured rather than asserted: were
gain a monotone function of difficulty it would sit near -1 and this precheck
would be redundant with the difficulty results.

Mean expected accuracy along the curve:

| model | a(p,1) | a(p,2) | a(p,4) | a(p,8) |
|---|---:|---:|---:|---:|
| qwen | 0.657 | 0.670 | 0.685 | 0.691 |
| deepseek | 0.785 | 0.786 | 0.794 | 0.796 |
| deepseek_llama | 0.604 | 0.594 | 0.641 | 0.674 |

## The gate

Out-of-fold prediction of `g(p)` on the frozen prompt folds. `R^2` is against a
**cross-fitted constant** -- the training-fold mean -- so the baseline is held out
exactly as the readouts are.

| model | Spearman geometry | Spearman output | Spearman both | R^2 geometry | R^2 output | R^2 both |
|---|---|---|---|---|---|---|
| qwen | -0.042 [-0.064, 0.011] | 0.073 [0.031, 0.102] | 0.102 [0.068, 0.172] | -0.004 [-0.005, -0.002] | 0.005 [-0.007, 0.012] | 0.009 [-0.003, 0.024] |
| deepseek | -0.057 [-0.138, 0.009] | 0.032 [-0.034, 0.075] | 0.015 [-0.017, 0.052] | -0.006 [-0.023, 0.000] | 0.001 [-0.025, 0.021] | -0.003 [-0.021, 0.021] |
| deepseek_llama | -0.074 [-0.123, -0.055] | -0.031 [-0.131, 0.009] | -0.009 [-0.039, 0.019] | 0.001 [-0.004, 0.003] | -0.002 [-0.013, 0.006] | 0.007 [-0.002, 0.017] |

| model | geometry beats constant | geometry adds over output (paired Spearman) | passes |
|---|---|---|---|
| qwen | no (R^2 -0.004) | yes (0.033 [-0.002, 0.070]) | **fail** |
| deepseek | no (R^2 -0.006) | no (-0.007 [-0.053, 0.033]) | **fail** |
| deepseek_llama | yes (R^2 0.001) | yes (0.026 [-0.013, 0.097]) | **PASS** |

Pre-declared rule: geometry alone beats the cross-fitted constant (R^2 > 0) and adds over output-alone in out-of-fold Spearman, on at least 2 of the models; medians over the stage-1 draws.

Passing: deepseek_llama (1/3). Gate: **FAIL** -- step 3 is not run; geometry contributes nothing to allocation and that is the finding.

## Diagnostics (not the gate)

These explain a failure; they do not decide one.

`AUROC vs prompt outcome` holds the target fixed at the eight-sibling plurality
outcome and varies only the feature, so it is directly comparable to the
sibling-mean marginal AUROC in the 2026-08-10 table (0.806 / 0.686 / 0.709) and
measures how much `rmd_tail_q20` degrades at n = 1. `AUROC vs own trace` is the
n = 1 decision problem itself.

| model | AUROC vs prompt outcome | AUROC vs own trace | rho(geometry, pass rate) | rho(geometry, g) |
|---|---|---|---|---|
| qwen | 0.790 [0.782, 0.799] | 0.784 [0.763, 0.817] | 0.512 [0.501, 0.526] | -0.021 [-0.043, 0.015] |
| deepseek | 0.674 [0.629, 0.697] | 0.663 [0.626, 0.707] | 0.235 [0.170, 0.258] | 0.035 [-0.020, 0.080] |
| deepseek_llama | 0.688 [0.680, 0.718] | 0.701 [0.663, 0.718] | 0.366 [0.333, 0.397] | -0.006 [-0.020, 0.034] |

Flagged with |rho(geometry, pass rate)| >= 0.20 and |rho(geometry, g)| < 0.10: qwen, deepseek, deepseek_llama.

On these models **geometry reads difficulty but not marginal gain** -- the
specific failure mode this precheck exists to catch. It is consistent with the
2026-08-10 peer control, which found most of the `rmd_tail_q20` increment is
prompt difficulty: difficulty is exactly the thing that does *not* order
prompts by how much another sample would help.

## Harness checks

| model | a(p,8) == frozen outcome | max |a(p,1) - cached pass rate| | prompts differing |
|---|---|---:|---:|
| qwen | yes | 0.0000 | 0/392 |
| deepseek | yes | 0.0000 | 0/393 |
| deepseek_llama | yes | 0.0000 | 0/408 |

`a(p,8)` reproducing the frozen prompt outcome is an identity -- C(8,8) is the one
subset containing every sibling -- and it is asserted at run time, not merely
reported. `a(p,1)` against the cached `is_correct` column is *not* an identity:
the first recomputes the answer match through the frozen parser, the second is the
collector's stored verdict, and any gap is answer-normalization drift.
