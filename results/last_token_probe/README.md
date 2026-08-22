# The last-token probe, reproduced and decomposed

The 2026-08-21 review's fourth blocker. The between/within decomposition has so
far been applied to this project's own scores. It has not been applied to the
object whose claim shape it corrects: a supervised probe on the **last token**
hidden state, of the kind a high pooled trace AUROC is reported for in the probe
literature. Until that is done, the paper is a self-audit rather than a
correction of a published claim shape.

## Why the repository's existing probe is not that probe

`applications/prompt_decomposition` already fits `probe_hidden_*`. It is a
different object in three ways that all matter:

| | `probe_hidden_tail_q20` (frozen) | `last_token_probe` (here) |
|:--|:--|:--|
| Feature | mean of PCA-projected states over the final 20% of tokens | the final token's raw hidden state |
| Readout | LDA with automatic shrinkage on `pca_dim=128` | L2 logistic on the full model width |
| Layer | every probed layer scored; the choice is made downstream | chosen inside each training split |

A tail-pooled LDA on 128 principal components is not a reproduction of a
last-token probe, so the frozen numbers cannot answer the review. Both are in
the report, side by side, on the same traces.

## Protocol

- **Feature.** The hidden state at the final generated token, at one layer.
- **Readout.** L2 logistic regression, liblinear dual (it solves in the sample
  count rather than the feature count, which is what makes in-fold selection
  affordable at this width).
- **Folds.** Prompt-disjoint 5-fold, `KFold(shuffle=True, random_state=42)` over
  sorted prompt ids -- the frozen pipeline's own folds, via its own
  `make_prompt_folds`.
- **Selection.** Layer *and* penalty are chosen together inside each outer
  training split, by prompt-disjoint inner 5-fold CV. Neither is ever chosen on
  the split it is scored on.
- **Population.** `parseable` is primary and matches the frozen probe's training
  rule; `all_traces` is reported alongside it so the truncation-detection
  inflation is visible rather than assumed away.

The penalty is selected rather than fixed because it is not a detail: on Qwen
layer 21 the held-out pooled AUROC runs 0.828 at `C=1` and 0.895 at `C=1e-3`.
Reproducing a published claim means reproducing it at its strength, and a fixed
loose penalty would have understated the very number this exists to decompose.

## The three readouts, and the populations they live on

| Readout | Counts | Single-outcome prompts |
|:--|:--|:--|
| `pooled` | every held-out trace, prompt identity ignored | their traces are included |
| `micro` | every within-prompt (correct, incorrect) pair, pair-weighted | contribute no pairs |
| `macro` | per-prompt AUROC, each prompt counting once | undefined, excluded |

`pooled` is the published number. The three are **not** defined on the same
population, and the report states the counts for each. That difference is the
finding, not a caveat: a pooled score can be high because it ranks prompts, and
ranking prompts is not the same capability as ranking a prompt's own samples.

## What it found

On `parseable`, all three models put a published-shape pooled trace AUROC in the
low 0.90s and lose most of it once prompt identity is conditioned on.

| Model | Pooled | Macro prompt | Pooled - macro | 95% CI | Mixed prompts | Pairs |
|:--|---:|---:|---:|:--|---:|---:|
| qwen | 0.9013 | 0.6444 | 0.2569 | [0.1980, 0.3148] | 117 | 1104 |
| deepseek | 0.9139 | 0.5823 | 0.3316 | [0.2260, 0.4499] | 49 | 409 |
| deepseek_llama | 0.9032 | 0.7177 | 0.1855 | [0.1410, 0.2325] | 158 | 1636 |

Every interval excludes zero. DeepSeek is the extreme case -- 0.914 pooled,
0.582 within prompt -- and also the least certain, because only 49 of its 493
prompts have both outcomes. That count, not the bootstrap, is what bounds the
within-prompt evidence, which is why the report prints it beside every readout.

Two patterns hold across models.

**The collapse is specific to hidden-state scores.** `probe_hidden_tail_q20`
drops 0.30 / 0.21 / 0.15 across the three models and `rmd_tail_q20` drops
0.21 / 0.21 / 0.14, while `mean_entropy` and `mean_logprob` drop around 0.06 or
go *negative* on Qwen -- there they pool near 0.60 and score 0.65-0.67 within
prompt. So this is not a generic artifact of the decomposition. The output-side
scores keep their signal under it; the geometry-based ones do not.

**The probe that wins pooled does not win the within-prompt task.** On Qwen the
last-token probe leads `mean_entropy` by 30 points pooled (0.9013 vs 0.5951) and
trails it within prompt (0.6444 vs 0.6602). A reader who takes the pooled number
as a per-sample selection capability is reading it as something it is not.

Layer selection does not tell one story. Qwen picks the earliest offered layer
(L7) in 4 of 5 parseable folds, which is what prompt-difficulty encoding would
look like; but Llama picks its middle layer in 10 of 10 folds and DeepSeek picks
L14 in most. The early-layer observation is Qwen's, not a general result, and is
not evidence for the difficulty reading on its own.

## Continuity with the frozen report

The reference scores are recomputed here from the same traces and reproduce the
frozen Qwen layer-21 `prompt_decomposition` report on all four of its columns:

| Score | Frozen (pooled / centered / macro / pair) | Here |
|:--|:--|:--|
| `entropy` | 0.571 / 0.559 / 0.599 / 0.595 | 0.5708 / 0.5588 / 0.5992 / 0.5948 |
| `logprob` | 0.575 / 0.559 / 0.595 / 0.589 | 0.5747 / 0.5594 / 0.5947 / 0.5892 |
| `length` | 0.737 / 0.563 / 0.581 / 0.582 | 0.7367 / 0.5629 / 0.5807 / 0.5820 |
| `rmd_tail_q20` | 0.839 / 0.640 / 0.658 / 0.653 | 0.8393 / 0.6396 / 0.6584 / 0.6527 |

`length` is `-log1p(token count)`, matching the frozen definition. The transform
matters for exactly one column: pooled, micro and macro are rank statistics and
cannot see it, but the prompt-centered readout subtracts a per-prompt mean from
the raw score and can. Reproducing three columns and missing the fourth is the
signature of a monotone mismatch, and it is pinned by a test.

## Files

| File | What it is |
|:---|:---|
| `last_token_probe_results.json` | every score, readout, fold, selection grid and interval |
| `last_token_probe_report.md` | what was fitted; the decomposition per population; the in-fold selection trace |
| `cache/` | extracted last-token states (gitignored; one pass over the trace batches rebuilds it) |

```
uv run python -m controls.last_token_probe \
  --model qwen:data/qwen_bestofn_full/math500:7,14,21 \
  --oof qwen:results/qwen_bestofn_full/math500/math500_prompt_decomposition_oof.csv:21 \
  --output_dir results/last_token_probe_qwen
# ... deepseek (7,14,21) and deepseek_llama (8,16,24) likewise, then:
uv run python -m controls.last_token_probe --merge <each>/last_token_probe_results.json
```

Not a DVC stage, and it does not touch one. The frozen pipeline is left exactly
as committed: this reads the same trace batches independently, so no stage is
invalidated and nothing needs re-running to reproduce the paper's other
exhibits.

## Extraction

The published protocol needs one vector per (trace, layer), not the whole token
sequence, so the analysis does not go near the frozen pipeline's ~140 GB peak.
One pass over the collected batches pulls the final token's state at each probed
layer into a small cache (~66 MB per model); everything after that is CPU
minutes on cached arrays. The stored arrays are DEFLATE members, so the tail
cannot be seeked to and the whole array is inflated and discarded -- that is the
entire cost, and it is paid once per model.

## Uncertainty

Intervals resample prompts with the fit held fixed -- folds, layer, penalty and
coefficients all frozen -- exactly as every other interval in this project does.
They do not carry the uncertainty of the fitting path itself. That is the
outer-refit blocker, and it is the next rung.
