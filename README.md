# reasoning-geometry-probe

Probing hidden-state geometry in math reasoning.

The core question: when a language model generates a reasoning trace, does the *geometry* of its hidden states tell us something about correctness that token-level entropy misses? We fit a reference manifold (PCA + Gaussian) on hidden states from correct traces, then measure Mahalanobis distance for each new trace. The hypothesis is that incorrect reasoning deviates from this manifold even when the output distribution looks similar token-by-token.

Current evidence is the Best-of-8 MATH-500 runs on three models. Historical
greedy, transfer, temperature, and prefix runs remain under `results/` for
provenance but are not current evidence. See [FINDINGS.md](FINDINGS.md) and
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

Several experiment families have been **retired after returning negative or null
results** — Best-of-N geometry reranking, prefix abort-and-retry filtering,
functional-trajectory encoding, and the PCA-dimension sweep. They are not
pending work; the questions are answered. The verdict and evidence for each is
in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) under *2026-07-25: DVC graph
restructure*, and [notebooks/README.md](notebooks/README.md) indexes which
notebooks are current evidence versus archived diagnostics.

## Models and datasets

| Model | Architecture | Decoding |
|---|---|---|
| Qwen2.5-7B-Instruct | Qwen2.5, 28 layers, hidden dim 3584 | Best-of-8, layer 21, 1024-token budget |
| DeepSeek-R1-Distill-Qwen-7B | Qwen2.5 lineage, reasoning distill | Best-of-8, layer 21, 8192-token budget |
| DeepSeek-R1-Distill-Llama-8B | Llama lineage, reasoning distill | Best-of-8, layer 24, 12288-token budget |
| Llama-3.1-8B-Instruct | Llama lineage, no reasoning distill | **Queued**: Best-of-8, layer 24, 1024-token budget |

Datasets: **MATH-500** (500 problems, 5 difficulty levels, 7 subjects), **GSM8K** test set
(~1300 problems, single greedy trace only), and **OlympiadBench** (`OE_TO_maths_en_COMP`,
501 of 674 rows after the single-numerical-answer filter) — **queued**.

Every current result rests on Best-of-8 MATH-500 alone. GSM8K was collected at one sample
per problem, so it supports no claim that needs siblings, and its greedy pass rate (0.91 /
0.91 / 0.63) leaves too few errors for an abstention curve to rank. The two queued collects
address the two things that scope cannot fix on its own: OlympiadBench is a second prompt
set, and Llama-3.1-8B-Instruct is a second *non-distilled* model, so that "distilled vs not"
stops being confounded with "Qwen vs not". Neither has run; see `params.yaml` and the
`probe_dataset` gate.

## Current result

**Hidden-state geometry adds selective-prediction value on top of a
self-consistency baseline, on three models.** The claim is an *increment*, not a
score for the feature alone. Baseline `B0 = (length, entropy, logprob,
vote_agreement)` aggregated over 8 sibling traces; `B1 = B0 + rmd_tail_q20`,
the **mean** per-token relative Mahalanobis distance over the **final 20% of
trace tokens** (`prompt_decomposition.py::score_localized_rmd`; the `q20` in
the name is the size of the tail window, not a quantile of the distances).
Out-of-fold logistic readouts, prompt-clustered paired bootstrap, 1000 draws.

On the primary `full_population` estimand -- correctness available at the stated
generation budget over all 500 prompts -- in **AURC** (area under the
risk-coverage curve, lower is better):

| Model | n | `B1 − B0` |
|---|---:|---|
| Qwen2.5-7B-Instruct | 500 | −0.0520 [−0.0845, −0.0218] |
| DeepSeek-R1-Distill-Qwen-7B | 500 | −0.0284 [−0.0526, −0.0048] |
| DeepSeek-R1-Distill-Llama-8B | 500 | −0.0469 [−0.0743, −0.0162] |

Controls it survives: three difficulty controls, one of them MATH-500's exogenous
human-annotated level; a length residualization; DeepConf (arXiv:2508.15260) as
a prompt-level score, as a confidence-weighted vote, and as a confidence filter,
with all four of its statistics; and a vote-proxy control answering Orgad et al.
(arXiv:2410.02707) — geometry scores AUROC 0.71–0.83 *inside* the stratum where
the eight siblings agree unanimously and self-consistency is silent.

The vote-proxy control now also holds on the **whole** population, not just that
stratum (2026-08-10). Adding the full answer-distribution entropy to `B0` buys
nothing (−0.0006 / −0.0035 / +0.0003, all p > 0.05), and `rmd_tail_q20` still
adds on top of it on all three models, Holm-corrected over the pre-declared
family of six (Holm p 0.000 / 0.048 / 0.008; DeepSeek-R1-Distill-Qwen-7B is a
borderline pass — raw p=0.016 against a 0.0167 threshold, and the bootstrap only
resolves p to 1/1000). The reason the histogram is empty is worth reporting:
at 8 samples on MATH-500, **70% / 89% / 53% of prompts are unanimous**, so every
answer-distribution statistic is constant by construction on most of the data.
That is a limit on self-consistency baselines at this sample count, not a
property of this feature.

**A deployable peer is a genuine competitor, not a mechanism control.** A peer's
agreement with the target answer needs no gold label, but it costs extra
generations; `B1` uses states from the target's existing eight traces. At the
cheapest deployable rung, the six target-peer comparisons give four ties, one
RMD win and one peer win after Holm correction over the six tests. No peer rung
is exactly cost-matched to `B1`, so the supported advantage is zero additional
generations, not superiority to peer uncertainty. The older conclusion that
peer pass rates absorb most of the increment used a gold-aware diagnostic and
is withdrawn.

**It does not extend to sample allocation (2026-08-10).** A pre-declared gate
asked whether single-trace geometry predicts the *gain from buying more samples*,
`g(p) = a(p,8) − a(p,1)`, with `a(p,k)` the expected plurality-vote correctness
over all `C(8,k)` sibling subsets. It does not: geometry ranks the gain backwards
(Spearman −0.042 / −0.057 / −0.074) while correlating +0.51 / +0.24 / +0.37 with
the pass rate, and the gate fails on 1 of 3 models with the one pass sitting at
R² = +0.0005. That is the expected shape — gain is non-monotone in difficulty, and
a prompt at 0/8 and one at 8/8 both gain nothing. It is not a sample-size problem:
at one trace the feature holds AUROC 0.790 / 0.674 / 0.688 against its 0.806 /
0.686 / 0.709 at eight. **Ruled out: ranking prompts by predicted gain from more
samples. Untouched: ranking them by difficulty, for abstention or routing.**

**The tail window is a Qwen-specific localization, not part of the method.**
The untailed whole-trace mean `rmd_full` — Vazhentsev et al.'s ATRMD — recovers
almost the entire increment by itself on both reasoning-distilled models
(−0.0335 of −0.0355; −0.0509 of −0.0560), and the tail adds nothing separable
from zero there. Only on Qwen2.5-7B-Instruct does the tail carry the result.
Window size does not explain the split: inside Qwen the tail advantage *grows*
with window size (−0.042 below the median window, −0.116 above it), and the
Llama short stratum matched to Qwen on window median and base accuracy still
shows no tail effect. The split follows reasoning distillation, on one
non-distilled model, with trace style, budget and base accuracy still collinear
with it. A third non-distilled model is the test that would settle it.

**Pooled trace discrimination is not sibling verification.** A nested
last-token probe reaches pooled AUROC 0.901 / 0.914 / 0.903, but macro
within-prompt AUROC is 0.644 / 0.582 / 0.718. RMD shows the same qualitative
gap. Length also loses substantial pooled discrimination, while entropy and
log-probability lose less. The supported claim is that the prompt-level
increment survives; a high pooled trace AUROC alone does not establish which
sibling trace is correct. See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

*Metric note: AURC and AUACC are affinely related at fixed n and both inherit the
base accuracy, so levels are not comparable across models — only deltas are.
AUROC is the base-rate-free metric and is used wherever a comparison crosses
models.*

## Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

Requires Python 3.11+.

## Running the pipeline

The pipeline is managed with [DVC](https://dvc.org). Each stage collects hidden states or runs analysis for one model × dataset combination.

```bash
# Reproduce the current CPU analysis
CUDA_VISIBLE_DEVICES="" uv run dvc repro --single-item evaluate_prompt_decomposition@0
CUDA_VISIBLE_DEVICES="" uv run dvc repro --single-item evaluate_wave1_experiments@0
CUDA_VISIBLE_DEVICES="" uv run dvc repro --single-item evaluate_abstention_baselines@0

# Inspect the active graph
uv run dvc status
```

The default DAG contains the Qwen baseline, dense-layer/PCA checks,
truncation-budget diagnostics, Qwen Best-of-N decomposition, and Wave-1 CPU
follow-ups. Retired model-family and application stages are not
default dependencies; see the experiment log for their status.

Configuration is in [params.yaml](params.yaml): model names, layer indices, PCA dimension, bootstrap count, etc.

## Scripts

| Script | Purpose |
|---|---|
| `collect_data.py` | Run autoregressive generation, capture hidden states and per-token entropy, save `.npz` files |
| `analyze.py` | Fit reference manifold, extract features, run 5-fold CV logistic regression |
| `probe.py` | Trajectory probe: resample Mahalanobis sequence to fixed length, run functional PCA |
| `best_of_n.py` | Best-of-N reranking using geometry scores |
| `prefix_analysis.py` | Retired early-prefix diagnostic (historical) |
| `prefix_filter.py` | Retired abort/retry diagnostic (historical) |
| `summarize.py` | Aggregate the current result profile into `results/SUMMARY.md` |

## Output structure

```
data/
  {model}/{dataset}/      # .npz files with hidden states (DVC-tracked, not in git)
results/
  {model}/{dataset}/      # metrics JSON + plots
  SUMMARY.md              # aggregated results table
```

## Features

Each trace is represented by 12 scalar features extracted from the per-token sequence:

- **Entropy-only (5)**: mean, max, std, 90th percentile, fraction above median
- **Mahalanobis-only (7)**: mean, max, std, 90th percentile distance; mean/max distance at high-entropy tokens; entropy–Mahal correlation
- **Combined (12)**: union of the above

The reference manifold is fitted on PCA(128) of hidden states from correct traces only, with a regularized Gaussian covariance.
