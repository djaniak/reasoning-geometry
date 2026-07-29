# reasoning-geometry-probe

Probing hidden-state geometry in math reasoning.

The core question: when a language model generates a reasoning trace, does the *geometry* of its hidden states tell us something about correctness that token-level entropy misses? We fit a reference manifold (PCA + Gaussian) on hidden states from correct traces, then measure Mahalanobis distance for each new trace. The hypothesis is that incorrect reasoning deviates from this manifold even when the output distribution looks similar token-by-token.

Current evidence is the clean Qwen Best-of-8 MATH-500 rerun. Historical
DeepSeek, transfer, temperature, and prefix runs remain under `results/` for
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
| Qwen2.5-7B-Instruct | Qwen2.5, 28 layers, hidden dim 3584 | Greedy / Best-of-8 |
| DeepSeek-R1-Distill-Qwen-7B | Historical diagnostic; clean replication pending | — |
| Llama-3.1-8B-Instruct | Historical diagnostic; clean replication pending | — |
| DeepSeek-R1-Distill-Llama-8B | Historical diagnostic; clean replication pending | — |

Datasets: **MATH-500** (500 problems, 5 difficulty levels, 7 subjects) and **GSM8K** test set (~1300 problems).

## Current result

On clean Qwen Best-of-8 traces, RMD over the highest-entropy 20% of tokens
beats full-trace RMD at all three layers and beats a matched random-token
control. It remains competitive with output baselines, adds only suggestive
incremental value, and does not improve Best-of-N tie-breaking. The strongest
surviving interpretation is prompt-level difficulty/abstention, not a reliable
per-attempt correctness detector.

**The within-prompt result is Qwen-specific; the between-prompt result is not.**
A pre-registered cross-model gate on DeepSeek-R1-Distill-Qwen-7B failed
(2026-07-29): the localization effect is +0.004 [−0.016, +0.027], excluding
Qwen's +0.058, and *every* within-prompt AUC on that model — geometry and output
baselines alike — sits at or below chance. On the same data, prompt-level
abstention **did** replicate, beating the length confound baseline by +0.030
AURC [+0.014, +0.048], p < 0.001.

The cross-model claim is therefore: **hidden-state geometry indicates which
problems are hard, not which attempt is right.** See
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

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
CUDA_VISIBLE_DEVICES="" uv run dvc repro --single-item evaluate_prompt_selection@0

# Inspect the active graph
uv run dvc status
```

The default DAG contains the Qwen baseline, dense-layer/PCA checks,
truncation-budget diagnostics, Qwen Best-of-N decomposition/selection, and
Wave-1 CPU follow-ups. Retired model-family and application stages are not
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
