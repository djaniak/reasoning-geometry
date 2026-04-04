# reasoning-geometry-probe

Probing hidden-state geometry as a signal for math reasoning errors in LLMs.

The core question: when a language model generates a reasoning trace, does the *geometry* of its hidden states tell us something about correctness that token-level entropy misses? We fit a reference manifold (PCA + Gaussian) on hidden states from correct traces, then measure Mahalanobis distance for each new trace. The hypothesis is that incorrect reasoning deviates from this manifold even when the output distribution looks similar token-by-token.

See [FINDINGS.md](FINDINGS.md) for full results.

## Models and datasets

| Model | Architecture | Decoding |
|---|---|---|
| Qwen2.5-7B-Instruct | Qwen2.5, 28 layers, hidden dim 3584 | Greedy |
| DeepSeek-R1-Distill-Qwen-7B | Same architecture, reasoning-distilled | Greedy / T=0.6 |

Datasets: **MATH-500** (500 problems, 5 difficulty levels, 7 subjects) and **GSM8K** test set (~1300 problems).

## Key results

Geometry adds signal beyond entropy in all four model × dataset conditions, even after controlling for trace length. The effect is largest for DeepSeek, where long reasoning chains neutralize entropy as a discriminator:

| Condition | Entropy AUC | Combined AUC | Δ (length-controlled) |
|---|---|---|---|
| Qwen MATH-500 | 0.713 | 0.772 | +0.027 |
| Qwen GSM8K | 0.760 | 0.781 | +0.014 |
| DeepSeek MATH-500 | 0.776 | 0.859 | +0.044 |
| DeepSeek GSM8K | 0.728 | 0.835 | +0.080 |

The geometry signal is **bimodal** across layers (peaks at early ~L6–L10 and late ~L24–L26, trough at L14), suggesting two distinct processing stages: problem comprehension and solution execution. Cross-model transfer shows the manifold shape is partially shared across architecturally identical but differently trained models, while decision boundaries are model-specific.

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
# Run a single stage
dvc repro collect_qwen_gsm8k
dvc repro analyze_qwen_gsm8k

# Run everything
dvc repro

# Check metrics
dvc metrics show
dvc metrics diff
```

Stage names follow the pattern `{collect|analyze}_{model}_{dataset}`. Available models: `qwen`, `deepseek`, `qwen_dense`, `deepseek_temp`. Available datasets: `gsm8k`, `math500`. See [dvc.yaml](dvc.yaml) for the full DAG.

Configuration is in [params.yaml](params.yaml): model names, layer indices, PCA dimension, bootstrap count, etc.

## Scripts

| Script | Purpose |
|---|---|
| `collect_data.py` | Run autoregressive generation, capture hidden states and per-token entropy, save `.npz` files |
| `analyze.py` | Fit reference manifold, extract features, run 5-fold CV logistic regression |
| `probe.py` | Trajectory probe: resample Mahalanobis sequence to fixed length, run functional PCA |
| `best_of_n.py` | Best-of-N reranking using geometry scores |
| `prefix_analysis.py` | Early-prefix classification: how predictive are the first N tokens? |
| `prefix_filter.py` | Prefix-based filtering to prune likely-incorrect samples before full generation |
| `summarize.py` | Aggregate all result JSONs into `results/SUMMARY.md` |

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
