# Application alignment

Exploratory descriptive correlations only; the number of independent model-layer conditions is too small for significance claims.

| Model | Layer | Method | Within AUC | Centered AUC | ICC | Prompt corr | Top-1 gain | Gap to majority | Selective gain |
|:---|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| deepseek | 7 | raw | 0.318 | 0.362 | 0.878 | -0.444 | -0.075 | -0.112 | -0.200 |
| deepseek | 7 | rmd | 0.931 | 0.841 | 0.878 | 0.682 | +0.109 | +0.072 | +0.126 |
| deepseek | 14 | raw | 0.338 | 0.371 | 0.906 | -0.484 | -0.045 | -0.082 | -0.202 |
| deepseek | 14 | rmd | 0.927 | 0.837 | 0.920 | 0.694 | +0.111 | +0.074 | +0.125 |
| deepseek | 21 | raw | 0.353 | 0.369 | 0.904 | -0.613 | -0.051 | -0.088 | -0.234 |
| deepseek | 21 | rmd | 0.930 | 0.797 | 0.831 | 0.698 | +0.109 | +0.072 | +0.133 |
| qwen | 7 | raw | 0.491 | 0.484 | 0.878 | -0.195 | -0.011 | -0.050 | -0.103 |
| qwen | 7 | rmd | 0.551 | 0.555 | 0.943 | 0.452 | -0.007 | -0.046 | +0.073 |
| qwen | 14 | raw | 0.484 | 0.476 | 0.901 | -0.175 | -0.035 | -0.074 | -0.104 |
| qwen | 14 | rmd | 0.550 | 0.550 | 0.970 | 0.499 | -0.005 | -0.044 | +0.086 |
| qwen | 21 | raw | 0.441 | 0.467 | 0.901 | -0.357 | -0.035 | -0.074 | -0.158 |
| qwen | 21 | rmd | 0.602 | 0.592 | 0.960 | 0.547 | +0.007 | -0.032 | +0.100 |

## Descriptive correlations

- `pooled/within_auc_vs_top1_gain`: Spearman=0.961, n=12
- `pooled/prompt_correlation_vs_selective_gain`: Spearman=0.986, n=12
- `pooled/icc_vs_selective_gain`: Spearman=0.007, n=12
- `raw/within_auc_vs_top1_gain`: Spearman=0.928, n=6
- `raw/prompt_correlation_vs_selective_gain`: Spearman=0.943, n=6
- `raw/icc_vs_selective_gain`: Spearman=-0.543, n=6
- `rmd/within_auc_vs_top1_gain`: Spearman=0.754, n=6
- `rmd/prompt_correlation_vs_selective_gain`: Spearman=0.943, n=6
- `rmd/icc_vs_selective_gain`: Spearman=-0.829, n=6
