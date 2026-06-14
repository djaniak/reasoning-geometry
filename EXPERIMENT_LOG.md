# Experiment Log

This ledger tracks completed evidence, artifact compatibility, and the next
smallest runnable stages. Dates are UTC. DVC stage completion means the output
is recorded in `dvc.lock`; it does not by itself imply that an artifact uses the
latest schema.

## 2026-06-14: Confidence Decomposition and Mechanism Experiments

### Status

| Experiment family | Conditions | Status |
|:---|:---|:---|
| Enriched prompt decomposition | Qwen and DeepSeek, 500 prompts x 8 traces | Complete |
| OOF prompt selection | Qwen and DeepSeek, 500 prompts x 8 traces | Complete |
| Application alignment | Qwen and DeepSeek, raw/RMD x 3 layers | Complete |
| Fair supervised RMD probe | Qwen, DeepSeek, Llama, DeepSeek-Llama | Complete |
| One-class mechanism sweep | Four models x 3 layers x 8 dimensions | Complete |

Artifacts:

- `results/qwen_bestofn_full/math500/math500_prompt_decomposition_results.json`
- `results/qwen_bestofn_full/math500/math500_prompt_decomposition_oof.csv`
- `results/deepseek_bestofn_full/math500/math500_prompt_decomposition_results.json`
- `results/deepseek_bestofn_full/math500/math500_prompt_decomposition_oof.csv`
- `results/qwen_bestofn_full/math500/math500_prompt_selection_results.json`
- `results/deepseek_bestofn_full/math500/math500_prompt_selection_results.json`
- `results/application_alignment/math500_application_alignment_results.json`
- `results/*_selective/math500/math500_selective_prediction_results.json`
- `results/*_one_class/math500/math500_one_class_sweep_results.json`

Protocol:

- Five prompt-grouped folds.
- PCA, correct-trace reference, and RMD background fitted on training prompts.
- Evaluation on held-out prompts.
- 1,000 prompt-cluster bootstrap replicates over fixed OOF predictions.
- All configured layers and dimensions are reported without post-hoc selection.
- The enriched OOF CSV contains answer metadata, entropy, log-probability,
  length, activation norm, centroid, raw Mahalanobis, and RMD scores.

### Prompt Decomposition

| Model | Layer | Method | Pooled AUC | Prompt-centered AUC | Within-prompt AUC | ICC | Prompt-score/pass-rate Spearman |
|:---|---:|:---|---:|---:|---:|---:|---:|
| Qwen | 7 | RMD | 0.736 | 0.555 | 0.551 | 0.943 | 0.452 |
| Qwen | 14 | RMD | 0.763 | 0.550 | 0.550 | 0.970 | 0.499 |
| Qwen | 21 | RMD | 0.786 | 0.592 | 0.602 | 0.960 | 0.547 |
| DeepSeek | 7 | RMD | 0.885 | 0.841 | 0.931 | 0.878 | 0.682 |
| DeepSeek | 14 | RMD | 0.887 | 0.837 | 0.927 | 0.920 | 0.694 |
| DeepSeek | 21 | RMD | 0.892 | 0.797 | 0.930 | 0.831 | 0.698 |

DeepSeek RMD beats entropy on within-prompt pairwise AUC by 0.134-0.138
across all three layers. The paired prompt-bootstrap intervals exclude zero:

| Layer | RMD minus entropy within-prompt AUC | 95% CI |
|---:|---:|:---|
| 7 | +0.138 | [+0.108, +0.169] |
| 14 | +0.134 | [+0.105, +0.165] |
| 21 | +0.138 | [+0.108, +0.168] |

For Qwen, RMD does not beat entropy, log-probability, or length within prompts
at any layer with a confidence interval excluding zero. Its pooled strength is
therefore primarily a between-prompt solvability signal.

### Prompt Selection

| Model | Layer | Random | Entropy | Length | RMD top-1 | Strict majority | Oracle Pass@8 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Qwen | 7 | 0.557 | 0.572 | 0.566 | 0.550 | 0.596 | 0.676 |
| Qwen | 14 | 0.557 | 0.572 | 0.566 | 0.552 | 0.596 | 0.676 |
| Qwen | 21 | 0.557 | 0.572 | 0.566 | 0.564 | 0.596 | 0.676 |
| DeepSeek | 7 | 0.416 | 0.488 | 0.506 | 0.524 | 0.452 | 0.546 |
| DeepSeek | 14 | 0.416 | 0.488 | 0.506 | 0.526 | 0.452 | 0.546 |
| DeepSeek | 21 | 0.416 | 0.488 | 0.506 | 0.524 | 0.452 | 0.546 |

Paired bootstrap reanalysis of saved prompt outcomes:

- DeepSeek RMD top-1 beats random by 0.109-0.111, entropy by 0.036-0.038,
  and length by 0.018-0.020. All corresponding 95% intervals exclude zero.
- Qwen RMD top-1 differs from random by -0.007 to +0.007, and every 95%
  interval includes zero.
- Under the strict invalid-output policy, DeepSeek RMD rank-weighted voting
  reaches 0.488 versus 0.452 for majority vote, but remains below RMD top-1.
  Qwen RMD rank-weighted voting reaches 0.582-0.584 versus 0.596 for majority.

Voting has a major parser limitation. Unparsed answers are excluded from the
vote, while answer parsing is also required for a trace to be labeled correct:

| Model | Correct parse rate | Incorrect parse rate | Prompts with no parsed answer |
|:---|---:|---:|---:|
| Qwen | 1.000 | 0.815 | 2 / 500 |
| DeepSeek | 1.000 | 0.224 | 136 / 500 |

The original parsed-only vote silently removed invalid traces, producing the
artificial DeepSeek result `majority = Oracle Pass@8 = 0.546`. The corrected
strict vote counts an unparsed response as an explicit invalid output and
scores invalid winners as failures.

The historical NPZ files do not contain generated text or token arrays, so the
missing answers cannot be reparsed. Future collections now persist both token
strings and generated text, and use balanced-brace parsing for nested
`\\boxed{}` / `\\fbox{}` answers.

### Fair Supervised RMD Probe

Best MATH-500 AUSC across configured layers:

| Model | Entropy | Unsupervised RMD | Old entropy+raw LR | Entropy+RMD LR | Gain over entropy | Gain over unsupervised RMD |
|:---|---:|---:|---:|---:|---:|---:|
| Qwen | 0.621 | 0.721 | 0.701 | 0.737 | +0.116 | +0.016 |
| DeepSeek | 0.500 | 0.633 | 0.620 | 0.639 | +0.139 | +0.006 |
| Llama | 0.384 | 0.493 | 0.465 | 0.507 | +0.123 | +0.013 |
| DeepSeek-Llama | 0.442 | 0.506 | 0.481 | 0.526 | +0.084 | +0.020 |

The fair supervised probe confirms that the old supervised baseline was using
the weaker raw geometry. Entropy+RMD is best in every model, but most of its
signal is already present in the unsupervised RMD score.

### One-Class Mechanism Sweep

Mean pooled ROC-AUC across each model's three sparse layers:

| Model | RMD dim 8 | RMD dim 32 | RMD dim 128 | Raw Ledoit-Wolf dim 128 |
|:---|---:|---:|---:|---:|
| Qwen | 0.717 | 0.762 | 0.772 | 0.379 |
| DeepSeek | 0.867 | 0.870 | 0.869 | 0.225 |
| Llama | 0.750 | 0.778 | 0.781 | 0.396 |
| DeepSeek-Llama | 0.783 | 0.786 | 0.792 | 0.352 |

- Diagonal, empirical-ridge, and Ledoit-Wolf target-only Mahalanobis AUCs
  differ by less than 0.001 throughout the sweep.
- Background subtraction is the load-bearing mechanism. Target-only distances
  are often strongly anti-predictive, especially for DeepSeek, while RMD is
  strongly predictive.
- A universal rank-1 mechanism is rejected. DeepSeek reaches its plateau near
  dimension 8 and DeepSeek-Llama near 4-8, while Qwen and Llama continue to
  improve through 64-128 dimensions.
- Input normalization does not provide a consistent advantage over ordinary
  RMD once more than a few components are retained.

### Current Interpretation

1. RMD is not merely a prompt-difficulty signal. For DeepSeek it is a strong
   trace-level correctness signal and a useful within-prompt selector.
2. The same score is model-conditional. Qwen RMD is primarily useful for
   between-prompt abstention and provides no reliable top-1 reranking gain.
3. The mechanism is relative geometry, not covariance estimation. Subtracting
   the generic background distribution reverses a strongly misleading raw
   distance signal.
4. Variance structure predicts application fit: within-prompt AUC tracks top-1
   gain, and prompt-score/pass-rate correlation tracks selective-prediction
   gain. These correlations remain exploratory because there are only two
   models and three correlated layers per model.
5. ICC alone is not an application selector. It is essentially uncorrelated
   with selective-prediction gain in the current conditions.

### Limitations and Compatibility

- The bootstrap resamples fixed OOF predictions; it does not refit PCA and
  covariance references inside every bootstrap replicate.
- Prompt-selection voting is confounded by answer-parser missingness.
- Application-alignment correlations reuse layers from the same models and are
  not independent replications.
- The Qwen and DeepSeek checkpoint comparison is not a clean causal
  distillation intervention because their training lineages differ.
- Selective-prediction results currently lack paired problem-bootstrap
  intervals for scorer differences.

### Next Experiments

| Priority | Experiment | Purpose | Cost |
|---:|:---|:---|:---|
| 1 | Add paired bootstrap intervals for selection and selective AUSC deltas | Quantify application-level uncertainty | Cheap reanalysis |
| 2 | Length-matched and confidently-wrong controls | Test whether RMD remains informative beyond length and confidence | Cheap reanalysis |
| 3 | Replicate enriched decomposition on Llama and DeepSeek-Llama Best-of-N traces | Test whether application alignment generalizes across architecture families | Requires Best-of-N inference |
| 4 | Matched Qwen2.5-Math-7B comparison | Separate reasoning distillation from base-model/math-training differences | Requires inference |

## Logging Convention

For every completed experiment, append:

1. exact DVC stage and parameterization;
2. artifact paths and schema;
3. primary point estimates and uncertainty;
4. interpretation and claims ruled in or out;
5. limitations and next dependent stage.
