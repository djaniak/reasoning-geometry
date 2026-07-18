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

## 2026-06-14: Truncation-Confound Audit of the DeepSeek Within-Prompt Result

### Status

| Experiment family | Conditions | Status |
|:---|:---|:---|
| Within-prompt decomposition re-audit | Qwen and DeepSeek, existing 500x8 OOF CSVs | Complete (reanalysis, no new compute) |

This is a code- and CSV-level audit of the within-prompt correctness claim, not a
new collection run. No NPZ/hidden-state access was used; all numbers come from the
already-written OOF CSVs.

Artifacts (inputs, unchanged):

- `results/deepseek_bestofn_full/math500/math500_prompt_decomposition_oof.csv`
- `results/qwen_bestofn_full/math500/math500_prompt_decomposition_oof.csv`

Code changes:

- `prompt_decomposition.py`: added `is_unparsed`, `truncation_report`,
  `parseable_within_prompt_metrics`; `analyze_oof_scores` now emits a top-level
  `truncation` block and per-layer `truncation` + `parseable_only` blocks; new
  `--max_new_tokens` CLI arg (inferred from max observed length if omitted);
  Markdown report gains a truncation/parseability section.
- `dvc.yaml`: `evaluate_prompt_decomposition` now passes
  `--max_new_tokens ${item.max_new_tokens}` so capped-trace diagnostics are exact.
- `tests/test_prompt_decomposition.py`: added 5 tests (26 pass).

### Mechanism

`collect_data.py:313`:
`is_correct = answers_match(predicted_answer, gold) if (predicted_answer and gold) else False`.
Any trace with no parseable final answer is auto-labeled incorrect. The
decomposition consumed `is_correct` with no parseability filter, so non-answers
entered the "incorrect" class.

### Primary findings (per layer, all 4000 traces/layer)

| Quantity | DeepSeek | Qwen |
|:---|---:|---:|
| Unparsed (no final answer) | 1814/4000 (45.4%) | 328/4000 (8.2%) |
| Of unparsed, length-capped at max_new_tokens | 99.4% (1804 at exactly 2048) | ~all at 1024 |
| Unparsed share of the incorrect class | 77.6% | — |
| within_macro RMD, ALL traces (L7/14/21) | 0.931 / 0.931 / 0.933 | 0.557 |
| within_macro RMD, PARSEABLE-only | 0.266 / 0.274 / 0.279 | 0.503 |
| within_macro entropy, PARSEABLE-only | 0.348 | 0.660 |
| Mixed-prompt count, ALL -> PARSEABLE | 166 -> 13 | 131 -> 117 |

DeepSeek `max_new_tokens=2048` is too small for R1-Distill on MATH500: 45% of
generations hit the cap before emitting `\boxed{}`. RMD scores these as strongly
anomalous (mean rmd_score correct=0.42, parseable-wrong=0.36, unparsed=0.11 at
L7; gap widens at deeper layers). Mean length: correct=1371, parseable-wrong=1455,
unparsed=2043.

### Claims ruled out

- RULED OUT (high confidence): "DeepSeek within-prompt AUC ~0.93 measures
  within-trace reasoning correctness." It is overwhelmingly a truncation /
  termination detector. Removing non-answers collapses the mixed-prompt set 166->13
  (92% of within-prompt mixedness was correct-vs-truncated, not correct-vs-wrong).
- RULED OUT (high confidence): the cross-model thesis "distillation reshapes
  geometry from between-problem solvability (Qwen) to within-trace correctness
  (DeepSeek)" as currently evidenced. The Qwen(0.55) vs DeepSeek(0.93) within-prompt
  gap tracks the differential truncation rate (8% vs 45%), not distillation. Qwen
  RMD is at chance within-prompt with or without filtering.
- SUPERSEDES "Current Interpretation" point 1 (2026-06-14 entry, line ~144) and
  upgrades the limitation at line ~162 from "confounded by parser missingness" to
  "dominated by truncation" for the within-prompt metric specifically.

### Claims still standing

- What RMD genuinely detects here is degenerate / non-terminating generations.
  That is real and plausibly useful for Best-of-N rejection, but is confounded
  with length and is not evidence of within-trace correctness geometry.
- The parseable-only contrast (n=13 mixed prompts) is too small to pin RMD's true
  within-prompt sign; the only firm claim is that the 0.93 headline does not survive.

### Limitations

- Parseable-only DeepSeek estimate rests on 13 mixed prompts -> noisy.
- True lengths of truncated traces are censored at 2048; existing data cannot say
  what `max_new_tokens` is sufficient.

### Next dependent stage

- BLOCKER for the Llama decomposition: `deepseek_llama` is also `max_new_tokens=2048`
  (params.yaml) and will inherit the identical artifact. Before the full 500x8
  campaign, run a small smoke test (limit ~30, T=0.6) at a raised budget (try 8192)
  on `deepseek` and `deepseek_llama`, measure cap-hit rate, pick the smallest budget
  with single-digit truncation (watch hidden-state storage ~ tokens x layers), then
  collect full. Do NOT run full 500x8 at 2048.

## 2026-07-11: Prompt-Local RMD and Current Evidence Reconciliation

### Status

| Experiment family | Condition | Status |
|:---|:---|:---|
| Prompt-local RMD | Qwen MATH-500, 500 prompts x 8 traces, layers 7/14/21 | Complete |
| Prompt-local top-1 selection | Same Qwen OOF scores | Complete |
| DeepSeek-Qwen budget probe | 8192 tokens, 24 traces | Complete; 12.5% capped/unparsed |
| DeepSeek-Llama budget probe | 12288 tokens, 24 traces | Complete; 0% capped/unparsed |
| DeepSeek prompt-local RMD | Historical 2048-token Best-of-N data | Deliberately not interpreted; truncation-contaminated |
| Clean cross-model prompt decomposition | Re-collected Best-of-N data | Not run |

Artifacts:

- `results/qwen_bestofn_full/math500/math500_prompt_decomposition_results.json`
- `results/qwen_bestofn_full/math500/math500_prompt_decomposition_oof.csv`
- `results/qwen_bestofn_full/math500/math500_prompt_decomposition_report.md`
- `results/qwen_bestofn_full/math500/math500_prompt_selection_results.json`
- `results/qwen_bestofn_full/math500/math500_prompt_selection_report.md`
- `results/truncation_probe/deepseek_8192.json`
- `results/truncation_probe/deepseek_llama_12288.json`

### Prompt-Local Protocol

For every held-out prompt and trace, the score uses the global OOF PCA and
correct-trace reference fitted on training prompts. Its local background is a
diagonal Gaussian fitted to tokens from the other seven attempts of that same
held-out prompt. The scored trace is excluded from its local background. The
fixed-orientation confidence score is the mean local-background distance minus
the global raw correct-manifold distance.

This is a quick test of whether removing prompt-shared semantic variation
reveals a same-prompt correctness residual. It uses no correctness labels from
the held-out prompt, but it is transductive because sibling attempts are
available at scoring time.

### Primary Results

| Layer | Prompt-local pooled AUC | Prompt-local centered AUC | Prompt-local within pair AUC | ICC | Top-1 Pass@1 |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.379 | 0.501 | 0.499 | 0.965 | 0.558 |
| 14 | 0.402 | 0.480 | 0.480 | 0.955 | 0.532 |
| 21 | 0.320 | 0.523 | 0.529 | 0.940 | 0.548 |

Selection references are random trace `0.557`, strict majority vote `0.596`,
and oracle Pass@8 `0.676`. Prompt-local RMD does not improve on random and is
consistently below majority vote.

For comparison, global RMD pooled AUC is `0.736/0.763/0.786` and global RMD
within-prompt pair AUC is `0.551/0.550/0.602` at layers 7/14/21. Prompt-local
subtraction removes the useful between-prompt component without exposing a
strong same-prompt component.

On parseable-only traces (117 mixed prompts), prompt-local within-macro AUC is
`0.446/0.436/0.483`, while global RMD is `0.503/0.515/0.574` and log-probability
is `0.649` at every layer. The apparent L21 all-trace prompt-local pair AUC of
`0.529` therefore does not survive the stricter correctness population.

### Interpretation

- Rejected for this estimator and Qwen dataset: same-prompt full-trace residual
  geometry is sufficient for correctness ranking.
- Supported: the useful global RMD signal is largely tied to prompt-level
  semantic/difficulty structure rather than an attempt-specific offset that can
  be recovered with a sibling-trace Gaussian.
- Supported: full-trace averaging is likely too coarse for local arithmetic,
  sign, or late-answer errors. The next geometry tests should localize scoring
  to high-entropy tokens, the trace tail, answer regions, or step transitions.
- This negative result does not rule out all prompt-conditional geometry. It
  rules out this simple leave-one-trace-out diagonal local-background method on
  Qwen MATH-500.

### Length and Truncation Context

The Qwen global RMD-minus-length contrast is strongest at L21: pooled `+0.055`
with 95% CI `[+0.021, +0.093]`, centered `+0.092` with
`[+0.047, +0.134]`, and within macro `+0.116` with `[+0.065, +0.171]` on all
traces. Parseable-only within-prompt performance is much weaker, so these
all-trace contrasts must not be presented as clean trace-correctness estimates.

The budget probes establish collection settings, not final scientific results:

| Model | Budget | n | Capped | Unparsed | Completed p95 | Completed max |
|:---|---:|---:|---:|---:|---:|---:|
| DeepSeek-R1-Distill-Qwen-7B | 8192 | 24 | 12.5% | 12.5% | 3924 | 5193 |
| DeepSeek-R1-Distill-Llama-8B | 12288 | 24 | 0% | 0% | 10237 | 11163 |

### Repository and DVC State

The result files exist, but the current worktree is not globally DVC-clean.
`dvc status` reports many changed dependencies because analysis and collection
code plus `params.yaml` have evolved since `dvc.lock`. In particular,
`evaluate_prompt_decomposition@0` and `evaluate_prompt_selection@0` report
changed dependencies despite the new Qwen outputs being present. Do not treat
an existing artifact as proof that its current stage definition is reproduced.

No files were staged or committed as part of this documentation update.

### Next Dependent Experiments

1. Implement and run high-entropy-token and tail-only RMD on the existing Qwen
   OOF protocol. These are the smallest tests of localized error geometry.
2. Add answer-cluster geometry to prompt selection using the existing enriched
   OOF CSV before collecting more hidden states.
3. Re-run parseable-only selective prediction with paired problem-bootstrap
   intervals against length, entropy/log-probability, and a trained linear
   probe. This determines whether the abstention application survives.
4. Only after the cheap gates pass, collect clean Best-of-N data for additional
   model families using architecture-specific token budgets. Do not rerun the
   old DeepSeek 2048-token decomposition as evidence.

## Logging Convention

For every completed experiment, append:

1. exact DVC stage and parameterization;
2. artifact paths and schema;
3. primary point estimates and uncertainty;
4. interpretation and claims ruled in or out;
5. limitations and next dependent stage.
