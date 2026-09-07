# Hidden-State Geometry Improves Post-Generation Abstention on MATH-500

Working manuscript draft, 2026-09-07. Full-refit results remain pending.
Authors, affiliations, release identifiers, and final figures are not yet supplied.
The numerical source is the [claim–evidence table](2026-09-07-claim-evidence-table.md).

## Abstract

Does hidden-state geometry improve confidence estimates after several answers have
already been generated? We evaluate a relative Mahalanobis distance feature on
500 MATH-500 problems with eight generations per problem, using three 7–8B models.
Adding the feature to length, token entropy, log probability, and vote agreement
reduces area under the risk–coverage curve by 0.0520, 0.0284, and 0.0469. Paired
bootstrap intervals exclude zero for the frozen fitting pipeline. Additional
answer-distribution and confidence-weighted-vote controls retain the increment
where their required inputs are available. However, conditioning on the other
models’ gold-scored success rates reduces the increment by 87–98%, and none of
the residuals survives the registered multiple-comparison correction. We therefore
interpret the feature as substantially a prompt-difficulty proxy. Whole-trace and
random-window scores show that a tail-specific interpretation is not established
on the two distilled models. The evidence supports post-generation abstention
within this benchmark and budget, while limiting claims about reasoning quality,
within-problem answer verification, and sample allocation.

## 1. Introduction

A confidence score can help decide which answers to return. When a model generates
several answers to the same problem, their agreement already provides confidence
information. A hidden-state score is useful only if it adds information beyond the
available output measures under a specified prediction target and budget.

We study whether the plurality answer is correct after eight completed generations.
Our comparison adds one geometric feature to a fixed baseline of length, entropy,
log probability, and vote agreement. The contribution is the controlled evaluation,
not a new distance statistic. We distinguish this practical comparison from the
question of what the feature measures. A score can improve abstention by identifying
hard problems without identifying which competing solution is correct.

The primary population includes all 500 problems. Earlier analyses excluded some
capped or unparseable outputs. Such filtering changes the question by conditioning
on outcomes of the generation process. We retain those earlier analyses as
sensitivity results and base current claims on correctness available at the stated
budget.

## 2. Related work

Vazhentsev et al. define token-level density-based uncertainty measures, including
relative Mahalanobis distance and whole-sequence aggregation, and combine density
features with output probability. Our use of whole-trace RMD follows that precedent;
we test its contribution with vote agreement included in the baseline.
[Token-Level Density-Based Uncertainty Quantification](https://arxiv.org/abs/2502.14427).

Orgad et al. study correctness information in hidden states, including answer-token
localization and relations to consistency across sampled answers. This motivates
our agreement controls and the distinction between pooled trace discrimination
and discrimination among answers to the same problem.
[LLMs Know More Than They Show](https://arxiv.org/abs/2410.02707).

Kirin reports hidden-state information before the answer region in a looped model.
On Horizon Logic, hidden-plus-shortcut scores improve accuracy–coverage area by
0.0124 [0.0048, 0.0207], with 92.6% versus 88.4% selective accuracy at 70% coverage.
That evaluation reports coverage 0.50–1.00. Its prediction unit, model, and timing
differ from ours, so the effect sizes are not directly ranked. Our tail window can
include answer tokens, and we do not establish a pre-answer result.
[Operational Proto-Introspection, v3 §8.6 and Appendix L.1](https://arxiv.org/html/2607.18553v3).

We also evaluate available DeepConf statistics and confidence-weighted-vote features
as stronger output controls. Exact inputs exist for the two distilled models but
are missing from the current Qwen cache.
[DeepConf](https://arxiv.org/abs/2508.15260).

## 3. Experimental setup

### Models, budget, and target

We use Qwen2.5-7B-Instruct, DeepSeek-R1-Distill-Qwen-7B, and
DeepSeek-R1-Distill-Llama-8B. Each has eight cached generations for each of the
500 MATH-500 problems. Generation caps are 1,024, 8,192, and 12,288 tokens,
respectively. The frozen geometry layers are 21, 21, and 24.

The target is correctness of the plurality answer among parseable generations.
Vote agreement is the fraction of parseable answers equal to that winner. When
answers tie in count, the implementation uses the highest per-trace log-probability
score among tied answers; a remaining tie follows the deterministic answer order.
If no answer is parseable, the outcome is failure. Capped and unparseable traces
remain in the primary population. This is fixed-budget correctness, not eventual
correctness if more tokens were allowed.

### Features and fitting

Let B0 contain prompt-aggregated length, token entropy, log probability, and vote
agreement. B1 adds `rmd_tail_q20`: negative mean tokenwise relative Mahalanobis
distance over the final 20% of each trace, averaged across the prompt’s traces.
The sign makes larger feature values correspond to smaller relative distances.
Reference geometry uses training prompts, with PCA dimension 128 and the frozen
reference-fitting settings. The prompt-level logistic readouts use the recorded
five prompt folds. All siblings remain assigned to their prompt’s fold.

The geometry feature requires access to states from existing generations. It adds
no generations, but this is not a claim of zero computation, memory, or fitting cost.
The readouts are evaluated after all eight generations; these experiments do not
measure online early stopping or pre-answer confidence.

### Metrics and uncertainty

We rank prompts by their predicted correctness and report area under the
risk–coverage curve (AURC; lower is better). AUACC is the corresponding
accuracy–coverage area on the same grid. Paired AURC differences are primary;
AUROC is a discrimination diagnostic. Absolute levels across models have different
base accuracies and should not be interpreted as comparable improvements.

The frozen headline uses seed 42 and 1,000 paired prompt-bootstrap draws. Its
intervals hold predictions fixed and therefore exclude variation from refitting
geometry, folds, and readouts. Control runs have their own recorded bootstrap
settings. We quote each interval from its named source rather than substitute
intervals from another run with the same point estimate. A stored bootstrap
p-value of zero denotes finite resampling resolution, not a zero probability.

## 4. Results

### 4.1 Geometry improves the fixed-budget abstention baseline

| Model | B1 − B0 AURC | 95% paired bootstrap interval |
|---|---:|---|
| Qwen2.5-7B-Instruct | −0.0520 | [−0.0845, −0.0218] |
| DeepSeek-R1-Distill-Qwen-7B | −0.0284 | [−0.0526, −0.0048] |
| DeepSeek-R1-Distill-Llama-8B | −0.0469 | [−0.0743, −0.0162] |

Source: each model’s committed `math500_incremental_abstention_results.json`,
`populations.full_population.paired_deltas.B1_minus_B0_aurc`.

Adding answer-distribution entropy to B0 does not show a significant improvement.
Geometry added on top of that strengthened baseline retains negative AURC
increments on all three models, with Holm-adjusted p-values 0.020/0.032/0.032.
Within the unanimous-answer stratum, the geometry feature alone has AUROC
0.827/0.717/0.739 on 302/424/253 prompts. This stratum result is a discrimination
diagnostic, not another measurement of the multivariate AURC increment.

For the two distilled models, adding geometry to B0 plus the DeepConf tail-q20
weighted-vote feature gives −0.02849 [−0.05259, −0.00427] and
−0.04630 [−0.07775, −0.01857]. This comparison includes the DeepConf feature on
both sides. It is distinct from comparing frozen B1 against B0 plus DeepConf.
The missing Qwen inputs prevent a three-model DeepConf claim.

### 4.2 Most of the increment overlaps with peer-measured difficulty

The diagnostic adds the other two models’ gold-scored pass rates to both B0 and
B1. It uses information unavailable for an unlabeled deployment prompt.

| Model | Geometry increment given peers | 95% interval | Attenuation | Holm p |
|---|---:|---|---:|---:|
| Qwen | −0.0067 | [−0.0164, −0.0002] | 87% | 0.126 |
| DeepSeek-Qwen | −0.0006 | [−0.0016, +0.0003] | 98% | 0.388 |
| DeepSeek-Llama | −0.0025 | [−0.0069, +0.0026] | 95% | 0.388 |

Two raw intervals overlap zero, triggering the registered stop rule. None of the
three residuals survives Holm correction. Following that rule, we describe the
increment as substantially a prompt-difficulty proxy. The attenuation is
observational and does not identify a causal mechanism or prove a zero residual.
It limits the interpretation of the practical gain without changing its measured
value against B0. The earlier cap-free run did not trigger this rule; that
population-dependent result remains secondary.

### 4.3 The exact cutoff is not required; localization claims are limited

The registered window rule requires 10%, 20%, and 50% tail features each to
improve AURC over B0 on every model. All nine intervals exclude zero. Qwen’s
50% result has an upper endpoint of −0.0012, so this cell is close to zero.
The 10% point estimate is more favorable than 20% on all models, but this does
not establish superiority or change the frozen feature.

On the distilled models, whole-trace RMD alone gives increments of −0.0287 and
−0.0445. Adding the tail to B0 plus whole-trace RMD gives −0.0030
[−0.0109,+0.0035] and −0.0041 [−0.0118,+0.0035]. These conditional increments
are unresolved. On Qwen the same conditional contrast is −0.0464
[−0.0724,−0.0224]. This is not a direct comparison between the separate readouts.

Random 20% windows yield increments of −0.0295 and −0.0444 on the distilled
models, near the corresponding tail estimates of −0.0284 and −0.0469.
Qwen’s random-window estimate is −0.0184 versus −0.0520 for the tail. No paired
random-versus-tail contrast was computed. These point estimates motivate a
limited interpretation; they establish neither equivalence nor superiority.
The saved sensitivity run followed an audit-time in-memory inspection, recorded
in the experiment log. The original cutoff rule predates that inspection.

### 4.4 Other applications and competing signals

The last-token probe’s pooled trace AUROC is 0.901/0.914/0.903, compared with
0.644/0.582/0.718 averaged within mixed-correctness prompts. Only 117/49/158
prompts contribute to those within-prompt estimates. The DeepSeek-Qwen macro
interval includes 0.5. These differences show why pooled discrimination alone
cannot establish sibling verification.

In six comparisons with a single additional peer generation, raw paired
intervals favor geometry once, the peer once, and are inconclusive four times.
The comparison family is post hoc and the compute is not exactly matched.

The allocation precheck passes on the primary population and fails on two models
on the cap-free population. Its draw ranges are not confidence intervals, and
an actual allocation-policy benefit remains unestablished. We make no general
positive or negative allocation claim.

## 5. Refit stability and limitations

[PENDING FULL-REFIT RESULTS.] The two distilled models have results at seeds
42/101/202. Their B1−B0 point estimates retain their sign, with ranges
[−0.0343,−0.0240] and [−0.0469,−0.0294]. Seed 42 reproduces the frozen fit;
only 101 and 202 are new partitions. Qwen has no new-partition result yet.
All missing peer steps and seed 303 are required to complete the registered
protocol. Small observed ranges do not establish bootstrap interval coverage.

The evidence covers one dataset, one eight-generation budget, three related
7–8B models, and one collected trace set. Model architecture, distillation,
trace length, and accuracy vary together. We do not attribute differences
causally to distillation. Answer-region inclusion prevents a pre-answer claim.
The refit sweep changes partitions of existing traces, not collection randomness.

## 6. Discussion

The useful empirical result is an improvement in confidence ranking after
completion of the generation budget. The difficulty diagnostic limits a stronger
claim that geometry measures reasoning quality independent of task difficulty.
The evaluation also separates prompt-level abstention from choosing among
siblings and from deciding where to spend additional samples. These require
different targets and cannot be inferred from pooled correctness discrimination.

## Release checklist for this draft

- Replace the pending refit paragraph after all registered checks finish.
- Verify generation sampling settings, model revisions, and reference-fitting
  details against the release configuration before submission.
- Add final risk–coverage figures from the frozen source predictions.
- Add authors, affiliations, bibliography formatting, and the reproducible release
  identifier. Check the cited paper versions and integration conventions.
- Verify a clean checkout can inspect all cited JSONs; distinguish this from a
  full rerun, which requires the separately stored inputs and compute.
