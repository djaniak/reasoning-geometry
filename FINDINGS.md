# Hidden-State Geometry Predicts Math Reasoning Errors

*Qwen2.5-7B-Instruct and DeepSeek-R1-Distill-Qwen-7B on MATH-500 and GSM8K.*
*Includes dense 14-layer sweep (Qwen) and temperature sampling (DeepSeek T=0.6).*
*Code: `collect_data.py`, `analyze.py`. Pipeline: `dvc.yaml`. Summary: `results/SUMMARY.md`.*

---

> ## ⚠️ Correction & contamination notice (2026-06-19)
>
> A truncation artifact was found that contaminates many DeepSeek results below.
> Read this before trusting any DeepSeek headline number. Full audit:
> `EXPERIMENT_LOG.md` (2026-06-14 audit entry).
>
> **Mechanism.** `collect_data.py` auto-labels any trace with no parseable `\boxed{}`
> as `is_correct=False`. With the generation cap, a large fraction of DeepSeek traces
> hit the cap *without emitting an answer* and were counted as "wrong." Mahalanobis/RMD
> then flags these off-manifold/non-terminating traces — so geometry was substantially
> detecting **truncation**, not **wrong reasoning**.
>
> **Measured contamination (cheap metadata scan, no rerun):**
> | Data | unparsed | length-capped | unparsed share of "incorrect" class |
> |---|---:|---:|---:|
> | DeepSeek greedy MATH-500 (main table, selective pred, stratification) | **43%** | **51%** | **75–79%** |
> | DeepSeek Best-of-N MATH-500 (within-prompt decomposition) | **45%** | ~55% | **78%** |
> | Qwen greedy MATH-500 | 8% | 8% | 17% |
> | Qwen Best-of-N MATH-500 | 8% | 8% | — |
>
> **RETRACTED:** the 2026-06-14 claim *"DeepSeek RMD is genuinely trace-level,
> within-prompt AUC 0.927–0.931"* (below) is the artifact. On parseable-only Best-of-N
> traces the within-prompt mixed-prompt set collapses 166→13 and within-macro falls to
> ~chance (0.27). The "strongest paper spine" built on it is withdrawn.
>
> **De-confounded estimate we DO have (Best-of-N OOF, parseable-only, pooled AUC):**
> DeepSeek correctness RMD **0.636** vs length **0.545** vs entropy 0.515 (was 0.887 with
> truncated traces included). The RMD-over-length margin (+0.05–0.09, CI excludes zero)
> survives and is **between-prompt (solvability)**; within-prompt (per-attempt) is at chance.
>
> **Contaminated and NOT yet re-validated (pending parseable-only rerun via
> `selective_prediction.py --exclude_unparsed`):** the main-results table (DeepSeek
> combined 0.859), selective-prediction AUSC (DeepSeek 0.633), and the difficulty/subject
> stratification — all DeepSeek-greedy-derived, where 75–79% of the "incorrect" class is
> non-answers. Treat their DeepSeek numbers as upper bounds.
>
> **Largely clean:** Qwen results (8% contamination) and any signal evaluated on
> parseable traces or about *geometry* rather than the correctness label (mechanism /
> low-dim / transfer), though AUC-based ones still warrant a parseable check.
>
> **Cross-model caveat:** the "DeepSeek shows a much larger geometry effect than Qwen"
> story tracks the **differential truncation rate (43% vs 8%)**, not cleanly distillation.

## Current evidence (updated 2026-07-31)

> **Length control (2026-07-31): the between-prompt claim strengthens.** RMD's
> advantage over length is not just a margin on the same ranking — its
> length-orthogonal component alone reaches +0.161 [+0.128, +0.194] AURC over an
> uninformative scorer on Qwen and +0.107 [+0.077, +0.135] on DeepSeek, while
> `entropy` and `logprob` fall to zero on DeepSeek (Holm 1.000). A supervised
> LDA on the same activations does **not** reliably beat RMD once length is
> partialled out of both (Holm 0.090 Qwen / 0.126 DeepSeek), which is the
> strongest available form of the label-light argument. Caution: the raw
> Spearman-vs-length table looks like a collapse for RMD on DeepSeek (rho +0.82)
> and is misleading read alone. Full entry: "Supervised Probe Ceiling and Length
> Residualization" below.

> **Scope correction (2026-07-29): the localization result is Qwen-specific.**
> A pre-registered cross-model gate on DeepSeek-R1-Distill-Qwen-7B (MATH-500
> Best-of-8, 8,192-token budget) **failed on both confirmatory tests**:
> `rmd_high_entropy_q20 − rmd` = +0.004 [−0.016, +0.027] and
> `− rmd_random_q20` = +0.001 [−0.023, +0.026] at L21 (Holm p = 1.000 for both).
> Qwen's +0.058 effect lies outside the DeepSeek interval, so this is an
> informative null rather than an underpowered one — though power is lower
> (49 mixed prompts vs 117). Notably **every within-prompt AUC on DeepSeek is at
> or below chance, including entropy and logprob**: the phenomenon is absent in
> that model, not merely undetected by geometry. Wherever this document says
> localization "replicates across layers", read that as *across layers within
> Qwen*. The `deepseek_llama` collect was cancelled by the gate.
>
> **The between-prompt claim did replicate.** On the same DeepSeek data, E1
> prompt abstention beats the length confound baseline —
> `rmd_tail_q20 − length` = +0.030 AURC [+0.014, +0.048], p < 0.001 — with the
> same scorer ordering as Qwen, though the effect is ~2.3x smaller and clears
> length on AURC only, not at 50% coverage. So the two regimes now separate
> cleanly across models: **geometry indicates which problems are hard, not which
> attempt is right.** Full entries: `EXPERIMENT_LOG.md` (2026-07-29, both).

The load-bearing result is the clean Qwen Best-of-8 MATH-500 rerun. Highest-entropy
20% RMD beats full-trace RMD at all three layers and beats a matched random-token
control. It is competitive with output baselines, has only suggestive incremental
probe value, and does not improve Best-of-N tie-breaking. The defensible claim is
prompt-level difficulty/abstention; within-prompt correctness geometry is small
**and does not generalize to the reasoning-distilled model tested.**

All DeepSeek 2,048-token analyses, temperature runs, transfer grids, old selective
prediction, and one-class sweeps are historical diagnostics. They remain on disk for
provenance but are excluded from the active DVC graph and current summary. The
parseable-only C1/C2 audit rejects the old low-dimensional-distillation claim and
weakens transfer to an exploratory result.

---

## Introduction

Token-level entropy — the standard uncertainty signal from language models — becomes
unreliable for reasoning-distilled models. When a model generates long chains of thought,
both correct and incorrect traces contain many high-entropy "thinking" tokens. Entropy
loses discriminative power because the model is visibly uncertain throughout, regardless
of whether it ultimately gets the answer right.

We test whether **hidden-state geometry** provides a complementary signal. The approach:
fit a reference manifold (PCA + Gaussian) on hidden states from correct traces, then
measure each trace's Mahalanobis distance from that manifold. The hypothesis is that
correct reasoning stays on-manifold while incorrect reasoning deviates — even when the
output distribution (entropy) looks similar token-by-token.

---

## Method

**Models**: Qwen2.5-7B-Instruct (greedy decoding); DeepSeek-R1-Distill-Qwen-7B (greedy).
Both are 28-layer Qwen2.5 architecture, hidden dim 3584 — direct comparison without
architectural confounds.

**Datasets**: MATH-500 (500 problems; 56% solved by Qwen, 43% by DeepSeek) and GSM8K
test set (1319 problems for Qwen, 1269 for DeepSeek; ~90% solved by both).

**Data collection**: Token-by-token autoregressive generation, hidden states captured at
layers 7, 14, 21 (~25/50/75% of depth). Per-token entropy from output logit distribution.

**Reference manifold**: PCA(128) on hidden states from correct traces only -> Gaussian
with regularized covariance. Mahalanobis distance measures deviation from the
correct-reasoning manifold in that PCA basis.

**Features**:
- *Entropy-only* (5): mean, max, std, 90th-pct, fraction-above-median
- *Mahalanobis-only* (7): mean, max, std, 90th-pct distance; mean/max distance at
  high-entropy tokens; entropy-Mahal correlation
- *Combined* (12): union of entropy and Mahalanobis features

**Evaluation**: Stratified 5-fold CV, fixed fold assignments across all conditions,
logistic regression with class-balanced weighting. Length-controlled baseline adds
log(trace_length) to entropy features.

---

## Main Results

All numbers below report at each model/dataset's **best combined layer** (the layer
where entropy + geometry together achieves highest AUC). Deltas are always vs the
entropy-only baseline; length-controlled deltas use entropy + log(trace_length) as
the baseline.

| Condition | Qwen MATH-500 | Qwen GSM8K | DeepSeek MATH-500 | DeepSeek GSM8K |
|---|---|---|---|---|
| N (correct / incorrect) | 281 / 219 | 1204 / 115 | 215 / 285 | 1144 / 125 |
| Best combined layer | L7 | L21 | L7 | L7 |
| Entropy-only | 0.713 | 0.760 | 0.776 | 0.728 |
| Mahalanobis-only | 0.742 | 0.690 | 0.826 | 0.806 |
| **Combined** | **0.772** | **0.781** | **0.859** | **0.835** |
| Raw Δ (combined − entropy) | +0.059 | +0.021 | +0.083 | +0.106 |
| Length-controlled Δ | +0.027 | +0.014 | +0.044 | +0.080 |

Note: the best *Mahalanobis-only* layer sometimes differs from the best combined layer.
For DeepSeek MATH-500, L21 Mahalanobis-only (0.827) marginally exceeds L7 (0.826);
we report at the best combined layer (L7, combined 0.859) since the combined model is
the primary claim.

**The core finding**: Geometry adds signal beyond entropy in all four conditions, even
after controlling for trace length. The effect is largest for DeepSeek (+4.4 to +8.0
length-controlled AUC points) and smaller but consistent for Qwen (+1.4 to +2.7 points).

**Why DeepSeek shows a larger effect**: For DeepSeek, Mahalanobis-only *already beats*
entropy-only in every condition — notably on GSM8K (0.806 vs 0.728, Mahal vs entropy). DeepSeek
generates long reasoning chains (max 2048 tokens) where both correct and incorrect
traces contain many high-entropy "thinking" tokens, neutralizing entropy as a
discriminator. But hidden-state geometry still separates the trajectories: correct
reasoning stays on-manifold, incorrect reasoning deviates even when the output
distribution looks similar token-by-token. For Qwen, entropy retains more signal
because the model answers more directly without extended reasoning, so the geometry
gain is smaller.

**Best layer differs by dataset, not by model.** L7 is best combined on MATH-500 for
both models; L21 is best for Qwen GSM8K (L7 for DeepSeek GSM8K). Plausible mechanism:
MATH-500 problems are structurally diverse (algebra, geometry, number theory), and
layer 7 encodes whether the model has correctly identified the problem type. GSM8K
problems are structurally similar — the signal emerges later where execution quality
matters.

## Retired pre-fix sweep (historical, not current evidence)

A later full summary now includes `llama`, `deepseek_llama`, `deepseek_temp`,
robust/relative Mahalanobis controls, and label-informed subspace analyses. That
broader sweep sharpens the story:

- **Raw one-class geometry is strongest for reasoning-distilled models.** DeepSeek remains
  the clearest result (`0.835` on GSM8K, `0.859` on MATH-500; length-controlled gains
  `+0.080` and `+0.044`). `deepseek_temp` on GSM8K is similarly strong (`+0.100`
  length-controlled). Qwen remains positive but modest. Base Llama is only slightly
  positive on GSM8K and is negative on MATH-500 under raw combined scoring.
- **Cross-architecture replication narrows the claim.** The earlier Qwen-family story
  does not generalize cleanly to Llama-family models under the same raw global
  Mahalanobis setup. The right claim is not "universal hidden-state geometry works";
  it is "raw one-class geometry works best in reasoning-distilled settings and degrades
  under architecture/training shifts."
- **RMD / norm-RMD are the strongest continuation.** Relative / robust Mahalanobis
  improves almost every condition and often rescues weak raw cells. The clearest example
  is Llama on MATH-500: raw combined is negative versus entropy, while robust variants
  become positive. Intuitively this is the right correction under architecture shift:
  raw Mahalanobis conflates correctness with background representation spread, while RMD
  subtracts that background and asks whether correct traces are specifically closer than
  the generic token cloud. That makes robust geometry the most defensible next main-table
  upgrade.
- **Low-rank contrast results are informative but not the same claim.** The contrast
  subspace and low-rank sweeps use correctness labels at fit time. They should be framed
  as supervised upper bounds or appendix analyses, not as a continuation of the current
  one-class geometry story.
- **Subject and difficulty effects are heterogeneous, not universal.** Some model-dataset
  cells show strong positive pockets, but others are flat or negative. The older
  "geometry/precalculus strongest, algebra weakest" phrasing is too broad for the full
  sweep. The safer claim is that stratification is model-dependent and useful for
  diagnosing where the detector works.
- **Cross-model transfer still supports a manifold-shape story.** Same-family transfer
  retains most native Mahalanobis signal while frozen classifiers transfer poorly. The
  manifold seems more shared than the decision rule.
- **Downstream control remains negative, but for two different reasons.** Best-of-N looks
  structurally mismatched: it asks a pooled trace-level score to rank samples within one
  problem, where majority vote has direct within-problem agreement signal. Prefix
  filtering is less settled: the current global-threshold policy may simply be the wrong
  implementation, so that branch should be framed as unresolved calibration mismatch
  unless prompt-conditioned normalization also fails.

## Experiment completion log (2026-06-08)

The previously missing dense-merge, selective-prediction, Best-of-N pilot, and
concordance stages are now complete and incorporated into `results/SUMMARY.md`.

- **Dense Qwen MATH-500 merge:** all 14 even layers from L0-L26 are present. Base
  combined scoring peaks at L8 (`0.760`), while one-class RMD is strongest later:
  raw RMD reaches approximately `0.827` at L20 and norm-RMD approximately `0.832`.
- **Selective prediction:** RMD improves over entropy for all four MATH-500 models.
  Best AUSC is `0.721` vs `0.621` for Qwen, `0.633` vs `0.500` for DeepSeek,
  `0.493` vs `0.384` for Llama, and `0.506` vs `0.442` for DeepSeek-Llama.
  Raw Mahalanobis is consistently weak; background subtraction is the load-bearing
  correction. ⚠️ **The DeepSeek/DeepSeek-Llama AUSC numbers are contamination-suspect**
  (43% of DeepSeek greedy traces are unparsed/truncated, all labeled incorrect — RMD can
  "abstain" on non-answers, inflating AUSC). Needs the parseable-only rerun
  (`selective_prediction.py --exclude_unparsed`) before use. Qwen (8%) is largely clean.
- **Best-of-N pilot (`N=8`, 100 prompts):** Qwen majority vote scores `0.690`,
  ahead of best raw combined geometry (`0.650`) and RMD-only (`0.620`).
  DeepSeek majority vote and best RMD-only both score `0.570`; RMD+entropy reaches
  `0.560`. The pilot therefore does not show a general geometry reranking win,
  although DeepSeek RMD matches majority vote.
- **Legacy concordance diagnostic:** Qwen raw-Mahalanobis concordance is `0.536` at
  L7 and below chance at L14/L21; DeepSeek is `0.248-0.264` across layers. These are
  descriptive in-sample raw-Mahalanobis diagnostics, not the confirmatory OOF/RMD
  decomposition.

## Outer-loop synthesis: confidence decomposition and mechanism (2026-06-14)

The enriched OOF decomposition, prompt-selection analysis, fair supervised
probes, application-alignment analysis, and one-class mechanism sweep are now
complete. They replace the earlier binary framing of "trace correctness versus
prompt difficulty" with a model-conditional result.

- **DeepSeek RMD is genuinely trace-level.** ⚠️ **RETRACTED — see the correction
  notice at the top of this file. This is the truncation artifact.** Across L7/L14/L21,
  within-prompt pairwise AUC is `0.927-0.931`. RMD exceeds entropy within prompts by
  `0.134-0.138`, with paired prompt-bootstrap intervals excluding zero. In
  top-1 selection it gains `0.109-0.111` Pass@1 over a random trace and
  `0.018-0.020` over the strong length baseline. *(All of this is driven by RMD ranking
  correct traces above unparsed/truncated ones, not above genuinely wrong answers; on
  parseable-only traces the within-prompt signal is at chance.)*
- **Qwen RMD is primarily a solvability signal.** Its pooled AUC rises to
  `0.736-0.786`, but within-prompt AUC is only `0.550-0.602`. It does not beat
  entropy, log-probability, or length within prompts with a confidence interval
  excluding zero, and top-1 Pass@1 differs from random by only
  `-0.007` to `+0.007`.
- **Relative geometry, not covariance estimation, is the mechanism.**
  Diagonal, empirical-ridge, and Ledoit-Wolf target-only Mahalanobis are
  numerically almost identical throughout the dimension sweep. Background
  subtraction changes the result by tens of AUC points and often reverses a
  strongly anti-predictive target-only distance.
- **The one-class signal is low-dimensional for distilled checkpoints, but not
  rank-1 or universal.** DeepSeek plateaus near dimension 8 and
  DeepSeek-Llama near 4-8. Qwen and Llama continue improving through
  dimensions 64-128. This is consistent with, but does not causally establish,
  a distillation-associated low-dimensional contrast.
- **A fair supervised RMD probe is strongest but only modestly improves on
  unsupervised RMD.** Best entropy+RMD AUSC is `0.737` for Qwen, `0.639` for
  DeepSeek, `0.507` for Llama, and `0.526` for DeepSeek-Llama. Gains over the
  best unsupervised RMD score are only `0.006-0.020`.
- **Application alignment is promising but exploratory.** Within-prompt AUC
  tracks top-1 gain, while prompt-score/pass-rate correlation tracks selective
  AUSC gain. ICC alone does not. The current correlations reuse three layers
  from only two models, so they are a hypothesis generator rather than a law.
- **Strict voting confirms the parser artifact and preserves an RMD
  advantage.** The answer parser succeeds on all correct traces but only
  `22.4%` of DeepSeek incorrect traces. Parsed-only voting had made DeepSeek
  majority vote equal Oracle Pass@8 (`0.546`). Counting unparsed traces as
  explicit invalid outputs lowers strict majority to `0.452`; RMD
  rank-weighted voting reaches `0.488`, while direct RMD top-1 remains best at
  `0.524-0.526`. For Qwen, strict majority is `0.596`, still above RMD top-1
  (`0.550-0.564`).
- **Historical answers cannot be recovered from these files.** The existing
  Best-of-N NPZs contain no generated text or token arrays. Future collection
  now persists generated text and uses balanced parsing for nested
  `\\boxed{}` and `\\fbox{}` expressions.

The strongest paper spine is now: ⚠️ **WITHDRAWN — rested on the retracted
within-prompt result. See the correction notice at the top.** The revised, defensible
spine is in the paper-strategy work: geometry is a length-and-entropy-beating
*between-prompt solvability* signal (for abstention / compute allocation), not a
within-prompt correctness reranker; prior "trace-correctness" readings were confounded
by trace length and truncation.

> ~~Relative hidden-state geometry separates correctness-specific structure from
> generic representation spread. Whether that signal supports abstention or
> within-prompt selection depends on its prompt-level decomposition, and that
> decomposition differs sharply across model training regimes.~~

The next smallest useful work is to add paired application-level uncertainty
and run length-matched/confidently-wrong controls. Broader
application-alignment claims require additional independent model conditions
rather than more layers from the same two checkpoints.

---

## Difficulty Stratification (MATH-500)

Difficulty stratification at each model's best combined layer (L7), 200-iteration
stratified bootstrap with 3-fold CV within each bootstrap sample for 95% CI.
Mahalanobis reference fitted fresh per stratum from correct traces in that stratum.

### DeepSeek

| Level | n (incorr) | Entropy AUC | Combined AUC | Δ | Combined 95% CI | Entropy 95% CI |
|---|---|---|---|---|---|---|
| 1 (easiest) | 43 (12) | 0.704 | 0.744 | +0.040 | [0.726, 1.000] | [0.614, 0.896] |
| 2 | 90 (31) | 0.740 | 0.831 | +0.091 | [0.854, 0.990] | [0.604, 0.879] |
| 3 | 105 (56) | 0.692 | 0.678 | −0.014 | [0.656, 0.899] | [0.538, 0.816] |
| 4 | 128 (80) | 0.771 | 0.808 | +0.038 | [0.803, 0.948] | [0.681, 0.879] |
| 5 (hardest) | 134 (106) | 0.790 | 0.897 | +0.107 | [0.915, 0.990] | [0.711, 0.904] |
| Easy (1-2) | 133 (43) | 0.722 | 0.843 | +0.122 | [0.819, 0.961] | [0.624, 0.834] |
| Hard (4-5) | 262 (186) | 0.813 | 0.884 | +0.071 | [0.865, 0.947] | [0.733, 0.873] |

### Qwen

| Level | n (incorr) | Entropy AUC | Combined AUC | Δ | Combined 95% CI | Entropy 95% CI |
|---|---|---|---|---|---|---|
| 1 (easiest) | 43 (8) | 0.614 | 0.786 | +0.171 | [0.691, 1.000] | [0.321, 0.900] |
| 2 | 90 (22) | 0.633 | 0.623 | −0.010 | [0.552, 0.917] | [0.380, 0.760] |
| 3 | 105 (44) | 0.564 | 0.474 | −0.090 | [0.480, 0.828] | [0.444, 0.749] |
| 4 | 128 (55) | 0.701 | 0.683 | −0.017 | [0.667, 0.878] | [0.590, 0.803] |
| 5 (hardest) | 134 (90) | 0.766 | 0.774 | +0.008 | [0.759, 0.954] | [0.682, 0.871] |
| Easy (1-2) | 133 (30) | 0.603 | 0.773 | +0.170 | [0.662, 0.917] | [0.448, 0.743] |
| Hard (4-5) | 262 (145) | 0.754 | 0.776 | +0.022 | [0.755, 0.884] | [0.669, 0.815] |

**DeepSeek shows consistent difficulty-level gains; Qwen is mixed.** For DeepSeek,
combined beats entropy at 5 of 5 individual levels (except level 3, Δ = −0.014), with
the largest gains at easy (levels 1-2, Δ = +0.122) and hard (levels 4-5, Δ = +0.071)
problems. The level 5 gain (Δ = +0.107) is the best-powered single-level result.

For Qwen, the picture is less clear. The easy group (1-2) shows a strong gain
(Δ = +0.170), but levels 2-4 individually show negative or flat deltas. The
hard group (4-5) shows a modest gain (Δ = +0.022). The wide bootstrap CIs reflect
the small per-level sample sizes — many individual-level results are not distinguishable
from zero.

The grouped Easy (1-2) result is the best-powered comparison. For DeepSeek: combined CI
[0.819, 0.961] vs entropy CI [0.624, 0.834] — minimal overlap. For Qwen: combined CI
[0.662, 0.917] vs entropy CI [0.448, 0.743] — overlapping but combined is higher.

---

## Subject Stratification (MATH-500)

Subject stratification at each model's best combined layer (L7). Mahalanobis reference
fitted fresh per subject from correct traces in that subject. 200-iteration stratified
bootstrap for 95% CI.

### DeepSeek

| Subject | n (incorr) | Entropy AUC | Mahal AUC | Combined AUC | Δ | Combined 95% CI |
|---|---|---|---|---|---|---|
| Algebra | 124 (42) | 0.631 | 0.614 | 0.622 | −0.009 | [0.645, 0.854] |
| Counting & Prob. | 38 (25) | 0.807 | 0.747 | 0.713 | −0.093 | [0.662, 1.000] |
| Geometry | 41 (28) | 0.707 | 0.800 | 0.707 | +0.000 | [0.558, 1.000] |
| Interm. Algebra | 97 (80) | 0.799 | 0.857 | 0.898 | +0.099 | [0.872, 1.000] |
| Number Theory | 62 (29) | 0.731 | 0.699 | 0.718 | −0.013 | [0.702, 0.970] |
| Prealgebra | 82 (34) | 0.700 | 0.637 | 0.683 | −0.017 | [0.674, 0.927] |
| Precalculus | 56 (47) | 0.752 | 0.833 | 0.833 | +0.081 | [0.451, 0.986] |

### Qwen

| Subject | n (incorr) | Entropy AUC | Mahal AUC | Combined AUC | Δ | Combined 95% CI |
|---|---|---|---|---|---|---|
| Algebra | 124 (31) | 0.660 | 0.633 | 0.668 | +0.008 | [0.614, 0.860] |
| Counting & Prob. | 38 (20) | 0.692 | 0.750 | 0.808 | +0.117 | [0.615, 1.000] |
| Geometry | 41 (25) | 0.780 | 0.553 | 0.553 | −0.227 | [0.543, 0.939] |
| Interm. Algebra | 97 (60) | 0.715 | 0.688 | 0.724 | +0.010 | [0.677, 0.925] |
| Number Theory | 62 (14) | 0.666 | 0.686 | 0.593 | −0.073 | [0.584, 0.947] |
| Prealgebra | 82 (29) | 0.581 | 0.536 | 0.515 | −0.065 | [0.509, 0.875] |
| Precalculus | 56 (40) | 0.671 | 0.542 | 0.535 | −0.135 | [0.480, 0.894] |

**Subject-level results are mixed and model-dependent.** The aggregate geometry gain
(visible in the main results) does not replicate uniformly across subjects. For
DeepSeek, only Intermediate Algebra (Δ = +0.099) and Precalculus (Δ = +0.081) show
clear positive combined deltas; most other subjects are flat or slightly negative.
For Qwen, Counting & Probability (Δ = +0.117) is the only clear winner, while
Geometry (Δ = −0.227) and Precalculus (Δ = −0.135) show substantial negative deltas
where adding Mahalanobis features hurts.

The wide bootstrap CIs reflect small per-subject samples — many effects are not
distinguishable from zero. The negative deltas likely reflect overfitting: with few
traces per subject (e.g. 8-14 incorrect), adding 7 Mahalanobis features to a 5-feature
entropy model can degrade performance in CV. This is consistent with the aggregate
result being positive while per-subject results are noisy.

**Interpretation**: The geometry signal is most reliably detected at the aggregate
level where sample sizes support the 12-feature combined model. Per-subject claims
require either larger datasets or dimensionality reduction of the Mahalanobis features.

---

## Cross-Model Transfer (historical exploratory result)

These numbers were produced before the clean-budget revalidation and are retained
only to document why the transfer branch was retired from the active pipeline.

Both models share the Qwen2.5-7B architecture but differ in training (base instruction-tuned
vs reasoning-distilled). Two transfer tests probe whether the geometry signal is model-specific
or shared.

### Geometry transfer

Fit the Mahalanobis reference on one model's correct traces, compute distances for the
other model's traces, then train+evaluate a classifier on the target model's data using
the cross-model features. This tests whether the *manifold shape* transfers.

| Eval model | Dataset | Layer | Native Mahal AUC | Cross Mahal AUC | Transfer % |
|---|---|---|---|---|---|
| DeepSeek | GSM8K | L21 | 0.831 | 0.821 | 99% |
| DeepSeek | MATH-500 | L14 | 0.788 | 0.796 | 101% |
| Qwen | GSM8K | L21 | 0.690 | 0.745 | 108%* |
| Qwen | MATH-500 | L7 | 0.742 | 0.711 | 96% |

\* Qwen GSM8K cross-model Mahal (0.745) *exceeds* native Mahal (0.690). Plausible:
Qwen's native manifold at L21 is poorly discriminative because 91% of traces are
correct (the manifold is nearly everything). DeepSeek's manifold provides a tighter
reference.

### Classifier transfer

Train a logistic regression on Model A's entropy + Mahalanobis features (using Model A's
own reference manifold), then evaluate that frozen classifier on Model B's features (using
Model A's reference for Mahalanobis computation). This tests whether the *decision boundary*
— the relationship between entropy and geometry — transfers across models.

| Eval model | Dataset | Layer | Clf Mahal AUC | Clf Combined AUC |
|---|---|---|---|---|
| DeepSeek | GSM8K | L7 | 0.600 | 0.713 |
| DeepSeek | MATH-500 | L7 | 0.705 | 0.723 |
| Qwen | GSM8K | L7 | 0.536 | 0.687 |
| Qwen | MATH-500 | L7 | 0.351 | 0.650 |

Only L7 results shown — L14 and L21 classifier transfer largely fails (AUC near or
below 0.5 in most conditions). Full per-layer results in `results/SUMMARY.md`.

**Geometry transfers; decision boundaries do not.** Cross-model Mahal retention is **not**
a single tight band: the full L7/L14/L21 grid in `results/SUMMARY.md` spans roughly **82%**
(weakest: DeepSeek on GSM8K at L7, native Mahal 0.806 → cross 0.663) up through **~101%**
at some cells (e.g. DeepSeek MATH-500 L14: 0.788 → 0.796), with Qwen GSM8K reaching
**above 100%** in all layers (native manifold there is weak; the other model’s reference
can help). **Late layers tend to retain native Mahal more reliably for geometry-only
transfer** (e.g. **94–99%** on DeepSeek at L14–L21), while **L7 can drop as low as ~82%**
on the same benchmark — i.e. early-layer geometry is more model-specific, late-layer
geometry more shared. That pattern fits a two-phase story (comprehension vs execution)
better than cherry-picking only the strongest cells.

A frozen classifier trained on one model still rarely works on the other: the
*relationship* between entropy and geometry differs across models even when manifold
shape partially aligns. (Classifier-transfer numbers in the table are shown at L7 only;
L14/L21 are worse — see `results/SUMMARY.md`.)

### Interpretation

The geometry transfer grid suggests the correct-reasoning manifold is **partially** shared
across models with the same architecture but different training, with **layer-dependent**
universality (late layers often closer to native Mahal than early layers on the same
dataset). The classifier transfer failure shows that the *feature-to-correctness mapping*
remains model-specific.

This connects to the Platonic Representation Hypothesis: models trained on the same
distribution converge on similar internal representations (manifold shape transfers),
but the functional use of those representations diverges (decision boundaries don't
transfer).

---

## Mechanistic Negatives

Two targeted analyses tested whether the geometry advantage localizes to specific tokens.
Both were null for DeepSeek. The null results *sharpen* the trace-level claim.

**Confident-wrong analysis**: Isolate only low-entropy tokens (bottom 25% globally).
Compare mean Mahalanobis distance at those tokens between correct and incorrect traces.

| Model | Dataset | p-value | Significant? |
|---|---|---|---|
| Qwen | MATH-500 | 0.10 | No |
| Qwen | GSM8K | <0.01 | Yes |
| DeepSeek | GSM8K | 0.91 | No |
| DeepSeek | MATH-500 | 1.00 | No |

For DeepSeek, the confident-wrong analysis is null everywhere — geometry does not
concentrate at individually confident tokens. The Qwen GSM8K result (p < 0.01) is the
exception; it does not replicate in the more diverse MATH-500 dataset.

**Interpretation**: The geometry signal is a *trajectory-level* property — a diffuse
deviation of the entire hidden-state path from the correct manifold, not a local
artifact at individual uncertain tokens. The intermediate representations encode a
correctness-relevant signal that is not surfaced in the output distribution. This is
what makes geometry complementary to entropy rather than redundant with it.

(Note: the data shows that hidden states carry this signal. It does not show that the
model *uses* this information — the geometric deviation may be a side effect of
processing that the model never acts on.)

---

## Trace-Length Confound

Incorrect traces might be systematically shorter or longer, and Mahalanobis summary
statistics could correlate with length.

### MATH-500

| Model | Layer | Entropy-only | Entropy+length | Raw Δ | Length-controlled Δ |
|---|---|---|---|---|---|
| Qwen | L7 | 0.713 | 0.778 | +0.059 | +0.027 |
| DeepSeek | L7 | 0.776 | 0.826 | +0.083 | +0.044 |

Trace length is itself predictive on MATH-500: adding it to entropy raises AUC by +0.065
(Qwen) and +0.050 (DeepSeek). The raw geometry delta roughly halves after controlling
for length. **Honest framing: after length control, geometry adds +2.7 AUC points for
Qwen and +4.4 for DeepSeek.** The signal is real but partially mediated by length.

### GSM8K

| Model | Layer | Raw Δ | Length-controlled Δ |
|---|---|---|---|
| Qwen | L21 | +0.021 | +0.014 |
| DeepSeek | L7 | +0.106 | +0.080 |

On GSM8K the length control changes little. For DeepSeek, +8.0 points survive length
control — this is not a length artifact.

---

## Dense Layer Sweep (MATH-500)

Qwen2.5-7B-Instruct on MATH-500, every 2nd layer (L0–L26, 14 layers total). Same model
and data as the sparse sweep — this resolves whether the geometry signal is monotone,
peaked, or concentrated at a specific transition.

| Layer | Mahal-only AUC | Combined AUC | Δ (raw) | Δ (len-ctrl) | Conf-wrong p |
|---|---|---|---|---|---|
| L0 | 0.616 | 0.716 | +0.002 | +0.002 | 4.4e-05 *** |
| L2 | 0.705 | 0.744 | +0.031 | +0.011 | 2.9e-05 *** |
| L4 | 0.689 | 0.740 | +0.027 | +0.013 | 9.1e-07 *** |
| L6 | 0.726 | 0.758 | +0.045 | +0.021 | 3.7e-03 *** |
| L8 | 0.733 | **0.760** | +0.047 | +0.018 | 5.7e-03 *** |
| L10 | 0.720 | 0.748 | +0.035 | +0.008 | 9.2e-03 *** |
| L12 | 0.703 | 0.744 | +0.030 | +0.006 | 3.3e-02 * |
| L14 | 0.678 | 0.728 | +0.015 | −0.006 | 3.7e-02 * |
| L16 | 0.682 | 0.727 | +0.014 | −0.007 | 6.6e-02 ns |
| L18 | 0.700 | 0.730 | +0.017 | −0.011 | 1.3e-01 ns |
| L20 | 0.725 | 0.749 | +0.036 | −0.001 | 2.6e-02 * |
| L22 | 0.712 | 0.743 | +0.030 | −0.002 | 1.2e-01 ns |
| L24 | 0.689 | 0.731 | +0.018 | −0.003 | 1.4e-05 *** |
| L26 | 0.728 | 0.754 | +0.040 | +0.015 | 6.7e-07 *** |

Entropy-only baseline: 0.713. Best combined layer: L8 (0.760). Sparse sweep had
L7 = 0.772 — the dense sweep's nearest layers L6/L8 hit 0.758/0.760, consistent within
fold variance.

**The signal is bimodal, not monotone.** Mahalanobis-only AUC rises from L0 (0.616) to
L8 (0.733), forming a first peak. It then dips to a trough at L14 (0.678) before
recovering to a second peak at L26 (0.728). L14 is consistently the weakest geometry
layer — both in the dense sweep and the original sparse results (sparse L14 combined =
0.728 vs L7 = 0.772 and L21 = 0.737).

**Two-phase interpretation**: The bimodal profile suggests geometry encodes correctness-
relevant information at two distinct processing stages:

1. **Early layers (L6–L10)**: Problem comprehension — whether the model has correctly
   identified the problem type and set up the right approach.
2. **Late layers (L20–L26)**: Solution execution — whether the intermediate computation
   has stayed on track. The second peak emerges where the model resolves final answer
   computation.
3. **L14 trough**: A transition zone where representations are mid-computation — too late
   to reflect problem setup, too early to reflect execution quality. The geometry signal
   is weakest here.

Note: length-controlled deltas are negative or near-zero for most layers beyond L12,
meaning much of the late-layer geometry gain is mediated by trace length. The raw signal
is still positive everywhere, but the length-independent component concentrates in
early layers (L0–L8).

**Confident-wrong tokens localize to early and late layers.** The confident-wrong
analysis (Mann-Whitney test on low-entropy tokens) shows a striking layer dependence.
Early layers (L0, L2, L4) show highly significant separation (p < 1e-4) — at these
layers, even individually confident tokens carry a geometry signal that distinguishes
correct from incorrect traces. Mid-layers (L16–L22) lose significance, and it
re-emerges at L24–L26 (p < 1e-5). This mirrors the bimodal AUC profile and suggests
that early-layer confident-wrong detection reflects misidentified problem types, while
late-layer detection reflects execution errors the model has committed to.

---

## Temperature Sampling (historical diagnostic)

This run used the old DeepSeek budget and is not a current correctness result.

DeepSeek-R1-Distill-Qwen-7B on GSM8K: greedy (T=0) vs temperature sampling (T=0.6).
This tests whether entropy collapse — the primary motivation for geometry — persists
under non-greedy decoding.

| Condition | Greedy (T=0) | Temperature (T=0.6) |
|---|---|---|
| N (correct / incorrect) | 1144 / 125 | 1189 / 130 |
| Entropy-only AUC | 0.728 | 0.699 |
| Best Mahal-only AUC (layer) | 0.806 (L7) | 0.784 (L7) |
| Best combined AUC (layer) | 0.835 (L7) | 0.822 (L7) |
| Raw Δ (combined − entropy) | +0.106 | +0.124 |
| Length-controlled Δ | +0.080 | +0.100 |

**Temperature makes entropy worse, but geometry degrades less.** Under T=0.6, entropy-
only AUC drops from 0.728 to 0.699 (−0.029) — sampling noise blurs the already-weak
entropy signal. Geometry also degrades but by less: Mahalanobis-only drops from 0.806
to 0.784 (−0.022), and combined from 0.835 to 0.822 (−0.013). As a result, the geometry
advantage *increases*: the length-controlled delta grows from +8.0 to +10.0 AUC points.

**Entropy collapse persists and worsens under temperature.** The hypothesis was that
greedy decoding might artificially suppress entropy, making entropy look worse as a
discriminator than it truly is. The opposite happens: adding temperature noise makes
entropy even less discriminative, presumably because the sampling randomness adds
entropy to all tokens regardless of whether the reasoning is correct.

**Geometry is more robust to decoding strategy.** Hidden states are computed before the
sampling step — they reflect the model's internal representation, not the stochastic
output. The −0.022 drop in Mahalanobis AUC likely reflects genuinely different reasoning
trajectories under temperature sampling (the model takes different paths), not
measurement noise from the decoding process.

Per-layer detail at T=0.6:

| Layer | Mahal-only | Combined | Δ (raw) | Δ (len-ctrl) |
|---|---|---|---|---|
| L7 | 0.784 | **0.822** | +0.124 | +0.100 |
| L14 | 0.787 | 0.791 | +0.093 | +0.072 |
| L21 | 0.804 | 0.815 | +0.116 | +0.093 |

The layer profile under temperature is similar to greedy: L7 is best combined, L21 is
best Mahalanobis-only. The trough at L14 persists. This stability across decoding
strategies reinforces the finding that the layer profile reflects architectural structure,
not an artifact of greedy decoding.

---

## Functional Trajectory Encoding (Track A)

The finished `trajectory_*` stages in `dvc.yaml` run `probe.py` on the same four
model/dataset pairs as the scalar probe, but with **Track A only**:
`--methods fpca_mahal`, `--layers 7,14,21`, `--target_len 64`, `--pca_dim 128`.
Within each CV fold, `probe.py` fits the Mahalanobis reference on the train fold's
correct traces, resamples each trace's Mahalanobis-distance sequence to 64 points,
fits PCA on the flattened trajectories, and trains a balanced logistic regressor on
the resulting coefficients.

| Condition | Best fPCA trajectory AUC | Entropy | Best scalar Mahal | Best combined |
|---|---|---|---|---|
| Qwen MATH-500 | 0.695 (L21) | 0.713 | 0.742 (L7) | 0.772 (L7) |
| Qwen GSM8K | 0.538 (L7) | 0.760 | 0.690 (L21) | 0.781 (L21) |
| DeepSeek MATH-500 | 0.802 (L21) | 0.776 | 0.827 (L21) | 0.859 (L7) |
| DeepSeek GSM8K | 0.808 (L21) | 0.728 | 0.831 (L21) | 0.835 (L7) |

**Main result**: Track A is a negative result. The functional trajectory encoder never
beats the existing scalar Mahalanobis summary features at any layer or in any condition.
Its best case is DeepSeek GSM8K at L21 (0.808), which still trails scalar Mahalanobis
at the same layer (0.831) and the best combined model (0.835).

**Late layers dominate.** Unlike the scalar combined probe, whose best layer can be
early (L7 on both MATH-500 conditions and DeepSeek GSM8K), the trajectory encoder
peaks at L21 in three of four settings and is weakest or mediocre at earlier layers.
This suggests the resampled-distance path mostly captures late execution-stage
variation, not the earlier problem-setup signal that drives the strongest scalar gains.

**Model dependence is stark.** On DeepSeek, the trajectory encoder remains above the
entropy baseline (+0.027 on MATH-500, +0.080 on GSM8K), so sequence-shape information
is not useless. On Qwen it fails outright: it underperforms entropy on both datasets,
and on GSM8K it is close to chance (0.519-0.538 ROC AUC across all three layers).

**Interpretation**: for the current Track A implementation, learning a fold-local PCA
basis over the full Mahalanobis trajectory does not improve on simple summary
statistics. The sequence representation appears to add variance faster than signal.
So the scalar Mahalanobis summaries remain the right headline result, and any follow-up
on trajectory modeling should focus on the unfinished ablations (`fpca_combined` and
`probseq_joint`) rather than treating `fpca_mahal` as a replacement.

---

## Early-Layer Prefix Detection (Qwen MATH-500 pilot)

Qwen2.5-7B-Instruct on MATH-500, shallow dense-sweep layers only (L0/L2/L4), using
prefixes of the first `k` generated tokens. This tests the "prune bad traces early"
hypothesis from the roadmap.

| Prefix k | Entropy AUC | L0 Combined | L2 Combined | L4 Combined |
|---|---|---|---|---|
| 5 | 0.501 | 0.501 | **0.631** | 0.619 |
| 10 | 0.507 | 0.504 | 0.550 | 0.524 |
| 20 | 0.469 | 0.531 | 0.512 | 0.494 |
| 40 | 0.533 | 0.554 | 0.533 | 0.489 |

Best setting: **L2 at k=5**, combined AUC **0.631** vs entropy **0.501**
(Δ = **+0.130**). Mahalanobis-only is slightly stronger still at the very shortest
prefixes (L2/L4 Mahal-only 0.659/0.670 at k=5), but the effect does **not** strengthen
as more prefix tokens are observed.

**Interpretation**: there is a real but weak early geometry signal in the first few
tokens, especially at shallow layers. But it is **not** yet strong or stable enough to
support an adaptive early-pruning policy. Performance is mediocre in absolute terms and
non-monotone in `k`: after the initial 5-token bump, combined AUC falls back toward
~0.50-0.55 for k=10-40.

This changes the roadmap priority. Prefix detection should now be treated as a
**secondary follow-up**, not the main downstream application. The immediate next step is
Best-of-N replication on models/settings where the full-trajectory probe signal is
stronger. The first Qwen run is now complete and does **not** show a geometry-guided
selector win, which makes prefix pruning even less urgent.

---

## Best-of-N Selection (Qwen MATH-500, full 500 problems, N=8)

> **Status note (2026-06):** the negative result below was measured with **base
> Mahalanobis** — the *weak* geometry variant. The RMD selector rerun is complete
> on the 100-problem pilot, but the full 500-problem result below has not been rerun
> with RMD. On the pilot, RMD does not rescue Qwen reranking and only matches majority
> vote for DeepSeek. Prefix filtering remains negative — see the prefix section and
> the note in `dvc.yaml`.

Qwen2.5-7B-Instruct on the full MATH-500 benchmark with `N=8` sampled traces per
problem. Selectors and Mahalanobis references are fit leakage-safely on train folds
only; the reported metric is mean Pass@1 across the 5 fixed folds.

| Selector | Pass@1 |
|---|---|
| Random | 0.557 |
| Oracle Pass@8 | 0.676 |
| Majority vote | 0.620 |
| Mean log-prob | 0.574 |
| Entropy-only | 0.590 |
| Best Mahal-only | 0.576 (L21) |
| Best Combined | 0.582 (L14) |

**Main result**: this run does **not** validate geometry-guided reranking for Qwen.
The best combined selector improves over random by only +0.025 Pass@1 and over
mean log-prob by +0.008, but it still trails entropy-only by -0.008 and majority vote
by -0.038. The practical gap to the oracle remains large (0.620 vs 0.676 for majority,
0.582 vs 0.676 for best combined).

**Geometry helps a little, but not enough.** Adding entropy to Mahalanobis improves the
same-layer selector at L7 and L14 (0.576 vs 0.554 at L7; 0.582 vs 0.566 at L14), so the
probe features are not useless. But the gains are too small to beat the simpler
baselines. For Qwen at `N=8`, the ranking is: majority vote > entropy > best combined
~ best Mahal/log-prob > random.

**Difficulty breakdown weakens the Best-of-N story further.** Majority vote is strongest
across easy, medium, and hard groups. Entropy ties the best combined selector on the
medium stratum (0.638) and beats it on the hard group (0.481 vs 0.466). The hoped-for
"geometry rescues hard problems" effect is absent here.

**Subject-level behavior is mixed and does not match the earlier structural-subject
hypothesis.** Combined does beat entropy in a few pockets, including Counting &
Probability (0.500 vs 0.447), Intermediate Algebra (0.464 vs 0.433), and Number Theory
(0.823 vs 0.806). But it loses in Algebra (0.750 vs 0.774), Geometry (0.390 vs 0.439),
and Precalculus (0.286 vs 0.304) — exactly the sort of structurally distinct subjects
where a geometry-guided selector was supposed to be most plausible.

**Interpretation**: the probe signal that predicts single-trace correctness does not yet
translate into a strong multi-sample reranker for Qwen. Agreement between samples
(majority vote) is a better heuristic than hidden-state geometry in this regime. The
next question is whether this is a Qwen-specific limitation or whether geometry-guided
selection only becomes competitive for more reasoning-distilled models and/or larger `N`.

### RMD pilot update (100 problems, N=8)

| Model | Random | Majority | Entropy | Best raw combined | Best RMD | Best RMD+entropy |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 0.606 | **0.690** | 0.620 | 0.650 (L14) | 0.620 (L14) | 0.580 (L14) |
| DeepSeek | 0.439 | **0.570** | 0.500 | 0.550 (L7) | **0.570 (L21)** | 0.560 (L14/L21) |

The pilot sharpens rather than reverses the negative application result. RMD is highly
useful for single-trace abstention, but it does not consistently rank samples from the
same prompt. DeepSeek is the more promising case, where RMD-only matches majority vote,
but this requires confirmation on the full 500-problem set with OOF RMD scoring.

---

## Localized Geometry, Contrastive Readouts, and Selection (Qwen MATH-500, 2026-07-18)

Full rerun of the Qwen Best-of-8 prompt decomposition and OOF prompt selection
after the truncation-bias fix (`evaluate_prompt_decomposition@0`,
`evaluate_prompt_selection@0`; 500 prompts x N=8, layers 7/14/21, 5 prompt
folds, 1,000-draw prompt-cluster bootstrap). Full ledger entry with all paired
contrasts: `EXPERIMENT_LOG.md` (2026-07-18).

**Headline: the two-regime dissociation is confirmed on clean Qwen data, and
localization to high-entropy tokens is the one within-prompt effect that
robustly replicates across all three layers.**

**Within-prompt (parseable traces, 117 mixed prompts).** Restricting RMD to
the highest-entropy 20% of tokens improves centered AUC by +0.052/+0.055/
+0.058 at L7/14/21 (all p ≤ 0.006) over full-trace RMD, and the gain is
entropy-specific — a matched random-20% token control tracks full-trace RMD
almost exactly. The effect grows with depth: at L21, rmd_high_entropy_q20
reaches 0.654 within-macro / 0.605 centered. But this only *ties* the free
output baselines (entropy 0.660 / 0.611; logprob 0.649 / 0.609):
rmd_he_q20 − logprob at L21 is −0.004 centered (p=0.926), and geometry is
*below* logprob at L7/L14. Supervised prompt-contrastive directions are real
(OOF alignment beats the shuffle null, strongest at L21: 0.18–0.22 vs null
≈0.10) but add nothing over the matched localized RMD (p ≥ 0.118). Adding
geometry to a cross-fitted output-feature probe helps only at L21 within-macro
(+0.049, p=0.024, unadjusted, 1 of 6 cells).

**Yet the signal is not entropy in disguise.** Label-free residualization of
within-prompt-centered rmd_he_q20 on entropy+logprob+length keeps nearly all
its discrimination at L21 (residual within-macro 0.645 [0.587, 0.697] vs
0.654 raw). Geometry and output uncertainty are linearly complementary; the
flat incremental-probe result reflects saturation on 117 mixed prompts, not
redundancy.

**Best-of-8 selection stays negative, now with the mechanism.** Majority vote
0.596 pass@1 (random 0.557, oracle 0.676); every geometry/logprob tie-break
variant lands within ±0.006 (15/15 paired deltas p ≥ 0.248), and weighted RMD
voting *underperforms* majority (0.582–0.584). The negative is structural:
only 39/500 prompts have a tied top answer at N=8, and only ~10 ties contain
both a correct and an incorrect option — a ~2-point ceiling no tie-breaker
can exceed.

**Where geometry wins: abstention, not reranking.** Exploratory risk–coverage
on the same OOF scores (L21): trace-level parseable acc@50% coverage — rmd
0.784 vs entropy 0.676. Prompt-level abstention with majority-vote answering
(full-coverage 0.616): rmd_tail_q20 reaches **0.836 at 50% coverage** vs
length 0.740, logprob 0.680, entropy 0.672 — i.e. geometry beats the
length-confound baseline by ~+0.10, so this is not purely truncation
detection. (No bootstrap CIs on these selective numbers yet; confirmatory run
via the selective-prediction stages is the next dependent experiment.)

**Superseded numbers.** This rerun replaces the earlier Qwen decomposition
entries: the L21 RMD-minus-length centered contrast is now null (+0.029,
p=0.194; pooled +0.049 [0.018, 0.083] remains, but the pooled view is
length/truncation-confounded — length alone pools at 0.737 and collapses to
0.478 within-prompt on parseable traces).

---

## Supervised Probe Ceiling and Length Residualization (Qwen + DeepSeek MATH-500, 2026-07-31)

Two additions to the Best-of-8 decomposition, run on both models at L21 (500
prompts, N=8, 5 prompt folds, 1,000-draw prompt-cluster bootstrap):

- **`probe_hidden_*`** — a cross-fitted supervised LDA (`lsqr`, Ledoit-Wolf
  shrinkage) on PCA-projected region means, trained on pooled labels over
  parseable training traces only. This is the SEP-style ceiling: how much of the
  geometry signal can supervision on the *same activations* recover? It is
  distinct from `contrast_*`, which is prompt-centered by construction and so
  targets the within-prompt regime.
- **E1R** — the E1 prompt-abstention metrics recomputed with trace length
  partialled out of every scorer in rank space. E1 shows whether RMD beats
  length; only E1R shows whether RMD carries anything length does not already
  supply.

**Both are exploratory, not pre-registered.** Ledger: `EXPERIMENT_LOG.md`
(2026-07-31). Code: `prompt_decomposition.py` (`fit_hidden_state_probe`,
`length_collapse_diagnostics`), `wave1_experiments.py`
(`length_residualized_abstention`).

**Headline: RMD's rank correlation with length is high but not exclusive. Its
length-orthogonal component is the strongest unsupervised prompt-level signal on
both models, and the supervised probe does not reliably beat it once length is
controlled. On DeepSeek, entropy and logprob — not RMD — are the scorers that
collapse to length.**

### Raw prompt abstention with the probe (AURC, L21)

| Scorer | Qwen | DeepSeek |
|---|---:|---:|
| `probe_hidden_tail_q20` | 0.853 | 0.904 |
| `probe_hidden_full` | 0.806 | 0.882 |
| `probe_hidden_high_entropy_q20` | 0.779 | 0.880 |
| `rmd_tail_q20` | 0.828 | 0.856 |
| `rmd_high_entropy_q20` | 0.789 | 0.832 |
| `length` | 0.759 | 0.826 |
| `logprob` / `entropy` | 0.666 / 0.660 | 0.788 / 0.788 |

Base accuracy (full coverage) is 0.620 Qwen / 0.750 DeepSeek, so AURC levels are
not comparable across models; the paired deltas are.

`probe_hidden_tail_q20 − rmd_tail_q20` is **+0.025 [+0.002, +0.046] p=0.028**
(Qwen, Holm 0.056 — does not survive) and **+0.048 [+0.018, +0.079] p=0.002**
(DeepSeek, Holm 0.006 — survives). Same sign both models. On Qwen the other two
probe regions *lose* to RMD; on DeepSeek all three favor it.

Note the DeepSeek RMD-vs-length picture: `rmd_tail_q20 − length` = +0.030
[+0.014, +0.048] holds, but `rmd_high_entropy_q20 − length` = +0.005
[−0.011, +0.025] p=0.506 is **indistinguishable from length**. Only the tail
region clears the confound baseline on that model.

### The length-collapse diagnostic, and why it misleads on its own

Spearman of each score against `length_score`, parseable traces only (n=3,672
Qwen / 3,649 DeepSeek), L21:

| Scorer | Qwen | DeepSeek |
|---|---:|---:|
| `rmd` | +0.658 | **+0.820** |
| `rmd_high_entropy_q20` | +0.615 | +0.808 |
| `rmd_tail_q20` | +0.675 | +0.805 |
| `probe_hidden_tail_q20` | +0.425 | **+0.223** |
| `logprob` / `entropy` | −0.134 / −0.163 | **+0.369 / +0.350** |

Two things stand out. RMD is far more length-coupled on the reasoning-distilled
model (rho +0.82 at L21, rising monotonically with depth from +0.70 at L7),
while a supervised probe on the same activations sits near +0.22. And the sign
of the entropy/logprob coupling **flips** between models: negative on Qwen,
strongly positive on DeepSeek.

**Read on its own, this table invites the wrong conclusion.** A rho of +0.82
does not imply RMD is a length proxy — length explains only part of the
solvability ranking, so a scorer can track length closely and still carry a
large independent component. E1R is the test that settles it.

### E1R — abstention with length partialled out

Length removed in rank space (Spearman's own linear component, so monotone
non-linear coupling is removed too). The reference is an uninformative scorer,
whose expected AURC equals base accuracy; a scorer with no length-independent
signal lands at zero. Negative control on the same data — a synthetic scorer
that is length plus sub-tie jitter — lands at +0.008 (Qwen) / −0.007 (DeepSeek),
p ≥ 0.82.

| Scorer | Qwen Δ vs uninformative | DeepSeek Δ vs uninformative |
|---|---|---|
| `probe_hidden_tail_q20` | +0.190 [+0.155, +0.224] | +0.140 [+0.110, +0.168] |
| `rmd_tail_q20` | **+0.161 [+0.128, +0.194]** | **+0.107 [+0.077, +0.135]** |
| `rmd_high_entropy_q20` | +0.111 [+0.074, +0.148] | +0.063 [+0.027, +0.096] |
| `logprob` | +0.058 [+0.014, +0.097] | +0.009 [−0.029, +0.046] |
| `entropy` | +0.057 [+0.013, +0.097] | +0.011 [−0.028, +0.047] |

Holm within model across the seven scorers: every geometry row survives at
p < 0.01 on both models; `entropy`/`logprob` survive on Qwen (Holm 0.016) and
**fail completely on DeepSeek (Holm 1.000)**. On DeepSeek, RMD's orthogonal
component alone (+0.107) exceeds length's own raw advantage over the null
(+0.076) — so the high rank correlation notwithstanding, RMD is not a length
proxy, and the output-side baselines are.

**Probe vs RMD once neither scorer may use length:**

| Contrast | Qwen | DeepSeek |
|---|---|---|
| `probe_hidden_tail_q20` − `rmd_tail_q20` | +0.029 [−0.003, +0.060] p=0.090 | +0.033 [+0.001, +0.064] p=0.042 |
| `probe_hidden_full` − `rmd_tail_q20` | −0.037 [−0.074, −0.004] p=0.034 | +0.012 [−0.021, +0.043] p=0.422 |
| `probe_hidden_high_entropy_q20` − `rmd_tail_q20` | −0.062 [−0.102, −0.023] p=0.006 | +0.004 [−0.033, +0.037] p=0.778 |

Holm within model (3 comparisons): **nothing favoring the probe survives**
(Qwen 0.090, DeepSeek 0.126). The single surviving cell is Qwen's
`high_entropy_q20` probe *losing* to RMD (Holm 0.018). So the DeepSeek probe
advantage in raw E1 (+0.048, Holm 0.006) was substantially the probe being less
length-dependent than RMD rather than extracting more geometry: strip length
from both and it falls to +0.033 and stops surviving correction.

### Caveats

- The residual retains small rank correlation with length (+0.13 DeepSeek,
  +0.05 Qwen). Rank-space OLS zeroes the Pearson correlation of ranks, not the
  Spearman correlation of the residual's own ranks, so removal is near-complete,
  not exact.
- E1R is a stricter test than incremental value. It shows RMD's orthogonal part
  ranks prompts on its own; it does not show that `length + RMD` beats `length`.
- Supervision is not a drop-in substitute for RMD's label-light use case. The
  probe is a ceiling and a diagnostic, not a competing deployment story.
- Two models, both Qwen-lineage. The `deepseek_llama` (Llama-architecture)
  collect was cancelled by the localization gate, so no cross-architecture
  replication of any result in this section exists.

---

## Discussion (current interpretation)

The current result is narrower: entropy-localized RMD is a reproducible prompt-level
difficulty/abstention signal on clean Qwen Best-of-8 traces. It does not establish a
general per-attempt correctness detector, a transfer law, or a distillation effect.
Trajectory encoding, prefix filtering, and Best-of-N tie-breaking are negative
follow-ups; retain them as diagnostics rather than headline applications.

**Limitations**:

- **Best-layer selection bias**: Headline numbers are reported at the layer where
  entropy + geometry achieves the highest combined AUC on the **same** data used for
  evaluation. That can inflate the apparent gain vs entropy-only. **Mitigation (planned
  for the paper):** report **all three sparse layers (L7, L14, L21)** in the main table and
  let the layer pattern speak; the Qwen dense sweep already shows the full curve on
  MATH-500. **Stronger alternative (heavier):** nested CV — in each outer fold, run inner
  CV on the train split to pick a layer, then evaluate only that layer on the outer test
  fold.
- **Cross-architecture story is still incomplete**: Llama-family runs now exist and
  they materially narrow the raw-geometry claim, but this document's detailed sections
  are still centered on the original Qwen-family experiments. Cross-architecture
  transfer, robust-geometry comparisons, and a fully integrated all-model writeup still
  need cleaner reporting.
- **Answer extraction noise**: MATH-500 uses `\boxed{}` regex; GSM8K uses `####` marker.
  Correct answers that fail to parse count as incorrect, inflating the "incorrect" rate.
- **PCA dim fixed at 128**: Not swept. May not be optimal.
- **Trajectory experiment only covers Track A**: The finished DVC run used
  `fpca_mahal` only. The entropy+trajectory (`fpca_combined`) and probabilistic
  sequence (`probseq_joint`) variants were implemented in `probe.py` but not run in
  the completed pipeline.
- **Dense sweep is single-model, single-dataset**: The bimodal layer profile is
  established for Qwen on MATH-500. Whether it replicates for DeepSeek or on GSM8K is
  untested.

**Completed follow-ups**:

1. ~~Dense layer sweep~~ — **Done.** Signal is bimodal (peaks at L6–L10 and L20–L26,
   trough at L14), not monotone. Two processing phases: problem comprehension and
   solution execution.
2. ~~Temperature sampling~~ — **Done.** Entropy collapse persists and worsens under
   T=0.6. Geometry advantage increases from +8.0 to +10.0 length-controlled AUC points.
3. ~~Early-layer prefix pilot~~ — **Done (mixed/negative).** A shallow-layer prefix
   signal exists at the first ~5 generated tokens (best: Qwen MATH-500, L2, AUC 0.631),
   but it is too weak and unstable to justify an early-pruning policy yet.
4. ~~Best-of-N run on Qwen MATH-500 (`N=8`)~~ — **Done (negative for geometry);
   RMD pilot also complete.**
   Majority vote is best at Pass@1 = 0.620, ahead of entropy-only (0.590), best combined
   (0.582, L14), best Mahal-only (0.576, L21), and mean log-prob (0.574). Geometry-aware
   reranking is not yet the strongest downstream application on Qwen. In the 100-problem
   RMD pilot, Qwen RMD-only reaches 0.620 vs majority 0.690; DeepSeek RMD-only matches
   majority at 0.570.
5. ~~Functional trajectory encoding (Track A)~~ — **Done (negative).** The completed
   `probe.py` / `trajectory_*` DVC run never beats scalar Mahalanobis summaries. Best
   trajectory AUC is 0.808 (DeepSeek GSM8K, L21), still below scalar Mahal (0.831) and
   combined (0.835).
6. ~~Cross-architecture raw replication + robust controls~~ — **Done at summary level.**
   Llama-family runs show that raw one-class Mahalanobis is not universally strong, while
   RMD / norm-RMD recover several weak cells. This now changes the main interpretation
   and next-step priority.

**Next steps and implementation roadmap**:

1. **Fix reporting hygiene first.**
   - Keep leakage-safe scoring end-to-end.
   - Move away from same-fold best-layer headlines: either report all sparse layers
     (`L7/L14/L21` or `L8/L16/L24`) or use nested CV for layer choice.
   - Use the full transfer grid and weak cells in the text, not only best-case numbers.

2. **Promote robust geometry into the main claim.**
   - Make raw Mahalanobis the baseline and **RMD / norm-RMD** the primary unsupervised
     comparison.
   - Add focused ablations on normalization, background correction, PCA dimension, and
     covariance regularization.
   - Use Llama-family failures to explain why raw global covariance is insufficient and
     why relative / robust geometry helps.

3. **Keep low-rank geometry, but label it honestly.**
   - Report low-rank centroid / Mahalanobis sweeps as **label-informed upper bounds**.
   - Emphasize that low ranks often work best and higher ranks hurt, but do not present
     this as an unsupervised continuation unless a one-class low-rank variant is added.

4. **Change the downstream story.**
   - Stop treating Best-of-N and prefix filtering as likely headline applications.
   - First measure **within-problem score spread** versus between-problem score spread.
     If the score does not vary meaningfully within prompt, reranking will keep failing.
   - For prefixes, split the question cleanly: test prompt-conditioned normalization
     before claiming the early signal itself is absent.
   - A better-aligned downstream task is **single-trace selective prediction /
     abstention / escalation**: after one trace, decide whether to trust it, sample more,
     or hand off to a stronger model.
   - Only revisit Best-of-N with a group-aware objective or hybrid with answer agreement.

5. **Execution order.**
   - **5.1** Rebuild headline tables around raw vs RMD / norm-RMD with all layers shown.
   - **5.2** Add the within-problem variance audit and score-spread plots for current
     Best-of-N data before running more downstream sweeps.
   - **5.3** Run a selective-prediction / abstention benchmark on the strongest settings
     first (DeepSeek Qwen-family, then DeepSeek-Llama MATH-500 with robust geometry).
   - **5.4** Keep cross-architecture replication, dense-layer replication, and limited
     temperature / PCA ablations as robustness checks.
   - **5.5** Revisit heavier branches (`fpca_combined`, `probseq_joint`, MFA) only if they
     beat the scalar robust baseline somewhere nontrivial.

6. **Risks to track during execution.**
   - Domain shift between fit source and eval source can reduce transfer.
   - Parsing/extraction noise can distort both majority vote and Pass@1.
   - Gains may be concentrated in specific subjects or model families; report that
     heterogeneity directly rather than smoothing it away.
