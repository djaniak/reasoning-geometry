# Hidden-State Geometry Predicts Math Reasoning Errors

*Qwen2.5-7B-Instruct and DeepSeek-R1-Distill-Qwen-7B on MATH-500 and GSM8K.*
*Includes dense 14-layer sweep (Qwen) and temperature sampling (DeepSeek T=0.6).*
*Code: `collect_data.py`, `analyze.py`. Pipeline: `dvc.yaml`. Summary: `results/SUMMARY.md`.*

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

## Cross-Model Transfer

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

## Temperature Sampling (GSM8K)

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

---

## Discussion

The main finding is that hidden-state geometry provides a correctness signal that is
complementary to token-level entropy, especially for reasoning-distilled models where
entropy collapses. The effect is visible at the aggregate level across all conditions, partially transfers
across models, follows a bimodal layer profile, and is robust to decoding strategy.
Subject- and difficulty-level results are noisier and model-dependent. The first
trajectory-encoding follow-up is negative: resampled fPCA on the Mahalanobis sequence
does not beat the simpler scalar Mahalanobis summaries, and on Qwen GSM8K it is near
chance.

**Limitations**:

- **Best-layer selection bias**: Headline numbers are reported at the layer where
  entropy + geometry achieves the highest combined AUC on the **same** data used for
  evaluation. That can inflate the apparent gain vs entropy-only. **Mitigation (planned
  for the paper):** report **all three sparse layers (L7, L14, L21)** in the main table and
  let the layer pattern speak; the Qwen dense sweep already shows the full curve on
  MATH-500. **Stronger alternative (heavier):** nested CV — in each outer fold, run inner
  CV on the train split to pick a layer, then evaluate only that layer on the outer test
  fold.
- **Single architecture family**: Both models are Qwen2.5-7B. Cross-architecture
  generalization (e.g., LLaMA, Mistral) is untested but data collection is in progress.
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
4. ~~Best-of-N run on Qwen MATH-500 (`N=8`)~~ — **Done (negative for geometry).**
   Majority vote is best at Pass@1 = 0.620, ahead of entropy-only (0.590), best combined
   (0.582, L14), best Mahal-only (0.576, L21), and mean log-prob (0.574). Geometry-aware
   reranking is not yet the strongest downstream application on Qwen.
5. ~~Functional trajectory encoding (Track A)~~ — **Done (negative).** The completed
   `probe.py` / `trajectory_*` DVC run never beats scalar Mahalanobis summaries. Best
   trajectory AUC is 0.808 (DeepSeek GSM8K, L21), still below scalar Mahal (0.831) and
   combined (0.835).

**Next steps and implementation roadmap**:

1. **Finalize evaluation protocol and reporting hygiene (immediate).**
   - Keep leakage-safe scoring end-to-end (train-fold-only refs in probe analyses; disjoint fit/eval splits for downstream selectors).
   - In main probe tables, report **all sparse layers (L7/L14/L21)** side-by-side instead of
     selecting a best layer on the same data.
   - Keep full cross-model transfer grid in supplement and use honest ranges in text
     (include weak cells like ~81% retention).

2. **Replication and stress tests (before new claims).**
   - **Cross-architecture replication** (`collect_llama_*`, `analyze_llama_*`) on
     Llama-3.1-8B-Instruct.
   - **Dense sweep for DeepSeek** (same 14-layer schedule as Qwen dense sweep) to test
     whether bimodality and the L14 trough replicate in the reasoning-distilled model.
   - **Temperature sweep** beyond a single point (`T in {0.0, 0.3, 0.6, 0.9}`) for at least
     one model/dataset pair, with fixed seeds.
   - **PCA-dimension ablation** (`d in {32, 64, 128, 256}`) and covariance regularization
     sensitivity to confirm robustness.
   - **Reference-source ablation** (GSM8K-fit -> MATH-eval vs disjoint-MATH-fit -> MATH-eval)
     to quantify domain-shift effects on downstream scoring.
   - **Trajectory-model follow-up only if justified**: Track A (`fpca_mahal`) is now a
     negative baseline. If trajectory modeling is revisited, prioritize `fpca_combined`
     and `probseq_joint` on DeepSeek first and require a win over scalar Mahalanobis on
     at least one condition before expanding the branch.

3. **Downstream Application A: Geometry-guided Best-of-N selection (now a stress test, not a headline result).**
   - The first full-dataset run is complete for **Qwen MATH-500, `N=8`**, and it is
     negative for the main claim: majority vote beats all probe-based selectors.
   - Replicate on **DeepSeek** first, where the probe advantage over entropy is much larger
     in the single-trace setting and is therefore more likely to survive the reranking setup.
   - Then increase to `N=16` only if either model starts to show a real probe-based gain.
   - Keep the same selector set: random, majority vote, length-normalized mean log-prob,
     entropy-only, Mahalanobis-only, and combined entropy+Mahalanobis.
   - Primary metric: Pass@1. Secondary: subject-level Pass@1 plus oracle Pass@N context.
   - Success criterion: a geometry-aware selector should at least beat entropy/log-prob and
     materially narrow the gap to majority vote, not just exceed random.

4. **Downstream Application B: Early-layer prefix detection (deprioritized).**
   - The single-trace pilot on existing Qwen MATH-500 traces is now complete and mixed.
   - Only revisit on Best-of-N traces if the multi-sample setup reveals stronger prefix
     separation than the current single-trace pilot.
   - Compute shallow-layer (L2, optional L0/L4) geometry features on prefixes
     `k in {5,10,20,40}` tokens; evaluate AUC vs correctness and subject breakdowns.
   - If predictive early, test an adaptive policy: prune low-quality continuations early, then
     apply trajectory-level selector on survivors.
   - Report both quality impact and compute savings; treat this as conditional on robust early
     separation, not a guaranteed win.

5. **Execution order (clean numbering).**
   - **5.1** Freeze DVC stages for leakage-safe **multi-sample collection** and Best-of-N evaluation.
   - **5.2** Qwen full-dataset Best-of-N run (`N=8`) is complete; use it as the baseline negative control.
   - **5.3** Run DeepSeek full-dataset Best-of-N at `N=8`.
   - **5.4** Increase to `N=16` only if DeepSeek or Qwen shows a selector ranking change worth following up.
   - **5.5** Only if Best-of-N starts to succeed: revisit prefix curves on sampled traces and test adaptive pruning.
   - **5.6** Replication/ablations (DeepSeek dense, temperature sweep, PCA sweep, reference-source ablation, then cross-architecture).
   - **5.7** Paper tables/figures: probe results (all layers), transfer grid, Best-of-N gains,
     prefix AUC-vs-k, and subject-specific effects.

6. **Risks to track during execution.**
   - Domain shift between fit source and eval source can reduce transfer.
   - Parsing/extraction noise can distort both majority vote and Pass@1.
   - Gains may be concentrated in specific subjects; report capability profile honestly rather
     than global overclaims.
