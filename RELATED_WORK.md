# Related work: primary-source check

Scope of the check: the frozen selective-prediction result — MATH-500, 8 sampled traces per
prompt, plurality-vote answer, prompt-level out-of-fold logistic readouts B0 (length, token
entropy, token logprob, vote agreement) vs B1 = B0 + `rmd_tail_q20`, scored by AUACC with a
prompt-clustered paired bootstrap, on Qwen2.5-7B-Instruct (1024-token budget) and
DeepSeek-R1-Distill-Qwen-7B (8192-token budget), geometry read at layer 21 in both.

Every claim below was traced to an arXiv/published PDF, an official repo, or a first-party
page, and most were verified by extracting the text of the PDF rather than reading a summary.
Where a claim is my inference rather than something a paper states, it is labelled
**[inference]**. Where I could not find a primary source, it says **not found**.

---

## What this means for the write-up

- **The "increment over self-consistency" framing survives, and it is the single defensible
  novelty claim here.** I found no paper that adds a hidden-state geometry feature *on top of*
  a vote-agreement/self-consistency baseline and reports the increment. Probe papers either
  report the probe alone (arXiv:2504.05419 reports ROC-AUC with no output-side baseline at
  all), or benchmark geometry as a *replacement* for consistency methods (INSIDE,
  arXiv:2402.03744; LM-Polygraph, TACL 2025). Say this explicitly and cite the absence.
- **`rmd_tail_q20` is not a new statistic and should not be presented as one.** Ren et al.
  (arXiv:2106.09022) own RMD; Vazhentsev et al. (NAACL 2025, arXiv:2502.14427) already compute
  **token-level RMD over generated tokens** with foreground = tokens from correct responses and
  background = all generated tokens, aggregate it over the trace, and feed it to a supervised
  regressor alongside sequence probability. That is the same construction. **The aggregator is
  not ours either**: `rmd_tail_q20` is the *mean* of per-token RMD over the final 20% of tokens
  (`prompt_decomposition.py::score_localized_rmd`), which is Vazhentsev's trace mean restricted
  to a tail window. The `q20` in the name is the window size, not a quantile of the distances.
  What is left as ours is the *evaluation* (prompt-level, over siblings, against a
  vote-agreement baseline). Frame the contribution there, not on the feature and not on the
  aggregator.
- **The data now says the same thing, independently (2026-08-10).** The untailed whole-trace
  mean `rmd_full` — which *is* Vazhentsev's ATRMD — was run against `B0` directly. On both
  reasoning-distilled models it recovers essentially the whole increment on its own
  (DeepSeek-R1-Distill-Qwen-7B −0.0335 of −0.0355, p=0.034; DeepSeek-R1-Distill-Llama-8B
  −0.0509 of −0.0560, p=0.004), and adding the tail restriction on top adds nothing separable
  from zero. Only on Qwen2.5-7B-Instruct is the tail load-bearing: there `rmd_full` alone does
  not beat `B0` (−0.0137, p=0.428) while the tail does. So the aggregator leg is dead twice —
  once on the code description, once on the numbers. Cite ATRMD as the statistic and report
  the tail as a **model-dependent localization**, not as the method.
- **The tail window is DeepConf's move, and must be cited rather than claimed:
  DeepConf** (arXiv:2508.15260) uses exactly it — bottom-10% group confidence and tail
  confidence over the last 2048 tokens — on the *output* side. `rmd_tail_q20` is the same
  aggregation shape applied to geometry instead of token confidence. The repo already has
  `deepconf_exact.py` and `B0_plus_DeepConf_tail_q20` specs; those must be in the paper, because
  a reviewer will ask whether the geometry increment is just a worse-instrumented DeepConf.
  Note that the borrowed move only pays off on one of the three models (see above), so do not
  present the tail window as a design choice that transfers.
- **"AUACC" is a real, citable name but it is the minority convention.** Its clearest primary
  use is Chen et al. (arXiv:2310.11689, EMNLP Findings 2023), which states outright that AUACC
  is "the common metric used for evaluating selective prediction performance". But the
  dominant convention in selective classification is AURC / risk-coverage (Geifman et al.,
  arXiv:1805.08206; Kamath et al., ACL 2020), and Nature-level UQ work uses AURAC (Farquhar et
  al. 2024). Recommendation: report AURC as the primary number and AUACC as the mirror, not the
  other way round — the repo already computes both and they are affinely related.
  **Actioned 2026-08-06**, with one correction to the reasoning above: the swap buys the
  convention and nothing else. AURC inherits the base accuracy exactly as AUACC does, just
  with the sign flipped — an uninformative scorer sits at `(1 − 1/n) − base`, not at zero.
  AUROC is the only base-rate-free metric in play. So: AURC primary, AUACC mirror, AUROC
  whenever a comparison crosses models, and never a bare level across models in either.
- **Threat to novelty worth naming out loud:** Orgad et al. (arXiv:2410.02707) resample K=30
  responses per prompt and show hidden-state probes predict *which* resampling regime a prompt
  is in. That is the opposite-direction result — geometry predicts agreement — and a hostile
  reviewer will read it as "so the geometry feature is a proxy for the vote-agreement feature
  you already have in B0." The difficulty controls partly answer this, but the paper should
  answer it directly with the B0/B1 residual structure.
  **Answered 2026-08-06** (`orgad_agreement_control.py`, three models,
  `cap_free_valid_plurality`). The two features correlate at Pearson 0.36 / 0.11 / 0.27
  (Qwen / DeepSeek-Qwen / Llama), so at most 13% of either is the other. Inside the
  *unanimous* stratum — where agreement is constant by construction and carries no
  information — `rmd_tail_q20` scores AUROC 0.829 / 0.714 / 0.756, at or above its pooled
  figure, on 70% / 89% / 52% of prompts. After an out-of-fold linear residualization on the
  vote it keeps 0.744 / 0.660 / 0.670. Swapping geometry in *for* the vote still beats the
  full B0 on two of three models, while adding the vote back on top of geometry buys ≤0.012
  AURC. Cite Orgad et al. as the motivation for the control, and report the control.
- **Second threat:** arXiv:2607.18553 (July 2026) reports the same *shape* of claim — hidden
  states add +0.066 AUROC on GSM8K over length and log-probability shortcut features, with a CI
  excluding zero. Different model class (a looped 2.6B transformer), different metric, no
  self-consistency control — but it is a prior instance of "hidden states add incrementally over
  cheap output features", and it should be cited rather than discovered by a reviewer.
- **Adaptive sample allocation is not available as a contribution (§6, added 2026-08-10).**
  "Spend a fixed budget non-uniformly across prompts and beat uniform" is Adaptive-Consistency
  (EMNLP 2023), ESC (ICLR 2024), Damani et al. (arXiv:2410.04707) and ReASC
  (arXiv:2601.02970). If the allocation rung reports anything, it is geometry's increment over
  a count-based stopping rule at one-to-two samples per prompt — the only budget where a
  vote-based rule has nothing to count. Say that before the table, not after it.

---

## 1. Has this been done? Incremental value over a self-consistency baseline

**Short answer: I did not find it.** No primary source I could locate reports a hidden-state or
representation-geometry feature adding measurable selective-prediction value *on top of* a
self-consistency / vote-agreement baseline. The papers closest to it fall into three groups.

### 1a. Probe papers with no output-side baseline at all

- **Zhang et al., "Reasoning Models Know When They're Right: Probing Hidden States for
  Self-Verification"** ([arXiv:2504.05419](https://arxiv.org/abs/2504.05419), NYU / NYU
  Shanghai, Apr 2025). Trains an MLP/linear probe on hidden states of R1-Distill-Llama-8B/70B,
  R1-Distill-Qwen-1.5B/7B/32B and QwQ-32B, on GSM8K, MATH, AIME and KnowLogic. Verified from the
  PDF: §4.1–4.2 report **ROC-AUC, ECE and Brier for the probe only**. There is no length
  baseline, no entropy baseline, no logprob baseline, and no self-consistency baseline anywhere
  in the experimental section. Reported ROC-AUC is "above 0.7" in-distribution across all
  probes, with R1-Distill-Qwen-32B "over 0.9 ROC-AUC on AIME". Downstream use is a
  confidence-thresholded early exit, cutting inference tokens by 24%.
  This is the most direct precedent for the repo's *feature* (hidden states of an R1-Distill-Qwen
  model on MATH) and it establishes the bar the repo clears: it has no baseline to control for.
- **Sun et al., "LLM Reasoning as Trajectories: Step-Specific Representation Geometry and
  Correctness Signals"** ([arXiv:2604.05655](https://arxiv.org/abs/2604.05655), Apr 2026).
  GSM8K and MATH-500. Reports "ROC–AUC of 0.87 in predicting final-answer correctness prior to
  the emission of the final answer" from step-specific representation-subspace features, then
  builds trajectory-based steering. Verified from the PDF that the datasets are GSM8K +
  MATH-500; I found **no self-consistency or majority-vote baseline** in the text I extracted.

### 1b. Geometry benchmarked as a *replacement* for consistency, not an increment

- **Chen et al., "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection"**
  ([arXiv:2402.03744](https://arxiv.org/abs/2402.03744), ICLR 2024). This is the paper that
  comes closest to "controls for self-consistency", and it is worth being precise about what it
  actually does. EigenScore is
  `E(Y|x,θ) = (1/K) log det(Σ + α·I_K)`, the log-determinant of the covariance matrix of the
  sentence embeddings `Z = [z_1 … z_K]` of K sampled responses — i.e. it *is* a self-consistency
  measure, computed in embedding space rather than answer space. Verified from the PDF, its
  baselines are Perplexity, Length-normalised Entropy (Malinin & Gales), and **Lexical
  Similarity**, defined in their Eq. 3 as the mean pairwise ROUGE-L across the K sampled
  answers — an explicit consistency baseline. Metrics are AUROC and PCC, on QA benchmarks.
  **Difference from the repo's design:** INSIDE compares geometry *against* a consistency
  baseline head-to-head. It does not fit a readout on consistency and then ask what geometry
  adds. The repo's B1 − B0 is the question INSIDE does not ask.
- **Vashurin et al., "Benchmarking Uncertainty Quantification Methods for Large Language Models
  with LM-Polygraph"** ([TACL 2025, 13:220–248](https://aclanthology.org/2025.tacl-1.11/);
  [arXiv:2406.15627](https://arxiv.org/abs/2406.15627)). Benchmarks information-based,
  consistency-based (semantic entropy, lexical similarity) and density-based (Mahalanobis, RDE)
  methods across eleven tasks. Again head-to-head ranking, not incremental. Note for the
  write-up: Vazhentsev et al. cite this benchmark for the finding that "the reported performance
  of density-based scores for text generation so far has been notably low" — useful framing.

### 1c. Hybrid / stacked scores — the nearest thing to an incremental design

- **Vazhentsev et al., "Token-Level Density-Based Uncertainty Quantification Methods for
  Eliciting Truthfulness of Large Language Models"**
  ([arXiv:2502.14427](https://arxiv.org/abs/2502.14427), NAACL 2025). §4.2–4.3 (verified from
  PDF) train a linear regressor on PCA'd per-layer token-MD/RMD features **plus the sequence
  probability** (`SATRMD+MSP`), and separately combine density and information scores via HUQ
  (Vazhentsev et al. 2023). That is structurally the same "cheap output feature + density
  feature in one readout" design as B0→B1. **The output feature is sequence probability, not
  vote agreement**, the task is single-generation selective generation on QA / summarisation /
  fact-checking (eleven datasets), and the metric is PRR, not AUACC. So it is a design
  precedent, not a result precedent.
- **Kirin, "Operational Proto-Introspection in Looped Language Models"**
  ([arXiv:2607.18553](https://arxiv.org/abs/2607.18553), Jul 2026). Reports hidden-state
  features beating surface features (length, log-probability) on GSM8K: AUROC 0.797 vs 0.731,
  increment **+0.066 [+0.021, +0.112]**; Horizon Logic +0.111 [+0.056, +0.169]. Explicitly
  frames this as "hidden-state-based scores improve risk-coverage over shortcut-only scores".
  This is the closest published instance of the repo's *claim shape*. Differences: a frozen 2.6B
  looped transformer (Ouro-RLTT), not a standard decoder; AUROC/risk-coverage, not AUACC; and
  **no vote-agreement term in the baseline** — the shortcut baseline is length + logprob only,
  which is B0 minus entropy and minus vote agreement.

### 1d. The opposite-direction result that threatens the framing

- **Orgad et al., "LLMs Know More Than They Show: On the Intrinsic Representation of LLM
  Hallucinations"** ([arXiv:2410.02707](https://arxiv.org/abs/2410.02707), ICLR 2025).
  Verified from PDF §5.1: they sample **K = 30 responses at T = 1** per example and build a
  taxonomy of error types from the resulting answer distribution (i.e. from agreement), then
  show probes on exact-answer-token activations predict that taxonomy. Their UQ baselines are
  Logits-mean, Logits-min, P(True), and probing — **no consistency baseline**. The relevance is
  adversarial: it is published evidence that hidden states *encode* the resampling agreement
  structure, which is exactly the confound B0's `vote_agreement` term is meant to absorb.
- **Ding, "When LLMs Agree, Are They Right? Auditing Self-Consistency and Cross-Model Agreement
  as Confidence Signals"** ([arXiv:2607.08065](https://arxiv.org/abs/2607.08065), Jul 2026).
  53 runners, 265k samples on GPQA-Diamond and AIME. Concludes agreement is "a positive but
  weak predictor" with correlations 0.20–0.59, and is "a conditional proxy for correctness, not
  a standalone confidence score" — 77% of GPQA entries had ≥0.8 agreement yet 48% of those were
  wrong. Useful for motivating *why* there is headroom above a vote-agreement baseline.

---

## 2. Relative Mahalanobis distance: provenance

### 2a. The originating definitions

- **Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution Samples and
  Adversarial Attacks"** ([arXiv:1807.03888](https://arxiv.org/abs/1807.03888), NeurIPS 2018).
  The Mahalanobis OOD baseline: fit class-conditional Gaussians `N(µ_k, Σ)` with a **shared**
  covariance to penultimate-layer features, score by `−min_k MD_k`.
- **Ren, Fort, Liu, Guha Roy, Padhy, Lakshminarayanan, "A Simple Fix to Mahalanobis Distance
  for Improving Near-OOD Detection"** ([arXiv:2106.09022](https://arxiv.org/abs/2106.09022),
  Jun 2021; presented at the ICML 2021 Uncertainty & Robustness in Deep Learning workshop).
  This is the origin of RMD, and the definition is verbatim from the PDF §2:

  > `RMD_k(z′) = MD_k(z′) − MD_0(z′)`

  where `MD_0(z′)` is the Mahalanobis distance to a Gaussian `N(µ_0, Σ_0)` "fitted to the entire
  training data **not considering the class labels**", `µ_0 = (1/N) Σ z_i`, `Σ_0 = (1/N) Σ
  (z_i−µ_0)(z_i−µ_0)^T`. The paper states RMD "is equivalent to computing a likelihood ratio
  `max_k (log p_k(z′) − log p_0(z′))`", inheriting the foreground/background likelihood-ratio
  idea from Ren et al. 2019. Reported: CIFAR-100 vs CIFAR-10 AUROC 74.91 (MD) → 81.01 (RMD);
  Genomics OOD 53.10 → 68.98; CLINC intent OOD with pretrained BERT 75.48 → 91.98. A secondary
  finding directly relevant to this repo: RMD is *stable across training* while MD's AUROC
  rises then collapses (their Fig. 2).
- **Ren, Luo, Zhao, Krishna, Saleh, Lakshminarayanan, Liu, "Out-of-Distribution Detection and
  Selective Generation for Conditional Language Models"**
  ([arXiv:2209.15558](https://arxiv.org/abs/2209.15558), ICLR 2023). Carries RMD into language.
  Verified from PDF §2.2: the input embedding is `z = (1/L) Σ_i h_i`, the **average of the
  encoder's final-layer hidden states over the input tokens**, and the output embedding is
  `w = (1/T) Σ_i g_i`, the average of the decoder's final-layer hidden states over the output
  tokens. `RMD_input(z_test) := MD_input(z_test) − MD_0(z_test)`, noting explicitly that the
  class-conditional form "cannot be directly applied for CLMs because outputs are not just class
  labels". Selective generation combines the OOD score with sequence likelihood.

### 2b. Does applying it to intermediate-token hidden states within a trace have precedent?

**Yes, and this is the finding that most constrains the novelty claim.** In Ren et al. 2021 and
Ren et al. 2023 the distance is computed on **one vector per input or per whole output** — a
whole-input embedding or a mean-pooled output embedding. Per-token application inside a
generated sequence is *not* in those papers.

It is in **Vazhentsev et al., NAACL 2025** ([arXiv:2502.14427](https://arxiv.org/abs/2502.14427);
code at [github.com/ArtemVazh/token_mahalanobis_distance](https://github.com/ArtemVazh/token_mahalanobis_distance)).
Verified from PDF §4.1, their construction is:

- For each layer `l`, fit `µ_{E_l}`, `Σ_{E_l}` on the set `E_l` of **token** embeddings drawn
  from *responses that pass a correctness criterion* `Q(ỹ_j) > τ` — i.e. foreground = tokens of
  correct traces, which is the repo's foreground.
- "For the token-level RMD, we additionally compute the background covariance matrix `Σ⁰_l` and
  the background centroid `µ⁰_l` using the embeddings of **all generated tokens** for the input
  prompts from some background dataset" — i.e. background = all generated tokens, which is the
  repo's background.
- Per-token score `U_MD(t^k_i, l) = (h_l(t^k_i) − µ_{E_l})^T Σ⁻¹_{E_l} (h_l(t^k_i) − µ_{E_l})`,
  aggregated as **ATRMD = the mean over the tokens of the generated sequence**, then stacked
  across layers via PCA into a supervised linear regressor (`SATRMD`), optionally + MSP.

So: token-level RMD inside a generated trace, correct-trace foreground, all-token background,
multi-layer, is published. Their baselines include Semantic Entropy, Lexical Similarity,
EigenScore, SAPLMA, SAR and Factoscope; their metric is PRR (Prediction Rejection Ratio, from
Malinin & Gales); their tasks are QA / summarisation / fact-checking, not math with sibling
traces.

**What is left unprecedented, as far as I could find:** (a) aggregation from trace to *prompt*
over sibling samples; (b) evaluation against a vote-agreement baseline under difficulty
controls. Claim (a)–(b), not "we apply RMD to trace hidden states". **[inference]** — the papers
do not say they omit these; I simply found no paper doing them.

**Withdrawn (2026-08-10):** a third leg previously claimed here — "the *aggregator*, a low
quantile over a tail region rather than a mean over the whole trace" — was based on a wrong
description of the code. `rmd_tail_q20` computes a *mean* over the final 20% of tokens, so the
aggregator is Vazhentsev's trace mean under DeepConf's tail windowing: the composition of two
published moves, with nothing added. Do not claim it.

The same day, the numbers withdrew it a second time. `rmd_full` — the untailed trace mean,
i.e. ATRMD itself — was scored against `B0` on all three models. It recovers the whole
increment on both reasoning-distilled models, where the tail restriction then adds nothing.
The tail only earns its place on Qwen2.5-7B-Instruct. A window-size explanation was tested
and falsified (see the 2026-08-10 log entry): inside Qwen the tail advantage *grows* with
window size rather than shrinking, and DeepSeek-R1-Distill-Llama-8B's short stratum — matched
to Qwen on both window median (110 vs 87) and base accuracy (0.688 vs 0.693) — still shows no
tail effect (−0.0088 [−0.0214, +0.0026]) against Qwen's −0.0633 [−0.1009, −0.0274]. So the
split follows reasoning distillation, not window length. It rests on one non-distilled model,
and training data, trace style, budget and base accuracy remain collinear with distillation.

Two differences from Ren's original worth stating in a footnote so a reviewer does not think
they were missed: the repo has binary correctness labels rather than Ren's semantic class labels,
so its foreground is *correct traces* and there is no `min_k`; and it fits both Gaussians on a PCA
projection with Ledoit-Wolf shrinkage (`analyze.py::fit_relative_mahalanobis_reference`),
whereas Ren et al. use raw penultimate features with an empirical covariance.

---

## 3. The competing signals

### Self-consistency — Wang et al., ICLR 2023

[arXiv:2203.11171](https://arxiv.org/abs/2203.11171). **The statistic:** sample a diverse set of
reasoning paths, marginalise out the paths, and take the most consistent (plurality/majority)
answer. **Reported:** +17.9% absolute on GSM8K, +11.0% SVAMP, +12.2% AQuA, +6.4% StrategyQA,
+3.9% ARC-challenge, over chain-of-thought greedy decoding, with LaMDA-137B / PaLM-540B / GPT-3 /
UL2-20B.

Critically for B0, the *confidence* reading of self-consistency is also from this paper, §"we
found that the consistency (in terms of % of decodes agreeing with the final aggregated answer)
is highly correlated with accuracy (Figure 5, over GSM8K). This suggests that one can use
self-consistency to provide an uncertainty estimate ... self-consistency confers some ability
for the model to 'know when it doesn't know'." That sentence is the citation for B0's
`vote_agreement` feature; use it.

### Semantic entropy — Kuhn et al. ICLR 2023, and Farquhar et al. Nature 2024

- **Kuhn, Gal, Farquhar, "Semantic Uncertainty: Linguistic Invariances for Uncertainty
  Estimation in Natural Language Generation"**
  ([arXiv:2302.09664](https://arxiv.org/abs/2302.09664), ICLR 2023). **The statistic:** cluster
  sampled generations into semantic equivalence classes `c ∈ C` by bidirectional entailment,
  give each class probability `p(c|x) = Σ_{s∈c} p(s|x) = Σ_{s∈c} Π_i p(s_i|s_<i, x)`, and take
  the entropy over classes rather than sequences. Length-normalisation of log-probabilities is
  discussed as a modelling choice (§3.3). Evaluation is free-form QA with OPT models; the paper
  reports semantic entropy is "more predictive of model accuracy on question answering data sets
  than comparable baselines". **No math benchmark.**
- **Farquhar, Kossen, Kuhn, Gal, "Detecting hallucinations in large language models using
  semantic entropy"**, Nature 630, 2024,
  [doi:10.1038/s41586-024-07421-0](https://doi.org/10.1038/s41586-024-07421-0). **The
  statistic:** `SE(x) = −Σ_c P(c|x) log P(c|x)`; the **discrete** variant estimates `P(c|x)`
  from the *count* of generations in each cluster, ignoring token probabilities — which makes
  discrete semantic entropy the direct generalisation of vote agreement to free-form answers,
  and therefore the honest comparator for B0's `vote_agreement` term. **Metrics:** AUROC, and
  **AURAC — "the total area enclosed by the accuracies at all cut-off percentages X%"**, i.e.
  area under the rejection-accuracy curve. **Reported:** semantic entropy 0.790 AUROC vs
  baselines 0.687–0.698, over TriviaQA, SQuAD, BioASQ, NQ-Open, **SVAMP** (elementary-school
  math) and FactualBio. Baselines: naive entropy, P(True), and a supervised embedding-regression
  probe. **The only math benchmark is SVAMP — there is no MATH/MATH-500 or GSM8K number in
  either semantic-entropy paper.** So "we beat semantic entropy on MATH-500" is a claim only the
  repo's own re-implementation can support; it cannot be sourced to their reported numbers.
- **Kossen et al., "Semantic Entropy Probes"**
  ([arXiv:2406.15927](https://arxiv.org/abs/2406.15927), Jun 2024). Linear probes on hidden
  states trained to predict binarised semantic entropy, evaluated by AUROC against binarised SE
  and against accuracy, with an accuracy probe (scikit-learn logistic regression, default L2 +
  LBFGS) as the comparator. Relevant as the "hidden states can stand in for a consistency
  measure" precedent — again a *substitution*, not an increment.

### DeepConf — Fu, Wang, Tian, Zhao, "Deep Think with Confidence"

[arXiv:2508.15260](https://arxiv.org/abs/2508.15260), Meta AI + UCSD, submitted 21 Aug 2025;
project page [jiaweizzhao.github.io/deepconf](https://jiaweizzhao.github.io/deepconf/); official
code [github.com/facebookresearch/deepconf](https://github.com/facebookresearch/deepconf). This
is the paper `deepconf_exact.py` implements. All definitions below are verbatim from the PDF
§2–3.2:

- **Token confidence** (their Eq. 2): `C_i = −(1/k) Σ_{j=1..k} log P_i(j)` over the **top-k**
  tokens at position `i`. This matches `deepconf_exact.py::topk_token_confidence`.
- **Average trace confidence / self-certainty** (Eq. 3): `C_avg = (1/N) Σ_i C_i`, credited to
  Kang et al., **"Scalable Best-of-N Selection for Large Language Models via Self-Certainty"**
  ([arXiv:2502.18581](https://arxiv.org/abs/2502.18581), NeurIPS 2025).
- **Group confidence** (Eq. 4): mean token confidence over a sliding window `G_i` of n = 1024 or
  2048 previous tokens, overlapping.
- **Bottom-10% group confidence** (Eq. 5): mean of the lowest-decile group confidences in the
  trace.
- **Lowest group confidence** (Eq. 6): `min_{G_j} C_{G_j}`. This is the online early-stopping
  signal.
- **Tail confidence** (Eq. 7): mean token confidence over the **last K tokens** (they use 2048),
  motivated by "reasoning quality often degrades toward the end of long chains of thought".

Offline use is **confidence-weighted majority voting** — `V(a) = Σ_t C_t · I(answer(t)=a)` —
plus **confidence filtering** to the top η% of traces, η ∈ {10%, 90%}.

**Reported math numbers** (their Table 1, accuracy %, K=512, averaged over 64 repetitions):

| Model | Dataset | Pass@1 | Cons@512 | Mean@512 (η=10) | Bottom-10 (η=10) | Tail (η=10) |
|---|---|---|---|---|---|---|
| DeepSeek-8B | AIME24 | 83.0 | 86.7 | 86.7 | 93.3 | 93.3 |
| DeepSeek-8B | AIME25 | 76.9 | 82.3 | 82.3 | 87.5 | 87.4 |
| DeepSeek-8B | HMMT25 | 58.1 | 69.6 | 69.9 | 79.5 | 83.9 |
| Qwen3-32B | AIME24 | 80.6 | 85.3 | 85.7 | 90.8 | 89.4 |
| GPT-OSS-120B | AIME25 | 91.8 | 97.0 | 97.1 | 98.1 | **99.9** |

Online: DeepConf-low cuts tokens 43–79% vs majority voting at K=512; the headline 84.7% token
reduction is the GPT-OSS-120B AIME25 figure. "DeepSeek-8B" in that paper is
DeepSeek-R1-0528-Qwen3-8B, *not* the R1-Distill-Qwen-7B this repo uses — do not conflate them.

**Why this matters most for positioning:** DeepConf's contribution is precisely the argument
that a *low-order statistic over a localised region of the trace* beats a global mean, and that
the tail is a privileged region. `rmd_tail_q20` is that same argument transplanted to
hidden-state geometry. The paper should say so, and should show the geometry increment survives
`B0 + DeepConf_tail_q20` — which the repo's `incremental_abstention.py` already specifies as
`B0_plus_DeepConf_tail_q20_plus_RMD`.

---

## 4. Metric conventions: AUACC vs AURC vs coverage-at-risk

Both names are real. They come from different literatures, and the risk-coverage one is older
and more standard.

**Risk-coverage / AURC (lower is better) — the dominant convention.**
- The risk-coverage curve itself traces to **El-Yaniv & Wiener, JMLR 2010**, "On the Foundations
  of Noise-free Selective Classification" — cited as the origin by both Kamath et al. and Xin et
  al. below. (I verified the *citation*, not the JMLR PDF itself; the primary paper is
  [jmlr.org/papers/v11/el-yaniv10a.html](https://jmlr.org/papers/v11/el-yaniv10a.html).)
- **Geifman, Uziel, El-Yaniv, "Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers"**
  ([arXiv:1805.08206](https://arxiv.org/abs/1805.08206), ICLR 2019) names AURC and E-AURC.
  Verified from PDF §3: "We propose to measure the performance of a κ function as the area under
  the risk-coverage curve (AURC) ... we subtract the AURC of the best κ in hindsight ... We term
  this normalized metric 'excess AURC' (E-AURC)."
- **Kamath, Jia, Liang, "Selective Question Answering under Domain Shift"** (ACL 2020,
  [arXiv:2006.09462](https://arxiv.org/abs/2006.09462)) — the canonical NLP selective-prediction
  paper. Verified from PDF §3.1: "the risk-coverage curve provides a standard way to evaluate
  selective prediction methods (El-Yaniv and Wiener, 2010) ... We plot risk versus coverage and
  evaluate on the area under this curve (AUC), as well as the maximum possible coverage for a
  desired risk level." So: **AURC + coverage@risk**, no AUACC.
- **Xin, Tang, Yu, Lin, "The Art of Abstention"** (ACL 2021,
  [aclanthology.org/2021.acl-long.84](https://aclanthology.org/2021.acl-long.84/)) also reports
  "the area under the risk–coverage curve". **Note a citation error to avoid inheriting:** the
  "Know Your Limits" survey attributes AUACC to Xin et al. 2021, but I grepped the Xin PDF and
  found no "accuracy-coverage" phrasing anywhere — they use risk-coverage. Do not cite Xin et
  al. for AUACC.

**Accuracy-coverage / AUACC (higher is better) — real, but the minority convention.**
- Clearest primary source: **Chen, Yoon, Ebrahimi, Arık, Pfister, Jha, "Adaptation with
  Self-Evaluation to Improve Selective Prediction in LLMs"**
  ([arXiv:2310.11689](https://arxiv.org/abs/2310.11689), EMNLP Findings 2023). Verified from PDF
  §3: "We use the area under the accuracy-coverage curve (AUACC) metric to measure selective
  prediction performance ... **AUACC is the common metric used for evaluating selective
  prediction performance**". They report e.g. CoQA AUACC 91.23% → 92.63%.
- **Wen et al., "Know Your Limits: A Survey of Abstention in Large Language Models"**
  ([arXiv:2407.18418](https://arxiv.org/abs/2407.18418), TACL). Verified from PDF p.11, it lists
  both, side by side: "**Area Under Risk-Coverage Curve (AURCC)** ... Lower AURCC indicates
  better selective QA performance" and "**Area Under Accuracy-Coverage Curve (AUACC)** ...
  Higher AUACC indicates better performance", and its summary box recommends balancing "ER,
  C@Acc, AUROC, AUACC, AURCC". Note the survey spells it **AURCC**, with two Cs.
  Caveat on its citations: it attributes AUACC to Cole et al. 2023 and Xin et al. 2021, and I
  verified that **neither uses that metric** — Cole et al.
  ([arXiv:2305.14613](https://arxiv.org/abs/2305.14613)) report ECE, ROC-AUC and **Coverage@Acc**.
- **AURAC** — Farquhar et al., Nature 2024 (above) — "the total area enclosed by the accuracies
  at all cut-off percentages X%". Same integral, parameterised by rejection rate instead of
  coverage. This is the name a hallucination/UQ reviewer will expect.

**Recommendation for the repo.** `abstention_baselines.py` already documents the exact identity
`AURC = (1 − 1/n) − AUACC` on a shared coverage grid, so the two are affinely equivalent here
and a sign flip converts the headline. Given that Kamath, Xin, Geifman and the whole selective-
classification line use risk-coverage, and given that the repo already reports a DeepSeek
abstention result in AURC units in the README, leading with **AURC (lower better)** and giving
AUACC as the mirror is the more recognisable choice. If AUACC stays primary, cite Chen et al.
2023 for the name and Wen et al. for its currency, and state the identity in the metrics section
so nobody has to guess the direction.

---

## 5. Nearest neighbours

Ordered roughly by closeness. "Differs" is the one-line honest statement.

1. **Vazhentsev et al., NAACL 2025 — Token-level density-based UQ**
   ([arXiv:2502.14427](https://arxiv.org/abs/2502.14427)). *The closest work, and a scoop of
   the feature.* Token-level RMD inside generated traces, correct-response foreground,
   all-token background, multi-layer, stacked with sequence probability in a supervised readout.
   Their ATRMD *aggregator* — the same trace mean, computed here on our own reference at a
   single layer, as `rmd_full`; we do not reproduce their multi-layer supervised SATRMD
   pipeline — **recovers our entire increment on
   two of the three models** (2026-08-10); the tail window we add on top of it is load-bearing
   only on Qwen2.5-7B-Instruct. Treat the statistic as theirs.
   **Differs:** single-generation QA/summarisation/fact-checking (no sibling traces, no
   plurality vote), no prompt-level aggregation, PRR not AUACC, and no consistency feature in
   the combined readout. Those four are the whole distance between the papers.
2. **Zhang et al., 2025 — Reasoning models know when they're right**
   ([arXiv:2504.05419](https://arxiv.org/abs/2504.05419)). Hidden-state probes on
   R1-Distill-Qwen-7B and friends, on MATH/GSM8K/AIME, ROC-AUC > 0.7 (up to > 0.9 on AIME).
   **Differs:** supervised MLP probe on raw activations rather than a positive-only density
   score; **no baseline of any kind** — no length, entropy, logprob or vote agreement; per-trace
   verification for early exit, not prompt-level abstention.
3. **Fu et al., 2025 — DeepConf** ([arXiv:2508.15260](https://arxiv.org/abs/2508.15260)).
   Tail-region and bottom-decile confidence statistics over traces, confidence-weighted voting
   and filtering, on AIME/HMMT/BRUMO/GPQA. **Differs:** purely output-side (top-k logprob), and
   the goal is answer *selection and token savings*, not an abstain/answer decision — but it
   owns the "low-order statistic over the tail beats the global mean" idea that `rmd_tail_q20`
   reuses. On our data that idea only holds for Qwen2.5-7B-Instruct; on both reasoning-distilled
   models the global mean is as good, so we borrow the move without confirming it.
4. **Chen et al., ICLR 2024 — INSIDE / EigenScore**
   ([arXiv:2402.03744](https://arxiv.org/abs/2402.03744)). Covariance-spectrum geometry over K
   sampled responses' embeddings, benchmarked against Lexical Similarity, LN-Entropy and
   perplexity. **Differs:** geometry *across siblings* replaces consistency rather than
   supplementing it; QA hallucination, not math; AUROC/PCC.
5. **Ren et al., ICLR 2023 — OOD detection & selective generation for CLMs**
   ([arXiv:2209.15558](https://arxiv.org/abs/2209.15558)). RMD on mean-pooled encoder/decoder
   embeddings, combined with sequence likelihood for selective generation. **Differs:** one
   embedding per input/output rather than per token; summarisation and translation under domain
   shift; the target is OOD-ness, not correctness of a math answer.
6. **Sun et al., 2026 — LLM Reasoning as Trajectories**
   ([arXiv:2604.05655](https://arxiv.org/abs/2604.05655)). Step-specific representation-subspace
   geometry on GSM8K and MATH-500, ROC-AUC 0.87 for mid-reasoning correctness, plus trajectory
   steering. **Differs:** per-trace mid-generation prediction and intervention, not prompt-level
   abstention; no self-consistency control found in the text I extracted.
7. **Kirin, 2026 — Operational Proto-Introspection in Looped LMs**
   ([arXiv:2607.18553](https://arxiv.org/abs/2607.18553)). Hidden states add **+0.066 AUROC
   [+0.021, +0.112]** over length + logprob on GSM8K, framed as improving risk-coverage over
   "shortcut-only" scores. **Differs:** looped 2.6B transformer, GSM8K/Horizon Logic, AUROC, and
   the baseline lacks both entropy and vote agreement — but this is the closest published
   instance of the exact claim *shape*, including the CI-excluding-zero increment.
8. **Orgad et al., ICLR 2025 — LLMs Know More Than They Show**
   ([arXiv:2410.02707](https://arxiv.org/abs/2410.02707)). Probes on exact-answer-token
   activations; error taxonomy built from K=30 resamples. **Differs — and cuts against us:**
   shows hidden states predict the *resampling agreement structure*, i.e. the thing B0's
   `vote_agreement` measures. Also documents that error detectors do not transfer across tasks,
   which is consistent with the repo's failed cross-model within-prompt gate.
9. **Kossen et al., 2024 — Semantic Entropy Probes**
   ([arXiv:2406.15927](https://arxiv.org/abs/2406.15927)). Hidden-state probes predicting
   semantic entropy, AUROC vs SE and vs accuracy. **Differs:** supervised, targets a consistency
   quantity rather than adding to one; QA, not math.
10. **Vashurin et al., TACL 2025 — LM-Polygraph benchmark**
    ([aclanthology.org/2025.tacl-1.11](https://aclanthology.org/2025.tacl-1.11/),
    [arXiv:2406.15627](https://arxiv.org/abs/2406.15627)). Eleven tasks, head-to-head across
    information-, consistency- and density-based UQ. **Differs:** ranking, not incremental
    analysis — and it is the source of the "density-based scores underperform for generation"
    prior that a positive geometry increment has to argue against.
11. **Ni et al., 2025 — ReProbe** ([arXiv:2511.06209](https://arxiv.org/abs/2511.06209),
    accepted ACL 2026). Transformer probe on frozen-LLM internal states scoring reasoning-step
    credibility for test-time scaling; claims parity with PRMs up to 810× larger. **Differs:**
    step-level credibility for search/scaling, not abstention. I could **not** confirm from the
    abstract page whether MATH-500 is among its benchmarks or whether self-consistency is a
    baseline — **not verified**, check the PDF before citing specifics.

### Explicitly not found

- No paper reporting a hidden-state/geometry feature's **incremental** selective-prediction value
  over a self-consistency or vote-agreement baseline. **Not found.**
- No semantic-entropy paper reporting numbers on **MATH-500 or MATH**; SVAMP is the only math
  benchmark in Farquhar et al. **Not found.**
- ~~No prior use of a **low quantile of RMD over a trace tail region** as an uncertainty
  statistic.~~ **Withdrawn 2026-08-10.** The repo computes no such quantile. `rmd_tail_q20` is a
  *mean* of per-token RMD over the final 20% of tokens, which is exactly Vazhentsev et al.'s
  token-RMD mean under DeepConf's tail window. Both components are published; the composition is
  not claimed as a gap. Confirmed on data the same day: the untailed mean recovers the whole
  increment on two of three models.
- No published selective-prediction result on MATH-500 that controls for the dataset's
  **human-annotated level 1–5** difficulty. **Not found** — though absence here is weaker
  evidence, since a difficulty control is the kind of thing buried in an appendix.

---

## 6. Adaptive sample allocation — the crowded area the allocation rung enters

Added **2026-08-10, before the allocation precheck ran**, so the baselines are on record
ahead of the numbers. The relevant claim shape is *"spend a fixed total sampling budget
non-uniformly across prompts, and beat spending it uniformly."* That claim is not available:
it is four papers old and the strongest versions of it are output-side and cheap. Nothing in
the allocation rung is a contribution on adaptive allocation. **The only thing that could be
new is geometry's increment over these policies at a budget where they are weak**, and the
regime where they are weak is small — one or two samples per prompt, where a vote-based rule
has nothing to count yet.

- **Aggarwal, Madaan, Yang, Mausam, "Let's Sample Step by Step: Adaptive-Consistency for
  Efficient Reasoning and Coding with LLMs"** ([arXiv:2305.11860](https://arxiv.org/abs/2305.11860),
  EMNLP 2023). Draws samples sequentially per question and stops on a **lightweight posterior
  over the answer counts**. Verified from the paper: it proposes Beta, Dirichlet, Chinese
  Restaurant Process, entropy-based, majority-based and random criteria, and **Beta is the
  recommended default** — it integrates the posterior probability that the runner-up answer
  overtakes the leader, `∫₀^0.5 p₂^{v₂}(1−p₂)^{v₁} dp₂` over the top-two counts `v₁, v₂`, and
  stops once that probability falls below a confidence threshold `C_thresh` (default 0.95).
  Reports up to 7.9× budget reduction at <0.1% mean accuracy drop across 17 datasets and three
  LLMs. **This is the baseline the allocation claim has to beat**, and its knob — one scalar
  threshold — is what we tune out of fold to hit a matched average budget.
- **Li, Yuan, Feng, Pan, Wang, Sun, Wang, Li, "Escape Sky-high Cost: Early-stopping
  Self-Consistency for Multi-step Reasoning"** ([arXiv:2401.10480](https://arxiv.org/abs/2401.10480),
  ICLR 2024). The same idea with a simpler, stricter rule: sampling is divided into windows of
  size `w` (they use **w = 8 on MATH**, 5 elsewhere) and stops as soon as **every sample inside
  one window agrees** — zero answer entropy in the window. They add a scheme for choosing
  `(w, L)` from a target budget and performance level using a first window of `w₀ ≈ 5` samples.
  Reported −33.8% samples on MATH, up to −84.2% on Coin Flip. Note the direct collision with our
  setup: with 8 cached siblings and 70% / 89% / 53% of prompts already unanimous, ESC's MATH
  configuration would stop most of our prompts immediately, which is exactly why the small-budget
  end is the only regime where geometry has room.
- **Damani, Shenfeld, Peng, Bobu, Andreas, "Learning How Hard to Think: Input-Adaptive
  Allocation of LM Computation"** ([arXiv:2410.04707](https://arxiv.org/abs/2410.04707), Oct
  2024). The closest work *by target*: it trains a predictor of the **reward distribution given
  an input and a computation budget** and allocates extra computation where it is predicted to
  help most, over programming, mathematics and dialogue. Claims ~50% compute saved at no quality
  loss, or up to +10% quality at fixed budget. **This is the paper that owns "predict the gain
  from more compute, then allocate"** — our `g(p) = a(p,8) − a(p,1)` target is a coarse special
  case of their reward-vs-budget model. **Differs:** their difficulty features are output-side
  and prompt-side, not hidden-state geometry; the allocation is over a routing/reranking
  pipeline; no per-prompt hidden-state signal is used.
- **Kim, Yang, Min, Jung, "Reliability-Aware Adaptive Self-Consistency for Efficient Sampling
  in LLM Reasoning"** ([arXiv:2601.02970](https://arxiv.org/abs/2601.02970), Jan 2026; v2 Apr
  2026). **Identifier verified against the arXiv listing** — it exists and is what the title
  says. ReASC replaces frequency-only stopping with **response-level confidence**: a first phase
  resolves single high-confidence responses, then a second accumulates frequency *weighted by
  confidence*. Up to 70% cost reduction on GSM8K. This is the closest thing to "put a confidence
  score inside the stopping rule", which is structurally what a geometry-driven allocator does —
  the difference is only which score. Cite it as the reason the contribution cannot be
  "confidence-aware allocation" in general.

**What this means for the allocation rung.** The contribution, if any, is *not* adaptive
allocation and *not* confidence-aware stopping. It is the narrower question of whether a
**single-trace hidden-state feature** predicts the marginal gain from more samples well enough
to beat count-based stopping **at budgets where counting has not started yet**. Write the
expected-loss-at-larger-budgets result down in advance, since ESC and Adaptive-Consistency both
get strong the moment there are votes to count.
