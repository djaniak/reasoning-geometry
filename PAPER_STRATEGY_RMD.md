# Paper Strategy (RMD): Hidden-State Geometry for Reasoning Reliability

*Durable working document (2026-06-19). Survives context compaction. Seeds the
second literature-grounding pass. Sources: `EXPERIMENT_LOG.md` (2026-06-14 audit),
`FINDINGS.md` (now carries a correction banner), deep-research run `wf_ea6e6b93-88e`.*

> **Split out of `PAPER_STRATEGY.md`, 2026-08-15.** This file covers the
> selective-prediction / Mahalanobis-geometry paper only. The arithmetic-DAG
> activation-patching thread is a **separate paper** with a separate document,
> [`PAPER_STRATEGY_DAG.md`](PAPER_STRATEGY_DAG.md). It uses a different model,
> different data, and a different claim, and shares no result with this one.
> Nothing in this file speaks to it, and it should not be cited as if it did.
> Section numbers below are unchanged from the pre-split file, so existing `§n`
> references still resolve.

---

## 1. Current thesis (2026-08-22)

> **On all 500 MATH-500 prompts under a fixed eight-sample budget, adding a
> relative-Mahalanobis tail score to mean length, token entropy, token log
> probability and plurality agreement improves prompt-level selective prediction
> on three 7B/8B checkpoints. The fixed-pipeline AURC increments are −0.0520,
> −0.0284 and −0.0469. A high pooled trace AUROC does not establish sibling-level
> verification: a reproduced last-token probe falls from 0.901/0.914/0.903 pooled
> to 0.644/0.582/0.718 macro within prompt. A deployable peer-agreement baseline
> is a genuine paid competitor, not a mechanism control; at one extra generation
> the six model-pair comparisons yield four ties, one RMD win and one peer win
> after Holm correction.**

The contribution of this work is the evaluation rather than a new geometry
statistic. Token-level RMD and whole-trace ATRMD are already defined by Vazhentsev
et al. On both reasoning-distilled models, ATRMD recovers nearly the full increment.
Restricting the mean to the final 20% of tokens matters only on Qwen2.5-7B-Instruct.
The paper should therefore present the tail as a model-dependent localization, and
should foreground the controlled increment over a strong self-consistency baseline.

The current evidence supports post-generation abstention and risk ranking after
eight traces. It does not establish single-trace verification, pre-generation
routing, adaptive compute allocation, or within-prompt reranking. The allocation
precheck failed: single-trace geometry ranks the gain from another sample in the
wrong order on all three models. The remaining evidence gate is the registered
full-refit stability sweep; no new score or application experiment precedes it.

The label-efficiency result is limited to a small budget. At 50 labelled prompts,
geometry leads a pooling-matched linear probe by −0.033 AURC. The gap disappears at
100 prompts, and the probe leads at larger budgets. General sample efficiency should
not be claimed.

The second contribution of the paper is methodological. It shows how trace length,
generation caps, parse failures, layer selection, weak difficulty controls, and
fixed-prediction uncertainty can inflate claims about reasoning geometry. Prior
trace-correctness and tail-mechanism claims shrink or fail once these controls are
applied.

Full evidence: `EXPERIMENT_LOG.md` entries dated 2026-08-06 through 2026-08-10 and
`RELATED_WORK.md` §nearest-neighbours.

## 2. Verified, de-confounded findings (what we can claim)

The 2026-07-18 Qwen localized rerun is the result the current argument rests on.
The older cross-model, temperature, low-dimensional, and selective-prediction claims
listed below remain historical until clean-budget replications are collected.

- **Length is a strong correctness baseline** (pooled AUC ~0.74 Qwen / ~0.83
  DeepSeek) that prior geometry and UQ work never benchmarked against as a standalone
  predictor.
- **On clean Qwen Best-of-8 traces, entropy-localized RMD beats full-trace RMD and a
  matched random-token control at all three layers.** It remains competitive with
  output baselines and has only suggestive incremental probe value.
- **The signal is between-prompt (solvability) rather than within-prompt.** Pooled
  parseable RMD is positive, whereas within-prompt (per-attempt) performance is at
  chance (Qwen 0.515, DeepSeek 0.27 at n=13).
- **Termination is mostly recoverable from length** (Qwen length detects unparsed at
  0.996, against RMD 0.84). The claim "RMD rejects non-terminating traces" should
  therefore not be used as a headline.
- **The earlier low-dimensional distillation and transfer claims are rejected or
  weakened by parseable-only audits.** They should not be used as current paper
  pillars.
- **Selection is closed.** Geometry does not improve Best-of-8 tie-breaking. The
  useful downstream direction is prompt-level abstention and compute allocation.
- **The scarce-label advantage is two thirds decision-function form and one third
  positive-only supervision, and neither component survives 100 labels** (2026-08-08
  ladder, §7f). This claim replaced the earlier statement that one-class geometry is
  unusually sample-efficient, which had been measured against a probe differing in
  three respects at once.

## 3. Contamination ledger (CRITICAL: re-validate before using)

Mechanism: `collect_data.py` auto-labels no-parseable-answer traces as incorrect; truncated
(length-capped) generations dominate the "incorrect" class.

| Data | unparsed | capped | unparsed share of incorrect |
|---|---:|---:|---:|
| DeepSeek greedy MATH-500 | 43% | 51% | 75–79% |
| DeepSeek Best-of-N MATH-500 | 45% | ~55% | 78% |
| Qwen greedy / BoN MATH-500 | 8% | 8% | 17% |

**Contaminated, NOT yet re-validated** (need parseable-only rerun; mechanism added:
`selective_prediction.py --exclude_unparsed`): main-results table (DeepSeek combined 0.859),
selective-prediction AUSC (DeepSeek 0.633), difficulty/subject stratification. The
"DeepSeek ≫ Qwen geometry effect" tracks differential truncation (43% vs 8%).
**RETRACTED:** within-prompt 0.93 "genuinely trace-level" claim.

**Amended 2026-08-03.** The mechanism stated above is correct about the *label* and
incorrect about the *traces*. Capped traces resumed to 16,384 tokens terminate 70%
[0.56, 0.81] of the time, and are correct in 46% of those cases, against 5.6% as
scored. Capping is a budget shortfall, so the contaminated rows are censored
observations rather than failures. This has two consequences for the paper. First,
exclusion is defensible as missing-data handling, which is a stronger footing than
dropping degenerate traces. Second, the paper should not state that geometry detects
non-termination, because what it detects is an unfinished trace, and the distinction
is checkable. Reviewers of a truncation-heavy table are likely to raise this point,
and the continuation study provides the answer. Ledger: `EXPERIMENT_LOG.md`
(2026-08-03).

## 4. Literature map (deep-research wf_ea6e6b93-88e, 21/25 claims confirmed)

- **The RMD primitive is NOT novel.** NAACL 2025 [arXiv:2502.14427] (SAT(R)MD) already
  computes token-level RMD against a C4 background. *This is the closest overlap and
  should be read first.* Differentiation rests on: a trace-level single score, a
  positive-only one-class fit, length control, and the reasoning/BoN application.
- **Trained probes are strong baselines that reviewers will expect us to beat:**
  PCA+LDA ~80% on MATH [SWIFT, arXiv:2505.12225]; Semantic Entropy Probes
  [arXiv:2406.15927]; TrajSelector [arXiv:2510.16449]. A positive-only RMD fit is
  likely to lose to these at raw correctness.
- **Within-prompt geometry has repeatedly failed in prior work.** INSIDE/EigenScore
  [arXiv:2402.03744] is within-prompt self-consistency and scores *below random* on
  GSM8K in the ACL-2025 survey [2025.findings-acl.1101]. Between-prompt difficulty is
  therefore the open direction.
- **The length-as-baseline gap is real**, but it should be stated as "under-controlled
  in prior work" rather than "never studied" (2-1 vote). Semantic entropy [Nature 2024]
  captures only confabulations and misses systematic reasoning errors, which motivates
  a single-trace representation signal.
- **Refuted or over-claimed in prior work:** SE-as-universal-SOTA,
  latent-reranking-beats-majority, GenRM-as-SOTA, SWIFT-beats-heavy-RMs. This reduces
  the competitive pressure and is consistent with our at-chance within-prompt finding.

## 5. Three candidate framings (ranked)

1. **Measurement-and-reframe paper (recommended):** "What does hidden-state geometry
   actually measure in reasoning models? A length-controlled re-examination." Length
   confound + truncation artifact + between/within decomposition + between-prompt
   solvability application beating length, and competitive with a trained probe at
   full labels. ~~and ahead of it below ~100 (§7f)~~ **[Dropped 2026-08-22 with
   claim 3 below, for the same reason.]** This framing covers only claims that the
   evidence supports, and it does not require us to beat trained probes at raw
   correctness.
2. **Between-prompt difficulty for ~~test-time compute allocation~~ / routing** (constructive).
   **[Narrowed 2026-08-10.]** The allocation half is closed; see the amendment in §1.
   What remains is routing and abstention by predicted difficulty, which the same
   precheck supports at one trace (AUROC 0.790 / 0.674 / 0.688).
3. ~~**Single-trace label-efficient RMD beating length+entropy**~~ **[Cut from the
   paper 2026-08-22.]** Stating this claim correctly requires three qualifications
   at once. The effect is confined to 25–100 labels; §7f attributes it to the
   quadratic decision function rather than to the geometry; and the three
   label-efficiency runs score different eval sets (the complement of each run's
   *largest* budget, 400 vs 100), so crossing points do not transfer between
   them. A short paper cannot carry all three qualifications, and omitting any one
   of them over-claims. The record is kept in notebook 17 §8. If a single sentence
   is ever required, it should state that a positive-only fit is an inexpensive
   route to a quadratic decision boundary in the scarce-label regime. It should
   not state that geometry is label-efficient.

## 6. Baselines / experiments a top venue will demand

- ~~Trained linear probe (PCA+LDA, SEP-style): the required baseline. Does it also collapse
  to length?~~ **DONE 2026-07-31, both models; see §7e.** The probe
  does not collapse (rho ~0.22 DeepSeek), and neither does RMD, whereas entropy and logprob
  do. Under length control, the probe does not reliably beat RMD on either model. The
  RMD-versus-EigenScore comparison on reasoning is still open.
- ~~Matched comparator ladder: is the probe gap supervision, decision-function form, or
  pooling order?~~ **DONE 2026-08-08, three models; see §7f.** A reviewer who reads
  "one-class geometry needs fewer labels" is likely to ask this question, because
  `probe_hidden_tail_q20` differed from `rmd_tail_q20` in all three respects at once. Two thirds
  of the 50-label gap is decision-function form and one third is supervision, while pooling
  order runs in the opposite direction. The decomposition should be reported without waiting
  for a reviewer to request it.
- Length baseline everywhere (done: `rmd_minus_length` wired into prompt_decomposition).
- Pilot re-collection @ larger `max_new_tokens` (CONFIRMED by probe: Qwen-distill 8192,
  Llama-distill 12288 → 0% truncation) to get clean, well-powered parseable numbers.
- Parseable-only re-validation of selective prediction + stratification (flag exists, not run).
- Confirm the truncation/auto-label pitfall is undocumented (targeted search).
- Selective-prediction / compute-allocation eval protocol (no standard benchmark exists).

## 7b. Pass-2 results (wf_a8915e2f-3b3, 21/25 confirmed) + FINAL recommendation

Grounding of the four FINDINGS-specific results:
1. **Distillation→low-dim correctness geometry: MOST NOVEL.** No prior work links
   distillation/R1-RL to compression of a hidden-state *correctness* manifold. Adjacent:
   intrinsic-dim work is SFT-vs-ICL / prompting / parameter-subspace only; truth-subspace
   work (universal truthfulness hyperplane = **arXiv:2407.08582**; LID↔truthfulness =
   arXiv:2402.18048) never covers distilled checkpoints. ⚠️ The scope must be stated
   carefully: instruction tuning *increases* intrinsic dimension (2402.18048), which is the
   opposite direction, and parameter-subspace ID is not activation-manifold ID. The support
   is therefore analogical, and direct evidence has to be supplied.
2. **"Manifold shape transfers, readout does not": NOVEL framing**, grounded by the Platonic
   Representation Hypothesis (arXiv:2405.07987) and Linear Representation Transferability
   (arXiv:2506.00653), and complemented by Orgad et al. (error detectors do not
   transfer, arXiv:2410.02707) and SEP, which do not pre-empt it. ⚠️ LRT transfers steering
   vectors within one family, not covariance manifolds, so the support is analogical only.
3. **Selective prediction/abstention: crowded.** Established metrics are **AURCC / AUACC /
   Coverage@Acc / R-Acc / ER** (TACL 2025 'Know Your Limits', arXiv:2407.18418). These should be used,
   NOT "AUSC". Strong single-pass baselines to beat: supervised latent correctness probe
   **~0.84 AUC on MATH** (arXiv:2511.14773), Semantic Entropy Probes. Competitors: SelectLLM,
   AbstentionBench. Novelty rests only on the combination of **positive-only supervision, a
   single forward pass, and improvement over length, entropy and SE**. Motivation:
   **reasoning fine-tuning degrades abstention by 24%** (AbstentionBench,
   arXiv:2506.09038, NeurIPS 2025).
4. **Truncation pitfall: WEAK on its own.** A related difficulty-driven sample-selection artifact
   in long-CoT probing is already published (arXiv:2511.14773). It should be included as
   methodology rather than as a separate contribution.

**Blocking citation fix:** arXiv:2412.06245 = Janapati & Ji (intrinsic dim, SFT-vs-ICL), NOT
the truthfulness-hyperplane paper (that is arXiv:2407.08582).

### FINAL recommended paper, MECHANISM-first (deviates from pass-2's "lead with abstention")

Rationale for deviating: our de-confounded correctness AUC (DeepSeek RMD 0.636) is well below
the supervised latent probe (~0.84). Leading with the application invites a comparison that we
lose. The paper should lead with the mechanism, where the literature confirms genuine novelty,
and use abstention as a demonstration with a clearly stated scope.

> **Title direction:** "How Reasoning Distillation Reshapes the Geometry of Correctness."
> **C1 (mechanism, novel):** distillation/RL compresses the correctness-relevant hidden-state
>   structure into a low-dimensional subspace (dim~8 distilled vs 64–128 instruct).
> **C2 (mechanism, novel):** the correct-reasoning manifold *shape* transfers across
>   same-architecture instruct↔distilled models, whereas the accuracy-trained *readout* does
>   not (PRH/LRT).
> **C3 (application):** positive-only, single-forward-pass RMD for selective prediction on
>   reasoning. It improves on length, entropy and semantic entropy at zero supervision, while
>   supervised probes perform better but require labels. The motivation is that reasoning
>   fine-tuning degrades abstention.
> **Rigor thread:** length is the baseline to beat, and the truncation and
>   auto-label-incorrect confounds must be controlled.
> **Positioning:** RMD is trace-level (vs SAT(R)MD token-level), positive-only (vs PCA+LDA/SEP/
>   SelectLLM), between-prompt/cross-model (vs INSIDE within-prompt).

### GATING REALITY (critical)
Every number that C1, C2 and C3 currently rest on comes from **artifact-contaminated data**;
the low-dim sweep, the transfer grid and the AUSC were all computed before the fix. Before any
of this becomes a paper, the following are required:
- Pilot re-collection at proper budget → clean parseable data. **Budgets confirmed by probe
  (2026-06-19):** deepseek(-Qwen) 8192 (residual ~12% non-terminating), deepseek-llama 12288 (0%
  truncation, heavy ~11k tail). Wired into `params.yaml bestofn_matrix` with per-arch layers
  (Qwen 7/14/21, Llama 8/16/24); pilot deepseek bumped to 8192.
- Re-validate C1 (low-dim sweep), C2 (transfer grid), C3 (risk-coverage) **parseable-only**,
  with the full baseline suite: length, entropy, **Semantic Entropy Probes**, and the
  **supervised latent probe (~0.84)** head-to-head.
- Realistic expectation: RMD (0.64) is below the supervised probe on raw AUC. The case for the
  paper rests on *positive-only supervision, a single pass, cross-model evidence, and the
  mechanism*, rather than on raw accuracy.

## 7c. DE-RISK OUTCOME (2026-06-19, parseable-only on existing greedy data)

Ran C1 (one_class dim sweep) and C2 (cross-model transfer) ALL vs parseable. Verdict:

- **C1 REJECTED.** DeepSeek's "dim~8 plateau" was 100% a truncation artifact: on contaminated
  data RMD reaches 0.86 at dim 2, because truncated traces are trivially separable, whereas on
  parseable-only data DeepSeek climbs monotonically to dim 128 (max 0.77), exactly as Qwen does.
  There is no distillation-induced low-dimensional compression.
- **"DeepSeek > Qwen geometry effect" REJECTED.** DS parseable best 0.77 < Qwen 0.79.
- **C2 WEAKENED/PARTIAL.** The claim that the readout does not transfer survives (classifier
  transfer is near chance in both directions). The claim that shape transfers survives in one
  direction only (Qwen-eval/DS-ref 95–99% retention) and rests on a weak raw-Mahalanobis
  signal (~0.7); the DS-eval direction is uninformative, since its own parseable raw-Mahalanobis
  is ~0.55. The result is neither RMD-based nor cross-architecture, and it is no longer a strong
  standalone contribution.
- **SURVIVING POSITIVES (de-confounded, parseable):**
  - **RMD ≫ raw Mahalanobis: +0.177 (DS), +0.143 (Qwen).** Background subtraction is the
    mechanism, and raw Mahalanobis is close to useless on reasoning. The effect strengthens
    after de-confounding, so this is a robust contribution.
  - **RMD > length + entropy** (between-prompt pooled; +0.05–0.09 over length, CI excludes zero).
  - **Dissociation:** geometry measures between-prompt solvability rather than within-prompt
    correctness.

### REVISED paper (mechanism-first is OFF; C1 dead, C2 weak)
> **"What hidden-state geometry actually measures in reasoning models, once length
> and truncation are controlled."** (1) Cautionary and methodological part: geometry
> "correctness detectors" are largely length and truncation detectors, demonstrated by
> collapsing a 0.93→chance within-prompt result and a dim-8→dim-128 plateau. (2) Surviving
> part: a between-prompt SOLVABILITY signal in which RELATIVE geometry (RMD) is essential
> (≫ raw Mahalanobis, and above length+entropy). It is useful for abstention and compute
> allocation, but not for per-attempt reranking.
Strength: findings or short-paper level, and plausible for a main conference given a clean
difficulty/abstention benchmark and cross-model breadth. The claims are modest but supported.

### Remaining cheap checks (analysis-only, existing data)
1. C3: `selective_prediction.py --exclude_unparsed`. Does RMD abstention beat length+entropy
   parseable? (decides whether the abstention application stays). MOST IMPORTANT un-run number.
2. RMD-vs-raw across all 4 models parseable (incl. Llama greedy data), for breadth on the mechanism.
3. ~~Pass-rate/difficulty correlation parseable, for the compute-allocation motivation.~~
   **[Done 2026-08-10, and it went further than a correlation.]** Experiment 2 puts
   peer-model pass rates *inside* the readout as a control rather than correlating
   them alongside: `EXPERIMENT_LOG.md` (2026-08-10). The hook survives, at ~20% of
   the increment's original size.

## 7d. 2026-07-18 UPDATE (Qwen BoN full rerun: localization + abstention numbers land)

Fresh `evaluate_prompt_decomposition@0` / `evaluate_prompt_selection@0` rerun post
truncation-fix. Ledger: `EXPERIMENT_LOG.md` 2026-07-18; narrative: `FINDINGS.md`
"Localized Geometry" section. Impact on the revised paper of §7c:

- **The §7c thesis SURVIVES and acquires a second component.** The between-prompt and
  within-prompt dissociation replicates on clean Qwen data (ICC 0.94–0.97; within-prompt
  geometry ties entropy and logprob but never exceeds them). The abstention application now
  has numbers as well. Prompt-level abstention at 50% coverage gives rmd_tail_q20 **0.836**
  against length 0.740, logprob 0.680 and entropy 0.672 (exploratory, and confidence
  intervals are still required from the selective-prediction stages). This is
  §7c-check-1 "MOST IMPORTANT un-run number" in trace form, and it passes.
- **NEW contribution-bearing positive: entropy-localized RMD.** rmd over highest-entropy
  20% tokens beats full-trace rmd at ALL THREE layers (+0.05 centered, p ≤ 0.006) with a
  matched random-token control failing. The test was prespecified, entropy-specific and depth-monotone.
  Plus: label-free residualization shows the L21 signal is linearly complementary to
  entropy+logprob+length (residual within-macro 0.645 vs 0.654 raw). This strengthens the
  mechanism account: the trace-level information is concentrated in geometry measured at
  *uncertainty forks*, even though head-to-head it only ties the output baselines.
- **Selection and reranking are now CLOSED, with a structural explanation.** At N=8 only
  39 of 500 prompts tie and about 10 have headroom (a ceiling of roughly 2 points); all
  tie-break deltas are ≤0.006 with p ≥ 0.248, and weighted-RMD voting underperforms
  majority voting. No further compute should be spent in this direction.
- **Contrastive supervision is a NEGATIVE result and should be reported briefly.** OOF
  cross-prompt directions are real (alignment 0.18–0.22 against a null of 0.10 at L21) but
  add nothing over matched localized RMD. The determining factor is therefore the choice of
  region rather than supervision.
- **Multiplicity discipline for the paper:** the only unadjusted-significant incremental
  probe cell (L21 macro +0.049, p=0.024) must be framed as suggestive; pre-specify the
  deepest layer and the two surviving contrasts (localization, entropy-specificity) for
  the cross-model confirmation on deepseek/llama/deepseek_llama full runs
  (deepseek_llama decomposition outputs are currently deleted and must be regenerated first).

## 7e. 2026-07-31 UPDATE (supervised probe ceiling + length residualization, BOTH models)

Ran the baseline flagged as required in §6, the trained PCA+LDA probe, on both models,
plus E1R (abstention with trace length partialled out in rank space). Narrative:
`FINDINGS.md` "Supervised Probe Ceiling and Length Residualization". Exploratory,
not pre-registered.

**§6 asked whether the trained probe also collapses to length. It does not, and neither
does RMD; the output baselines are the ones that collapse.**

- **The label-efficiency argument is now at its strongest form.** Once length is partialled
  out of both scorers, the supervised probe does **not** reliably beat the positive-only
  RMD fit on either model (probe − rmd_tail_q20: +0.029 Holm 0.090 Qwen, +0.033 Holm 0.126
  DeepSeek). The one Holm-surviving cell has the probe *losing* (Qwen he_q20, 0.018).
  **Qualified 2026-08-08 (§7f):** this is a full-budget, length-residualized *tie*, and
  reading it as a positive-only *win* over-claims in two respects. The probe here also differs
  in pooling order and in being linear, and at 400 labels it takes the lead on solo AUROC.
  This is the comparison that §7b's "GATING REALITY" expected to lose ("honest expectation:
  RMD loses raw AUC to the supervised probe"); at prompt-level abstention, under length
  control, it does not. The scope should be noted: this is between-prompt abstention AURC and
  not raw trace-correctness AUC, where the ~0.84 supervised probe number still stands.
- **RMD survives length control on both models; entropy/logprob do not on DeepSeek.**
  Δ vs an uninformative scorer: rmd_tail_q20 +0.161 [+0.128, +0.194] Qwen /
  +0.107 [+0.077, +0.135] DeepSeek, both Holm < 0.01; entropy and logprob
  +0.009–0.011 on DeepSeek, Holm 1.000. This sharpens §7c's surviving positive
  "RMD > length + entropy" into a stronger claim: **RMD exceeds length on a component that
  length cannot supply, and on the reasoning-distilled model it is the output-side baselines
  that act as length proxies.** A negative control validates the removal
  (a pure-length scorer reaches +0.008 / −0.007, p ≥ 0.82).
- **The raw Spearman-versus-length table should not be used as a headline.** RMD reaches
  rho +0.82 with length on DeepSeek L21, against +0.22 for the probe, which appears to be a
  collapse but is not one. A high rank correlation with length can coexist with a large
  length-independent component, because length itself explains only part of solvability. The
  correlation should therefore be reported together with E1R; reported alone it invites an
  incorrect reading, which this document itself drew before E1R was run.
- **Raw (uncontrolled) probe-versus-RMD, for completeness:** probe_hidden_tail_q20 exceeds
  rmd_tail_q20 by +0.025 (Qwen, Holm 0.056, does not survive) and +0.048 (DeepSeek,
  Holm 0.006, survives). Reviewers will want both the raw and the length-controlled
  version, and the difference between them is itself the finding.
- **One DeepSeek weakness to disclose:** `rmd_high_entropy_q20 − length` = +0.005
  [−0.011, +0.025] p=0.506, so the entropy-localized region does not clear length on
  that model. Only `rmd_tail_q20` does (+0.030 [+0.014, +0.048]). This is consistent with
  §7d's failed localization gate: entropy-localization is Qwen-specific, whereas tail
  localization replicates.
  **[Updated 2026-08-09]** Both halves of that last sentence now require qualification.
  Entropy-localization remains Qwen-specific, but the §7d gate run on
  `deepseek_llama` L24, the model after which its own layer column is named, *passes* on both
  tests. It remains Qwen-specific only because the localized score there is
  0.491, that is, at chance, and because the gate never required either score to exceed it.
  The statement that tail localization is what replicates is now incorrect as a claim about
  the aggregator: between prompts, the untailed `rmd_full` recovers the whole increment
  on both distilled models, and the tail is required only on Qwen. See
  `EXPERIMENT_LOG.md` (2026-08-09, both entries).
  **[Sharpened 2026-08-22, on `full_population`.]** Experiment 1b now says this
  with intervals, and the missing half changes the framing. `rmd_full` (which
  *is* Vazhentsev's ATRMD, published prior art) over `B0`: −0.0287 [−0.0534,
  −0.0055] DeepSeek-Qwen and −0.0445 [−0.0739, −0.0151] Llama, essentially the
  whole increment; but −0.0178 [−0.0435, **+0.0105**] on Qwen, where it does not
  clear zero at all. The correct reading is therefore not that the tail is redundant on
  two models. Each architecture requires a different region, and the two scores
  correlate at Pearson 0.93–0.96. **The increment should be written as the claim and the
  localization as a scope note.** `rmd_tail_q20` is the region that works
  everywhere, not an established contribution. Notebook 14 §4a Table 5a.
- **Breadth, rather than rigor, is now the binding constraint.** ~~n=2 models, both
  Qwen-lineage. Every claim in this section needs the Llama-architecture replication
  (`deepseek_llama`, cancelled by the §7d gate)~~ **[Stale; corrected
  2026-08-09.]** The `deepseek_llama` collect was restored for other reasons and
  finished 2026-08-03, so this section's claims now rest on **three models spanning
  two architecture families**. However, two of the three are reasoning-distilled and
  two are Qwen-lineage, all are 7–8B, and all share one task and one prompt set.
  Breadth is still the binding constraint, and it is now **single-dataset scope**
  rather than model count. Note also that "a property of reasoning-distilled models"
  is exactly the axis the 2026-08-09 localization split falls along, and it rests on
  a single non-distilled model.

## 7f. 2026-08-08 UPDATE (the supervision and form ladder: attribution moved, regime bounded)

Ran the comparator ladder that §6 now records as required. Ledger:
`EXPERIMENT_LOG.md` 2026-08-08 "Splitting the label-efficiency gap into supervision
and decision-function form". Three models, budgets 25/50/100, 30 label draws.

**§7e compared `rmd_tail_q20` against a probe that differed in three respects at once:**
positive-only versus a labelled negative class, quadratic versus linear, and
score-then-pool versus pool-then-score. `qmd_tail_q20` (RMD's own estimator with the
unconditional background replaced by a Gaussian over incorrect traces) separates these
factors.

- **The headline sentence changes.** At 50 labels the −0.033 AURC gap against a
  pooling-matched linear probe decomposes into approximately **−0.011 supervision and −0.018
  decision-function form**. The statement that the positive-only inductive bias accounts for
  the label efficiency must therefore quote −0.011 rather than −0.033. The defensible version
  is that *a positive-only one-class fit is an inexpensive way to obtain a quadratic decision
  function, and that the quadratic accounts for most of the scarce-label advantage.*
- **Both components expire by 100 labels** (+0.000 each, 15/30 and 14/30). The quadratic
  captures what a hyperplane cannot *express*: on a pure shape axis the correct class is 2.9x
  wider with no mean shift, and no linear boundary encodes "too wide". This stops
  costing the probe AURC once the probe can fit its own direction. It should not be written
  that geometry retains an advantage at scale, because the claim is restricted to the
  scarce-label regime.
- **Pooling order runs opposite to the intuition.** Score-then-pool *costs* the LDA
  approximately +0.016 at 50 labels, so §7e's region-mean probe was in fact helped by
  averaging first. The original comparison was therefore conservative in the probe's favour
  rather than inflated in favour of geometry. This should be stated without waiting to be
  asked, since it is the first point a reviewer is likely to suspect.
- **QMD is a strong comparator.** `B0+qmd − B0` reaches −0.058 at 100 (28/30). The labelled
  quadratic is a strong feature that simply requires more labels than the positive-only one.
- **Mechanism, for the figure and the intuition section.** RMD's background Gaussian is a
  *mixture containing the positives*, not a rival class: its mean sits 0.44 from
  `mu_correct` in that class's own metric against the incorrect class's 0.90. So RMD
  already points along the contrast with the magnitude damped (score range 2.0 vs 4.7), so
  labelling the negative class removes the dilution rather than revealing a new direction.
  The identity `RMD − QMD = d_incorrect − d_background` holds exactly, because `d_correct`
  cancels, so the supervision component has a closed form. Figure: `rmd_qmd_geometry.py`.
- **Scope caveat that prevents these numbers from being combined directly with §7e.** These
  budgets cap at 100, leaving ~314–328 evaluation prompts against the frozen run's ~80. Only
  the paired within-run deltas transfer; AURC *levels* and crossing budgets do not.

## 7. Open questions for the SECOND research pass (a): what research #1 did NOT see

Research #1 saw only a curated brief and not FINDINGS.md. The second pass must ground the
following:
- **Distillation reshaping representation geometry** (our low-dim contrast, bimodal layers).
  Is there prior work on RL or distillation changing internal geometry? (research flagged
  this as ungrounded)
- **Cross-model manifold transfer and the Platonic Representation Hypothesis** for
  correctness. Is there prior work? (not evaluated in pass #1)
- **Selective prediction and abstention for reasoning** as the headline application: prior
  art, benchmarks, and which AUSC values are considered strong.
- **Truncation, generation caps and non-termination** as an evaluation confound in o1/R1-style
  reasoning-model papers, including a check on whether this is first documented here.
