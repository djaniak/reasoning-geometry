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
   claim 3 below — same reason.]** Every claim we can support; avoids the fight
   (beating trained probes at raw correctness) we'd lose.
2. **Between-prompt difficulty for ~~test-time compute allocation~~ / routing** (constructive).
   **[Narrowed 2026-08-10.]** The allocation half is closed — see the amendment in §1.
   What survives is routing/abstention by predicted difficulty, which the same
   precheck supports at one trace (AUROC 0.790 / 0.674 / 0.688).
3. ~~**Single-trace label-efficient RMD beating length+entropy**~~ **[Cut from the
   paper 2026-08-22.]** Stating it correctly needs three qualifications at once —
   the effect is confined to 25–100 labels, §7f moves the attribution off the
   geometry and onto the quadratic decision function, and the three
   label-efficiency runs score different eval sets (the complement of each run's
   *largest* budget, 400 vs 100), so crossing points do not transfer between
   them. A short paper cannot carry all three, and dropping any one over-claims.
   Kept as the record in notebook 17 §8. If a sentence is ever wanted: *a
   positive-only fit is a cheap route to a quadratic decision boundary in the
   scarce-label regime* — never that geometry is label-efficient.

## 6. Baselines / experiments a top venue will demand

- ~~Trained linear probe (PCA+LDA, SEP-style) — *the* required baseline; does it ALSO collapse
  to length? (the killer experiment).~~ **DONE 2026-07-31, both models — see §7e.** The probe
  does not collapse (rho ~0.22 DeepSeek); neither does RMD; entropy/logprob do. Length-
  controlled, the probe does not reliably beat RMD on either model. RMD-vs-EigenScore
  head-to-head on reasoning is still open.
- ~~Matched comparator ladder: is the probe gap supervision, decision-function form, or
  pooling order?~~ **DONE 2026-08-08, three models — see §7f.** A reviewer who reads
  "one-class geometry needs fewer labels" asks exactly this, because
  `probe_hidden_tail_q20` differed from `rmd_tail_q20` in all three at once. Two thirds
  of the 50-label gap is form, one third supervision, and pooling order runs the *other*
  way. Volunteer the decomposition; do not wait to be asked for it.
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
   arXiv:2402.18048) never covers distilled checkpoints. ⚠️ Scope carefully: instruction
   tuning INCREASES intrinsic dim (2402.18048) — opposite direction; parameter-subspace ID
   ≠ activation-manifold ID. Support is analogical → we must supply direct evidence.
2. **"Manifold shape transfers, readout doesn't": NOVEL framing**, grounded by Platonic
   Representation Hypothesis (arXiv:2405.07987) + Linear Representation Transferability
   (arXiv:2506.00653); complemented (not pre-empted) by Orgad et al. (error detectors don't
   transfer, arXiv:2410.02707) and SEP. ⚠️ LRT transfers steering vectors (not covariance
   manifolds) within one family — analogical support only.
3. **Selective prediction/abstention: crowded.** Established metrics are **AURCC / AUACC /
   Coverage@Acc / R-Acc / ER** (TACL 2025 'Know Your Limits', arXiv:2407.18418) — use these,
   NOT "AUSC". Strong single-pass baselines to beat: supervised latent correctness probe
   **~0.84 AUC on MATH** (arXiv:2511.14773), Semantic Entropy Probes. Competitors: SelectLLM,
   AbstentionBench. Novelty ONLY via **positive-only + single-forward-pass + beats
   length/entropy/SE**. Hook: **reasoning fine-tuning DEGRADES abstention 24%** (AbstentionBench,
   arXiv:2506.09038, NeurIPS 2025).
4. **Truncation pitfall: WEAK alone.** A related difficulty-driven sample-selection artifact
   in long-CoT probing is already published (arXiv:2511.14773). Fold in as methodology.

**Blocking citation fix:** arXiv:2412.06245 = Janapati & Ji (intrinsic dim, SFT-vs-ICL), NOT
the truthfulness-hyperplane paper (that is arXiv:2407.08582).

### FINAL recommended paper — MECHANISM-first (deviates from pass-2's "lead with abstention")

Rationale for deviating: our de-confounded correctness AUC (DeepSeek RMD 0.636) is WELL BELOW
the supervised latent probe (~0.84). Leading with the application invites the comparison we
lose. Lead instead with the mechanism (where the lit confirms genuine novelty), use abstention
as an honestly-scoped demonstration.

> **Title direction:** "How Reasoning Distillation Reshapes the Geometry of Correctness."
> **C1 (mechanism, novel):** distillation/RL compresses the correctness-relevant hidden-state
>   structure into a low-dimensional subspace (dim~8 distilled vs 64–128 instruct).
> **C2 (mechanism, novel):** the correct-reasoning manifold *shape* transfers across
>   same-architecture instruct↔distilled models; the accuracy-trained *readout* does not (PRH/LRT).
> **C3 (application):** positive-only, single-forward-pass RMD for selective prediction on
>   reasoning — beats length/entropy/semantic-entropy at zero supervision (supervised probes do
>   better but need labels); motivated by reasoning-FT degrading abstention.
> **Rigor thread:** length is the baseline to beat; truncation/auto-label-incorrect confound.
> **Positioning:** RMD is trace-level (vs SAT(R)MD token-level), positive-only (vs PCA+LDA/SEP/
>   SelectLLM), between-prompt/cross-model (vs INSIDE within-prompt).

### GATING REALITY (critical)
Every load-bearing number for C1/C2/C3 is currently on **artifact-contaminated data** (the
low-dim sweep, the transfer grid, and the AUSC were all computed pre-fix). Before any of this
is a paper:
- Pilot re-collection at proper budget → clean parseable data. **Budgets confirmed by probe
  (2026-06-19):** deepseek(-Qwen) 8192 (residual ~12% non-terminating), deepseek-llama 12288 (0%
  truncation, heavy ~11k tail). Wired into `params.yaml bestofn_matrix` with per-arch layers
  (Qwen 7/14/21, Llama 8/16/24); pilot deepseek bumped to 8192.
- Re-validate C1 (low-dim sweep), C2 (transfer grid), C3 (risk-coverage) **parseable-only**,
  with the full baseline suite: length, entropy, **Semantic Entropy Probes**, and the
  **supervised latent probe (~0.84)** head-to-head.
- Honest expectation: RMD (0.64) loses raw AUC to the supervised probe; the paper wins on
  *positive-only + single-pass + cross-model + mechanism*, not on raw accuracy.

## 7c. DE-RISK OUTCOME (2026-06-19, parseable-only on existing greedy data)

Ran C1 (one_class dim sweep) and C2 (cross-model transfer) ALL vs parseable. Verdict:

- **C1 REJECTED.** DeepSeek's "dim~8 plateau" was 100% truncation artifact: on contaminated data
  RMD hits 0.86 at dim 2 (truncated traces trivially separable); parseable-only, DeepSeek climbs
  monotonically to dim 128 (max 0.77) exactly like Qwen. No distillation low-dim compression.
- **"DeepSeek > Qwen geometry effect" REJECTED.** DS parseable best 0.77 < Qwen 0.79.
- **C2 WEAKENED/PARTIAL.** "Readout doesn't transfer" survives (clf transfer ~chance both ways).
  "Shape transfers" survives ONE direction (Qwen-eval/DS-ref 95–99% retention) on a weak raw-Mahal
  signal (~0.7); DS-eval direction uninformative (its own parseable raw-Mahal ~0.55). Not RMD-based,
  not cross-arch. No longer a strong standalone contribution.
- **SURVIVING POSITIVES (de-confounded, parseable):**
  - **RMD ≫ raw Mahalanobis: +0.177 (DS), +0.143 (Qwen)** — background subtraction is the mechanism;
    raw Mahal near-useless on reasoning. STRENGTHENS after de-confounding. (Robust contribution.)
  - **RMD > length + entropy** (between-prompt pooled; +0.05–0.09 over length, CI excludes zero).
  - **Dissociation:** geometry = between-prompt solvability, NOT within-prompt correctness.

### REVISED paper (mechanism-first is OFF; C1 dead, C2 weak)
> **"What hidden-state geometry actually measures in reasoning models — once you control for length
> and truncation."** (1) Cautionary/rigor hook: geometry "correctness detectors" are largely
> length/truncation detectors — demonstrated by collapsing a 0.93→chance within-prompt result and a
> dim-8→dim-128 plateau. (2) What survives: a between-prompt SOLVABILITY signal where RELATIVE
> geometry (RMD) is essential (≫ raw Mahal; beats length+entropy), useful for abstention / compute
> allocation, not per-attempt reranking.
Strength: findings/short-paper level; main-conf-plausible with a clean difficulty/abstention
benchmark + cross-model breadth. Honest and contribution-bearing, not splashy.

### Remaining cheap checks (analysis-only, existing data)
1. C3: `selective_prediction.py --exclude_unparsed` — does RMD abstention beat length+entropy
   parseable? (decides whether the abstention application stays). MOST IMPORTANT un-run number.
2. RMD-vs-raw across all 4 models parseable (incl. Llama greedy data) — breadth for the mechanism.
3. ~~Pass-rate/difficulty correlation parseable — the compute-allocation hook.~~
   **[Done 2026-08-10, and it went further than a correlation.]** Experiment 2 puts
   peer-model pass rates *inside* the readout as a control rather than correlating
   them alongside: `EXPERIMENT_LOG.md` (2026-08-10). The hook survives, at ~20% of
   the increment's original size.

## 7d. 2026-07-18 UPDATE (Qwen BoN full rerun: localization + abstention numbers land)

Fresh `evaluate_prompt_decomposition@0` / `evaluate_prompt_selection@0` rerun post
truncation-fix. Ledger: `EXPERIMENT_LOG.md` 2026-07-18; narrative: `FINDINGS.md`
"Localized Geometry" section. Impact on the revised paper of §7c:

- **The §7c thesis SURVIVES and gains a second leg.** The between-prompt/within-prompt
  dissociation replicates on clean Qwen data (ICC 0.94–0.97; within-prompt geometry ties
  but never beats entropy/logprob). NEW: the abstention application now has numbers —
  prompt-level abstention at 50% coverage: rmd_tail_q20 **0.836** vs length 0.740 /
  logprob 0.680 / entropy 0.672 (exploratory, needs CIs via selective-prediction stages;
  this is §7c-check-1 "MOST IMPORTANT un-run number" in trace form, and it passes).
- **NEW contribution-bearing positive: entropy-localized RMD.** rmd over highest-entropy
  20% tokens beats full-trace rmd at ALL THREE layers (+0.05 centered, p ≤ 0.006) with a
  matched random-token control failing — prespecified, entropy-specific, depth-monotone.
  Plus: label-free residualization shows the L21 signal is linearly complementary to
  entropy+logprob+length (residual within-macro 0.645 vs 0.654 raw). This upgrades the
  mechanism story: geometry at *uncertainty forks* is where the trace-level information
  lives, even though it only ties output baselines head-to-head.
- **Selection/reranking is now CLOSED with a structural explanation:** at N=8 only
  39/500 prompts tie and ~10 have headroom (~2-pt ceiling); all tie-break deltas ≤0.006,
  p ≥ 0.248; weighted-RMD voting underperforms majority. Do not spend more compute here.
- **Contrastive supervision is a NEGATIVE worth one sentence:** OOF cross-prompt
  directions are real (alignment 0.18–0.22 vs null 0.10 at L21) but add nothing over
  matched localized RMD — region choice, not supervision, is what matters.
- **Multiplicity discipline for the paper:** the only unadjusted-significant incremental
  probe cell (L21 macro +0.049, p=0.024) must be framed as suggestive; pre-specify the
  deepest layer and the two surviving contrasts (localization, entropy-specificity) for
  the cross-model confirmation on deepseek/llama/deepseek_llama full runs
  (deepseek_llama decomposition outputs currently deleted — regenerate first).

## 7e. 2026-07-31 UPDATE (supervised probe ceiling + length residualization, BOTH models)

Ran §6's flagged "killer experiment" — the trained PCA+LDA probe — on both models,
plus E1R (abstention with trace length partialled out in rank space). Narrative:
`FINDINGS.md` "Supervised Probe Ceiling and Length Residualization". Exploratory,
not pre-registered.

**§6 asked: does the trained probe ALSO collapse to length? The answer is the
inverse of the framing — the probe does not collapse, and neither does RMD, but the
output baselines do.**

- **The label-efficiency argument is now at its strongest form.** Once length is partialled
  out of both scorers, the supervised probe does **not** reliably beat the positive-only
  RMD fit on either model (probe − rmd_tail_q20: +0.029 Holm 0.090 Qwen, +0.033 Holm 0.126
  DeepSeek). The one Holm-surviving cell has the probe *losing* (Qwen he_q20, 0.018).
  **Qualified 2026-08-08 (§7f):** this is a full-budget, length-residualized *tie*, and
  reading it as a positive-only *win* over-claims twice — the probe here also differs in
  pooling order and in being linear, and at 400 labels it takes the lead on solo AUROC.
  This is the comparison §7b's "GATING REALITY" expected to lose ("honest expectation:
  RMD loses raw AUC to the supervised probe") — at prompt-level abstention, length-
  controlled, it does not. Note the scope: this is between-prompt abstention AURC, not
  raw trace-correctness AUC, where the ~0.84 supervised probe number still stands.
- **RMD survives length control on both models; entropy/logprob do not on DeepSeek.**
  Δ vs an uninformative scorer: rmd_tail_q20 +0.161 [+0.128, +0.194] Qwen /
  +0.107 [+0.077, +0.135] DeepSeek, both Holm < 0.01; entropy and logprob
  +0.009–0.011 on DeepSeek, Holm 1.000. This upgrades §7c's surviving positive
  "RMD > length + entropy" to the sharper claim: **RMD beats length on a component
  length cannot supply, and the output-side baselines are the ones that are length
  proxies on the reasoning-distilled model.** Negative control validates the removal
  (a pure-length scorer lands at +0.008 / −0.007, p ≥ 0.82).
- **Do not headline the raw Spearman-vs-length table.** RMD reaches rho +0.82 with
  length on DeepSeek L21 (vs +0.22 for the probe), which reads as a collapse and is
  not one. High rank correlation with length coexists with a large length-independent
  component because length itself explains only part of solvability. Report the
  correlation *and* E1R together, or the correlation invites a wrong reading — this
  document previously drew that wrong reading before E1R was run.
- **Raw (uncontrolled) probe-vs-RMD, for completeness:** probe_hidden_tail_q20 beats
  rmd_tail_q20 by +0.025 (Qwen, Holm 0.056, does not survive) and +0.048 (DeepSeek,
  Holm 0.006, survives). Reviewers will want both the raw and the length-controlled
  version; the gap between them *is* the finding.
- **One DeepSeek weakness to disclose:** `rmd_high_entropy_q20 − length` = +0.005
  [−0.011, +0.025] p=0.506 — the entropy-localized region does not clear length on
  that model. Only `rmd_tail_q20` does (+0.030 [+0.014, +0.048]). Consistent with
  §7d's failed localization gate: entropy-localization is Qwen-specific, tail
  localization is what replicates.
  **[Updated 2026-08-09]** Both halves of that last sentence now need qualifying.
  Entropy-localization stays Qwen-specific, but the §7d gate run on
  `deepseek_llama` L24 — the model its own layer column names — *passes* on both
  tests; it only stays Qwen-specific once you notice the localized score there is
  0.491, i.e. at chance, and that the gate never required either score to beat it.
  And "tail localization is what replicates" is now wrong as a statement about the
  aggregator: between prompts, the untailed `rmd_full` recovers the whole increment
  on both distilled models, and the tail is load-bearing only on Qwen. See
  `EXPERIMENT_LOG.md` (2026-08-09, both entries).
  **[Sharpened 2026-08-22, on `full_population`.]** Experiment 1b now says this
  with intervals, and the missing half changes the framing. `rmd_full` — which
  *is* Vazhentsev's ATRMD, published prior art — over `B0`: −0.0287 [−0.0534,
  −0.0055] DeepSeek-Qwen and −0.0445 [−0.0739, −0.0151] Llama, essentially the
  whole increment; but −0.0178 [−0.0435, **+0.0105**] on Qwen, where it does not
  clear zero at all. So it is not that the tail is redundant on two models — it
  is that each architecture needs a different region, and the two scores
  correlate at Pearson 0.93–0.96. **Write the increment as the claim and the
  localization as a scope note.** `rmd_tail_q20` is the region that works
  everywhere, not an established contribution. Notebook 14 §4a Table 5a.
- **Breadth is now the binding constraint, not rigor.** ~~n=2 models, both
  Qwen-lineage. Every claim in this section needs the Llama-architecture replication
  (`deepseek_llama`, cancelled by the §7d gate)~~ **[Stale — corrected
  2026-08-09.]** The `deepseek_llama` collect was restored for other reasons and
  finished 2026-08-03, so this section's claims now rest on **three models spanning
  two architecture families** — but two of the three are reasoning-distilled and
  two are Qwen-lineage, all are 7–8B, and all share one task and one prompt set.
  Breadth is still the binding constraint, and it is now **single-dataset scope**
  rather than model count. Note also that "a property of reasoning-distilled models"
  is exactly the axis the 2026-08-09 localization split falls along, and it rests on
  a single non-distilled model.

## 7f. 2026-08-08 UPDATE (the supervision/form ladder — attribution moved, regime bounded)

Ran the comparator ladder §6 now records as required. Ledger:
`EXPERIMENT_LOG.md` 2026-08-08 "Splitting the label-efficiency gap into supervision
and decision-function form". Three models, budgets 25/50/100, 30 label draws.

**§7e compared `rmd_tail_q20` against a probe that differed in three ways at once** —
positive-only vs labelled negative class, quadratic vs linear, score-then-pool vs
pool-then-score. `qmd_tail_q20` (RMD's own estimator with the unconditional background
swapped for a Gaussian over incorrect traces) makes the rungs separable.

- **The headline sentence changes.** At 50 labels the −0.033 AURC gap against a
  pooling-matched linear probe is roughly **−0.011 supervision + −0.018
  decision-function form**. "The positive-only inductive bias is what buys the label
  efficiency" must quote −0.011, not −0.033. The defensible version: *a positive-only
  one-class fit is a cheap way to obtain a quadratic decision function, and the
  quadratic is where most of the scarce-label advantage lives.*
- **Both rungs expire by 100 labels** (+0.000 each, 15/30 and 14/30). The quadratic is
  what a hyperplane cannot *express* — on a pure shape axis the correct class is 2.9x
  wider with no mean shift, and no linear boundary encodes "too wide" — but that stops
  costing the probe AURC once it can fit its own direction. **Do not write that geometry
  keeps an edge at scale.** The regime is the claim.
- **Pooling order runs opposite to the intuition.** Score-then-pool *costs* the LDA
  ≈ +0.016 at 50 labels, so §7e's region-mean probe was being helped by averaging first.
  The original comparison was conservative in the probe's favour, not inflated in
  geometry's — a point worth volunteering, since it is the first thing a reviewer
  suspects.
- **QMD is not a strawman.** `B0+qmd − B0` reaches −0.058 at 100 (28/30). The labelled
  quadratic is a strong feature that simply needs more labels than the positive-only one.
- **Mechanism, for the figure and the intuition section.** RMD's background Gaussian is a
  *mixture containing the positives*, not a rival class: its mean sits 0.44 from
  `mu_correct` in that class's own metric against the incorrect class's 0.90. So RMD
  already points along the contrast with the magnitude damped (score range 2.0 vs 4.7),
  and labelling the negative class undilutes rather than reveals. `RMD − QMD =
  d_incorrect − d_background` exactly — `d_correct` cancels — so the supervision rung has
  a closed form. Figure: `rmd_qmd_geometry.py`.
- **Scope caveat that blocks a direct splice into §7e.** These budgets cap at 100, leaving
  ~314–328 evaluation prompts against the frozen run's ~80. Only the paired within-run
  deltas transfer; AURC *levels* and crossing budgets do not.

## 7. Open questions for the SECOND research pass (a) — what research #1 did NOT see

Research #1 only saw a curated brief, not FINDINGS.md. The second pass must ground:
- **Distillation reshaping representation geometry** (our low-dim contrast, bimodal layers) —
  is there prior work on RL/distillation changing internal geometry? (research flagged ungrounded)
- **Cross-model manifold transfer / Platonic Representation Hypothesis** for correctness —
  prior work? (not evaluated in pass #1)
- **Selective prediction / abstention for reasoning** as the headline application — prior art,
  benchmarks, and what AUSC numbers are considered strong.
- **Truncation / generation-cap / non-termination** as an evaluation confound in o1/R1-style
  reasoning-model papers — first-to-document check.
