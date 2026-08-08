# Paper Strategy — Hidden-State Geometry for Reasoning Reliability

*Durable working doc (2026-06-19). Survives context compaction. Seeds the second
literature-grounding pass. Sources: `EXPERIMENT_LOG.md` (2026-06-14 audit),
`FINDINGS.md` (now carries a correction banner), deep-research run `wf_ea6e6b93-88e`.*

---

## 1. Revised thesis (defensible)

> **Relative hidden-state geometry (RMD) is a selective-prediction *between-prompt solvability*
> signal that beats both trace-length and entropy baselines — and beats them on a
> component length cannot supply, which a supervised probe on the same activations does
> not reliably improve on. It is useful for abstention / compute allocation / routing —
> NOT for within-prompt (per-attempt) reranking (Best-of-N is weak). Prior
> "trace-correctness" readings of such geometry were confounded by trace length and by a
> truncation/auto-label-as-incorrect artifact.**

*Second clause added 2026-07-31 from the length-residualized (E1R) + supervised-probe
runs; see §7e. Scope: between-prompt abstention on two Qwen-lineage models.*

Novelty rests on the **length/truncation rigor critique + the between-prompt application**,
NOT on the RMD primitive (already precedented — see §4).

## 2. Verified, de-confounded findings (what we can claim)

The 2026-07-18 Qwen localized rerun is the current load-bearing result. Older
cross-model, temperature, low-dimensional, and selective-prediction claims below
are historical until clean-budget replications are collected.

- **Length is a deceptively strong correctness baseline** (pooled AUC ~0.74 Qwen / ~0.83
  DeepSeek) that prior geometry/UQ work never benchmarked against as a standalone predictor.
- **On clean Qwen Best-of-8 traces, entropy-localized RMD beats full-trace RMD and a
  matched random-token control at all three layers.** It remains competitive with
  output baselines and has only suggestive incremental probe value.
- **The signal is between-prompt (solvability), not within-prompt.** Pooled parseable RMD
  is positive; within-prompt (per-attempt) is at chance (Qwen 0.515, DeepSeek 0.27 at n=13).
- **Termination is mostly recoverable from length** (Qwen length detects unparsed at 0.996 >
  RMD 0.84) — do NOT headline "RMD rejects non-terminating traces."
- **The old low-dimensional distillation and transfer claims are rejected or weakened by
  parseable-only audits.** Do not use them as current paper pillars.
- **Selection is closed:** geometry does not improve Best-of-8 tie-breaking; the useful
  downstream direction is prompt-level abstention/compute allocation.

## 3. Contamination ledger (CRITICAL — re-validate before using)

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

**Amended 2026-08-03.** The mechanism line above is right about the *label* and wrong
about the *traces*. Capped traces resumed to 16,384 tokens terminate 70% [0.56, 0.81]
of the time and are correct 46% of those times, versus 5.6% as scored. Capping is a
budget shortfall, so the contaminated rows are censored observations rather than
failures. Two consequences for the paper: (i) exclusion is defensible as missing-data
handling, which is a stronger footing than "drop the degenerate traces"; (ii) do not
write that geometry detects non-termination — it detects *unfinished*, and the
distinction is checkable. Reviewers of a truncation-heavy table will ask; the
continuation study is the answer. Ledger: `EXPERIMENT_LOG.md` (2026-08-03).

## 4. Literature map (deep-research wf_ea6e6b93-88e, 21/25 claims confirmed)

- **RMD primitive is NOT novel.** NAACL 2025 [arXiv:2502.14427] (SAT(R)MD) already does
  token-level RMD vs a C4 background. *Most dangerous overlap — read first.* Differentiate on:
  trace-level single score, unsupervised, length-controlled, reasoning/BoN application.
- **Trained probes are strong baselines reviewers will demand we beat:** PCA+LDA ~80% on MATH
  [SWIFT, arXiv:2505.12225]; Semantic Entropy Probes [arXiv:2406.15927]; TrajSelector
  [arXiv:2510.16449]. An unsupervised RMD likely loses to these at raw correctness.
- **Within-prompt geometry is a graveyard:** INSIDE/EigenScore [arXiv:2402.03744] is
  within-prompt self-consistency, scores *below random* on GSM8K in ACL-2025 survey
  [2025.findings-acl.1101]. → between-prompt difficulty is the white space.
- **Length-as-baseline gap is real** but state as "under-controlled in prior work," not
  "never studied" (2-1 vote). Semantic entropy [Nature 2024] only catches confabulations,
  misses systematic reasoning errors → motivates a single-trace representation signal.
- **Refuted/over-claimed:** SE-as-universal-SOTA, latent-reranking-beats-majority,
  GenRM-as-SOTA, SWIFT-beats-heavy-RMs — reduces competitive pressure, consistent with our
  at-chance within-prompt finding.

## 5. Three candidate framings (ranked)

1. **Measurement-and-reframe paper (recommended):** "What does hidden-state geometry
   actually measure in reasoning models? A length-controlled re-examination." Length
   confound + truncation artifact + between/within decomposition + between-prompt
   solvability application beating length & a trained probe. Every claim we can support;
   avoids the fight (beating trained probes at raw correctness) we'd lose.
2. **Between-prompt difficulty for test-time compute allocation / routing** (constructive).
3. **Single-trace label-light RMD beating length+entropy** (riskiest — primitive precedented).

## 6. Baselines / experiments a top venue will demand

- ~~Trained linear probe (PCA+LDA, SEP-style) — *the* required baseline; does it ALSO collapse
  to length? (the killer experiment).~~ **DONE 2026-07-31, both models — see §7e.** The probe
  does not collapse (rho ~0.22 DeepSeek); neither does RMD; entropy/logprob do. Length-
  controlled, the probe does not reliably beat RMD on either model. RMD-vs-EigenScore
  head-to-head on reasoning is still open.
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
   AbstentionBench. Novelty ONLY via **unsupervised + single-forward-pass + beats
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
> **C3 (application):** unsupervised, single-forward-pass RMD for selective prediction on
>   reasoning — beats length/entropy/semantic-entropy at zero supervision (supervised probes do
>   better but need labels); motivated by reasoning-FT degrading abstention.
> **Rigor thread:** length is the baseline to beat; truncation/auto-label-incorrect confound.
> **Positioning:** RMD is trace-level (vs SAT(R)MD token-level), unsupervised (vs PCA+LDA/SEP/
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
  *unsupervised + single-pass + cross-model + mechanism*, not on raw accuracy.

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
3. Pass-rate/difficulty correlation parseable — the compute-allocation hook.

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

- **The label-light argument is now at its strongest form.** Once length is partialled
  out of both scorers, the supervised probe does **not** reliably beat unsupervised RMD
  on either model (probe − rmd_tail_q20: +0.029 Holm 0.090 Qwen, +0.033 Holm 0.126
  DeepSeek). The one Holm-surviving cell has the probe *losing* (Qwen he_q20, 0.018).
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
- **Breadth is now the binding constraint, not rigor.** n=2 models, both Qwen-lineage.
  Every claim in this section needs the Llama-architecture replication
  (`deepseek_llama`, cancelled by the §7d gate) before it can be stated as a property
  of reasoning-distilled models rather than of two checkpoints.

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
