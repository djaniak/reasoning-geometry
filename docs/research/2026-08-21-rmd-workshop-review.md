# RMD workshop review

Date: 2026-08-21
Revised: 2026-08-22
Experiment repository: `raw/repos/reasoning-geometry` at `ca7c0e4ef01c95a88b02966c99622787ba1d3523` (2026-08-20)
Experiment log: 42 dated entries, 4,943 lines
Constraint used here: short GPU access; no full new three-model collection assumed

## Decision

The RMD work can support a workshop paper, but the current notebook is not yet a submission. The paper should be a controlled measurement study. Its main result is not a new score. It is the separation of pooled correctness prediction into prompt difficulty, within-prompt discrimination, and outcome construction.

Four items block submission:

1. define the budget-indexed outcomes and report the continuation study as a separate one-model sensitivity analysis;
2. replace the current peer-control framing with a baseline and cost comparison;
3. apply the between/within decomposition to a published-style supervised probe;
4. refit the full pipeline across outer prompt partitions.

A second prompt set would provide confirmation outside MATH-500, but it is optional for a workshop submission. Run it only after the four blockers are resolved and the analysis is frozen.

## Orientation and status of the previous review

The most recent RMD-specific review asked for an allocation gate, a supervised prompt-state ceiling, outer-partition refits, and a second dataset. Current status:

| Recommendation | Status on 2026-08-21 |
|---|---|
| Marginal-value allocation gate | Done. It failed. Geometry has Spearman \(-0.042/-0.057/-0.074\) with gain and the predeclared gate passes only a negligible \(R^2=0.0005\) case. |
| Supervised prompt-state ceiling | Not resolved by the current literature pass. Do not add it to the workshop unless the paper claims that information is created during generation. |
| Outer-partition and refit stability | Open. The bootstrap still resamples fixed OOF predictions and all models use one outer partition. |
| Second dataset | Wired but not run. The OlympiadBench probe and Best-of-8 collection remain queued. |
| Held-out-sibling forecasting | Closed with the allocation direction. The gate found no useful marginal-value signal. |

The current RMD output is the literature pass dated 2026-08-21. The older `RELATED_WORK.md` and `PAPER_STRATEGY_RMD.md` contain stale claims identified by that pass. There is no manuscript file. The paper-facing artifact is notebook 14.

### Literature-first record

A new pre-registration is impossible because the user prompt and the literature report expose the results. I reuse the field-level test recorded before the latest results: a contribution would require hidden-state geometry to add value over self-consistency, a discriminating experiment to identify what the score measures, or a policy benefit beyond output-side methods. Repackaging would include ATRMD as a new score, a tail mean as a new method, or geometry beating length without a consistency baseline.

The 2026-08-21 literature pass changes the second criterion. Prompt-difficulty prediction is already reported by Masoomi et al. The remaining contribution is the controlled attenuation under empirical difficulty and the between/within decomposition, not the qualitative statement that geometry tracks hard problems.

Two relevant papers are missing from the pass:

- [Who Thinks Best Depends on How Long You Let Them](https://arxiv.org/abs/2608.12150) varies token budgets across four models and three reasoning benchmarks and reports ranking reversals. This directly precedes the broad claim that generation configuration changes evaluation conclusions.
- [Beyond Accuracy: Decomposing the Reasoning Efficiency of LLMs](https://arxiv.org/abs/2602.09805) separates completion rate, correctness conditional on completion, and generated length. The conceptual separation between completion and correctness is therefore precedented. The continuation of capped traces remains the distinctive part of this project.

## A. Stage diagnosis

The core RMD thread is in late Understanding. The project knows the main effect, its strongest empirical control, and several failed downstream uses. It has not completed the stability and confirmation tests needed for Distillation.

The paper documents behave as if Distillation has started. Notebook 14 states one sentence of the paper and lists standing claims. Several are stale: it calls the score a single-forward-pass signal although the target and baseline use eight completed siblings; it treats peer pass rates only as a non-deployable control; and it states a three-model DeepConf result although the exact control is absent on Qwen and does not clear the increment on DeepSeek-Qwen in the incremental comparison.

The budget thread has moved backwards from Distillation to Understanding. The continuation run established that many capped traces terminate under a larger budget, but the paper has not defined the outcome at each budget.

## B. Truth-seeking audit

### 1. The paper does not define its budget-indexed outcomes

Let \(C_B\) denote whether a correct, assessable answer is available by token budget \(B\), under a fixed decoding and extraction configuration. No answer by \(B\) is an observed failure under this protocol. Let \(C_{B\rightarrow B'}\) denote correctness after continuing the stored prefix from \(B\) to a larger budget \(B'\), under an explicit continuation configuration. These quantities answer different questions. Neither is a unique label of eventual correctness: an answer can change as generation continues.

The current notebook calls all capped rows censored and treats their exclusion as missing-data handling. This is not valid for \(C_B\). Complete-case filtering estimates correctness conditional on avoiding a cap, not \(C_{B\rightarrow B'}\), because capping is related to prompt difficulty and correctness.

The distinction matters empirically. Removing prompts with a capped sibling changes the population from 500 prompts to 392/393/408 and raises base accuracy from 0.620/0.750/0.634 to 0.691/0.796/0.674. Capping is prompt-structured and concentrated near the budget edge. Complete-case filtering therefore conditions on difficulty.

The continuation result is useful but narrower than the current prose. It samples 50 traces from 370 non-looping capped DeepSeek traces; 70% terminate and 45.7% of terminated traces are correct. It does not recreate the original random stream and covers one model. Also, 38 of 374 capped traces already contain a parseable answer at the cap. Their stopping time remains censored, but their answer correctness at \(B\) is observable if the extraction rule scores the available answer.

Required correction:

- Report \(C_B\) on the full population.
- Report \(C_{B\rightarrow B'}\) only for the sampled DeepSeek continuation study. Treat it as a case study, not as the label for the full dataset or other models.
- Report cap-free results as conditional results. Do not use them as estimates of \(C_{B\rightarrow B'}\).

A common 1024-token recomputation would estimate a valid \(C_{1024}\), but it would not separate reasoning correctness from termination. The existing 2048-token audit already shows the failure mode: 45.4% of DeepSeek traces are unparsed, 99.4% of them end at the cap, and the within-prompt RMD result falls from about 0.93 to 0.27 on parseable traces; mixed prompts fall from 166 to 13. Do not spend another analysis on the 1024-token version.

### 2. "Absorption" is descriptive attenuation, not a mechanism estimate

The peer control reduces the AURC increment from \(-0.0585/-0.0355/-0.0560\) to \(-0.0108/-0.0004/-0.0125\). This establishes shared predictive information with empirical peer-model difficulty. It does not identify what the hidden state encodes. A fitted readout with correlated predictors is not a variance decomposition or a causal mediation analysis.

The `99%` value should not appear as evidence of redundancy. On DeepSeek-Qwen, `B0+peer` is only 0.0045 AURC above the oracle floor. The experiment has almost no headroom there.

Use "attenuation under peer control" in the paper. Report the two paired deltas as primary results. If a percentage is retained, bootstrap the ratio and mark it descriptive. The safe interpretation is: roughly four fifths of the original increment is shared with empirical peer-model difficulty on Qwen and Llama; the DeepSeek-Qwen attenuation cannot distinguish overlap from saturation.

### 3. The peer control is also a competing method

The internal log says `B0+peer` is never a baseline. Hamidieh et al. use a scale-matched model ensemble as a deployable uncertainty method, so a reviewer can treat the same resource as a baseline. The current data already show peer features beat `B1` on Qwen and DeepSeek-Qwen and tie it on Llama.

The comparison is not compute matched. The peer feature consumes 16 additional generations from two models. RMD uses hidden states from the target model's eight generations. The paper should show accuracy/risk against model calls or generated tokens. At minimum, report one peer, two peers, one sample per peer, eight samples per peer, `B1`, and `B0+peer`.

### 4. The published-style probe decomposition is still missing

The current within-prompt analysis concerns existing RMD and output scores. It does not yet show that a high pooled AUROC from the probe literature collapses after conditioning on prompt identity. The literature report correctly identifies this as the analysis that changes the paper from a self-audit into a correction of a published claim shape.

The new probe must reproduce the comparison before decomposing it: last-token hidden state, prompt-disjoint train/test folds, layer selection inside training data, and the same trace population. Report pooled trace AUROC, micro within-prompt pair AUROC, and macro per-prompt AUROC. State how prompts without both outcomes enter each metric. A tail-pooled LDA already in the repository is not a reproduction of the last-token probe used by Yuan et al.

### 5. Current intervals omit the uncertainty that matters

The paired bootstrap holds PCA, Gaussian references, feature selection, and readouts fixed, and all models share one outer prompt partition. This omission matters for residual AURC changes near \(-0.01\). The outer refit must repeat the whole fitting path, including layer selection if the paper treats the selected layer as data-dependent.

Repeated analysis on the same 500 MATH prompts also makes the local p-values exploratory. Outer resplits measure fit stability; they do not create an untouched confirmation set. The queued OlympiadBench collection can provide confirmation, but it is not required before a workshop draft.

### 6. Reproducibility is not submission-ready

The result JSONs and OOF tables are gitignored and absent from this checkout; there is no DVC remote. Notebook 14 reads committed artifacts from another machine and has no manuscript export. Before submission, archive the exact result tables, fold assignments, population IDs, model revisions, prompts, decoding configuration, extraction rule, and commands needed to rebuild every paper exhibit.

## C. Prioritisation

The project spent enough time on new score variants. The useful work now is evaluation closure. More tail windows, geometry families, layers, and label budgets would add rows without changing the paper.

The workshop north star should be:

> Determine which reliability conclusion follows under each stated evaluation unit, token-budget outcome, and difficulty control, then test whether that conclusion survives a published-style probe.

This also fits the PhD better than a transfer claim. It gives Paper 5 a direct role in the dissertation thesis about representation signals under model, generation, outcome, and evaluation choices.

## D. Literature gaps

The 2026-08-21 pass correctly removes novelty from ATRMD, tail aggregation, generic prompt-difficulty prediction, label efficiency, reranking, and adaptive allocation. It also identifies the closest threats: Vazhentsev et al., Yuan et al., Hamidieh et al., Masoomi et al., and Gu et al.

Three corrections remain:

1. Add the two budget/completion papers named above. They narrow the budget-related contribution to the continuation experiment and its interaction with geometry.
2. Rewrite `RELATED_WORK.md`; it still claims that probe papers lack an output-side baseline and that the incremental design is unique. Yuan et al. and Hamidieh et al. invalidate the broad form of both claims.
3. Do not claim the first between/within decomposition in general. Claim only that the authors did not find this decomposition applied to hidden-state correctness probes over sibling reasoning traces.

## E. Next experiments, ranked

### 1. Define the outcomes and run a sensitivity table

**Stage:** Understanding.
**Cost:** CPU, less than one day on existing outputs.

**Question:** Does the paper's conclusion hold for \(C_B\), for correctness conditional on avoiding the cap, and in the sampled \(C_{B\rightarrow B'}\) continuation case?

Report \(C_B\) on the full population and identify cap-free results as conditional. Score parseable answers present at the cap separately from the stopping event. Report \(C_{B\rightarrow B'}\) only for the 50 sampled DeepSeek traces. Do not extrapolate it to all capped traces or other models. A changed ordering supports a protocol-dependence result.

### 2. Peer baseline and cost ladder

**Stage:** Distillation.
**Cost:** CPU analysis on existing generations.

**Question:** How does target-model RMD compare with one or two peer models at matched model-call or token cost?

Report one peer, two peers, one sample per peer, eight samples per peer, `B1`, and `B0+peer` where the stored outputs permit these comparisons. If a cheap peer wins, report that result directly. If RMD wins at a matched cost, report the cost definition and the observed margin. The current unmatched comparison does not establish practical superiority for either method.

### 3. Published-style probe with between/within decomposition

**Stage:** Understanding.
**Cost:** CPU hours; cached hidden states.

**Question:** Does a probe that reproduces the high pooled trace AUROC retain discrimination after conditioning on prompt identity?

A collapse after reproducing the pooled result supports the measurement correction. If the probe reproduces the pooled result and retains within-prompt performance, limit the claim to RMD. If it does not reproduce the pooled result, do not claim that this analysis corrects the probe literature.

### 4. Outer-partition and full-refit stability

**Stage:** Understanding.
**Cost:** 5–15 CPU hours.

**Question:** Do `B1-B0`, the peer-controlled residual, and the between/within result survive refitting the complete pipeline?

If the residual changes sign or varies widely, keep the original increment and demote the residual. More fixed-prediction bootstrap draws cannot replace this test.

### 5. Freeze the analysis, write the manuscript, and archive its evidence

**Stage:** Distillation.
**Cost:** writing and artifact packaging.

Use four main sections: budget-indexed outcomes and the continuation case study; pooled versus within-prompt decomposition; RMD, peer baselines, and attenuation; full-refit stability and limitations. Archive the exact tables and population IDs before drafting claims from them. Put closed downstream directions in the appendix or remove them.

### 6. Optional OlympiadBench confirmation

**Stage:** Confirmation.
**Cost:** short GPU run: the queued 64-problem OlympiadBench gate, then 200–250 prompts at \(N=8\) only if the base rate is usable.

Run this only if time and GPU access remain after the workshop analysis is frozen. Test the original increment and the between/within direction. Peer attenuation is optional unless matching peer outputs are already available. A workshop submission can proceed without this collection if its scope remains a controlled MATH-500 measurement study.

## F. What to stop

- Stop `m\cdot|s|` and `\varphi_\beta` transfer experiments for the workshop. They test a different claim and Paper 5 does not need them.
- Stop tail-localisation, entropy-region, layer, covariance, and label-budget claims and sweeps.
- Stop comparing effect magnitudes as properties of the model families. The evaluated configurations use different token budgets, and the existing 2048-token audit shows that termination dominates the apparent DeepSeek within-prompt result. Report the three model configurations separately. It is safe to state only that `B1-B0` has the same direction in all three.
- Stop the distillation and cross-model mechanism claims. The current data do not separate model lineage from generation budget and termination.
- Stop calling the method single-forward-pass, single-trace verification, routing, OOD detection, or adaptive allocation.
- Stop calling peer pass rates only a diagnostic control.
- Stop using `78–99% absorbed` as the main quantitative claim. Report paired AURC changes and the DeepSeek-Qwen headroom.
- Stop adding controls to MATH-500 after the frozen closure set. Use new compute for confirmation.
