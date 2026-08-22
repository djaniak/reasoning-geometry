# RMD literature novelty check

Date: 2026-08-22

Scope: `raw/repos/reasoning-geometry`, especially `RELATED_WORK.md`, `PAPER_STRATEGY_RMD.md`, `EXPERIMENT_LOG.md`, notebook 14, and `docs/research/2026-08-21-rmd-workshop-review.md`. Literature claims below were checked against the linked primary paper, proceedings page, or official preprint.

## Verdict

RMD has a possible workshop paper, but only as a controlled measurement study. It is not a new geometry statistic, a new tail aggregator, or evidence that hidden states generally discover prompt difficulty. The current publishable candidate is narrower:

> Under a stated eight-sample, fixed-budget protocol, a published token-level RMD score adds prompt-level selective-prediction value over a target-model self-consistency readout; that increment attenuates under an empirical peer-difficulty baseline, while the score does not verify which sibling trace is correct.

This needs the validation work below before it is a paper claim. The existing `notebooks/14_rmd_paper_story.ipynb` is the right paper-facing artifact. Do not start a second RMD story notebook before the analysis is frozen; revise notebook 14 after the closure runs.

The recent [TrAC preprint](https://arxiv.org/abs/2608.00422) removes any broad claim that a trace-derived signal newly improves eight-sample self-consistency. Its Consensus Fusion adds a short trace-conditioned answer re-elicitation and token-uncertainty profile to `SC@8` on MATH500 and other math benchmarks. RMD must be framed as a hidden-state density feature under a different target and cost profile, not as the current best way to augment self-consistency.

## Claim ledger

| Candidate claim | Novelty verdict | Evidence and safe boundary |
|---|---|---|
| Relative Mahalanobis distance is the method contribution | Published | [Ren et al.](https://arxiv.org/abs/2106.09022) introduce RMD. [Vazhentsev et al.](https://aclanthology.org/2025.naacl-long.113/) apply token-level RMD to generated text, fit a correct-response token reference plus an all-token background, average token scores into ATRMD, and train a readout over it. Call the score ATRMD/RMD, not a new manifold method. |
| `rmd_tail_q20` is a new aggregation | Not publishable as a method | The repository correctly identifies it as a mean over the final 20% of tokens, not a distance quantile. Vazhentsev et al. already own the trace mean; [DeepConf](https://arxiv.org/abs/2508.15260) precedes tail-focused confidence aggregation. The tail effect can appear only as a one-model localization observation. |
| A trace-derived signal improves eight-sample self-consistency | Published, and close | [TrAC](https://arxiv.org/html/2608.00422v2) fuses an active re-elicited answer with passive token-probability/entropy features and `SC@8` consensus. It reports an AURC reduction and AUROC gain at the same eight full generations plus one short cached answer decode, including on MATH500. It does not use hidden states or RMD, but it occupies the main practical claim shape. |
| Multi-trace path scoring supports prompt-level abstention | Published | [Pause and Reflect](https://arxiv.org/html/2605.14098v1) combines multiple CoTs, path scores, weighted answer aggregation, and conformal abstention. It does not use hidden-state density, but it blocks a broad claim of a new prompt-level path-scoring abstention system. |
| Hidden states add value beyond cheap surface features | Precedented | [Kirin](https://arxiv.org/abs/2607.18553) reports a hidden-state-plus-shortcuts AUROC gain over length and log-probability on GSM8K. [Yuan et al.](https://arxiv.org/abs/2605.09502) report a large hidden-state versus text-surface gap. Do not claim the first incremental hidden-state reliability result. |
| Geometry predicts task difficulty | Published | [Masoomi et al.](https://arxiv.org/abs/2607.01571) report 0.93 AUC for hidden-trajectory effective dimension on easy versus hard MATH-500 items. The paper cannot headline "geometry measures hard problems." |
| Adding a feature to self-consistency is novel | Precedented design; narrow feature-level gap remains | TrAC and [Hamidieh et al.](https://arxiv.org/abs/2604.17112) both augment self-consistency and report selective-prediction gains. Vazhentsev et al. combine token-density features with sequence probability. The defensible gap is narrower: this review did not find a paper that measures a positive-only hidden-state density score's increment over a fitted answer-agreement readout at the *prompt* level. Treat that as a search finding, not a priority claim. |
| The signal is between-prompt, not within-prompt | Measurement candidate, not a first | [Yuan et al.](https://arxiv.org/html/2605.09502) compare a last-token hidden-state probe against `N=5` self-consistency and test correct versus wrong traces on the same mixed-outcome problems. [Masoomi et al.](https://arxiv.org/html/2607.01571) contrast question-level held-out evaluation with a knowingly question-leaky split. The repository can add a stricter, like-for-like audit of RMD and a published-style probe, but cannot claim the first within/between or sibling analysis. |
| Peer pass rates expose what RMD measures | Potentially publishable as a controlled observation | On the cap-free population, the recorded AURC increment changes from \(-0.0585/-0.0355/-0.0560\) to \(-0.0108/-0.0004/-0.0125\) after adding two peer models' pass rates (`EXPERIMENT_LOG.md`, 2026-08-10). Say "attenuation under peer control," not "78–99% mechanism explained." The DeepSeek-Qwen arm has only 0.0045 AURC headroom above an oracle, so its 99% attenuation cannot separate saturation from redundancy. |
| Peer pass rates are only a diagnostic control | False | Hamidieh et al. make a small scale-matched ensemble a deployable UQ method. Therefore `B0+peer` is a competing baseline. It must receive a cost ladder, not a footnote saying it is unavailable at decision time. |
| Cap-free analysis establishes eventual correctness | False as written | [Kaiser et al.](https://arxiv.org/html/2602.09805v2) already separate completion, correctness conditional on completion, and length; [Guedes de Souza and Panisson](https://arxiv.org/abs/2608.12150) show that token budgets change reasoning evaluations and model rankings. In this repository, a cap-free population conditions on avoiding a cap; it does not estimate correctness after a longer continuation. The continuation result is a one-model sensitivity case, not a replacement label. |

## What remains worth publishing

The paper's candidate contribution is an evaluation correction with a positive empirical result, not a new score:

1. Report the incremental result only for an explicit deployment outcome: target-model prompt-level abstention after eight traces at a stated token budget. It is a feature-specific result, not a claim that no trace-derived signal improves consensus.
2. Show that the apparent gain changes when the analysis distinguishes prompt difficulty, sibling-level correctness, and budget-conditioned answer availability.
3. Reproduce a published-style hidden-state correctness probe, then report pooled trace AUROC beside micro and macro within-prompt AUROC. This reconciles existing probe results under one explicit unit of analysis; it is not the first such distinction.
4. Compare RMD against exact-answer agreement, vote agreement, and peer-model disagreement at matched model-call or generated-token cost. A result where an output-side method wins is still publishable if the paper reports it directly and the measurement contribution survives.

Do not claim novelty for: ATRMD/RMD itself, the tail window, generic difficulty prediction, single-trace verification, adaptive sample allocation, Best-of-N reranking, broad label efficiency, cross-model/distillation mechanism, or causal use of the score.

## Submission blockers and cheapest order

1. **Define the outcome table on existing outputs.** Report correctness available by budget \(C_B\) on all 500 prompts, correctness conditional on avoiding a cap, and the sampled DeepSeek continuation outcome \(C_{B\rightarrow B'}\) separately. Score already-parseable capped answers at \(B\). The current cap-free headline is conditional on a difficulty-related event.
2. **Align against TrAC before making a utility claim.** TrAC targets individual response correctness and needs a short cached re-elicitation; RMD targets plurality-vote correctness and needs white-box hidden states. State this unit/cost difference. If compute allows, add TrAC's PCE/consensus-fusion baseline to the same prompt-level target. If not, do not make a best-system or broad self-consistency-augmentation claim.
3. **Run the peer baseline and cost ladder.** Compare one and two peers, one and eight samples per peer where cached data permit it, against `B1`. Report calls/tokens as well as AURC.
4. **Run the published-style probe decomposition.** Use last-token states, prompt-disjoint folds, train-only layer selection, and the same trace population. Report pooled, micro within-prompt, and macro within-prompt AUROC, with the rule for single-label prompts stated.
5. **Repeat the full pipeline over outer prompt partitions.** Refit PCA, reference Gaussians, feature/layer selection, and the readout. Fixed-OOF bootstrap intervals do not test this uncertainty.
6. **Freeze evidence before prose.** Archive result tables, fold IDs, population IDs, model revisions, prompts, decoding/extraction configuration, and exact commands. The current notebook reads artifacts absent from this checkout.

Only after these steps, use the queued OlympiadBench collection as confirmation. More geometry variants, layer/tail sweeps, or allocation experiments would not improve the claim.

## Writing boundary for a workshop draft

Use language like this only if the closure work supports it:

> On MATH-500 under a fixed eight-sample protocol, ATRMD adds prompt-level selective-prediction value over a target-model self-consistency readout. A peer-model difficulty baseline attenuates much of that increment, and a matched probe audit reports both cross-prompt and within-prompt discrimination. These conclusions depend on the stated budget, outcome protocol, and the availability of white-box hidden states.

This says what the data can establish. It does not turn attenuation into a representation mechanism or conditional cap-free results into eventual correctness.

## Primary sources checked

- [Ren et al., RMD](https://arxiv.org/abs/2106.09022)
- [Vazhentsev et al., token-level RMD / ATRMD, NAACL 2025](https://aclanthology.org/2025.naacl-long.113/)
- [Wang et al., self-consistency, ICLR 2023](https://openreview.net/forum?id=1PL1NIMMrw)
- [DeepConf](https://arxiv.org/abs/2508.15260)
- [TrAC, trace-conditioned answer consistency](https://arxiv.org/abs/2608.00422)
- [Gu et al., Pause and Reflect](https://arxiv.org/abs/2605.14098)
- [Zhang et al., hidden-state self-verification](https://arxiv.org/abs/2504.05419)
- [Yuan et al., hidden error awareness](https://arxiv.org/abs/2605.09502)
- [Hamidieh et al., cross-model disagreement plus self-consistency](https://arxiv.org/abs/2604.17112)
- [Masoomi et al., reasoning geometry and task hardness](https://arxiv.org/abs/2607.01571)
- [Kirin, hidden-state increment over shortcut features](https://arxiv.org/abs/2607.18553)
- [Guedes de Souza and Panisson, budget-dependent rankings](https://arxiv.org/abs/2608.12150)
- [Kaiser et al., Beyond Accuracy](https://arxiv.org/abs/2602.09805)
