# RMD and DAG patching: workshop decision and restart guide

Date: 2026-08-22  
Raw repository reviewed: `raw/repos/reasoning-geometry` at `139fd1a9b7d8a7078f4164becc5878e14547aa71`  
Scope: decide which thread is ready for a workshop paper, identify RMD closure work, and decide whether a new notebook is needed.

## Decision

Write the DAG workshop paper now. It has a narrow claim, committed evidence, and a dedicated workshop storyboard. Do not spend the DAG workshop sprint on another experiment.

RMD can become a workshop paper, but it needs a short analysis-closure pass first. Its contribution is a controlled measurement study, not a new geometry score. The useful result is whether a hidden-state-density score has a prompt-level increment under one stated protocol after strong output and peer baselines, and how that conclusion changes across sibling-level and token-budget outcome units. [PAPER_STRATEGY_RMD.md:17-39] [docs/research/2026-08-21-rmd-workshop-review.md:9-20]

The two papers should remain separate. They use different models, data, metrics, and claims. [PAPER_STRATEGY_DAG.md:5-10]

No current deadline or compute budget was supplied. The RMD ordering below assumes a workshop scope, existing cached outputs, and no new multi-model collection.

| Thread | Current stage | Workshop decision | Smallest next move |
|---|---|---|---|
| DAG patching | Distillation | Draft now | Turn notebook 16 into manuscript figures, tables, and prose. |
| RMD | Late Understanding | Close four analysis blockers, then draft | Start with a budget-outcome table and an artifact archive. |

## A. Stage diagnosis

### DAG patching

The workshop claim is already supported by two complementary controls. In the registered matched comparison, a one-step site produced the implied digit in 24/24 pairs and a two-step site in 0/24; the exact paired test is \(2^{-24}=5.96\times10^{-8}\). [PAPER_STRATEGY_DAG.md:77-92] The larger campaign then shows one-step success in 555/603 eligible sites and zero implied-digit wins in 432 multi-step sites, including same-trace positive controls that make the multi-step null interpretable. [PAPER_STRATEGY_DAG.md:68-75] [EXPERIMENT_LOG.md:57-76]

This is a real workshop-sized result because the evidence tests the important alternatives: clean/implied/raw outcomes separate transformation, copying, and clean preservation; token-distance bands overlap; and the same trace contains a working one-step intervention beside a failed multi-step ancestor intervention. [PAPER_STRATEGY_DAG.md:42-48] [docs/research/2026-08-16-dag-literature-and-claim-boundary.md:26-39]

The stage is Distillation. The project document explicitly calls the workshop paper submittable and says the missing work is writing rather than evidence. [PAPER_STRATEGY_DAG.md:305-330]

### RMD

RMD is in late Understanding. It has a three-model prompt-level AURC increment over the fitted target-model baseline, but the paper-facing notebook still presents conclusions that need correction. The documented submission blockers are: define budget-indexed outcomes, make peer models a costed baseline, run a published-style probe decomposition, and refit the full pipeline across outer prompt partitions. [docs/research/2026-08-21-rmd-workshop-review.md:9-20] [docs/research/2026-08-21-rmd-workshop-review.md:47-53]

The result is not weak; its current interpretation is incomplete. The original AURC increments are \(-0.0585/-0.0355/-0.0560\) across the three models, but after adding peer pass rates they become \(-0.0108/-0.0004/-0.0125\). The peer control has almost no remaining headroom for DeepSeek-Qwen, so its near-zero residual cannot distinguish redundancy from saturation. [EXPERIMENT_LOG.md:1845-1894]

## B. Truth-seeking audit

### What DAG can claim

Use this scoped thesis: in one pretrained reasoning-tuned 1.5B model and one fully written synthetic arithmetic format, a native-position residual transplant often controls the next arithmetic update, while exact semantic control is absent after two or more written operations; distributional effects remain. [PAPER_STRATEGY_DAG.md:52-66] [docs/research/2026-08-16-dag-literature-and-claim-boundary.md:10-24]

Do not call the step contrast a pure graph-depth causal effect. Adding a step also changes an operation, a written intermediate, a variable binding, and local context. The valid wording is that the contrast persists after matching clean confidence and ancestor token distance. [EXPERIMENT_LOG.md:134-148] Do not treat 1,035 sites as independent trials; report item- or seed-level summaries. [docs/research/2026-08-16-dag-literature-and-claim-boundary.md:37-39]

The clearest novelty is the arithmetic-specific, per-item clean/implied/raw measurement and the observed immediate-read boundary. It is not the first causal scratchpad-state intervention or the first computation-versus-copy test: Shih, Winnicki, and Darve already provide close causal controls. [docs/research/2026-08-16-dag-literature-and-claim-boundary.md:56-107]

### What RMD can claim after closure

RMD/ATRMD and the tail aggregation are not new methods. The repository's literature record assigns token-level RMD and trace aggregation to prior work, and shows the tail restriction is load-bearing only on Qwen. [PAPER_STRATEGY_RMD.md:28-33] [RELATED_WORK.md:18-51]

The safe candidate contribution is narrower: under a fixed eight-sample, stated-budget MATH-500 protocol, report whether ATRMD adds prompt-level selective-prediction value beyond a fitted self-consistency readout, how that increment attenuates under peer-model difficulty, and whether a high pooled hidden-state correctness score remains useful within a sibling set. This is an evaluation result, not a representation mechanism or the first pooled-versus-within-prompt comparison. [docs/research/2026-08-21-rmd-workshop-review.md:75-99] [outputs/reviews/2026-08-22-rmd-literature-novelty-check.md:28-36]

The current cap-free headline is conditional on avoiding a cap and cannot stand for eventual correctness. Capping changes the analysis population from 500 prompts to 392/393/408 and is related to prompt difficulty. The sampled continuation study is valuable but only covers 50 DeepSeek traces, one model, and a different continuation configuration. [docs/research/2026-08-21-rmd-workshop-review.md:57-73] [EXPERIMENT_LOG.md:3770-3829]

The RMD paper also needs a current competitor check. The recent TrAC preprint adds a response-correctness score to self-consistency and reports improvements with either one complete trace plus a short answer probe or an existing eight-sample pool. It does not test RMD or the peer-control/decomposition question, but it rules out a broad claim that this project is first to augment self-consistency at eight samples. [TrAC, arXiv:2608.00422](https://arxiv.org/abs/2608.00422)

## C. Prioritisation

For the DAG workshop, do not run another experiment. Preserve the narrow scope, show the pre-registered pair result first, then the same-trace control and semantic-versus-distributional contrast, and give the single-model/single-format limits visible space. [PAPER_STRATEGY_DAG.md:305-330]

For RMD, stop exploring score variants, layer sweeps, tail windows, label budgets, reranking, and adaptive allocation. The allocation test ran and failed: single-trace geometry ranks marginal value of another sample backwards, so no allocation policy was written. [PAPER_STRATEGY_RMD.md:41-52] The project review also identifies more geometry variants as low-information work. [docs/research/2026-08-21-rmd-workshop-review.md:105-113]

Do not make a second dataset the first RMD task. OlympiadBench is useful confirmation only after the four closure analyses are frozen; a narrowly scoped MATH-500 workshop paper can proceed without it. [docs/research/2026-08-21-rmd-workshop-review.md:154-175]

The post-training reasoning-prefix note is a third, optional direction. Keep it out of both workshop sprints unless you explicitly choose to open a separate project.

## D. Literature gaps and claim boundaries

The RMD draft must cite and compare against token-level RMD, supervised hidden-state correctness probes, self-consistency and cross-model uncertainty, task-difficulty geometry, conformal path aggregation, and recent response-level UQ. The present literature record removes novelty from the RMD primitive, generic prompt-difficulty prediction, the between-/within-question contrast itself, label efficiency, reranking, and allocation. The possible contribution is a cleaner, feature-specific test with exact-answer agreement, peer-cost controls, explicit outcome units, and formal pooled/micro/macro prompt-conditional reporting. [outputs/reviews/2026-08-22-rmd-literature-novelty-check.md:11-54] [docs/research/2026-08-21-rmd-workshop-review.md:115-123]

For DAG, cite the closest state-intervention work and state the difference as an empirical measurement contrast, not a priority claim. The work does not establish that written text overwrites latent state or that transformers cannot maintain a latent state across several steps. [docs/research/2026-08-16-dag-literature-and-claim-boundary.md:22-24] [docs/research/2026-08-16-dag-literature-and-claim-boundary.md:81-108]

## E. Next steps, in order

### DAG: make the workshop draft

1. Use notebook 16 as the figure and table source.
2. Draft the short manuscript around D1 (matched pair), D7/D8 (same-trace control), D2 (semantic cliff with distributional decay), and D3/D4 as limited supporting evidence.
3. Report seed/item summaries and the limitations explicitly. Freeze citations and artifacts.

Notebook 16 already contains the short workshop story; notebook 15 is the full ledger. Both are generated and must be changed through their builders, not by hand-editing the notebook. [notebooks/README.md:17-21] [notebooks/README.md:62-87]

### RMD: close the paper before writing

1. **Outcome and provenance table — CPU.** Archive the result tables, prompt IDs, folds, model revisions, prompts, decoding/extraction rules, and commands. Then report \(C_B\) on all prompts, cap-free results as conditional, parseable-at-cap answers separately, and the sampled \(C_{B\rightarrow B'}\) continuation case separately. The notebook currently depends on result artifacts absent from this checkout. [docs/research/2026-08-21-rmd-workshop-review.md:57-73] [docs/research/2026-08-21-rmd-workshop-review.md:101-103]
2. **TrAC claim alignment — desk work.** Cite TrAC and state the target and cost difference: it scores response correctness with a short trace-conditioned answer probe, while RMD scores a target-model plurality outcome from white-box states. A full TrAC reproduction is optional; do not make a best-system or broad self-consistency-augmentation claim without it. [outputs/reviews/2026-08-22-rmd-literature-novelty-check.md:15-15] [outputs/reviews/2026-08-22-rmd-literature-novelty-check.md:44-53]
3. **Peer cost ladder — CPU.** Compare B0, B0+RMD, one peer, two peers, and available peer sample counts at explicit model-call or generated-token cost. Peer features already beat or tie RMD on the three models, but the present comparison spends 16 additional peer generations and is not cost matched. [docs/research/2026-08-21-rmd-workshop-review.md:83-87]
4. **Published-style probe decomposition — CPU on cached states.** Reproduce the last-token, prompt-disjoint supervised-probe setup, select layers inside training folds, and report pooled trace AUROC next to micro and macro within-prompt AUROC. This validates the paper's interpretation; a collapse is a feature-specific measurement result, not a claim to have invented the decomposition. [docs/research/2026-08-21-rmd-workshop-review.md:89-93] [outputs/reviews/2026-08-22-rmd-literature-novelty-check.md:28-36]
5. **Outer-refit stability — CPU.** Refit PCA, reference Gaussians, feature/layer choices, and readouts over outer prompt partitions. A bootstrap over fixed out-of-fold predictions does not test this uncertainty. [docs/research/2026-08-21-rmd-workshop-review.md:95-99]
6. **Only then update notebook 14 and draft.** If the residual is unstable or the probe retains within-prompt signal, narrow the paper further rather than adding more experiments. [docs/research/2026-08-21-rmd-workshop-review.md:145-168]

## F. Notebook decision and reacquaintance path

Do **not** create a new RMD notebook. Reuse `notebooks/14_rmd_paper_story.ipynb` after the closure analyses. It is the existing paper-facing artifact, but it currently calls the score a single-forward-pass signal, treats cap-free analysis as missing-data handling, and frames peer rates only as a non-deployable control. Those statements need replacement. [notebooks/README.md:17-24] [notebooks/14_rmd_paper_story.ipynb:18-26] [notebooks/14_rmd_paper_story.ipynb:155-162] [notebooks/14_rmd_paper_story.ipynb:490-504]

The revised RMD notebook needs only five exhibits:

1. the full-population budget outcome table plus the continuation case;
2. a costed B0/RMD/peer ladder;
3. raw versus peer-conditioned AURC increments, with the DeepSeek-Qwen headroom marked;
4. pooled versus micro/macro within-prompt results for RMD and the matched probe;
5. full-refit stability and a short list of closed applications.

Use this reading order to regain context without reopening every experiment:

1. DAG: `notebooks/16_dag_workshop_story.ipynb` → `PAPER_STRATEGY_DAG.md` → `notebooks/15_dag_paper_story.ipynb` for provenance.
2. RMD: `docs/research/2026-08-21-rmd-workshop-review.md` → `PAPER_STRATEGY_RMD.md` → `notebooks/14_rmd_paper_story.ipynb`, treating its cap and peer language as stale.
3. For both: read the respective claim-boundary/literature note before drafting a title or abstract.

## Files and checks

- Wrote this review and retained the separate literature memo at `outputs/reviews/2026-08-22-rmd-literature-novelty-check.md`.
- Left all files under `raw/`, including the source repository, unchanged.
- Did not execute notebooks or models. The RMD result artifacts needed for execution are absent from this checkout, so only documentary and source-level checks were performed.
