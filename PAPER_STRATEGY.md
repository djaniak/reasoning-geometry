# Paper Strategy: index

*Split 2026-08-15.* This repository carries **two independent papers**. They were
tracked in a single strategy document until now, which made it easy for a limitation of
one to be applied to the other. They share no model, no dataset, no metric, and no claim.

| Doc | Thread | Status |
|:---|:---|:---|
| [`PAPER_STRATEGY_RMD.md`](PAPER_STRATEGY_RMD.md) | Relative Mahalanobis distance for selective prediction on MATH-500; 7–8B models | Evidence largely frozen. Binding constraint is **breadth** (single dataset, §7e), plus two unrun baselines (semantic entropy / EigenScore). |
| [`PAPER_STRATEGY_DAG.md`](PAPER_STRATEGY_DAG.md) | Residual-stream activation patching on a synthetic arithmetic DAG; DeepSeek-R1-Distill-Qwen-1.5B | **Workshop-submittable now**, and at its registered N as of 2026-08-16. Main-conference path runs through a clean-valid multi-step format (§6b, G1/G2). The node-influence matrix E4 is **answered and negative on this format**, so it is not pending, and it survives only as a post-G1 experiment. |

Section numbers in `PAPER_STRATEGY_RMD.md` are unchanged from the pre-split file,
so existing `§n` references (e.g. `params.yaml`, `EXPERIMENT_LOG.md`) still
resolve. The filename in those references is stale, whereas the section number is not.

The evidence for each paper is held in its own document. A sentence that requires both
documents is therefore incorrect.
