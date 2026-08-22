# Paper Strategy — index

*Split 2026-08-15.* This repo carries **two independent papers**. They were
tracked in one strategy doc until now, which made it easy to let a limitation of
one qualify the other. They share no model, no dataset, no metric, and no claim.

| Doc | Thread | Status |
|:---|:---|:---|
| [`PAPER_STRATEGY_RMD.md`](PAPER_STRATEGY_RMD.md) | Relative Mahalanobis distance for selective prediction on MATH-500; 7–8B models | Evidence largely frozen. Binding constraint is **breadth** (single dataset, §7e), plus two unrun baselines (semantic entropy / EigenScore). |
| [`PAPER_STRATEGY_DAG.md`](PAPER_STRATEGY_DAG.md) | Residual-stream activation patching on a synthetic arithmetic DAG; DeepSeek-R1-Distill-Qwen-1.5B | **Workshop-submittable now**, and at its registered N as of 2026-08-16. Main-conference path runs through a clean-valid multi-step format (§6b, G1/G2). The node-influence matrix E4 is **answered and negative on this format** — not pending — and survives only as a post-G1 experiment. |

Section numbers in `PAPER_STRATEGY_RMD.md` are unchanged from the pre-split file,
so existing `§n` references (e.g. `params.yaml`, `EXPERIMENT_LOG.md`) still
resolve — the filename in those references is stale, the section is not.

Neither paper's evidence is in the other's doc. If a sentence needs both, it is
wrong.
