# Notebooks

Current evidence lives here; finished diagnostics live in [`archive/`](archive/).
Each notebook also states its own status in its first cell.

Numbering is historical and has gaps — retired notebooks (04 cross-model
transfer, 05 temperature robustness, 06 prefix/BoN followups, 07 discourse
eigenspectrum, 00 tidy results) were deleted rather than renumbered, and moving
a notebook to `archive/` does not renumber it, so existing references to
notebook numbers stay valid.

Archived notebooks still run in place — `_viz_utils` and `results/` are located
by walking up to the repo root, so `archive/` needs no path changes.

## Current evidence

| Notebook | Regime | Bottom line |
|:---|:---|:---|
| [14_rmd_paper_story](14_rmd_paper_story.ipynb) | Between-prompt | **Start here.** The narrative spine of the selective-prediction half: the object of study, what collapsed under control, the increment on three models, the four controls it survives, and the four things geometry does not do. Reads every number out of a committed artifact. |
| [15_dag_paper_story](15_dag_paper_story.ipynb) | Causal (synthetic) | **The other half, on its own.** Activation patching on a synthetic arithmetic DAG, written as the paper argument: three claims, then the whole intervention — its vocabulary, one real item end to end, and the headline result — in one opening figure, the depth ladder and its specificity profile, the distance / cross-item / written-vs-omitted objections, then E2 stage A's measured confound and stage B's pre-registered 24/24 vs 0/24. The only write-up of that thread outside `results/dag_patching/` and `EXPERIMENT_LOG.md`. |
| [13_deepconf_null_and_label_efficiency](13_deepconf_null_and_label_efficiency.ipynb) | Between-prompt | **Most recent.** DeepConf is at chance on both models in all three framings, so no external baseline is left to beat. Against a supervised probe on the same states, one-class geometry leads below a crossing at 60–226 labels — but because the LDA collapses there, not because the Gaussian excels. ~2× label saving, confined to 25–100 labels. |
| [11_prompt_geometry_core_experiments](11_prompt_geometry_core_experiments.ipynb) | Within-prompt | **Primary analysis.** Entropy-localized RMD beats full-trace RMD at every layer (+0.052/+0.055/+0.058, p ≤ 0.006) and is entropy-specific, but only *ties* free output baselines. Sample selection is negative with a structural ceiling. |
| [12_wave1_abstention](12_wave1_abstention.ipynb) | Between-prompt | **Headline positive.** `rmd_tail_q20` hits 0.852 acc@50% coverage vs length 0.748 / entropy 0.692; beats the length confound baseline by +0.069 AURC (p < 0.001). Mechanism follow-ups E4–E7 all negative. |
| [01_main_effect_overview](01_main_effect_overview.ipynb) | Pooled (legacy) | Geometry-vs-entropy at each condition's best layer. **Read the length-controlled bars only** — pooled all-trace AUCs are length-confounded. |
| [02_layer_dynamics](02_layer_dynamics.ipynb) | Pooled | Layer profile is **bimodal**, peaks near L6–L10 and L20–L26 with a trough at L14. Motivates the sparse L7/14/21 probe layers. Single model, single dataset. |

## `archive/` — negative or null results

Kept because they close questions, moved out of the way because nothing here is
pending. None correspond to an active DVC stage; their outputs remain under
`results/` as provenance.

| Notebook | Verdict |
|:---|:---|
| [08_trajectory_fpca_vs_scalar](archive/08_trajectory_fpca_vs_scalar.ipynb) | **Negative.** Functional-PCA trajectory encoding never beats scalar Mahalanobis summaries in any of 4 conditions; near chance on Qwen GSM8K. Explains why the headline score is a scalar summary, not a shape. |
| [09_pca_ablation_analysis](archive/09_pca_ablation_analysis.ipynb) | **Null.** `pca_dim` spread is ~±0.03 and non-monotone; 128 is best or near-best in 9/12 cells. Closes the "PCA dim never swept" limitation. |
| [10_prefix_filter_analysis](archive/10_prefix_filter_analysis.ipynb) | **Negative.** Zero of 270 abort-and-retry operating points achieve positive token savings; false-abort rate tracks the base rate. |
| [03_math500_stratification](archive/03_math500_stratification.ipynb) | **Stale inputs.** Reads single-sample MATH-500, a stage no longer in the graph, with pooled per-stratum AUCs that carry the length confound. Needs redoing within-prompt if stratified claims matter. |

`archive/findings_corrected.html` is a rendered snapshot of a deleted notebook,
kept for provenance only.

## Reading order

Start at 14 for the selective-prediction argument end to end, then follow it into
the detail: 12 (where geometry wins), 11 (where it does not), 13 for what
survives once the baselines and the supervised probe have both had their turn,
then 02 for the layer story. 01 is the historical headline and should be read
with its caveat. 15 is independent of all of them and can be read first or last.

14 and 15 load committed result JSON and format it, so they stay honest by
breaking when an artifact moves; the aggregation they do do (medians over stored
per-item gate diagnostics, the arm inventory, Fisher on the stage-B 2×2 beside
the exact paired test the artifact carries) is stated in the cell that does it. 15 also regenerates two item traces for its
first figure with `dag/dag_tasks.py`, from the identity each arm stored for the
item, and checks them against the digits that arm recorded before drawing them.
Skim `archive/` before proposing any follow-up in those directions — the
questions are answered.

Retirement rationale for every removed DVC stage, with the evidence behind each
verdict, is in [`EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md) under
**2026-07-25: DVC graph restructure**.
