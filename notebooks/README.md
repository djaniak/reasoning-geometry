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
| [16_dag_workshop_story](16_dag_workshop_story.ipynb) | Causal (synthetic) | **Start here for the DAG paper.** Three figures and three tables: the clean/implied/raw assay, the E3 semantic cliff with distributional decay and same-trace control, the pre-registered matched confirmation, the stated-result ablation, and the controls that limit the claim. Notebook 15 remains the full experiment ledger. |
| [14_rmd_workshop_story](14_rmd_workshop_story.ipynb) | Between-prompt | **Start here for the RMD paper.** The narrative spine of the selective-prediction half: the two estimands one pooled AUROC gets asked to answer, the outcome protocol, the decomposition, the increment that survives it, why the feature is measured on the trace tail (§4a), and the peer cost ladder. Reads every number out of a committed artifact; the full-refit gate in §4 is still open. Notebooks 11–13 remain the long-form record. |
| [15_dag_paper_story](15_dag_paper_story.ipynb) | Causal (synthetic) | **Full DAG experiment ledger.** The intervention, pilot ladder, control failures, E2 matching, E3 campaign, gap/confidence analysis, quorum defect, and artifact provenance. Use 16 for the workshop story and 15 to check how the result was established. |
| [13_deepconf_null_and_label_efficiency](13_deepconf_null_and_label_efficiency.ipynb) | Between-prompt | **Most recent.** DeepConf is at chance on both models in all three framings, so no external baseline is left to beat. Against a supervised probe on the same states, one-class geometry leads below a crossing at 60–226 labels — but because the LDA collapses there, not because the Gaussian excels. ~2× label saving, confined to 25–100 labels. |
| [11_prompt_geometry_core_experiments](11_prompt_geometry_core_experiments.ipynb) | Within-prompt | **Primary analysis.** Entropy-localized RMD beats full-trace RMD at every layer (+0.052/+0.055/+0.058, p ≤ 0.006) and is entropy-specific, but only *ties* free output baselines. Sample selection is negative with a structural ceiling. |
| [12_wave1_abstention](12_wave1_abstention.ipynb) | Between-prompt | **Headline positive.** `rmd_tail_q20` hits 0.852 acc@50% coverage vs length 0.748 / entropy 0.692; beats the length confound baseline by +0.069 AURC (p < 0.001). Mechanism follow-ups E4–E7 all negative. |

## `archive/` — closed, superseded, or null

Kept because they close questions or record how a claim used to be stated, moved
out of the way because nothing here is pending. A notebook lands here when the
paper takes at most a sentence from it and that sentence has been placed
somewhere the paper actually reads — the sentence moves, the notebook retires.
Archiving is not a verdict on the result: 02 below is a standing positive.

Two of these still back live machinery. `02_layer_dynamics` is why
`collect_qwen_dense_math500` remains in the DVC graph, and it is cited from
notebook 14's scope paragraph as the justification for the L7/14/21 grid.

| Notebook | Verdict |
|:---|:---|
| [08_trajectory_fpca_vs_scalar](archive/08_trajectory_fpca_vs_scalar.ipynb) | **Negative.** Functional-PCA trajectory encoding never beats scalar Mahalanobis summaries in any of 4 conditions; near chance on Qwen GSM8K. Explains why the headline score is a scalar summary, not a shape. |
| [09_pca_ablation_analysis](archive/09_pca_ablation_analysis.ipynb) | **Null.** `pca_dim` spread is ~±0.03 and non-monotone; 128 is best or near-best in 9/12 cells. Closes the "PCA dim never swept" limitation. |
| [10_prefix_filter_analysis](archive/10_prefix_filter_analysis.ipynb) | **Negative.** Zero of 270 abort-and-retry operating points achieve positive token savings; false-abort rate tracks the base rate. |
| [01_main_effect_overview](archive/01_main_effect_overview.ipynb) | **Superseded.** The historical pooled headline: geometry vs entropy at each condition's best layer, length-controlled bars beside length-confounded ones. Notebook 14 §2–§3 makes the same correction on a stated population with intervals, so nothing here is quoted any more. |
| [02_layer_dynamics](archive/02_layer_dynamics.ipynb) | **Positive, absorbed.** Layer profile is bimodal — peaks near L6–L10 and L20–L26, trough at L14 — which is what motivates probing L7/14/21 rather than the last layer alone. That sentence now lives in notebook 14 §6. Qwen MATH-500 only; replication on the distill models is untested. |
| [03_math500_stratification](archive/03_math500_stratification.ipynb) | **Stale inputs.** Reads single-sample MATH-500, a stage no longer in the graph, with pooled per-stratum AUCs that carry the length confound. Needs redoing within-prompt if stratified claims matter. |

`archive/findings_corrected.html` is a rendered snapshot of a deleted notebook,
kept for provenance only.

## Reading order

Start at 14 for the selective-prediction argument end to end, then follow it into
the detail: 12 (where geometry wins), 11 (where it does not), 13 for what
survives once the baselines and the supervised probe have both had their turn,
then into `archive/` only to check how something used to be stated. The DAG
thread is independent: read 16 for the short paper
story and 15 for its full evidence trail.

14, 15, and 16 load committed result JSON and format it, so they stay honest by
breaking when an artifact moves; the aggregation they do do (medians over stored
per-item gate diagnostics, the arm inventory, Fisher on the stage-B 2×2 beside
the exact paired test the artifact carries) is stated in the cell that does it. 15 also regenerates two item traces for its
first figure with `dag/dag_tasks.py`, from the identity each arm stored for the
item, and checks them against the digits that arm recorded before drawing them.
Skim `archive/` before proposing any follow-up in those directions — the
questions are answered.

## 14, 15 and 16 are generated, not hand-edited

`15_dag_paper_story.ipynb` is written by
[`build_15_dag_paper_story.py`](build_15_dag_paper_story.py) and then executed in
place by [`execute_notebook.py`](execute_notebook.py), which drives a kernel
through `jupyter_client` because `nbconvert` and `nbclient` are not in this
environment. Edit the generator and rebuild; a hand-patch of the `.ipynb` is
overwritten by the next build.

`16_dag_workshop_story.ipynb` is the shorter paper storyboard. Its generator
reuses the maintained loading, styling, and intervention figure from 15, then
builds the paper-specific figures and prose around the same committed artifacts.

`14_rmd_workshop_story.ipynb` is the RMD-side storyboard and is built the same
way from [`build_14_rmd_workshop_story.py`](build_14_rmd_workshop_story.py). It
has no ledger generator behind it yet — 11, 12 and 13 are the long-form record
for their own eras, and the closure experiments (`budget_outcomes`,
`peer_cost_ladder`, `last_token_probe`) plus the controls that appear in no
notebook at all (`closest_baselines`, `orgad_agreement_control`,
`peer_difficulty_control`, `application_alignment`, `allocation_precheck`,
`label_efficiency_supervision_ladder`, `label_efficiency_token_pooling`) are
currently documented only in `EXPERIMENT_LOG.md`.

Two kinds of guard sit on it. At **build** time the generator asserts, over the
assembled prose, that five claims withdrawn in the 2026-08-22 rewrite appear
only in cells that retract them, and that the code-cell count has not drifted —
so a rebuild that reintroduces a withdrawn claim fails rather than publishing
it. At **execution** time §3 asserts that the reference scores still reproduce
the frozen Qwen layer-21 `prompt_decomposition` report on all four of its
columns, to 1e-3, so a silent monotone change of scale cannot pass.

```
uv run python notebooks/build_14_rmd_workshop_story.py
uv run --with pandas --with jinja2 --with jupyter_client --with ipykernel \
  python notebooks/execute_notebook.py notebooks/14_rmd_workshop_story.ipynb notebooks

uv run python notebooks/build_15_dag_paper_story.py
uv run --with pandas --with jinja2 --with jupyter_client --with ipykernel \
  python notebooks/execute_notebook.py notebooks/15_dag_paper_story.ipynb notebooks

uv run python notebooks/build_16_dag_workshop_story.py
uv run --with pandas --with jinja2 --with jupyter_client --with ipykernel \
  python notebooks/execute_notebook.py notebooks/16_dag_workshop_story.ipynb notebooks
```

Then **look at the figures**. A layout collision renders silently and does not
raise, so a build that reports zero failed cells can still have produced an
unreadable panel.

Retirement rationale for every removed DVC stage, with the evidence behind each
verdict, is in [`EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md) under
**2026-07-25: DVC graph restructure**.
