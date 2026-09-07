# Claim–evidence table

Built 2026-09-07 from the repaired write-ups. **This is the source for manuscript
drafting**: a claim not in this table is not ready to appear in the paper, and every
number in the paper should be traceable to a row here.

**Every contrast below records its exact left and right feature sets and the JSON
key it is read from.** Feature sets use the frozen definitions stored in each
artifact's `feature_definitions` block:

```
B0            = [length, entropy, logprob, vote_agreement]
B1            = B0 + [rmd_tail_q20]
B0_plus_H     = B0 + [neg_answer_entropy]
B0_plus_peer  = B0 + [peer_pass_rate__<other1>, peer_pass_rate__<other2>]
B0_plus_dcvote= B0 + [dcvote_deepconf_tail_q20]
rmd_tail_q20  = negative mean per-token RMD over the final 20% of generated tokens
rmd_full      = negative mean per-token RMD over the whole trace (Vazhentsev ATRMD)
```

A contrast written `X − Y` is `AURC(X readout) − AURC(Y readout)`, paired on the
same prompts, so negative favours `X`. Every readout is the frozen cross-fitted
logistic on the frozen prompt folds.

Conventions. *Prediction unit* is what a single scored item is — a prompt or a trace.
*Feature timing* is when the feature could be computed in deployment: `post-8` means
after all eight traces exist, `post-1` after a single trace, `oracle` means it needs
information not available at decision time (a gold label, or other models' runs).
AURC is lower-is-better; a negative delta favours the left-hand readout. Intervals are
the frozen prompt-clustered paired bootstrap over fixed OOF predictions and do not
propagate reference refitting.

---

## A. The headline claim

**A1. Adding `rmd_tail_q20` to a self-consistency baseline lowers AURC on all three
models.**

| | |
|---|---|
| Models | all three |
| Population | `full_population`, n=500 |
| Prediction unit | prompt |
| Feature timing | post-8 (aggregates over the eight traces) |
| Artifact | `results/{model}_bestofn_full/math500/math500_incremental_abstention_results.json` — **the intervals come only from these**. `closest_baselines_report.md` gives the same point estimates with different intervals ([−0.0832,−0.0218] / [−0.0531,−0.0060] / [−0.0773,−0.0178]) because it uses a different bootstrap seed offset. Quote one source per interval and say which. |
| Contrast | left `B1` = [length, entropy, logprob, vote_agreement, rmd_tail_q20] · right `B0` = [length, entropy, logprob, vote_agreement] |
| JSON key | `populations.full_population.paired_deltas.B1_minus_B0_aurc` |
| Value | qwen −0.05201702 [−0.08447729, −0.02181222] p=0.000; deepseek −0.02836542 [−0.05261266, −0.00481978] p=0.010; deepseek_llama −0.04694501 [−0.07425313, −0.01616323] p=0.004 |
| Provenance | seed 42, `n_bootstrap` 1000, layers 21 / 21 / 24, caps 1024 / 8192 / 12288, target `plurality_vote_correctness`. sha256 (first 16) `1a8941c499dcb81a` / `221cd8260a34ec72` / `089f8b1bfdba1d01`. **Now git-tracked** (`.gitignore` exception added 2026-09-07). |
| Checks completed | reproduced from JSON to the digit; survives the answer-histogram control (A2); reproduced bit-identically at refit seed 42; sign-stable across three refits on the two models refitted, with spread narrower than the bootstrap interval |
| Limitations | Qwen has no refit at a *new* partition (its seed-42 refit is the frozen-partition reproduction check and matches bit-for-bit). The frozen deepseek_llama value is the **largest** of its three refits (mean −0.0372). Single dataset (MATH-500), single budget (N=8), one shared outer prompt partition across all three models (B9, still open). |

**A2. The increment is not the answer histogram.**

| | |
|---|---|
| Models | all three · Population `full_population` · unit prompt · timing post-8 |
| Artifact | `results/closest_baselines/closest_baselines_results.json` |
| Contrasts | `H over B0`: left `B0+[neg_answer_entropy]` · right `B0`. `rmd_tail over B0+H`: left `B0+[neg_answer_entropy, rmd_tail_q20]` · right `B0+[neg_answer_entropy]` |
| JSON key | `models[].populations.full_population.paired_deltas_aurc.{H_over_B0, rmd_tail_over_B0_plus_H}` — the node is `paired_deltas_aurc` and its value *is* the AURC record, so there is no trailing `.aurc`. Model identity is the `label` field, not `model`. **The Holm values are not in this JSON**: `holm_adjusted` runs at report-generation time (`baselines/closest_baselines.py:697`) and writes only to `closest_baselines_report.md`, which is the sole source for 0.020 / 0.032 / 0.032. |
| Value | `H over B0` −0.0003 p=0.598 / +0.0001 p=0.792 / +0.0016 p=0.056; `rmd_tail over B0+H` −0.0519 / −0.0281 / −0.0462, Holm p 0.020 / 0.032 / 0.032 |
| Checks completed | family pre-declared at `EXPERIMENT_LOG.md, section “2026-08-09: The two closest cheap baselines, and whether the tail is a window artifact”` ("Pre-declared rules, written before the run"), which is the pre-registration itself — not `:3298`, which is the later results write-up asserting it; Holm computed by `baselines/closest_baselines.py:448` |
| Limitations | The bootstrap resolves p to 1/1000; these clear the threshold but not by a wide margin. The other five contrasts per model are exploratory and unadjusted. |

## B. Controls the increment survives — with the coverage each actually has

**No control below is a clean three-model replication on the primary population.**
This is the single largest honesty exposure in the paper and each row states its own gap.

| # | Control | Models | Population | Artifact | Result | Gap |
|---|---|---|---|---|---|---|
| B1 | Vote proxy (Orgad et al. 2410.02707) | all three | **`full_population`** (2026-09-07) | `results/orgad_agreement_control_full_population/` | unanimous-stratum AUROC of `rmd_tail_q20` (single feature, no readout) 0.827 [0.774,0.876] qwen / 0.717 [0.660,0.770] deepseek / 0.739 [0.670,0.802] llama, on n = 302 / 424 / 253, where `vote_agreement` is constant. Key `[<i>].strata.unanimous.auroc` (this JSON's top level is a **list** of model records, not an object with a `models` field). **List order is `[0]=DeepSeek-Qwen, [1]=Llama, [2]=Qwen`** — the reverse of this table's usual qwen/deepseek/llama order, because the recorded command passes the models in that order with those labels. Index by `label`, not position. | runs on the primary population now, but this row is a single-feature AUROC inside a stratum, not a replication of the increment contrast — it does not make the control a clean three-model replication of `B1 − B0` |
| B2 | MATH-500 annotated level + budget-edge difficulty | qwen, deepseek | — | `results/{qwen,deepseek}_bestofn_full/math500/math500_difficulty_control_results.json` | increment remains after the tested annotated-level and budget-edge controls; this does not exclude other difficulty signals | **deepseek_llama never run** (`EXPERIMENT_LOG.md, section “2026-08-03: The increment is not a prompt-difficulty proxy (BOTH models)”`) |
| B3 | Cross-model peer pass rate | all three | **`full_population`** (2026-09-07) | `results/peer_difficulty_control_full_population/` | pre-declared stop rule **TRIGGERED** (deepseek + deepseek_llama overlap zero; rule needs 2 of 3). Contrast `B1_minus_B0_given_peer`: left `B0+peer+[rmd_tail_q20]` · right `B0+peer`, key `models[].populations.full_population.paired_deltas.B1_minus_B0_given_peer_aurc`. −0.0067 [−0.0164,−0.0002] p=0.042 / −0.0006 [−0.0016,+0.0003] p=0.194 / −0.0025 [−0.0069,+0.0026] p=0.280. 87/98/95% absorbed; headroom share removed 20/8/3%; **Holm significant on none** (0.126/0.388/0.388) | none — this is now run on the primary population. The cap-free run (`results/peer_difficulty_control/`) did **not** trigger; both are retained and the primary one governs |
| B4 | Length residualization | qwen, deepseek | — | `EXPERIMENT_LOG.md, section “2026-07-31: Supervised probe ceiling + length residualization (BOTH models)”` | increment survives | **deepseek_llama never run** (predates that collect) |
| B5 | DeepConf (2508.15260), 3 forms, 4 statistics | deepseek, deepseek_llama | **`full_population`** (2026-09-07) | `results/deepconf_asymmetry_full_population/`, `results/deepconf_weighted_vote_full_population/` | Contrast: left `B0+[dcvote_deepconf_tail_q20, rmd_tail_q20]` · right `B0+[dcvote_deepconf_tail_q20]`, key `models[].populations.full_population.paired_deltas.B1_minus_B0_plus_dcvote_deepconf_tail_q20_aurc` → **−0.02849 [−0.05259, −0.00427] p=0.022** / **−0.04630 [−0.07775, −0.01857] p=0.002**. (Do not use the script's stdout line: it prints `B1-B0+dcvote=+0.0286 … p=0.020`, which is `bottom10_group_confidence` in **AUACC**, a different statistic on a different metric.) DeepConf single-score AUROC point estimates near 0.5 (not an equivalence claim): 0.497/0.542/0.485/0.485 and 0.496/0.519/0.488/0.491 | **unavailable from the existing Qwen cache** — no token arrays in `data/qwen_bestofn_full` (`EXPERIMENT_LOG.md, section “2026-08-05: The increment does NOT clear DeepConf's tail statistic (DeepSeek)”`); requires additional collection or recovery of missing inputs, outside this repair |

**The B3 wording that is required.** On the primary population the pre-declared stop
rule fires, so the registered consequence binds: *report the increment as
substantially a prompt-difficulty proxy, and narrow the "geometry adds beyond
output-side confidence" framing accordingly.* Do not report the cap-free run's
non-triggering as the result. Two limits on how far this cuts: `B0 + peer` is not
deployable, so nothing here competes with the headline as a method and the headline
increment is unchanged; the peer feature uses gold-scored outcomes from other models. It is a diagnostic, not a deployable feature or a causal identification of difficulty. What narrows is the mechanism claim.

**The B5 wording that is required.** `EXPERIMENT_LOG.md, section “2026-08-05: The increment does NOT clear DeepConf's tail statistic (DeepSeek)”` instructs that any
write-up must state that this baseline cannot run from the existing Qwen cache "rather than implying
replication". Qwen is the model where the increment is strongest.

## C. Comparison against the closest published statistic

**C1. Adding the tail to a baseline that already includes whole-trace RMD gives no statistically resolved increment on the two distilled models.**

| Field | Evidence |
|---|---|
| Scope | deepseek, deepseek_llama; `full_population`; prompt; post-8 |
| Artifact | `results/closest_baselines/closest_baselines_results.json` |
| Contrast | left `B0+[rmd_full, rmd_tail_q20]`; right `B0+[rmd_full]` |
| JSON key | `models[].populations.full_population.paired_deltas_aurc.rmd_tail_over_rmd_full` |
| Result | −0.0030 [−0.0109,+0.0035], p=0.436; −0.0041 [−0.0118,+0.0035], p=0.320 |
| Interpretation | These intervals describe the extra contribution of the tail after whole-trace RMD is included. They do not compare the separate tail-only and whole-trace-only readouts, establish equivalence, or bound every possible tail advantage. |
| Separate descriptive results | Whole-trace over B0: −0.0287 / −0.0445. Tail over B0: −0.0284 / −0.0469. These are point estimates from separate contrasts. |

**C2. On Qwen, adding the tail to B0 plus whole-trace RMD improves AURC.**

| | |
|---|---|
| Model | qwen · Population `full_population` · unit prompt · timing post-8 |
| Contrast and key | left `B0+[rmd_full, rmd_tail_q20]`; right `B0+[rmd_full]`; `models[].populations.full_population.paired_deltas_aurc.rmd_tail_over_rmd_full` |
| Value | `rmd_tail over rmd_full` −0.0464 [−0.0724, −0.0224] p=0.000 |
| Limitations | Trace style, budget, base accuracy and distillation are all collinear across the three models. A third non-distilled model is the test that would separate them, and has not been run. |

**C3. Exploratory Qwen window-size strata show a non-monotone pattern of point estimates.**

| | |
|---|---|
| Model | qwen · Population `full_population` (primary) · unit prompt |
| Artifact | `results/closest_baselines/closest_baselines_report.md` §"1b follow-up", computed on `populations[0]` (`baselines/closest_baselines.py:381`); strata sum to 500 |
| Value | terciles −0.0448 [−0.0993,−0.0044] / −0.1083 [−0.1682,−0.0489] / −0.0491 [−0.0868,−0.0084]; below median −0.0563 [−0.1038,−0.0163], above median −0.0662 [−0.0998,−0.0288] |
| Checks completed | both tercile and median splits present on the primary population |
| Limitations | **Do not write "the advantage grows with window size."** That reading comes from the `cap_free_all_eight_parseable` run (−0.0419 / −0.1158, `EXPERIMENT_LOG.md, section “2026-08-09: The two closest cheap baselines, and whether the tail is a window artifact”`), where the gradient is 2.8×; on the primary population it is 1.2× and the terciles are non-monotone (an inverted U). These strata do not establish a monotone trend or exclude window size as an explanation of cross-model differences. The median split was adopted **after** the tercile split was refused for a small minority class, stated as such at the time; every stratified contrast is exploratory and unadjusted (`:3435`). The Llama matched-window stratum (n=154, base acc 0.688 vs Qwen 0.693, −0.0088 p=0.136) exists **only** on `cap_free_all_eight_parseable` and has not been repeated on the primary population; DeepSeek's matched stratum is refused at n=38 with 4 incorrect. |

**C4. The increment persists at the registered 10%, 20%, and 50% tail cutoffs.**

| | |
|---|---|
| Models | all three · Population `full_population` · unit prompt · timing post-8 |
| Artifact | `results/rmd_window_sensitivity/rmd_window_sensitivity_results.json` |
| Contrast | left `B0 + [<detector>]` · right `B0`, for each detector in {`rmd_tail_q10`, `rmd_tail_q20`, `rmd_tail_q50`, `rmd_full`, `rmd_high_entropy_q20`, `rmd_random_q20`} |
| Registered rule | fixed 2026-08-22 before q10/q50 were computed: q10, q20 and q50 must each improve AURC over `B0` with a 95% interval below zero on every checkpoint |
| Verdict | **PASSES** on all three models. The sanctioned sentence is "we fixed 20% as a simple localized window; sensitivity analysis shows the result is not specific to this exact cutoff." |
| Values | q10 / q20 / q50: qwen −0.0559 / −0.0520 / −0.0304; deepseek −0.0342 / −0.0284 / −0.0376; llama −0.0572 / −0.0469 / −0.0477 |
| Mandatory disclosures | (i) `rmd_tail_q10` has a more favorable point estimate than frozen q20 on all three; no superiority test is reported here. The registration forbids calling 20% optimal or replacing the frozen feature — so do neither, but state it. (ii) On both distilled models a random 20% window lands at the same place as the tail: −0.0295 vs −0.0284 (deepseek) and −0.0444 vs −0.0469 (llama), against −0.0184 vs −0.0520 on Qwen. **No paired contrast between `rmd_random_q20` and `rmd_tail_q20` was computed**, so this is a comparison of point estimates, not a test — state it as "random-window and tail point estimates are similar on the distilled models and farther apart on Qwen; the differences have not been tested", not as an established equivalence. The direction cuts against the paper, which is why it must be disclosed, but it is not measured. (iii-bis) Qwen's q50 clears the registered rule by 0.0012 (−0.0304 [−0.0596, −0.0012]); by this repo's own convention for near-threshold results, read that as borderline rather than a clean pass. (iii) `rmd_high_entropy_q20` is weak everywhere (−0.0148 n.s. / −0.0052 n.s. / −0.0274), consistent with the 2026-07-29 entropy gate failure. |
| Limitations | The cutoff rule was registered before these outcomes were inspected. The audit-time in-memory inspection preceded the saved run and is recorded in the experiment log. Descriptive detector comparisons are exploratory; this analysis does not select a detector and does not restore the withdrawn tail-aggregator novelty claim. Reads the three seed-42 refit CSVs, so it inherits that partition. |

## D. Deployable-peer comparison

**D1. The six single-peer comparisons favor RMD once and the peer once; four are inconclusive under the raw intervals.**

| | |
|---|---|
| Models | all three · Population `full_population` · unit prompt · timing post-8 for `B1`, post-8+1 for the peer |
| Artifact | `results/peer_cost_ladder/peer_cost_ladder_report.md` §4 |
| Value | six single-peer deployable comparisons: 4 inconclusive comparisons, 1 RMD win (deepseek vs deepseek_llama peer, −0.0341 [−0.0576, −0.0108]), 1 peer win (deepseek_llama vs qwen peer, +0.0544 [+0.0240, +0.0838]); raw p = 0.170 / 0.062 / 0.008 / 0.408 / 0.878 / 0.000 |
| Checks completed | verdicts reproduced from `cheapest_peer_verdict`; raw p-values read from `contrasts` |
| Limitations | **This family is post-hoc**, selected after seeing the ladder, and `controls/peer_cost_ladder.py` computes no multiplicity correction. Report the raw intervals and say so. No peer rung is exactly cost-matched to `B1`, so the supported claim is "uses no additional generations beyond the target’s eight", never "beats peer uncertainty". |

## E. Negative and bounding results the paper keeps

**E1. Whether geometry predicts the gain from more samples is UNRESOLVED — the
pre-declared gate reverses between populations. The former negative claim is removed.**

| | |
|---|---|
| Models | all three · unit prompt · timing post-1 |
| Artifacts | `results/allocation_precheck_full_population/` (primary, 2026-09-07) and `results/allocation_precheck/` (cap-free, original) |
| JSON key | `gate.per_model.<model>.{geometry_r2_median, spearman_gain_from_geometry_median, passes}`; `gate.models_passing` |
| Target | `g(p) = a(p,8) − a(p,1)`, expected plurality-vote correctness over all `C(8,k)` sibling subsets |
| Value — `full_population` | gate **PASSES 3/3**. R² geometry vs cross-fitted constant +0.019 [0.011,0.033] / +0.137 [0.119,0.157] / +0.016 [0.010,0.019] |
| Value — `cap_free_valid_plurality` | gate **FAILS 2/3**. R² −0.0037 / −0.0065 / +0.0005168 |
| Checks completed | same pre-declared rule, both populations; diagnostics read on both |
| **Bracket semantics — read this before quoting E1** | The brackets in this row are **not bootstrap confidence intervals**. `applications/allocation_precheck.py:553` formats them as `median [min, max]` over the **8 stage-1 draws**. They are a spread across resamples of the subset-enumeration, not an inference statement, and they must never be reported as CIs or used to claim significance. This is the one row in this document where the convention stated at the top does not apply. |
| Limitations | **Do not report either direction as the finding.** Three reasons. (i) The gate's second leg ("geometry adds over output-alone") is decided on a median over 8 draws, and that median is small on every model: +0.018 [−0.002,+0.039] qwen, +0.006 [−0.016,+0.024] deepseek, +0.012 [+0.000,+0.065] deepseek_llama. Two of the three draw-ranges cross zero; deepseek_llama's minimum is +0.000485, so its range is strictly positive — but with n=8 draws and no interval estimate, none of this is a significance claim in either direction. (ii) The diagnostic the precheck exists to catch fires on **2 of 3** models on the primary population — `difficulty_not_gain` is `{qwen: true, deepseek: false, deepseek_llama: true}`, where on the cap-free population it fired on all three. Raw `rho(geometry, g)` is negative on all three (−0.064 / −0.293 / −0.060) while `rho(geometry, pass rate)` is +0.625 / +0.493 / +0.441; the report's own conclusion is "geometry reads difficulty but not marginal gain". On deepseek the flag clears only because |rho| = 0.293 is far from zero, i.e. geometry is *strongly* anti-correlated with gain there. A fitted readout can exploit an inverse relation, which is why the legs disagree. (iii) The gate was *registered* on the cap-free population; the primary population was adopted later (2026-08-22) but before this run, so the switch is not post-hoc — yet the reversal means the gate does not discriminate what it was built to discriminate. |

**E2. Pooled trace discrimination is not sibling verification.**

| | |
|---|---|
| Models | all three · Population `parseable` (trace-level) · unit **trace** · timing post-1 |
| Artifact | `results/last_token_probe/last_token_probe_results.json` |
| Value | probe pooled 0.9013 / 0.9139 / 0.9032 against macro within-prompt 0.6444 / 0.5823 / 0.7177; RMD shows the same gap; mixed prompts 117 / 49 / 158 |
| Checks completed | three readouts on one protocol; pooled−macro intervals exclude zero |
| Limitations | The deepseek macro 0.582 has a 95% interval of [0.463, 0.688], which **contains chance**. RMD's within-prompt macro on that model is **0.461, below chance**. Entropy and log-probability "lose less" than length, but on both distilled models all three baselines sit at or under 0.50 within prompts, so that is a statement about the size of the drop, not about retained signal. |

## F. Claims explicitly cut — must not reappear

| Claim | Why cut | Recorded at |
|---|---|---|
| Label efficiency: geometry leads a probe by −0.033 AURC at 50 labels | Cut 2026-08-22. The −0.033 **is** the pooling-matched comparison (against `probe_token_tail_q20`), but its interval spans zero: −0.033 [−0.044, +0.024], sign p=0.109. It decomposes into ≈−0.011 supervision + −0.018 decision-function form, so only a claim about the one-class inductive bias must quote −0.011. | `EXPERIMENT_LOG.md, section “2026-08-22: Scope — the label-efficiency claim is cut from the paper”`, `:3557`, `:3562`; `results/label_efficiency_token_pooling/label_efficiency_report.md` |
| Entropy-localized RMD beats full-trace and a random control (as a general claim) | Pre-registered gate **failed** on DeepSeek-R1-Distill-Qwen-7B on 2026-07-29 (+0.004 p=0.674; +0.001 p=0.924). Demoted to Qwen-specific by the decision rule fixed in advance. The Llama decomposition collect was cancelled *at that time* but later ran, so `rmd_random_q20` **does** exist for Llama; what was never re-run there is the gate itself. | `EXPERIMENT_LOG.md, section “2026-07-29: GATE FAILED — localization is Qwen-specific”`; `results/deepseek_llama_bestofn_full/math500/math500_prompt_decomposition_results.json` |
| Aggregator novelty (leg a) | Withdrawn; the contribution is the evaluation, not a new geometry statistic. | `EXPERIMENT_LOG.md, section “2026-08-09: The two closest cheap baselines, and whether the tail is a window artifact”` |
| Peer pass rates "absorb most of the increment", control withdrawn | The withdrawal was itself wrong and is reversed — see B3. | this table, `EXPERIMENT_LOG.md` 2026-09-07 |

## G. Evidence gates still open

| Gate | State | What closes it |
|---|---|---|
| Registered full-refit stability sweep | **partially closed, not closed** — 3 of 4 registered seeds, 2 of 3 models, peer step skipped at every seed | qwen at seeds 101/202; the peer step at each seed; seed 303. see the job specification; decomposition has a rough 140 GB scheduling estimate |
| Registered tail-window sensitivity | **CLOSED 2026-09-07 — rule passes on all three models** | done; artifact at `results/rmd_window_sensitivity/`. See C4 |
| Five controls on the primary population | **CLOSED 2026-09-07 — all five run** | done; see B1, B3, B5, E1 and the commands as actually executed in the 2026-09-07 log entry |
| Shared outer prompt partition (B9) | open, and not optional before submission | all three models share one partition; noted in the 1a/1b limitations |
