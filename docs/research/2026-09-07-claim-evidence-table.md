# Claim–evidence table

Built 2026-09-07 from the repaired write-ups. **This is the source for manuscript
drafting**: a claim not in this table is not ready to appear in the paper, and every
number in the paper should be traceable to a row here.

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
| Value | qwen −0.0520 [−0.0845, −0.0218]; deepseek −0.0284 [−0.0526, −0.0048]; deepseek_llama −0.0469 [−0.0743, −0.0162] |
| Checks completed | reproduced from JSON to the digit; survives the answer-histogram control (A2); reproduced bit-identically at refit seed 42; sign-stable across three refits on the two models refitted, with spread narrower than the bootstrap interval |
| Limitations | **The three abstention JSONs carrying these intervals are not git-tracked**, so the headline is not reproducible from a fresh clone — fix before submission. Qwen has no refit at a *new* partition (its seed-42 refit is the frozen-partition reproduction check and matches bit-for-bit). The frozen deepseek_llama value is the **largest** of its three refits (mean −0.0372). Single dataset (MATH-500), single budget (N=8), one shared outer prompt partition across all three models (B9, still open). |

**A2. The increment is not the answer histogram.**

| | |
|---|---|
| Models | all three · Population `full_population` · unit prompt · timing post-8 |
| Artifact | `results/closest_baselines/closest_baselines_report.md` |
| Value | `H over B0` −0.0003 / +0.0001 / +0.0016, all p > 0.05; `rmd_tail over B0+H` −0.0519 / −0.0281 / −0.0462, Holm p 0.020 / 0.032 / 0.032 |
| Checks completed | family pre-declared at `EXPERIMENT_LOG.md:2976` ("Pre-declared rules, written before the run"), which is the pre-registration itself — not `:3050`, which is the later results write-up asserting it; Holm computed by `baselines/closest_baselines.py:448` |
| Limitations | The bootstrap resolves p to 1/1000; these clear the threshold but not by a wide margin. The other five contrasts per model are exploratory and unadjusted. |

## B. Controls the increment survives — with the coverage each actually has

**No control below is a clean three-model replication on the primary population.**
This is the single largest honesty exposure in the paper and each row states its own gap.

| # | Control | Models | Population | Artifact | Result | Gap |
|---|---|---|---|---|---|---|
| B1 | Vote proxy (Orgad et al. 2410.02707) | all three | `cap_free_valid_plurality` | `results/orgad_agreement_control/` | AUROC 0.829 / 0.714 / 0.756 inside the unanimous stratum, where `vote_agreement` is constant | never run on `full_population`; its report never names its population |
| B2 | MATH-500 annotated level + budget-edge difficulty | qwen, deepseek | — | `results/{qwen,deepseek}_bestofn_full/math500/math500_difficulty_control_results.json` | increment is not a difficulty proxy | **deepseek_llama never run** (`EXPERIMENT_LOG.md:4416`) |
| B3 | Cross-model peer pass rate | all three | `cap_free_valid_plurality` | `results/peer_difficulty_control/` | pre-declared stop rule **not triggered** (1 of 3 overlaps zero, rule needs 2); 81% / 99% / 78% attenuated; Holm leaves only deepseek_llama (p_holm 0.012 vs 0.072, 0.544) | never run on `full_population` |
| B4 | Length residualization | qwen, deepseek | — | `EXPERIMENT_LOG.md:4827` | increment survives | **deepseek_llama never run** (predates that collect) |
| B5 | DeepConf (2508.15260), 3 forms, 4 statistics | deepseek, deepseek_llama | `cap_free_valid_plurality` (asymmetry); `full_population` present in the weighted-vote JSON | `results/deepconf_asymmetry/`, `results/deepconf_weighted_vote/` | `B1 − (B0 + dcvote_deepconf_tail_q20)` AURC on `full_population`: −0.0285 [−0.0526, −0.0043] p=0.022 and −0.0463 [−0.0778, −0.0186] p=0.002 | **structurally impossible on Qwen** — no token arrays in `data/qwen_bestofn_full` (`EXPERIMENT_LOG.md:4398`); permanent, not a missing run |

**The B3 wording that is supported**: the increment survives on 2 of 3 models on the
raw intervals and 1 of 3 after the pre-declared Holm correction; on
DeepSeek-R1-Distill-Qwen-7B it is **eliminated**, not attenuated (−0.0004 [−0.0016,
+0.0005] p=0.544 against a remaining headroom of 0.0045 — the control's own report
warns that a small delta against no headroom is no evidence at all). Say both
readings. Not "absorbed everywhere" and not "withdrawn": the control is a confound
control, never a deployable baseline, so its gold-derivedness is not a defect — the
same is true of MATH-500's human-annotated level.

**The B5 wording that is required.** `EXPERIMENT_LOG.md:4400` instructs that any
write-up must state that this baseline can never run on Qwen "rather than implying
replication". Qwen is the model where the increment is strongest.

## C. Comparison against the closest published statistic

**C1. On the two reasoning-distilled models, no difference between whole-trace ATRMD and the tail is detectable at this n.**

| | |
|---|---|
| Models | deepseek, deepseek_llama · Population `full_population` · unit prompt · timing post-8 |
| Artifact | `results/closest_baselines/closest_baselines_report.md`, §1b |
| Value | `rmd_full over B0` −0.0287 / −0.0445 against `rmd_tail over B0` −0.0284 / −0.0469; `rmd_tail over rmd_full` p = 0.436 / 0.320 |
| Checks completed | both directions tested; neither separable from zero |
| Limitations | Not equivalence. The intervals ([−0.0109,+0.0035], [−0.0118,+0.0035]) bound any tail advantage at about 0.012 against an increment of 0.03–0.05; state the bound, do not declare a tie. Neither ordering of the point estimates is measurable, so the paper must not present the tail as recovering more. |

**C2. Only on Qwen does the tail carry the result.**

| | |
|---|---|
| Model | qwen · Population `full_population` · unit prompt · timing post-8 |
| Value | `rmd_tail over rmd_full` −0.0464 [−0.0724, −0.0224] p=0.000 |
| Limitations | Trace style, budget, base accuracy and distillation are all collinear across the three models. A third non-distilled model is the test that would separate them, and has not been run. |

**C3. Window size does not explain the Qwen/distilled split — but the advantage is
non-monotone in window size, not increasing in it.**

| | |
|---|---|
| Model | qwen · Population `full_population` (primary) · unit prompt |
| Artifact | `results/closest_baselines/closest_baselines_report.md` §"1b follow-up", computed on `populations[0]` (`baselines/closest_baselines.py:381`); strata sum to 500 |
| Value | terciles −0.0448 [−0.0993,−0.0044] / −0.1083 [−0.1682,−0.0489] / −0.0491 [−0.0868,−0.0084]; below median −0.0563 [−0.1038,−0.0163], above median −0.0662 [−0.0998,−0.0288] |
| Checks completed | both tercile and median splits present on the primary population |
| Limitations | **Do not write "the advantage grows with window size."** That reading comes from the `cap_free_all_eight_parseable` run (−0.0419 / −0.1158, `EXPERIMENT_LOG.md:3089`), where the gradient is 2.8×; on the primary population it is 1.2× and the terciles are non-monotone (an inverted U). The claim the data supports is only that the advantage does not *decay* with window size, which is what the window hypothesis predicted. The median split was adopted **after** the tercile split was refused for a small minority class, stated as such at the time; every stratified contrast is exploratory and unadjusted (`:3187`). The Llama matched-window stratum (n=154, base acc 0.688 vs Qwen 0.693, −0.0088 p=0.136) exists **only** on `cap_free_all_eight_parseable` and has not been repeated on the primary population; DeepSeek's matched stratum is refused at n=38 with 4 incorrect. |

## D. Deployable-peer comparison

**D1. At one extra generation, `B1` is not beaten by a cheap peer.**

| | |
|---|---|
| Models | all three · Population `full_population` · unit prompt · timing post-8 for `B1`, post-8+1 for the peer |
| Artifact | `results/peer_cost_ladder/peer_cost_ladder_report.md` §4 |
| Value | six single-peer deployable comparisons: 4 ties, 1 RMD win (deepseek vs deepseek_llama peer, −0.0341 [−0.0576, −0.0108]), 1 peer win (deepseek_llama vs qwen peer, +0.0544 [+0.0240, +0.0838]); raw p = 0.170 / 0.062 / 0.008 / 0.408 / 0.878 / 0.000 |
| Checks completed | verdicts reproduced from `cheapest_peer_verdict`; raw p-values read from `contrasts` |
| Limitations | **This family is post-hoc**, selected after seeing the ladder, and `controls/peer_cost_ladder.py` computes no multiplicity correction. Report the raw intervals and say so. No peer rung is exactly cost-matched to `B1`, so the supported claim is "wins at zero additional generations", never "beats peer uncertainty". |

## E. Negative and bounding results the paper keeps

**E1. Geometry does not predict the gain from more samples.**

| | |
|---|---|
| Models | all three · Population `cap_free_valid_plurality` · unit prompt · timing post-1 |
| Artifact | `results/allocation_precheck/allocation_precheck_results.json` |
| Value | gate **fails on 2 of 3** (qwen R² −0.0037, deepseek R² −0.0065, deepseek_llama PASS at R² +0.0005168); Spearman with gain −0.042 / −0.057 / −0.074 against +0.51 / +0.24 / +0.37 with the pass rate |
| Checks completed | pre-declared gate; verdicts read from `gate.per_model`; not a power artifact — at one trace the feature still holds AUROC 0.790 / 0.674 / 0.688 |
| Limitations | Not run on `full_population`; every number in this row and in the matching README paragraph is `cap_free_valid_plurality` (n = 392/393/408), including the "0.806 / 0.686 / 0.709 at eight" comparison, whose primary-population values are 0.819 / 0.714 / 0.712. Reported as a negative, so the population gap is the low-risk one — but the verdict must be re-read after the re-run, not assumed. |

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
| Label efficiency: geometry leads a probe by −0.033 AURC at 50 labels | Cut 2026-08-22. The −0.033 **is** the pooling-matched comparison (against `probe_token_tail_q20`), but its interval spans zero: −0.033 [−0.044, +0.024], sign p=0.109. It decomposes into ≈−0.011 supervision + −0.018 decision-function form, so only a claim about the one-class inductive bias must quote −0.011. | `EXPERIMENT_LOG.md:417`, `:3309`, `:3314`; `results/label_efficiency_token_pooling/label_efficiency_report.md` |
| Entropy-localized RMD beats full-trace and a random control (as a general claim) | Pre-registered gate **failed** on DeepSeek-R1-Distill-Qwen-7B on 2026-07-29 (+0.004 p=0.674; +0.001 p=0.924). Demoted to Qwen-specific by the decision rule fixed in advance. The Llama decomposition collect was cancelled *at that time* but later ran, so `rmd_random_q20` **does** exist for Llama; what was never re-run there is the gate itself. | `EXPERIMENT_LOG.md:5186`; `results/deepseek_llama_bestofn_full/math500/math500_prompt_decomposition_results.json` |
| Aggregator novelty (leg a) | Withdrawn; the contribution is the evaluation, not a new geometry statistic. | `EXPERIMENT_LOG.md:3177` |
| Peer pass rates "absorb most of the increment", control withdrawn | The withdrawal was itself wrong and is reversed — see B3. | this table, `EXPERIMENT_LOG.md` 2026-09-07 |

## G. Evidence gates still open

| Gate | State | What closes it |
|---|---|---|
| Registered full-refit stability sweep | **partially closed, not closed** — 3 of 4 registered seeds, 2 of 3 models, peer step skipped at every seed | qwen at seeds 101/202; the peer step at each seed; seed 303. ~2.5 h at `peak_gb=8` |
| Registered tail-window sensitivity | **not run** — `results/rmd_window_sensitivity/` does not exist | the registered command at `EXPERIMENT_LOG.md:367`. An in-memory replication during the 2026-09-06 audit is not this artifact and must not be cited |
| Five controls on the primary population | not run | `docs/research/2026-09-07-primary-population-controls.md` |
| Shared outer prompt partition (B9) | open, and not optional before submission | all three models share one partition; noted in the 1a/1b limitations |
