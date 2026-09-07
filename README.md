# reasoning-geometry-probe

Probing hidden-state geometry in math reasoning.

The core question: when a language model generates a reasoning trace, does the *geometry* of its hidden states tell us something about correctness that token-level entropy misses? We fit a reference manifold (PCA + Gaussian) on hidden states from correct traces, then measure Mahalanobis distance for each new trace. The hypothesis is that incorrect reasoning deviates from this manifold even when the output distribution looks similar token-by-token.

Current evidence is the Best-of-8 MATH-500 runs on three models. Historical
greedy, transfer, temperature, and prefix runs remain under `results/` for
provenance but are not current evidence. See [FINDINGS.md](FINDINGS.md) and
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

Several experiment families have been **retired after returning negative or null
results** — Best-of-N geometry reranking, prefix abort-and-retry filtering,
functional-trajectory encoding, and the PCA-dimension sweep. They are not
pending work; the questions are answered. The verdict and evidence for each is
in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) under *2026-07-25: DVC graph
restructure*, and [notebooks/README.md](notebooks/README.md) indexes which
notebooks are current evidence versus archived diagnostics.

## Models and datasets

| Model | Architecture | Decoding |
|---|---|---|
| Qwen2.5-7B-Instruct | Qwen2.5, 28 layers, hidden dim 3584 | Best-of-8, layer 21, 1024-token budget |
| DeepSeek-R1-Distill-Qwen-7B | Qwen2.5 lineage, reasoning distill | Best-of-8, layer 21, 8192-token budget |
| DeepSeek-R1-Distill-Llama-8B | Llama lineage, reasoning distill | Best-of-8, layer 24, 12288-token budget |
| Llama-3.1-8B-Instruct | Llama lineage, no reasoning distill | **Queued**: Best-of-8, layer 24, 1024-token budget |

Datasets: **MATH-500** (500 problems, 5 difficulty levels, 7 subjects), **GSM8K** test set
(~1300 problems, single greedy trace only), and **OlympiadBench** (`OE_TO_maths_en_COMP`,
501 of 674 rows after the single-numerical-answer filter) — **queued**.

Every current result rests on Best-of-8 MATH-500 alone. GSM8K was collected at one sample
per problem, so it supports no claim that needs siblings, and its greedy pass rate (0.91 /
0.91 / 0.63) leaves too few errors for an abstention curve to rank. The two queued collects
address the two things that scope cannot fix on its own: OlympiadBench is a second prompt
set, and Llama-3.1-8B-Instruct is a second *non-distilled* model, so that "distilled vs not"
stops being confounded with "Qwen vs not". Neither has run; see `params.yaml` and the
`probe_dataset` gate.

## Current result

**Hidden-state geometry adds selective-prediction value on top of a
self-consistency baseline, on three models.** The claim is an *increment*, not a
score for the feature alone. Baseline `B0 = (length, entropy, logprob,
vote_agreement)` aggregated over 8 sibling traces; `B1 = B0 + rmd_tail_q20`,
the **mean** per-token relative Mahalanobis distance over the **final 20% of
trace tokens** (`prompt_decomposition.py::score_localized_rmd`; the `q20` in
the name is the size of the tail window, not a quantile of the distances).
Out-of-fold logistic readouts, prompt-clustered paired bootstrap, 1000 draws.

On the primary `full_population` estimand -- correctness available at the stated
generation budget over all 500 prompts -- in **AURC** (area under the
risk-coverage curve, lower is better):

| Model | n | `B1 − B0` (frozen seed 42) | across-refit range |
|---|---:|---|---|
| Qwen2.5-7B-Instruct | 500 | −0.0520 [−0.0845, −0.0218] | not yet refitted |
| DeepSeek-R1-Distill-Qwen-7B | 500 | −0.0284 [−0.0526, −0.0048] | −0.0343 … −0.0240 (mean −0.0289) |
| DeepSeek-R1-Distill-Llama-8B | 500 | −0.0469 [−0.0743, −0.0162] | −0.0469 … −0.0294 (mean −0.0372) |

The left column is the frozen primary fit and stays the headline. The right column
is the partial refit sweep (seeds 42/101/202, two of three models, `--skip_peer`).
Under the decision rule registered on 2026-08-22, both quantities are sign-stable
with a spread narrower than their bootstrap interval, so the claim stands and both
are reported. Read the frozen Llama value knowing it is the **largest** of its three
refits, not the middle one: the refit mean is −0.0372, about a quarter smaller. The
same caution applies to the probe's DeepSeek macro AUROC quoted below — the frozen
0.582 is the **minimum** of its three refits, whose mean is 0.644. Qwen has no refit
at a new partition, and no seed has run the peer step, so the refit gate is
**partially closed, not closed**. (Qwen has a seed-42 refit, which is the
frozen-partition reproduction check and reproduces the headline bit-for-bit; what
it lacks is a refit at any *new* partition.)

Controls it survives, **with the coverage each one actually has** — no control
below is a clean three-model replication, and the differences matter:

| Control | Models | Population | Outcome on the primary population |
|---|---|---|---|
| MATH-500 annotated level + budget-edge difficulty (`difficulty_control`) | qwen, deepseek | includes `full_population` | survives; **DeepSeek-R1-Distill-Llama-8B never run** |
| Cross-model peer pass rate (`peer_difficulty_control`) | all three | `full_population` (2026-09-07) + `cap_free_valid_plurality` | **stop rule triggers** (2 of 3 overlap zero); none of the three survives Holm; see below |
| Length residualization | qwen, deepseek | — | survives; **DeepSeek-R1-Distill-Llama-8B never run** |
| DeepConf (arXiv:2508.15260), three forms, all four statistics | deepseek, deepseek_llama | `full_population` (2026-09-07) + `cap_free_valid_plurality` | survives — all four DeepConf statistics sit at chance (AUROC 0.485–0.542); **structurally impossible on Qwen** |
| Vote proxy answering Orgad et al. (arXiv:2410.02707) | all three | `full_population` (2026-09-07) + `cap_free_valid_plurality` | survives — unanimous-stratum AUROC 0.827 / 0.717 / 0.739 on n = 302 / 424 / 253 |

The DeepConf gap is not a scheduling gap. The exact statistic needs cached token
IDs and `data/qwen_bestofn_full` stores no token arrays, so it *can never* run on
Qwen — the model where the increment is strongest. That is a permanent limit of
this control, not a missing run.

Inside the stratum where the eight siblings agree unanimously and self-consistency
is silent, geometry scores AUROC **0.72–0.83** on the primary population (0.827
Qwen / 0.717 DeepSeek-Qwen / 0.739 Llama, on n = 302 / 424 / 253). At 8 samples on
MATH-500, **60% / 85% / 51% of the 500 prompts are unanimous**, so every
answer-distribution statistic is constant by construction on most prompts. That is
a limit on self-consistency baselines at this sample count, not a property of this
feature. (The corresponding cap-free figures are 70% / 89% / 52% of 392 / 393 / 408,
with AUROC 0.829 / 0.714 / 0.756 — the same picture on a cleaner subset.)

The vote-proxy control also holds across all agreement levels, not just that
stratum (2026-08-10). On the primary population, adding the full
answer-distribution entropy to `B0` buys nothing (−0.0003 / +0.0001 / +0.0016,
all p > 0.05), and `rmd_tail_q20` still adds on top of it on all three models,
Holm-corrected over the family of two contrasts × three models pre-declared at
`EXPERIMENT_LOG.md:3224` (Holm p 0.020 / 0.032 / 0.032). The bootstrap resolves
p to 1/1000, so read those as clearing the threshold, not as clearing it by a
wide margin.

**A deployable peer is a genuine competitor, not a mechanism control.** A peer's
agreement with the target answer needs no gold label, but it costs extra
generations; `B1` uses states from the target's existing eight traces. At the
cheapest deployable rung, the six target-peer comparisons give four ties, one
RMD win and one peer win, read off the raw 95% intervals (`B1` over the peer:
p = 0.170 / 0.062 / 0.008 / 0.408 / 0.878 / 0.000). These six were selected after
seeing the ladder, are not a pre-declared family, and `controls/peer_cost_ladder.py`
computes no multiplicity correction over them — so no corrected claim is made here.
No peer rung is exactly cost-matched to `B1`, so the supported advantage is zero
additional generations, not superiority to peer uncertainty.

The cross-model difficulty control is a **control, not a competing baseline**, and
it is reported rather than withdrawn — but on the primary population it comes out
against us, and that is now the headline reading of it.

Run on `full_population` on 2026-09-07, its pre-declared stop rule — call the
increment substantially a difficulty proxy if two or more models have an interval
overlapping zero — **triggers**. Two of three models overlap:

| model | `B1 − B0` given two peers' pass rates | absorbed | Holm p |
|---|---|---:|---:|
| Qwen2.5-7B-Instruct | −0.0067 [−0.0164, −0.0002] p=0.042 | 87% | 0.126 |
| DeepSeek-R1-Distill-Qwen-7B | −0.0006 [−0.0016, +0.0003] p=0.194 | 98% | 0.388 |
| DeepSeek-R1-Distill-Llama-8B | −0.0025 [−0.0069, +0.0026] p=0.280 | 95% | 0.388 |

Under Holm over that pre-declared family of three, **none of the three survives**.
The share of remaining headroom the tail removes is 20% / 8% / 3%. The pre-declared
consequence, written before the run, applies: the increment is reported as
substantially a prompt-difficulty proxy, and the "geometry adds beyond output-side
confidence" framing narrows accordingly.

Two things this does **not** mean. `B0 + peer` is not deployable — you cannot run
two other eight-sample models to decide whether to trust this one — so nothing here
competes with the headline as a method, and the headline increment itself is
unchanged. And an earlier `cap_free_valid_plurality` run of the same control did
*not* trigger the rule (1 of 3 overlapping, 81% / 99% / 78% absorbed). Both runs
are kept; the primary-population one governs, per the reporting rule adopted at
`EXPERIMENT_LOG.md:1132`. What narrows is the mechanism claim, not the measurement:
most of what `rmd_tail_q20` contributes over `B0` is prompt difficulty, which an
exogenous difficulty signal captures too. Two other models' pass rates are not available
at decision time, so this never competes with the headline; that the measure is
gold-derived is not a defect in a confound control, exactly as MATH-500's
human-annotated level is gold-derived.

**Sample allocation: the pre-declared gate passes on the primary population and
fails on the cap-free one, and the paper should claim neither direction.** Run on
`full_population` on 2026-09-07, the gate passes 3 of 3 (R² for geometry alone
against a cross-fitted constant: +0.019 / +0.137 / +0.016, all intervals above
zero). On `cap_free_valid_plurality` the same gate fails 2 of 3 (−0.0037 / −0.0065
/ +0.0005). The earlier flat claim that geometry "does not extend to sample
allocation" was a cap-free result and is **withdrawn**.

The pass should not be read as an allocation result either. The gate's second leg —
geometry adds over output-alone in out-of-fold Spearman — is decided on a **median
over eight stage-1 draws**, and the brackets this precheck prints are `median [min,
max]` across those draws, **not** bootstrap confidence intervals
(`applications/allocation_precheck.py:553`). The medians are small on every model:
+0.018 [−0.002, +0.039]; +0.006 [−0.016, +0.024]; +0.012 [+0.000, +0.065]. Two of
the three draw-ranges cross zero and the third is strictly positive by 0.0005, but
with eight draws and no interval estimate none of that is a significance claim
either way.

Meanwhile the diagnostic the precheck exists to catch fires on **2 of 3** models
(`difficulty_not_gain` is true for Qwen and Llama, false for DeepSeek), where on the
cap-free population it fired on all three. The raw correlation between geometry and
the gain is negative on all three (−0.064 / −0.293 / −0.060) while its correlation
with the pass rate is +0.63 / +0.49 / +0.44. The report's own words: "geometry reads
difficulty but not marginal gain" — and DeepSeek clears the flag only because its
anti-correlation with gain is *strong*, at −0.293. A fitted readout can exploit an
inverse relationship, which is why the two legs disagree. What the paper can say is
that this is unresolved and population-dependent, and that writing `allocation.py`
is now a licensed next step rather than a ruled-out one.

The rest of this paragraph is the original `cap_free_valid_plurality` analysis
(n = 392 / 393 / 408), retained because it is what the pre-declared gate was
registered on. A pre-declared gate
asked whether single-trace geometry predicts the *gain from buying more samples*,
`g(p) = a(p,8) − a(p,1)`, with `a(p,k)` the expected plurality-vote correctness
over all `C(8,k)` sibling subsets. It does not: geometry ranks the gain backwards
(Spearman −0.042 / −0.057 / −0.074) while correlating +0.51 / +0.24 / +0.37 with
the pass rate, and the gate fails on 2 of 3 models, with the single pass sitting
at R² = +0.0005. That is the expected shape — gain is non-monotone in difficulty, and
a prompt at 0/8 and one at 8/8 both gain nothing. It is not a sample-size problem:
at one trace the feature holds AUROC 0.790 / 0.674 / 0.688 against its 0.806 /
0.686 / 0.709 at eight (both on `cap_free_valid_plurality`; the primary-population
eight-trace values are 0.819 / 0.714 / 0.712). **Ruled out: ranking prompts by predicted gain from more
samples. Untouched: ranking them by difficulty, for abstention or routing.** *(That
"ruled out" is the cap-free verdict and no longer holds on the primary population —
see above. Neither direction is established.)*

**The tail window is a Qwen-specific localization, not part of the method.**
The untailed whole-trace mean `rmd_full` — Vazhentsev et al.'s ATRMD — recovers
essentially the whole increment by itself on both reasoning-distilled models
(101% and 95% of the tail's point estimate). On the primary
population no difference between them is detectable at this n — `rmd_full` −0.0287
against the tail's −0.0284 on DeepSeek-R1-Distill-Qwen-7B, −0.0445 against −0.0469
on DeepSeek-R1-Distill-Llama-8B. The tail over `rmd_full` is −0.0030 [−0.0109,
+0.0035] p=0.436 and −0.0041 [−0.0118, +0.0035] p=0.320: those intervals exclude
any advantage larger than about 0.012, against an increment of 0.03–0.05. That is
a bound on how much the tail can be adding, not a demonstration that it adds
exactly nothing. Only on Qwen2.5-7B-Instruct does the tail carry the result.
Window size does not explain the split, but not in the way an earlier draft of
this page claimed. On the primary population the tail's advantage inside Qwen is
**non-monotone** in window size — terciles −0.0448 / −0.1083 / −0.0491, and
−0.0563 below the median window against −0.0662 above it. The advantage neither
shrinks with window size, which is what the window hypothesis predicts, nor grows
with it: it peaks in the middle. (The steeper −0.042 / −0.116 gradient quoted
previously is the `cap_free_all_eight_parseable` stratification at
`EXPERIMENT_LOG.md:3337`; on the primary population the gap between the two halves
is 0.010, not 0.074.) The Llama short stratum matched to Qwen on window median and
base accuracy still shows no tail effect, and that matched-window comparison has
**not** been repeated on the primary population. Every stratified contrast here is
exploratory and unadjusted (`:3435`).

**The 20% cutoff is not load-bearing (registered sensitivity analysis, run
2026-09-07).** The rule registered on 2026-08-22, before q10 or q50 were computed,
required q10, q20 and q50 each to improve AURC over `B0` with a 95% interval below
zero on every checkpoint. It **passes** on all three models, so the paper may state
that the result is not specific to this exact cutoff — noting that Qwen's q50 clears
by 0.0012 (−0.0304 [−0.0596, −0.0012]), which by this repo's own convention for
near-threshold results is a borderline pass rather than a clean one. Three disclosures come with
it, and the registration forbids acting on any of them:

| model | `q10` | frozen `q20` | `q50` | `rmd_random_q20` |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | −0.0559 | −0.0520 | −0.0304 | −0.0184 (n.s.) |
| DeepSeek-R1-Distill-Qwen-7B | −0.0342 | −0.0284 | −0.0376 | **−0.0295** |
| DeepSeek-R1-Distill-Llama-8B | −0.0572 | −0.0469 | −0.0477 | **−0.0444** |

First, `q10` beats the frozen `q20` on all three models. The registration says 20%
is not to be called optimal and the frozen feature is not to be replaced, so it is
not — but a reader will notice, and the paper should say it rather than be caught
with it. Second, and more damaging: on both distilled models a **random** 20% window lands
where the tail does (−0.0295 against −0.0284; −0.0444 against −0.0469), while on
Qwen it does not (−0.0184 against −0.0520). No paired contrast between the two was
computed, so this compares point estimates rather than testing a difference — the
supported statement is that the tail shows no measured advantage over a same-sized
random window on the distilled models, and a large one on Qwen. Read that as the
absence of evidence for tail-specific localization off Qwen, not as proof that
window position carries nothing. Third, `rmd_high_entropy_q20` is weak
everywhere (−0.0148 n.s. / −0.0052 n.s. / −0.0274), which is consistent with the
entropy-localization gate that failed in 2026-07-29.
Artifact: `results/rmd_window_sensitivity/`. The split follows reasoning distillation, on one
non-distilled model, with trace style, budget and base accuracy still collinear
with it. A third non-distilled model is the test that would settle it.

**Pooled trace discrimination is not sibling verification.** A nested
last-token probe reaches pooled AUROC 0.901 / 0.914 / 0.903, but macro
within-prompt AUROC is 0.644 / 0.582 / 0.718 — and on DeepSeek-R1-Distill-Qwen-7B
that 0.582 carries a 95% interval of [0.463, 0.688], which contains chance. RMD
shows the same qualitative gap and goes further: its within-prompt macro on that
model is **0.461, below chance**. Length also loses substantial pooled
discrimination, while entropy and log-probability lose less of theirs — though on
both distilled models all three of those baselines sit at or under 0.50 within
prompts, so "loses less" is a statement about the size of the drop, not about
retaining usable within-prompt signal. The supported claim is that the
prompt-level increment survives; a high pooled trace AUROC alone does not
establish which sibling trace is correct, and on one model the within-prompt
ranking is not established at all. Trace-level numbers here are on the `parseable`
population. See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

*Metric note: AURC and AUACC are affinely related at fixed n and both inherit the
base accuracy, so levels are not comparable across models — only deltas are.
AUROC is the base-rate-free metric and is used wherever a comparison crosses
models.*

## Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

Requires Python 3.11+.

## Running the pipeline

The pipeline is managed with [DVC](https://dvc.org). Each stage collects hidden states or runs analysis for one model × dataset combination.

```bash
# Reproduce the current CPU analysis
CUDA_VISIBLE_DEVICES="" uv run dvc repro --single-item evaluate_prompt_decomposition@0
CUDA_VISIBLE_DEVICES="" uv run dvc repro --single-item evaluate_wave1_experiments@0
CUDA_VISIBLE_DEVICES="" uv run dvc repro --single-item evaluate_abstention_baselines@0

# Inspect the active graph
uv run dvc status
```

The default DAG contains the Qwen baseline, dense-layer/PCA checks,
truncation-budget diagnostics, Qwen Best-of-N decomposition, and Wave-1 CPU
follow-ups. Retired model-family and application stages are not
default dependencies; see the experiment log for their status.

Configuration is in [params.yaml](params.yaml): model names, layer indices, PCA dimension, bootstrap count, etc.

## Scripts

| Script | Purpose |
|---|---|
| `collect_data.py` | Run autoregressive generation, capture hidden states and per-token entropy, save `.npz` files |
| `analyze.py` | Fit reference manifold, extract features, run 5-fold CV logistic regression |
| `probe.py` | Trajectory probe: resample Mahalanobis sequence to fixed length, run functional PCA |
| `best_of_n.py` | Best-of-N reranking using geometry scores |
| `prefix_analysis.py` | Retired early-prefix diagnostic (historical) |
| `prefix_filter.py` | Retired abort/retry diagnostic (historical) |
| `summarize.py` | Aggregate a result profile into `results/SUMMARY.md` (**stale**: the checked-in copy is from 2026-07-29, greedy-era, and does not reflect any Best-of-8 result on this page) |

## Output structure

```
data/
  {model}/{dataset}/      # .npz files with hidden states (DVC-tracked, not in git)
results/
  {model}/{dataset}/      # metrics JSON + plots
  SUMMARY.md              # aggregated results table (stale, 2026-07-29 greedy era)
```

## Features

Each trace is represented by 12 scalar features extracted from the per-token sequence:

- **Entropy-only (5)**: mean, max, std, 90th percentile, fraction above median
- **Mahalanobis-only (7)**: mean, max, std, 90th percentile distance; mean/max distance at high-entropy tokens; entropy–Mahal correlation
- **Combined (12)**: union of the above

The reference manifold is fitted on PCA(128) of hidden states from correct traces only, with a regularized Gaussian covariance.
