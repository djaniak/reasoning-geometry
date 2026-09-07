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

| Control | Models | Population | Note |
|---|---|---|---|
| MATH-500 annotated level + budget-edge difficulty (`difficulty_control`) | qwen, deepseek | includes `full_population` | DeepSeek-R1-Distill-Llama-8B never run |
| Cross-model peer pass rate (`peer_difficulty_control`) | all three | `cap_free_valid_plurality` | not run on the primary population |
| Length residualization | qwen, deepseek | — | predates the Llama collect |
| DeepConf (arXiv:2508.15260), three forms, all four statistics | deepseek, deepseek_llama | `cap_free_valid_plurality` | **structurally impossible on Qwen** |
| Vote proxy answering Orgad et al. (arXiv:2410.02707) | all three | `cap_free_valid_plurality` | not run on the primary population |

The DeepConf gap is not a scheduling gap. The exact statistic needs cached token
IDs and `data/qwen_bestofn_full` stores no token arrays, so it *can never* run on
Qwen — the model where the increment is strongest. That is a permanent limit of
this control, not a missing run.

Inside the stratum where the eight siblings agree unanimously and self-consistency
is silent, geometry scores AUROC 0.71–0.83. That stratum is most of the data: at
8 samples on MATH-500, **70% / 89% / 52% of prompts are unanimous** (of the 392 /
393 / 408 prompts in `cap_free_valid_plurality`, not of the 500 in the primary
population), so every answer-distribution statistic is constant by construction on
most prompts. That is a limit on self-consistency baselines at this sample count,
not a property of this feature.

The vote-proxy control also holds across all agreement levels, not just that
stratum (2026-08-10). On the primary population, adding the full
answer-distribution entropy to `B0` buys nothing (−0.0003 / +0.0001 / +0.0016,
all p > 0.05), and `rmd_tail_q20` still adds on top of it on all three models,
Holm-corrected over the family of two contrasts × three models pre-declared at
`EXPERIMENT_LOG.md:2976` (Holm p 0.020 / 0.032 / 0.032). The bootstrap resolves
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
it is reported rather than withdrawn. Its own pre-declared stop rule — call the
increment substantially a difficulty proxy if two or more models have an interval
overlapping zero — did **not** trigger: only DeepSeek-R1-Distill-Qwen-7B overlaps
(1 of 3). But the two readings of the same run point different ways and both must
be given: on the raw intervals the increment survives on two of three models, while
under Holm over that pre-declared family of three only DeepSeek-R1-Distill-Llama-8B
stays significant (Holm p 0.012 against 0.072 and 0.544). Absorption is 81% / 99% /
78%. On DeepSeek-R1-Distill-Qwen-7B the increment is **eliminated**, not attenuated:
−0.0004 [−0.0016, +0.0005] p=0.544, against a remaining headroom of 0.0045, and the
control's own report warns that a small delta against no headroom is no evidence at
all. The supported summary is: attenuated but present on two models, gone on the
third. Two other models' pass rates are not available
at decision time, so this never competes with the headline; that the measure is
gold-derived is not a defect in a confound control, exactly as MATH-500's
human-annotated level is gold-derived.

**It does not extend to sample allocation (2026-08-10).** *(Every number in this
paragraph is on `cap_free_valid_plurality`, n = 392 / 393 / 408 — the allocation
precheck has not been run on the primary population. It is reported as a negative,
so the population gap understates rather than flatters, but the verdict must be
re-read after that run, not assumed.)* A pre-declared gate
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
samples. Untouched: ranking them by difficulty, for abstention or routing.**

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
`EXPERIMENT_LOG.md:3089`; on the primary population the gap between the two halves
is 0.010, not 0.074.) The Llama short stratum matched to Qwen on window median and
base accuracy still shows no tail effect, and that matched-window comparison has
**not** been repeated on the primary population. Every stratified contrast here is
exploratory and unadjusted (`:3187`). The split follows reasoning distillation, on one
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
