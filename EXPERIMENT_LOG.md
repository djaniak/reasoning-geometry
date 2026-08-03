# Experiment Log

This ledger tracks completed evidence, artifact compatibility, and the next
smallest runnable stages. Dates are UTC. DVC stage completion means the output
is recorded in `dvc.lock`; it does not by itself imply that an artifact uses the
latest schema.

## 2026-08-03: Budget-limited noncompletion — capping is a budget shortfall

Supersedes the guard described in the entry below, and answers the question that
entry left open by naming it prematurely: cap hits are **not** non-convergence.

### 1. Stages and parameterization

No DVC stage. One CPU pass over cached artifacts, one GPU continuation run.

```
python sibling_structure.py --model_label deepseek \
    --oof_csv results/deepseek_bestofn_full/math500/math500_prompt_decomposition_oof.csv \
    --data_dir data/deepseek_bestofn_full/math500
python sibling_structure.py --model_label qwen ... --data_dir data/qwen_bestofn_full/math500
python continue_capped.py --data_dir data/deepseek_bestofn_full/math500 \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --n_traces 50 --extra_tokens 8192 --batch_size 6 --seed 42 --num_shards 3 --shard {0,1,2}
```

Artifacts: `results/{deepseek,qwen}_bestofn_full/math500/math500_sibling_structure_{results.json,report.md}`,
`results/deepseek_bestofn_full/math500/math500_continue_capped_{results,traces}.json`.

### 2. The cap guard, corrected

The 2026-08-03 guard below rejected any cap above every observed trace length.
That is the signature of a wrong budget, but equally of a **clean collect**:
DeepSeek-Llama runs at 12288 and need never reach it, so the guard would have
refused a correct cap. Observed lengths are not evidence and no longer decide.

`collect_data.py` stores no run-level budget, so `trace_caps.resolve_cap` now
recovers it from the two records that are authoritative — `dvc.lock` (the collect
command that actually ran) and `dvc.yaml` + `params.yaml` (the declared stage
config) — keyed by the data directory. A caller-supplied cap that contradicts
either raises, and so does disagreement between the two; that is what catches the
original Qwen-at-8192 defect, without the length heuristic. Directories outside
the pipeline resolve to an unvalidated cap that says so: `Cap.provenance` is
written into every report rather than implying the count was checked.

Verified live: Qwen 1024 and DeepSeek 8192 from both records; DeepSeek-Llama
12288 from `params.yaml` alone (pending collect, absent from `dvc.lock`) and
**accepted** despite no trace reaching it. Qwen's corrected abstention run
reproduces the table in the entry below bit-identically apart from the new
provenance field.

### 3. Sibling structure — both models

Counts only, no fitting. This needs lengths and answers rather than tokens, so
unlike the loop study it **is** a two-model result.

| | DeepSeek (8192) | Qwen (1024) |
|---|---|---|
| prompts with >=1 capped sibling | 107/500 | 108/500 |
| prompts with all eight capped | 9 | 2 |
| P(another sibling finishes given a cap) | 0.916 | 0.981 |
| P(a finisher is correct given a cap) | 0.570 | 0.454 |

Finished-sibling accuracy falls monotonically with the number of capped siblings
— DeepSeek 0.785 at zero capped through 0.250 at seven; Qwen 0.657 through 0.333.
Capping tracks prompt difficulty; it is not an independent sampling accident.

The control that gives "borderline" content: the **longest finishing sibling uses
a median 88% of the budget at affected prompts against 35% at unaffected ones**
(Qwen: 93% against 52%). Prompts that cap are prompts already pressed against the
cap. Regime split among affected prompts (definitions in `sibling_structure.py`):

| regime | DeepSeek | Qwen |
|---|---|---|
| prompt-limited (>=5 of 8 capped) | 34 (32%) | 27 (25%) |
| budget-borderline (longest finisher >=90% of budget) | 31 (29%) | 53 (49%) |
| trajectory-limited (a sibling finished correctly) | 29 (27%) | 15 (14%) |
| unresolved | 13 (12%) | 13 (12%) |

### 4. Continuation — what capped traces were actually doing

50 capped DeepSeek traces sampled at seed 42 from the 370 that were not already
looping (4 excluded), resumed from prompt + their own 8192 stored tokens — which
round-trip exactly through `convert_tokens_to_ids` — and run to 16384 at the
collection temperature 0.6. Intervals are Wilson 95%.

| outcome | n | share |
|---|---|---|
| completed, correct | 16 | 0.32 [0.21, 0.46] |
| completed, incorrect | 18 | 0.36 |
| still unfinished at 16384 | 13 | 0.26 [0.16, 0.40] |
| degenerate loop | 3 | 0.06 [0.02, 0.16] |

**70% [0.56, 0.81] terminate given 8192 more tokens, and 45.7% [0.31, 0.62] of
those are correct** — against the 5.6% accuracy these same traces are scored at
when judged truncated. Extra tokens needed by the finishers: median 2846,
mean 3386, p90 7014; 21 of 35 fit in +4096.

Zero traces answered and then kept going *in the continuation*, but at population
scale **38 of 374 capped traces (10.2%) already carried a parseable answer** when
the budget ran out — those were never budget-limited, only bad at stopping.

The gate for entering geometry — two reproducible regimes — is met, by the
section 3 labels:

| regime | n | correct | incorrect | unfinished | loop |
|---|---|---|---|---|---|
| prompt-limited | 31 | 7 | 12 | 10 | 2 |
| budget-borderline | 11 | 5 | 5 | 1 | 0 |
| trajectory-limited | 5 | 4 | 0 | 1 | 0 |
| unresolved | 3 | 0 | 1 | 1 | 1 |

Termination is 0.61 [0.44, 0.76] under prompt-limited against 0.84 [0.62, 0.95]
otherwise. The direction is consistent and the cells are tiny; treat the ordering
as real and the magnitudes as unestimated.

### 5. Claims ruled in and out

- **Ruled in.** A cap hit is predominantly a **budget shortfall**, not a failure
  to converge. The name "non-convergence" was premature and is retired.
- **Ruled in.** Capping is prompt-structured, not sample-structured: affected
  prompts sit at the budget edge and their finishers are less accurate. Two
  models.
- **Ruled out.** That most capped traces are stuck. 6% degenerate on continuation,
  1% at the cap itself (entry below) — the same order, still not the story.
- **Not established.** Any prompt-level accuracy gain from a larger budget. Only
  the sampled capped traces were continued, not their siblings, so nothing here
  measures a plurality vote. The trace-level flip rate is 0.32.
- **Costing, for the budget-engineering framing.** Continuing only the capped
  traces to 16384 costs ~4.8k tokens each in expectation, ~1.8M over the 370
  coherent ones, **+14.4% on a 12.4M-token run**, to flip ~32% of them.

### 6. Limitations and the next dependent stage

Continuation cannot reproduce the original sampling stream — the RNG state is
gone — so this measures what the model does next from that prefix, not what it
did on the day. n=50 is a marginal sample of capped *traces*, so it is dominated
by prompt-limited prompts (31 of 50), which is the honest population weighting
but leaves the other cells at n=3-11.

Next stage, now gated open: geometry on fixed prefixes (512/1024/2048 tokens),
comparing **capped against completed siblings of the same prompt**, which removes
prompt difficulty without another global correctness score. Plots first — path
efficiency, recurrence, velocity, distance to a successful sibling, sibling
dispersion, with entropy and log-probability controls. The target is the regime
label from section 3, not correctness.

Not started, and deliberately: held-out-sibling forecasting (the backup branch),
and any continuation of Qwen, which stores no tokens and so cannot be resumed.

## 2026-08-03: Cap-population fix, and loop precursors KILLED (DeepSeek only)

### 1. Stages and parameterization

No DVC stage. Two ad-hoc, CPU-only passes over cached artifacts:

```
python incremental_abstention.py --model_label qwen --max_new_tokens 1024 --layer 21 \
    --oof_csv results/qwen_bestofn_full/math500/math500_prompt_decomposition_oof.csv
python loop_precursors.py --data_dir data/deepseek_bestofn_full/math500 \
    --max_new_tokens 8192 --ngram 8 --window 200 --threshold 0.5 --seed 42
```

### 2. The n=498 cap-population bug

A previously reported Qwen "cap-free valid" population of n=498 was impossible
against the audit finding of 108/500 Qwen prompts with a capped sibling. Cause:
the ad-hoc `incremental_abstention.py` run was passed **DeepSeek's
`--max_new_tokens 8192`** for traces collected at **Qwen's 1024**
(`params.yaml:69-70`). No trace can reach 8192, so every cap count was zero and
`cap_free_valid_plurality` silently equalled `valid_plurality`. Not the
`max_new_tokens is None` path — that was a separate latent defect, now also closed.

`trace_caps.resolve_cap` now rejects a missing cap; `truncation_report`,
`answer_cluster_eligibility`, and `prompt_accounting` all route through it.
(Its first version also rejected a cap above every observed length. That was
wrong and was replaced on the same day — see the 2026-08-03 budget-limited
noncompletion entry.) A repo-wide sweep found no other
result file carrying a mismatched cap, and DeepSeek's correctly-capped run
reproduces bit-identically.

**Corrected Qwen populations** (MATH-500, layer 21, 1,000-draw bootstrap):

| population | n | prompts w/ ≥1 capped sibling | B1−B0 AUACC |
|---|---|---|---|
| full_population | 500 | 108 | 0.052 [0.019, 0.083] p=0.002 |
| valid_plurality | 498 | 106 | 0.052 [0.018, 0.083] p=0.002 |
| cap_free_valid_plurality | 392 | 0 | 0.059 [0.023, 0.096] p=0.002 |
| all_eight_parseable | 392 | 1 | 0.059 [0.021, 0.099] p=0.004 |

Automatic failures: 2. **The tail-RMD increment survives the correction** and is
slightly larger on the cap-free population. The 108 figure now matches the audit.

### 3. Loop precursors: the premise is false

Scope: DeepSeek only. `data/qwen_bestofn_full` stores no `tokens_*` and no
`generated_text`, so no token- or text-level analysis can run on Qwen. **Nothing
in this section is a cross-model replication.**

Population: 4,000 traces, 500 prompts, cap 8192. 374 traces capped (9.3%),
consuming **24.7% of the 12.4M generated tokens**, at accuracy **0.056**.
107/500 prompts have ≥1 capped sibling; 9 have all eight capped.

An 8-gram prefix-novelty detector (200-token window, 0.5 threshold) flags 80% of
capped traces at a median onset of **44.9% of budget** — which looked like a
large early-stop prize. It is not:

- it also flags **21.1% of uncapped traces**, and those are only mildly less
  accurate than unflagged ones (0.712 vs 0.782), so the flag is not reading a
  pathology;
- reading 21 capped traces **at the onset position** (7 each from early-onset,
  late-onset, and no-onset strata), only **2 are degenerate loops**. The other 19
  are coherent, unfinished reasoning — symmetric case analysis, re-verification of
  a shoelace computation, Asymptote code re-reading, second-approach checks. The
  detector fires on structural repetition intrinsic to mathematical reasoning.

Quantified with a tail-periodicity statistic (best repeating period in the last
500 tokens, calibrated on the two hand-labelled loops, which scored 1.000 and
0.423 against a 0.188 maximum for the other nineteen):

| threshold | degenerate share of capped | of uncapped |
|---|---|---|
| ≥0.20 | 0.070 (26) | 0.043 (157) |
| ≥0.30 | 0.011 (4) | 0.012 (44) |
| ≥0.50 | 0.005 (2) | 0.005 (17) |

**Degenerate looping occurs at ~1% in both populations and is therefore not what
causes capping.** The result is insensitive to the threshold across 0.20–0.90.

### 4. Claims ruled out

- **Ruled out:** capping in DeepSeek-R1-Distill-Qwen-7B on MATH-500 is a
  degenerate-loop phenomenon. It is not. Capped traces are overwhelmingly hard
  problems the model does not finish in 8192 tokens.
- **Ruled out:** "detect the loop, early-stop, recover the compute". There is no
  loop to detect in ~99% of capped traces.
- **Not run, by the pre-registered kill criterion:** the geometric precursor test
  (L2) and the tokens-saved/answers-lost curve (L3). Both target loop onset, and
  the object does not exist at population scale.

### 5. Limitations and next dependent stage

The hand taxonomy is 21 traces, stratified rather than random; the population
periodicity statistic is what carries the claim, not the reading. The periodicity
threshold rests on two hand-labelled positives, which is why §3 reports a
threshold sweep instead of a single number.

The 24.7% of budget spent on 5.6%-accurate capped traces is real and still
unrecovered. Recovering it means predicting **non-convergence**, not detecting a
loop — a different and harder target, and a scope change. Gate it behind an
explicit decision rather than drifting into it.

Artifacts: `loop_precursors.py`, `trace_caps.py`, `tests/test_loop_precursors.py`,
`tests/test_trace_caps.py`. Run outputs are scratch-only and not checked in.

## 2026-07-31: Supervised probe ceiling + length residualization (BOTH models)

### 1. Stages and parameterization

```
evaluate_prompt_decomposition@0  (qwen)      evaluate_prompt_decomposition@1  (deepseek)
evaluate_wave1_experiments@0     (qwen)      evaluate_wave1_experiments@1     (deepseek)
```

Both models: MATH-500 Best-of-8, 500 prompts, layers 7/14/21, 5 prompt folds,
1,000-draw prompt-cluster bootstrap. New params in `params.yaml`:
`prompt_decomposition.hidden_probe_regions: "full,high_entropy_q20,tail_q20"`
(`random_q20` omitted — it is a control for localization claims and the probe
makes none). Both stages pinned to `CUDA_VISIBLE_DEVICES=""`; CPU-only.

Two additions:

- **`probe_hidden_*`** — cross-fitted supervised LDA (`solver=lsqr`,
  `shrinkage=auto`) on PCA-projected region means, pooled labels, trained on
  parseable training traces only (unparsed traces are auto-labeled incorrect
  upstream, so training on them would let the probe win by detecting
  truncation). 45 fits per model = 5 folds × 3 layers × 3 regions. Distinct from
  `contrast_*`, which is prompt-centered and targets the within-prompt regime.
- **E1R** (`length_residualized_abstention`) — E1 abstention metrics with
  `length_score` partialled out of every scorer in rank space, refit inside each
  bootstrap draw. Reference is an uninformative scorer (expected AURC = base
  accuracy).

**Both exploratory, not pre-registered** — recorded as `prespecified: false` in
the emitted JSON.

### 2. Artifacts and schema

- `results/{qwen,deepseek}_bestofn_full/math500/math500_prompt_decomposition_results.json`
  — new `settings.hidden_state_probe` provenance block; new
  `layers.<L>.parseable_only.length_collapse` and `hidden_probe_paired_deltas`.
- `results/{qwen,deepseek}_bestofn_full/math500/math500_wave1_results.json`
  — new top-level `e1r_length_residualized_abstention`.
- `..._prompt_decomposition_oof.csv` — three new `probe_hidden_*_score` columns
  (37 columns total).
- Tests: `tests/test_hidden_state_probe.py` (10), `tests/test_length_residualization.py` (6).

Regression check: re-running wave1 changed **0 of 10,338** shared scalars in both
models; the E1R block is purely additive.

### 3. Point estimates and uncertainty

Raw E1 prompt abstention, AURC at L21 (base accuracy 0.620 Qwen / 0.750 DeepSeek):

| Scorer | Qwen | DeepSeek |
|---|---:|---:|
| `probe_hidden_tail_q20` | 0.853 | 0.904 |
| `rmd_tail_q20` | 0.828 | 0.856 |
| `rmd_high_entropy_q20` | 0.789 | 0.832 |
| `length` | 0.759 | 0.826 |
| `logprob` / `entropy` | 0.666 / 0.660 | 0.788 / 0.788 |

`probe_hidden_tail_q20 − rmd_tail_q20`: +0.025 [+0.002, +0.046] p=0.028 Qwen
(Holm 0.056, does not survive); +0.048 [+0.018, +0.079] p=0.002 DeepSeek
(Holm 0.006, survives).

`rmd_high_entropy_q20 − length` on DeepSeek: +0.005 [−0.011, +0.025] p=0.506 —
not distinguishable from length. `rmd_tail_q20 − length` = +0.030 [+0.014, +0.048].

Length collapse (Spearman vs `length_score`, parseable, L21): `rmd` +0.658 Qwen /
**+0.820** DeepSeek; `probe_hidden_tail_q20` +0.425 / +0.223; `entropy` −0.163 /
**+0.350** (sign flips between models).

E1R, Δ AURC vs an uninformative scorer:

| Scorer | Qwen | DeepSeek |
|---|---|---|
| `probe_hidden_tail_q20` | +0.190 [+0.155, +0.224] | +0.140 [+0.110, +0.168] |
| `rmd_tail_q20` | +0.161 [+0.128, +0.194] | +0.107 [+0.077, +0.135] |
| `rmd_high_entropy_q20` | +0.111 [+0.074, +0.148] | +0.063 [+0.027, +0.096] |
| `logprob` | +0.058 [+0.014, +0.097] | +0.009 [−0.029, +0.046] |
| `entropy` | +0.057 [+0.013, +0.097] | +0.011 [−0.028, +0.047] |

Holm across the 7 scorers within model: all geometry rows p < 0.01 both models;
entropy/logprob Holm 0.016 Qwen but **1.000 DeepSeek**.

E1R probe vs RMD: +0.029 [−0.003, +0.060] p=0.090 Qwen, +0.033 [+0.001, +0.064]
p=0.042 DeepSeek. Holm over 3 comparisons: 0.090 / 0.126 — neither survives. Only
surviving cell is Qwen `probe_hidden_high_entropy_q20` *losing* (−0.062, Holm 0.018).

Negative control (synthetic scorer = length + sub-tie jitter): +0.008 Qwen /
−0.007 DeepSeek, p ≥ 0.82.

### 4. Claims ruled in and out

**Ruled IN:**

- RMD carries substantial between-prompt solvability signal that length cannot
  supply, on both models. Upgrades the §7c "RMD > length + entropy" positive.
- On DeepSeek, `entropy` and `logprob` are the length proxies, not RMD.
- A supervised probe on the same activations does not reliably beat unsupervised
  RMD at length-controlled prompt abstention on either model. Strongest available
  form of the label-light argument (`PAPER_STRATEGY.md` §6 killer experiment).

**Ruled OUT:**

- "RMD collapses to length on reasoning-distilled models." The rho +0.82
  diagnostic does not support this; E1R refutes it. Do not report the Spearman
  table without E1R alongside it.
- "Supervision recovers materially more geometry signal than RMD." True in raw
  E1 on DeepSeek (Holm 0.006) but the advantage is largely reduced length
  dependence — it does not survive length control.

### 5. Limitations and next dependent stage

- Residual retains small rank correlation with length (+0.13 DeepSeek, +0.05
  Qwen): rank-space OLS zeroes Pearson-of-ranks, not Spearman of the residual's
  own ranks. Removal is near-complete, not exact.
- E1R is stricter than incremental value: it shows the orthogonal component ranks
  prompts alone, not that `length + RMD` beats `length`.
- The probe is supervised and is a ceiling/diagnostic, not a deployment
  alternative to RMD.
- Scope is between-prompt abstention AURC. The ~0.84 supervised-probe figure from
  arXiv:2511.14773 is raw trace-correctness AUC and is not contradicted here.
- n=2, both Qwen-lineage. **Next dependent stage:** the `deepseek_llama`
  (Llama-architecture) collect, cancelled by the 2026-07-29 gate, is now the
  binding constraint on every claim above.

## 2026-07-25: DVC graph restructure — retired experiment families

### Status

The active graph was cut from 24 stages to 12 and re-pointed at a
**3-model x 2-dataset preliminary matrix**: `qwen` (Qwen2.5-7B-Instruct, 1,024
tok, L7/14/21), `deepseek` (DeepSeek-R1-Distill-Qwen-7B, 8,192 tok, L7/14/21),
`deepseek_llama` (DeepSeek-R1-Distill-Llama-8B, 12,288 tok, L8/16/24); each with
GSM8K single-sample greedy (limit 500) and MATH-500 Best-of-8 (T=0.6, N=8, limit
500). Single-sample MATH-500 is dropped — the Best-of-8 data supersedes it.

**This is a scope cut, not a data deletion.** Every retired stage's outputs
remain under `results/` and every number below is still reproducible from those
JSONs. What changed is what the default graph, `results/SUMMARY.md`, and the
paper claim as *current evidence*.

Cache-safety constraint applied throughout: whole-matrix entries
(`bestofn_matrix`, `wave1_matrix`, ...) were removed from every foreach `do:`
block's `params:` list, so adding a model row cannot invalidate a finished cell.
Item values still appear in `cmd`, so real changes still trigger reruns.

### Why each family was retired

Three distinct reasons. Only the first is a scientific negative.

**(a) Negative or null result — the experiment answered its question, and the
answer was no.**

| Retired stage | Verdict | Evidence |
|:---|:---|:---|
| `evaluate_prefix_filter`, `collect_prefix_filter` | **Negative.** Abort-and-retry prefix filtering never pays for itself. | 135 cells/model (3 prefix lengths x 3 score kinds x 3 layers x 5 thresholds). **Zero cells with positive token savings** — best is −0.015 (i.e. 1.5% *more* tokens) for both Qwen and DeepSeek. Best pass@1 delta +0.016 (Qwen) / +0.024 (DeepSeek), and DeepSeek's best cells are all `entropy_only`, so geometry contributes nothing. False-abort rate ~0.5 ≈ base rate. |
| `evaluate_prompt_selection`, `evaluate_bestofn_full/_pilot/_concordance` | **Negative, with a structural ceiling.** Geometry does not rerank same-prompt samples. | Qwen MATH-500 N=8: majority vote 0.596 pass@1 (random 0.557, oracle 0.676); all 15 geometry/logprob tie-break variants within ±0.006, **15/15 paired deltas p ≥ 0.248**; RMD rank-weighted voting 0.582–0.584 *underperforms* majority. The ceiling is structural: only 39/500 prompts have a tied top answer at N=8, and only ~10 of those ties contain both a correct and an incorrect option — **~2 points of headroom no tie-breaker can exceed.** Retiring this is closing a question, not abandoning it. |
| `trajectory` (Track A, `fpca_mahal`) | **Negative.** Functional trajectory encoding never beats scalar Mahalanobis summaries. | 4 model/dataset conditions x 3 layers. Best case DeepSeek GSM8K L21 = 0.808, still below scalar Mahal at the same layer (0.831) and best combined (0.835). On Qwen GSM8K it is near chance (0.519–0.538 across all layers) and below the entropy baseline. Sequence representation adds variance faster than signal. |
| `analyze_pca_ablation_runs/_merge` | **Null — and the null is the point.** `pca_dim` is not a tuned knob. | 4 conditions x 3 layers x {32, 128, 512, max}. Combined AUC spread across dims is ~±0.03 and non-monotone in every cell; dim 128 is best or within 0.01 of best in 9/12 cells. Closes the "PCA dim fixed at 128, not swept" limitation rather than leaving it open. |
| `analyze_cross` | **Split verdict, retired as out-of-scope.** Manifold shape partially transfers; decision boundaries do not. | Geometry-only cross-model Mahal retention spans ~82% (DeepSeek GSM8K L7) to ~101% (DeepSeek MATH-500 L14), with late layers retaining more reliably (94–99% at L14–L21). Frozen classifier transfer fails: L7 cross-model clf Mahal AUC 0.351–0.705, and L14/L21 are at or below chance in most cells. Retired because it predates the truncation-bias fix and is a separate paper. |

**(b) Superseded by a stricter protocol — the numbers were confounded, not
wrong-hypothesis.**

| Retired stage | Reason |
|:---|:---|
| `evaluate_selective_prediction` | Superseded by `evaluate_wave1_experiments` E1, which does the same risk–coverage comparison **with prompt-cluster bootstrap CIs** and the length baseline. The old stage reported point estimates only. |
| `evaluate_one_class_sweep` | Ran on **all traces**, so its pooled AUCs carry the length/truncation confound that the 2026-07-18 fix exposed (length alone pools at 0.737 but collapses to 0.478 within-prompt on parseable traces). Its mechanistic conclusion survives and is recorded below; the stage does not. |
| `analyze_subspace` | Contrast-direction analysis is subsumed by the `contrast_*` regions inside `evaluate_prompt_decomposition`, which are OOF cross-fitted and tested against a 1,000-draw shuffle null. |
| `evaluate_application_alignment` | Correlations over 2 models x 3 correlated layers — not enough independent cells to support the claim. Re-derivable from the OOF CSVs if the 3-model matrix gives it more support. |

**(c) Retired inputs — collected under budgets now known to be too short.**

DeepSeek 2,048-token analyses and the `deepseek_temp` sweep. The truncation
audit showed the 2,048 budget censored the incorrect class, so the affected
`results/deepseek/{gsm8k,math500}` and `results/deepseek_temp` artifacts are
provenance only. `data/deepseek/gsm8k_stale_2048` and
`data/deepseek_llama/gsm8k_stale_2048` are the corresponding retired inputs.
`collect_qwen_dense_math500` and its analyze/merge stages remain in the graph
**frozen** — the dense layer sweep is a standing positive result and its cache
must not be disturbed.

### Mechanistic conclusion preserved from the retired one-class sweep

Worth keeping in front of the reader even though the stage is gone, because it
explains *why* RMD rather than raw Mahalanobis is the headline score:

| Model | RMD dim 8 | RMD dim 32 | RMD dim 128 | Raw Ledoit-Wolf dim 128 |
|:---|---:|---:|---:|---:|
| Qwen | 0.717 | 0.762 | 0.772 | 0.379 |
| DeepSeek | 0.867 | 0.870 | 0.869 | 0.225 |
| Llama | 0.750 | 0.778 | 0.781 | 0.396 |
| DeepSeek-Llama | 0.783 | 0.786 | 0.792 | 0.352 |

**Background subtraction is the load-bearing mechanism**, not covariance
estimation: diagonal, empirical-ridge, and Ledoit-Wolf target-only variants
differ by < 0.001 throughout, while target-only raw distance is strongly
*anti*-predictive (0.225–0.396) and RMD is strongly predictive. A universal
rank-1 mechanism is rejected — DeepSeek plateaus near dim 8, Qwen and Llama
keep improving through 64–128. (Pooled all-trace AUCs; length-confounded in
absolute level, but the raw-vs-RMD reversal is far too large to be a length
artifact.)

### Wave-1 mechanism follow-ups: all four negative

Recorded here because they are the newest negatives and are easy to misread as
supporting results. Qwen MATH-500 Best-of-8, prompt-cluster bootstrap, 1,000
draws (`results/qwen_bestofn_full/math500/math500_wave1_results.json`):

- **E5 (event-locked RMD) — negative, on the control.** RMD is elevated in the
  window before a high-entropy event at L21 (pre = +0.0147 [+0.0044, +0.0263]),
  but the matched **random-event control is statistically indistinguishable**
  (+0.0135 [+0.0019, +0.0239]). The elevation is a property of the window, not
  of the event. Post-event slopes null at all three layers. Do not cite the
  pre-event CI without its control.
- **E4 (entropy-trajectory autopsy) — negative.** Of four trajectory-shape
  features, three are null vs mean entropy and `mean_peak_position` is
  significantly worse (−0.162 [−0.243, −0.077], p < 0.001). Entropy carries a
  level, not a shape.
- **E6 (log-norm LVE) — negative in all 15 cells** (3 layers x 5 variants),
  every one significantly below plain logprob. Token-order shuffle controls
  match the unshuffled variants, so LVE is order-insensitive.
- **E7 (sibling eligibility) — power audit, not a test.** Of 500 prompts only
  59 have >= 2 correct *and* >= 2 incorrect siblings after censoring (315 have
  >= 2 correct, 242 >= 2 incorrect). This is the structural reason the
  within-prompt selection tests are underpowered, independent of scorer quality.

E1 (prompt abstention) is the one Wave-1 positive: `rmd_tail_q20` AURC 0.828,
acc@50% 0.852 vs length 0.748 / logprob 0.704 / entropy 0.692; paired deltas vs
**length** +0.069 AURC [+0.043, +0.096] and +0.104 acc@50% [+0.064, +0.144],
both p < 0.001. Beating the length baseline is what separates this from
truncation detection.

### Numerical-precision change to `evaluate_prompt_decomposition`

New params `prompt_decomposition.hidden_dtype: float16` and
`compute_dtype: float32`, plumbed through `analyze.set_compute_dtype()` and the
trace loader. Motivation is capacity, not speed: the distill models' traces are
~5x longer than Qwen's, and the reference fit's float64 concatenation is the
binding RAM constraint (~199 GB for DeepSeek MATH-500 alone). float16 storage is
lossless here because hidden states come from a **bf16** forward pass (8-bit
mantissa into a 10-bit mantissa; max observed |value| 2,512 vs the 65,504
overflow limit).

Verified on real Qwen L21 Best-of-8 data (8 batches, 400 traces): raw and RMD
per-trace scores both **Spearman 1.00000000 / Pearson 1.00000000**, max abs diff
1e-6; pooled AUC float64 0.918138 vs float16/float32 0.918138 (**delta
0.000000**). This invalidates the `evaluate_prompt_decomposition@0` (Qwen) cache
entry by command hash even though the outputs are numerically identical.

### Notebook audit accompanying the cut

`notebooks/README.md` is the index. Every notebook's first cell now states its
status, its inputs, and its bottom line, so a negative cannot be mistaken for a
pending experiment. Finished diagnostics moved to `notebooks/archive/`; the
top-level directory holds current evidence only.

- **Current (`notebooks/`):** `11_prompt_geometry_core_experiments`
  (within-prompt, primary), `12_wave1_abstention` (**new** — between-prompt; E1
  is the headline positive and previously had no notebook),
  `01_main_effect_overview` (pooled legacy view, now carrying an explicit
  length-confound caveat), `02_layer_dynamics`.
- **`notebooks/archive/` — negatives and nulls, kept deliberately:**
  `08_trajectory_fpca_vs_scalar`, `09_pca_ablation_analysis` (its trailing
  "interpretation prompts" cell replaced with the actual conclusion),
  `10_prefix_filter_analysis` (stale "re-run `evaluate_prefix_filter`"
  instructions removed — that stage no longer exists).
- **`notebooks/archive/` — stale inputs:** `03_math500_stratification` reads
  single-sample MATH-500, which no longer has a collect stage.
  Difficulty/subject metadata is absent from the Best-of-8 OOF CSV, so the
  stratification has not been redone under the corrected protocol. If stratified
  claims are needed, join MATH-500 metadata onto the OOF CSV by `prompt_id` and
  redo it within-prompt.
- Archived notebooks still execute in place: `_viz_utils` and `results/` are
  found by walking up to the repo root, so `archive/` needed no path edits
  (verified by running all four).
- Notebook numbering keeps its gaps (00, 04-07 deleted) and archiving does not
  renumber, so existing references stay valid.

### Active graph after the cut

`collect_qwen_arch`, `collect_arch`, `analyze_base`, `analyze_controls`,
`merge_analyze` (GSM8K only), the frozen `collect_qwen_dense_math500` trio,
`truncation_probe`, `collect_bestofn_full`, `evaluate_prompt_decomposition`,
`evaluate_wave1_experiments`, `summarize`.

### Next dependent stage

Cross-model confirmation of localization (`rmd_high_entropy_q20 − rmd`) and
entropy-specificity (`− rmd_random_q20`) at each model's pre-specified deepest
layer. **Gate:** if localization fails on DeepSeek at L21, the localization claim
demotes to Qwen-specific and the `deepseek_llama` MATH-500 collect does not run.

## 2026-07-29: E1 abstention REPLICATES on DeepSeek (between-prompt regime)

### Outcome

`evaluate_wave1_experiments@1`, DeepSeek-R1-Distill-Qwen-7B, MATH-500 Best-of-8,
500 prompts, deepest layer 21, 1,000-draw prompt-cluster bootstrap. Companion to
the failed within-prompt gate below: **the between-prompt abstention claim
survives cross-model where the within-prompt localization claim did not.**

**Not pre-registered.** Unlike the localization gate, no E1 criterion was fixed
in advance. Twelve contrasts, unadjusted. Reported as a replication check run
after the fact, not as a committed test. (The headline contrast would survive
Holm across all 12 — p < 0.001 x 12 is still < 0.05 — but that is a
reassurance, not a substitute for pre-registration.)

### Risk–coverage, both models

| Method | DeepSeek AURC | acc@50% | Qwen AURC |
|:---|---:|---:|---:|
| rmd_tail_q20 | **0.856** | 0.856 | 0.828 |
| rmd_high_entropy_q20 | 0.832 | 0.840 | 0.788 |
| length | 0.826 | 0.828 | 0.759 |
| logprob | 0.788 | 0.788 | 0.666 |
| entropy | 0.788 | 0.792 | 0.660 |

Identical ordering on both models: same winning region (`tail_q20`), same
runner-up, same losers, and length again the strongest free baseline.
Full-coverage accuracy differs substantially — DeepSeek 0.750 vs Qwen 0.620 —
so DeepSeek offers any scorer less headroom.

### The confound-clearing contrast

| Contrast | Metric | DeepSeek | Qwen |
|:---|:---|:---|:---|
| `rmd_tail_q20 − length` | AURC | **+0.030 [+0.014, +0.048], p<0.001** | +0.069 [+0.043, +0.096], p<0.001 |
| `rmd_tail_q20 − length` | acc@50% | +0.028 [−0.004, +0.072], p=0.094 | +0.104 [+0.064, +0.144], p<0.001 |
| `rmd_tail_q20 − entropy` | AURC | +0.068 [+0.037, +0.104], p<0.001 | +0.168, p<0.001 |
| `rmd_tail_q20 − logprob` | AURC | +0.068 [+0.036, +0.102], p<0.001 | +0.162, p<0.001 |
| `rmd_he_q20 − length` | AURC | +0.005 [−0.011, +0.025], p=0.506 | +0.030, p=0.040 |

Beating **length** is what separates this from truncation detection, and it
replicates. Entropy and logprob are weak baselines on this task in both models,
so those contrasts are large but not very informative.

### Three limits on the replication

1. **Effect is ~2.3x smaller** — +0.030 vs Qwen's +0.069 against length.
   Replicates in sign and significance, not in magnitude.
2. **Only AURC clears length; acc@50% does not** (+0.028, p=0.094). On Qwen both
   did. The replication is strongest on the integrated measure, not at the
   specific operating point.
3. **Only the tail region survives.** `rmd_he_q20 − length` is null on DeepSeek
   (+0.005, p=0.506) where it was marginal on Qwen (+0.030, p=0.040). Region
   choice does not transfer as cleanly as the overall effect.

### Combined interpretation across the two 2026-07-29 entries

| Regime | Question | Qwen | DeepSeek |
|:---|:---|:---|:---|
| Within-prompt | which of N attempts is correct? | small effect, ties output baselines | **absent — all AUCs at/below chance** |
| Between-prompt | should the model answer this problem? | beats length, p<0.001 | **beats length, p<0.001** |

The defensible cross-model claim is therefore narrower and better supported than
the one this project started with: **hidden-state geometry indicates which
problems are hard, not which attempt is right.** The failed localization gate is
load-bearing evidence for that framing rather than a setback — it rules out the
per-attempt reading that the Qwen-only data would otherwise permit.

## 2026-07-29: GATE FAILED — localization is Qwen-specific

### Outcome

The pre-registered gate below (written 2026-07-28, before any DeepSeek
decomposition output existed) **fails on both confirmatory tests**. Per the
decision rule fixed in advance: the localization claim demotes to
**Qwen-specific**, and the `deepseek_llama` MATH-500 best-of-8 collect
**does not run**.

`evaluate_prompt_decomposition@1`, DeepSeek-R1-Distill-Qwen-7B, MATH-500,
500 prompts x N=8, 8,192-token budget, layers 7/14/21, pca_dim 128, 5
prompt-grouped folds, 1,000-draw prompt-cluster bootstrap. Data audit clean:
500/500 complete prompts, `partial_data=false`, 12,000 traces, no duplicates.

### Confirmatory tests (L21, parseable, `prompt_centered_auc`)

| # | Contrast | Delta | 95% CI | raw p | Holm p | Verdict |
|:--|:---|---:|:---|---:|---:|:---|
| 1 | `rmd_high_entropy_q20 − rmd` | +0.004 | [−0.016, +0.027] | 0.674 | 1.000 | **FAIL** |
| 2 | `rmd_high_entropy_q20 − rmd_random_q20` | +0.001 | [−0.023, +0.026] | 0.924 | 1.000 | **FAIL** |

### This is an informative null, not merely low power

Qwen's L21 effect was **+0.058**. The DeepSeek 95% interval tops out at
**+0.027**, so a Qwen-sized effect is *excluded*, not just unresolved. The
conclusion is "the Qwen effect is not present here", not "we could not tell".

Power is nonetheless materially lower and must be stated: **49 mixed prompts /
409 within-prompt pairs**, against Qwen's 117 / 1,104. Censoring is also
heavier — 8.8% unparsed, 9.4% cap-hit, and **unparsed traces are 29.3% of the
incorrect class** (Qwen: 18.5%), so the parseable-only survivors are a more
selected subset.

### The larger finding: no within-prompt signal at all on DeepSeek

Within-prompt AUCs (macro / centered, parseable, 49 mixed prompts):

| Method | L7 | L14 | L21 |
|:---|:---|:---|:---|
| rmd | 0.458 / 0.465 | 0.470 / 0.479 | 0.473 / 0.447 |
| rmd_high_entropy_q20 | 0.484 / 0.478 | 0.520 / 0.498 | 0.507 / 0.451 |
| rmd_random_q20 | 0.447 / 0.467 | 0.453 / 0.471 | 0.456 / 0.450 |
| rmd_tail_q20 | 0.463 / 0.500 | 0.524 / 0.512 | 0.461 / 0.467 |
| entropy | 0.468 / 0.459 | — | — |
| logprob | 0.478 / 0.462 | — | — |

(entropy and logprob are layer-invariant.)

**Every cell is at or below chance.** Geometry did not lose to entropy — the
free output baselines fail too. There is no within-prompt correctness signal in
this model/dataset to detect, so the negative is about the absence of the
phenomenon rather than the inadequacy of the readout. This is a materially
different claim from "geometry is worse than entropy" and should be reported as
such.

### Exploratory (not part of the gate; no layer may be substituted post hoc)

`rmd_he_q20 − rmd` centered: L7 +0.012 (p=0.456), L14 +0.019 (p=0.304),
L21 +0.004 (p=0.674). No layer rescues the claim.

Pooled all-trace AUCs at L21 remain high — rmd 0.757 (ICC 0.898), rmd_tail_q20
0.762, against length 0.701 and entropy 0.609 — but these are the
length/truncation-confounded view that the 2026-07-18 audit disqualified as a
headline. They are consistent with a surviving *between-prompt* signal, which
`evaluate_wave1_experiments@1` (E1 abstention) tests separately. **The gate
governs the within-prompt localization claim only; it does not decide the
abstention claim.**

### Consequences

- `deepseek_llama` MATH-500 best-of-8 collect: **cancelled**. `bestofn_matrix`
  and `wave1_matrix` retain the row so the cell can be run later if the claim is
  reformulated, but it is not scheduled.
- The headline localization result stands **for Qwen only** and must be worded
  that way in `FINDINGS.md` and the paper.
- Two-regime framing survives and is arguably strengthened: within-prompt
  correctness detection now looks model-specific and fragile, while
  between-prompt abstention is the claim with a chance of generalizing.

### Infrastructure note

Both long-trace stages needed memory work before they would run at all.
`prompt_decomposition.py` and `wave1_experiments.py` now share three levers
(`--hidden_dtype float16`, `--compute_dtype float32`,
`--max_reference_tokens 2000000`); see the 2026-07-25 entry for the dtype
rationale and `analyze.set_max_reference_tokens` for the cap. Peak RAM per stage
drops from ~243–330 GB to ~140 GB. The cap does not bind for Qwen (~550k
correct-training tokens vs the 2M cap), verified bit-identical
(`max abs diff 0.000e+00`); under a deliberately harsh 40k cap the scores still
track at Spearman 0.9993, so the DeepSeek 5.7M -> 2M reduction is not a
plausible cause of the null above.

## 2026-07-28: Pre-registered gate criterion (written before DeepSeek results existed)

Recorded while `evaluate_prompt_decomposition@1` was still running, with no
DeepSeek decomposition output on disk. Timestamped here precisely so the gate
decision cannot be a post-hoc choice of threshold.

**Confirmatory set — 2 tests, no others:**

| # | Contrast | Layer | Metric |
|:--|:---|:---|:---|
| 1 | `rmd_high_entropy_q20 − rmd` | deepest (deepseek L21, deepseek_llama L24) | `prompt_centered_auc` |
| 2 | `rmd_high_entropy_q20 − rmd_random_q20` | deepest | `prompt_centered_auc` |

Both on the parseable-only within-prompt population, using the paired
prompt-cluster bootstrap already computed by the stage.

**Adjustment:** Holm across the 2 tests, family-wise alpha 0.05. Applied to the
saved JSON at gate time — no pipeline edit needed, so this costs no compute and
does not invalidate any stage.

**Decision rule, fixed in advance:**

- **Both pass** (Holm-adjusted p < 0.05, point estimate positive) -> localization
  replicates cross-model. Proceed to `deepseek_llama` MATH-500 collect.
- **Test 1 passes, test 2 fails** -> localization replicates but
  entropy-specificity does not. Report as "localization is real, mechanism
  unconfirmed"; still proceed, since test 1 is the primary claim.
- **Test 1 fails** -> claim demotes to Qwen-specific. **Stop.** Do not run the
  `deepseek_llama` collect. Report the negative.

`prompt_centered_auc` is named as primary because it is the metric the Qwen
headline (+0.052/+0.055/+0.058) was quoted on. `within_prompt_macro` is reported
alongside as secondary and does not enter the gate.

Everything else in the 16-pair x 3-layer x 2-metric grid is **exploratory** and
will be reported with raw p-values and an explicit exploratory label. The single
Qwen incremental-probe cell at p=0.024 is in that exploratory set and does not
survive correction; it is not a claim.

## 2026-07-19: Active-pipeline cleanup

The active DVC graph now contains the Qwen baseline, Qwen dense/PCA checks,
truncation probes, Qwen Best-of-N decomposition/selection, Wave-1 CPU follow-ups,
and the Qwen trajectory negative control. Historical DeepSeek 2,048-token analyses,
temperature, transfer, pilot, prefix, legacy selective-prediction, application-
alignment, and all-trace one-class stages remain on disk but are retired from the
default graph and current summary. Their results are provenance, not current claims.

Clean replication budgets remain recorded in `params.yaml` (`8192` for
DeepSeek-Qwen, `12288` for DeepSeek-Llama) without active collection stages.

## 2026-06-14: Confidence Decomposition and Mechanism Experiments

### Status

| Experiment family | Conditions | Status |
|:---|:---|:---|
| Enriched prompt decomposition | Qwen and DeepSeek, 500 prompts x 8 traces | Complete |
| OOF prompt selection | Qwen and DeepSeek, 500 prompts x 8 traces | Complete |
| Application alignment | Qwen and DeepSeek, raw/RMD x 3 layers | Complete |
| Fair supervised RMD probe | Qwen, DeepSeek, Llama, DeepSeek-Llama | Complete |
| One-class mechanism sweep | Four models x 3 layers x 8 dimensions | Complete |

Artifacts:

- `results/qwen_bestofn_full/math500/math500_prompt_decomposition_results.json`
- `results/qwen_bestofn_full/math500/math500_prompt_decomposition_oof.csv`
- `results/deepseek_bestofn_full/math500/math500_prompt_decomposition_results.json`
- `results/deepseek_bestofn_full/math500/math500_prompt_decomposition_oof.csv`
- `results/qwen_bestofn_full/math500/math500_prompt_selection_results.json`
- `results/deepseek_bestofn_full/math500/math500_prompt_selection_results.json`
- `results/application_alignment/math500_application_alignment_results.json`
- `results/*_selective/math500/math500_selective_prediction_results.json`
- `results/*_one_class/math500/math500_one_class_sweep_results.json`

Protocol:

- Five prompt-grouped folds.
- PCA, correct-trace reference, and RMD background fitted on training prompts.
- Evaluation on held-out prompts.
- 1,000 prompt-cluster bootstrap replicates over fixed OOF predictions.
- All configured layers and dimensions are reported without post-hoc selection.
- The enriched OOF CSV contains answer metadata, entropy, log-probability,
  length, activation norm, centroid, raw Mahalanobis, and RMD scores.

### Prompt Decomposition

| Model | Layer | Method | Pooled AUC | Prompt-centered AUC | Within-prompt AUC | ICC | Prompt-score/pass-rate Spearman |
|:---|---:|:---|---:|---:|---:|---:|---:|
| Qwen | 7 | RMD | 0.736 | 0.555 | 0.551 | 0.943 | 0.452 |
| Qwen | 14 | RMD | 0.763 | 0.550 | 0.550 | 0.970 | 0.499 |
| Qwen | 21 | RMD | 0.786 | 0.592 | 0.602 | 0.960 | 0.547 |
| DeepSeek | 7 | RMD | 0.885 | 0.841 | 0.931 | 0.878 | 0.682 |
| DeepSeek | 14 | RMD | 0.887 | 0.837 | 0.927 | 0.920 | 0.694 |
| DeepSeek | 21 | RMD | 0.892 | 0.797 | 0.930 | 0.831 | 0.698 |

DeepSeek RMD beats entropy on within-prompt pairwise AUC by 0.134-0.138
across all three layers. The paired prompt-bootstrap intervals exclude zero:

| Layer | RMD minus entropy within-prompt AUC | 95% CI |
|---:|---:|:---|
| 7 | +0.138 | [+0.108, +0.169] |
| 14 | +0.134 | [+0.105, +0.165] |
| 21 | +0.138 | [+0.108, +0.168] |

For Qwen, RMD does not beat entropy, log-probability, or length within prompts
at any layer with a confidence interval excluding zero. Its pooled strength is
therefore primarily a between-prompt solvability signal.

### Prompt Selection

| Model | Layer | Random | Entropy | Length | RMD top-1 | Strict majority | Oracle Pass@8 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Qwen | 7 | 0.557 | 0.572 | 0.566 | 0.550 | 0.596 | 0.676 |
| Qwen | 14 | 0.557 | 0.572 | 0.566 | 0.552 | 0.596 | 0.676 |
| Qwen | 21 | 0.557 | 0.572 | 0.566 | 0.564 | 0.596 | 0.676 |
| DeepSeek | 7 | 0.416 | 0.488 | 0.506 | 0.524 | 0.452 | 0.546 |
| DeepSeek | 14 | 0.416 | 0.488 | 0.506 | 0.526 | 0.452 | 0.546 |
| DeepSeek | 21 | 0.416 | 0.488 | 0.506 | 0.524 | 0.452 | 0.546 |

Paired bootstrap reanalysis of saved prompt outcomes:

- DeepSeek RMD top-1 beats random by 0.109-0.111, entropy by 0.036-0.038,
  and length by 0.018-0.020. All corresponding 95% intervals exclude zero.
- Qwen RMD top-1 differs from random by -0.007 to +0.007, and every 95%
  interval includes zero.
- Under the strict invalid-output policy, DeepSeek RMD rank-weighted voting
  reaches 0.488 versus 0.452 for majority vote, but remains below RMD top-1.
  Qwen RMD rank-weighted voting reaches 0.582-0.584 versus 0.596 for majority.

Voting has a major parser limitation. Unparsed answers are excluded from the
vote, while answer parsing is also required for a trace to be labeled correct:

| Model | Correct parse rate | Incorrect parse rate | Prompts with no parsed answer |
|:---|---:|---:|---:|
| Qwen | 1.000 | 0.815 | 2 / 500 |
| DeepSeek | 1.000 | 0.224 | 136 / 500 |

The original parsed-only vote silently removed invalid traces, producing the
artificial DeepSeek result `majority = Oracle Pass@8 = 0.546`. The corrected
strict vote counts an unparsed response as an explicit invalid output and
scores invalid winners as failures.

The historical NPZ files do not contain generated text or token arrays, so the
missing answers cannot be reparsed. Future collections now persist both token
strings and generated text, and use balanced-brace parsing for nested
`\\boxed{}` / `\\fbox{}` answers.

### Fair Supervised RMD Probe

Best MATH-500 AUSC across configured layers:

| Model | Entropy | Unsupervised RMD | Old entropy+raw LR | Entropy+RMD LR | Gain over entropy | Gain over unsupervised RMD |
|:---|---:|---:|---:|---:|---:|---:|
| Qwen | 0.621 | 0.721 | 0.701 | 0.737 | +0.116 | +0.016 |
| DeepSeek | 0.500 | 0.633 | 0.620 | 0.639 | +0.139 | +0.006 |
| Llama | 0.384 | 0.493 | 0.465 | 0.507 | +0.123 | +0.013 |
| DeepSeek-Llama | 0.442 | 0.506 | 0.481 | 0.526 | +0.084 | +0.020 |

The fair supervised probe confirms that the old supervised baseline was using
the weaker raw geometry. Entropy+RMD is best in every model, but most of its
signal is already present in the unsupervised RMD score.

### One-Class Mechanism Sweep

Mean pooled ROC-AUC across each model's three sparse layers:

| Model | RMD dim 8 | RMD dim 32 | RMD dim 128 | Raw Ledoit-Wolf dim 128 |
|:---|---:|---:|---:|---:|
| Qwen | 0.717 | 0.762 | 0.772 | 0.379 |
| DeepSeek | 0.867 | 0.870 | 0.869 | 0.225 |
| Llama | 0.750 | 0.778 | 0.781 | 0.396 |
| DeepSeek-Llama | 0.783 | 0.786 | 0.792 | 0.352 |

- Diagonal, empirical-ridge, and Ledoit-Wolf target-only Mahalanobis AUCs
  differ by less than 0.001 throughout the sweep.
- Background subtraction is the load-bearing mechanism. Target-only distances
  are often strongly anti-predictive, especially for DeepSeek, while RMD is
  strongly predictive.
- A universal rank-1 mechanism is rejected. DeepSeek reaches its plateau near
  dimension 8 and DeepSeek-Llama near 4-8, while Qwen and Llama continue to
  improve through 64-128 dimensions.
- Input normalization does not provide a consistent advantage over ordinary
  RMD once more than a few components are retained.

### Current Interpretation

1. RMD is not merely a prompt-difficulty signal. For DeepSeek it is a strong
   trace-level correctness signal and a useful within-prompt selector.
2. The same score is model-conditional. Qwen RMD is primarily useful for
   between-prompt abstention and provides no reliable top-1 reranking gain.
3. The mechanism is relative geometry, not covariance estimation. Subtracting
   the generic background distribution reverses a strongly misleading raw
   distance signal.
4. Variance structure predicts application fit: within-prompt AUC tracks top-1
   gain, and prompt-score/pass-rate correlation tracks selective-prediction
   gain. These correlations remain exploratory because there are only two
   models and three correlated layers per model.
5. ICC alone is not an application selector. It is essentially uncorrelated
   with selective-prediction gain in the current conditions.

### Limitations and Compatibility

- The bootstrap resamples fixed OOF predictions; it does not refit PCA and
  covariance references inside every bootstrap replicate.
- Prompt-selection voting is confounded by answer-parser missingness.
- Application-alignment correlations reuse layers from the same models and are
  not independent replications.
- The Qwen and DeepSeek checkpoint comparison is not a clean causal
  distillation intervention because their training lineages differ.
- Selective-prediction results currently lack paired problem-bootstrap
  intervals for scorer differences.

### Next Experiments

| Priority | Experiment | Purpose | Cost |
|---:|:---|:---|:---|
| 1 | Add paired bootstrap intervals for selection and selective AUSC deltas | Quantify application-level uncertainty | Cheap reanalysis |
| 2 | Length-matched and confidently-wrong controls | Test whether RMD remains informative beyond length and confidence | Cheap reanalysis |
| 3 | Replicate enriched decomposition on Llama and DeepSeek-Llama Best-of-N traces | Test whether application alignment generalizes across architecture families | Requires Best-of-N inference |
| 4 | Matched Qwen2.5-Math-7B comparison | Separate reasoning distillation from base-model/math-training differences | Requires inference |

## 2026-06-14: Truncation-Confound Audit of the DeepSeek Within-Prompt Result

### Status

| Experiment family | Conditions | Status |
|:---|:---|:---|
| Within-prompt decomposition re-audit | Qwen and DeepSeek, existing 500x8 OOF CSVs | Complete (reanalysis, no new compute) |

This is a code- and CSV-level audit of the within-prompt correctness claim, not a
new collection run. No NPZ/hidden-state access was used; all numbers come from the
already-written OOF CSVs.

Artifacts (inputs, unchanged):

- `results/deepseek_bestofn_full/math500/math500_prompt_decomposition_oof.csv`
- `results/qwen_bestofn_full/math500/math500_prompt_decomposition_oof.csv`

Code changes:

- `prompt_decomposition.py`: added `is_unparsed`, `truncation_report`,
  `parseable_within_prompt_metrics`; `analyze_oof_scores` now emits a top-level
  `truncation` block and per-layer `truncation` + `parseable_only` blocks; new
  `--max_new_tokens` CLI arg (inferred from max observed length if omitted);
  Markdown report gains a truncation/parseability section.
- `dvc.yaml`: `evaluate_prompt_decomposition` now passes
  `--max_new_tokens ${item.max_new_tokens}` so capped-trace diagnostics are exact.
- `tests/test_prompt_decomposition.py`: added 5 tests (26 pass).

### Mechanism

`collect_data.py:313`:
`is_correct = answers_match(predicted_answer, gold) if (predicted_answer and gold) else False`.
Any trace with no parseable final answer is auto-labeled incorrect. The
decomposition consumed `is_correct` with no parseability filter, so non-answers
entered the "incorrect" class.

### Primary findings (per layer, all 4000 traces/layer)

| Quantity | DeepSeek | Qwen |
|:---|---:|---:|
| Unparsed (no final answer) | 1814/4000 (45.4%) | 328/4000 (8.2%) |
| Of unparsed, length-capped at max_new_tokens | 99.4% (1804 at exactly 2048) | ~all at 1024 |
| Unparsed share of the incorrect class | 77.6% | — |
| within_macro RMD, ALL traces (L7/14/21) | 0.931 / 0.931 / 0.933 | 0.557 |
| within_macro RMD, PARSEABLE-only | 0.266 / 0.274 / 0.279 | 0.503 |
| within_macro entropy, PARSEABLE-only | 0.348 | 0.660 |
| Mixed-prompt count, ALL -> PARSEABLE | 166 -> 13 | 131 -> 117 |

DeepSeek `max_new_tokens=2048` is too small for R1-Distill on MATH500: 45% of
generations hit the cap before emitting `\boxed{}`. RMD scores these as strongly
anomalous (mean rmd_score correct=0.42, parseable-wrong=0.36, unparsed=0.11 at
L7; gap widens at deeper layers). Mean length: correct=1371, parseable-wrong=1455,
unparsed=2043.

### Claims ruled out

- RULED OUT (high confidence): "DeepSeek within-prompt AUC ~0.93 measures
  within-trace reasoning correctness." It is overwhelmingly a truncation /
  termination detector. Removing non-answers collapses the mixed-prompt set 166->13
  (92% of within-prompt mixedness was correct-vs-truncated, not correct-vs-wrong).
- RULED OUT (high confidence): the cross-model thesis "distillation reshapes
  geometry from between-problem solvability (Qwen) to within-trace correctness
  (DeepSeek)" as currently evidenced. The Qwen(0.55) vs DeepSeek(0.93) within-prompt
  gap tracks the differential truncation rate (8% vs 45%), not distillation. Qwen
  RMD is at chance within-prompt with or without filtering.
- SUPERSEDES "Current Interpretation" point 1 (2026-06-14 entry, line ~144) and
  upgrades the limitation at line ~162 from "confounded by parser missingness" to
  "dominated by truncation" for the within-prompt metric specifically.

### Claims still standing

- What RMD genuinely detects here is degenerate / non-terminating generations.
  That is real and plausibly useful for Best-of-N rejection, but is confounded
  with length and is not evidence of within-trace correctness geometry.
- The parseable-only contrast (n=13 mixed prompts) is too small to pin RMD's true
  within-prompt sign; the only firm claim is that the 0.93 headline does not survive.

### Limitations

- Parseable-only DeepSeek estimate rests on 13 mixed prompts -> noisy.
- True lengths of truncated traces are censored at 2048; existing data cannot say
  what `max_new_tokens` is sufficient.

### Next dependent stage

- BLOCKER for the Llama decomposition: `deepseek_llama` is also `max_new_tokens=2048`
  (params.yaml) and will inherit the identical artifact. Before the full 500x8
  campaign, run a small smoke test (limit ~30, T=0.6) at a raised budget (try 8192)
  on `deepseek` and `deepseek_llama`, measure cap-hit rate, pick the smallest budget
  with single-digit truncation (watch hidden-state storage ~ tokens x layers), then
  collect full. Do NOT run full 500x8 at 2048.

## 2026-07-11: Prompt-Local RMD and Current Evidence Reconciliation

### Status

| Experiment family | Condition | Status |
|:---|:---|:---|
| Prompt-local RMD | Qwen MATH-500, 500 prompts x 8 traces, layers 7/14/21 | Complete |
| Prompt-local top-1 selection | Same Qwen OOF scores | Complete |
| DeepSeek-Qwen budget probe | 8192 tokens, 24 traces | Complete; 12.5% capped/unparsed |
| DeepSeek-Llama budget probe | 12288 tokens, 24 traces | Complete; 0% capped/unparsed |
| DeepSeek prompt-local RMD | Historical 2048-token Best-of-N data | Deliberately not interpreted; truncation-contaminated |
| Clean cross-model prompt decomposition | Re-collected Best-of-N data | Not run |

Artifacts:

- `results/qwen_bestofn_full/math500/math500_prompt_decomposition_results.json`
- `results/qwen_bestofn_full/math500/math500_prompt_decomposition_oof.csv`
- `results/qwen_bestofn_full/math500/math500_prompt_decomposition_report.md`
- `results/qwen_bestofn_full/math500/math500_prompt_selection_results.json`
- `results/qwen_bestofn_full/math500/math500_prompt_selection_report.md`
- `results/truncation_probe/deepseek_8192.json`
- `results/truncation_probe/deepseek_llama_12288.json`

### Prompt-Local Protocol

For every held-out prompt and trace, the score uses the global OOF PCA and
correct-trace reference fitted on training prompts. Its local background is a
diagonal Gaussian fitted to tokens from the other seven attempts of that same
held-out prompt. The scored trace is excluded from its local background. The
fixed-orientation confidence score is the mean local-background distance minus
the global raw correct-manifold distance.

This is a quick test of whether removing prompt-shared semantic variation
reveals a same-prompt correctness residual. It uses no correctness labels from
the held-out prompt, but it is transductive because sibling attempts are
available at scoring time.

### Primary Results

| Layer | Prompt-local pooled AUC | Prompt-local centered AUC | Prompt-local within pair AUC | ICC | Top-1 Pass@1 |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.379 | 0.501 | 0.499 | 0.965 | 0.558 |
| 14 | 0.402 | 0.480 | 0.480 | 0.955 | 0.532 |
| 21 | 0.320 | 0.523 | 0.529 | 0.940 | 0.548 |

Selection references are random trace `0.557`, strict majority vote `0.596`,
and oracle Pass@8 `0.676`. Prompt-local RMD does not improve on random and is
consistently below majority vote.

For comparison, global RMD pooled AUC is `0.736/0.763/0.786` and global RMD
within-prompt pair AUC is `0.551/0.550/0.602` at layers 7/14/21. Prompt-local
subtraction removes the useful between-prompt component without exposing a
strong same-prompt component.

On parseable-only traces (117 mixed prompts), prompt-local within-macro AUC is
`0.446/0.436/0.483`, while global RMD is `0.503/0.515/0.574` and log-probability
is `0.649` at every layer. The apparent L21 all-trace prompt-local pair AUC of
`0.529` therefore does not survive the stricter correctness population.

### Interpretation

- Rejected for this estimator and Qwen dataset: same-prompt full-trace residual
  geometry is sufficient for correctness ranking.
- Supported: the useful global RMD signal is largely tied to prompt-level
  semantic/difficulty structure rather than an attempt-specific offset that can
  be recovered with a sibling-trace Gaussian.
- Supported: full-trace averaging is likely too coarse for local arithmetic,
  sign, or late-answer errors. The next geometry tests should localize scoring
  to high-entropy tokens, the trace tail, answer regions, or step transitions.
- This negative result does not rule out all prompt-conditional geometry. It
  rules out this simple leave-one-trace-out diagonal local-background method on
  Qwen MATH-500.

### Length and Truncation Context

The Qwen global RMD-minus-length contrast is strongest at L21: pooled `+0.055`
with 95% CI `[+0.021, +0.093]`, centered `+0.092` with
`[+0.047, +0.134]`, and within macro `+0.116` with `[+0.065, +0.171]` on all
traces. Parseable-only within-prompt performance is much weaker, so these
all-trace contrasts must not be presented as clean trace-correctness estimates.

The budget probes establish collection settings, not final scientific results:

| Model | Budget | n | Capped | Unparsed | Completed p95 | Completed max |
|:---|---:|---:|---:|---:|---:|---:|
| DeepSeek-R1-Distill-Qwen-7B | 8192 | 24 | 12.5% | 12.5% | 3924 | 5193 |
| DeepSeek-R1-Distill-Llama-8B | 12288 | 24 | 0% | 0% | 10237 | 11163 |

### Repository and DVC State

The result files exist, but the current worktree is not globally DVC-clean.
`dvc status` reports many changed dependencies because analysis and collection
code plus `params.yaml` have evolved since `dvc.lock`. In particular,
`evaluate_prompt_decomposition@0` and `evaluate_prompt_selection@0` report
changed dependencies despite the new Qwen outputs being present. Do not treat
an existing artifact as proof that its current stage definition is reproduced.

No files were staged or committed as part of this documentation update.

### Next Dependent Experiments

1. Implement and run high-entropy-token and tail-only RMD on the existing Qwen
   OOF protocol. These are the smallest tests of localized error geometry.
2. Add answer-cluster geometry to prompt selection using the existing enriched
   OOF CSV before collecting more hidden states.
3. Re-run parseable-only selective prediction with paired problem-bootstrap
   intervals against length, entropy/log-probability, and a trained linear
   probe. This determines whether the abstention application survives.
4. Only after the cheap gates pass, collect clean Best-of-N data for additional
   model families using architecture-specific token budgets. Do not rerun the
   old DeepSeek 2048-token decomposition as evidence.

## 2026-07-18: Qwen Best-of-8 Localized Geometry, Contrastive Readouts, and Selection

### Stage and parameterization

`evaluate_prompt_decomposition@0` and `evaluate_prompt_selection@0` (qwen full
item of `bestofn_matrix`), rerun 2026-07-18 on CPU (`CUDA_VISIBLE_DEVICES=""`).
Qwen2.5-7B-Instruct, MATH-500, 500 prompts x N=8, max_new_tokens=1024, layers
7/14/21, pca_dim=128, 5 prompt-grouped folds, 1,000 prompt-cluster bootstrap
replicates over fixed OOF predictions, contrastive regions
full/high_entropy_q20/tail_q20/random_q20, 1,000 alignment shuffles. Data audit
clean: 500/500 complete prompts, `partial_data=false`. These numbers supersede
the 2026-06-14 Qwen decomposition entry (post truncation-bias fix, commit
`d54906a`); e.g. the L21 RMD-minus-length centered contrast is now null
(+0.029, p=0.194) where the old entry reported +0.092 significant.

### Artifacts

- `results/qwen_bestofn_full/math500/math500_prompt_decomposition_results.json`
  (45 probe diagnostics, alignment diagnostics, parseable-only blocks)
- `results/qwen_bestofn_full/math500/math500_prompt_decomposition_oof.csv`
  (12,000 rows = 3 layers x 4,000 traces, 28 cols incl. probe/contrast scores)
- `results/qwen_bestofn_full/math500/math500_prompt_selection_results.json`
- both `_report.md` companions

### Primary estimates and uncertainty

Truncation context: 8.2% unparsed, 8.45% cap-hit, unparsed = 18.5% of the
incorrect class. Length pools at AUC 0.737 on all traces but collapses to
0.478 within-macro on parseable traces, so all-trace pooled AUCs (rmd_tail_q20
up to 0.839 at L21) remain length/truncation-confounded. RMD ICC 0.94–0.97.

Parseable-only within-prompt (117 mixed prompts, ~1,104 pairs; output
baselines layer-invariant: entropy 0.660 macro / 0.611 centered, logprob
0.649 / 0.609, cross-fitted output probe 0.634 / 0.592):

| Method (within macro / centered) | L7 | L14 | L21 |
|:--|:--|:--|:--|
| rmd | 0.503 / 0.513 | 0.515 / 0.503 | 0.574 / 0.547 |
| rmd_high_entropy_q20 | 0.588 / 0.564 | 0.588 / 0.557 | 0.654 / 0.605 |
| contrast_high_entropy_q20 | 0.617 / 0.574 | 0.640 / 0.590 | 0.660 / 0.617 |
| probe_outputs + rmd_he_q20 | 0.662 / 0.600 | 0.629 / 0.590 | 0.683 / 0.609 |

Prespecified paired contrasts (parseable, centered AUC unless noted):

- **Localization supported at all layers:** rmd_he_q20 − rmd = +0.052/+0.055/
  +0.058 (L7/14/21), p ≤ 0.006; within macro +0.073…+0.085, p ≤ 0.002.
  Tail-20% weaker, mostly ns.
- **Entropy-specificity supported for RMD:** rmd_he_q20 − rmd_random_q20 =
  +0.049/+0.058/+0.057, p ≤ 0.014; the matched random-20% control tracks
  full-trace rmd. Contrast version mixed at L21 (centered p=0.128, macro
  p=0.032).
- **Contrastive readout partial:** OOF cross-prompt directions beat the
  shuffle null (L21 alignment 0.180–0.222 vs null ≈0.101, p ≤ 0.005 across
  folds; mean pairwise cosine only 0.02–0.04). Contrast beats plain rmd
  (L14 +0.088 p=0.002; L21 +0.070 p=0.018) but never beats matched rmd_he_q20
  (p ≥ 0.118).
- **No geometry readout beats free output baselines:** rmd_he_q20 − logprob at
  L21 = −0.004 centered, p=0.926; negative at L7/L14 (rmd_tail_q20 − logprob
  at L14 significantly negative, −0.072, p=0.048).
- **Incremental probes weak:** probe+rmd_he_q20 − probe = +0.049 macro
  [0.006, 0.091], p=0.024 at L21 only (centered ns; 1/6 cells, unadjusted).
  L21 fold-averaged coefficients: rmd_he_q20 +0.39±0.11, entropy collapses
  +0.28→+0.06, length goes negative.
- **Selection null:** majority vote 0.596 pass@1 (random 0.557, oracle
  0.676); all tie-break variants within ±0.006, 15/15 paired deltas p ≥
  0.248. Structural ceiling: only 39/500 prompts tie at N=8 and only ~10
  ties have correctness headroom (~2 pts max). rmd_rank_weighted_vote
  0.582–0.584 < majority; all top1 selectors ≤ 0.582.

Exploratory follow-ups on the OOF CSV (unregistered, label-free
residualization, no CIs on selective numbers except where stated):

- **Residualization:** within-prompt-centered rmd_he_q20 projected onto
  entropy+logprob+length keeps its discrimination at L21: residual
  within-macro 0.645 [0.587, 0.697] vs 0.654 raw (R² vs outputs 0.227).
  Geometry is linearly complementary to output features; the near-null
  incremental probes reflect saturation on 117 mixed prompts.
- **Selective prediction (L21, parseable, base acc 0.606):** acc@50%
  coverage — rmd 0.784, rmd_he_q20 0.766 vs entropy 0.676, logprob 0.675.
- **Prompt-level abstention with majority-vote answering (full-coverage acc
  0.616):** acc@50% — rmd_tail_q20 0.836, rmd 0.796 vs length 0.740,
  logprob 0.680, entropy 0.672. Geometry beats the length-confound baseline
  by ~+0.10 at 50% coverage.

### Interpretation

Two-regime story confirmed on clean Qwen data. Within-prompt: a small,
depth-increasing, entropy-localized correctness signal exists (best 0.654
macro at L21), is entropy-specific (random-token control fails), is linearly
complementary to output features, but only ties the free baselines and does
not translate into Best-of-8 selection. Between-prompt: geometry is a strong
difficulty/abstention signal that clearly beats entropy, logprob, and length
in risk–coverage terms. Ruled out: geometry-guided tie-breaking at N=8
(no-op by construction); all-trace pooled AUC as a headline (length
confound); contrastive supervision adding anything beyond region choice.

### Limitations and next dependent stage

Bootstrap CIs do not propagate reference refitting; 21 contrasts x 3 layers x
2 metrics unadjusted (the single L21 incremental p=0.024 would not survive
correction); parseable-only conditions on an outcome-correlated event
(correct traces parse at 1.000 vs 0.815); within-prompt inference rests on
117 mixed prompts. Next stages: (1) cross-model confirmation of localization
+ entropy-specificity at the pre-specified deepest layer on
deepseek/llama/deepseek_llama full runs (deepseek_llama decomposition outputs
currently deleted per `dvc status` — regenerate first); (2) run the
risk–coverage comparison with prompt-cluster bootstrap CIs through the
selective-prediction stages; (3) entropy-residualized geometry as a
registered contrast; (4) token-level audit of what the high-entropy 20%
localizes.

## Logging Convention

For every completed experiment, append:

1. exact DVC stage and parameterization;
2. artifact paths and schema;
3. primary point estimates and uncertainty;
4. interpretation and claims ruled in or out;
5. limitations and next dependent stage.
