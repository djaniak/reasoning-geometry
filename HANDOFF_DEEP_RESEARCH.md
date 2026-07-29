# Handoff: Hidden-State Geometry for Reasoning Reliability — Findings, Literature, Codebase

*Self-contained handoff for a fresh (Claude) deep-research context. Written 2026-06-19.
Companion docs in repo: `FINDINGS.md` (now carries a correction banner), `EXPERIMENT_LOG.md`,
`PAPER_STRATEGY.md`, `results/SUMMARY.md`. Two prior deep-research passes are summarized in §7.*

---

## 0. TL;DR

We study whether **hidden-state geometry** predicts correctness/reliability of LLM math-reasoning
traces, via **Relative Mahalanobis Distance (RMD)**. A major **truncation artifact** was found this
session that contaminates many headline numbers (especially for the distilled model). After
de-confounding, the defensible result is: **RMD is a between-prompt *solvability* signal that beats
trace-length and entropy baselines; it is NOT a within-prompt (per-attempt) correctness reranker.**
The novel, literature-checked angles are **(a)** distillation compressing correctness geometry into a
low-dimensional subspace and **(b)** the correct-reasoning manifold *shape* transferring across models
while the trained *readout* does not. All load-bearing numbers still need re-validation on clean
(non-truncated) data.

---

## 1. Research question & method

**Question:** Do intermediate hidden states encode whether a reasoning trace is correct/reliable,
beyond what token-level entropy (and trace length) already reveal — and how does this depend on
model training (instruct vs reasoning-distilled)?

**RMD (the core signal):**
1. Fit PCA (128 dims) on hidden states of **correct** traces → Gaussian (regularized covariance) =
   "correct-reasoning manifold."
2. Raw Mahalanobis distance = distance of a trace's tokens to that manifold.
3. **Relative** MD = raw MD **minus** distance to a generic *background* manifold (fit on all/ generic
   tokens). Background subtraction is load-bearing — raw MD alone is weak/anti-predictive; RMD fixes it.
4. Per-trace score = − mean token RMD (higher = more correct-like). Leakage-safe: PCA + both manifolds
   fit on training prompts only, scored out-of-fold (OOF).

**Models (HuggingFace):** Qwen2.5-7B-Instruct, DeepSeek-R1-Distill-Qwen-7B (same 28-layer Qwen2.5 arch,
hidden 3584 — controlled instruct-vs-distilled pair); Llama-3.1-8B-Instruct, DeepSeek-R1-Distill-Llama-8B
(second family). **Data:** MATH-500 (500 problems), GSM8K test. Hidden states captured at layers 7/14/21
(Qwen arch) or 8/16/24 (Llama arch) — ~25/50/75% depth. Two decoding regimes: greedy single-trace, and
Best-of-N (N=8, T=0.6) for within/between decomposition.

---

## 2. Codebase map

Pipeline orchestrated by **DVC** (`dvc.yaml` + `params.yaml`); summaries in `results/SUMMARY.md`.

| File | Purpose |
|---|---|
| `collect_data.py` | Autoregressive generation; captures per-token hidden states + entropy + logprobs → `data/{model}/{dataset}/batch_*.npz`. **Auto-labels `is_correct=False` when no `\boxed{}` parses — the source of the truncation artifact (line ~313).** |
| `analyze.py` | Fits PCA+Gaussian on correct traces; computes Mahalanobis / RMD; 5-fold logistic-regression AUC (entropy vs Mahal vs combined). Core scoring library reused elsewhere. |
| `prompt_decomposition.py` | Within-prompt vs between-prompt decomposition of Best-of-N traces (OOF RMD scores + prompt-cluster bootstrap). **Patched this session:** `truncation`/`parseable_only` diagnostics, `is_unparsed`, `--max_new_tokens`, and `rmd_minus_length` as primary contrast. |
| `selective_prediction.py` | Trust/abstain (risk–coverage) downstream task. **Patched:** `--exclude_unparsed` flag (parseable-only re-validation; not yet run). |
| `best_of_n.py` | Leakage-safe Best-of-N reranking (selectors: random/oracle/majority/entropy/logprob/RMD/combined). |
| `prefix_filter.py`, `prefix_analysis.py` | Early-abort / prefix-based detection simulations. |
| `one_class_sweep.py` | PCA-dimension + covariance-estimator (diagonal/ridge/Ledoit-Wolf) mechanism sweep. |
| `prompt_selection.py`, `application_alignment.py` | Prompt-level selectors from OOF scores; relate variance structure to application performance. |
| `probe.py`, `trajectory_encoders.py`, `trajectory_preprocessing.py` | Functional-trajectory (fPCA-on-Mahalanobis-sequence) encoding (Track A). |
| `pca_ablation.py`, `merge_results.py`, `summarize.py` | Aggregation / reporting. |
| `truncation_probe.py` | **New.** Budget-sizing probe: batched generation (no hidden states) measuring cap-hit rate + completion-length distribution. DVC stage `truncation_probe`. |
| `reproduce_findings.py` | **New.** One-command cheap reproduction of the corrected findings (contamination ledger + 3-tier AUC + within-prompt collapse) from existing artifacts. |
| `tests/` | pytest suite (≈27 tests on `prompt_decomposition`, parsing, etc.). |

**DVC stages:** collect_{qwen,llama}_arch → analyze_{base,controls,subspace,pca_ablation} → merge/cross →
collect/evaluate_bestofn_{pilot,full} → evaluate_{selective_prediction, prompt_decomposition,
prompt_selection, application_alignment, one_class_sweep, prefix_filter} → trajectory → summarize.
Plus the new `truncation_probe` (GPU, diagnostic, run deliberately).

---

## 3. THE CORRECTION (read before trusting any DeepSeek number)

**Mechanism:** `collect_data.py` labels any trace with no parseable `\boxed{}` as incorrect. With the
generation cap, a large fraction of DeepSeek-R1-distill traces hit the cap *without emitting an answer*
and were counted "wrong." RMD then flags these off-manifold/non-terminating traces — so geometry was
substantially detecting **truncation/non-termination**, not **wrong reasoning**.

**Measured contamination (cheap; `reproduce_findings.py`):**

| Data | unparsed | length-capped | unparsed share of "incorrect" class |
|---|---:|---:|---:|
| DeepSeek **greedy** MATH-500 (main table, selective pred, stratification) | 43% | 51% | 75% |
| DeepSeek **Best-of-N** MATH-500 (decomposition) | 45% | ~55% | 78% |
| Qwen greedy / Best-of-N | 8% | 8% | 17–18% |

**Retracted:** the prior headline *"DeepSeek RMD is genuinely trace-level, within-prompt AUC 0.93,
beats entropy +0.13."* On parseable-only Best-of-N traces the within-prompt mixed-prompt set
collapses **166 → 13** and within-macro falls to **~0.27 (chance/below)**.

---

## 4. Experiments & results (corrected status flagged)

**(i) Single-trace correctness (greedy, main table).** Combined entropy+Mahal AUC: Qwen MATH-500 0.772,
DeepSeek 0.859; GSM8K 0.781 / 0.835. Length-controlled deltas +1.4–8.0 pts. ⚠️ **DeepSeek
contamination-suspect** (43% truncated); Qwen largely clean.

**(ii) The decisive de-confounded result (Best-of-N OOF, parseable-only, pooled AUC):**
DeepSeek correctness RMD **0.636** vs length **0.545** vs entropy 0.515 (was 0.887 with truncated traces
included). Qwen RMD 0.70–0.75 vs length 0.68 vs entropy 0.58. **RMD-over-length margin +0.05–0.09,
prompt-bootstrap CI excludes zero, holds at every layer.** Signal is **between-prompt** (pooled positive)
but **within-prompt at chance** (per-attempt). → solvability/difficulty, not per-attempt reranking.

**(iii) Termination is mostly length.** Predicting parsed-vs-unparsed: Qwen length 0.996 > RMD 0.84
(do NOT headline "RMD rejects non-terminating traces"); DeepSeek RMD 0.95 ≳ length 0.91.

**(iv) Selective prediction / abstention (greedy).** RMD > entropy all four models (Qwen AUSC 0.721 vs
0.621; DeepSeek 0.633 vs 0.500; Llama 0.493 vs 0.384; DS-Llama 0.506 vs 0.442). Background subtraction
load-bearing. ⚠️ **DeepSeek/DS-Llama numbers contamination-suspect** (RMD can "abstain" on non-answers).
Re-validate parseable-only (`--exclude_unparsed`).

**(v) Best-of-N reranking — NEGATIVE.** Majority vote beats geometry (Qwen Pass@1 0.620 vs best-combined
0.582; RMD pilot doesn't rescue). Consistent with within-prompt being at chance.

**(vi) Mechanism (about geometry, less label-dependent — but AUC parts still need parseable check):**
- **Low-dimensional contrast for distilled models:** DeepSeek plateaus ~PCA dim 8, DS-Llama ~4–8;
  Qwen/Llama keep improving to 64–128.
- **RMD-not-covariance:** diagonal/ridge/Ledoit-Wolf nearly identical; *background subtraction* changes
  results by tens of AUC pts and reverses anti-predictive raw distance.
- **Bimodal two-phase layer profile** (Qwen dense sweep): peaks L6–10 (comprehension) and L20–26
  (execution), trough L14.
- **Cross-model transfer:** correct-manifold *shape* transfers 82–101% across instruct↔distilled
  (same arch); frozen classifier (readout) transfers poorly; late layers transfer better.

**(vii) Robustness / negatives:** geometry advantage *grows* under T=0.6; confident-wrong analysis null
for DeepSeek (diffuse trajectory-level signal, not token-local); functional-trajectory (fPCA) encoding
negative (doesn't beat scalar Mahal summaries); prefix/early-detection weak.

**(viii) Stratification (greedy MATH-500):** difficulty/subject deltas heterogeneous, mostly noisy;
DeepSeek-positive pockets (Interm. Algebra, Precalc). ⚠️ contamination-suspect.

---

## 5. Defensible (de-confounded) thesis

> **Relative hidden-state geometry (RMD) is a calibrated between-prompt *solvability* signal that beats
> both trace-length and entropy baselines, useful for abstention / compute allocation — NOT a
> within-prompt correctness reranker (Best-of-N is weak). Prior "trace-correctness" readings of such
> geometry are confounded by trace length and by a truncation/auto-label-as-incorrect artifact.**

---

## 6. Critical caveats / known pitfalls for any follow-up
- **Length is the baseline that matters** (pooled AUC ~0.74 Qwen / ~0.83 DeepSeek). Benchmark RMD
  against length, not just entropy. Entropy is a weak baseline (often <0.6).
- **Truncation contaminates labels.** Always report parseable-only + truncation rate. Re-collect at a
  larger `max_new_tokens` — **budgets confirmed by probe (2026-06-19, n=24/model):** Qwen-distill
  **8192** (completions finish ~5.2k; residual ~12% are non-terminating runaways a bigger budget won't
  save); Llama-distill **12288** (0% truncation, completions to ~11k — heavy tail, n=24 so some tail
  risk). The two same-recipe distilled models have very different length regimes — a single global
  budget is wrong. Wired into `params.yaml` (`bestofn_matrix`, with per-arch `layers`).
- **The RMD primitive is NOT novel** (see §7) — novelty must come from the application/mechanism, not
  "we use relative Mahalanobis."

---

## 7. Literature review (two adversarial deep-research passes; ~46 claims verified)

**Activation-geometry error detection (established, not white space):**
- INSIDE / EigenScore (ICLR 2024, arXiv:2402.03744): covariance/LogDet of K resampled embeddings;
  **within-prompt self-consistency**; scores *below random on GSM8K* in ACL-2025 survey re-eval
  (aclanthology 2025.findings-acl.1101). → leaves **between-prompt** difficulty as white space.
- **NAACL 2025 SAT(R)MD (arXiv:2502.14427): already implements token-level RMD vs a C4 background**
  across 11 datasets. **Most dangerous overlap.** Differentiate on: trace-level single score,
  unsupervised, length-controlled, reasoning/Best-of-N application.
- Mahalanobis OOD on transformers (AAAI 2021); single-mode Gaussian fails in near-OOD (Mahalanobis++,
  Ren-fix). Semantic Entropy (Nature 2024) — unsupervised but only catches confabulations, misses
  systematic reasoning errors. Semantic Entropy Probes (arXiv:2406.15927) — single-pass hidden-state
  probe; baseline + supports the single-trace direction.

**Trained probes / verifiers (baselines reviewers will demand we beat):**
- PCA+LDA ~80% on MATH; SWIFT (arXiv:2505.12225); TrajSelector (arXiv:2510.16449); **supervised latent
  correctness probe ~0.84 AUC on MATH (arXiv:2511.14773)**. PRMs/GenRM (arXiv:2408.15240). Our
  unsupervised RMD (0.64) **loses raw AUC to these** — must win on unsupervised/single-pass/cross-model.

**Distillation / RL reshaping representation geometry (Result 1 — MOST NOVEL):**
- No prior work links distillation/R1-RL to compression of a hidden-state *correctness* manifold.
  Adjacent only: intrinsic-dim SFT-vs-ICL / prompting / parameter-subspace (arXiv:2412.06245, 2602.09276);
  truth subspaces emerge over training (arXiv:2510.15804); **instruction tuning INCREASES intrinsic dim
  (arXiv:2402.18048 — opposite direction, must address)**. Support is analogical → we supply direct evidence.

**Cross-model transfer / PRH (Result 2 — NOVEL framing):**
- Platonic Representation Hypothesis (arXiv:2405.07987); Linear Representation Transferability
  (arXiv:2506.00653, steering vectors transfer near-causally within a family). Complemented (not
  pre-empted) by Orgad et al. (error detectors don't transfer, arXiv:2410.02707) and SEP (accuracy
  probes transfer worse OOD). "Shape transfers, readout doesn't" is novel — but support is analogical
  (LRT transfers vectors, not covariance manifolds, within one family).

**Selective prediction / abstention (Result 3 — crowded; metrics matter):**
- Established metrics = **AURCC / AUACC / Coverage@Acc / R-Acc / ER** (TACL 2025 "Know Your Limits",
  arXiv:2407.18418) — use these, NOT "AUSC". Competitors: SelectLLM, AbstentionBench (arXiv:2506.09038,
  NeurIPS 2025 — **reasoning fine-tuning DEGRADES abstention 24%**, a strong motivating hook).

**Truncation pitfall (Result 4 — weak alone):** a related difficulty-driven sample-selection artifact
in long-CoT probing is already published (arXiv:2511.14773). Fold in as methodology, not headline.

**Citation fix (blocking):** arXiv:2412.06245 = Janapati & Ji (intrinsic dim), **NOT** the universal
truthfulness hyperplane paper (that is **arXiv:2407.08582**).

---

## 8. Recommended paper (mechanism-first)

> **"How Reasoning Distillation Reshapes the Geometry of Correctness."**
> - **C1 (mechanism):** distillation compresses correctness structure into a low-dim subspace (dim~8 vs 64–128).
> - **C2 (mechanism):** correct-reasoning manifold *shape* transfers instruct↔distilled; trained *readout* does not.
> - **C3 (application):** unsupervised, single-forward-pass RMD abstention — beats length/entropy/semantic-entropy
>   *at zero supervision* (supervised probes do better but need labels); motivated by reasoning-FT degrading abstention.
> - **Rigor thread:** length baseline + truncation confound.
> - **Positioning:** trace-level (vs SAT(R)MD), unsupervised (vs PCA+LDA/SEP/SelectLLM), cross-model (vs INSIDE).

(Deviates from "lead with abstention" because RMD 0.64 < supervised 0.84 — don't headline a comparison we lose.)

---

## 9. Open questions for deep research (highest value)
1. **R1-style RL / reasoning-distillation × representation geometry (2025–2026):** any work on how it
   changes hidden-state intrinsic dimensionality or a correctness/linear subspace? (firms up C1 novelty)
2. **Cross-model transfer of a covariance-defined Mahalanobis manifold** (vs steering vectors): any
   direct prior art? Does "shape transfers, readout doesn't" hold cross-architecture (Qwen↔Llama)?
3. **Strong AUSC/AURCC/Coverage@Acc reference numbers on MATH-500/GSM8K** — what counts as SOTA, and
   does unsupervised single-pass RMD beat the supervised latent probe (~0.84) and SEP head-to-head?
4. **Generation-cap / unparseable-auto-label pitfall:** has any o1/R1 evaluation or UQ paper documented
   it explicitly (vs difficulty-selection)? Confirm/deny "first-to-document."

## 10. Gating reality (what must happen before submission)
Every load-bearing number for C1/C2/C3 was computed pre-fix on contaminated data. Required:
(a) pilot re-collection at proper budget → clean parseable data; (b) re-validate C1 (low-dim sweep),
C2 (transfer grid), C3 (risk–coverage) **parseable-only** with full baselines: length, entropy,
**Semantic Entropy Probes**, and the **supervised latent probe (~0.84)** head-to-head.

## 11. Reproduction
```bash
uv run python reproduce_findings.py          # corrected findings, cheap, no GPU (~5s)
uv run --with jupyter jupyter nbconvert --to notebook --execute --inplace notebooks/findings_corrected.ipynb
PYTHONPATH=. uv run python -m pytest tests/test_prompt_decomposition.py -q
CUDA_VISIBLE_DEVICES=<free> uv run dvc repro 'truncation_probe@0'   # GPU budget probe (deliberate)
```
