# Missing primary-population controls: what to run before submission

Written 2026-09-07 during the preprint-readiness repair. **Nothing here was run.**
Each entry gives the exact command, the inputs it needs, the outputs it writes, an
estimated cost, and the retained claim that depends on it.

## Why these exist

`EXPERIMENT_LOG.md:884` adopted `full_population` as the primary estimand and fixed
the composition rule: deltas taken under any attenuation must share that base, or two
corrections land on different denominators and will not compose. Five controls still
default to `--population cap_free_valid_plurality` and have only ever been run there.
That is a reporting gap, not a measurement error — each is a legitimate result on the
population it ran on. But a control on n≈392 cannot be cited in support of a headline
on n=500 without either re-running it or saying which population it came from, and
the manuscript will want the former.

All five re-read cached OOF rows and import the frozen aggregation, folds, readout,
bootstrap and seed convention. None is a DVC stage; none re-generates traces or
hidden states; none touches `.dvc/cache`. Precedent for the cost estimate is the
2026-08-22 re-run of experiments 1a/1b on the primary population, recorded at four
CPU-minutes.

Shared inputs, all verified present on 2026-09-07:

```
OOF   results/{qwen,deepseek,deepseek_llama}_bestofn_full/math500/math500_prompt_decomposition_oof.csv   (6.6-6.7 MB each)
DATA  data/{qwen,deepseek,deepseek_llama}_bestofn_full/math500
```

Use `.venv/bin/python`; bare `python` is not on PATH. Run from the repo root.

---

## 1. Vote-proxy control (Orgad et al.) — **highest priority**

**Claim it supports.** "A vote-proxy control answering Orgad et al. (arXiv:2410.02707)":
that `rmd_tail_q20` is not a worse-instrumented `vote_agreement`. This is the control a
reviewer who has read Orgad will look for first, and it is the one whose report never
names its population at all.

```bash
.venv/bin/python controls/orgad_agreement_control.py \
  --output_dir results/orgad_agreement_control_full_population \
  --population full_population \
  --model "DeepSeek-Qwen:results/deepseek_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/deepseek_bestofn_full/math500" \
  --model "Llama:results/deepseek_llama_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/deepseek_llama_bestofn_full/math500" \
  --model "Qwen:results/qwen_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/qwen_bestofn_full/math500"
```

**Writes** `orgad_agreement_control_{report.md,results.json}` in that directory.
**Cost** ~2-5 CPU-minutes, no GPU.
**Expect** the unanimous-stratum denominators to change from 274/392, 349/393, 214/408
to shares of 500, which is what fixes the `README.md` unanimity sentence at source.
**Note** a separate `--output_dir` is used so the existing `cap_free` artifact is not
overwritten; keep both and report the primary one.

## 2. Cross-model peer difficulty control

**Claim it supports.** "The increment is attenuated by problem difficulty, not
eliminated by it" — including the pre-declared stop rule and its Holm table, which are
currently evaluated on `cap_free_valid_plurality` only. This control accepts repeated
`--population`, so the primary population can be added as the headline while the
existing one is retained as sensitivity.

```bash
.venv/bin/python controls/peer_difficulty_control.py \
  --output_dir results/peer_difficulty_control_full_population \
  --population full_population --population cap_free_valid_plurality \
  --model "qwen:results/qwen_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/qwen_bestofn_full/math500" \
  --model "deepseek:results/deepseek_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/deepseek_bestofn_full/math500" \
  --model "deepseek_llama:results/deepseek_llama_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/deepseek_llama_bestofn_full/math500"
```

**Writes** `peer_difficulty_control_{report.md,results.json}`; the first `--population`
becomes the headline the stop rule and Holm table are evaluated on.
**Cost** ~1-2 CPU-hours (the 2026-08-09 estimate at `EXPERIMENT_LOG.md:3215`).
**Watch for** whether `stop_rule.triggered` stays `false` on the primary population.
It is `false` on `cap_free` with exactly one model overlapping zero, so this is the
one re-run that could change a retained claim rather than only its denominator.

## 3. Allocation precheck

**Claim it supports.** "It does not extend to sample allocation" — the Spearman
figures, the pass-rate correlations, and the fail/fail/PASS gate verdict quoted in
`README.md`.

```bash
.venv/bin/python applications/allocation_precheck.py \
  --output_dir results/allocation_precheck_full_population \
  --population full_population \
  --model "qwen:results/qwen_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/qwen_bestofn_full/math500" \
  --model "deepseek:results/deepseek_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/deepseek_bestofn_full/math500" \
  --model "deepseek_llama:results/deepseek_llama_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/deepseek_llama_bestofn_full/math500"
```

**Writes** `allocation_precheck_{report.md,results.json}`.
**Cost** ~10-30 CPU-minutes; it enumerates `C(8,k)` sibling subsets over 500 prompts
and runs a bootstrap over stage-1 draws, so it is the slower of the two CPU-minute-class
jobs.
**Note** this is a claim the paper reports as a *negative*. Re-running it on a larger,
harder population is more likely to strengthen the negative than to weaken it, so the
risk here is low — but the gate verdict must be re-read, not assumed.

## 4-5. DeepConf, two forms

**Claim they support.** "DeepConf (arXiv:2508.15260) as a prompt-level score, as a
confidence-weighted vote, and as a confidence filter, with all four of its statistics."

**These can never become three-model controls.** The exact DeepConf statistic requires
teacher-forced cached token IDs, and `data/qwen_bestofn_full` stores no token arrays
(`EXPERIMENT_LOG.md:4398`). Re-running on the primary population fixes the population
gap on the two models that can carry it; it does not fix the coverage gap, which is
permanent and must be stated in the manuscript rather than worked around.

```bash
.venv/bin/python baselines/deepconf_asymmetry.py \
  --output_dir results/deepconf_asymmetry_full_population \
  --population full_population \
  --model "deepseek_qwen:results/deepseek_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/deepseek_bestofn_full/math500:results/deepseek_bestofn_full/math500/deepconf_exact_full/deepconf_exact_full.npz:results/deepseek_bestofn_full/math500/math500_incremental_abstention_results.json" \
  --model "llama:results/deepseek_llama_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/deepseek_llama_bestofn_full/math500:results/deepseek_llama_bestofn_full/math500/deepconf_exact_llama/deepconf_exact_llama.npz:results/deepseek_llama_bestofn_full/math500/math500_incremental_abstention_results.json"

.venv/bin/python baselines/deepconf_weighted_vote.py \
  --output_dir results/deepconf_weighted_vote_full_population \
  --population full_population \
  --model "deepseek_qwen:results/deepseek_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/deepseek_bestofn_full/math500:results/deepseek_bestofn_full/math500/deepconf_exact_full/deepconf_exact_full.npz" \
  --model "llama:results/deepseek_llama_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/deepseek_llama_bestofn_full/math500:results/deepseek_llama_bestofn_full/math500/deepconf_exact_llama/deepconf_exact_llama.npz"
```

**Writes** `deepconf_asymmetry_{report.md,results.json}` and
`deepconf_weighted_vote_{report.md,results.json}` in their directories.
**Cost** ~5-15 CPU-minutes each; both read a merged exact-DeepConf npz that already
exists, so no teacher-forcing pass is repeated.
**Inputs verified present** on 2026-09-07:
`results/deepseek_bestofn_full/math500/deepconf_exact_full/deepconf_exact_full.npz`,
`results/deepseek_llama_bestofn_full/math500/deepconf_exact_llama/deepconf_exact_llama.npz`,
and the three `math500_incremental_abstention_results.json`. Note the abstention JSONs
are **not git-tracked**, so a fresh clone cannot run `deepconf_asymmetry` until they are
regenerated or committed.
**Already partly clear, in the JSON only.** `deepconf_weighted_vote_results.json`
already carries a `full_population` block, and on it the DeepConf-controlled increment
`B1 − (B0 + dcvote_deepconf_tail_q20)` in AURC clears on both models that can run it:

| model | `full_population` | `cap_free_valid_plurality` |
|---|---|---|
| DeepSeek-Qwen | −0.0285 [−0.0526, −0.0043] p=0.022 | −0.0356 [−0.0629, −0.0102] p=0.006 |
| Llama | −0.0463 [−0.0778, −0.0186] p=0.002 | −0.0561 [−0.0930, −0.0207] p=0.002 |

So this particular contrast is already available on the primary population and does not
need the re-run; what the re-run buys is a **report** that presents it, since the
committed `deepconf_weighted_vote_report.md` does not. `deepconf_asymmetry` has no
`full_population` block at all and does need the run.

Before citing either, reconcile against `EXPERIMENT_LOG.md:4395`, which records that
"once DeepConf is *added to B0*, the remaining geometry margin is not significant at
this sample size on the clean population." That statement is about a different rung —
DeepConf added as a raw feature, not as a confidence-weighted vote — and about the
`cap_free` population. The two are not in conflict on their face, but the manuscript
must not quote the significant weighted-vote contrast as though it settles the
raw-feature one. Establish which rung each sentence refers to before writing either.

---

## Also outstanding, and not in this file's scope

* **The registered tail-window sensitivity run** (`EXPERIMENT_LOG.md:367`). Registered
  rule, registered command, `results/rmd_window_sensitivity/` does not exist. An
  in-memory replication during the 2026-09-06 audit is not this artifact and must not
  be cited; see the 2026-09-07 log entry.
* **The peer-residual half of the refit sweep**: qwen at seeds 101 and 202, then the
  peer step at each of the three seeds. ~2.5 h at `peak_gb=8`, needs the layer-cache
  memmap and `PR_SET_THP_DISABLE` set. This is the multi-hour job; run it in tmux.
* **Seed 303**, the fourth registered seed, untouched.
