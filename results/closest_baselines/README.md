# The two closest cheap baselines (experiments 1a and 1b)

Both pre-declared stop rules for the RMD closure path are read out of this
directory. `1a` asks whether the increment survives against the cheapest thing
that could explain it away — the answer histogram of the same eight samples.
`1b` asks *where in the trace* the score has to be measured.

AURC throughout, **lower is better**, so a negative delta favours the left-hand
readout. Every readout is the frozen cross-fitted logistic on the frozen prompt
folds, and every interval is the frozen paired prompt bootstrap over fixed OOF
predictions — it does not propagate reference refitting. That is the open gate
`controls/refit_stability.py` is registered against.

## The three features being separated

| Feature | What it is |
|:---|:---|
| `neg_answer_entropy` (`H`) | minus the Shannon entropy of the normalized exact-answer histogram over parseable siblings — self-consistency, sharpened |
| `rmd_full` | whole-trace mean of per-token RMD. This is Vazhentsev et al.'s ATRMD, i.e. **published prior art**, not a strawman |
| `rmd_tail_q20` | the same mean restricted to the final 20% of tokens — the paper's feature |

## Populations

`full_population` is the headline and is what the stop rules are evaluated on:
all 500 prompts, with unparsed traces scored as incorrect rather than dropped.
The two `cap_free_*` rows are sensitivity analyses that additionally drop any
prompt with a generation-capped sibling; they are more permissive, and reading
a delta off them overstates the primary effect by roughly 11–20% depending on
the model. Notebook 17 §1 tabulates that gap. Do not quote a `cap_free_*`
number as the result.

**2026-08-22.** `full_population` was added on this date. The two `cap_free_*`
populations reproduce the 2026-08-10 artifact to 1e-12 on every field — the
re-run is a strict superset, not a restatement.

## What they returned

`1a` does not trigger on any population: `rmd_tail_q20` over `B0 + H` clears
zero on all three models (`full_population`: −0.0519 / −0.0281 / −0.0462). The
increment is not the answer histogram in disguise.

`1b` was a branch rule, not a gate — *no region or percentile sweep follows,
whichever way this lands* — and it lands split. The tail beats the untailed
ATRMD on Qwen (−0.0464 [−0.0724, −0.0224]) and ties on both distilled models
(−0.0030 and −0.0041, intervals spanning zero). Read the other direction, which
is the more useful one: `rmd_full` alone over `B0` is −0.0287 and −0.0445 on the
distilled pair — essentially the whole increment — but only −0.0178 [−0.0435,
+0.0105] on Qwen, where it does not clear zero at all. **The increment
replicates on all three models; which localization delivers it does not.** The
tail is a Qwen-specific refinement of a published feature, and the two scores
correlate at ρ 0.88–0.94.

## Files

| File | What it is |
|:---|:---|
| `closest_baselines_results.json` | per model and population: marginal AUROCs, redundancy, all six readouts, seven paired deltas, both stop-rule verdicts |
| `closest_baselines_report.md` | the same, as tables, one section per population |

```
uv run python -m baselines.closest_baselines \
  --model qwen:results/qwen_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/qwen_bestofn_full/math500 \
  --model deepseek:results/deepseek_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/deepseek_bestofn_full/math500 \
  --model deepseek_llama:results/deepseek_llama_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/deepseek_llama_bestofn_full/math500 \
  --population full_population \
  --population cap_free_valid_plurality \
  --population cap_free_all_eight_parseable
```

Not a DVC stage. It re-reads cached OOF rows and imports the frozen folds,
populations, readout, bootstrap and seed convention from
`incremental_abstention` rather than restating them, so a change there
propagates here rather than diverging silently.
