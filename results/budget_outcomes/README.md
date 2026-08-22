# Budget-indexed outcomes

`C_B` for the RMD prompt-level result: correctness available by the generation
budget, under the fixed decoding and extraction rule, on all 500 prompts. A
prompt with no parseable answer by `B` is an observed failure, not a missing
value.

The headline population in `incremental_abstention` is
`cap_free_valid_plurality`, which drops any prompt with a capped sibling. That
filter conditions on an event correlated with difficulty, so it estimates
correctness *given the cap was avoided*. These tables put the four populations
side by side so the paper can report the unconditional number as primary and
name the conditional one as conditional.

## Files

| File | What it is |
|:---|:---|
| `budget_outcomes.json` | every number below, machine-readable |
| `budget_outcomes_report.md` | the four tables and the reporting rule |

```
uv run python -m analysis.budget_outcomes
```

Nothing here is fitted. `analysis/budget_outcomes.py` reads the committed
`incremental_abstention`, `continue_capped` and OOF artifacts, re-indexes them
by population, and writes the tables; it loads no model and runs no bootstrap
of its own. The paired deltas are the ones already stored in the locked
artifact, so any disagreement between this report and that artifact is a bug
here, not a new result.

## What it establishes

1. **The increment survives `C_B` on all three models.** Every `B1 - B0` AURC
   interval on `full_population` excludes zero.
2. **The headline population overstates it** — by 11% on Qwen, 20% on
   DeepSeek-Qwen, 16% on Llama. Deltas computed under peer control or any other
   attenuation must be taken against the `full_population` base, or the two
   corrections are applied to different denominators and will not compose.
3. **Capping tracks difficulty, measurably.** Traces that hit the budget but
   still carried an answer are far less accurate than traces that finished
   (Qwen 0/11; DeepSeek 0.553 vs 0.771; Llama 0.286 vs 0.587). That is the
   mechanism the complete-case filter was silently exploiting.
4. **`C_{B->B'}` is one model's case study.** 50 resumed DeepSeek traces, never
   a label for the other models, the unsampled capped traces, or the dataset.

## One denominator to state explicitly

`continue_capped` reports `accuracy_of_completions` over traces that
*terminated*, which includes a degenerate loop that ran to a stop and can never
be correct: 16/35 = 0.4571. Over traces labelled completed it is 16/34 = 0.471.
Both are defensible. The report carries both and names which is which; quoting
either without saying so is the error.
