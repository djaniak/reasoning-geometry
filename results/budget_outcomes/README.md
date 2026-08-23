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
| `budget_outcomes_report.md` | population, operating-point, cap, and continuation tables |

```
uv run python -m analysis.budget_outcomes
```

`analysis/budget_outcomes.py` reads the committed `incremental_abstention`,
`continue_capped` and OOF artifacts. It re-indexes the stored deltas by
population and rebuilds the frozen prompt-level `B0` and `B1` logistic readouts
for the operating-point curves. A runtime assertion requires their AURCs to
match the locked artifact. The pointwise bands use 1,000 prompt-bootstrap draws
with the readouts fixed. The script does not refit hidden-state references or
run a model.

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
5. **The integrated gain has an operational counterpart.** At 50% abstention,
   `B1` reaches 0.876 / 0.880 / 0.796 accuracy on answered prompts, compared
   with 0.780 / 0.836 / 0.736 for `B0`.

## One denominator to state explicitly

`continue_capped` reports `accuracy_of_completions` over traces that
*terminated*, which includes a degenerate loop that ran to a stop and can never
be correct: 16/35 = 0.4571. Over traces labelled completed it is 16/34 = 0.471.
Both are defensible. The report carries both and names which is which; quoting
either without saying so is the error.
