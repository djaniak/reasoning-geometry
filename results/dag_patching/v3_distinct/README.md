# The v3_distinct family

The depth ladder and the cross-item donor control, rerun on the generator that
keeps all three competing digits apart. See `EXPERIMENT_LOG.md`, 2026-08-14,
"The floor changes four verdicts and v3 confirms depth 1 on a well-posed
family."

`v2_paired` keeps the *implied* value off the clean answer but nothing kept the
**raw** digit -- the one standing at the patched result position, which a
readout that copies what it finds there would emit -- off it. Where they
coincide, "copied the digit" and "did not move" are the same prediction and the
item cannot answer the question. That happened in 2 of 20 ancestor items and 1
of 20 cross-item items under `v2_paired`. Under `v3_distinct` it happens in
none, which is what makes the counts below whole-batch rather than filtered.

The extra rejection moves the random stream, so this is a different family from
`v2_paired`, not a corrected version of it. Item *i* here is not item *i*
there, and the two are not comparable item by item. `v2_paired` stays reachable
and unchanged.

| File | Arm |
|:---|:---|
| `depth{1,2,3}_gap0.json` | ladder, depth sweep |
| `depth1_gap{1,2}.json` | ladder, distance-matched controls |
| `cross_seed{0,1,2,3}.json` | cross-item donor control, four seeds |

```
uv run python dag_patching.py \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --condition both --seed 0 --n_items 5 --depth 1 --gap 0 \
  --generator v3_distinct \
  --output results/dag_patching/v3_distinct/depth1_gap0.json
```

Run commit `af0870f`. Seed 0 for the ladder, seeds 0-3 for the cross-item arm,
`n_items 5`, `n_decoys 6`, `condition both`. Not in `MANIFEST.json`, which
covers only the eight archived, provenance-recovered runs; every field those
had to infer is recorded directly here.

## What these are for

Two things, and they are independent.

**The ladder is scored under the absolute-effect floor from the start.** It
reproduces on a new family what the rescore showed on the old one: depth 1
replaces the answer at every gap, and depth 2 and depth 3 are scientific
negatives whose clean answer keeps 0.96 and 0.99 of the readout. The floor is
not what makes them negative -- the measurement was always that -- but it is
what makes the verdict say so.

**The cross-item arm's donor batch is selected on the same rule**, threaded
into donor eligibility, because the raw digit there comes from another item and
the within-item rejection cannot reach it. As before, this arm's value
distribution is not the ladder's and its numbers are not comparable to it item
by item.

## Reading the mass, not the log-ratio

The informative statistic is which digit the patched readout actually lands on,
not `delta_toward` minus `delta_toward_raw`. That margin is a ratio against a
clean baseline which itself varies by digit, and it tracks how far apart the two
candidate digits happen to sit rather than which one won. `probs_patched` in
the rows makes the direct comparison available to any rescore.
