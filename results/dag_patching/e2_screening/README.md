# E2 stage A — screening

Clean forward passes and nothing else. There is no patched number in this
directory and no code path in `dag_screening.py` that could produce one: stage A
decides *which items are comparable*, and it has to be able to say so without
having seen how any of them respond to a patch.

The protocol is registered in `EXPERIMENT_LOG.md`, 2026-08-15, written before
any of these items existed.

## Why it exists

The depth ladder collapses after depth 1, and depth is confounded with two
things that move with it. In the archived runs the eligible clean `p(target)`
supports do not touch — 0.666–0.961 at depth 1 against 0.966–0.999 at depth 2 —
so every depth-1 success is on an item the model was unsure of and every depth-2
failure on one it was sure of. `ancestor_distance` is `{11, 24}` against
`{23, 36}`. Stage A screens a large pool at both depths and finds the items where
those two quantities overlap.

## Files

| File | What it is |
|:---|:---|
| `depth1.json`, `depth1_more.json` | 630 depth-1 items, seeds 11-31, gaps 0/1/2 |
| `depth2.json`, `depth2_more.json` | 600 depth-2 items, seeds 11-40, gap 0 |
| `SELECTION.json` | all 1,230 screened items, plus the registered selection over them |

Seeds are disjoint from 0-3, which the archived runs used. Depth 1 is screened at
three gap placements because its ancestor distances start below depth 2's; the
gap sweep is a distance sampler, and one placement per spine survives selection.

```
uv run python dag_screening.py --depths 2 --gaps 0 \
  --seeds 11 12 13 14 15 16 17 18 19 20 --n_items 20 \
  --output results/dag_patching/e2_screening/depth2.json

uv run python dag_screening.py --depths 1 2 \
  --screened results/dag_patching/e2_screening/depth{1,1_more,2,2_more}.json \
  --output results/dag_patching/e2_screening/SELECTION.json
```

The second command runs no model. Selection is pure and re-derivable from the
screened records alone.

## What it found

`float32`, so the readout is off the 0.125-nat bfloat16 grid.

| depth | screened | eligible | clean ties | p(target) min / median / max |
|:---|---:|---:|---:|:---|
| 1 | 630 | 480 | **0** | 0.430 / 0.707 / 0.990 |
| 2 | 600 | 599 | **0** | 0.684 / 0.992 / 1.000 |

Not one tied clean readout in 1,230 items, against 5 of 33 in bfloat16. That is
the correction in `EXPERIMENT_LOG.md` confirmed from the other direction: the
ties were the recording precision, not the model.

Window `(0.684, 0.990)`. **24 matched pairs**, at the registered ceiling and
above the floor of 16, so the registered decision is to proceed to stage B.
Every pair is matched to within **0.0007** in clean `p(target)` and **1 token**
of ancestor distance, over 24 distinct spines per depth.

## One thing to know about the rule

The registered rule bounds ancestor distance to ±2 tokens but bounds confidence
only by *ordering the greedy*. On the first 410-item screen that showed: greedy
filled to the ceiling with pairs as far apart as 0.165 in `p(target)`, which is
not matched in any sense that matters here. Tripling the screen removed the
problem — the worst pair is now 0.0007 — so the rule was left exactly as
registered rather than amended. If a future screen is small, this hole is still
there.
