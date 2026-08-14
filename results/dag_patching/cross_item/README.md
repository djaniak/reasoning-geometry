# Cross-item donor control

Four runs of the cross-item donor control, seeds 0-3, at depth 1 / gap 0. See
`EXPERIMENT_LOG.md`, 2026-08-14, "The cross-item donor control fails its
specificity leg."

Every other edit in this project rewrites the recipient's *own* trace, so none
of them touches the sharpest objection to the ancestor gap: those two token
positions might simply be perturbation-sensitive. This arm writes **another
item's** residual state at the same positions — same span, same token width,
same formatting — under a derangement, so no item donates to itself.

| File | Seed |
|:---|:---|
| `depth1_gap0.json` | 0 |
| `depth1_gap0_seed{1,2,3}.json` | 1, 2, 3 |

```
uv run python dag_patching.py \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --condition both --seed 0 --n_items 5 --depth 1 --gap 0 \
  --generator v2_paired --cross_item \
  --output results/dag_patching/cross_item/depth1_gap0.json
```

Run commit `140c335`. Each report records `cross_item`, `donor_map`,
`generator` and `n_decoys` directly, so nothing here needs the provenance
recovery the archived eight required. These runs are not in the archived
package's `MANIFEST.json`.

## The batch is not the ladder's

`--cross_item` selects a mutually donatable batch out of a wider candidate
pool. Two selections happen, and they differ in kind:

- **by ancestor line position** — so donor and recipient share the patched
  positions. Formatting only; nothing the readout measures depends on it.
- **by value compatibility** — the donor's value carried through the
  recipient's chain has to land on a digit, and not on the clean answer. That
  depends on sampled values. It is a constraint of the ten-way readout rather
  than a preference over outcomes, but it does mean this arm is **not** the
  same value distribution as the depth ladder, and its numbers are not
  comparable to it item by item.

## What it predicts

The chain is affine, so donor value `v_j` read through recipient `i`'s chain
implies `v_j + delta_i` — neither the clean answer nor the donor's own digit.
Batch selection keeps those three digits distinct, which is what separates
"propagated the donor's value" from "copied the patched token" from "did not
move". `delta_toward_raw` in the rows is the third of these; it needs the clean
log-odds and so cannot be recovered by a rescore.

Eligibility is decided by the spine alone, not by walking the chain — walking
it would range-check intermediate values, which is depth-dependent and would
hand a different donor map to each depth, exactly the desynchronisation the
`v1_unpaired` generator suffered.

## Scoring

The `cross_item_donor` gate is registered with the joint-layer rule from the
start: it has no archived verdict to preserve, so there was no reason to repeat
the `any(layer)` mistake. It is **reported beside the verdict and never binds
it** — the verdict space says whether the *within-item* intervention was valid,
and folding a new statistic into it before its null is known is the post-hoc
move the previous two checkpoints were spent undoing.
