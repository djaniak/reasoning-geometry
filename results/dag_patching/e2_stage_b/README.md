# E2 stage B — the patched run over the matched pairs

The 24 pairs stage A chose, patched at layer 13. Protocol registered in
`EXPERIMENT_LOG.md`, 2026-08-15, in `6f1e9a7` — before the selection rule was
written, before the items existed, and before any of these numbers did.

Stage A is next door in `../e2_screening/`. It holds clean forward passes only,
and the selection it holds was computed before this directory existed.

## Files

| File | What it is |
|:---|:---|
| `depth1.json`, `depth2.json` | one arm per depth, 24 items each, in the archived row schema |
| `ANALYSIS.json` | every registered reading of the two arms, plus `primary.exact_paired` |

```
CUDA_VISIBLE_DEVICES=1 uv run python dag_stage_b.py \
  --selection results/dag_patching/e2_screening/SELECTION.json \
  --n_decoys 6 --output_dir results/dag_patching/e2_stage_b
```

`ANALYSIS.json` re-derives from the two arm files by `dag_stage_b.analyse` with
no GPU, which is how `primary.exact_paired` was added on 2026-08-15 after the run:

```
uv run python -m dag.dag_stage_b \
  --reanalyse results/dag_patching/e2_stage_b \
  --output_dir results/dag_patching/e2_stage_b
```

The regenerated file is identical to the one the run wrote apart from that one
added key, checked field by field. The arm files are read and never written. Every item was checked against its stage-A
measurement before being patched — ancestor distance, target value, gap, and the
clean readout to 1e-6 — because the screening file records `(depth, seed, index,
gap)` but not `n_decoys`, and `n_decoys` changes the trace.

## The result

The depth contrast survives matching, at full strength.

| depth | n | implied uniquely top | ties | clean p(target) min / median / max | ancestor distance |
|---:|---:|:---|---:|:---|:---|
| 1 | 24 | **24/24** | 0 | 0.696 / 0.914 / 0.990 | 24–37 |
| 2 | 24 | **0/24** | 0 | 0.696 / 0.913 / 0.990 | 23–36 |

Difference 1.00. Three readings of it, and only the first was registered:

| reading | value | what it assumes |
|:---|:---|:---|
| paired bootstrap over whole pairs, 1,000 replicates | interval [1.000, 1.000] | nothing; it has nothing to resample |
| **exact paired, one-sided** | **p = 2⁻²⁴ = 5.96e-8** | 24 matched pairs |
| Fisher's exact, one-sided | p = 3.1e-14 | 48 independent observations |

**Quote 5.96e-8.** The interval is degenerate because the separation is perfect,
not because the estimate is precise: with 24 pairs and no discordant one there is
nothing for a resampler to vary, and zero width means "no item went the other
way", not a tight bound. Fisher's exact is smaller by six orders of magnitude
because it treats the two arms as independent samples, which a design that
matched them item for item is not. The exact paired test — the sign test on the
discordant pairs, which is exact McNemar — is the one this design licenses.
Neither p-value was registered; the registration named the bootstrap and no
p-value at all.

This is the second of the three registered outcomes. What it establishes:

> **The one-versus-two-step contrast persists after matching on clean confidence
> and ancestor token distance.**

Those two are now matched — the depth-1 and depth-2 confidence quantiles agree to
three decimals and the distances overlap — and the rates did not move toward each
other at all. It does **not** establish that the contrast is about graph depth.
Depth bundles an added operation, a written intermediate result, a new variable
binding and a changed local context, and matching two observed covariates removes
two rival explanations without unbundling the rest. `dag_tasks` states the bundle
in its module docstring; the ladder measures whether a patched state still moves
the answer when a written intermediate contradicts it.

## The validity gate, and a disagreement it does not settle

The registered gate is null flips: an arm whose nulls move the answer on 20% or
more of its null rows is an invalid test. Neither arm is close.

| depth | null | surface_null | non_ancestor | ancestor |
|---:|:---|:---|:---|:---|
| 1 | 0/144 | 0/24 | 0/24 | 24/24 |
| 2 | 0/144 | 0/24 | 0/24 | 0/24 |

`control_specificity` at layer 13: depth 1 lands on the implied digit 24/24 with
0/192 control rows moving at all. Depth 2 lands on it 0/24, also against 0/192.

**The arm scorer disagrees about depth 2, and the disagreement is recorded rather
than resolved.** `dag_patching` scores `depth2.json` as an *invalid test*, on
`directional_control_failed` and `surface_above_null` — its gates are relative,
and at depth 2 there is no movement for a relative gate to be relative to. By the
gate registered for E2 the arm is valid and negative; by the arm scorer it is
unreadable. Both labels are in the artifacts.

They answer different questions. The arm scorer asks whether one arm's own gates
can be read, and a manipulation that does nothing leaves them unreadable. E2 asks
whether the ancestor edit installs its digit at one depth and not the other, and
"it does nothing at depth 2" is that measurement, not an obstacle to it. The
registration named the null-flip gate as stage B's validity criterion and did not
anticipate the arm scorer reaching a different verdict on the same rows; that it
did is a limitation of the registration, and the honest reading is that the
contrast stands on the paired comparison while the depth-2 arm, taken alone,
is not independently scoreable.

## The depth-2 patch is not inert

Median total-variation distance at layer 13, clean to patched:

| depth | ancestor | non_ancestor | null | surface_null |
|---:|---:|---:|---:|---:|
| 1 | 0.9868 | 0.0219 | 0.0043 | 0.0051 |
| 2 | 0.0877 | 0.0075 | 0.0046 | 0.0041 |

At depth 1 the transplanted state replaces the readout outright. At depth 2 it
moves it about twenty times as much as a null edit does and roughly a tenth as
much as at depth 1 — so the state is reaching the read position and having an
effect there. What it does not do is put the implied digit anywhere near the top.

The four-way level split, medians over the 24 eligible items per depth:

| depth | p(implied) | p(raw) | p(target) | remaining |
|---:|:---|:---|:---|:---|
| 1 | 0.0006 → 0.8579 | 0.0005 → 0.1065 | 0.9136 → 0.0023 | 0.0781 → 0.0150 |
| 2 | 0.0006 → 0.0013 | 0.0006 → 0.0028 | 0.9132 → 0.8040 | 0.0777 → 0.1616 |

Two readings a reader will want:

- **The donor's literal digit is promoted at depth 1 too**, 0.0005 → 0.1065,
  about two hundredfold. It is never uniquely on top (0/24), so the transformed
  digit wins every time — but "the recipient transforms the donor value" stays
  too clean a sentence, exactly as the 2026-08-15 correction said.
- **At depth 2 the displaced mass does not go to the implied digit.** p(target)
  falls 0.913 → 0.804 and the remaining-digit mass roughly doubles, 0.078 →
  0.162, while p(implied) stays at 0.001. The patch adds noise, not an answer.

## Limits

- **Layer 13 is inherited** from the `v3_distinct` discovery table, not
  re-searched. This run is confirmatory for the depth *contrast*. The depth-1
  rate is not a fresh test of anything, because the layer was chosen on data that
  produced it.
- **High-confidence regime only.** The matched window is p(target) 0.696 to
  0.990, median 0.914. That covers the upper half of the range the original
  depth-1 result was obtained in (0.53–0.96) and not the lower half.
- **Four row kinds, not five — a deviation from the registered protocol.** The
  registration named all five; a cross-item batch is selected for mutual
  donatability, so it is a different batch from a plain run at the same seed and
  the matched items do not occur in one. Recorded in `unreachable_row_kinds` in
  both arms. It does not touch the registered ancestor outcome, but it leaves the
  matched run without its strongest portable-state control — the one that would
  show the transplanted state carries content *across* items rather than within
  one. The cross-item donor claim is exploratory in the registration and is
  untouched by this run.
- **24 pairs, 24 distinct spines per depth**, seeds 11–40, disjoint from the
  archived 0–3.
- **float32 is a model run, not a readout.** The dtype reaches `from_pretrained`,
  so every matmul in the forward pass ran at it. Both depths ran at the same
  precision, which is what E2's internal validity needs, but this is **not** a
  same-precision replication of the archived bfloat16 depth-1 result and these
  counts may not be pooled with the eight archived arms — `dag_pooling.pool`
  refuses to. The readout itself was float32 in every run, archived ones
  included: `digit_readout` casts before it softmaxes.
- Nothing here speaks to depth 3, to the omission arms, or to any mechanism.
