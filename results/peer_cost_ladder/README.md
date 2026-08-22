# Peer baseline and cost ladder

The 2026-08-21 review's second rung. `peer_difficulty_control` says `B0 + peer`
is "a control, never a baseline the headline has to beat"; the review rejects
that, because a scale-matched peer ensemble is a deployable uncertainty method
in the literature and a reviewer may read it as one. This puts both on a cost
axis instead of arguing about the label.

## The cost model

Every rung pays the target's eight generations. `B1` adds `rmd_tail_q20` read
from the hidden states of *those same* generations: **zero extra calls, zero
extra tokens**. A peer rung at `m` samples from `k` peers adds `k*m` calls and
the corresponding tokens. So no peer rung is cost matched to `B1` — the
cheapest purchasable thing is one extra generation, and `B1` is free at the
margin.

## The distinction the frozen control was missing

The peer feature it uses is the fraction of a peer's siblings that were
**correct**. That needs the gold answer, so it cannot be computed at decision
time. It bounds the peer family from above; it is not a baseline anyone could
deploy. The ladder therefore reads each purchase two ways:

| Kind | Feature | Deployable |
|:---|:---|:--|
| `graded` | fraction of the peer's samples that were correct | no — needs gold |
| `agree` | fraction of the peer's samples returning the target's own answer | yes |

Same generations, same price, different readout.

## Files

| File | What it is |
|:---|:---|
| `peer_cost_ladder_results.json` | every rung, cost, delta and flag |
| `peer_cost_ladder_report.md` | five tables: floors, the ladder in cost order, deltas against `B1`, the one-extra-generation question, saturation |

```
uv run python -m controls.peer_cost_ladder \
  --model {qwen,deepseek,deepseek_llama}:results/{model}_bestofn_full/math500/math500_prompt_decomposition_oof.csv:data/{model}_bestofn_full/math500
```

Three CPU minutes. Not a DVC stage; it re-reads cached OOF rows and imports the
frozen aggregation, folds, populations, readout, bootstrap and seed convention.

**Continuity check.** At `k=2, m=8` the graded rung is `peer_difficulty_control`'s
`B0_plus_peer`. On `cap_free_valid_plurality` the ladder reproduces the frozen
`B0`, `B1` and `B0_plus_peer` AURCs to six decimals on all three models.
Disagreement there is a bug here, not a new result.

## What it shows

1. **The peer control's dominance was substantially a graded-readout artifact.**
   At the frozen control's own cost (16 extra generations, both peers), the
   graded rung beats `B1` by 0.0561/0.0946/0.0295 AURC; the deployable
   agreement rung at the identical cost gives 0.0409/0.0169/0.0569 — it beats
   `B1` on Qwen and Llama and ties on DeepSeek-Qwen, a different pattern with a
   different model flipping.
2. **At one extra generation, deployable peers do not establish a win.** Six
   target/peer pairs: four ties, one `B1` win (DeepSeek-Qwen against a
   DeepSeek-Llama peer, −0.0341 [−0.0576, −0.0108]), one peer win (Llama
   against a Qwen peer, +0.0544 [+0.0240, +0.0838]).
3. **Agreement with a weak peer can be worse than nothing.** On DeepSeek-Qwen,
   the DeepSeek-Llama agreement feature raises AURC above `B0` at every sample
   count — negative headroom removed. Buying a peer is not monotonically good.
4. **The DeepSeek-Qwen graded comparison is saturated.** `B0_graded_both_m{2,4,8}`
   remove 90–95% of `B0`'s headroom, sitting on the oracle floor. There, "the
   peer wins" and "there was nothing left to remove" are not distinguishable,
   which is the review's own caution, now localised to the graded family.

## Two uncertainty sources, kept apart

The interval is the frozen prompt bootstrap with the pipeline held fixed, taken
on the median draw. The spread across 25 independent re-draws of *which*
siblings were bought is reported separately, as `sign stable`. Folding them
together would make these intervals incomparable with every other interval in
the paper; combining them is the outer-refit rung's job.

## What is not charged for

`B1` needs the hidden states retained and a Mahalanobis readout over them —
real work, but not a generation, and it does not scale with how many models you
are willing to run. The `graded` rungs are not charged for the gold answer they
consume, because it cannot be bought at decision time at any price.
