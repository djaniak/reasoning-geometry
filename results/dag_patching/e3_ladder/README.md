# E3 at the registered N, with the chain-node arm

Two questions, one set of runs. 48 items per cell, seeds 0/1/2, `v3_distinct`,
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` at float32, layers 6/13/20/27 with 27
dropped from scoring.

**E3.** The paired ladder's numbers came from 5 items at one seed. E3's spec was
40–60 items per cell and at least three seeds. This is that, for all five arms.

**The chain arm.** The depth ladder patches the ancestor and leaves every written
intermediate clean, so an ancestor that does nothing at depth 2 has two readings:
no value crossed the step, or none crossed those tokens. `--chain_edits` patches
the intermediate itself — the same two positions, the same donor arithmetic, the
same clean readout, one step from the target instead of `depth`. The ancestor and
the intermediate are then compared **inside one item**, which is what the depth
ladder cannot do.

## Files

| File | What it is |
|:---|:---|
| `depth{1,2,3}_gap{0,1,2}_seed{0,1,2}.json` | one arm per cell, in the archived row schema |
| `*.log` | the gate table each run printed, including its chain rows |
| `ANALYSIS.json` | every reading below, plus the per-site records |

```
uv run python -m dag.dag_patching \
  --n_items 48 --n_decoys 6 --generator v3_distinct \
  --depth D --gap G --seed S [--chain_edits] \
  --output results/dag_patching/e3_ladder/depthD_gapG_seedS.json
```

`--chain_edits` on depth 2 and 3 only; at depth 1 there is no intermediate and
the flag is a no-op, which makes depth 1 the contrast's own control. Every arm
regenerates from the settings its report records — checked on
`depth3_gap0_seed0` against tokens, gap, target value, chain nodes and ancestor
distance.

`ANALYSIS.json` and every table below are re-derived from the arm files by
`dag/dag_e3_ladder.py`, which loads no model:

```
uv run python -m dag.dag_e3_ladder
```

It reads the arms and writes `ANALYSIS.json`; `--output -` prints the tables and
writes nothing. The layer and the primary outcome are stage B's, imported rather
than restated, and `tests/test_dag_e3_ladder.py` asserts the three copies of the
layer agree.

## Reading

Stage B's registered outcome, unchanged so the numbers mean what its numbers
mean: **the donor-implied digit alone on top at layer 13**, among items whose
clean answer was alone on top to begin with. `steps_to_target` counts written
lines on the path from the patched line to the target; `distance_to_read` counts
tokens. The two fall together as depth grows, which is what the gap arms exist to
separate.

## The result

1,152 patch sites over 720 items; 1,035 sit on an item whose clean answer was
alone on top (90%) and are what the rates below are over.

| site | steps | tokens | implied alone on top | clean held | median TV | per seed |
|:---|---:|:---|:---|:---|---:|:---|
| depth 1 gap 0, ancestor | 1 | 11–24 | **104/117 (88.9%)** | 0/117 | 0.987 | 35/39 · 33/37 · 36/41 |
| depth 1 gap 1, ancestor | 1 | 24–37 | **87/104 (83.7%)** | 0/104 | 0.983 | 32/39 · 23/29 · 32/36 |
| depth 1 gap 2, ancestor | 1 | 37–50 | **76/94 (80.9%)** | 0/94 | 0.982 | 27/32 · 22/28 · 27/34 |
| depth 2 gap 0, chain `n` | 1 | 11 | **144/144 (100%)** | 0/144 | 0.996 | 48/48 · 48/48 · 48/48 |
| depth 2 gap 0, ancestor | 2 | 23–36 | **0/144 (0%)** | 144/144 | 0.035 | 0/48 · 0/48 · 0/48 |
| depth 3 gap 0, chain `n` | 1 | 11 | **144/144 (100%)** | 0/144 | 0.999 | 48/48 · 48/48 · 48/48 |
| depth 3 gap 0, chain `m` | 2 | 23 | **0/144 (0%)** | 144/144 | 0.090 | 0/48 · 0/48 · 0/48 |
| depth 3 gap 0, ancestor | 3 | 35–48 | **0/144 (0%)** | 144/144 | 0.003 | 0/48 · 0/48 · 0/48 |

### Steps, not tokens

Sites banded by token distance. Every band that holds both a one-step and a
multi-step site splits on the step count and not on the band.

| tokens to the read position | 1 step | 2 steps | 3 steps |
|:---|:---|:---|:---|
| 0–15 | 338/344 (98.3%) | — | — |
| 16–30 | 97/113 (85.8%) | **0/217** | — |
| 31–45 | 77/96 (80.2%) | **0/71** | **0/73** |
| 46–60 | 43/50 (86.0%) | — | **0/71** |

A one-step patch still lands the implied digit 86% of the time at 46–60 tokens.
A two-step patch lands it 0% of the time at 16–30. The gap ladder was already
the control for this; what the chain arm adds is a two-step site at 23 tokens —
*nearer* the read position than the one-step sites that work — which is the cell
that had never been measured.

### The same trace, two patch sites

Paired on the item, so the clean readout, the token count, the null spread and
the surface control are all held fixed. The sign test is over discordant pairs.

| contrast | n | chain only | ancestor only | both | neither | p |
|:---|---:|---:|---:|---:|---:|:---|
| depth 2: ancestor (2 steps) vs chain (1 step) | 144 | **144** | 0 | 0 | 0 | 9.0e-44 |
| depth 3: ancestor (3 steps) vs chain (1 step) | 144 | **144** | 0 | 0 | 0 | 9.0e-44 |
| depth 3: ancestor (3 steps) vs chain `m` (2 steps) | 144 | 0 | 0 | 0 | **144** | 1.0 |

The third row is the one that says the effect is a cliff and not a decay: two
steps and three steps are both dead, and nothing separates them.

### Carried, or copied?

`v3_distinct` keeps the implied digit, the digit the donor line writes, and the
clean answer mutually distinct, so the three are separable predictions.

| site | implied | raw | log-odds moved further toward implied than raw |
|:---|:---|:---|:---|
| depth 1 gap 0, ancestor | 88.9% | 9.4% | 63/117 |
| depth 1 gap 1, ancestor | 83.7% | 16.3% | 60/104 |
| depth 1 gap 2, ancestor | 80.9% | 19.1% | 51/94 |
| depth 2 gap 0, chain `n` | 100% | 0% | **141/144** |
| depth 3 gap 0, chain `n` | 100% | 0% | **144/144** |

Unpredicted, and worth saying so: the chain edit separates carrying from copying
far more sharply than the ancestor edit does — 141/144 and 144/144 against
roughly half for the depth-1 ancestor arms. The readout does not reproduce the
digit written at the patched position; it applies the target's own remaining step
to it. One step of arithmetic is being done on the patched value.

### What the arm verdicts do at this N, and why they are not the reading

The frozen quorum is `max(1, n - 1)` — "all but one". At n=5 that asks for 80%
of items; at n=48 it asks for 47/48, or 97.9%. The surface control's actual pass
rate is 85–100% in **every** arm here, so which side of the line an arm falls on
turns on one or two items:

| arm | best scoring layer | needed | verdict |
|:---|:---|---:|:---|
| `depth1_gap0_seed{0,1,2}` | 46/48, 45/48, 45/48 | 47 | invalid test ×3 |
| `depth1_gap1_seed{0,1,2}` | 47/48, 47/48, 48/48 | 47 | positive ×3 |
| `depth1_gap2_seed{0,1,2}` | 46/48, 48/48, 46/48 | 47 | invalid test, positive, invalid test |
| `depth2_gap0_seed{0,1,2}` | 47/48, 47/48, 47/48 | 47 | scientific negative ×3 |
| `depth3_gap0_seed{0,1,2}` | 48/48, 47/48, 48/48 | 47 | scientific negative ×3 |

So E3 returns a second finding it was not looking for: **the scoring policy does
not survive its own N.** A fixed-count quorum calibrated at five items becomes a
97.9% requirement at forty-eight, and the depth-1 positive control fails it while
its ancestor gap is 48/48 and its directional control 47/48.

The gate is **not** changed here. Rewriting a quorum after seeing which arms it
fails is the retroactive policy move this project refuses everywhere else, and it
would be made on evidence produced by the very run being scored. The rates above
are reported instead, and the per-layer counts are in each arm's `gates` block
for anyone who wants to re-derive them. Deciding the rule is a separate,
pre-registered piece of work.

Note also that the depth-1 arms are unscreened, unlike `../e2_stage_b/`, whose
24 items came out of a clean-forward-pass selection: 80–89% here against 24/24
there is the cost of dropping the screen, not a disagreement.

### What this settles, and what it does not

The depth-2 arm in `../e2_stage_b/` could only be scored an **invalid test** —
directional control failed and the surface edit was loud — and an invalid test is
not evidence about the model. The chain row is the positive control that arm
never had: patching a line in the *same trace*, against the *same* clean readout
and null spread, moves the answer onto the implied digit 144/144. The
intervention works there. The ancestor's silence is therefore about the model.

Still open: every site measured here is one step or more from the target through
a *written* value. Nothing in this directory shows what happens when the
intermediate is not written down — that is the `omit` contrast in
`../written_vs_omitted/`, at n=5, and the format search it points at.

## What no chain row is allowed to do

No gate reads them. The verdict function is frozen at `v2_gap_and_floor`, and
binding a verdict to a reading introduced in the same change would let the arm's
own result decide how the arm is scored. The chain rows are recorded, printed
beside the verdict by `dag_patching.print_chain_rows`, and inert — which is a
claim about the scorer, so `tests/test_dag_patching.py` tests it rather than
trusting it.
