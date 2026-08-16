# Paper Strategy (DAG) — Where a Patched Residual State Is Read in a Written Chain of Thought

*Durable working doc, opened 2026-08-15. Survives context compaction.*

> **Split out of `PAPER_STRATEGY.md`, 2026-08-15.** That file is now
> [`PAPER_STRATEGY_RMD.md`](PAPER_STRATEGY_RMD.md) and covers the
> selective-prediction / Mahalanobis thread only. **The two are separate papers.**
> They share no model (1.5B distill here, 7–8B there), no data (synthetic
> arithmetic DAG here, MATH-500 there), no metric, and no claim. Do not staple
> them; do not let a sentence from one qualify the other.

Sources of record: `EXPERIMENT_LOG.md` entries 2026-08-13 through 2026-08-16,
`results/dag_patching/` (eight archived pilot artifacts plus seven later
sub-packages), and `notebooks/15_dag_paper_story.ipynb`, which is the only
narrative write-up outside those two.

*Updated 2026-08-16 for E3 at its registered N and the chain-node arm
(`results/dag_patching/e3_ladder/`). That run changed §1, §2 (D2, and new D7–D9),
§3.1, §3.7, a new §3.10, §4, §5 and §6b. It did not change §6a: the workshop
paper was complete before it and is stronger after it.*

---

## 0. Setup, in one paragraph

`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (revision
`ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`, 28 layers) is prompted with a
synthetic arithmetic program whose dependency graph is known by construction:
every node is a single digit, every line states an operation and its result, and
the answer is a ten-way readout over digits. `dag_tasks` generates the items;
`dag_patching` runs residual-stream activation patching at layers 6, 13, 20 and
27 (27 produces no movement and is dropped). A **donor** trace differs from the
**clean** trace in one node's value; patching writes the donor's residual state
at the clean trace's own token positions.

Three digits are named for every patched row, and keeping them distinct is what
makes the experiment well-posed: the **clean/target** digit (the true answer),
the **implied** digit (the donor's value carried through the recipient's
remaining arithmetic — the chain is affine, so donor `v_j` through recipient `i`
implies `v_j + delta_i`), and the **raw** digit (the one literally written at the
patched position). `v3_distinct` is the generator that guarantees all three
differ; earlier families did not, which is a defect it was built to fix.

---

## 1. Current thesis (2026-08-16)

> **In a written chain of thought, a patched residual state changes the answer
> only at the node the answer reads directly. One step further up the chain the
> same intervention does nothing, and the reason is that the model is reading the
> written intermediate value rather than recomputing it — unwriting that value
> does not restore the patch, it destroys the model's ability to answer at all.**

Since 2026-08-16 the middle sentence has a sharper form, and it is a **step
cliff, not a decay**. Over 1,035 patch sites (E3, 15 cells of 48 items, three
seeds), a site **one** step from the target lands the implied digit 555/603
(92%), at every token distance from 11 to 60; a site **two or more** steps away
lands it **0/432**, at every distance. Nothing in between was observed. Two steps
and three steps are not merely both small — patched in the same item, they are
indistinguishable (0 discordant pairs of 144).

The pre-registered form of the first half, from `results/dag_patching/e2_stage_b/`:

> The one-versus-two-step contrast persists after matching on clean confidence
> and ancestor token distance.

**24 matched pairs, layer 13, implied digit uniquely on top 24/24 at depth 1
against 0/24 at depth 2.** Difference 1.00; one-sided exact paired test (sign
test = exact McNemar) p = 2⁻²⁴ = **5.96e-8**. Registered in `6f1e9a7` before the
selection rule was written, before the items existed, and before any of these
numbers did.

**Quote 5.96e-8 and nothing else.** The registered bootstrap interval
[1.000, 1.000] is degenerate — every replicate is 1 by construction — and is the
absence of a counterexample, not a bound. Fisher's exact (3.1e-14) is the wrong
model: it credits a matched-pair design with twice the independent observations
it has.

---

## 2. What is established (and the artifact behind each)

| # | Claim | Evidence | Strength |
|:--|:---|:---|:---|
| **D1** | The depth-1/depth-2 dissociation survives matching on clean confidence and ancestor token distance | `e2_stage_b`, 24 pairs, 24/24 vs 0/24, p = 5.96e-8 | **Pre-registered, confirmatory** |
| **D2** | The collapse is not a token-distance artifact | `e3_ladder`, 1,035 eligible sites: banded by token distance, every band holding both a one-step and a multi-step site splits on the step count and not the band — at 16–30 tokens, 97/113 (85.8%) for one step against **0/217** for two; at 31–45, 77/96 against 0/71 and 0/73 | **At registered N**: 48 items × 3 seeds × 5 arms |
| **D3** | The model *reads* the written intermediate rather than computing it latently | `written_vs_omitted`: omitting the chain value drops clean p(target) to 0.240 (depth 2) and 0.050 (depth 3) from 0.996/0.999 — the patch is not restored, the task is destroyed | Strong, with its own control |
| **D4** | D3's collapse is attributable to the missing path value alone | `--omit decoy` arms are indistinguishable from written arms: 5/5 clean at p(target) 0.997/0.999, nothing moved | Clean separation of the two changes |
| **D5** | The depth-2 patch is not inert — it reaches the read position and does nothing useful there | median TV at L13: depth-1 ancestor 0.9868, depth-2 ancestor 0.0877, nulls ~0.004; at depth 2 p(target) 0.913→0.804 and remaining-digit mass 0.078→0.162 while p(implied) stays 0.001 | Descriptive, matched batch |
| **D6** | The intervention is quiet where it should be | 0/144 null rows flipped at either depth; 0/192 control rows moved at layer 13 | Registered validity gate |
| **D7** | The two sites dissociate **inside one item**, holding the clean readout, the token count, the null spread and the surface control fixed | `e3_ladder` chain arm: at depth 2, ancestor (2 steps) vs the written intermediate (1 step) is 144 chain-only / 0 ancestor-only, sign test p = 9.0e-44; identically at depth 3 | **Within-item, at N**; the design D1 could not run |
| **D8** | The depth-2 silence is about the model, not the intervention | The chain edit in the *same trace*, at 11 tokens, moves the answer onto the implied digit 144/144 while the ancestor at 23–36 tokens moves it 0/144 | The in-arm positive control `e2_stage_b/depth2` never had |
| **D9** | At the site where it works, the readout **carries** the donor value rather than copying it | Chain edit: log-odds moves further toward implied than raw in 141/144 (depth 2) and 144/144 (depth 3); implied uniquely on top 100%, raw 0% | Strong — **and specific to this site**, see §3.1 |

**D3+D4 are the most under-appreciated pair in the thread.** They convert the
depth result from "patches decay with distance" — which is a boring claim and
also not what was measured — into a statement about *what the written trace is
for*: at depth ≥ 2 the answer is determined by the written intermediate token,
the patch does not touch that token, and nothing downstream reads the state the
patch does write.

---

## 3. What is NOT established (the honest limits, in rank order of danger)

*1–9 are in rank order. 10 and 11 were added on 2026-08-16 and are appended in
discovery order rather than inserted, so that the §3.n references already written
elsewhere keep resolving; by danger, **10 belongs around third** — it is the one
that will silently produce a wrong sentence in a draft.*

1. **The mechanism is shown to be computation at one site, and is still a coin
   flip at the site the paper is built on.** The contrast that separates "the
   recipient transformed the donor value" from "the readout copied the written
   digit" is the log-odds margin `delta_toward − delta_toward_raw`. E3 measured it
   at N on both sites, and they came apart:

   | edit | margin favours implied | n |
   |:---|:---|---:|
   | ancestor, depth 1 gap 0 / 1 / 2 | 53.8% / 57.7% / 54.3% | 117 / 104 / 94 |
   | chain (the written intermediate) | **97.9% / 100%** | 144 / 144 |

   So the coin flip is real, it survives a twentyfold increase in N, and it is
   **a property of the ancestor edit specifically** — not of the readout, and not
   of the method. Where the patch lands on the line the answer reads directly,
   the readout demonstrably applies the target's own remaining step to the donor
   value (D9). Where it lands one line above, whatever gets through does not
   separate carrying from copying at all.

   This matters for how the paper is framed: it is the *ancestor* edit — the one
   D1 and the whole depth contrast are built on — whose mechanism remains
   ambiguous. Do not let D9 launder that. *(The earlier readings stand as
   recorded: `v2_paired` 20 items gives 10/8, the cross-item donor leans to
   copying at 6/13, and a post-hoc recount on the 24 matched E2 pairs gives 19/5
   with raw never uniquely on top, 0/24. The E2 recount is post-hoc on a
   registered run and its batch differs from `v2_paired` in three ways at once, so
   it attributes nothing; it is still not written into an artifact, see §7. E3's
   numbers are pre-N and unscreened, which is the cleanest of the four and the one
   to quote.)*
2. **The cross-item donor control failed its specificity leg on every seed**
   (1/5, 2/5, 2/5, 2/5 against a 4/5 quorum), while its direction leg passed 5/5
   everywhere. A foreign item's state perturbs the readout about as much as the
   native edit (median TV 0.971 vs 0.984). The control built to close selectivity
   reopened it.
3. **"Depth" is a bundle, not a variable.** It adds an operation, a written
   intermediate result, a new variable binding and a changed local context
   simultaneously. `dag_tasks` says so in its own module docstring and has since
   before the run. Matching two observed covariates removes two rival
   explanations; it does not unbundle the rest. Never write "the depth result is
   about graph depth" — that sentence was retracted on 2026-08-15.
4. **Layer 13 is inherited**, from the `v3_distinct` discovery table, not
   re-searched. E2 is confirmatory for the *contrast*; the depth-1 rate is not a
   fresh test of anything.
5. **n = 1 checkpoint, n = 1 task family, one operation set, one notation.** A
   model that never learned to carry an unstated intermediate is not the same
   claim as a model that cannot.
6. **Four registered row kinds out of five.** Cross-item was unreachable in the
   matched design (a cross-item batch is *selected* for mutual donatability, so
   it is a different batch from the one stage A matched). Recorded in
   `unreachable_row_kinds` in both arms and labelled a **protocol deviation**.
   This label stays regardless of what is run next.
7. **Two verdict systems disagree on depth 2 — now with the tie broken.** E2's
   registered null-flip gate calls it valid-and-negative; `dag_patching`'s arm
   scorer calls it an *invalid test* (`directional_control_failed`,
   `surface_above_null`) because its gates are relative and there is no movement
   to be relative to. Both labels are in the artifacts and both stay. What has
   changed is that an invalid test is no longer the *only* thing on offer: an
   invalid test is not evidence about the model, and D8 supplies the in-arm
   positive control that arm never had — patching a line in the same trace, at
   the same clean readout and null spread, moves the answer 144/144. The
   intervention demonstrably works there, so the ancestor's silence is
   attributable. **Do not retro-score the E2 arm on this.** It is a separate run;
   what E3 licenses is the claim about the model, not a relabelling of an
   archived verdict.
8. **High-confidence regime only.** Matched window p(target) 0.696–0.990, median
   0.914 — the upper half of the range the original depth-1 result came from.
9. **float32 is a model run, not a readout.** E2 ran every matmul in float32; the
   eight archived arms are bfloat16. E2 is internally valid (both depths at one
   precision) but is **not** a same-precision replication, and `dag_pooling.pool`
   refuses to merge across the difference.
10. **The arm scorer's quorum does not survive its own N, so E3's arm verdicts
    are not quotable.** `_quorum(n) = max(1, n − 1)` was calibrated at n = 5,
    where "all but one" asks for 80%. At n = 48 the same rule asks for 97.9%. The
    surface control's actual pass rate is 85–100% in every E3 arm, so which side
    of the line an arm falls on turns on one or two items: `depth1_gap0` scores
    *invalid test* on all three seeds at 46/48, 45/48, 45/48 — while its ancestor
    gap is 48/48 and its directional control 47/48. **The gate was deliberately
    not changed**, because rewriting a quorum after seeing which arms it fails is
    a retroactive policy move made on evidence produced by the run being scored.
    Consequence for the paper: quote E3's *rates* (D2, D7, D9, all of which are
    computed over items and not over arms) and never its verdict labels. Fixing
    the rule is its own pre-registered piece of work, §7.
11. **E3's depth-1 arms are unscreened, and read lower than E2's.** 80–89% here
    against 24/24 in `e2_stage_b`, whose items came through a clean-forward-pass
    selection. That is the cost of dropping the screen, not a disagreement
    between the runs, and it is the honest number for an unselected item.

---

## 4. The methodological contribution (do not undersell this)

The thread produced six transferable results about how to run and score
activation-patching experiments. For an interpretability audience these may be
worth more than D1.

- **A directional gate cannot separate propagation from copying.** "The answer
  moved toward the value the donor implies" is satisfied by a readout that simply
  copied the digit the donor wrote, whenever those two digits differ — and they
  do differ, by construction, at depth 1. The fix is one extra recorded quantity
  (`delta_toward_raw`) at zero extra forward passes. Every patching paper that
  reports a directional statistic is exposed to this.
- **Relative gates are blind at both ends.** Every gate was a ratio or a
  one-sided comparison against a null, so none noticed when both sides were
  ≈ 0 (fixed by an absolute floor, which flipped four verdicts to *scientific
  negative*) — and none noticed the mirror case where *everything* moves. In the
  omission arms the nulls flip 23/40 and 33/40 and a comment-tag rewrite flips
  the answer 4/5, so a collapsed model clears every relative gate and scores
  `positive`. `control_specificity` is the diagnostic that catches it, and it is
  reported and never binding, on purpose.
- **A three-valued verdict space is necessary.** *invalid test* / *positive* /
  *scientific negative*. A patch that was directional, quiet, selective and
  simply did not change the answer is a finding; collapsing it into "failed" and
  a broken intervention into the same bucket destroys information.
- **Measurement and scoring must be separable.** Rows are the measurement; gates
  are a policy over them. `--rescore` and `--reanalyse` re-derive every verdict
  with no GPU and no rewrite of an archived file, which is what made three rounds
  of external review answerable at all.
- **A fixed-count quorum is a moving significance level.** "All but one" is a
  natural-sounding validity rule and it is 80% at n = 5 and 97.9% at n = 48, so
  scaling a pilot to its registered N silently tightens every gate that uses one
  — and tightens it hardest on the *positive control*, which is the arm most
  likely to sit just under a very high bar. E3 hit this exactly (§3.10). Anyone
  freezing a gate at pilot N is exposed; the rule has to be written as a rate, or
  as a test, before the N changes. Note that this was found the expensive way,
  by running the registered N and watching a working arm score *invalid test*.
- **The within-item design is available more often than it is used.** A depth
  ladder compares conditions across items and inherits every between-item
  difference; patching two sites in the *same* trace holds the clean readout, the
  token count, the null spread and the surface control fixed by construction, and
  turns a descriptive gap into a sign test on discordant pairs (D7, p = 9.0e-44
  on 144 pairs). The extra cost was one additional patched row per item.

Supporting the whole thread: the eight pilot artifacts are committed
byte-for-byte outside DVC, content-addressed, with v0 schema fields *derived*
(regenerated and checked against the archived measurements) rather than
backfilled, and *inferred* fields kept separately and flagged.

---

## 5. Where the novelty is, and where the fight is

**The increment is not "activation patching works."** It is:

1. **A ground-truth dependency graph.** Most patching work on reasoning has no
   independent statement of which computation should influence which; here the
   graph is generated, so ancestor/non-ancestor is not a judgement call and the
   counterfactual answer is known in closed form.
2. **The written-vs-latent question, answered by an omission control** (D3+D4).
   This is the piece that speaks directly to the faithfulness-of-CoT literature,
   which mostly perturbs the *text*; the omission arm perturbs the text while
   holding token count, position and notation fixed, and pairs it with a decoy
   arm that separates "the notation is unreadable" from "the value is missing."
3. **The propagate-vs-copy contrast as a general critique** (§4, bullet 1).
4. **A within-item dissociation between two patch sites in one trace** (D7). The
   depth ladder can only compare an item at depth 1 against a *different* item at
   depth 2. The chain arm patches the ancestor and the written intermediate in the
   same trace, against the same clean readout and the same null spread, which
   converts the central claim from a between-condition gap into a paired test —
   and produces the cell that had never been measured anywhere: a two-step site
   *nearer* the read position than the one-step sites that work, and dead
   (0/144 at 23 tokens, against 85.8% for one step at 16–30).

**Literature grounding is the biggest un-run task on this paper.** No
DAG-specific literature pass has been done — `RELATED_WORK.md` covers the RMD
thread. Before drafting, ground against at minimum: causal tracing / ROME-style
patching methodology, path patching and causal scrubbing (which is the closest
prior art for "does the model's computation respect the stated graph"), CoT
faithfulness work (perturbation-based and filler-token), and any synthetic-DAG
or algorithmic-task interpretability benchmark. **Assume something close exists
and find it before writing**, not after review.

Expected fight: a reviewer who reads "1.5B distill, synthetic arithmetic,
digit readout" and asks whether any of it transfers. §6 is the answer.

---

## 6. Two venues, two scopes

### 6a. Workshop paper — submittable now

Everything in §2 exists on disk, is committed, and has survived three rounds of
external review. What is missing is **the write-up and the literature pass**, not
evidence.

- Title direction: *"One step, and no further: where a patched residual state is
  read in a written chain of thought."*
- Spine: D1 (pre-registered, matched, p = 5.96e-8) → D7 (the same dissociation
  *within* one item, p = 9.0e-44, which is also D8, the positive control that
  makes the depth-2 silence attributable) → D2 (a step cliff, not distance and
  not decay) → D3+D4 (the model reads, it does not compute) → §3 stated as
  limits, with §3.1 and §3.2 given real space rather than a footnote.
- E3 changed the spine's shape, not its length: D1 stays the headline because it
  is the pre-registered one, and D7 goes immediately after it because it answers
  the first question a reader has about D1 — whether the depth-2 null is the
  model or the intervention.
- Quote rates, not arm verdicts, from `e3_ladder` (§3.10). The prose must not
  say "positive in three of five arms."
- The negative and the methodology are **part of the contribution**, not a
  limitations section. A workshop that takes "here is a control that failed and
  here is what it cost us to notice" is the right room for this.
- Cost: the draft, plus the literature pass. No GPU.

### 6b. Main conference — and why E4 is now answered rather than pending

This is a genuine main-conference candidate, and it is the stronger of the two
threads in this repo, because the effect is large, pre-registered, and cleanly
controlled. What it is missing is not rigor but **estimand and scope**: it is a
contrast on one checkpoint and one format — better measured than it was on
2026-08-15, but not broader.

> **E4 — the node-by-node influence matrix against the transitive reduction — is
> dead on this format and gated on any other.** An earlier draft ranked it first;
> that was wrong. The 2026-08-15 revision gated it on two predictions, and E3 has
> now **measured both**, which upgrades the argument from a forecast to a result:
>
> 1. **The matrix's answer is no longer a prediction — it has been observed.**
>    The gating argument was that every intermediate is written (`dag_tasks`
>    states k−1 results at depth k), so by D1 and D3 the influence of any
>    non-adjacent node must be zero *because the written token determines the
>    answer and the patch does not touch it*. E3 measured exactly that quantity
>    across eight sites at step counts 1, 2 and 3: the three sites more than one
>    step out are **0/432**, spanning 23 to 48 tokens from the read position — a
>    range that overlaps the one-step sites, which land 92%. There is no decay
>    structure to find between two steps and three either (D7's third row: 0
>    discordant pairs of 144). The influence matrix on this format is therefore
>    the adjacency matrix of the last step, and running it would spend roughly
>    thirty times the forward passes to re-measure an off-diagonal that has been
>    observed to be empty at every step count the format admits. **Not gated —
>    answered, and negative.** It belongs in the paper as a measured result, not
>    in a future-work list.
> 2. **A nonzero cell is still not interpretable, and now we know at which site.**
>    §3.1 measured propagate-vs-copy at N and it came apart by site: 97.9–100%
>    for the chain edit, but **53.8–57.7% for the ancestor edit**, which is the
>    edit every cell of the matrix would be built from. The coin flip did not go
>    away with n = 20 → n ≈ 100; it localised. So this reason has not weakened —
>    it has hardened, and it now applies precisely to E4's unit of measurement.

**What survives of E4.** Only the version run on a format that clears G1, where
the intermediate is *unwritten* and a non-adjacent cell could be nonzero for an
interesting reason. That is a different experiment from the one in the original
plan, and it inherits none of this format's results. Everything below is about
reaching that format.

**The gate, and why its two halves are one experiment.** A format in which the
intermediate is *unwritten* is itself the intervention that does not copy a
written value: with no digit at the patched position to copy, the propagate/copy
confound largely dissolves at depth ≥ 2. So "clean-valid multi-step format" and
"selective propagation" are not two runs in sequence — they are one screen with
two acceptance criteria.

- **G1 — clean validity.** A (model, format) pair where at least one intermediate
  on the dependency path is unwritten *and* the model still solves the clean task
  at high p(target). The naive version is already known to fail: D3's
  comment-padded omission drops clean p(target) to 0.240 at depth 2 and 0.050 at
  depth 3. This is the hard part and it may not be reachable on the current
  checkpoint at all — the log states the reason plainly, that this distill was
  never trained on traces with unstated steps, and "never learned to" is not the
  same claim as "cannot".
- **G2 — selective propagation.** On a format that clears G1: the patch moves the
  answer toward the implied value, and does so for ancestors and not for
  non-ancestors, with the raw-digit reading no longer available as an explanation.
  This is the claim §3.1 currently cannot support. E3 sharpened what G2 has to
  show: the chain arm demonstrates that this model *can* apply one remaining
  arithmetic step to a patched value and does so essentially always (D9), so G2 is
  not asking whether the capability exists. It is asking whether the capability is
  reachable across a step the model would otherwise satisfy by reading a token —
  which is a narrower and more answerable question than it was on 2026-08-15.

**Only if G1 and G2 both clear does E4 become worth running** — and at that point
it is genuinely the experiment that converts a two-point contrast into a claim
about structure, which is why it stays in the plan rather than being dropped.

**What this does to the ordering.** E5 moves up and changes role. It was a
breadth follow-on; it is now the most likely *route to G1*, because if no
available checkpoint holds a multi-step format then G1 fails on the model rather
than on the format, and that is worth knowing before any format search. Tokenizer
alignment across the triple is already verified and committed
(`tokenizer_alignment.json`), which was E5's hard precondition.

So:

0. **Fix the quorum rule, before any further registered run.** No GPU, and it
   now blocks the others: every run below is at E3's N or larger, where the
   frozen `max(1, n − 1)` is a 97.9% validity bar that a working positive control
   fails (§3.10). Decide it as a rate or a test, pre-register the decision, and
   apply it going forward only — E3's own arms keep the labels they were scored
   with, and its rates are unaffected either way.
1. **E5 as a G1 screen** — Base / Instruct / Distill, plus scale, asking one
   question: does any of them solve a task with an unstated intermediate? Cheap,
   and it is the fork the rest of the plan hangs on.
2. **Format search for G1**, informed by 1. Second operation/format family folds
   in here rather than being a separate item, since one notation and one
   operation set also confounds "depth" with a rendering.
3. **G2 on whatever clears G1.** Registered in advance, with a caliper on the
   matching rule and the screening cap fixed beforehand — the standing rule from
   the 2026-08-15 corrections. Include a chain-edit arm: it is one extra row per
   item and it supplies the in-arm positive control that made E3's negative
   attributable (D8). On a G1 format there is no written intermediate to patch,
   so the chain site has to be redefined for that design — do that at
   registration time, not after.
4. **E4 on a G1 format, only then** — and not the E4 in the original plan, which
   §6b now reports as measured and negative rather than pending.

GPU cost is genuinely small: a current arm is ~225 forward passes of 127-token
sequences on a 1.5B model. Item-family design, not compute, is the constraint —
that has been true at every step of this thread and should be planned for.

**Timeline honesty.** This is a longer path than "run E4 next", and G1 is a real
risk of failing outright. That does not touch §6a: the workshop paper is complete
on the evidence that exists, and none of it depends on this gate clearing.

---

## 7. Standing rules carried forward

- Do not backfill or rewrite the archived JSON files. Anything recovered after
  the fact is stored beside them.
- The verdict space stays three-valued.
- Do not add an epsilon for the 0.0153-vs-0.0152 surface case. It stays a
  failure; the 4/5 rule already handles it.
- No DVC for this thread — no remote on this host, and `.dvc/cache` would be the
  only copy.
- `notebooks/15_dag_paper_story.ipynb` is generated by a script and executed by a
  `jupyter_client` runner; never hand-edit the `.ipynb`.
- **A reading introduced by a run does not gate that run.** No gate reads a chain
  row; `dag_patching.print_chain_rows` reports them beside the verdict and
  `tests/test_dag_patching.py` asserts they move nothing. Binding a verdict to a
  quantity added in the same change lets an arm's own result decide how the arm
  is scored. Same reason the quorum was left alone in §3.10.
- **Analysis that produces a table in a paper lives in `dag/`, not in a
  scratchpad.** `dag_e3_ladder.py` re-derives every E3 number from the arm files
  with no GPU and no model, and imports its layer and primary outcome from
  `dag_stage_b` rather than restating them, so the numbers stay comparable
  instead of merely adjacent.
- **Open write-up debt**, in the order it should be cleared:
  1. The post-hoc propagate/copy recount on the 24 matched E2 pairs (§3.1) is
     computed but not yet in `ANALYSIS.json`, the E2 README, or
     `EXPERIMENT_LOG.md`. It needs a `--reanalyse` pass following the
     `exact_paired` precedent — added-after-the-run, exploratory, reported and
     never gated. E3 raised the value of doing this: §3.1 is now a table with
     four entries in it and the E2 recount is one of them.
  2. `notebooks/15_dag_paper_story.ipynb` predates E3 and does not contain D7,
     D8, D9 or the step-cliff framing. Its generator script must be recovered or
     rewritten first — the generators live outside the repo, which is the same
     fragility `dag_e3_ladder.py` was extracted to fix.
  3. `notebooks/14_rmd_paper_story.ipynb` still cites `PAPER_STRATEGY.md` §3/§7c
     in a markdown cell; the file split on 2026-08-15 and the reference is stale
     (the section numbers still resolve, the filename does not).
  4. The DAG literature pass, §5. Still the biggest un-run task on this paper.
- `EXPERIMENT_LOG.md` correction #4 currently ends "Restoring it is the next
  small run." That is no longer the plan: the cross-item arm exists and has run
  (four seeds, 2026-08-14); what E2 lacks is only the *matched-pairs* version,
  and the protocol-deviation label stands either way.
