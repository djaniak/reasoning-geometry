# Paper Strategy (DAG): Where a Patched Residual State Is Read in a Written Chain of Thought

*Durable working document, opened 2026-08-15. Survives context compaction.*

> **Split out of `PAPER_STRATEGY.md`, 2026-08-15.** That file is now
> [`PAPER_STRATEGY_RMD.md`](PAPER_STRATEGY_RMD.md) and covers the
> selective-prediction / Mahalanobis thread only. **The two are separate papers.**
> They share no model (1.5B distill here, 7–8B there), no data (synthetic
> arithmetic DAG here, MATH-500 there), no metric, and no claim. They should not be
> combined, and a sentence from one should not be used to qualify the other.

Sources of record: `EXPERIMENT_LOG.md` entries 2026-08-13 through 2026-08-16,
`results/dag_patching/` (eight archived pilot artifacts plus seven later
sub-packages), `notebooks/16_dag_workshop_story.ipynb` (short paper
storyboard), `notebooks/15_dag_paper_story.ipynb` (full experiment ledger), and
the literature and claim audit in
[`docs/research/2026-08-16-dag-literature-and-claim-boundary.md`](docs/research/2026-08-16-dag-literature-and-claim-boundary.md).

*Updated 2026-08-16 for E3 at its registered N and the chain-node arm
(`results/dag_patching/e3_ladder/`). That run changed §1, §2 (D2, and new D7–D9),
§3.1, §3.7, a new §3.10, §4, §5 and §6b. It did not change §6a: the workshop
paper was already complete before that run, and it is stronger after it.*

*Corrected 2026-08-16 against the literature pass and paired gap analysis. The
strategy now separates the semantic cliff from distributional decay, adds
§3.12, and narrows the E4 claim in §6b.*

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
remaining arithmetic; the chain is affine, so donor `v_j` through recipient `i`
implies `v_j + delta_i`), and the **raw** digit (the one literally written at the
patched position). `v3_distinct` is the generator that guarantees all three
differ. Earlier families did not guarantee this, which is the defect it was built
to correct.

---

## 1. Current thesis (2026-08-16)

> **In a pretrained reasoning-tuned 1.5B model, a native-position residual-state
> transplant usually controls the recipient's next arithmetic update, but exact
> semantic control is absent when two or more written operations remain. The
> readout still changes distributionally at those multi-step sites. Same-trace
> and token-distance controls locate an immediate-read boundary in the semantic
> outcome, while the clean/implied/raw assay separates recipient-side
> transformation, literal donor copying and preservation of the clean
> computation.**

This thesis is scoped to one checkpoint and one fully written synthetic format.
The omission pilot supports reliance on the written intermediate but does not
identify an overwrite mechanism. The literature-qualified claim boundary is
recorded in the research note linked above.

Since 2026-08-16 the result has a sharper form: a **semantic cliff with
distributional decay**. Across 1,035 eligible site observations, clustered
within items, arms and seeds, a site **one** step from the target lands the
implied digit 555/603 (92%), at token distances from 11 to 60. A site **two or
more** steps away lands it **0/432**. Probability and total-variation effects
remain at the multi-step sites, so the patch cannot be said to have no
influence. On the registered semantic outcome, the two-step and three-step sites
are indistinguishable within the same items (0 discordant pairs of 144).

The pre-registered form of the first half, from `results/dag_patching/e2_stage_b/`:

> The one-versus-two-step contrast persists after matching on clean confidence
> and ancestor token distance.

**24 matched pairs, layer 13, implied digit uniquely on top 24/24 at depth 1
against 0/24 at depth 2.** Difference 1.00; one-sided exact paired test (sign
test = exact McNemar) p = 2⁻²⁴ = **5.96e-8**. Registered in `6f1e9a7` before the
selection rule was written, before the items existed, and before any of these
numbers did.

**Quote 5.96e-8 and nothing else.** The registered bootstrap interval
[1.000, 1.000] is degenerate, because every replicate is 1 by construction, and it
records the absence of a counterexample rather than a bound. Fisher's exact (3.1e-14) is the wrong
model: it credits a matched-pair design with twice the independent observations
it has.

---

## 2. What is established (and the artifact behind each)

| # | Claim | Evidence | Strength |
|:--|:---|:---|:---|
| **D1** | The depth-1/depth-2 dissociation survives matching on clean confidence and ancestor token distance | `e2_stage_b`, 24 pairs, 24/24 vs 0/24, p = 5.96e-8 | **Pre-registered, confirmatory** |
| **D2** | Token distance alone does not explain the semantic cliff | `e3_ladder`, 1,035 eligible site observations, clustered within items, arms and seeds: at 16–30 tokens, implied is top at 97/113 one-step sites and **0/217** two-step sites; at 31–45, 77/96 against 0/71 and 0/73 | Registered-N descriptive rates; interpret with matched D1 |
| **D3** | The written intermediate is behaviorally required in this format | `written_vs_omitted`: omitting the chain value drops clean p(target) to 0.240 (depth 2) and 0.050 (depth 3) from 0.996/0.999; the patch is not restored | Supporting pilot, n = 5 |
| **D4** | The comment-padded omission notation remains legible when applied off the dependency path | `--omit decoy` arms are indistinguishable from written arms: 5/5 clean at p(target) 0.997/0.999, nothing moved | Rules out a notation-wide failure, not every path-specific alternative |
| **D5** | Distributional influence remains after exact semantic control disappears | median TV at L13: depth-1 ancestor 0.9868, depth-2 ancestor 0.0877, nulls ~0.004; at depth 2 p(target) 0.913→0.804 and remaining-digit mass 0.078→0.162 while p(implied) stays 0.001 | Descriptive, matched batch |
| **D6** | The intervention is quiet where it should be | 0/144 null rows flipped at either depth; 0/192 control rows moved at layer 13 | Registered validity gate |
| **D7** | The two sites dissociate semantically **inside one item**, holding the clean readout, token count, null spread and surface control fixed | `e3_ladder` chain arm: at depth 2, ancestor (2 steps) vs written intermediate (1 step) is 144 chain-only / 0 ancestor-only, sign test p = 9.0e-44; identically at depth 3 | **Within-item, at N**; the design D1 could not run |
| **D8** | The multi-step semantic null is specific to the patch site, not a generic intervention failure | The chain edit in the *same trace*, at 11 tokens, puts the implied digit on top 144/144 times while the ancestor at 23–36 tokens does so 0/144 times | The in-arm positive control `e2_stage_b/depth2` never had |
| **D9** | Successful one-step edits usually transform the donor value; copying is site-dependent | Chain: implied uniquely top 288/288, raw 0/288. Depth-1 ancestor: implied 267/315, raw 46/315, clean 0/315 | Strong semantic outcome; the change-score margin is diagnostic only |

**D3+D4 argue against a simple latent-recovery story.** Removing the written
path value does not restore the ancestor patch; it makes the clean task fail.
The decoy arm shows that the omission notation remains legible when used off the
dependency path. Together they support reliance on the written path value. They
do not isolate the missing value as the only cause, prove that text overwrites
latent state, or show that no downstream component reads the patched state.

---

## 3. What is NOT established (the honest limits, in rank order of danger)

*1–9 are in rank order. 10–12 were added on 2026-08-16 and are appended in
discovery order rather than inserted, so that the §3.n references already written
elsewhere keep resolving. Ordered by danger, **10 belongs around third**, because it
is the one most likely to produce an incorrect sentence in a draft without being
noticed.*

1. **The ancestor edit is a mixture, not pure symbolic propagation.** The
   registered semantic outcome is clear: across the depth-1 ancestor arms, the
   implied digit is top in 267/315 cases, the raw digit in 46/315, and the clean
   digit in 0/315. Recipient-side transformation dominates, but literal copying
   remains a real minority outcome.

   The change-score margin `delta_toward − delta_toward_raw` is less decisive:

   | edit | margin favours implied | n |
   |:---|:---|---:|
   | ancestor, depth 1 gap 0 / 1 / 2 | 53.8% / 57.7% / 54.3% | 117 / 104 / 94 |
   | chain (the written intermediate) | **97.9% / 100%** | 144 / 144 |

   Raw log-odds often rise almost as far as implied log-odds because raw starts
   from a lower base. The margin therefore does not estimate the semantic
   mixture well. Use the registered argmax outcome for the main claim and the
   margin as a diagnostic. The chain site remains the cleaner transformation
   control (D9). Do not claim that the ancestor edit implements pure symbolic
   propagation. *(Earlier smaller runs remain exploratory: `v2_paired` gives
   10/8, the cross-item donor 6/13, and a post-hoc E2 recount 19/5. E3 is the
   unscreened at-N result to quote.)*
2. **The cross-item donor control failed its specificity leg on every seed**
   (1/5, 2/5, 2/5, 2/5 against a 4/5 quorum), while its direction leg passed 5/5
   everywhere. A foreign item's state perturbs the readout about as much as the
   native edit (median TV 0.971 vs 0.984). The control built to close selectivity
   reopened it.
3. **"Depth" is a bundle of several factors rather than a single variable.** It adds an operation, a written
   intermediate result, a new variable binding and a changed local context
   simultaneously. `dag_tasks` says so in its own module docstring and has since
   before the run. Matching two observed covariates removes two rival
   explanations; it does not separate the rest. The statement that the depth result
   is about graph depth should never be written; it was retracted on 2026-08-15.
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
7. **Two verdict systems disagree on depth 2, and the tie is now broken.** E2's
   registered null-flip gate calls it valid-and-negative; `dag_patching`'s arm
   scorer calls it an *invalid test* (`directional_control_failed`,
   `surface_above_null`) because its gates are relative and there is no movement
   to be relative to. Both labels are in the artifacts and both remain. What has
   changed is that an invalid test is no longer the only available reading. An
   invalid test is not evidence about the model, and D8 supplies the in-arm
   positive control that the arm previously lacked: patching a line in the same
   trace, at the same clean readout and null spread, moves the answer 144/144. The
   intervention demonstrably works there, so the ancestor's silence can be
   attributed. **The E2 arm should not be retro-scored on this basis.** It is a
   separate run, and what E3 licenses is a claim about the model rather than a
   relabelling of an archived verdict.
8. **High-confidence regime only.** Matched window p(target) 0.696–0.990, median
   0.914, which is the upper half of the range the original depth-1 result came from.
9. **float32 is a property of the model run rather than of the readout.** E2 ran every matmul in float32; the
   eight archived arms are bfloat16. E2 is internally valid (both depths at one
   precision) but is **not** a same-precision replication, and `dag_pooling.pool`
   refuses to merge across the difference.
10. **The arm scorer's quorum does not survive its own N, so E3's arm verdicts
    are not quotable.** `_quorum(n) = max(1, n − 1)` was calibrated at n = 5,
    where "all but one" asks for 80%. At n = 48 the same rule asks for 97.9%. The
    surface control's actual pass rate is 85–100% in every E3 arm, so which side
    of the line an arm falls on turns on one or two items: `depth1_gap0` scores
    *invalid test* on all three seeds at 46/48, 45/48, 45/48, while its ancestor
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
12. **Gap placement bundles token distance with clean difficulty.** This is an
    exploratory, complete-case analysis of 81 items eligible at all three
    placements. From gap 0 to gap 2, implied outcomes fall from 76/81 to 67/81
    and raw outcomes rise from 3/81 to 14/81. The 11 items that switch from
    implied to raw lose a median 0.129 in clean p(target), against 0.045 for the
    65 that remain implied (Mann–Whitney p = 0.0022); the groups do not clearly
    differ at gap 0 (p = 0.165). Longer placement, lower clean confidence and
    more copying move together within items. This design does not identify which
    drives which. D1 is unaffected because it matched on both clean confidence
    and ancestor distance.

---

## 4. The methodological contribution (do not undersell this)

The thread produced seven transferable results about how to run and score
activation-patching experiments. For an interpretability audience these may be
worth more than D1.

- **A directional gate cannot separate propagation from copying.** "The answer
  moved toward the value the donor implies" is satisfied by a readout that simply
  copied the digit the donor wrote, whenever those two digits differ, and they
  do differ by construction at depth 1. The fix is one extra recorded quantity
  (`delta_toward_raw`) at zero extra forward passes. Every patching paper that
  reports a directional statistic is exposed to this.
- **Relative gates are blind at both ends.** Every gate was a ratio or a
  one-sided comparison against a null, so none noticed when both sides were
  ≈ 0 (fixed by an absolute floor, which flipped four verdicts to *scientific
  negative*), and none noticed the opposite case in which *everything* moves. In the
  omission arms the nulls flip 23/40 and 33/40 and a comment-tag rewrite flips
  the answer 4/5, so a collapsed model clears every relative gate and scores
  `positive`. `control_specificity` is the diagnostic that catches it, and it is
  reported and never binding, on purpose.
- **A three-valued verdict space is necessary.** *invalid test* / *positive* /
  *scientific negative*. A patch that was directional, quiet, selective and
  simply did not change the answer is a finding. Placing it and a broken
  intervention in a single "failed" category destroys information.
- **Measurement and scoring must be separable.** Rows are the measurement; gates
  are a policy over them. `--rescore` and `--reanalyse` re-derive every verdict
  with no GPU and no rewrite of an archived file, which is what made three rounds
  of external review answerable at all.
- **A fixed-count quorum is a moving significance level.** "All but one" is a
  natural-sounding validity rule and it is 80% at n = 5 and 97.9% at n = 48, so
  scaling a pilot to its registered N tightens every gate that uses one without
  this being visible, and it tightens the *positive control* most, since that is the
  arm most likely to sit just below a very high bar. E3 encountered this exactly
  (§3.10). Any gate frozen at pilot N is exposed in the same way, so the rule has to
  be written as a rate, or as a test, before the N changes. Note that this was found the expensive way,
  by running the registered N and watching a working arm score *invalid test*.
- **The within-item design is available more often than it is used.** A depth
  ladder compares conditions across items and inherits every between-item
  difference; patching two sites in the *same* trace holds the clean readout, the
  token count, the null spread and the surface control fixed by construction, and
  turns a descriptive gap into a sign test on discordant pairs (D7, p = 9.0e-44
  on 144 pairs). The extra cost was one additional patched row per item.
- **A placement manipulation can also move clean difficulty.** In the paired gap
  arms, the items that switch from transformation to copying lose about three
  times as much clean p(target) as the items that keep transforming (§3.12).
  Written-trace patching should measure clean confidence for each placement and
  either match on it or report it beside the patch outcome. Token distance and
  clean difficulty cannot be separated in this gap sweep.

Supporting the whole thread: the eight pilot artifacts are committed
byte-for-byte outside DVC, content-addressed, with v0 schema fields *derived*
(regenerated and checked against the archived measurements) rather than
backfilled, and *inferred* fields kept separately and flagged.

---

## 5. Where the novelty is, and where the fight is

The literature pass is complete. The closest prior is Shih, Winnicki and Darve
(2026), which already edits internal written state, predicts an exact downstream
counterfactual and controls for copying. Geiger formalizes the interchange test;
Tan and Kudo cover causal arithmetic chain-of-thought; Brinkmann covers known
task structure plus mechanistic intervention; Patchscopes and Mehrafarin cover
hidden-state transplantation. Do not claim novelty for any one of those
ingredients.

The defensible increment is:

1. **A clean/implied/raw semantic assay** that separates recipient-side
   transformation, literal donor copying and clean preservation on every item.
2. **A same-trace immediate-read boundary**: the last intermediate controls the
   answer 144/144 times while its ancestor controls it 0/144 times.
3. **A larger one-step versus multi-step boundary** (555/603 against 0/432) with
   overlapping token-distance controls.
4. **A measurement result:** probability and total-variation effects can remain
   after exact semantic counterfactual control disappears.

The omission arm is supporting evidence, not proof that written text overwrites
latent state. The ancestor edit is a measured mixture (implied 267/315, raw
46/315, clean 0/315). It is therefore neither a pure propagation mechanism nor an
unresolved semantic outcome. See the linked research note for sources and wording
guardrails.

Expected objection: a reviewer who reads "1.5B distill, synthetic arithmetic,
digit readout" and asks whether any of it transfers. §6 is the answer.

---

## 6. Two venues, two scopes

### 6a. Workshop paper, submittable now

Everything in §2 exists on disk, is committed, and has survived three rounds of
external review. The literature pass is complete. What is missing is the
write-up, not evidence.

- Title direction: *"One step, and no further: where a patched residual state is
  read in a written chain of thought."*
- Spine: D1 (pre-registered, matched, p = 5.96e-8) → D7 (the same dissociation
  *within* one item, p = 9.0e-44, which is also D8, the positive control that
  makes the depth-2 semantic null attributable) → D2 (a semantic cliff with
  distributional decay, not token distance alone) → D3+D4 (the written path
  value is behaviorally required) → §3
  stated as limits, with §3.1 and §3.2 given real space rather than a footnote.
- E3 changed the spine's shape, not its length: D1 stays the headline because it
  is the pre-registered one, and D7 goes immediately after it because it answers
  the first question a reader has about D1, namely whether the depth-2 null is the
  model or the intervention.
- Quote rates, not arm verdicts, from `e3_ladder` (§3.10). The prose must not
  say "positive in three of five arms."
- The negative and the methodology are **part of the contribution**, not a
  limitations section. A workshop that takes "here is a control that failed and
  here is what it cost us to notice" is the right room for this.
- Cost: the draft and clustered summaries from existing artifacts. No GPU.

### 6b. Main conference, and why E4 is not on the critical path

This is a genuine main-conference candidate, and it is the stronger of the two
threads in this repo, because the effect is large, pre-registered, and cleanly
controlled. What it is missing is not rigor but **estimand and scope**: it is a
contrast on one checkpoint and one format. It is better measured than it was on
2026-08-15, but it is not broader.

> **E4, the node-by-node influence matrix against the transitive reduction, is
> not on this paper's critical path.** An earlier draft ranked it first. E3
> answers the decision that motivated it, but it does not measure every possible
> influence-matrix estimand:
>
> 1. **The exact-semantic result is already sufficient for the paper's claim.**
>    At every measured path site with two or more remaining operations, the
>    implied digit is top 0/432 times; one-step sites land it 555/603 times across
>    overlapping token-distance bands. A full semantic matrix is unlikely to
>    change the immediate-read result. This does not show that off-diagonal
>    activation influence is zero. D5 measures smaller probability and TV
>    effects at multi-step sites. Mapping those effects would be a different
>    experiment with a different estimand.
> 2. **The clean/implied/raw assay makes a nonzero semantic cell interpretable.**
>    The chain edit is implied 288/288 and raw 0/288. The depth-1 ancestor edit is
>    implied 267/315, raw 46/315 and clean 0/315. The ancestor is a measured
>    mixture dominated by recipient-side transformation, not the unresolved coin
>    flip suggested by the change-score margin. Any future matrix should report
>    these semantic classes rather than the margin alone.

**What survives of E4.** The version relevant to the paper's structural claim
runs on a format that clears G1, where the intermediate is *unwritten* and a
non-adjacent semantic cell could be nonzero. A distributional matrix on the
current format is also possible, but it asks where weaker activation influence
remains, not whether the transplanted value controls the computation. Everything
below is about reaching the G1 format.

**The gate, and why its two halves are one experiment.** A format in which the
intermediate is *unwritten* is itself the intervention that does not copy a
written value: with no digit at the patched position to copy, the propagate/copy
confound largely dissolves at depth ≥ 2. So "clean-valid multi-step format" and
"selective propagation" are not two runs in sequence. They form one screen with
two acceptance criteria.

- **G1, clean validity.** A (model, format) pair where at least one intermediate
  on the dependency path is unwritten *and* the model still solves the clean task
  at high p(target). The naive version is already known to fail: D3's
  comment-padded omission drops clean p(target) to 0.240 at depth 2 and 0.050 at
  depth 3. This is the hard part and it may not be reachable on the current
  checkpoint at all. The log states the reason plainly: this distill was
  never trained on traces with unstated steps, and "never learned to" is not the
  same claim as "cannot".
- **G2, selective propagation.** On a format that clears G1: the patch moves the
  answer toward the implied value, and does so for ancestors and not for
  non-ancestors, with the raw-digit reading no longer available as an explanation.
  E3 shows that one-step ancestor edits usually transform the donor value and
  that the chain edit does so in every eligible item (D9). G2 asks whether this
  selective semantic control survives another operation when the model cannot
  read the needed intermediate from the trace.

**Only if G1 and G2 both clear does the structural E4 become worth running.** At
that point it converts a two-point contrast into a claim about structure, which is
why it remains in the plan rather than being dropped.

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
   apply it going forward only. E3's own arms keep the labels they were scored
   with, and its rates are unaffected either way.
1. **E5 as a G1 screen**: Base / Instruct / Distill, plus scale, asking one
   question: does any of them solve a task with an unstated intermediate? It is
   inexpensive, and the rest of the plan depends on its outcome.
2. **Format search for G1**, informed by 1. Second operation/format family folds
   in here rather than being a separate item, since one notation and one
   operation set also confounds "depth" with a rendering.
3. **G2 on whatever clears G1.** Registered in advance, with a caliper on the
   matching rule and the screening cap fixed beforehand, following the standing rule
   from the 2026-08-15 corrections. Include a chain-edit arm: it is one extra row per
   item and it supplies the in-arm positive control that made E3's negative
   attributable (D8). On a G1 format there is no written intermediate to patch,
   so the chain site has to be redefined for that design. This should be done at
   registration time rather than afterwards.
4. **E4 on a G1 format, only then.** The current results answer the decision to
   prioritize it; they do not claim to contain a full node-by-node matrix.

GPU cost is genuinely small: a current arm is ~225 forward passes of 127-token
sequences on a 1.5B model. The constraint is item-family design rather than compute.
This has been true at every step of this thread and should be planned for.

**Timeline honesty.** This is a longer path than "run E4 next", and G1 is a real
risk of failing outright. That does not touch §6a: the workshop paper is complete
on the evidence that exists, and none of it depends on this gate clearing.

---

## 7. Standing rules carried forward

- Do not backfill or rewrite the archived JSON files. Anything recovered after
  the fact is stored beside them.
- The verdict space stays three-valued.
- No epsilon should be added for the 0.0153-vs-0.0152 surface case. It remains a
  failure, and the 4/5 rule already handles it.
- No DVC for this thread: there is no remote on this host, and `.dvc/cache` would be
  the only copy.
- `notebooks/15_dag_paper_story.ipynb` and
  `notebooks/16_dag_workshop_story.ipynb` are generated by scripts and executed
  by a `jupyter_client` runner; never hand-edit the `.ipynb` files.
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
  1. Turn `notebooks/16_dag_workshop_story.ipynb` into the manuscript. It now
     carries the three-figure, three-table workshop spine; notebook 15 remains
     the full evidence ledger.
  2. `notebooks/14_rmd_paper_story.ipynb` still cites `PAPER_STRATEGY.md` §3/§7c
     in a markdown cell; the file split on 2026-08-15 and the reference is stale
     (the section numbers still resolve, the filename does not).
  3. Convert the literature and claim audit already summarized in notebook 16
     into the manuscript's related-work section.
- The post-hoc carry/copy recount on the 24 matched E2 pairs is no longer open
  debt. E3 supplies the committed, at-N semantic analysis. Keep the E2 19/5
  result as exploratory history; add a matched-family sensitivity artifact only
  if a reviewer asks for it.
- `EXPERIMENT_LOG.md` correction #4 currently ends "Restoring it is the next
  small run." That is no longer the plan: the cross-item arm exists and has run
  (four seeds, 2026-08-14); what E2 lacks is only the *matched-pairs* version,
  and the protocol-deviation label stands either way.
