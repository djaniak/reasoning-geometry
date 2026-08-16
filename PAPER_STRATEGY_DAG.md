# Paper Strategy (DAG) — Where a Patched Residual State Is Read in a Written Chain of Thought

*Durable working doc, opened 2026-08-15. Survives context compaction.*

> **Split out of `PAPER_STRATEGY.md`, 2026-08-15.** That file is now
> [`PAPER_STRATEGY_RMD.md`](PAPER_STRATEGY_RMD.md) and covers the
> selective-prediction / Mahalanobis thread only. **The two are separate papers.**
> They share no model (1.5B distill here, 7–8B there), no data (synthetic
> arithmetic DAG here, MATH-500 there), no metric, and no claim. Do not staple
> them; do not let a sentence from one qualify the other.

Sources of record: `EXPERIMENT_LOG.md` entries 2026-08-13 through 2026-08-15,
`results/dag_patching/` (eight archived pilot artifacts plus five later
sub-packages), and `notebooks/15_dag_paper_story.ipynb`, which is the only
narrative write-up outside those two.

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

## 1. Current thesis (2026-08-15)

> **In a written chain of thought, a patched residual state changes the answer
> only at the node the answer reads directly. One step further up the chain the
> same intervention does nothing, and the reason is that the model is reading the
> written intermediate value rather than recomputing it — unwriting that value
> does not restore the patch, it destroys the model's ability to answer at all.**

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
| **D2** | The collapse is not a token-distance artifact | `paired_ladder`: depth2_gap0 (dist 23) `median_delta_toward` 1.59 vs depth1_gap1 (dist 24) 7.67 at L13; depth3_gap0 (dist 35) 1.06 vs depth1_gap2 (dist 37) 6.94 | Descriptive, 5 items × 1 seed, no test |
| **D3** | The model *reads* the written intermediate rather than computing it latently | `written_vs_omitted`: omitting the chain value drops clean p(target) to 0.240 (depth 2) and 0.050 (depth 3) from 0.996/0.999 — the patch is not restored, the task is destroyed | Strong, with its own control |
| **D4** | D3's collapse is attributable to the missing path value alone | `--omit decoy` arms are indistinguishable from written arms: 5/5 clean at p(target) 0.997/0.999, nothing moved | Clean separation of the two changes |
| **D5** | The depth-2 patch is not inert — it reaches the read position and does nothing useful there | median TV at L13: depth-1 ancestor 0.9868, depth-2 ancestor 0.0877, nulls ~0.004; at depth 2 p(target) 0.913→0.804 and remaining-digit mass 0.078→0.162 while p(implied) stays 0.001 | Descriptive, matched batch |
| **D6** | The intervention is quiet where it should be | 0/144 null rows flipped at either depth; 0/192 control rows moved at layer 13 | Registered validity gate |

**D3+D4 are the most under-appreciated pair in the thread.** They convert the
depth result from "patches decay with distance" — which is a boring claim and
also not what was measured — into a statement about *what the written trace is
for*: at depth ≥ 2 the answer is determined by the written intermediate token,
the patch does not touch that token, and nothing downstream reads the state the
patch does write.

---

## 3. What is NOT established (the honest limits, in rank order of danger)

1. **The mechanism is not shown to be computation over the graph.** The one
   contrast that separates "the recipient transformed the donor value" from "the
   readout copied the written digit" is the log-odds margin
   `delta_toward − delta_toward_raw`. On `v2_paired`, 20 items, the within-item
   ancestor edit is **a coin flip: 10 propagated / 8 copied**, and the cross-item
   donor leans to copying, 6/13. The directional gate that reads 5/5 in every arm
   ever run never asked this question. *(A post-hoc recount on the 24 matched E2
   pairs gives 19/5 at depth 1, and the argmax reading there is cleaner still —
   raw is never uniquely on top, 0/24, despite carrying 0.1065 of the mass. This
   is reported **beside** the 10/8, never instead of it: it is post-hoc on a
   registered run, and the two batches differ in three ways at once —
   `v2_paired`→`v3_distinct`, bfloat16→float32, unmatched→matched — so nothing
   attributes the improvement to any one of them. Not yet written into an
   artifact; see §7.)*
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
7. **Two verdict systems disagree on depth 2.** E2's registered null-flip gate
   calls it valid-and-negative; `dag_patching`'s arm scorer calls it an *invalid
   test* (`directional_control_failed`, `surface_above_null`) because its gates
   are relative and there is no movement to be relative to. Both labels are in
   the artifacts. The contrast stands on the paired comparison; the depth-2 arm
   taken alone is not independently scoreable.
8. **High-confidence regime only.** Matched window p(target) 0.696–0.990, median
   0.914 — the upper half of the range the original depth-1 result came from.
9. **float32 is a model run, not a readout.** E2 ran every matmul in float32; the
   eight archived arms are bfloat16. E2 is internally valid (both depths at one
   precision) but is **not** a same-precision replication, and `dag_pooling.pool`
   refuses to merge across the difference.

---

## 4. The methodological contribution (do not undersell this)

The thread produced four transferable results about how to run and score
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
- Spine: D1 (pre-registered, matched, p = 5.96e-8) → D2 (not distance) → D3+D4
  (the model reads, it does not compute) → §3 stated as limits, with §3.1 and
  §3.2 given real space rather than a footnote.
- The negative and the methodology are **part of the contribution**, not a
  limitations section. A workshop that takes "here is a control that failed and
  here is what it cost us to notice" is the right room for this.
- Cost: the draft, plus the literature pass. No GPU.

### 6b. Main conference — and why E4 stays closed

This is a genuine main-conference candidate, and it is the stronger of the two
threads in this repo, because the effect is large, pre-registered, and cleanly
controlled. What it is missing is not rigor but **estimand and scope**: right now
it is a two-point contrast on one checkpoint.

> **E4 — the node-by-node influence matrix against the transitive reduction — is
> gated, not next.** An earlier draft of this section ranked it first. That was
> wrong, and the external review that said to keep it closed until a clean-valid
> multi-step format produces selective propagation is correct. Two reasons, both
> already in the evidence:
>
> 1. **The matrix's answer is already known on this format, for an uninteresting
>    reason.** Every intermediate is written — `dag_tasks` states k−1 results at
>    depth k — so by D1 and D3 the influence of any non-adjacent node is zero
>    *because the written token determines the answer and the patch does not touch
>    it*, not because the model's computation lacks that edge. The matrix would
>    reproduce the depth-1 adjacency and nothing else. That is D1 re-measured at
>    more positions, at roughly thirty times the forward passes, and it would be
>    written up as a structural result when it is not one.
> 2. **A nonzero cell is not yet interpretable.** §3.1: propagate-vs-copy is a
>    coin flip on the within-item edit (10/8) and the cross-item control fails
>    specificity on all four seeds. Every cell of the matrix inherits that
>    ambiguity, so building a 2-D structure out of it multiplies the problem
>    rather than resolving it.

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
  This is the claim §3.1 currently cannot support.

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

1. **E5 as a G1 screen** — Base / Instruct / Distill, plus scale, asking one
   question: does any of them solve a task with an unstated intermediate? Cheap,
   and it is the fork the rest of the plan hangs on.
2. **Format search for G1**, informed by 1. Second operation/format family folds
   in here rather than being a separate item, since one notation and one
   operation set also confounds "depth" with a rendering.
3. **G2 on whatever clears G1.** Registered in advance, with a caliper on the
   matching rule and the screening cap fixed beforehand — the standing rule from
   the 2026-08-15 corrections.
4. **E4, only then.**

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
- **Open write-up debt:** the post-hoc propagate/copy recount on the 24 matched
  E2 pairs (§3.1) is computed but not yet in `ANALYSIS.json`, the E2 README, or
  `EXPERIMENT_LOG.md`. It needs a `--reanalyse` pass following the
  `exact_paired` precedent — added-after-the-run, exploratory, reported and never
  gated.
- `EXPERIMENT_LOG.md` correction #4 currently ends "Restoring it is the next
  small run." That is no longer the plan: the cross-item arm exists and has run
  (four seeds, 2026-08-14); what E2 lacks is only the *matched-pairs* version,
  and the protocol-deviation label stands either way.
