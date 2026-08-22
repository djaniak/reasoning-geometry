# DAG document update and next steps

Date: 2026-08-20  
Reviewed experiment commit: `ca7c0e4ef01c95a88b02966c99622787ba1d3523`  
Comparison point: `ca7c0e4^`  
Experiment log entries reviewed: 42

## Orientation

The commit improves the literature and claim boundary. It adds no experiment.
That is the right direction: the workshop evidence is already sufficient, and
the paper strategy says the missing work is the write-up, not another model run
([strategy](../../raw/repos/reasoning-geometry/PAPER_STRATEGY_DAG.md:306)).

The notebook change does not complete the previous rewrite action. The source
cells before and after the commit have the same normalized hash. The 1,849-line
diff is generated JSON and output churn. The notebook still says that literature
positioning is unwritten
([notebook 15](../../raw/repos/reasoning-geometry/notebooks/15_dag_paper_story.ipynb:550)).

Status of the previous review's actions:

- Rewrite the paper story around the final claim boundary: **not completed by
  this commit**.
- Build final figures and compact tables: **not completed by this commit**.
- Draft the workshop paper before the main-track fork: **not done**.
- Defer new model runs until after the workshop draft: **done**. The experiment
  ledger has no new entry.

## Literature pre-registration

I recorded this view before rereading the experiment ledger and findings.

### Local literature read

- `raw/repos/reasoning-geometry/RELATED_WORK.md` in full. It concerns the older
  selective-prediction thread, not DAG patching
  ([literature pass](../../raw/repos/reasoning-geometry/docs/research/2026-08-16-dag-literature-pass.md:27)).
- Both DAG literature and claim-boundary notes in full.
- The local notes for Geiger, Kudo, Anthropic circuit tracing, Coconut,
  Patchscopes, latent CoT planning, causal-CoT dynamics, and reasoning-restored
  decision tokens.

### What would be new

A useful contribution combines an exact per-item clean/implied/raw outcome with
a within-trace intervention that varies the number of written updates remaining.
It shows a repeatable boundary after controlling token distance and separates
exact semantic control from weaker distributional influence. The claim is about
one model and format, not a universal transformer mechanism.

### What would be repackaging

Activation patching, exact high-level counterfactuals, generated task graphs,
causal use of written reasoning, hidden-state transplantation, downstream
transformation of an edited intermediate, and computation-versus-copy controls
all have prior art. Shih et al. is the closest design. The updated note now says
this clearly and limits the contribution to the assay's granularity plus the
observed boundary
([claim boundary](../../raw/repos/reasoning-geometry/docs/research/2026-08-16-dag-literature-and-claim-boundary.md:80)).

## A. Stage diagnosis

**Current stage: distillation for the workshop paper.** The core result is beyond
exploration. The matched E2 contrast is 24/24 against 0/24 with the correct
paired test, and E3 supplies the same-trace positive control and the larger
one-step/multi-step split
([strategy](../../raw/repos/reasoning-geometry/PAPER_STRATEGY_DAG.md:68)). The safe
thesis is narrow and suitable for a draft
([claim boundary](../../raw/repos/reasoning-geometry/docs/research/2026-08-16-dag-literature-and-claim-boundary.md:10)).

The main-track branch remains in understanding. It still has one checkpoint, one
task family, one operation set, and one notation
([strategy](../../raw/repos/reasoning-geometry/PAPER_STRATEGY_DAG.md:118)). Do not
let that branch delay the workshop manuscript.

## B. Truth audit

### What is supported

- The one-versus-two-step contrast survives matching on clean confidence and
  ancestor token distance: 24/24 against 0/24
  ([strategy](../../raw/repos/reasoning-geometry/PAPER_STRATEGY_DAG.md:79)).
- In E3, one-step sites land on the implied digit in 555/603 cases, while sites
  with two or more written operations do so in 0/432 cases
  ([strategy](../../raw/repos/reasoning-geometry/PAPER_STRATEGY_DAG.md:68)).
- Same-trace chain edits show that the multi-step null is not a generic failure
  of the intervention
  ([strategy](../../raw/repos/reasoning-geometry/PAPER_STRATEGY_DAG.md:106)).
- The revised claim note correctly treats Kudo as framing pressure rather than a
  contradiction, Garcia as a mitigated but unresolved confound, and supervision
  as a hypothesis rather than a causal explanation
  ([claim boundary](../../raw/repos/reasoning-geometry/docs/research/2026-08-16-dag-literature-and-claim-boundary.md:101)).

### What must be corrected before drafting

1. **Shih is described inaccurately in two ways.** The base model does not differ
   from the two LoRA arms only by whether intermediate states appear in targets;
   it receives no task-specific update. Only the final-answer and running-state
   LoRA arms isolate that target difference
   ([claim boundary](../../raw/repos/reasoning-geometry/docs/research/2026-08-16-dag-literature-and-claim-boundary.md:65)).
   The source matrix also says that no written digit exists or could be copied
   ([source matrix](../../raw/repos/reasoning-geometry/docs/research/2026-08-16-dag-literature-pass.md:52)).
   Shih patches at a printed state token and keeps that token fixed. The valid
   distinction is narrower: Shih does not score literal reproduction of the
   patch-site token as a separate outcome category.

2. **Claim ownership is still duplicated.** The source matrix says the claim
   note governs wording and that duplicate verdict material was removed
   ([source matrix](../../raw/repos/reasoning-geometry/docs/research/2026-08-16-dag-literature-pass.md:10)),
   but it retains a 16-item forbidden-claims section
   ([source matrix](../../raw/repos/reasoning-geometry/docs/research/2026-08-16-dag-literature-pass.md:353)).
   The claim note contains the governing version
   ([claim boundary](../../raw/repos/reasoning-geometry/docs/research/2026-08-16-dag-literature-and-claim-boundary.md:251)).
   Delete the duplicate section from the source matrix. Keep the matrix,
   per-family evidence, and verification record.

3. **The paper strategy still overstates the omission control.** D4 says the
   collapse is attributable to the missing path value alone
   ([strategy](../../raw/repos/reasoning-geometry/PAPER_STRATEGY_DAG.md:102)). The
   experiment correction says the decoys were not position-matched and therefore
   do not rule out every path-specific alternative
   ([experiment log](../../raw/repos/reasoning-geometry/EXPERIMENT_LOG.md:618)).
   Use the narrower wording already present in the claim note: the pilot supports
   dependence on a written intermediate but does not identify an overwrite
   mechanism
   ([claim boundary](../../raw/repos/reasoning-geometry/docs/research/2026-08-16-dag-literature-and-claim-boundary.md:47)).

4. **Notebook 15 is stale despite the large diff.** Its source did not change,
   and it still claims that the literature positioning is not written
   ([notebook 15](../../raw/repos/reasoning-geometry/notebooks/15_dag_paper_story.ipynb:550)).
   Do not treat the commit as a notebook rewrite.

The quantitative result remains strong after these corrections. They narrow the
story and remove factual errors; they do not change the measurements.

## C. Prioritization review

The research time in this commit went to the correct bottleneck: literature and
claim control. The notebook reserialization was wasted motion because it changed
no source cell. The separate post-training-prefix note is a different research
direction and should not consume the DAG workshop sprint.

The highest-value remaining work is manuscript production from existing
artifacts. The strategy already gives the spine: matched D1, same-trace D7/D8,
the larger D2 boundary, then the omission pilot and limits
([strategy](../../raw/repos/reasoning-geometry/PAPER_STRATEGY_DAG.md:312)).

## D. Literature gap

The literature coverage is now adequate for drafting. The nearest precedent is
identified, read in full, and used to narrow the claim. Kudo and Garcia are in
the argument rather than hidden in a matrix.

One narrow search gap remains: state-tracking vocabulary such as “scratchpad,”
“causal register,” “working memory,” and “transition system,” plus recent
workshop proceedings
([source matrix](../../raw/repos/reasoning-geometry/docs/research/2026-08-16-dag-literature-pass.md:487)).
Run that check during the citation-freeze pass, not as another broad literature
rewrite. The novelty statement must remain an absence-of-found-prior-work claim,
not a priority claim.

## E. Ranked next steps

1. **Fix the three document defects above.** Correct Shih in both notes, remove
   the duplicate forbidden-claims section from the source matrix, and narrow D4
   in the paper strategy. This is a small editorial pass with no compute.

2. **Finish the workshop storyboard and draft.** Use notebook 16 as the short
   manuscript spine and notebook 15 only as the full evidence ledger, as the
   strategy already specifies
   ([strategy](../../raw/repos/reasoning-geometry/PAPER_STRATEGY_DAG.md:456)).
   Generate the figures and tables from analysis code; do not hand-edit notebook
   JSON.

3. **Add clustered summaries and freeze citations.** Report seed- or item-level
   summaries because the 1,035 site observations are clustered
   ([claim boundary](../../raw/repos/reasoning-geometry/docs/research/2026-08-16-dag-literature-and-claim-boundary.md:37)).
   Recheck Shih, Garcia, Kudo, and the narrow state-tracking search immediately
   before submission.

4. **Only after a complete draft, choose the main-track extension.** Record the
   target venue, deadline, GPU budget, and desired paper scope first. The cheapest
   first gate is a clean-valid format with an unwritten dependency-path
   intermediate. The current naive omission format already fails this gate
   ([strategy](../../raw/repos/reasoning-geometry/PAPER_STRATEGY_DAG.md:373)).
   Stop if no format passes. If one passes, then test the same semantic assay.

5. **If broader mechanism is still the goal, run a matched supervision study.**
   Compare final-answer-only and running-state supervision on the same base model,
   task, and intervention. Do not infer the cause from the cross-paper contrast
   with Shih
   ([claim boundary](../../raw/repos/reasoning-geometry/docs/research/2026-08-16-dag-literature-and-claim-boundary.md:101)).
   Checkpoint breadth comes after this, not before it.

## F. What to stop doing

- Stop broad depth and gap sweeps for the workshop. The claim note already says
  the next step is the draft and clustered summaries
  ([claim boundary](../../raw/repos/reasoning-geometry/docs/research/2026-08-16-dag-literature-and-claim-boundary.md:303)).
- Stop re-executing notebook 15 when no source cell changed.
- Stop maintaining claim bans in two files.
- Stop treating the omission pilot as proof of overwrite or of one uniquely
  identified cause.
- Stop opening the post-training-prefix branch during the DAG workshop sprint.
- Stop considering a full node-by-node matrix part of the critical path. The
  strategy already rejects it for the current paper
  ([strategy](../../raw/repos/reasoning-geometry/PAPER_STRATEGY_DAG.md:339)).

## Standards

Summary: 0 P1, 2 P2, 1 P3. Worst issue: the notebook diff is large but carries
no source change.

- **P2 — `notebooks/15_dag_paper_story.ipynb`: generated-output churn.** The
  normalized source-cell hash is identical before and after the commit. The diff
  should either contain the intended source rewrite or omit the notebook.
- **P2 — `docs/research/2026-08-12-post-training-reasoning-prefix-novelty.md`: repository writing style.**
  The note is in Polish while the repository's research record and style rules
  use simple technical English. Translate it if it is meant to be a durable
  shared source; otherwise keep it outside the DAG workshop change.
- **P3 — `docs/research/2026-08-16-dag-literature-pass.md`: loaded prose.** Phrases
  such as “the paper that kills” make an evidence record harder to reuse
  ([source matrix](../../raw/repos/reasoning-geometry/docs/research/2026-08-16-dag-literature-pass.md:54)).
  Replace them during the manuscript edit, not in a separate cleanup project.

## Spec

Summary: 2 P1, 2 P2. Worst issue: the updated source matrix makes a false claim
about Shih's printed state token.

- **P1 — Shih token description is false.** Replace “no digit on the page could
  be copied” and “no written digit exists” with the narrower outcome-measure
  distinction described in the truth audit.
- **P1 — Claim ownership remains duplicated.** Remove the source matrix's
  forbidden-claims inventory; File A should be the only governing claim record.
- **P2 — Shih training-arm description is imprecise.** Only the two matched LoRA
  arms isolate supervision target; the pretrained base is unmodified.
- **P2 — The notebook part of the commit does not implement the requested
  rewrite.** It changes serialization and outputs while leaving source cells
  unchanged.

The requested Kudo, Garcia, supervision-causality, and full-text-Shih corrections
otherwise landed correctly.
