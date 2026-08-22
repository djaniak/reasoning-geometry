# DAG patching: literature and claim boundary

Date: 2026-08-16

This note records the literature audit that should govern the paper draft. It is
not a full bibliography. The experiment record remains
[`EXPERIMENT_LOG.md`](../../EXPERIMENT_LOG.md), and the paper plan remains
[`PAPER_STRATEGY_DAG.md`](../../PAPER_STRATEGY_DAG.md).

## Safe thesis

> We introduce a clean/implied/raw counterfactual assay for native-position
> activation transplantation in generated arithmetic traces. In a pretrained
> reasoning-tuned 1.5B model, a patched value usually controls the recipient's
> next arithmetic update, while exact semantic control is absent when two or more
> written operations remain. Distributional influence persists at those
> multi-step sites. Same-trace and token-distance controls locate this semantic
> boundary at the next read, while the three-way outcome separates recipient-side
> transformation, literal donor copying, and preservation of the clean
> computation.

This is a result about one checkpoint and one fully written synthetic format. It
is not evidence that transformers cannot maintain latent state across several
steps.

## What the evidence supports

| Result | Evidence | Interpretation |
|:---|:---|:---|
| One remaining written operation | Implied digit is top in 555/603 eligible E3 sites | The transplanted value often controls the next arithmetic update |
| Two or more remaining operations | Implied digit is top in 0/432 eligible E3 sites | Exact semantic control does not survive another written intermediate |
| Same trace, two sites | Last intermediate: 144/144; its ancestor: 0/144 | The boundary is not a generic failure of the intervention |
| Overlapping distance bands | One-step sites succeed while multi-step sites fail in the same token-distance bands | Token distance alone does not explain the boundary |
| Depth-1 ancestor outcomes | Implied 267/315, raw 46/315, clean 0/315 | Recipient-side transformation dominates, with literal copying as a minority outcome |
| Multi-step diagnostics | TV and probability movement remain after implied argmax falls to zero | “No exact semantic control” does not mean “the patch does nothing” |

The 1,035 sites are clustered within generated items, arms, gap placements, and
seeds. Report counts and seed- or item-level summaries. Do not describe them as
1,035 independent trials.

The exploratory paired gap analysis adds another design limit. Among 81 items
eligible at all three placements, the 11 that switch from implied to raw lose a
median 0.129 in clean p(target), compared with 0.045 for 65 items that remain
implied (Mann–Whitney p = 0.0022). Placement, clean confidence and copying move
together; this design does not separate their effects.

The omission pilot supports dependence on a written intermediate. It does not
identify an overwrite mechanism, and it should not carry the headline claim.

## Closest prior work

Source detail for every paper named here is in the companion
[literature pass](2026-08-16-dag-literature-pass.md). That note records the
evidence; this one governs the claims.

### The nearest precedent

[Shih, Winnicki, and Darve 2026](https://arxiv.org/abs/2606.29522) edit an
internal representation of a written state while keeping the visible scratchpad
text fixed, predict the exact downstream state from a known transition rule, and
run explicit controls against copying. Their two eight-state transition systems
(Q₈, D₈) split each state into a visible coordinate and an order-sensitive phase
bit; the edit is a rank-16 projection into a phase-bit subspace applied to
`resid_pre` at layer 12 of the current-state token, so the printed token and the
visible coordinate are unchanged by construction. They compare an unmodified
pretrained base with two matched LoRA conditions. The final-answer-only and
running-state models use identical move sequences and final states; only the
running-state targets include intermediate states.

Two of their controls are the ones that bear on us:

- **Move-swap.** The edited state is held fixed while the upcoming move changes
  (+0.57 selectivity in Q₈, +0.68 in D₈). This rules out a fixed answer bias: the
  prediction must combine the edited state with the current move.
- **Conflicting continuation** — their "computed, not copied" test. A real
  occurrence of the counterfactual state is injected from another context whose
  future disagrees with the current move. The running-state model follows the
  current move 80% (Q₈) and 90% (D₈) of the time against 20%/10% for the injected
  future.

This blocks any claim that our work is the first causal intervention on written
state, the first exact scratchpad counterfactual, or the first
computation-versus-copy test. **Do not present clean/implied/raw as a new kind of
control.** Present it as a more granular, arithmetic-specific measurement: their
copy competitor is another context's *continuation*, scored as a selectivity
contrast between two rule-defined branches, and literal token copying is never
scored as its own outcome category. Ours scores three digits that are distinct by
construction on every row, so recipient-side transformation, literal reproduction
of the digit at the patched position, and preservation of the clean computation
are separated per item rather than inferred from a contrast between conditions.

The useful contrast is empirical and scoped:

| Shih et al. | This work |
|:---|:---|
| Task-specific running-state supervision | Pretrained reasoning-tuned checkpoint, no task-specific state training |
| Learned low-rank edit in a phase-bit subspace, printed token fixed by construction | Whole residual state transplanted at its native token position |
| Persistent state can affect later updates (branch persistence reported to k=4) | Exact semantic control stops after the next written update |
| Copy competitor is another context's continuation, scored as a selectivity contrast | Clean/implied/raw arithmetic outcomes separate transformation, copying, and clean preservation per item |
| Phase bit is a single order-sensitive bit | Recipient must apply its own remaining arithmetic to a transplanted digit |

**Guardrail on this table.** The contrast is consistent with a supervision
hypothesis — that training a model to write running states is what produces a
persistent causal register — and Shih et al. do isolate supervision *within*
their own design. It does not follow that supervision explains the difference
between their persistence and our one-step boundary. The models, tasks,
interventions, and outcome measures all differ. State the contrast; do not
attribute it.

### Other boundaries set by prior work

- [Geiger et al. 2021](https://arxiv.org/abs/2106.02997) formalize interchange
  interventions and exact high-level counterfactual tests.
  [Boundless DAS on Alpaca 7B](https://arxiv.org/abs/2305.08809) carries the move
  onto an off-the-shelf pretrained model doing numeric reasoning, at ≈94%
  interchange-intervention accuracy. Together these own "a known computation as
  the interventional reference".
- [Tan 2023](https://aclanthology.org/2023.blackboxnlp-1.12/) and [Kudo et al.
  2026](https://aclanthology.org/2026.findings-eacl.59/) study causal use of
  arithmetic chain-of-thought.
- [Brinkmann et al. 2024](https://aclanthology.org/2024.findings-acl.242/) combine
  a known symbolic task structure with mechanistic intervention, and state in
  their abstract that intermediate results are stored in selected token
  positions.
- [Zhang et al. 2025](https://aclanthology.org/2025.acl-long.668/) study exact
  state tracking in chain-of-thought transformers.
- [Patchscopes](https://openreview.net/forum?id=5uwBzcn885) and [Mehrafarin et
  al. 2026](https://arxiv.org/abs/2604.23351) establish hidden-state
  transplantation as a way to expose or redirect reasoning information.
- [Circuit Tracing and On the Biology of a Large Language
  Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
  swap a latent intermediate (Texas→California) and read a transformed output
  ("Sacramento"). A patched intermediate being transformed rather than emitted is
  theirs. Their intermediate is latent, so no written digit could have been
  copied and the competitor never arises.
- [Cywiński et al. 2025](https://www.alignmentforum.org/posts/YGAimivLxycZcqRFR/can-we-interpret-latent-reasoning-using-current)
  patch latent thought vectors on a three-step word problem using a
  same-intermediate patch as a null. The null-patch design is theirs.
- [Lanham et al. 2023](https://arxiv.org/abs/2307.13702) provide the main
  behavioral intervention background for dependence on visible reasoning.

### Two papers to argue with by name

**[Kudo et al. 2026](https://aclanthology.org/2026.findings-eacl.59/) — pressure
on the framing, not a contradiction.** They title their paper *LLMs Faithfully
and Iteratively Compute Answers During CoT* and report sub-answers arising during
generation. Their patching is confined to the input/equation region before CoT
generation begins. That is a question about *when a value first becomes
available*; ours is about *what determines the answer once a value is written*.
The two can both hold. Say so explicitly in the related-work section — a reader
who sees only the titles will assume a conflict — but do not stage it as a
disagreement, because it is not one.

**[Garcia 2026](https://arxiv.org/abs/2605.10799) — a direct confound to
address.** The claim is that corruption sensitivity tracks the location of
explicit answer text rather than a fixed computational depth. This is the
alternative explanation our design most invites, and it must be named in the
manuscript rather than left for review. D2 (banding sites by token distance and
showing the step split inside bands that hold both) and D7 (two sites paired
within the same trace) **mitigate** it: they show the boundary is not explained by
placement alone. They do not dispose of it. Step count and site role remain
bundled, our arms are not an exhaustive sweep of placements, and Garcia's setting
is text corruption rather than activation patching, so the mapping between the
two designs is itself an assumption. Report the mitigation and the residual.

## Defensible contribution

The contribution is the assay plus the observed boundary, not a new list of
ingredients:

1. A clean/implied/raw semantic partition for each intervention. The claim is
   granularity, not priority: prior computation-versus-copy controls
   ([Shih et al.](https://arxiv.org/abs/2606.29522)) contrast conditions or
   continuations, while this partition scores three digits that are distinct by
   construction on every row, so transformation, literal reproduction of the
   digit at the patched position, and preservation of the clean computation are
   read off a single item.
2. A same-trace comparison between an immediately consumed state and its
   ancestor.
3. A one-step versus multi-step boundary that is not explained by token distance
   alone.
4. Evidence that distributional influence can remain after exact semantic
   counterfactual control disappears.

Avoid unqualified “first” claims. In particular, do not claim the first causal
test of chain-of-thought, first graph-grounded patching experiment, first hidden
reasoning-state transplant, or first computation-versus-copy control.

## Wording guardrails

Use:

- “interchange-style activation patching” or “native-position activation
  transplantation”. [Heimersheim &
  Nanda](https://arxiv.org/abs/2404.15255) treat activation patching, causal
  tracing, interchange intervention and resample ablation as one family; say so
  once and then pick one name;
- “noising direction” — a counterfactual state written into an otherwise clean
  run, per the same denoising/noising split;
- “counterfactual (symmetric-token-replacement-style) donor”, citing [Zhang &
  Nanda](https://arxiv.org/abs/2309.16042) for why an in-distribution
  counterfactual beats Gaussian noise;
- “clean/implied/raw counterfactual assay”;
- “interchange-intervention accuracy for a single aligned variable” for the
  registered outcome. This buys legibility with causal-abstraction reviewers at
  no cost, provided the qualifier is never dropped — full IIA is over an
  alignment of all variables;
- “log-odds margin toward the implied value relative to the raw value” for the
  propagate-versus-copy statistic. It is a logit-difference metric, which is what
  Zhang & Nanda recommend over probability;
- “number of written steps” or “step count”. Reserve “depth” for the generator
  parameter, with an explicit note that it bundles an operation, a written
  result, a binding and a context change;
- “immediate-read boundary”, defined in one sentence on first use. Acceptable
  alternatives if a reviewer objects to a coinage: “last-written-value locality”,
  “one-step read locality”;
- “the transplanted value does not control the final answer through a second
  explicitly written intermediate in this model and format.”

Avoid, as readings of our own result:

- “the model implements” or “we recovered” the ground-truth DAG;
- “depth causes the cliff”: depth also changes site role and local context;
- “the patch does nothing”: multi-step distributional effects remain;
- “written text overwrites latent state”: the omission pilot does not identify
  that mechanism;
- “the answer depends only on the most recent intermediate”;
- “transformers cannot propagate latent state”;
- “immediate-read *mechanism*”: we have not shown a mechanism;
- any claim that running-state supervision explains the difference between our
  boundary and [Shih et al.](https://arxiv.org/abs/2606.29522)'s persistence —
  the designs differ on several axes at once.

Avoid, as borrowed terms of art:

- “path patching” — a specific technique restricting the intervention to named
  paths ([Goldowsky-Dill et al.](https://arxiv.org/abs/2304.05969)). We overwrite
  a residual state and let every downstream path see it. Cite it as background,
  never as our method;
- “causal scrubbing” — resampling ablation under a hypothesis `(G, I, c)` scored
  by performance recovered. Different question, different statistic;
- “causal abstraction”, “faithful simplification” — reserved terms with formal
  content ([Geiger et al.](https://arxiv.org/abs/2301.04709)). Abstraction
  requires an alignment for every variable across the full intervention set; we
  test one variable at a time, and at ≥2 steps we get zero;
- “circuit” — we localize nothing at component level;
- “faithfulness” used loosely — it means both circuit-faithfulness ([Wang et
  al.](https://arxiv.org/abs/2211.00593)) and CoT-faithfulness ([Lanham et
  al.](https://arxiv.org/abs/2307.13702)). Pick one sense, define it, and never
  let the two share a sentence.

Avoid, as priority claims (each with the work that forecloses it; sources in the
[literature pass](2026-08-16-dag-literature-pass.md)):

- first use of a known ground-truth computation as an interventional reference —
  [Geiger et al. 2021](https://arxiv.org/abs/2106.02997), [Boundless
  DAS](https://arxiv.org/abs/2305.08809);
- first generated dependency graph with exact counterfactual answers —
  [iGSM](https://arxiv.org/abs/2407.20311), [Kudo et
  al.](https://aclanthology.org/2026.findings-eacl.59/);
- first activation patching on synthetic multi-step arithmetic in a pretrained
  model — [Kudo et al.](https://aclanthology.org/2026.findings-eacl.59/), ten
  models;
- first to patch inside a chain of thought and read the final answer —
  [Mehrafarin et al.](https://arxiv.org/abs/2604.23351), [Zhang et
  al.](https://arxiv.org/abs/2509.23676) on distilled DeepSeek-R1;
- showing that intermediate results are stored at token positions —
  [Brinkmann et al.](https://arxiv.org/abs/2402.11917), their abstract;
- showing that the model relies on its written chain of thought —
  [Lanham et al.](https://arxiv.org/abs/2307.13702), [Zhang et
  al.](https://arxiv.org/abs/2509.23676);
- first demonstration that a patched intermediate is transformed rather than
  emitted — [Anthropic's Texas→California
  swap](https://transformer-circuits.pub/2025/attribution-graphs/biology.html),
  [Patchscopes](https://arxiv.org/abs/2401.06102);
- first computation-versus-copy control — [Shih et
  al.](https://arxiv.org/abs/2606.29522);
- introducing the null-patch control — [Cywiński et
  al.](https://www.alignmentforum.org/posts/YGAimivLxycZcqRFR/can-we-interpret-latent-reasoning-using-current);
- first depth limit on multi-hop interventions — [Biran et
  al.](https://arxiv.org/abs/2406.12775) (layers), [Brinkmann et
  al.](https://arxiv.org/abs/2402.11917) (layers), [Liang &
  Pan](https://arxiv.org/abs/2602.00449) (hop length in latent CoT);
- a present-but-unused signal as a new phenomenon — [Sharma et
  al.](https://arxiv.org/abs/2604.22128).

One scoping correction to carry into the prose: the propagate-versus-copy
ambiguity we expose is not a general indictment of directional patching
statistics. It bites only where the patched position carries a value the readout
could emit verbatim, which is false for most of this literature — factual recall,
IOI, binding. Write “exposed whenever the patched position carries a value the
readout could emit verbatim”, never “exposed”.

## Related-work structure

Keep the section short and organized by the question each group answers:

1. High-level causal tests: Geiger; causal scrubbing and path-patching context.
2. Algorithmic and arithmetic mechanisms: Brinkmann, Stolfo, Zhang.
3. Causal use of written reasoning: Tan, Kudo, Shih, Lanham.
4. Hidden-state transplantation: Patchscopes, Mehrafarin, recent reasoning-model
   activation patching.

For a workshop submission, the next step is the draft and clustered summaries,
not another broad depth or gap sweep. A main-track extension needs broader scope
or a sharper supervision comparison; the current evidence does not support a
broad model-class claim.
