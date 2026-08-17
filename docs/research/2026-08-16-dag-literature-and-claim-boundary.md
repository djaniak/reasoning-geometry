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

The nearest precedent is [Shih, Winnicki, and Darve
2026](https://arxiv.org/abs/2606.29522). They edit an internal representation of
a written state while keeping visible text fixed, predict the exact downstream
state, and use controls against copying. Their task-specific running-state
training produces a persistent causal register. This blocks claims that our
work is the first causal intervention on written state, the first exact
scratchpad counterfactual, or the first computation-versus-copy test.

The useful contrast is empirical and scoped:

| Shih et al. | This work |
|:---|:---|
| Task-specific running-state supervision | Pretrained reasoning-tuned checkpoint, no task-specific state training |
| Learned low-rank state edit | Whole residual state transplanted at its native token position |
| Persistent state can affect later updates | Exact semantic control stops after the next written update |
| Controls combine edited state with recipient continuation | Clean/implied/raw arithmetic outcomes separate transformation, copying, and clean preservation |

Other boundaries set by prior work:

- [Geiger et al. 2021](https://arxiv.org/abs/2106.02997) formalize interchange
  interventions and exact high-level counterfactual tests.
- [Tan 2023](https://aclanthology.org/2023.blackboxnlp-1.12/) and [Kudo et al.
  2026](https://aclanthology.org/2026.findings-eacl.59/) study causal use of
  arithmetic chain-of-thought.
- [Brinkmann et al. 2024](https://aclanthology.org/2024.findings-acl.242/) combine
  a known symbolic task structure with mechanistic intervention.
- [Zhang et al. 2025](https://aclanthology.org/2025.acl-long.668/) study exact
  state tracking in chain-of-thought transformers.
- [Patchscopes](https://openreview.net/forum?id=5uwBzcn885) and [Mehrafarin et
  al. 2026](https://arxiv.org/abs/2604.23351) establish hidden-state
  transplantation as a way to expose or redirect reasoning information.
- [Lanham et al. 2023](https://arxiv.org/abs/2307.13702) provide the main
  behavioral intervention background for dependence on visible reasoning.

## Defensible contribution

The contribution is the assay plus the observed boundary, not a new list of
ingredients:

1. A clean/implied/raw semantic partition for each intervention.
2. A same-trace comparison between an immediately consumed state and its
   ancestor.
3. A one-step versus multi-step boundary that survives token-distance controls.
4. Evidence that distributional influence can remain after exact semantic
   counterfactual control disappears.

Avoid unqualified “first” claims. In particular, do not claim the first causal
test of chain-of-thought, first graph-grounded patching experiment, first hidden
reasoning-state transplant, or first computation-versus-copy control.

## Wording guardrails

Use:

- “interchange-style activation patching” or “native-position activation
  transplantation”;
- “clean/implied/raw counterfactual assay”;
- “immediate-read boundary”;
- “the transplanted value does not control the final answer through a second
  explicitly written intermediate in this model and format.”

Avoid:

- “the model implements” or “we recovered” the ground-truth DAG;
- “depth causes the cliff”: depth also changes site role and local context;
- “the patch does nothing”: multi-step distributional effects remain;
- “written text overwrites latent state”: the omission pilot does not identify
  that mechanism;
- “the answer depends only on the most recent intermediate”;
- “transformers cannot propagate latent state.”

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
