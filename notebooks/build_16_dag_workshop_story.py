"""Build the short DAG workshop storyboard from notebook 15's proven cells.

Run from the repository root:

    uv run python notebooks/build_16_dag_workshop_story.py
    uv run --with pandas --with jinja2 --with jupyter_client --with ipykernel \
        python notebooks/execute_notebook.py \
        notebooks/16_dag_workshop_story.ipynb notebooks

Notebook 15 remains the complete experiment ledger. This notebook keeps the
three figures and three tables needed to draft the workshop paper.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from build_15_dag_paper_story import CELLS as SOURCE_CELLS

REPO = Path(__file__).resolve().parents[1]
CELLS: list[dict] = []


def md(source: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": source.strip("\n").splitlines(keepends=True)})


def code(source: str) -> None:
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": source.strip("\n").splitlines(keepends=True)})


def reuse(index: int, marker: str, replacements: tuple[tuple[str, str], ...] = ()) -> None:
    """Reuse a maintained cell from notebook 15, failing if its role changed."""
    cell = deepcopy(SOURCE_CELLS[index])
    source = "".join(cell["source"])
    assert marker in source, f"notebook 15 cell {index} no longer contains {marker!r}"
    for old, new in replacements:
        assert old in source, f"notebook 15 cell {index} no longer contains {old!r}"
        source = source.replace(old, new)
    cell["source"] = source.splitlines(keepends=True)
    CELLS.append(cell)


md(r"""
# One update, then a semantic cliff

**Workshop storyboard: what a transplanted value controls in a fully written
reasoning trace.**

This notebook is the short paper story. Notebook 15 remains the complete
experiment record. Every number below comes from a committed artifact under
`results/dag_patching/`; the notebook runs no model.

## Paper claim

We transplant the residual state of one arithmetic line from a counterfactual
donor trace into the same token positions of a clean trace. The generated task
names three different outcomes before the intervention: the clean answer, the
answer implied by applying the recipient's remaining arithmetic to the donor
value, and the raw digit written by the donor.

In `DeepSeek-R1-Distill-Qwen-1.5B`, one-update sites land on the implied digit in
555/603 eligible cases. Sites with two or more written operations remaining do
so in 0/432 cases. A same-trace control gives the sharpest comparison: patching
the last intermediate succeeds in 144/144 items while patching its ancestor
succeeds in 0/144. Total variation still changes at multi-step sites, so the
result is a semantic cliff over a graded distributional effect.

The claim is limited to one checkpoint and one fully written synthetic format.
The design rules out token distance and clean confidence as sole explanations.
It does not isolate written-step count from operation, binding, local context,
or patch-site role.
""")

# Artifact loading, plotting style, and the table renderer are shared with 15.
reuse(1, "E3 = arm(\"e3_ladder/ANALYSIS.json\")")

md(r"""
## 1. The assay makes three semantic outcomes distinct

The clean and donor traces have equal length and differ in one arithmetic line.
At layer 13, the donor residual state at that line's two digit positions replaces
the clean state. The visible tokens remain clean.

Because the arithmetic chain is affine, the recipient predicts what should
happen if it uses the transplanted value. The generator keeps that **implied**
digit distinct from both the **raw** donor digit and the **clean** answer. One
intervention therefore distinguishes recipient-side transformation, literal
copying, and preservation of the original computation.
""")

reuse(
    3,
    "One patch, one item",
    replacements=(("how far the patched line sat from the answer (§7, §8)",
                   "how far the patched line sat from the answer (Figure 3)"),),
)

md(r"""
**Figure 1. Intervention and semantic outcomes.** The first three panels show
the clean run, donor run, and native-position transplant. The lower-left panel
shows one readout moving from the clean digit to the recipient-implied digit,
not to the raw donor digit. The lower-right panel gives the pre-registered
matched result: 24/24 one-update items and 0/24 two-update items land on the
implied digit, while matched control edits move 0/192 times. The example explains
the measurement; the aggregate result does not depend on this example.
""")

md(r"""
## 2. Headline result: semantic cliff, distributional decay

E3 scales the ladder to 48 generated items across three seeds and patches the
written intermediates as well as their ancestors. The registered outcome is the
implied digit uniquely on top at layer 13, conditional on a uniquely correct
clean readout.
""")

code(r'''
SITES = pd.DataFrame([
    {
        "site": row["label"],
        "operations remaining": row["steps"],
        "tokens to answer": f"{row['distance_min']}–{row['distance_max']}",
        "n": row["n"],
        "→ implied": f"{row['implied_top_unique']}/{row['n']}",
        "→ raw": f"{row['raw_top_unique']}/{row['n']}",
        "→ clean": f"{row['clean_top_unique']}/{row['n']}",
        "median TV": row["median_tv"],
        "per-seed implied": " · ".join(
            f"{seed['implied_top_unique']}/{seed['n']}" for seed in row["per_seed"]
        ),
    }
    for row in E3["by_site"]
])
table(
    SITES,
    f"Table 1 · semantic outcomes at every E3 patch site, layer {E3['layer']}",
    highlight=[row["steps"] == 1 for row in E3["by_site"]],
    **{"median TV": "{:.4f}"},
    note=("Tinted rows have one written operation remaining. They land on the "
          "recipient-implied digit in 555/603 eligible cases; all sites with two "
          "or more operations remaining land there 0/432 times and preserve the "
          "clean answer 432/432 times. The observations are clustered within "
          "items, placements, arms, and seeds, so the table reports counts and "
          "per-seed rates rather than treating 1,035 sites as independent."),
)
''')

code(r'''
fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.5))
MARK = {"ancestor": ("o", COLOR["ancestor"], "ancestor"),
        "chain": ("s", "#2c6fbb", "written intermediate")}

ax = axes[0, 0]
for kind, (marker, colour, label) in MARK.items():
    rows = [r for r in E3["by_site"] if r["kind"] == kind]
    ax.scatter([r["steps"] for r in rows], [r["median_tv"] for r in rows],
               marker=marker, s=46, color=colour, label=label, zorder=3)
ax.set_yscale("log")
ax.set_ylim(1.5e-3, 2.0)
ax.set_xticks([1, 2, 3])
ax.set_xlabel("written operations remaining")
ax.set_ylabel("median TV, clean → patched (log)")
ax.set_title("a. distributional influence decays", loc="left")
ax.legend(fontsize=7.5)

ax = axes[0, 1]
for kind, (marker, colour, label) in MARK.items():
    rows = [r for r in E3["by_site"] if r["kind"] == kind]
    ax.scatter([r["steps"] for r in rows],
               [r["implied_top_unique"] / r["n"] for r in rows],
               marker=marker, s=46, color=colour, label=label, zorder=3)
ax.set_ylim(-0.06, 1.08)
ax.set_xticks([1, 2, 3])
ax.set_xlabel("written operations remaining")
ax.set_ylabel("share landing on implied digit")
ax.set_title("b. exact semantic control falls to zero", loc="left")

ax = axes[1, 0]
labels, chain_share, ancestor_share = [], [], []
for row in E3["within_item"]:
    labels.append(f"{row['chain_steps']} vs {row['ancestor_steps']}")
    chain_share.append((row["chain_only"] + row["both"]) / row["n"])
    ancestor_share.append((row["ancestor_only"] + row["both"]) / row["n"])
x = range(len(labels))
ax.bar([i - 0.18 for i in x], chain_share, width=0.36, color="#2c6fbb",
       label="nearer site")
ax.bar([i + 0.18 for i in x], ancestor_share, width=0.36,
       color=COLOR["ancestor"], label="ancestor")
for i, (near, ancestor, row) in enumerate(zip(chain_share, ancestor_share,
                                               E3["within_item"])):
    ax.text(i - 0.18, near + 0.035, f"{round(near * row['n'])}/{row['n']}",
            ha="center", fontsize=7.5)
    ax.text(i + 0.18, ancestor + 0.035,
            f"{round(ancestor * row['n'])}/{row['n']}", ha="center", fontsize=7.5)
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylim(-0.06, 1.08)
ax.set_xlabel("operations remaining: nearer site vs ancestor")
ax.set_ylabel("share landing on implied digit")
ax.set_title("c. the boundary appears inside the same trace", loc="left")
ax.legend(fontsize=7.5)

ax = axes[1, 1]
for steps, colour, marker in ((1, COLOR[1], "o"), (2, COLOR[2], "s"),
                              (3, COLOR[3], "^")):
    rows = [r for r in E3["by_distance_band"] if r["steps"] == steps]
    if rows:
        ax.plot([sum(r["band"]) / 2 for r in rows],
                [r["implied_top_unique"] / r["n"] for r in rows],
                marker=marker, ms=6, mfc="none", color=colour, lw=1.4,
                label=f"{steps} operation" + "s" * (steps > 1))
ax.set_ylim(-0.06, 1.08)
ax.set_xlabel("token distance to answer (band midpoint)")
ax.set_ylabel("share landing on implied digit")
ax.set_title("d. token distance is not sufficient", loc="left")
ax.legend(fontsize=7.5)

fig.tight_layout()
plt.show()
''')

md(r"""
**Figure 2. The main empirical result.** Panel a shows a graded change in the
digit distribution: median TV falls from about 0.99 at one operation to
0.035–0.090 at two and 0.003 at three. Panel b shows a different pattern in the
semantic outcome: one-update sites often install the implied digit, while every
measured multi-update site remains at zero. Panel c makes the comparison within
the same trace. A last intermediate succeeds in 144/144 items where its ancestor
succeeds in 0/144; a two-update written intermediate and its three-update
ancestor both remain at zero. Panel d shows that one- and multi-update sites
remain separated inside overlapping token-distance bands. These controls locate
an immediate-read boundary in this format. They do not make written-step count
the only changing property.
""")

md(r"""
## 3. Pre-registered confirmation removes two measured confounds

The discovery items coupled the intervention with clean difficulty and token
distance. Stage A therefore screened clean traces without patching. A rule
registered before the selected items existed matched 24 depth-1 and depth-2
pairs on clean `p(target)` and ancestor distance. Stage B patched only those
pairs at the inherited layer 13.
""")

code(r'''
screened = pd.DataFrame(SELECTION["screened"])
pairs = SELECTION["selection"]["pairs"]
low, high = SELECTION["selection"]["window"]

fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2))
ax = axes[0]
for depth, frame in screened.groupby("depth"):
    eligible = frame[frame["clean_correct_unique"]]
    ax.hist(eligible["clean_target_share"], bins=40, alpha=0.7,
            label=f"depth {depth}", color=COLOR[depth])
ax.axvspan(low, high, color="#8a8a8a", alpha=0.16)
ax.set_xlabel("clean p(target)")
ax.set_ylabel("eligible items")
ax.set_title("a. overlap chosen before patching", loc="left")
ax.legend(fontsize=7.5)

ax = axes[1]
p1 = [pair[0]["clean_target_share"] for pair in pairs]
p2 = [pair[1]["clean_target_share"] for pair in pairs]
ax.scatter(p1, p2, s=26, color="#4f9153")
limits = [min(p1 + p2) - 0.01, max(p1 + p2) + 0.01]
ax.plot(limits, limits, color="#9a9a9a", lw=1, ls="--")
ax.set_xlim(limits)
ax.set_ylim(limits)
ax.set_xlabel("depth 1 clean p(target)")
ax.set_ylabel("depth 2 clean p(target)")
ax.set_title("b. 24 confidence-matched pairs", loc="left")
worst_p = max(abs(a["clean_target_share"] - b["clean_target_share"])
              for a, b in pairs)
worst_d = max(abs(a["ancestor_distance"] - b["ancestor_distance"])
              for a, b in pairs)
ax.text(0.04, 0.94, f"max Δp = {worst_p:.4f}\nmax Δdistance = {worst_d} token",
        transform=ax.transAxes, va="top", fontsize=8)

ax = axes[2]
spec = STAGE_B["control_specificity"]
control = spec["1"]["per_layer"][str(LAYER)]
assert all(spec[d]["per_layer"][str(LAYER)][key] == control[key]
           for d in ("1", "2") for key in ("control_moved", "n_control"))
control_moved, control_n = control["control_moved"], control["n_control"]
values = [PRIMARY["hits"]["1"] / PRIMARY["n"]["1"],
          PRIMARY["hits"]["2"] / PRIMARY["n"]["2"],
          control_moved / control_n]
labels = ["one update", "two updates", "controls"]
bars = ax.bar(labels, values, color=[COLOR[1], COLOR[2], "#9a9a9a"])
for bar, hit, total in zip(bars,
                           [PRIMARY["hits"]["1"], PRIMARY["hits"]["2"], control_moved],
                           [PRIMARY["n"]["1"], PRIMARY["n"]["2"], control_n]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.035,
            f"{hit}/{total}" + (" per arm" if total == control_n else ""),
            ha="center", fontsize=8.5, fontweight="600")
ax.set_ylim(0, 1.12)
ax.set_ylabel("share landing on implied digit")
ax.set_title("c. the matched intervention", loc="left")

fig.tight_layout()
plt.show()
''')

md(r"""
**Figure 3. Confirmatory matched test.** The matching procedure saw clean
forward passes but no patch outcomes. The 24 selected pairs differ by at most
0.0007 in clean `p(target)` and one token in ancestor distance. At layer 13, all
24 one-update interventions and no two-update interventions land on the implied
digit; matched non-ancestor, null, and surface controls move 0/192 times. The
one-sided exact paired test is `p = 2⁻²⁴ = 5.96e-8`. This confirms the contrast
in the high-confidence overlap region. It does not reselect the layer or estimate
a general population rate.
""")

md(r"""
## 4. The transplant carries the stated result, with some literal copying

The one-update ancestor result is a mixture. Across E3, the implied digit is top
in 267/315 cases, the raw donor digit in 46/315, and the clean digit in 0/315.
Recipient-side transformation dominates, but the ancestor edit is not pure
symbolic propagation. The written last-intermediate edit is cleaner: implied
288/288 and raw 0/288.

A small donor-split pilot asks which part of the donor line supplies the effect.
""")

reuse(
    11,
    "which half of the donor line does the work",
    replacements=(
        ("in claim 1", "in the one-update result"),
        ("it is the weakest ", "it is a supporting "),
        ("leg of the three claims", "pilot, not a headline result"),
    ),
)

md(r"""
## 5. Controls define the claim boundary

The failed or limited controls belong beside the positive result. They prevent a
semantic boundary from becoming a claim that the model implements the generated
DAG or that written text overwrites latent state.
""")

code(r'''
spec = STAGE_B["control_specificity"]
control = spec["1"]["per_layer"][str(LAYER)]
assert all(spec[d]["per_layer"][str(LAYER)][key] == control[key]
           for d in ("1", "2") for key in ("control_moved", "n_control"))
control_moved, control_n = control["control_moved"], control["n_control"]

cross = [arm(f"v3_distinct/cross_seed{seed}.json")["gates"]["cross_item_donor"]
         for seed in range(4)]

omit = []
for depth in (2, 3):
    payload = arm(f"written_vs_omitted/depth{depth}_chain.json")
    clean = payload["gates"]["clean_answer"]
    shares = [item["clean_probs"][item["target_value"]] for item in payload["items"]]
    omit.append(f"d{depth}: {clean['n_unique_correct']}/{clean['n_items']}, "
                f"median p={pd.Series(shares).median():.3f}")

gap = E3["by_distance_paired"]
near, far = gap["per_gap"][0], gap["per_gap"][-1]
rows = [
    {
        "check": "matched quiet controls",
        "observation": f"{control_moved}/{control_n} moved per arm at layer {LAYER}",
        "reading": "the matched intervention is not a generic token-position effect",
    },
    {
        "check": "foreign-item donor",
        "observation": f"specificity passed {sum(g['passes'] for g in cross)}/4 seeds",
        "reading": "portable-state selectivity remains unresolved",
    },
    {
        "check": "omit written path value",
        "observation": "; ".join(omit),
        "reading": "clean competence collapses, so overwrite is not identified",
    },
    {
        "check": "same items at three gaps",
        "observation": (f"implied {near['implied_top_unique']}/{near['n_complete_case']} → "
                        f"{far['implied_top_unique']}/{far['n_complete_case']}; "
                        f"raw {near['raw_top_unique']} → {far['raw_top_unique']}"),
        "reading": "placement and clean confidence move together",
    },
]
table(pd.DataFrame(rows), "Table 3 · controls that set the paper's claim boundary",
      note=("The first row supports the intervention. The other rows limit its "
            "interpretation. The omission pilot has five items per arm; the gap "
            "analysis is exploratory and clustered by item."))
''')

md(r"""
## 6. Relation to prior work

Interchange interventions already test whether neural states play the causal
role of variables in a high-level program ([Geiger et al.,
2021](https://arxiv.org/abs/2106.02997)). Prior work also studies causal
arithmetic chain-of-thought ([Tan,
2023](https://aclanthology.org/2023.blackboxnlp-1.12/)), task-grounded
mechanistic intervention ([Brinkmann et al.,
2024](https://aclanthology.org/2024.findings-acl.242/)), and hidden-state
transplantation ([Patchscopes](https://openreview.net/forum?id=5uwBzcn885)).

The closest precedent is [Shih, Winnicki, and Darve
2026](https://arxiv.org/abs/2606.29522). They compare an unmodified pretrained
base with matched final-answer-only and running-state LoRA models. Only the
running-state targets include intermediate states, and that model develops a
causal register that persists across later updates. Their conflicting-
continuation control already separates computation from copying. Our
clean/implied/raw assay adds a per-item arithmetic outcome rather than a new kind
of control.

[Kudo et al. 2026](https://aclanthology.org/2026.findings-eacl.59/) show that
models compute sub-answers during CoT generation. They ask when a value first
becomes available; we ask what controls the answer after the value is written.
The results address different stages of the trace.

[Garcia 2026](https://arxiv.org/abs/2605.10799) shows that corruption sensitivity
can track explicit answer placement. Our token-distance bands and same-trace
comparison rule out placement alone. They do not separate written-step count
from patch-site role or local context.

The contribution is therefore the arithmetic clean/implied/raw assay and the
controlled empirical boundary it reveals. It is not the first activation
intervention on chain-of-thought, the first graph-grounded patching study, or the
first computation-versus-copy control. The persistence difference from Shih is
consistent with a supervision hypothesis, but the papers differ in model, task,
intervention, and outcome. The comparison does not identify the cause.
""")

md(r"""
## 7. Paper conclusion and open scope

The transplant usually has the expected causal role in the recipient's next
arithmetic update. Through another explicitly written update, it still perturbs
the answer distribution but never installs the predicted counterfactual answer
in the measured sites. Exact semantic control and distributional influence are
therefore different estimands.

The result does not show that the model represents the generated DAG, that
transformers cannot maintain latent state, or that a written value overwrites a
latent one. The main limits are one checkpoint, one operation family, one
notation, teacher-forced traces, a layer inherited from discovery, a failed
cross-item specificity control, and a depth manipulation that also changes
operation, binding, context, and site role.

For the workshop paper, the experiments are complete. The next main-track fork
is a clean-only screen for a model and format that remain competent when a path
intermediate is unwritten. If none clears that gate, the sharper experiment is a
same-task comparison of final-answer-only and running-state supervision.

### Provenance

- Full experimental narrative: [notebook 15](15_dag_paper_story.ipynb)
- Paper strategy: [`PAPER_STRATEGY_DAG.md`](../PAPER_STRATEGY_DAG.md)
- Claim and literature boundary:
  [`docs/research/2026-08-16-dag-literature-and-claim-boundary.md`](../docs/research/2026-08-16-dag-literature-and-claim-boundary.md)
- Main analysis: [`results/dag_patching/e3_ladder/ANALYSIS.json`](../results/dag_patching/e3_ladder/ANALYSIS.json)
- Pre-registered matched analysis:
  [`results/dag_patching/e2_stage_b/ANALYSIS.json`](../results/dag_patching/e2_stage_b/ANALYSIS.json)
""")

# The shorter story must not quietly reintroduce the causal overclaims it replaces.
prose = "\n".join("".join(cell["source"]) for cell in CELLS
                  if cell["cell_type"] == "markdown")
for forbidden in ("step count alone", "reaches exactly one step",
                  "property of the step", "nothing else does"):
    assert forbidden not in prose
assert sum(cell["cell_type"] == "code" for cell in CELLS) == 7

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    target = REPO / "notebooks" / "16_dag_workshop_story.ipynb"
    for index, cell in enumerate(CELLS):
        cell["id"] = f"cell-{index:02d}"
    target.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"wrote {target} with {len(CELLS)} cells "
          f"({sum(c['cell_type'] == 'code' for c in CELLS)} code)")
