"""Build notebooks/17_rmd_experiment_ledger.ipynb. Edit this, never the .ipynb.

The RMD-side counterpart to `build_15_dag_paper_story.py`: the long-form record
of the closure era, and the notebook to open when you want to run your own
analysis rather than read a story. Notebook 14 is the storyboard and quotes a
deliberately small subset of these numbers; this one loads everything and hands
you the objects.

    uv run python notebooks/build_17_rmd_experiment_ledger.py
    uv run python notebooks/execute_notebook.py \
        notebooks/17_rmd_experiment_ledger.ipynb notebooks

Every number is read out of a committed artifact under `results/`. The
presentation helpers are not redefined here -- they are lifted out of
`build_14_rmd_workshop_story.py` at build time, so the two notebooks cannot
drift into different table styling, and a rename over there fails this build.
"""
from __future__ import annotations

import json
from pathlib import Path

from build_14_rmd_workshop_story import CELLS as STORY_CELLS

REPO = Path(__file__).resolve().parents[1]

CELLS: list[dict] = []


def md(source: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": source.strip("\n").splitlines(keepends=True)})


def code(source: str) -> None:
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": source.strip("\n").splitlines(keepends=True)})


def presentation() -> str:
    """The styling block from notebook 14, lifted rather than copied.

    Two notebooks in one paper should not diverge in table rules because
    somebody edited one of them. The markers below are asserted, so if 14's
    setup cell is restructured this build fails instead of silently emitting a
    notebook with stale styling.
    """
    setup = next("".join(cell["source"]) for cell in STORY_CELLS
                 if cell["cell_type"] == "code")
    start = setup.index("# Presentation only. Nothing below this line touches a number.")
    start = setup.rindex("# ---", 0, start)
    end = setup.index('\n\nCAP = BUDGET["cap_accounting"]')
    block = setup[start:end]
    assert "def table(" in block and "_RULES" in block, "styling block moved"
    assert "artifact(" not in block, "the lifted block must not load anything"
    return block


# ------------------------------------------------------------------ 0. what
md(r"""
# RMD experiment ledger

**The long-form record, and a working surface.** Notebook
[14](14_rmd_workshop_story.ipynb) is the storyboard: it makes one argument and
quotes the handful of numbers that argument needs. This notebook is the other
half -- every closure-era artifact loaded, including the seven controls that
appear in no other notebook, with the reasons a number here may not be
comparable to a number there stated up front rather than left to be discovered.

*Status: built 2026-08-22. Reads committed artifacts only; runs no model, fits
nothing, and takes seconds. The outer refit sweep is registered and pending, so
every interval below -- like every interval in 14 -- resamples prompts with
folds, layer and coefficients frozen at `seed=42`.*

## Two things to settle before reading any number

**1. There is more than one population, and the difference is not small.** The
closure experiments in sections 3--5 report on `cap_free_valid_plurality`, which
drops every prompt with a capped sibling -- 22%, 21% and 18% of prompts. The
paper's primary estimand is `full_population`: correctness at budget `B` over
all 500 prompts, with unparsed scored 0. Section 1 puts the two side by side.
The headline population is the more favourable one on all three models, so a
control that clears its bar on `cap_free_valid_plurality` has cleared a
slightly easier bar than the claim it is defending.

**2. `aurc` means opposite things in different files.** Some artifacts integrate
*risk* against coverage (lower is better, a negative delta is a gain); the
`wave1` E1 artifact integrates *accuracy* (higher is better). Both store the
number under the key `"aurc"`. Section 2 is the registry; nothing in this
notebook combines two conventions in one column.

## What this covers

| era | experiments |
|:--|:--|
| closure controls | closest baselines (1a/1b), Orgad agreement, peer difficulty |
| negative / exploratory | allocation precheck, application alignment |
| supervision | label-efficiency trio |
| in notebook 14 | budget outcomes, peer cost ladder, last-token probe |

Sections 3--8 are the material that has no other notebook, except the first of
the three label-efficiency runs, which notebook 13 covers. The three experiments
14 is built on are loaded here too, so an analysis can join across them, but
their story is told there and not repeated.
""")


# ----------------------------------------------------------------- 1. setup
code(r'''
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display

sys.path.insert(0, str(Path.cwd()))
import _viz_utils as vu

ROOT = vu.repo_root()
MODELS = ["qwen", "deepseek", "deepseek_llama"]
NICE = {"qwen": "Qwen-1.5B", "deepseek": "DeepSeek-7B", "deepseek_llama": "Llama-8B"}

# Three generations of this harness wrote three label vocabularies for the same
# three checkpoints -- `deepseek`, `deepseek_qwen` and `DeepSeek-Qwen` are one
# model. Joining artifacts on the raw string drops rows without complaining, so
# every table below goes through nice().
ALIASES = {"deepseek_qwen": "deepseek", "DeepSeek-Qwen": "deepseek",
           "Qwen": "qwen", "Llama": "deepseek_llama"}


def nice(label):
    return NICE[ALIASES.get(label, label)]

# The registry is declared, not discovered. An experiment that loses its
# artifact fails here, and one that is added without a stated estimand and
# sign convention cannot be loaded at all.
REGISTRY = {
    "budget_outcomes": dict(
        path="budget_outcomes/budget_outcomes.json",
        estimand="prompt abstention", population="all four, side by side",
        sign="aurc: lower is better", home="notebook 14"),
    "peer_cost_ladder": dict(
        path="peer_cost_ladder/peer_cost_ladder_results.json",
        estimand="prompt abstention", population="full_population",
        sign="aurc: lower is better", home="notebook 14"),
    "last_token_probe": dict(
        path="last_token_probe/last_token_probe_results.json",
        estimand="trace verification", population="parseable, all_traces",
        sign="auroc: higher is better", home="notebook 14"),
    "closest_baselines": dict(
        path="closest_baselines/closest_baselines_results.json",
        estimand="prompt abstention", population="cap_free_valid_plurality",
        sign="aurc: lower is better", home="here, section 3"),
    "orgad_agreement_control": dict(
        path="orgad_agreement_control/orgad_agreement_control_results.json",
        estimand="prompt abstention", population="cap_free_valid_plurality",
        sign="aurc: lower is better", home="here, section 4"),
    "peer_difficulty_control": dict(
        path="peer_difficulty_control/peer_difficulty_control_results.json",
        estimand="prompt abstention", population="cap_free_valid_plurality",
        sign="aurc: lower is better", home="here, section 5"),
    "allocation_precheck": dict(
        path="allocation_precheck/allocation_precheck_results.json",
        estimand="gain from more samples", population="cap_free_valid_plurality",
        sign="R^2 / rho: higher is better", home="here, section 6"),
    "application_alignment": dict(
        path="application_alignment/math500_application_alignment_results.json",
        estimand="within-prompt selection", population="not stated in artifact",
        sign="auc: higher is better", home="here, section 7"),
    "label_efficiency": dict(
        path="label_efficiency/label_efficiency_results.json",
        estimand="prompt abstention", population="cap_free_valid_plurality",
        sign="aurc: lower is better", home="here, section 8"),
    "label_efficiency_supervision_ladder": dict(
        path="label_efficiency_supervision_ladder/label_efficiency_results.json",
        estimand="prompt abstention", population="cap_free_valid_plurality",
        sign="aurc: lower is better", home="here, section 8"),
    "label_efficiency_token_pooling": dict(
        path="label_efficiency_token_pooling/label_efficiency_results.json",
        estimand="prompt abstention", population="cap_free_valid_plurality",
        sign="aurc: lower is better", home="here, section 8"),
    "wave1_qwen": dict(
        path="qwen_bestofn_full/math500/math500_wave1_results.json",
        estimand="prompt abstention", population="complete case (E1 censoring)",
        sign="aurc: HIGHER is better", home="notebook 14 section 4a"),
    "wave1_deepseek": dict(
        path="deepseek_bestofn_full/math500/math500_wave1_results.json",
        estimand="prompt abstention", population="complete case (E1 censoring)",
        sign="aurc: HIGHER is better", home="notebook 14 section 4a"),
}

A = {}
missing = []
for name, meta in REGISTRY.items():
    path = ROOT / "results" / meta["path"]
    if path.exists():
        A[name] = json.loads(path.read_text(encoding="utf-8"))
    else:
        missing.append(name)

assert "budget_outcomes" in A, "the primary artifact is gone; nothing below is safe"
print(f"loaded {len(A)} of {len(REGISTRY)} artifacts"
      + (f"; missing: {', '.join(missing)}" if missing else ""))

BUDGET, LADDER, PROBE = A["budget_outcomes"], A["peer_cost_ladder"], A["last_token_probe"]
CAP = BUDGET["cap_accounting"]
print("label vocabularies normalised: "
      + ", ".join(f"{k} -> {v}" for k, v in ALIASES.items()))
REFIT_PATH = ROOT / "results/refit_stability/refit_stability_results.json"
print(f"seed 42 everywhere; refit sweep "
      f"{'loaded' if REFIT_PATH.exists() else 'PENDING -- section 10'}")
''' + presentation() + r'''

for label in MODELS:
    print(f"{nice(label):14s} cap {CAP[label]['max_new_tokens']:>5}  "
          f"layer {CAP[label]['layer']:>3}  "
          f"{CAP[label]['n_capped']:>5} of {CAP[label]['n_traces']:>5} traces capped")
''')


# ------------------------------------------------------ 2. the population map
md(r"""
## 1. The population map

Every filter here is defensible on its own terms, and every one of them is also
a choice that moves the number. `full_population` is the paper's primary
estimand because it is the only one that does not condition on the budget under
evaluation: at a stated budget, an unfinished trace is the protocol's outcome,
not a missing observation.

Read the last column as *how much more favourable the headline filter is*. It is
positive on all three models, which is why the closure controls in sections
3--5 -- all of which report on `cap_free_valid_plurality` -- should be read as
defending the claim on the easier population, not the primary one.
""")

code(r'''
rows = []
for row in BUDGET["populations"]:
    rows.append({
        "model": nice(row["model"]),
        "population": row["population"],
        "definition": row["definition"],
        "n": row["n_prompts"],
        "kept": row["retained"],
        "base acc": row["base_accuracy"],
        "B1 - B0": row["delta_estimate"],
        "95% CI": f"[{row['delta_ci_low']:+.4f}, {row['delta_ci_high']:+.4f}]",
        "p": row["delta_p_two_sided"],
    })
POPS = pd.DataFrame(rows)

primary = POPS[POPS["population"] == "full_population"].set_index("model")["B1 - B0"]
headline = POPS[POPS["population"] == "cap_free_valid_plurality"].set_index("model")["B1 - B0"]
# Both are negative and lower is better, so the headline overstates when it is
# the more negative of the two. As a share of the headline effect -- the same
# formula notebook 14 section 2 prints, so the two notebooks cannot disagree.
OVERSTATEMENT = ((headline - primary) / headline).rename("headline vs primary")

POPS["headline overstates by"] = [
    f"{OVERSTATEMENT[r['model']]:+.0%}" if r["population"] == "cap_free_valid_plurality"
    else "" for _, r in POPS.iterrows()]

table(POPS, "Table 1 &middot; the same contrast on four populations",
      note=("AURC, <b>lower is better</b>, so a more negative <code>B1 - B0</code> is a "
            "bigger gain. <code>full_population</code> is the paper's primary estimand: "
            "all 500 prompts, unparsed scored 0, nothing conditioned on the budget. "
            "<code>cap_free_valid_plurality</code> is the population the closure controls "
            "in sections 3-5 ran on, and it is the more favourable one on every model."),
      highlight=(POPS["population"] == "full_population"),
      **{"kept": "{:.3f}", "base acc": "{:.3f}", "B1 - B0": "{:+.4f}", "p": "{:.3f}"})
''')


# ---------------------------------------------------------- 3. the sign map
md(r"""
## 2. The sign map

This exists because two artifacts in this repository store opposite quantities
under the key `"aurc"`, and one of them is quoted in notebook 14. Getting it
backwards inverts a conclusion without producing anything that looks wrong.

The rule: **selective-risk** curves integrate the error rate against coverage,
so lower is better and a negative paired delta is a gain. The `wave1` E1
artifact integrates **accuracy** against coverage, so higher is better and
`rmd_tail_q20` at 0.83 beats `entropy` at 0.66. Nothing in this notebook puts
the two in one column, and any analysis you build on top should not either.
""")

code(r'''
SIGNS = pd.DataFrame([
    {"artifact": name,
     "estimand": meta["estimand"],
     "population": meta["population"],
     "convention": meta["sign"],
     "written up in": meta["home"],
     "loaded": "yes" if name in A else "MISSING"}
    for name, meta in REGISTRY.items()
])

inverted = SIGNS["convention"].str.contains("HIGHER")
assert inverted.sum() == 2, "the sign registry no longer matches the artifacts"

table(SIGNS, "Table 2 &middot; estimand, population and sign convention per artifact",
      note=("The two highlighted rows integrate accuracy rather than risk, so for those "
            "and only those a <b>larger</b> <code>aurc</code> is better. They also carry "
            "E1's complete-case censoring -- unparsed and cap-hit traces excluded -- "
            "which is the filter section 1 argues against as a headline. Notebook 14 "
            "section 4a uses them for a region-vs-region contrast, where the filter "
            "applies identically to both arms."),
      highlight=inverted)
''')


# ------------------------------------------------- 4. closest baselines (1a/1b)
md(r"""
## 3. The two closest cheap baselines

The question a reviewer asks first: if you already have eight samples, you can
compute the entropy of their answer histogram for free. Does a hidden-state
score add anything over *that*, rather than over the weaker output features in
`B0`? And separately -- since the feature is measured on the trace tail -- does
the tail actually beat the whole-trace mean it is a restriction of?

Both were registered with a stop rule before the run. `H` is
`neg_answer_entropy`, minus the Shannon entropy of the normalized exact-answer
histogram over parseable siblings. `rmd_full` is the whole-trace mean of the
same per-token distance.

**1a passed and 1b did not, and 1b is the more interesting one.** The tail beats
the full trace on Qwen and ties or loses on the other two. So the localization
that section 4a of notebook 14 defends -- tail over high-entropy region -- is a
real ordering, but "tail over full trace" is not established on three models.
The honest statement is that the tail is never *worse* than the whole trace, not
that it is better.
""")

code(r'''
CB = A["closest_baselines"]
POP = "cap_free_valid_plurality"


def ci(entry, digits=4):
    return (f"{entry['point_estimate']:+.{digits}f} "
            f"[{entry['ci_low']:+.{digits}f}, {entry['ci_high']:+.{digits}f}]")


rows = []
for model in CB["models"]:
    pop = model["populations"][POP]
    deltas, readouts = pop["paired_deltas_aurc"], pop["readouts"]
    rows.append({
        "model": nice(model["label"]),
        "n": pop["n_prompts"],
        "B0": readouts["B0"]["aurc"],
        "B1": readouts["B1"]["aurc"],
        "B0+H": readouts["B0_plus_H"]["aurc"],
        "B0+rmd_full": readouts["B0_plus_rmd_full"]["aurc"],
        "1a: tail over B0+H": ci(deltas["rmd_tail_over_B0_plus_H"]),
        "1b: tail over full": ci(deltas["rmd_tail_over_rmd_full"]),
    })
CLOSEST = pd.DataFrame(rows)

rules = CB["stop_rules"]
verdict = (f"1a triggered: {rules['1a']['triggered']}  |  "
           f"1b tail wins on {rules['1b']['n_tail_wins']} of {rules['1b']['n_models']} "
           f"({', '.join(f'{k}={v}' for k, v in rules['1b']['branch_by_model'].items())})")

table(CLOSEST, "Table 3 &middot; against the answer histogram, and against the whole trace",
      note=("AURC, lower is better; a negative delta favours the tail. "
            f"<b>{verdict}</b>. 1a's stop rule was to withdraw the "
            "&ldquo;beyond self-consistency&rdquo; claim if two or more models had an "
            "interval overlapping zero &mdash; none did. 1b was pre-registered as "
            "terminal either way: no region or percentile sweep follows it. "
            f"Population <code>{POP}</code>; see Table 1 for what that costs."),
      **{c: "{:.4f}" for c in ("B0", "B1", "B0+H", "B0+rmd_full")})

print(verdict)
print("\nRedundancy of H with the vote it is computed from (Pearson / Spearman):")
for model in CB["models"]:
    red = model["populations"][POP]["redundancy"]["neg_answer_entropy_vs_vote_agreement"]
    print(f"  {nice(model['label']):14s} {red['pearson']:.3f} / {red['spearman']:.3f}")
print("H is nearly the same feature as vote_agreement, which is already in B0 --"
      "\nso 1a is a sharper test of redundancy than of an unseen baseline.")
''')


# ------------------------------------------------------- 5. Orgad agreement
md(r"""
## 4. Is the score a worse-instrumented vote?

The direct answer to Orgad et al. (arXiv:2410.02707): hidden states encode
resampling agreement, so a reviewer can reasonably read `rmd_tail_q20` as
`vote_agreement` measured badly. Three tests, in increasing severity.

**Inside a fixed level of agreement.** Agreement cannot separate anything within
its own stratum, so the unanimous stratum is where self-consistency has nothing
left to say. The geometry keeps most of its AUROC there on all three models.

**Substitution, both directions.** Swap the two features in `B0` and see which
one the readout misses. Replacing the vote with the geometry costs little;
adding the geometry to a voteless `B0` recovers a lot.

**Where it does not hold.** The `B1 - (rmd for vote)` column is the honest limit:
on Qwen the two are indistinguishable (p = 0.63). On that model the geometry can
largely *stand in for* the vote, which is a weaker claim than adding to it.
""")

code(r'''
ORGAD = A["orgad_agreement_control"]
rows = []
for entry in ORGAD:
    strata, deltas = entry["strata"], entry["paired_deltas_aurc"]
    rows.append({
        "model": nice(entry["label"]),
        "n": entry["n_prompts"],
        "rho(rmd, vote)": entry["redundancy"]["spearman"],
        "AUROC unanimous": strata["unanimous"]["auroc"]["point_estimate"],
        "n unanimous": strata["unanimous"]["n_prompts"],
        "rmd | vote": ci(entry["residual_auroc"]["geometry_given_agreement"], 3),
        "vote | rmd": ci(entry["residual_auroc"]["agreement_given_geometry"], 3),
        "B1 - B0": ci(deltas["B1_minus_B0"]),
        "B1 - (rmd for vote)": ci(deltas["B1_minus_B0_rmd_for_vote"]),
    })
AGREE = pd.DataFrame(rows)

table(AGREE, "Table 4 &middot; geometry against the vote it is scored beside",
      note=("The two residual columns are out-of-fold linear residuals: "
            "<code>rmd | vote</code> is what the geometry still ranks once the vote is "
            "projected out, and <code>vote | rmd</code> is the reverse. The asymmetry is "
            "the finding &mdash; the vote given the geometry sits at or below chance on "
            "two of three models. The last column is where the case is weakest: a "
            "geometry that can replace the vote has not thereby been shown to add to it."),
      **{"rho(rmd, vote)": "{:.3f}", "AUROC unanimous": "{:.3f}"})

print("Substitution detail, AURC (lower is better):")
for entry in ORGAD:
    d = entry["paired_deltas_aurc"]
    print(f"  {nice(entry['label']):14s} rmd added to a voteless B0: "
          f"{ci(d['B0_rmd_for_vote_minus_B0_voteless'])}  "
          f"p={d['B0_rmd_for_vote_minus_B0_voteless']['p_two_sided']:.3f}")
''')


# -------------------------------------------------- 6. peer difficulty control
md(r"""
## 5. The cross-model empirical difficulty control

The strongest available difficulty proxy: for each target, the other two models'
eight-sibling pass rates on the same prompt ids, entered into `B0` as two
features. Those pass rates reach AUROC 0.80--0.96 against the target's own
outcome -- a near-oracle. If `rmd_tail_q20` is difficulty in disguise, this is
where it should disappear.

`B0 + peer` is a **control, not a baseline**. Two other models' pass rates are
not available at decision time, so nothing here competes with the headline.

It survives, but not uniformly: under Holm correction across the three models,
only Llama-8B is significant at 0.05. The pre-registered stop rule asked
whether two or more models lost their interval, and one did.
""")

code(r'''
PEER = A["peer_difficulty_control"]
rows = []
for model in PEER["models"]:
    label = model["label"]
    pop = model["populations"][POP]
    deltas, holm = pop["paired_deltas"], PEER["holm"]["tests"][label]
    peers = model["peer_columns"]
    rows.append({
        "model": nice(label),
        "n": pop["n_prompts"],
        "peer AUROC (best)": max(pop["marginal_auroc"][c]["point_estimate"] for c in peers),
        "B1 - B0": ci(deltas["B1_minus_B0_aurc"]),
        "B1 - B0 | peer": ci(deltas["B1_minus_B0_given_peer_aurc"]),
        "p raw": holm["p_raw"],
        "p Holm": holm["p_holm"],
        "sig at .05": "yes" if holm["significant_at_0.05"] else "no",
        "headroom removed": pop["headroom_fraction_removed"]["B1_minus_B0_given_peer"],
    })
DIFFICULTY = pd.DataFrame(rows)

stop = PEER["stop_rule"]
table(DIFFICULTY, "Table 5 &middot; does the increment survive a near-oracle difficulty control",
      note=("AURC, lower is better. <code>B1 - B0 | peer</code> is the increment with both "
            "peer pass rates already in the baseline. <code>headroom removed</code> is the "
            "fraction of the gap to a per-prompt oracle that the increment closes. "
            f"Stop rule &mdash; withdraw if two or more models lose the interval &mdash; "
            f"<b>triggered: {stop['triggered']}</b> "
            f"({len(stop['models_with_interval_overlapping_zero'])} of {stop['n_models']} "
            "overlap zero). Holm is over the three models, so the single surviving "
            "significant result is Llama-8B; read this as the increment being attenuated "
            "by difficulty, not eliminated by it."),
      **{"peer AUROC (best)": "{:.3f}", "p raw": "{:.3f}", "p Holm": "{:.3f}",
         "headroom removed": "{:.3f}"})
''')


# ------------------------------------------------------ 7. allocation precheck
md(r"""
## 6. Allocation: a clean negative

A different question from every other rung. Not *is this prompt right*, but
*would buying more samples help* -- `g(p) = a(p,8) - a(p,1)`, the expected
plurality-vote gain from 1 to 8 samples over all `C(8,k)` sibling subsets.

Gain is not difficulty: a prompt solved 0/8 and one solved 8/8 both gain
nothing, and the measured `rho(pass rate, g)` is near zero rather than near -1,
so the question is genuinely separate. It is also where the geometry has nothing
to offer. Single-trace geometry does not beat a cross-fitted constant on two of
three models, the gate failed, and step 3 was never run.

This is in the paper as a scope sentence: the claim is abstention, not
allocation. It is worth keeping visible because it is the cheapest available
rebuttal to "why not use it to decide where to spend compute".
""")

code(r'''
ALLOC = A["allocation_precheck"]
rows = []
for model in ALLOC["models"]:
    label = model["label"]
    gate = ALLOC["gate"]["per_model"][label]
    gain = model["gain_summary"]
    rows.append({
        "model": nice(label),
        "n": model["n_prompts"],
        "mean g": gain["mean"],
        "share g = 0": gain["share_exactly_zero"],
        "rho(pass rate, g)": gain["spearman_pass_rate_vs_gain"],
        "geometry R^2": gate["geometry_r2_median"],
        "beats constant": "yes" if gate["geometry_beats_constant"] else "no",
        "adds over output": "yes" if gate["geometry_adds_over_output"] else "no",
        "passes": "yes" if gate["passes"] else "no",
    })
ALLOCATION = pd.DataFrame(rows)

gate = ALLOC["gate"]
table(ALLOCATION, "Table 6 &middot; can geometry predict the gain from more samples",
      note=("Medians over the eight choices of which trace you draw first &mdash; that "
            "is a random variable, and fixing it at <code>sample_id == 0</code> would "
            "hide its variance. <code>rho(pass rate, g)</code> near zero is what makes "
            "this a separate question from difficulty; were gain monotone in difficulty "
            f"it would sit near -1. Gate passed on {len(gate['models_passing'])} of "
            f"{gate['n_models']} models, so overall <b>passes: {gate['passes']}</b>. "
            f"Consequence as registered: {gate['consequence']}"),
      **{"mean g": "{:+.3f}", "share g = 0": "{:.3f}", "rho(pass rate, g)": "{:+.3f}",
         "geometry R^2": "{:+.4f}"})
''')


# ----------------------------------------------------- 8. application alignment
md(r"""
## 7. Application alignment (exploratory, weakest evidence here)

Twelve model-layer-method conditions, asking whether the within-prompt readouts
predict what a downstream application actually gets: top-1 selection gain over
random, and gain over majority vote. The correlations are high -- Spearman 0.96
between within-prompt AUC and top-1 gain -- but n = 12 conditions from 2 models,
and the artifact says so itself.

It earns a place in the ledger for one reason: it is the only artifact that
connects a readout to an application outcome, and it is the record of *why* the
project stopped reporting pooled AUC. The raw-distance rows have negative top-1
gain in every condition. Do not cite the correlations as evidence for anything;
cite the sign pattern.
""")

code(r'''
APP = A["application_alignment"]
ALIGN = pd.DataFrame(APP["conditions"])[[
    "model", "layer", "method", "within_prompt_pair_weighted", "prompt_centered_auc",
    "top1_gain_over_random", "top1_gap_to_majority", "selective_ausc"]]
ALIGN.columns = ["model", "layer", "method", "within AUC", "centered AUC",
                 "top-1 gain", "gap to majority", "selective AUSC"]

by_method = ALIGN.groupby("method")[["within AUC", "top-1 gain"]].mean().round(3)
print("mean by method:\n", by_method.to_string(), "\n")
print("raw conditions with positive top-1 gain: "
      f"{int((ALIGN[ALIGN['method'] == 'raw']['top-1 gain'] > 0).sum())} of "
      f"{int((ALIGN['method'] == 'raw').sum())}")

table(ALIGN, "Table 7 &middot; readouts against what an application gets",
      note=("<b>Exploratory.</b> " + APP["warning"] + " Every <code>raw</code> row has "
            "negative top-1 gain over random selection and negative gap to majority "
            "vote &mdash; picking by raw distance is worse than picking at random. That "
            "sign pattern, not the Spearman values, is what this artifact establishes."),
      highlight=(ALIGN["method"] == "raw"),
      **{c: "{:+.3f}" for c in ("within AUC", "centered AUC", "top-1 gain",
                                "gap to majority", "selective AUSC")})
''')


# ------------------------------------------------------- 9. label efficiency
md(r"""
## 8. The label-efficiency trio

Three runs of the same harness, and they are **not interchangeable**. Notebook
[13](13_deepconf_null_and_label_efficiency.ipynb) covers the first; the two
follow-ups have appeared in no notebook until now.

| run | budgets | extra arms | what it isolates |
|:--|:--|:--|:--|
| `label_efficiency` | 25--400 | -- | the original curve and crossing point |
| `label_efficiency_token_pooling` | 25--100 | `probe_token_tail_q20` | pooling order, supervision held fixed |
| `label_efficiency_supervision_ladder` | 25--100 | `+ qmd_tail_q20` | supervision, pooling *and* decision function held fixed |

The ladder is a strict superset of the pooling run, so if you only read one, read
the ladder.

**The comparability trap.** The evaluation set is the complement of the *largest*
budget in that run, and the runs have different largest budgets -- 400 versus
100. So the original scores about 80 prompts and the follow-ups score about 320.
Same rule, different eval sets: crossing points are not transferable between
rows of the table below, and the original's Qwen crossing at ~226 labels simply
lies outside the range the follow-ups tested, which is why the same model reads
`no` there.

**What the ladder adds.** `qmd_tail_q20` is RMD's own quadratic with the
unconditional background swapped for an incorrect-trace Gaussian, so
`rmd - qmd` is the gap left when *only* supervision differs. On Qwen it is
positive at 25--50 labels and gone by 100. That is the honest version of the
label-efficiency claim: it is a small-sample effect of the one-class fit, not a
property of the geometry.
""")

code(r'''
LE_RUNS = {
    "label_efficiency": "original (13)",
    "label_efficiency_token_pooling": "pooling-matched",
    "label_efficiency_supervision_ladder": "supervision ladder",
}

rows = []
for name, run_label in LE_RUNS.items():
    if name not in A:
        continue
    run = A[name]
    for model in run["models"]:
        crossing = model["crossing"]
        first = model["curves"][0]
        rows.append({
            "run": run_label,
            "model": nice(model["label"]),
            "budgets": f"{min(run['budgets'])}-{max(run['budgets'])}",
            "eval n": first["n_eval"]["median"],
            "crossed": "yes" if crossing["crossed"] else "no",
            "crossing budget": crossing["budget"] if crossing["budget"] else float("nan"),
            "bracket": str(crossing.get("bracket", "--")),
        })
LABELS = pd.DataFrame(rows)

table(LABELS, "Table 8 &middot; where the supervised probe overtakes one-class geometry",
      note=("&ldquo;Crossed&rdquo; means the probe overtakes <code>rmd_tail_q20</code> "
            "inside the budget range that run tested &mdash; so a <code>no</code> on a "
            "run that stops at 100 labels is not evidence of no crossing, only of none "
            "below 100. <b>Eval sets differ by run</b> (complement of that run's largest "
            "budget), so crossing budgets are comparable down a run and not across runs."),
      **{"eval n": "{:.0f}", "crossing budget": "{:.0f}"})

LADDER_RUN = A.get("label_efficiency_supervision_ladder")
if LADDER_RUN:
    rows = []
    for model in LADDER_RUN["models"]:
        for curve in model["curves"]:
            row = {"model": nice(model["label"]), "labels": curve["budget"]}
            for key, col in (("feature_auroc_delta", "rmd - probe"),
                             ("feature_auroc_delta_token", "rmd - token probe"),
                             ("feature_auroc_delta_qmd", "rmd - qmd")):
                entry = curve.get(key)
                row[col] = entry["median"] if entry else float("nan")
            rows.append(row)
    QMD = pd.DataFrame(rows)
    table(QMD, "Table 9 &middot; the supervision gap with pooling and decision function held fixed",
          note=("Median over 10 label draws, feature AUROC. Left to right the arms get "
                "closer to <code>rmd_tail_q20</code>: <code>probe</code> differs in "
                "supervision, pooling and decision function; <code>token probe</code> "
                "matches the pooling order; <code>qmd</code> matches the decision "
                "function too, so <code>rmd - qmd</code> is supervision alone. It shrinks "
                "toward zero by 100 labels &mdash; the advantage is a small-sample "
                "property of the one-class fit."),
          **{c: "{:+.3f}" for c in ("rmd - probe", "rmd - token probe", "rmd - qmd")})
''')


# -------------------------------------------------------- 10. working surface
md(r"""
## 9. Working surface

Everything above is loaded and named. `A` is the artifact dict keyed by the
registry names in Table 2; the frames built along the way (`POPS`, `CLOSEST`,
`AGREE`, `DIFFICULTY`, `ALLOCATION`, `ALIGN`, `LABELS`, `QMD`) are still in
scope. The cell below adds the two things that are awkward to assemble by hand:

- `REPLICATES` -- the per-replicate label-efficiency rows from all three runs
  concatenated with a `run` column, which is the only tidy long-format table
  in the closure era. The columns differ by run -- the ladder has arms the
  original does not -- so the missing cells are real, not a join failure.
- `OOF` -- a loader for the per-trace out-of-fold scores. **Rows are per (trace,
  layer)**; forget to select a layer and the geometry columns silently average
  over the sweep while the output-side columns reproduce exactly, which is a
  failure that leaves no trace in the numbers. The loader will not let you skip
  the argument.

If you take an analysis further than this notebook, the two things to carry with
you are Table 1 (which population) and Table 2 (which sign).
""")

code(r'''
frames = []
for name, run_label in LE_RUNS.items():
    path = ROOT / "results" / name / "label_efficiency_replicates.csv"
    if path.exists():
        frame = pd.read_csv(path)
        frame.insert(0, "run", run_label)
        frames.append(frame)
REPLICATES = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
print(f"REPLICATES: {len(REPLICATES)} rows x {REPLICATES.shape[1]} cols, "
      f"runs = {sorted(REPLICATES['run'].unique())}")


def OOF(label, layer):
    """Per-trace out-of-fold scores for one model at ONE layer.

    The layer is required on purpose: the csv holds a sweep, and a groupby that
    forgets it averages the geometry columns while length, entropy and logprob
    reproduce exactly.
    """
    path = ROOT / f"results/{label}_bestofn_full/math500/math500_prompt_decomposition_oof.csv"
    frame = pd.read_csv(path)
    available = sorted(frame["layer"].unique())
    if layer not in available:
        raise ValueError(f"{label} has layers {available}, not {layer}")
    return frame[frame["layer"] == layer].reset_index(drop=True)


print("\nOOF(label, layer) -> per-trace scores. Layers available:")
for label in MODELS:
    path = ROOT / f"results/{label}_bestofn_full/math500/math500_prompt_decomposition_oof.csv"
    if path.exists():
        head = pd.read_csv(path, usecols=["layer"])
        print(f"  {label:14s} {[int(x) for x in sorted(head['layer'].unique())]}  "
              f"(paper uses {CAP[label]['layer']})")

print("\nIn scope: A, REGISTRY, REPLICATES, OOF(), and the frames "
      "POPS CLOSEST AGREE DIFFICULTY ALLOCATION ALIGN LABELS QMD")
''')


# ------------------------------------------------------------- 11. boundaries
md(r"""
## 10. What is not in here

**The refit gate.** Every interval in this notebook resamples prompts with the
fitting path frozen at `seed=42` -- folds, layer, PCA basis and coefficients all
held fixed. The outer refit that would carry the fitting path's own uncertainty
is registered in `EXPERIMENT_LOG.md` (2026-08-22) and has not run. Until it
does, a control that clears its stop rule has cleared it conditional on one
prompt partition. `controls/refit_stability.py` is the harness.

**The pre-closure notebooks.** [11](11_prompt_geometry_core_experiments.ipynb)
is the within-prompt record and is Qwen-only, while notebook 14 claims three
models. [12](12_wave1_abstention.ipynb) states its headline on a population it
does not name; Table 1 above is what that number should be read against.
[13](13_deepconf_null_and_label_efficiency.ipynb) holds the DeepConf null and
the first label-efficiency curve. None of the three has been rewritten against
the closure-era definitions.

**Mechanism.** Nothing here isolates *why* hidden-state scores encode prompt
difficulty. Section 5 shows the increment is attenuated but not eliminated by a
near-oracle difficulty control; that is a bound on the confound, not an account
of it.

### Provenance

- Experiment ledger and protocol: [`EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md)
- Paper strategy: [`PAPER_STRATEGY_RMD.md`](../PAPER_STRATEGY_RMD.md)
- The storyboard this record backs: [notebook 14](14_rmd_workshop_story.ipynb)
- Per-experiment reports: `results/<name>/<name>_report.md` for every row of Table 2
""")


assert sum(cell["cell_type"] == "code" for cell in CELLS) == 10
assert sum(cell["cell_type"] == "markdown" for cell in CELLS) == 11

# The ledger's job is to state the two hazards, not to assume the reader knows
# them. If a rewrite drops either one the notebook stops being safe to reuse.
PROSE = "\n".join("".join(cell["source"]) for cell in CELLS
                  if cell["cell_type"] == "markdown")
REQUIRED = (
    "full_population",              # the primary estimand is named
    "cap_free_valid_plurality",     # so is the one the controls actually ran on
    "opposite quantities",          # the aurc sign collision is stated
    "control, not a baseline",      # the peer columns are not a competitor
    "refit",                        # the open gate is not quietly dropped
)
for required in REQUIRED:
    assert required in PROSE, f"the ledger stopped stating: {required!r}"

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
    target = REPO / "notebooks" / "17_rmd_experiment_ledger.ipynb"
    for index, cell in enumerate(CELLS):
        cell["id"] = f"cell-{index:02d}"
    target.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"wrote {target} with {len(CELLS)} cells "
          f"({sum(c['cell_type'] == 'code' for c in CELLS)} code)")
