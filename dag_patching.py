"""Five-item residual-stream patching feasibility check on ground-truth DAGs.

Prototype 1 of the causal-DAG fidelity direction. For each item we run the clean
trace, run each donor trace, then re-run the clean trace with the donor's
residual state written in at the edited token positions. We read the model's
ten-way digit distribution at the position that predicts the target's result.

The question this script answers is *not* "does the model have a causal graph".
It is the prior question: **is this intervention selective enough to support that
experiment at all**. It therefore reports three separate verdicts — positive,
scientific negative, and invalid test — and never reports "no causal graph" when
the intervention itself failed.

Layer indexing note: unlike ``collect_data.py``, which reads
``outputs.hidden_states[i]``, this script both reads and writes at the output of
``model.model.layers[j]``. Using one mechanism for both directions makes the
off-by-one between the two conventions impossible rather than merely tested.
``hidden_states[n_layers]`` is also post-final-norm, so it is not a valid patch
site at all.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from dag_tasks import (
    CHECKPOINTS,
    DEFAULT_GENERATOR,
    DONOR_CONDITIONS,
    GENERATORS,
    OMIT_MODES,
    generate_items,
)

LAYER_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
DIGITS = tuple(str(d) for d in range(10))

# Surface-control policies. See ``_surface_gate`` for why there are two.
GATE_POLICIES = ("v1_two_sided", "v2_one_sided")
ACTIVE_GATE_POLICY = "v2_one_sided"

# The absolute-effect floor's fallback threshold, for reports that stored no
# digit distribution. The question the floor asks is whether the patched readout
# still puts the clean answer on top; where `probs_patched` exists that is tested
# directly and this constant is unused.
#
# A half is *not* the argmax boundary, though this constant was first introduced
# claiming it was. A digit at 0.40 beats nine others averaging 0.067. For ten
# classes the tightest scalar-only sufficient condition is a share below 0.1. The
# share is exact in both tails -- at or above 0.5 the clean digit is necessarily
# still the argmax, below 0.1 it necessarily is not -- and can only err by
# over-calling movement in between. It is kept at the majority line because that
# is the boundary on the safe side: the fallback never misses a real move, it
# only admits some that did not happen. See ``_answer_moved_gate``.
ANSWER_MOVED_FALLBACK_MAX_CLEAN_SHARE = 0.5

# The verdict function's own version, which `gate_policy_version` is not: that
# names the surface-control policy alone. Adding `answer_moved` changed the
# verdict function while leaving the policy label untouched, so
# `paired_ladder/depth2_gap0.json` reads `v2_one_sided` / `positive` on disk
# while a rescore under that same label calls it a scientific negative. Reports
# carrying no version were scored by `v1_gap_only`.
VERDICT_VERSIONS = ("v1_gap_only", "v2_gap_and_floor")
VERDICT_VERSION = "v2_gap_and_floor"

# The layer-aggregation rule frozen for the next paired run. Every gate today
# aggregates independently with `any(layer)`, so each may clear at a different
# bin -- an arm-level positive with no single layer at which the patch was
# directional, quiet, and selective at once. `joint_layer` requires one such
# layer. It is computed and reported for the archived runs but does not decide
# their verdicts: applying it retroactively would be a third post-hoc policy
# move. See `_joint_layer_gate`.
PROSPECTIVE_LAYER_RULE = "joint_layer"


def layer_bins(n_layers: int, fractions=LAYER_FRACTIONS) -> list[int]:
    """Four relative-depth decoder-layer indices, fixed once per checkpoint.

    Frozen before the first run and reused for every checkpoint and prototype,
    per the pre-registered measurement conditions.
    """
    return sorted({max(0, round(fraction * n_layers) - 1) for fraction in fractions})


def decoder_layers(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError(f"{type(model).__name__} has no model.layers to hook")
    return layers


def _residual(output):
    return output[0] if isinstance(output, tuple) else output


def _rewrap(output, residual):
    if isinstance(output, tuple):
        return (residual,) + tuple(output[1:])
    return residual


def capture_states(model, token_ids, bins, positions):
    """Residual states at ``positions``, one tensor per layer bin, plus logits."""
    import torch

    store = {}
    handles = []
    for layer in bins:
        def hook(module, args, output, layer=layer):
            store[layer] = _residual(output)[:, positions, :].detach().clone()
        handles.append(decoder_layers(model)[layer].register_forward_hook(hook))
    try:
        with torch.no_grad():
            input_ids = torch.as_tensor([list(token_ids)], device=model.device)
            logits = model(input_ids, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()
    return store, logits


def run_patched(model, token_ids, layer, positions, donor):
    """Run ``token_ids`` with ``donor`` written into layer ``layer`` at ``positions``."""
    import torch

    def hook(module, args, output):
        residual = _residual(output).clone()
        residual[:, positions, :] = donor.to(residual.device, residual.dtype)
        return _rewrap(output, residual)

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            input_ids = torch.as_tensor([list(token_ids)], device=model.device)
            return model(input_ids, use_cache=False).logits
    finally:
        handle.remove()


def digit_readout(logits, read_position, digit_ids):
    """(probabilities over the ten digits, log-odds, total digit mass)."""
    import torch

    row = logits[0, read_position].detach().float()
    selected = row[list(digit_ids)]
    full = torch.softmax(row, dim=-1)
    return (
        torch.softmax(selected, dim=-1).cpu().numpy(),
        torch.log_softmax(selected, dim=-1).cpu().numpy(),
        float(full[list(digit_ids)].sum()),
    )


def digit_token_ids(tokenizer) -> list[int]:
    ids = []
    for digit in DIGITS:
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"digit {digit!r} is {len(encoded)} tokens, not 1")
        ids.append(encoded[0])
    if len(set(ids)) != 10:
        raise ValueError("digit token ids are not distinct")
    return ids


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def measure_item(model, item, bins, digit_ids) -> list[dict]:
    """One row per (layer bin, edit)."""
    _, clean_logits = capture_states(model, item.token_ids, bins, [item.read_position])
    probs, logodds, mass = digit_readout(clean_logits, item.read_position, digit_ids)

    rows = []
    for edit in item.edits:
        if max(edit.positions) >= item.read_position:
            raise ValueError("edit is not upstream of the read position")
        if len(edit.token_ids) != len(item.token_ids):
            raise ValueError("donor and clean traces differ in length")
        donor_states, _ = capture_states(
            model, edit.token_ids, bins, list(edit.positions)
        )
        for layer in bins:
            patched_logits = run_patched(
                model, item.token_ids, layer, list(edit.positions), donor_states[layer]
            )
            p_probs, p_logodds, p_mass = digit_readout(
                patched_logits, item.read_position, digit_ids
            )
            row = {
                "kind": edit.kind,
                "node": edit.node,
                "layer": layer,
                # The digit `delta_toward` is a delta toward. Without it the
                # delta says how far the readout moved but not toward what, so
                # reading the distribution below still needs the generator.
                "implied_value": edit.implied_target_value,
                # The third of the three competing digits, so `answer_moved` can
                # test the argmax of `probs_patched` from the row alone. It is
                # the same for every row of an item; a rescore recovers it for
                # runs that predate the field, but only where the summary
                # survived alongside the rows.
                "clean_value": item.target_value,
                "distance_to_read": edit.distance_to_read,
                "tv": float(0.5 * abs(p_probs - probs).sum()),
                "delta_toward": float(
                    p_logodds[edit.implied_target_value]
                    - logodds[edit.implied_target_value]
                ),
                "delta_away": float(
                    p_logodds[item.target_value] - logodds[item.target_value]
                ),
                # The baseline `delta_away` is a ratio against. Carrying it on
                # the row makes the clean answer's *patched* share computable
                # without joining back to the item summary.
                "clean_target_logodds": float(logodds[item.target_value]),
                "digit_mass_clean": mass,
                "digit_mass_patched": p_mass,
                # The whole readout, not just the projections of it this run
                # happened to ask for. Every scalar above is derivable from this
                # and `clean_probs` in the summary; the reverse is not true, and
                # finding that out cost a rerun. Ten floats a row.
                "probs_patched": [float(p) for p in p_probs],
            }
            if edit.donor_raw_value is not None:
                # Movement toward the digit the donor writes at the patched
                # position -- what a readout that copies what it finds there
                # would do, as opposed to carrying the value through the chain.
                # It needs the clean log-odds, which no rescore has, so it is
                # recorded here or not at all. Set for every value edit and for
                # the cross-item control; the tag edit writes no digit.
                row["delta_toward_raw"] = float(
                    p_logodds[edit.donor_raw_value]
                    - logodds[edit.donor_raw_value]
                )
                row["raw_value"] = edit.donor_raw_value
            if edit.donor_item is not None:
                row["donor_item"] = edit.donor_item
            rows.append(row)
    return rows, {
        "target_value": item.target_value,
        # Clean is a property of the item, so it is recorded once here rather
        # than repeated into every (edit, layer) row.
        "clean_probs": [float(p) for p in probs],
        "clean_top_digit": int(probs.argmax()),
        "clean_target_logodds": float(logodds[item.target_value]),
        "clean_digit_mass": mass,
    }


def scoring_layers(bins: list[int], n_layers: int | None) -> list[int]:
    """The layer bins a gate may be decided on.

    Patching the last decoder layer at a position upstream of the read position
    cannot reach that position: there is no later layer to carry it. Every TV
    there is 0 by construction, so a containment gate passes trivially
    (``0 <= 0 <= 0``) and an ``any(layer)`` rule would let the inert bin rescue a
    gate that fails at every informative layer. The prose summary already
    excluded it; this makes the scorer agree.

    ``n_layers`` is what identifies the model's final layer. Without it the
    caller has not said which bin that is, so every bin scores.
    """
    if n_layers is None:
        return list(bins)
    return [layer for layer in bins if layer != n_layers - 1]


def _quorum(n_items: int) -> int:
    """How many of ``n_items`` must clear a gate at a layer: all but one."""
    return max(1, n_items - 1)


def _answer_moved_gate(per_item: list[list[dict]], bins: list[int],
                       scored: list[int]) -> dict:
    """Did the patch actually change the answer, in absolute terms?

    Every other gate is a ratio or a one-sided comparison. ``ancestor_gap`` asks
    whether the ancestor edit perturbs the readout more than the controls do;
    ``directional_control`` asks whether it perturbs it the right way. Neither
    notices when both quantities are approximately zero. At ``depth2_gap0``,
    layer 6, the gap gate passes on ``tv_ancestor`` 0.026 against a null maximum
    of 0.0025 -- a clean 10x -- while the clean answer keeps 0.970 of the
    readout, and the directional gate passes 5/5 on log-ratio movement from
    about 1e-5 to about 1e-4. Both arms of the ladder past depth 1 were scored
    ``positive`` that way.

    Where the run stored ``probs_patched`` the question is settled exactly: the
    answer moved iff the clean digit is no longer alone at the top. A tie is not
    a move -- which of two co-maxima a bare argmax returns is an artefact of
    digit order, and the fresh ladder contains three exact top ties. Where the
    distribution is absent, ``delta_away`` plus the clean baseline gives the
    clean answer's patched share and ``ANSWER_MOVED_FALLBACK_MAX_CLEAN_SHARE``
    decides; ``test`` records which of the two ran.

    Failing this is a *scientific negative*, not an invalid test: such a patch
    was directional, quiet and selective, and simply did not change the answer.

    ``measured`` is False when neither test is available for any row, which is
    the state a pre-backfill legacy report is in. An unmeasurable floor must not
    read as a cleared one, so it fails.
    """
    def decide(row) -> tuple[bool, float, str] | None:
        probs, clean = row.get("probs_patched"), row.get("clean_value")
        if probs is not None and clean is not None:
            share = probs[clean]
            return share < max(probs), share, "argmax"
        baseline = row.get("clean_target_logodds")
        if baseline is None or "delta_away" not in row:
            return None
        share = math.exp(baseline + row["delta_away"])
        return share < ANSWER_MOVED_FALLBACK_MAX_CLEAN_SHARE, share, "majority"

    per_layer, tests = {}, set()
    for layer in bins:
        at_layer = [row for rows in per_item for row in rows
                    if row["kind"] == "ancestor" and row["layer"] == layer]
        decided = [d for d in map(decide, at_layer) if d is not None]
        tests.update(test for _, _, test in decided)
        per_layer[layer] = {
            "moved_items": sum(moved for moved, _, _ in decided),
            "n_items": len(at_layer),
            "median_clean_share": (
                statistics.median(share for _, share, _ in decided)
                if decided else None),
        }
    # One report's rows are all measured the same way, so a mixed set means the
    # rows disagree about what they carry and the gate should not claim either.
    measured = bool(tests)
    test = next(iter(tests)) if len(tests) == 1 else None
    return {
        "rule": ("clean answer no longer alone at the top of the digit readout"
                 if test == "argmax" else
                 f"clean answer below {ANSWER_MOVED_FALLBACK_MAX_CLEAN_SHARE} "
                 "of the digit readout")
                + ", for all but one item, at some scoring layer",
        "test": test,
        "measured": measured,
        "fallback_max_clean_share": ANSWER_MOVED_FALLBACK_MAX_CLEAN_SHARE,
        "per_layer": per_layer,
        "passes": test is not None and any(
            per_layer[layer]["moved_items"] >= _quorum(per_layer[layer]["n_items"])
            for layer in scored
        ),
        "applied_to_verdict": True,
    }


def _control_specificity_gate(per_item: list[list[dict]], bins: list[int]) -> dict:
    """Did the ancestor land on the digit it predicts, against a quiet background?

    ``answer_moved`` catches an arm where nothing moves. The written-versus-
    omitted run produced the mirror image. With the intermediate results
    unwritten the model stops solving the task, the clean readout goes nearly
    flat, and every edit flips the argmax: at ``depth2_omitted``, nulls 18/30
    and a comment-tag rewrite 3/5, on an arm the scorer calls positive. Every
    gate is relative, so a background that moves as much as the ancestor does
    clears all of them.

    The separating statistic is where the ancestor lands, not that it moved.
    ``ancestor_implied`` is the count landing on the donor-implied digit, which
    a control has no reason to produce; ``control_moved`` is how much of the
    background moved at all.

    Reported, never applied. Control flips are not unique to the broken arm --
    the published depth-1 positives have them at a lower rate -- so gating on
    them would be a retroactive policy move on evidence that does not yet
    support one. The archived eight store no distributions and cannot be checked
    this way at all.
    """
    controls = ("null", "surface_null", "non_ancestor")
    per_layer, measured = {}, False
    for layer in bins:
        counts = dict(ancestor_implied=0, ancestor_moved=0, n_items=0,
                      control_moved=0, n_control=0)
        for rows in per_item:
            for row in rows:
                probs, clean = row.get("probs_patched"), row.get("clean_value")
                if row["layer"] != layer or probs is None or clean is None:
                    continue
                measured = True
                moved = probs[clean] < max(probs)
                if row["kind"] == "ancestor":
                    counts["n_items"] += 1
                    counts["ancestor_moved"] += moved
                    counts["ancestor_implied"] += (
                        max(range(len(probs)), key=lambda d: probs[d])
                        == row.get("implied_value"))
                elif row["kind"] in controls:
                    counts["n_control"] += 1
                    counts["control_moved"] += moved
        per_layer[layer] = counts
    return {
        "rule": "the ancestor lands on the implied digit while the controls "
                "stay on the clean one",
        "measured": measured,
        "per_layer": per_layer,
        "applied_to_verdict": False,
    }


def _clean_answer_gate(items: list[dict] | None) -> dict:
    """Is the clean answer the model's own answer, uniquely?

    ``v3_distinct`` made the implied, raw and clean digits distinct. It did not
    make the model's clean behaviour correct: in the fresh depth-1 ladder the
    clean top digit disagrees with the target on 1/5, 1/5 and 2/5 items, and
    three observations are exact top ties. "The patch moved the answer off the
    clean target" is a counterfactual flip only where the clean target was the
    answer to begin with, so ``answer_moved`` is over-counting by however many
    items this gate reports.

    Reported, never applied. Binding the verdict to it would be a third
    retroactive policy move on runs already scored under two; the place to
    require clean correctness is the generator of the next family.

    ``n_tied`` is None for the archived reports, which store ``clean_top_digit``
    but no distribution: correctness is knowable there and uniqueness is not.
    """
    tops = [(item.get("clean_top_digit"), item.get("target_value"),
             item.get("clean_probs"))
            for item in items or []]
    known = [(top, target, probs) for top, target, probs in tops
             if top is not None and target is not None]
    with_probs = [probs for _, _, probs in known if probs]
    n_tied = sum(sorted(probs)[-1] == sorted(probs)[-2] for probs in with_probs)
    return {
        "rule": "the clean readout's top digit is the target value, uniquely",
        "measured": bool(known),
        "n_items": len(tops),
        "n_correct": sum(top == target for top, target, _ in known),
        "n_tied": n_tied if len(with_probs) == len(known) and known else None,
        "n_unique_correct": (
            sum(top == target and sorted(probs)[-1] > sorted(probs)[-2]
                for top, target, probs in known)
            if len(with_probs) == len(known) and known else None),
        "applied_to_verdict": False,
    }


def _joint_layer_gate(gates: dict, scored: list[int]) -> dict:
    """Which scoring layers clear every per-layer gate at once.

    An arm-level positive asserts that the patch was directional, quiet, and
    selective. Under `any(layer)` those three can each be satisfied at a
    different bin, and nothing in the report says so. This rule names the layers
    where they hold together.

    Fluency is arm-level, not per-layer -- it is measured over the ancestor rows
    of the whole run -- so it enters only through ``verdict_if_applied``.

    Reported, never applied: ``verdict_if_applied`` is what the arm would be
    called under this rule. The archived runs keep the verdicts their active
    policy gives them.
    """
    def clears(name: str, layer: int) -> bool:
        entry = gates[name]["per_layer"][layer]
        counted = ("n_positive" if name == "directional_control"
                   else "gap_items" if name == "ancestor_gap"
                   else "moved_items" if name == "answer_moved"
                   else "surface_items")
        if entry[counted] < _quorum(entry["n_items"]):
            return False
        return (name != "directional_control"
                or entry["median_delta_toward"] > 0)

    # Directional control and the surface gate say whether the patch measured
    # anything at that layer; the gap is only meaningful where both hold.
    valid = [layer for layer in scored
             if clears("directional_control", layer)
             and clears("surface_active", layer)]
    # A layer where the gap separates but the answer does not move is the
    # depth-2 case: real separation between two effects that are both
    # approximately nothing. It is not a layer at which the patch worked.
    joint = [layer for layer in valid
             if clears("ancestor_gap", layer)
             and gates["answer_moved"]["measured"]
             and clears("answer_moved", layer)]
    if not gates["fluency"]["passes"] or not valid:
        would_be = "invalid test"
    else:
        would_be = "positive" if joint else "scientific negative"
    return {
        "rule": PROSPECTIVE_LAYER_RULE,
        "layers": joint,
        "valid_layers": valid,
        "passes": bool(joint),
        "verdict_if_applied": would_be,
        "applied_to_verdict": False,
    }


def _cross_item_gate(per_item: list[list[dict]], bins: list[int],
                     scored: list[int]) -> dict:
    """The cross-item donor control: another item's state, same two positions.

    Every other edit rewrites the recipient's own trace, so none of them touches
    the sharpest objection to the ancestor gap -- that those positions are simply
    perturbation-sensitive, and any state written there would move the readout.
    This one writes a *foreign* state of the same span, width and formatting, and
    predicts a specific digit: the donor's value carried through the recipient's
    chain, which is neither the clean answer nor the donor's own digit.

    Two things must hold, and at the same layer:

    * ``n_toward`` -- the readout moves toward that predicted digit. The channel
      carries a value, and carries it out of the context it was measured in.
    * ``n_specific`` -- it moves toward the predicted digit more than toward the
      donor's own. Movement toward the donor's own digit is what copying the
      patched token looks like, and that is not a claim about the graph.

    The joint-layer rule applies from the start here. This gate has no archived
    verdict to preserve, so there is no reason to repeat the ``any(layer)``
    mistake that ``_joint_layer_gate`` exists to undo.

    Reported, never binding. The verdict space says whether the *within-item*
    intervention was valid; folding a new statistic into it before its null is
    known is the post-hoc move the last two checkpoints were spent undoing.
    """
    per_layer = {}
    for layer in bins:
        rows = [row for rows in per_item for row in rows
                if row["kind"] == "cross_item" and row["layer"] == layer]
        toward = [row["delta_toward"] for row in rows]
        per_layer[layer] = {
            "n_items": len(rows),
            "n_toward": sum(value > 0 for value in toward),
            "n_specific": sum(
                row["delta_toward"] > row["delta_toward_raw"] for row in rows
            ),
            "median_delta_toward": statistics.median(toward) if toward else 0.0,
            "median_tv": statistics.median([row["tv"] for row in rows])
            if rows else 0.0,
        }
    measured = any(entry["n_items"] for entry in per_layer.values())

    def clears(layer: int) -> bool:
        entry = per_layer[layer]
        return bool(
            entry["n_items"]
            and entry["n_toward"] >= _quorum(entry["n_items"])
            and entry["median_delta_toward"] > 0
            and entry["n_specific"] >= _quorum(entry["n_items"])
        )

    layers = [layer for layer in scored if clears(layer)] if measured else []
    return {
        "rule": PROSPECTIVE_LAYER_RULE,
        "measured": measured,
        "per_layer": per_layer,
        "layers": layers,
        "passes": bool(layers),
        "applied_to_verdict": False,
    }


def _surface_gate(passes_by_layer, bins, policy, n_by_layer, scored) -> dict:
    """Aggregate one surface policy with the same 4/5 rule as the other gates.

    ``v1_two_sided`` is the rule as originally registered: the surface
    perturbation must fall *inside* the range spanned by the null perturbations.
    That is a distributional-matching test, and the surface control was intended
    to test one-sided non-interference -- a computationally irrelevant edit must
    not move the readout *more* than an irrelevant value edit does.

    ``v2_one_sided`` states that role directly and is the active policy. It is a
    post-hoc amendment, so both policies stay runnable and both are reported.
    Passing v2 establishes only that the tag edit is quiet; it does not establish
    selectivity. The tag edit is a floor check, not a matched control -- the
    cross-item donor control is what would test selectivity.
    """
    per_layer = {
        layer: {
            "surface_items": sum(ok for lay, ok in passes_by_layer if lay == layer),
            "n_items": n_by_layer[layer],
        }
        for layer in bins
    }
    return {
        "policy": policy,
        "per_layer": per_layer,
        "failure_reason": (
            "surface_above_null" if policy == "v2_one_sided"
            else "surface_outside_null_spread"
        ),
        "passes": any(
            per_layer[layer]["surface_items"] >= _quorum(per_layer[layer]["n_items"])
            for layer in scored
        ),
    }


def evaluate_gates(per_item: list[list[dict]], bins: list[int],
                   policy: str = ACTIVE_GATE_POLICY,
                   n_layers: int | None = None,
                   items: list[dict] | None = None) -> dict:
    """The four continue-rules from the prototype specification.

    ``policy`` selects which surface rule the verdict is bound to. Both are
    always computed and reported; only the active one gates the verdict.
    """
    if policy not in GATE_POLICIES:
        raise ValueError(f"unknown gate policy {policy!r}, expected one of "
                         f"{GATE_POLICIES}")
    scored = scoring_layers(bins, n_layers)
    if not scored:
        raise ValueError(f"no scoring layers left in {bins} for {n_layers} layers")
    gates = {"gate_policy_version": policy, "scoring_layers": scored}

    # 1. Directional positive control. An ancestor patch must move the target
    #    toward the value implied by the donor, not merely away from the clean
    #    answer -- a patch that only corrupts the model also does the latter.
    directional = {}
    for layer in bins:
        toward = [
            row["delta_toward"]
            for rows in per_item
            for row in rows
            if row["kind"] == "ancestor" and row["layer"] == layer
        ]
        directional[layer] = {
            "n_positive": sum(value > 0 for value in toward),
            "n_items": len(toward),
            "median_delta_toward": statistics.median(toward) if toward else 0.0,
        }
    gates["directional_control"] = {
        "per_layer": directional,
        "passes": any(
            directional[layer]["n_positive"] >= _quorum(directional[layer]["n_items"])
            and directional[layer]["median_delta_toward"] > 0
            for layer in scored
        ),
    }

    # 2. Fluency. The answer distribution must stay on digits at all.
    ratios = [
        row["digit_mass_patched"] / row["digit_mass_clean"]
        for rows in per_item
        for row in rows
        if row["kind"] == "ancestor" and row["digit_mass_clean"] > 0
    ]
    gates["fluency"] = {
        "min_digit_mass_ratio": min(ratios) if ratios else 0.0,
        "passes": bool(ratios) and min(ratios) >= 0.5,
    }

    # 3. The ancestor-minus-non-ancestor gap must exceed the per-item null spread.
    # 4. The surface edit must not interfere; see ``_surface_gate`` for the two
    #    policies and why v2 is active.
    gap_pass, detail = [], []
    surface_pass = {name: [] for name in GATE_POLICIES}
    for rows in per_item:
        for layer in bins:
            at_layer = [row for row in rows if row["layer"] == layer]
            nulls = [row["tv"] for row in at_layer if row["kind"] == "null"]
            ancestor = next(r["tv"] for r in at_layer if r["kind"] == "ancestor")
            non_ancestor = next(r["tv"] for r in at_layer if r["kind"] == "non_ancestor")
            surface = next(r["tv"] for r in at_layer if r["kind"] == "surface_null")
            spread = max(nulls) - min(nulls)
            gap_pass.append((layer, (ancestor - non_ancestor) > spread))
            surface_pass["v1_two_sided"].append(
                (layer, min(nulls) <= surface <= max(nulls)))
            surface_pass["v2_one_sided"].append((layer, surface <= max(nulls)))
            detail.append({
                "layer": layer,
                "tv_ancestor": ancestor,
                "tv_non_ancestor": non_ancestor,
                "tv_surface_null": surface,
                "tv_null_min": min(nulls),
                "tv_null_max": max(nulls),
                "ancestor_above_all_nulls": ancestor > max(nulls),
                # Which side a v1 failure fell on. Every archived v1 failure but
                # one is "below", which is the direction the control wants.
                "surface_side": (
                    "below" if surface < min(nulls)
                    else "above" if surface > max(nulls)
                    else "in"
                ),
            })
    n_by_layer = {layer: sum(lay == layer for lay, _ in gap_pass) for layer in bins}
    gates["ancestor_gap"] = {
        "per_layer": {
            layer: {
                "gap_items": sum(ok for lay, ok in gap_pass if lay == layer),
                "n_items": n_by_layer[layer],
            }
            for layer in bins
        },
        "passes": any(
            sum(ok for lay, ok in gap_pass if lay == layer)
            >= _quorum(n_by_layer[layer])
            for layer in scored
        ),
    }
    for name in GATE_POLICIES:
        gates[f"surface_{name}"] = _surface_gate(
            surface_pass[name], bins, name, n_by_layer, scored
        )
    gates["surface_active"] = gates[f"surface_{policy}"]
    gates["cross_item_donor"] = _cross_item_gate(per_item, bins, scored)
    gates["answer_moved"] = _answer_moved_gate(per_item, bins, scored)
    gates["clean_answer"] = _clean_answer_gate(items)
    gates["control_specificity"] = _control_specificity_gate(per_item, bins)
    gates["prospective_joint_layer"] = _joint_layer_gate(gates, scored)
    gates["detail"] = detail
    return gates


def invalid_reasons(gates: dict) -> list[str]:
    """Why the intervention itself failed, if it did. Empty means it held.

    Directional control, fluency, and the active surface gate are validity
    requirements: they say whether the patch measured anything. Ancestor
    separation is only meaningful once all three hold, so it is not consulted
    here.
    """
    reasons = []
    if not gates["directional_control"]["passes"]:
        reasons.append("directional_control_failed")
    if not gates["fluency"]["passes"]:
        reasons.append("digit_mass_collapsed")
    if not gates["surface_active"]["passes"]:
        reasons.append(gates["surface_active"]["failure_reason"])
    return reasons


def verdict(gates: dict) -> str:
    """Positive, scientific negative, or invalid test -- kept strictly separate.

    An invalid test is not evidence about the model. Reporting one as a null is
    the specific failure the feasibility stage exists to catch. A loud surface
    edit belongs in that bucket: it means any two-token perturbation moves the
    readout, so the ancestor gap is not attributable to the edge.
    """
    if invalid_reasons(gates):
        return "invalid test"
    if gates["ancestor_gap"]["passes"] and gates["answer_moved"]["passes"]:
        return "positive"
    return "scientific negative"


# --------------------------------------------------------------------------
# offline rescoring
#
# The rows are the measurement; the gates and the verdict are a policy over
# them. Keeping the two separable is what lets a gate revision be re-run on an
# archived report instead of costing a GPU replay.
# --------------------------------------------------------------------------


def unflatten_rows(report: dict) -> list[list[dict]]:
    """Recover per-item row blocks from a stored report's flat ``rows``.

    ``measure_item`` emits ``for edit: for layer:``, so one item is a run of
    edit groups, each group one row per layer bin in ``layer_bins`` order. This
    validates that layout rather than assuming it: a silently regrouped file
    would mix two items into one block and produce a plausible-looking verdict.
    """
    rows = report["rows"]
    n_items = report["n_items"]
    bins = list(report["layer_bins"])
    if n_items <= 0 or len(rows) % n_items:
        raise ValueError(
            f"row count {len(rows)} does not divide into {n_items} items"
        )
    per_item = len(rows) // n_items
    if not bins or per_item % len(bins):
        raise ValueError(
            f"row count {per_item} per item is not a multiple of "
            f"{len(bins)} layer bins"
        )
    blocks = [rows[i * per_item:(i + 1) * per_item] for i in range(n_items)]
    for index, block in enumerate(blocks):
        for start in range(0, per_item, len(bins)):
            group = block[start:start + len(bins)]
            if [row["layer"] for row in group] != bins:
                raise ValueError(
                    f"item {index}: unexpected layer order "
                    f"{[row['layer'] for row in group]}, expected {bins}"
                )
            if len({row["kind"] for row in group}) != 1:
                raise ValueError(
                    f"item {index}: edit block does not have a single kind, got "
                    f"{sorted({row['kind'] for row in group})}"
                )
    return blocks


def _backfilled(per_item: list[list[dict]], items: list[dict] | None
                ) -> list[list[dict]]:
    """Recover the floor's inputs onto rows that predate them.

    Two fields, both stored once per item and needed per row. The archived
    reports have ``clean_target_logodds`` and ``delta_away``, which is enough for
    the fallback share; the runs since the row-schema change also have
    ``probs_patched`` and need only the clean digit -- ``target_value`` -- to be
    testable by argmax. Both are a join away, so neither costs a GPU replay or a
    rewrite of the archived files, which are immutable. Rows are copied rather
    than mutated so the caller's report is untouched.

    Reports supplying neither leave the floor unmeasurable, which
    ``_answer_moved_gate`` reports as a failure rather than a pass.
    """
    if not items:
        return per_item
    filled = []
    for block, summary in zip(per_item, items):
        recovered = {
            key: value for key, value in (
                ("clean_target_logodds", summary.get("clean_target_logodds")),
                ("clean_value", summary.get("target_value")),
            ) if value is not None
        }
        filled.append([
            row if row["kind"] != "ancestor"
            else {**recovered, **row} for row in block
        ])
    return filled


def rescore_report(report: dict, *, active: str = ACTIVE_GATE_POLICY) -> dict:
    """Re-run every gate policy over a stored report's rows.

    Returns a new report; the input and its rows are left untouched. Both
    policies' gates and verdicts are recorded so the legacy numbers stay
    auditable against the same measurement.
    """
    items = report.get("items")
    per_item = _backfilled(unflatten_rows(report), items)
    bins = list(report["layer_bins"])
    scoring = {}
    for policy in GATE_POLICIES:
        gates = evaluate_gates(per_item, bins, policy=policy,
                               n_layers=report.get("n_layers"), items=items)
        scoring[policy] = {
            "gates": gates,
            "verdict": verdict(gates),
            "invalid_reasons": invalid_reasons(gates),
        }
    rescored = dict(report)
    if "verdict" in report:
        rescored["original_verdict"] = report["verdict"]
        # Absent means the report was written before the floor existed, so the
        # verdict beside it came from the gap-only function.
        rescored["original_verdict_version"] = report.get(
            "verdict_version", VERDICT_VERSIONS[0])
    rescored["verdict_version"] = VERDICT_VERSION
    rescored["gate_policy_version"] = active
    rescored["gates"] = scoring[active]["gates"]
    rescored["verdict"] = scoring[active]["verdict"]
    rescored["scoring"] = scoring
    return rescored


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


def identity_patch_check(model, token_ids, bins, positions) -> dict:
    """Patch the clean run's own state back into itself; logits must not move.

    This is the one cheap check that pins the read and write sites to the same
    place. A silent mismatch there produces a plausible-looking null.
    """
    import torch

    states, clean_logits = capture_states(model, token_ids, bins, positions)
    worst = 0.0
    for layer in bins:
        patched = run_patched(model, token_ids, layer, positions, states[layer])
        worst = max(worst, float((patched - clean_logits).abs().max()))
    return {"max_abs_logit_change": worst, "passes": worst < 1e-3}


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def run(*, model_name: str, n_items: int, n_decoys: int, seed: int,
        output_path: str | None, self_test_only: bool,
        condition: str = "both", depth: int = 1, gap: int | None = None,
        generator: str = DEFAULT_GENERATOR, cross_item: bool = False,
        omit: str = "none") -> dict:
    from transformers import AutoTokenizer

    from collect_data import load_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    items = generate_items(
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        n_items=n_items,
        n_decoys=n_decoys,
        seed=seed,
        condition=condition,
        depth=depth,
        gap=gap,
        generator=generator,
        cross_item=cross_item,
        omit=omit,
    )
    digit_ids = digit_token_ids(tokenizer)

    model, _ = load_model(False, model_name=model_name)
    bins = layer_bins(model.config.num_hidden_layers)

    identity = identity_patch_check(
        model, items[0].token_ids, bins, list(items[0].edits[0].positions)
    )
    report = {
        "model": model_name,
        # Recorded from the start, so no later run has to have it inferred
        # the way the archived eight did.
        "generator": generator,
        "n_decoys": n_decoys,
        # The cross-item batch is selected for mutual donatability, so it is a
        # different batch from a plain run at the same seed. Recorded, not
        # inferable from the other settings.
        "cross_item": cross_item,
        "donor_map": [
            next((edit.donor_item for edit in item.edits
                  if edit.kind == "cross_item"), None)
            for item in items
        ] if cross_item else None,
        "condition": condition,
        "depth": depth,
        # Which lines state no result. Both the flag and the realised per-item
        # node names, because at depth 1 the flag is set and nothing is omitted,
        # and telling those apart is the contrast's own control.
        "omit": omit,
        "omitted_nodes": [list(item.omit) for item in items],
        "gap": [item.gap for item in items],
        "n_items": len(items),
        "seed": seed,
        "layer_bins": bins,
        "n_layers": model.config.num_hidden_layers,
        "n_tokens": [len(item.token_ids) for item in items],
        # Depth arms are matched against gap arms on this, not on the knobs.
        "ancestor_distance": [
            next(e.distance_to_read for e in item.edits if e.kind == "ancestor")
            for item in items
        ],
        "identity_patch": identity,
    }
    if not identity["passes"]:
        report["verdict"] = "invalid test"
        report["reason"] = "identity patch changed the logits; read and write sites differ"
        return report
    if self_test_only:
        report["verdict"] = "self test only"
        return report

    per_item, summaries = [], []
    for item in items:
        rows, summary = measure_item(model, item, bins, digit_ids)
        per_item.append(rows)
        summaries.append(summary)

    report["items"] = summaries
    report["rows"] = [row for rows in per_item for row in rows]
    # Score through the same path an archived report takes, so a fresh run and a
    # rescored one cannot disagree, and the row layout is validated on the way.
    report = rescore_report(report)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
    return report


def print_gate_table(report: dict) -> None:
    gates = report.get("gates")
    print(f"model          {report['model']}")
    # depth / gap / ancestor_distance postdate the first three artifacts. Show
    # them as unrecorded rather than inventing a default; `dag_evidence.py`
    # derives the true values by regenerating the items.
    print(f"condition      {report['condition']}  "
          f"depth {report.get('depth', '(v0: unrecorded)')}  "
          f"gap {report.get('gap', '(v0: unrecorded)')}")
    print(f"ancestor dist  {report.get('ancestor_distance', '(v0: unrecorded)')} "
          f"tokens to the read position")
    print(f"layer bins     {report['layer_bins']} of {report['n_layers']} layers")
    print(f"identity patch max |dlogit| = "
          f"{report['identity_patch']['max_abs_logit_change']:.2e} "
          f"({'pass' if report['identity_patch']['passes'] else 'FAIL'})")
    if not gates:
        print(f"verdict        {report['verdict']}")
        return
    print()
    print(f"gate policy    {report.get('gate_policy_version', ACTIVE_GATE_POLICY)}"
          f"   verdict fn {report.get('verdict_version', VERDICT_VERSIONS[0])}")
    print(f"{'gate':<32}{'passes':<10}detail")
    for name in ("directional_control", "fluency", "ancestor_gap",
                 "answer_moved",
                 *(f"surface_{policy}" for policy in GATE_POLICIES)):
        gate = gates[name]
        detail = {k: v for k, v in gate.items() if k != "passes"}
        print(f"{name:<32}{str(gate['passes']):<10}{json.dumps(detail)[:110]}")
    print()
    for policy, scored in report.get("scoring", {}).items():
        mark = " (active)" if policy == report.get("gate_policy_version") else ""
        reasons = ", ".join(scored["invalid_reasons"]) or "-"
        print(f"verdict[{policy}]{mark:<9} {scored['verdict']:<20} {reasons}")
    if "scoring" not in report:
        print(f"verdict        {report['verdict']}")
    joint = gates.get("prospective_joint_layer")
    if joint:
        print()
        print(f"{joint['rule']} (frozen for the next paired run, not applied here)")
        print(f"  layers where every gate clears together: "
              f"{joint['layers'] or 'none'}")
        print(f"  verdict if applied: {joint['verdict_if_applied']}")
    clean = gates.get("clean_answer")
    if clean and clean["measured"]:
        print()
        print("clean_answer (reported beside the verdict, never binding)")
        tied = "unknown (no clean distribution stored)" if clean["n_tied"] is None \
            else f"{clean['n_tied']}/{clean['n_items']}"
        print(f"  clean top digit is the target: "
              f"{clean['n_correct']}/{clean['n_items']}    top ties: {tied}")
    moved = gates.get("answer_moved")
    if moved and moved["measured"]:
        print()
        print(f"answer_moved [{moved['test'] or 'mixed'}]: {moved['rule']}")
        print(f"  {'layer':<8}{'moved':<10}median clean share")
        for layer, entry in moved["per_layer"].items():
            share = entry["median_clean_share"]
            print(f"  {str(layer):<8}{entry['moved_items']}/{entry['n_items']:<8}"
                  f"{'-' if share is None else format(share, '.3f')}")
    cross = gates.get("cross_item_donor")
    if cross and cross["measured"]:
        print()
        print("cross_item_donor (reported beside the verdict, never binding)")
        print(f"  {'layer':<8}{'toward':<10}{'specific':<11}"
              f"{'median toward':<16}median TV")
        for layer, entry in sorted(cross["per_layer"].items(),
                                   key=lambda pair: int(pair[0])):
            counted = f"{entry['n_toward']}/{entry['n_items']}"
            specific = f"{entry['n_specific']}/{entry['n_items']}"
            print(f"  {layer:<8}{counted:<10}{specific:<11}"
                  f"{entry['median_delta_toward']:<16.3f}"
                  f"{entry['median_tv']:.3f}")
        print(f"  layers clearing both: {cross['layers'] or 'none'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default=CHECKPOINTS[2])
    parser.add_argument("--n_items", type=int, default=5)
    parser.add_argument("--n_decoys", type=int, default=6)
    parser.add_argument("--generator", choices=GENERATORS,
                        default=DEFAULT_GENERATOR,
                        help="item family; v1_unpaired is the archived one "
                             "and is not paired across depth")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition", choices=DONOR_CONDITIONS, default="both",
                        help="which part of the edited line the donor rewrites")
    parser.add_argument("--depth", type=int, default=1,
                        help="steps from the edited ancestor to the target")
    parser.add_argument("--gap", type=int, default=None,
                        help="decoy lines between the chain and the target; the "
                             "distance control for --depth")
    parser.add_argument("--cross_item", action="store_true",
                        help="add the cross-item donor control; selects a "
                             "mutually donatable batch, so it is its own arm "
                             "and not comparable item-by-item to the ladder")
    parser.add_argument("--omit", choices=OMIT_MODES, default="none",
                        help="render some lines without stating their results, "
                             "padded to the same token count. 'chain' is the "
                             "written/omitted contrast for the depth collapse; "
                             "'decoy' omits as many values from lines the "
                             "target does not depend on, which is the control "
                             "for the notation itself. A no-op at depth 1")
    parser.add_argument("--output", default=None)
    parser.add_argument("--self_test", action="store_true",
                        help="run the identity patch and stop, no science")
    parser.add_argument("--rescore", default=None, metavar="REPORT_JSON",
                        help="re-run every gate policy over a stored report's "
                             "rows and stop; no model is loaded")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rescore:
        report = rescore_report(json.loads(Path(args.rescore).read_text()))
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2))
        print_gate_table(report)
        return
    report = run(
        model_name=args.model_name,
        n_items=args.n_items,
        n_decoys=args.n_decoys,
        generator=args.generator,
        seed=args.seed,
        condition=args.condition,
        depth=args.depth,
        gap=args.gap,
        cross_item=args.cross_item,
        omit=args.omit,
        output_path=args.output,
        self_test_only=args.self_test,
    )
    print_gate_table(report)


if __name__ == "__main__":
    main()
