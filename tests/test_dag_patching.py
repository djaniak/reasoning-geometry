"""Layer bins and gate logic for the DAG patching prototype.

Torch-free by design: ``dag_patching`` imports torch lazily inside the functions
that need it, so the part that decides positive / scientific negative / invalid
test can be tested anywhere. The hook mechanics live in
``test_dag_patching_hooks.py`` and need torch.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dag.dag_patching import (
    ACTIVE_GATE_POLICY,
    GATE_POLICIES,
    digit_token_ids,
    evaluate_gates,
    invalid_reasons,
    layer_bins,
    rescore_report,
    unflatten_rows,
    verdict,
)


def test_layer_bins_are_four_relative_depths():
    assert layer_bins(28) == [6, 13, 20, 27]
    assert layer_bins(24) == [5, 11, 17, 23]


def test_layer_bins_never_index_past_the_last_layer():
    # hidden_states[n_layers] is post-final-norm, so the deepest valid patch site
    # is the last decoder layer, not one past it.
    for n_layers in range(4, 81):
        assert max(layer_bins(n_layers)) == n_layers - 1
        assert min(layer_bins(n_layers)) >= 0


def test_digit_token_ids_rejects_a_multi_token_digit():
    class Splitting:
        def encode(self, text, add_special_tokens=False):
            return [1, 2]

    with pytest.raises(ValueError, match="not 1"):
        digit_token_ids(Splitting())


def test_digit_token_ids_rejects_colliding_ids():
    class Colliding:
        def encode(self, text, add_special_tokens=False):
            return [7]

    with pytest.raises(ValueError, match="not distinct"):
        digit_token_ids(Colliding())


# --------------------------------------------------------------------------
# gate logic
# --------------------------------------------------------------------------


def make_rows(layer, *, ancestor_toward, ancestor_tv, non_ancestor_tv,
              surface_tv, null_tvs, mass_ratio=1.0, cross=None,
              ancestor_away=-1.0, clean_target_logodds=-0.02,
              probs_patched=None, clean_value=None):
    rows = [{
        "kind": "ancestor", "node": "a", "layer": layer, "distance_to_read": 10,
        "tv": ancestor_tv, "delta_toward": ancestor_toward,
        "delta_away": ancestor_away,
        "clean_target_logodds": clean_target_logodds,
        "digit_mass_clean": 1.0, "digit_mass_patched": mass_ratio,
    }, {
        "kind": "non_ancestor", "node": "b", "layer": layer, "distance_to_read": 12,
        "tv": non_ancestor_tv, "delta_toward": 0.0, "delta_away": 0.0,
        "digit_mass_clean": 1.0, "digit_mass_patched": 1.0,
    }, {
        "kind": "surface_null", "node": None, "layer": layer, "distance_to_read": 20,
        "tv": surface_tv, "delta_toward": 0.0, "delta_away": 0.0,
        "digit_mass_clean": 1.0, "digit_mass_patched": 1.0,
    }]
    for index, value in enumerate(null_tvs):
        rows.append({
            "kind": "null", "node": f"n{index}", "layer": layer,
            "distance_to_read": 30 + index, "tv": value,
            "delta_toward": 0.0, "delta_away": 0.0,
            "digit_mass_clean": 1.0, "digit_mass_patched": 1.0,
        })
    if cross is not None:
        toward, raw, tv = cross
        rows.append({
            "kind": "cross_item", "node": "a", "layer": layer,
            "distance_to_read": 10, "tv": tv, "donor_item": 1,
            "delta_toward": toward, "delta_toward_raw": raw, "delta_away": 0.0,
            "digit_mass_clean": 1.0, "digit_mass_patched": 1.0,
        })
    if probs_patched is not None:
        rows[0]["probs_patched"] = list(probs_patched)
    if clean_value is not None:
        rows[0]["clean_value"] = clean_value
    return rows


def spread(clean_share, clean_value=3, n=10):
    """A ten-way distribution giving ``clean_value`` a share, rest uniform."""
    rest = (1.0 - clean_share) / (n - 1)
    return [clean_share if d == clean_value else rest for d in range(n)]


def item_rows(**kwargs):
    return [make_rows(0, **kwargs)]


def test_clean_positive_result():
    gates = evaluate_gates(item_rows(
        ancestor_toward=2.0, ancestor_tv=0.8, non_ancestor_tv=0.05,
        surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02],
    ), [0])
    assert verdict(gates) == "positive"


def test_valid_intervention_without_graph_structure_is_a_scientific_negative():
    # The directional control passes, so the patch works; the ancestor simply
    # does not stand out from the independent branch.
    gates = evaluate_gates(item_rows(
        ancestor_toward=2.0, ancestor_tv=0.30, non_ancestor_tv=0.28,
        surface_tv=0.27, null_tvs=[0.20, 0.25, 0.30, 0.28, 0.26, 0.22],
    ), [0])
    assert verdict(gates) == "scientific negative"


def test_undirected_movement_alone_is_an_invalid_test():
    # Large, selective-looking movement, but the target moves away from the
    # donor-implied value. A patch that merely corrupts the model does this.
    gates = evaluate_gates(item_rows(
        ancestor_toward=-3.0, ancestor_tv=0.9, non_ancestor_tv=0.05,
        surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02],
    ), [0])
    assert gates["directional_control"]["passes"] is False
    assert verdict(gates) == "invalid test"


def test_collapsed_digit_mass_is_an_invalid_test():
    gates = evaluate_gates(item_rows(
        ancestor_toward=2.0, ancestor_tv=0.9, non_ancestor_tv=0.05,
        surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02],
        mass_ratio=0.1,
    ), [0])
    assert gates["fluency"]["passes"] is False
    assert verdict(gates) == "invalid test"


# --------------------------------------------------------------------------
# surface control: v1 two-sided containment vs v2 one-sided non-interference
#
# v1 asked the surface perturbation to fall *inside* the range of the null
# perturbations, which is a distributional-matching test. The surface control was
# intended to test one-sided non-interference: a computationally irrelevant edit
# must not move the readout *more* than an irrelevant value edit does. v2 states
# that role directly. Passing v2 establishes only that the tag edit is quiet; it
# does not establish selectivity.
# --------------------------------------------------------------------------


def test_surface_below_the_null_spread_passes_v2_but_fails_v1():
    kwargs = dict(
        ancestor_toward=2.0, ancestor_tv=0.8, non_ancestor_tv=0.05,
        surface_tv=0.001, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02],
    )
    v2 = evaluate_gates(item_rows(**kwargs), [0], policy="v2_one_sided")
    assert v2["surface_active"]["passes"] is True
    assert verdict(v2) == "positive"

    v1 = evaluate_gates(item_rows(**kwargs), [0], policy="v1_two_sided")
    assert v1["surface_active"]["passes"] is False
    assert verdict(v1) == "invalid test"


def test_surface_above_the_null_spread_fails_both_policies():
    kwargs = dict(
        ancestor_toward=2.0, ancestor_tv=0.8, non_ancestor_tv=0.05,
        surface_tv=0.60, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02],
    )
    for policy in GATE_POLICIES:
        gates = evaluate_gates(item_rows(**kwargs), [0], policy=policy)
        assert gates["surface_active"]["passes"] is False, policy
        assert verdict(gates) == "invalid test", policy


def test_a_loud_surface_edit_cannot_produce_a_positive_verdict():
    # The ancestor separates cleanly from the non-ancestor, which under the old
    # scorer was enough. A surface edit louder than every null edit means any
    # two-token perturbation moves the readout, so the gap is not attributable
    # to the edge -- an intervention failure, not evidence about the model.
    gates = evaluate_gates(item_rows(
        ancestor_toward=2.0, ancestor_tv=0.8, non_ancestor_tv=0.05,
        surface_tv=0.60, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02],
    ), [0])
    assert gates["ancestor_gap"]["passes"] is True
    assert verdict(gates) == "invalid test"
    assert invalid_reasons(gates) == ["surface_above_null"]


def test_both_policies_are_reported_whichever_one_is_active():
    gates = evaluate_gates(item_rows(
        ancestor_toward=2.0, ancestor_tv=0.8, non_ancestor_tv=0.05,
        surface_tv=0.001, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02],
    ), [0], policy="v2_one_sided")
    assert gates["surface_v1_two_sided"]["passes"] is False
    assert gates["surface_v2_one_sided"]["passes"] is True
    assert gates["gate_policy_version"] == "v2_one_sided"


def test_the_active_policy_defaults_to_v2():
    assert ACTIVE_GATE_POLICY == "v2_one_sided"
    gates = evaluate_gates(item_rows(
        ancestor_toward=2.0, ancestor_tv=0.8, non_ancestor_tv=0.05,
        surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02],
    ), [0])
    assert gates["gate_policy_version"] == "v2_one_sided"


def test_surface_gate_uses_the_same_four_of_five_rule_as_the_other_gates():
    # Four quiet items and one loud one: the aggregation rule tolerates a single
    # dissenter, exactly as the directional and gap gates do.
    quiet = dict(ancestor_toward=1.5, ancestor_tv=0.7, non_ancestor_tv=0.05,
                 surface_tv=0.001, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02])
    rows = [make_rows(0, **quiet) for _ in range(4)]
    rows.append(make_rows(0, **{**quiet, "surface_tv": 0.60}))
    gates = evaluate_gates(rows, [0])
    assert gates["surface_active"]["per_layer"][0]["surface_items"] == 4
    assert gates["surface_active"]["passes"] is True
    assert verdict(gates) == "positive"

    rows.append(make_rows(0, **{**quiet, "surface_tv": 0.60}))
    gates = evaluate_gates(rows, [0])
    assert gates["surface_active"]["per_layer"][0]["surface_items"] == 4
    assert gates["surface_active"]["passes"] is False
    assert verdict(gates) == "invalid test"


def test_the_final_decoder_layer_is_not_a_scoring_layer():
    # Patching the last decoder layer at a position upstream of the read cannot
    # reach the read position: no later layer exists to move it. Every TV there
    # is 0, so a containment gate passes trivially (0 <= 0 <= 0) and `any(layer)`
    # would let that inert bin rescue a gate that fails at every real layer.
    loud = dict(ancestor_toward=2.0, ancestor_tv=0.8, non_ancestor_tv=0.05,
                surface_tv=0.60, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02])
    inert = dict(ancestor_toward=0.0, ancestor_tv=0.0, non_ancestor_tv=0.0,
                 surface_tv=0.0, null_tvs=[0.0] * 6)
    rows = [make_rows(20, **loud) + make_rows(27, **inert) for _ in range(5)]

    scored = evaluate_gates(rows, [20, 27], n_layers=28)
    assert scored["scoring_layers"] == [20]
    assert scored["surface_active"]["passes"] is False
    assert verdict(scored) == "invalid test"

    # Without n_layers the caller has not said which bin is the model's last, so
    # every bin scores -- the synthetic gate tests rely on that.
    unscoped = evaluate_gates(rows, [20, 27])
    assert unscoped["scoring_layers"] == [20, 27]
    assert unscoped["surface_active"]["passes"] is True


def test_rescore_excludes_the_final_layer_using_the_reports_n_layers():
    report = stored_report(surface_tv=0.60)
    rescored = rescore_report(report)
    assert rescored["gates"]["scoring_layers"] == [6, 13, 20]


# --------------------------------------------------------------------------
# the prospective joint-layer rule
#
# Every gate aggregates with `any(layer)`, so each one may clear at a different
# bin. That yields an arm-level positive with no single layer at which the patch
# was directional, quiet, and selective at once -- which is what an arm-level
# positive is meant to assert. The rule below requires one such layer. It is
# frozen for the next paired run and reported here alongside the active verdict;
# applying it to the archived reports would be a third post-hoc policy move.
# --------------------------------------------------------------------------


def split_layer_rows():
    """Directional control clears only at layer 0; the gap only at layer 1."""
    return [
        # Directional, quiet, but the gap does not exceed the null spread.
        make_rows(0, ancestor_toward=2.0, ancestor_tv=0.30, non_ancestor_tv=0.28,
                  surface_tv=0.20, null_tvs=[0.20, 0.25, 0.30, 0.28, 0.26, 0.22])
        # Selective and quiet, but the patch moves away from the donor value.
        + make_rows(1, ancestor_toward=-2.0, ancestor_tv=0.80, non_ancestor_tv=0.05,
                    surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02])
        for _ in range(5)
    ]


def test_the_joint_layer_rule_lists_every_layer_where_all_gates_clear_together():
    rows = [
        make_rows(0, ancestor_toward=1.5, ancestor_tv=0.7, non_ancestor_tv=0.05,
                  surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02])
        + make_rows(1, ancestor_toward=-1.0, ancestor_tv=0.05, non_ancestor_tv=0.05,
                    surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02])
        for _ in range(5)
    ]
    joint = evaluate_gates(rows, [0, 1])["prospective_joint_layer"]
    assert joint["layers"] == [0]
    assert joint["passes"] is True


def test_gates_clearing_at_different_layers_leave_no_joint_layer():
    joint = evaluate_gates(split_layer_rows(), [0, 1])["prospective_joint_layer"]
    assert joint["layers"] == []
    assert joint["passes"] is False
    assert joint["verdict_if_applied"] == "scientific negative"


def test_the_prospective_rule_does_not_touch_the_active_verdict():
    # The active `any(layer)` rule calls this positive: directional clears at
    # layer 0, the gap at layer 1. The prospective rule would not.
    gates = evaluate_gates(split_layer_rows(), [0, 1])
    assert gates["directional_control"]["passes"] is True
    assert gates["ancestor_gap"]["passes"] is True
    assert verdict(gates) == "positive"
    assert gates["prospective_joint_layer"]["applied_to_verdict"] is False
    assert gates["prospective_joint_layer"]["verdict_if_applied"] != "positive"


def test_a_quiet_loud_split_across_layers_is_an_invalid_test_under_the_rule():
    # Directional and selective at layer 0, but the surface edit is louder than
    # every null there. Validity is a per-layer property too.
    rows = [
        make_rows(0, ancestor_toward=2.0, ancestor_tv=0.80, non_ancestor_tv=0.05,
                  surface_tv=0.60, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02])
        + make_rows(1, ancestor_toward=-2.0, ancestor_tv=0.05, non_ancestor_tv=0.05,
                    surface_tv=0.01, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02])
        for _ in range(5)
    ]
    joint = evaluate_gates(rows, [0, 1])["prospective_joint_layer"]
    assert joint["verdict_if_applied"] == "invalid test"


def test_the_joint_rule_is_bounded_by_the_scoring_layers():
    # Layer 27 is given rows that would clear every gate, to show it is excluded
    # because it is the final decoder layer and not because its real rows are 0.
    clean = dict(ancestor_toward=2.0, ancestor_tv=0.80, non_ancestor_tv=0.05,
                 surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02])
    dead = dict(ancestor_toward=-2.0, ancestor_tv=0.05, non_ancestor_tv=0.05,
                surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02])
    rows = [make_rows(20, **dead) + make_rows(27, **clean) for _ in range(5)]
    gates = evaluate_gates(rows, [20, 27], n_layers=28)
    assert gates["prospective_joint_layer"]["layers"] == []
    assert evaluate_gates(rows, [20, 27])["prospective_joint_layer"]["layers"] == [27]


def test_rescore_reports_the_prospective_rule_for_every_policy():
    scoring = rescore_report(stored_report())["scoring"]
    for policy in GATE_POLICIES:
        joint = scoring[policy]["gates"]["prospective_joint_layer"]
        assert joint["rule"] == "joint_layer"
        assert joint["applied_to_verdict"] is False


def test_a_hairs_breadth_above_the_null_max_is_still_a_failure():
    # L6 item 1 of the archived result_only run: surface 0.0153 against a null
    # max of 0.0152. No epsilon; the 4/5 aggregation rule is what absorbs it.
    gates = evaluate_gates(item_rows(
        ancestor_toward=2.0, ancestor_tv=0.8, non_ancestor_tv=0.05,
        surface_tv=0.0153, null_tvs=[0.0006, 0.0152, 0.0035, 0.0011, 0.0009, 0.0003],
    ), [0])
    assert gates["surface_active"]["passes"] is False


def test_gap_must_exceed_the_null_spread_not_merely_be_positive():
    gates = evaluate_gates(item_rows(
        ancestor_toward=2.0, ancestor_tv=0.40, non_ancestor_tv=0.30,
        surface_tv=0.20, null_tvs=[0.05, 0.10, 0.20, 0.35, 0.15, 0.08],
    ), [0])
    assert gates["ancestor_gap"]["passes"] is False
    assert verdict(gates) == "scientific negative"


def test_a_single_passing_layer_bin_is_enough():
    rows = [
        make_rows(0, ancestor_toward=-1.0, ancestor_tv=0.05, non_ancestor_tv=0.05,
                  surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02])
        + make_rows(1, ancestor_toward=2.0, ancestor_tv=0.80, non_ancestor_tv=0.05,
                    surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02])
    ]
    gates = evaluate_gates(rows, [0, 1])
    assert verdict(gates) == "positive"
    assert gates["directional_control"]["per_layer"][0]["n_positive"] == 0
    assert gates["directional_control"]["per_layer"][1]["n_positive"] == 1


def test_gates_hold_across_five_items():
    rows = [
        make_rows(0, ancestor_toward=1.5, ancestor_tv=0.7, non_ancestor_tv=0.05,
                  surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02])
        for _ in range(5)
    ]
    gates = evaluate_gates(rows, [0])
    assert gates["directional_control"]["per_layer"][0]["n_items"] == 5
    assert gates["ancestor_gap"]["per_layer"][0]["gap_items"] == 5
    assert verdict(gates) == "positive"


def test_one_dissenting_item_out_of_five_still_passes():
    rows = [
        make_rows(0, ancestor_toward=1.5, ancestor_tv=0.7, non_ancestor_tv=0.05,
                  surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02])
        for _ in range(4)
    ]
    rows.append(make_rows(
        0, ancestor_toward=-0.2, ancestor_tv=0.05, non_ancestor_tv=0.05,
        surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02],
    ))
    gates = evaluate_gates(rows, [0])
    assert gates["directional_control"]["per_layer"][0]["n_positive"] == 4
    assert verdict(gates) == "positive"


def test_two_dissenting_items_out_of_five_fail_the_directional_control():
    rows = [
        make_rows(0, ancestor_toward=1.5, ancestor_tv=0.7, non_ancestor_tv=0.05,
                  surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02])
        for _ in range(3)
    ]
    rows += [
        make_rows(0, ancestor_toward=-0.2, ancestor_tv=0.05, non_ancestor_tv=0.05,
                  surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02])
        for _ in range(2)
    ]
    gates = evaluate_gates(rows, [0])
    assert verdict(gates) == "invalid test"


# --------------------------------------------------------------------------
# offline rescoring of an archived report
#
# The rows are the measurement; the gates and the verdict are a policy over
# them. Splitting the two is what makes a gate revision cost no GPU.
# --------------------------------------------------------------------------


def stored_report(n_items=5, bins=(6, 13, 20, 27), *, surface_tv=0.001,
                  verdict_str="positive", with_cross=False):
    """A report laid out the way ``measure_item`` flattens it: edit-major.

    ``measure_item`` loops ``for edit: for layer:``, so one item's block is nine
    consecutive edit groups of four layer rows each -- not the kind-major order
    ``make_rows`` produces for the gate tests. ``with_cross`` adds the tenth
    group, which is what a cross-item run stores.
    """
    kinds = ["ancestor", "non_ancestor", "surface_null"] + ["null"] * 6
    if with_cross:
        kinds.append("cross_item")
    tvs = {"ancestor": 0.8, "non_ancestor": 0.05, "surface_null": surface_tv,
           "null": 0.03, "cross_item": 0.5}
    rows = []
    for _ in range(n_items):
        for kind in kinds:
            for layer in bins:
                row = {
                    "kind": kind, "node": None, "layer": layer,
                    "distance_to_read": 11, "tv": tvs[kind],
                    "delta_toward": 2.0 if kind in ("ancestor", "cross_item")
                    else 0.0,
                    # The ancestor edit drives the clean answer off the readout,
                    # so these reports clear the absolute-effect floor and the
                    # verdicts here still mean what they meant before it existed.
                    "delta_away": -5.0 if kind == "ancestor" else 0.0,
                    "digit_mass_clean": 1.0, "digit_mass_patched": 1.0,
                }
                if kind == "ancestor":
                    row["clean_target_logodds"] = -0.02
                if kind == "cross_item":
                    row |= {"delta_toward_raw": 0.2, "donor_item": 1}
                rows.append(row)
    return {
        "model": "test", "condition": "both", "n_items": n_items, "seed": 0,
        "layer_bins": list(bins), "n_layers": 28, "rows": rows,
        "verdict": verdict_str,
        "items": [{"target_value": 3, "clean_target_logodds": -0.02,
                   "clean_top_digit": 3, "clean_digit_mass": 1.0}
                  for _ in range(n_items)],
    }


def test_unflatten_recovers_one_block_per_item():
    per_item = unflatten_rows(stored_report())
    assert len(per_item) == 5
    assert all(len(block) == 36 for block in per_item)
    assert {row["kind"] for row in per_item[0]} == {
        "ancestor", "non_ancestor", "surface_null", "null"}


def test_unflatten_rejects_a_row_count_that_does_not_divide_by_items():
    report = stored_report()
    report["rows"].pop()
    with pytest.raises(ValueError, match="row count"):
        unflatten_rows(report)


def test_unflatten_rejects_rows_in_an_unexpected_layer_order():
    # Kind-major within a layer instead of edit-major: the same rows, regrouped.
    # Silently accepting this would mix two items into one block.
    report = stored_report()
    report["rows"] = sorted(report["rows"], key=lambda row: row["layer"])
    with pytest.raises(ValueError, match="layer order"):
        unflatten_rows(report)


def test_unflatten_rejects_an_edit_block_with_mixed_kinds():
    report = stored_report()
    report["rows"][1]["kind"] = "null"
    with pytest.raises(ValueError, match="single kind"):
        unflatten_rows(report)


def test_rescore_reports_both_policies_and_preserves_the_original_verdict():
    report = rescore_report(stored_report(surface_tv=0.001))
    assert report["original_verdict"] == "positive"
    assert report["gate_policy_version"] == "v2_one_sided"
    assert report["verdict"] == "positive"
    assert report["scoring"]["v1_two_sided"]["verdict"] == "invalid test"
    assert report["scoring"]["v2_one_sided"]["verdict"] == "positive"
    assert set(report["scoring"]) == set(GATE_POLICIES)


def test_rescore_under_v1_reproduces_the_legacy_verdict():
    # The v1 scorer is kept runnable, not just described, so the published
    # legacy numbers stay auditable against the same rows.
    report = rescore_report(stored_report(surface_tv=0.03))
    assert report["scoring"]["v1_two_sided"]["verdict"] == "positive"
    assert report["scoring"]["v1_two_sided"]["invalid_reasons"] == []


def test_rescore_records_why_a_verdict_changed():
    report = rescore_report(stored_report(surface_tv=0.60))
    assert report["verdict"] == "invalid test"
    assert report["scoring"]["v2_one_sided"]["invalid_reasons"] == [
        "surface_above_null"]


def test_rescore_leaves_the_measurement_rows_untouched():
    original = stored_report()
    before = [dict(row) for row in original["rows"]]
    rescore_report(original)
    assert original["rows"] == before


# --------------------------------------------------------------------------
# the cross-item donor control
#
# Every other edit rewrites the recipient's own trace, so the sharpest objection
# to the ancestor gap survives all of them: those two positions might just be
# perturbation-sensitive. The cross-item donor writes another item's state at the
# same positions, same span and width, and predicts a specific digit -- the donor
# value carried through the recipient's chain, which is neither the clean answer
# nor the donor's own digit.
#
# Registered before the first run, with the joint-layer rule applied from the
# start: this gate has no archived verdicts to preserve, so there is no reason to
# repeat the `any(layer)` mistake here. Reported, never binding -- the verdict
# space is about whether the *within-item* intervention was valid, and folding a
# brand-new statistic into it before its null is known is the post-hoc move the
# previous two checkpoints existed to undo.
# --------------------------------------------------------------------------


def cross_rows(layer, cross):
    return make_rows(layer, ancestor_toward=2.0, ancestor_tv=0.8,
                     non_ancestor_tv=0.05, surface_tv=0.04,
                     null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02],
                     cross=cross)


def test_a_run_without_cross_item_rows_reports_the_control_as_unmeasured():
    gate = evaluate_gates(item_rows(
        ancestor_toward=2.0, ancestor_tv=0.8, non_ancestor_tv=0.05,
        surface_tv=0.04, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02],
    ), [0])["cross_item_donor"]
    assert gate["measured"] is False
    assert gate["passes"] is False
    assert gate["layers"] == []


def test_a_donor_carried_through_the_chain_clears_the_control():
    # Moves toward the propagated digit, and more than toward the donor's own.
    gates = evaluate_gates([cross_rows(0, (1.8, 0.2, 0.5)) for _ in range(5)], [0])
    gate = gates["cross_item_donor"]
    assert gate["measured"] is True
    assert gate["layers"] == [0]
    assert gate["passes"] is True


def test_a_donor_that_does_not_move_the_readout_fails_the_control():
    gate = evaluate_gates(
        [cross_rows(0, (-0.4, -0.3, 0.5)) for _ in range(5)], [0]
    )["cross_item_donor"]
    assert gate["per_layer"][0]["n_toward"] == 0
    assert gate["passes"] is False


def test_movement_toward_the_raw_donor_digit_is_not_specific():
    # The readout moves, and in the right direction, but the donor's own digit
    # rises more -- which is what copying the patched token would look like.
    gate = evaluate_gates(
        [cross_rows(0, (0.4, 2.0, 0.5)) for _ in range(5)], [0]
    )["cross_item_donor"]
    assert gate["per_layer"][0]["n_toward"] == 5
    assert gate["per_layer"][0]["n_specific"] == 0
    assert gate["passes"] is False


def test_the_control_uses_the_same_quorum_as_every_other_gate():
    rows = [cross_rows(0, (1.8, 0.2, 0.5)) for _ in range(4)]
    rows.append(cross_rows(0, (-1.0, 0.2, 0.5)))
    gate = evaluate_gates(rows, [0])["cross_item_donor"]
    assert gate["per_layer"][0]["n_toward"] == 4
    assert gate["passes"] is True, "4 of 5 is the quorum everywhere else"

    rows[3] = cross_rows(0, (-1.0, 0.2, 0.5))
    assert evaluate_gates(rows, [0])["cross_item_donor"]["passes"] is False


def test_direction_and_specificity_must_clear_at_the_same_layer():
    rows = [cross_rows(0, (1.8, 0.2, 0.5)) + cross_rows(1, (0.4, 2.0, 0.5))
            for _ in range(5)]
    gate = evaluate_gates(rows, [0, 1])["cross_item_donor"]
    assert gate["layers"] == [0], "layer 1 is directional but not specific"


def test_the_control_is_bounded_by_the_scoring_layers():
    # The inert final layer cannot carry this gate any more than the others.
    rows = [cross_rows(20, (0.0, 0.0, 0.0)) + cross_rows(27, (1.8, 0.2, 0.5))
            for _ in range(5)]
    assert evaluate_gates(rows, [20, 27], n_layers=28)["cross_item_donor"][
        "layers"] == []
    assert evaluate_gates(rows, [20, 27])["cross_item_donor"]["layers"] == [27]


def test_the_control_never_decides_the_verdict():
    # A control that fails everything must not turn a positive arm into anything
    # else; it is reported beside the verdict, not folded into it.
    gates = evaluate_gates(
        [cross_rows(0, (-2.0, 3.0, 0.9)) for _ in range(5)], [0])
    assert gates["cross_item_donor"]["passes"] is False
    assert gates["cross_item_donor"]["applied_to_verdict"] is False
    assert verdict(gates) == "positive"
    assert invalid_reasons(gates) == []


def test_a_report_with_cross_item_rows_still_rescores():
    report = stored_report(with_cross=True)
    rescored = rescore_report(report)
    assert rescored["verdict"] == "positive"
    for policy in GATE_POLICIES:
        gate = rescored["scoring"][policy]["gates"]["cross_item_donor"]
        assert gate["measured"] is True
        assert gate["applied_to_verdict"] is False


# --------------------------------------------------------------------------
# the absolute-effect floor
# --------------------------------------------------------------------------
#
# Every other gate is a ratio or a one-sided comparison: the ancestor edit must
# perturb the readout more than the controls do, and must move it in the right
# direction. None of them asks whether the answer changed. That let the depth-2
# and depth-3 ladder arms score `positive` on runs where the clean answer keeps
# 0.97 of the digit readout -- `tv_ancestor` 0.026 against `tv_null_max` 0.0025
# is a clean 10x between two numbers that are both approximately zero, and
# `median_delta_toward` of 1.84 nats is movement from about 1e-5 to about 1e-4.
#
# The floor asks whether the patched readout still puts the clean answer on top.
# These tests exercise the fallback share, which is all a report without a stored
# distribution can offer; the section further down covers the argmax test that
# supersedes it, and why the justification originally given for the half was
# wrong. Failing the floor is a scientific negative, not an invalid test -- the
# intervention was directional, quiet and selective, and simply did not change
# the answer.


def floor_rows(layer, *, away, clean=-0.02, gap=True, **kwargs):
    """Rows whose relative gates all pass, parameterised on absolute movement."""
    return make_rows(
        layer, ancestor_toward=2.0, ancestor_tv=0.8 if gap else 0.03,
        non_ancestor_tv=0.05 if gap else 0.028,
        surface_tv=0.04 if gap else 0.001,
        null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02] if gap
        else [0.002, 0.003, 0.005, 0.004, 0.003, 0.002],
        ancestor_away=away, clean_target_logodds=clean, **kwargs,
    )


def test_an_arm_that_never_moves_the_answer_is_not_positive():
    # The depth-2 case: the relative gates all pass, and the clean answer keeps
    # 0.97 of the readout.
    gates = evaluate_gates([floor_rows(0, away=-0.005)] * 5, [0])
    assert gates["ancestor_gap"]["passes"] is True
    assert gates["answer_moved"]["passes"] is False
    assert verdict(gates) == "scientific negative"


def test_an_arm_that_moves_the_answer_off_the_clean_digit_stays_positive():
    gates = evaluate_gates([floor_rows(0, away=-5.0)] * 5, [0])
    assert gates["answer_moved"]["passes"] is True
    assert verdict(gates) == "positive"


def test_failing_the_floor_is_a_negative_not_an_invalid_test():
    # The patch measured something; it just did not change the answer. Calling
    # that invalid would throw away a real null.
    gates = evaluate_gates([floor_rows(0, away=-0.005)] * 5, [0])
    assert "answer_did_not_move" not in invalid_reasons(gates)
    assert invalid_reasons(gates) == []


def test_the_floor_reads_the_clean_answers_share_of_the_patched_readout():
    # exp(clean_target_logodds + delta_away) is the clean answer's share after
    # patching, and half is the line.
    just_over = evaluate_gates([floor_rows(0, away=-0.6, clean=0.0)] * 5, [0])
    just_under = evaluate_gates([floor_rows(0, away=-0.8, clean=0.0)] * 5, [0])
    assert just_over["answer_moved"]["per_layer"][0]["moved_items"] == 0
    assert just_under["answer_moved"]["per_layer"][0]["moved_items"] == 5


def test_the_floor_needs_a_quorum_of_items_not_just_one():
    rows = ([floor_rows(0, away=-5.0)] * 2) + ([floor_rows(0, away=-0.005)] * 3)
    gates = evaluate_gates(rows, [0])
    assert gates["answer_moved"]["per_layer"][0]["moved_items"] == 2
    assert gates["answer_moved"]["passes"] is False


def test_the_joint_layer_requires_the_answer_to_have_moved_at_that_layer():
    # Layer 0 separates but does not move the answer; layer 1 does both. Without
    # the floor in the joint rule, layer 0 would qualify.
    rows = [floor_rows(0, away=-0.005) + floor_rows(1, away=-5.0)
            for _ in range(5)]
    gates = evaluate_gates(rows, [0, 1])
    assert gates["answer_moved"]["per_layer"][0]["moved_items"] == 0
    assert gates["prospective_joint_layer"]["layers"] == [1]


def test_a_report_without_the_clean_baseline_is_backfilled_on_rescore():
    # The archived reports predate the field: their rows carry `delta_away` and
    # their item summaries carry `clean_target_logodds`, so the join recovers it
    # rather than needing a GPU replay.
    report = stored_report()
    for row in report["rows"]:
        row.pop("clean_target_logodds", None)
    rescored = rescore_report(report)
    assert rescored["gates"]["answer_moved"]["measured"] is True


def test_a_report_that_cannot_supply_the_baseline_says_so_instead_of_passing():
    # An unmeasurable floor must not read as a cleared one.
    report = stored_report()
    for row in report["rows"]:
        row.pop("clean_target_logodds", None)
    for item in report["items"]:
        item.pop("clean_target_logodds", None)
    rescored = rescore_report(report)
    assert rescored["gates"]["answer_moved"]["measured"] is False
    assert rescored["gates"]["answer_moved"]["passes"] is False


# --------------------------------------------------------------------------
# the floor tests the argmax, and falls back to the majority only when blind
# --------------------------------------------------------------------------
#
# The floor was first written as "the clean answer no longer holds a majority
# of the readout", justified as the largest threshold that is not a free
# parameter because below a half the clean answer cannot still be the argmax.
# That justification is false. A half is the majority boundary, not the argmax
# boundary: a digit at 0.40 beats nine others averaging 0.067. The tightest
# scalar-only sufficient condition for ten classes is a share below 0.1.
#
# The share is exact in both tails -- at or above 0.5 the clean digit is
# necessarily still the argmax, below 0.1 it necessarily is not -- and can only
# err by over-calling movement in between. On the stored runs 37 of 360 ancestor
# rows sit in that band and 5 of them are called moved while the clean digit is
# still on top.
#
# Where the run stored `probs_patched`, none of this needs deciding: test the
# argmax. The share survives only as the fallback for the archived reports,
# which predate the field, and the gate records which test it ran.


def argmax_rows(layer, *, clean_share, stored=True, **kwargs):
    """Floor rows carrying a real ten-way distribution for the ancestor edit."""
    probs = spread(clean_share)
    return floor_rows(
        layer,
        # `delta_away` and the clean baseline are kept consistent with the
        # distribution, so the two tests disagree only where they genuinely do.
        away=math.log(clean_share) - math.log(0.98), clean=math.log(0.98),
        probs_patched=probs if stored else None,
        clean_value=3 if stored else None,
        **kwargs,
    )


def test_the_floor_tests_the_argmax_when_the_distribution_is_stored():
    # 0.45 is below the majority line but far above the other nine digits, so
    # the answer did not move. The majority rule would call this five for five.
    gates = evaluate_gates([argmax_rows(0, clean_share=0.45)] * 5, [0])
    assert gates["answer_moved"]["per_layer"][0]["moved_items"] == 0
    assert gates["answer_moved"]["passes"] is False
    assert verdict(gates) == "scientific negative"


def test_the_majority_fallback_is_what_a_blind_report_gets():
    # Same share, no distribution: the fallback over-calls it, which is the
    # error the archived reports are exposed to and cannot be rescued from.
    gates = evaluate_gates([argmax_rows(0, clean_share=0.45, stored=False)] * 5,
                           [0])
    assert gates["answer_moved"]["per_layer"][0]["moved_items"] == 5


def test_the_gate_records_which_test_decided_each_layer():
    stored = evaluate_gates([argmax_rows(0, clean_share=0.45)] * 5, [0])
    blind = evaluate_gates([argmax_rows(0, clean_share=0.45, stored=False)] * 5,
                           [0])
    assert stored["answer_moved"]["test"] == "argmax"
    assert blind["answer_moved"]["test"] == "majority"


def test_the_two_tests_agree_wherever_the_share_is_decisive():
    # Above a half and below a tenth the share settles the argmax on its own.
    for share in (0.7, 0.05):
        stored = evaluate_gates([argmax_rows(0, clean_share=share)] * 5, [0])
        blind = evaluate_gates(
            [argmax_rows(0, clean_share=share, stored=False)] * 5, [0])
        assert (stored["answer_moved"]["per_layer"][0]["moved_items"]
                == blind["answer_moved"]["per_layer"][0]["moved_items"])


def test_a_tie_at_the_top_does_not_count_as_having_moved_the_answer():
    # Two digits share the maximum, one of them the clean answer. Which one a
    # bare argmax returns is an artefact of digit order, so a tie is not a move.
    probs = [0.0] * 10
    probs[3] = probs[7] = 0.5
    rows = floor_rows(0, away=math.log(0.5), clean=0.0,
                      probs_patched=probs, clean_value=3)
    gates = evaluate_gates([rows] * 5, [0])
    assert gates["answer_moved"]["per_layer"][0]["moved_items"] == 0


def test_the_clean_digit_is_recovered_from_the_item_summary_on_rescore():
    # A run stored before `clean_value` existed still has the distribution and
    # an item summary naming the target, so the argmax test is a join away.
    report = stored_report()
    for row in report["rows"]:
        if row["kind"] == "ancestor":
            row["probs_patched"] = spread(0.45)
            row.pop("clean_value", None)
    rescored = rescore_report(report)
    assert rescored["gates"]["answer_moved"]["test"] == "argmax"
    assert rescored["gates"]["answer_moved"]["passes"] is False


# --------------------------------------------------------------------------
# is the clean answer the model's own answer?
# --------------------------------------------------------------------------
#
# `v3_distinct` made the three competing digits distinct. It did not make the
# model's clean behaviour correct: in the fresh depth-1 ladder the clean top
# digit disagrees with the target on 1/5, 1/5 and 2/5 items, and three
# observations are exact top ties. "The patch moved the answer off the clean
# target" is only a counterfactual flip when the clean target was the answer.
#
# Reported, never applied. Binding the verdict to it would be a third
# retroactive policy move on runs that are already scored; the fix belongs in
# the generator of the next family, not in a rescore of this one.


def clean_probs_for(top, share=0.6):
    return spread(share, clean_value=top)


def report_with_clean(tops, share=0.6):
    report = stored_report()
    report["items"] = [
        {"target_value": 3, "clean_target_logodds": -0.02,
         "clean_top_digit": top, "clean_digit_mass": 1.0,
         "clean_probs": clean_probs_for(top, share)}
        for top in tops
    ]
    return report


def test_the_diagnostic_counts_items_whose_clean_top_digit_is_the_target():
    rescored = rescore_report(report_with_clean([3, 3, 3, 2, 3]))
    gate = rescored["gates"]["clean_answer"]
    assert gate["n_items"] == 5
    assert gate["n_correct"] == 4


def test_a_tie_at_the_top_of_the_clean_readout_is_not_a_unique_answer():
    report = report_with_clean([3, 3, 3, 3, 3])
    tied = [0.0] * 10
    tied[3] = tied[8] = 0.5
    report["items"][0]["clean_probs"] = tied
    gate = rescore_report(report)["gates"]["clean_answer"]
    assert gate["n_tied"] == 1
    assert gate["n_unique_correct"] == 4


def test_the_clean_answer_diagnostic_never_decides_the_verdict():
    good = rescore_report(report_with_clean([3, 3, 3, 3, 3]))
    bad = rescore_report(report_with_clean([2, 2, 2, 2, 2]))
    assert bad["gates"]["clean_answer"]["n_correct"] == 0
    assert bad["gates"]["clean_answer"]["applied_to_verdict"] is False
    assert bad["verdict"] == good["verdict"]


def test_the_diagnostic_says_it_is_unmeasured_without_a_clean_distribution():
    # The archived reports store `clean_top_digit` but no distribution, so
    # correctness is knowable there and uniqueness is not.
    gate = rescore_report(stored_report())["gates"]["clean_answer"]
    assert gate["measured"] is True
    assert gate["n_tied"] is None


# --------------------------------------------------------------------------
# which verdict function produced this verdict
# --------------------------------------------------------------------------
#
# `gate_policy_version` names the surface-control policy, not the verdict
# function. Adding the floor changed the verdict function while leaving that
# label alone, so `paired_ladder/depth2_gap0.json` reads `v2_one_sided` and
# `positive` on disk while a rescore under the same label calls it a scientific
# negative. One name, two functions. The version below is what tells them apart.


def test_the_report_records_which_verdict_function_scored_it():
    from dag.dag_patching import VERDICT_VERSION

    rescored = rescore_report(stored_report())
    assert rescored["verdict_version"] == VERDICT_VERSION


def test_a_report_scored_before_the_floor_keeps_its_old_verdict_labelled():
    # The stored verdict and the version that produced it both survive the
    # rescore, so a changed verdict is attributable rather than mysterious.
    report = stored_report(verdict_str="positive")
    rescored = rescore_report(report)
    assert rescored["original_verdict"] == "positive"
    assert rescored["original_verdict_version"] == "v1_gap_only"


# --------------------------------------------------------------------------
# specificity: did the ancestor land somewhere, or did everything move?
# --------------------------------------------------------------------------
#
# `answer_moved` catches an arm where nothing moves. The written-versus-omitted
# run produced the mirror image: with the intermediate results unwritten, the
# model stops solving the task, the clean readout goes nearly flat, and *every*
# edit flips the argmax -- nulls 18/30 and a comment-tag rewrite 3/5, at
# `depth2_omitted`, which the scorer calls positive. Every existing gate is
# relative, so a background that moves as much as the ancestor does passes them.
#
# The separating statistic is not "did the answer move" but "did it land on the
# digit the edit predicts, against a background that did not move". Reported,
# never binding: control flips are not unique to the broken arm -- the published
# depth-1 positives have them too, at a lower rate -- so making this a gate
# would be a retroactive policy move on evidence that does not yet support one.


def specificity_rows(layer, *, ancestor_probs, control_probs, clean=3,
                     implied=7):
    rows = floor_rows(layer, away=-5.0, probs_patched=ancestor_probs,
                      clean_value=clean)
    rows[0]["implied_value"] = implied
    for row in rows[1:]:
        row |= {"probs_patched": list(control_probs), "clean_value": clean}
    return rows


def test_the_diagnostic_counts_controls_that_flip_the_answer():
    quiet = spread(0.9)
    loud = spread(0.05)
    gates = evaluate_gates(
        [specificity_rows(0, ancestor_probs=spread(0.9, clean_value=7),
                          control_probs=loud)] * 5, [0])
    entry = gates["control_specificity"]["per_layer"][0]
    assert entry["control_moved"] == entry["n_control"]

    gates = evaluate_gates(
        [specificity_rows(0, ancestor_probs=spread(0.9, clean_value=7),
                          control_probs=quiet)] * 5, [0])
    assert gates["control_specificity"]["per_layer"][0]["control_moved"] == 0


def test_the_diagnostic_counts_ancestors_landing_on_the_implied_digit():
    # Off the clean answer is not the same as onto the predicted one.
    onto = evaluate_gates(
        [specificity_rows(0, ancestor_probs=spread(0.9, clean_value=7),
                          control_probs=spread(0.9))] * 5, [0])
    elsewhere = evaluate_gates(
        [specificity_rows(0, ancestor_probs=spread(0.9, clean_value=1),
                          control_probs=spread(0.9))] * 5, [0])
    assert onto["control_specificity"]["per_layer"][0]["ancestor_implied"] == 5
    assert elsewhere["control_specificity"]["per_layer"][0]["ancestor_implied"] == 0
    assert elsewhere["control_specificity"]["per_layer"][0]["ancestor_moved"] == 5


def test_the_specificity_diagnostic_never_decides_the_verdict():
    gates = evaluate_gates(
        [specificity_rows(0, ancestor_probs=spread(0.9, clean_value=7),
                          control_probs=spread(0.05))] * 5, [0])
    assert gates["control_specificity"]["applied_to_verdict"] is False
    assert verdict(gates) == "positive"


def test_the_specificity_diagnostic_needs_the_distributions():
    # The archived eight store none, so this cannot be checked on them at all.
    gates = evaluate_gates([floor_rows(0, away=-5.0)] * 5, [0])
    assert gates["control_specificity"]["measured"] is False
