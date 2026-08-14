"""Layer bins and gate logic for the DAG patching prototype.

Torch-free by design: ``dag_patching`` imports torch lazily inside the functions
that need it, so the part that decides positive / scientific negative / invalid
test can be tested anywhere. The hook mechanics live in
``test_dag_patching_hooks.py`` and need torch.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dag_patching import (
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
              surface_tv, null_tvs, mass_ratio=1.0):
    rows = [{
        "kind": "ancestor", "node": "a", "layer": layer, "distance_to_read": 10,
        "tv": ancestor_tv, "delta_toward": ancestor_toward, "delta_away": -1.0,
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
    return rows


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
                  verdict_str="positive"):
    """A report laid out the way ``measure_item`` flattens it: edit-major.

    ``measure_item`` loops ``for edit: for layer:``, so one item's block is nine
    consecutive edit groups of four layer rows each -- not the kind-major order
    ``make_rows`` produces for the gate tests.
    """
    kinds = ["ancestor", "non_ancestor", "surface_null"] + ["null"] * 6
    tvs = {"ancestor": 0.8, "non_ancestor": 0.05, "surface_null": surface_tv,
           "null": 0.03}
    rows = []
    for _ in range(n_items):
        for kind in kinds:
            for layer in bins:
                rows.append({
                    "kind": kind, "node": None, "layer": layer,
                    "distance_to_read": 11, "tv": tvs[kind],
                    "delta_toward": 2.0 if kind == "ancestor" else 0.0,
                    "delta_away": 0.0,
                    "digit_mass_clean": 1.0, "digit_mass_patched": 1.0,
                })
    return {
        "model": "test", "condition": "both", "n_items": n_items, "seed": 0,
        "layer_bins": list(bins), "n_layers": 28, "rows": rows,
        "verdict": verdict_str,
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
