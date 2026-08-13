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
    digit_token_ids,
    evaluate_gates,
    layer_bins,
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


def test_surface_edit_outside_the_null_spread_does_not_flip_the_verdict():
    gates = evaluate_gates(item_rows(
        ancestor_toward=2.0, ancestor_tv=0.8, non_ancestor_tv=0.05,
        surface_tv=0.60, null_tvs=[0.02, 0.03, 0.05, 0.04, 0.03, 0.02],
    ), [0])
    assert gates["surface_inside_null_spread"]["passes"] is False
    assert verdict(gates) == "positive"


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
