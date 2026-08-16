"""Invariants of the E3 ladder reading.

The module turns arm files into the one table the depth claim is read off, so a
silent regrouping here would not raise -- it would produce a plausible number.
Three things are therefore pinned rather than trusted: the layer is stage B's,
the two chain lines of a depth-3 item stay apart, and the within-item contrast
pairs an ancestor with a chain line of the *same* item.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dag import dag_e3_ladder
from dag.dag_e3_ladder import (
    LAYER,
    build,
    by_distance_band,
    by_site,
    records,
    sign_test,
    within_item,
)

BINS = (6, 13, 20, 27)


def probs(top, *, share=0.9, n=10):
    rest = (1.0 - share) / (n - 1)
    return [share if d == top else rest for d in range(n)]


def arm(depth, *, n_items=2, seed=0, gap=0, lands_at_steps=(1,), clean_top=3):
    """One report in `measure_item`'s edit-major, layer-minor row layout.

    The ancestor is `depth` steps from the target and each chain line one fewer,
    which is the geometry the real generator produces and the only thing the
    grouping here depends on. ``lands_at_steps`` says which sites move the
    readout onto the implied digit; the default is the one the campaign measured,
    where a site lands if and only if it is one step from the target.
    """
    chain_nodes = ["m", "n"][-(depth - 1):] if depth > 1 else []
    sites = [("ancestor", None, depth, 12 * depth)]
    sites += [("chain", node, depth - 1 - index, 12 * (depth - 1 - index))
              for index, node in enumerate(chain_nodes)]
    rows = []
    for _ in range(n_items):
        for kind, node, steps, distance in sites:
            lands = steps in lands_at_steps
            for layer in BINS:
                rows.append({
                    "kind": kind, "node": node, "layer": layer,
                    "steps_to_target": steps, "distance_to_read": distance,
                    "implied_value": 7, "raw_value": 5, "clean_value": clean_top,
                    "tv": 0.9 if lands else 0.02,
                    "delta_toward": 8.0 if lands else 1.0,
                    "delta_toward_raw": 1.0,
                    "delta_away": -5.0 if lands else 0.0,
                    "clean_target_logodds": -0.02,
                    "digit_mass_clean": 1.0, "digit_mass_patched": 1.0,
                    "probs_patched": probs(7 if lands else clean_top),
                })
    return {
        "model": "test", "condition": "both", "generator": "v3_distinct",
        "depth": depth, "gap": [gap] * n_items, "seed": seed,
        "n_items": n_items, "layer_bins": list(BINS), "n_layers": 28,
        "omit": "none", "chain_edits": depth > 1, "rows": rows,
        "verdict": "scientific negative",
        "gate_policy_version": "v2_one_sided",
        "gates": {"scoring_layers": [6, 13, 20],
                  "surface_v2_one_sided": {"per_layer": {
                      "6": {"surface_items": 1}, "13": {"surface_items": 2},
                      "20": {"surface_items": 2}}}},
        "scoring": {"v2_one_sided": {"invalid_reasons": []}},
        "items": [{"target_value": clean_top, "clean_probs": probs(clean_top),
                   "clean_top_digit": clean_top, "clean_target_logodds": -0.02,
                   "clean_digit_mass": 1.0}
                  for _ in range(n_items)],
    }


@pytest.fixture
def ladder(tmp_path):
    for depth in (1, 2, 3):
        (tmp_path / f"depth{depth}_gap0_seed0.json").write_text(
            json.dumps(arm(depth)))
    return tmp_path


# --------------------------------------------------------------------------
# the inherited reading
# --------------------------------------------------------------------------


def test_the_layer_is_stage_bs_and_the_pooled_tables():
    # Three copies of one number. A change to any of them that left the others
    # alone would silently read a different depth of the residual stream than
    # the comparison these numbers are meant to be comparable to.
    from dag.dag_pooling import POOLED_LAYER
    from dag.dag_stage_b import LAYER as STAGE_B_LAYER

    assert LAYER == STAGE_B_LAYER == POOLED_LAYER == 13


# --------------------------------------------------------------------------
# the paired test
# --------------------------------------------------------------------------


def test_no_discordant_pairs_is_no_evidence_rather_than_an_error():
    # The depth-3 ancestor against the two-step chain line: both dead, nothing
    # separating them. Reporting that as significant either way would be wrong.
    assert sign_test(0, 0) == 1.0


def test_a_perfect_split_is_two_over_two_to_the_n():
    assert sign_test(5, 0) == pytest.approx(2 / 2 ** 5)
    assert sign_test(144, 0) == pytest.approx(8.97e-44, rel=1e-2)


def test_an_even_split_is_not_significant():
    assert sign_test(3, 3) == 1.0


def test_the_test_is_symmetric_in_its_two_directions():
    assert sign_test(9, 2) == sign_test(2, 9)


# --------------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------------


def test_only_the_sites_on_the_path_to_the_target_are_read():
    # The controls are the gates' business. Sweeping a null or a surface edit in
    # here would put a row with no step count into a table keyed on step count.
    report = arm(2, n_items=1)
    for row in report["rows"]:
        if row["kind"] == "ancestor":
            row["kind"] = "null"
            row["steps_to_target"] = None
    kinds = {record["kind"] for record in records(report, "depth2_gap0_seed0")}
    assert kinds == {"chain"}


def test_a_record_is_made_per_item_and_patch_site_at_one_layer():
    got = records(arm(3, n_items=4), "depth3_gap0_seed0")
    # Four items x (one ancestor + two chain lines), and one layer of the four.
    assert len(got) == 4 * 3
    assert all(record["arm"] == "depth3_gap0_seed0" for record in got)
    assert {record["item"] for record in got} == {0, 1, 2, 3}


def test_the_two_chain_lines_of_a_depth_three_item_stay_apart():
    # They are two steps and one step from the target. Collapsing them would
    # average the cell that answers the question with the cell that does not.
    rows = by_site(records(arm(3), "depth3_gap0_seed0"))
    chain = [row for row in rows if row["kind"] == "chain"]
    assert len(chain) == 2
    assert sorted(row["steps"] for row in chain) == [1, 2]
    assert len({row["node"] for row in chain}) == 2


def test_the_ancestor_carries_the_arms_depth_as_its_step_count():
    for depth in (1, 2, 3):
        rows = by_site(records(arm(depth), f"depth{depth}_gap0_seed0"))
        ancestor = next(row for row in rows if row["kind"] == "ancestor")
        assert ancestor["steps"] == depth


def test_rates_are_taken_only_over_items_whose_clean_answer_was_alone_on_top():
    report = arm(2, n_items=2)
    # Make one item's clean readout a tie, which is not an unambiguous answer
    # for a patch to move off.
    report["items"][0]["clean_probs"] = [0.5 if d in (3, 4) else 0.0
                                         for d in range(10)]
    got = records(report, "depth2_gap0_seed0")
    assert sum(record["eligible"] for record in got) == 2  # one item, two sites
    assert all(row["n"] == 1 for row in by_site(got))


# --------------------------------------------------------------------------
# the two tables the claim is read off
# --------------------------------------------------------------------------


def test_a_distance_band_splits_by_step_count():
    got = by_distance_band(records(arm(3), "depth3_gap0_seed0"),
                           bands=((0, 60),))
    assert {row["steps"] for row in got} == {1, 2, 3}
    landed = {row["steps"]: row["implied_top_unique"] for row in got}
    assert landed[1] and not landed[2] and not landed[3]


def test_a_band_holding_no_site_contributes_no_row():
    got = by_distance_band(records(arm(2), "depth2_gap0_seed0"),
                           bands=((0, 5), (0, 60)))
    assert all(row["band"] == [0, 60] for row in got)


def test_the_within_item_contrast_pairs_sites_of_the_same_item():
    got = within_item(records(arm(2, n_items=6), "depth2_gap0_seed0"))
    assert len(got) == 1
    row = got[0]
    assert row["n"] == 6
    assert row["chain_only"] == 6 and row["ancestor_only"] == 0
    assert row["both"] == row["neither"] == 0


def test_two_dead_sites_are_reported_as_no_difference():
    # Depth 3's ancestor against its two-step chain line, both inert.
    got = within_item(records(arm(3), "depth3_gap0_seed0"))
    two_step = next(row for row in got if row["chain_steps"] == 2)
    assert two_step["neither"] == two_step["n"]
    assert two_step["p"] == 1.0


def test_an_arm_with_no_chain_line_yields_no_within_item_contrast():
    assert within_item(records(arm(1), "depth1_gap0_seed0")) == []


# --------------------------------------------------------------------------
# the verdicts, and why they are reported rather than applied
# --------------------------------------------------------------------------


def test_each_arm_reports_the_quorum_that_decided_it():
    built = build(_written(arm(2, n_items=48)))
    entry = built["arm_verdicts"][0]
    assert entry["n_items"] == 48
    assert entry["quorum"] == 47  # "all but one" -> 97.9% at this N
    assert entry["surface_by_layer"] == {"6": 1, "13": 2, "20": 2}


def test_the_payload_says_chain_rows_gate_nothing():
    built = build(_written(arm(2)))
    assert built["chain_rows_gate_nothing"] is True


def _written(report, tmp=None):
    import tempfile
    directory = Path(tmp or tempfile.mkdtemp())
    (directory / f"depth{report['depth']}_gap0_seed0.json").write_text(
        json.dumps(report))
    return directory


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def test_build_reads_every_arm_in_the_directory(ladder):
    built = build(ladder)
    assert len(built["arm_verdicts"]) == 3
    assert built["n_items"] == 6  # 2 items in each of 3 arms
    assert built["n_sites"] == 2 * (1 + 2 + 3)


def test_the_analysis_file_is_not_read_back_in_as_an_arm(ladder):
    built = build(ladder)
    (ladder / "ANALYSIS.json").write_text(json.dumps(built) + "\n")
    assert build(ladder)["n_sites"] == built["n_sites"]


def test_an_empty_directory_is_refused_rather_than_summarised(tmp_path):
    with pytest.raises(FileNotFoundError, match="no arm files"):
        build(tmp_path)


def test_the_summary_prints_without_raising(ladder, capsys):
    dag_e3_ladder.print_summary(build(ladder))
    out = capsys.readouterr().out
    assert "Steps or tokens?" in out
    assert "Reported, not applied here." in out
