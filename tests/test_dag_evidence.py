"""Provenance and offline reconstruction of the archived DAG patching runs.

The first three artifacts predate the ``depth``/``gap``/``ancestor_distance``
fields. Those values are recoverable because generation is deterministic and
tokenizer-only, but a reconstruction that silently drifted from the run it
claims to describe would be worse than no reconstruction at all. These tests
pin the acceptance check, not the convenience.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dag_evidence import (
    RUN_COMMITS,
    SCHEMA_INTRODUCED,
    derive_v0_fields,
    inferred_fields,
    reconstruct_items,
    resolve_run_commit,
    schema_version,
    sha256_file,
    verdict_table,
    verify_reconstruction,
)
from test_dag_tasks import char_encode


def archived_report(*, n_items=3, n_decoys=6, seed=0, condition="both",
                    bins=(6, 13, 20, 27), n_layers=28):
    """A report built from real generated items, laid out as a run stores it."""
    items = reconstruct_items(
        {"n_items": n_items, "seed": seed, "condition": condition,
         "n_decoys": n_decoys},
        char_encode,
    )
    rows = []
    for item in items:
        for edit in item.edits:
            for layer in bins:
                rows.append({
                    "kind": edit.kind, "node": edit.node, "layer": layer,
                    "distance_to_read": edit.distance_to_read, "tv": 0.5,
                    "delta_toward": 2.0, "delta_away": -1.0,
                    "digit_mass_clean": 1.0, "digit_mass_patched": 1.0,
                })
    return {
        "model": "test", "condition": condition, "n_items": n_items,
        "seed": seed, "layer_bins": list(bins), "n_layers": n_layers,
        "n_tokens": [len(item.token_ids) for item in items],
        "items": [{"target_value": item.target_value, "clean_top_digit": 0,
                   "clean_target_logodds": 0.0, "clean_digit_mass": 1.0}
                  for item in items],
        "rows": rows, "verdict": "positive",
    }


# --------------------------------------------------------------------------
# schema detection
# --------------------------------------------------------------------------


def test_a_report_without_depth_and_gap_is_schema_v0():
    assert schema_version(archived_report()) == "v0"


def test_a_report_carrying_depth_and_gap_is_schema_v1():
    report = archived_report()
    report.update(depth=1, gap=[0, 0, 0], ancestor_distance=[11, 11, 11])
    assert schema_version(report) == "v1"


# --------------------------------------------------------------------------
# reconstruction acceptance
# --------------------------------------------------------------------------


def test_reconstruction_matches_the_archived_measurements():
    report = archived_report()
    items = reconstruct_items(report, char_encode)
    assert verify_reconstruction(report, items)["matches"] is True


def test_reconstruction_is_rejected_when_the_token_count_differs():
    report = archived_report()
    items = reconstruct_items(report, char_encode)
    report["n_tokens"][0] += 1
    with pytest.raises(ValueError, match="token count"):
        verify_reconstruction(report, items)


def test_reconstruction_is_rejected_when_an_edit_distance_differs():
    # distance_to_read pins the edited positions against the read position, so a
    # generator change that moved a line would surface here.
    report = archived_report()
    items = reconstruct_items(report, char_encode)
    report["rows"][0]["distance_to_read"] += 1
    with pytest.raises(ValueError, match="distance_to_read"):
        verify_reconstruction(report, items)


def test_reconstruction_is_rejected_when_the_edit_order_differs():
    report = archived_report()
    items = reconstruct_items(report, char_encode)
    n_bins = len(report["layer_bins"])
    for row in report["rows"][:n_bins]:
        row["kind"] = "null"
    with pytest.raises(ValueError, match="edit kind"):
        verify_reconstruction(report, items)


def test_reconstruction_is_rejected_when_a_target_value_differs():
    report = archived_report()
    items = reconstruct_items(report, char_encode)
    report["items"][0]["target_value"] = (
        report["items"][0]["target_value"] + 1) % 10
    with pytest.raises(ValueError, match="target value"):
        verify_reconstruction(report, items)


def test_reconstruction_is_rejected_when_the_seed_differs():
    report = archived_report(seed=0)
    items = reconstruct_items({**report, "seed": 1}, char_encode)
    with pytest.raises(ValueError):
        verify_reconstruction(report, items)


# --------------------------------------------------------------------------
# derived fields
# --------------------------------------------------------------------------


def test_derived_fields_are_produced_only_after_verification():
    report = archived_report()
    items = reconstruct_items(report, char_encode)
    derived = derive_v0_fields(report, items)
    assert derived["depth"] == 1
    assert len(derived["gap"]) == report["n_items"]
    assert len(derived["ancestor_distance"]) == report["n_items"]
    assert derived["derived"] is True


def test_the_earliest_report_predates_condition_and_defaults_to_both():
    report = archived_report(condition="both")
    del report["condition"]
    items = reconstruct_items(report, char_encode)
    assert verify_reconstruction(report, items)["matches"] is True


def test_a_report_that_never_recorded_its_condition_lists_it_as_inferred():
    # The donor condition changes only the donor text. Positions, token count,
    # target value, and every distance are identical across conditions, so the
    # reconstruction cannot confirm it -- it comes from the run order and the
    # log. Recording it as derived would overstate what was checked.
    report = archived_report()
    del report["condition"]
    assert "condition" in inferred_fields(report)


def test_a_report_that_does_record_its_condition_does_not_list_it_as_inferred():
    # Only `feasibility.json` predates the field. Marking it inferred on the two
    # runs that state it outright understates the provenance we have.
    assert "condition" not in inferred_fields(archived_report())


def test_every_report_lists_its_unrecorded_decoy_count_as_inferred():
    # No artifact records n_decoys, v1 included -- so the flag is not a v0
    # concern and must not be attached only to the reports that get a v0
    # derivation block.
    v1 = archived_report()
    v1.update(depth=1, gap=[0], ancestor_distance=[11])
    assert "n_decoys" in inferred_fields(v1)
    assert "depth" not in inferred_fields(v1)


def test_derived_fields_carry_only_what_the_reconstruction_checked():
    # condition and n_decoys are provenance, not derivation; they belong to the
    # manifest entry, not to the block whose contract is "verified against the
    # archived measurement".
    report = archived_report()
    items = reconstruct_items(report, char_encode)
    assert "condition" not in derive_v0_fields(report, items)


def test_derived_fields_refuse_to_run_on_a_mismatched_reconstruction():
    report = archived_report()
    items = reconstruct_items(report, char_encode)
    report["n_tokens"][0] += 1
    with pytest.raises(ValueError):
        derive_v0_fields(report, items)


def test_derived_ancestor_distance_agrees_with_the_archived_rows():
    report = archived_report()
    items = reconstruct_items(report, char_encode)
    derived = derive_v0_fields(report, items)
    n_bins = len(report["layer_bins"])
    per_item = len(report["rows"]) // report["n_items"]
    for index, distance in enumerate(derived["ancestor_distance"]):
        block = report["rows"][index * per_item:(index + 1) * per_item]
        ancestor = next(row for row in block if row["kind"] == "ancestor")
        assert ancestor["distance_to_read"] == distance


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# which commit produced each run
#
# No report records it. It is recovered from the artifact's mtime, bracketed
# against the commit timeline, and corroborated independently by which schema
# fields the report carries. mtime does not survive a clone, so the resolved
# values are frozen in `RUN_COMMITS` and the recovery is a check on them.
# --------------------------------------------------------------------------

TIMELINE = [("ccc", 300), ("bbb", 200), ("aaa", 100)]  # newest first


def test_the_run_commit_is_the_newest_commit_older_than_the_artifact():
    assert resolve_run_commit(250, TIMELINE, archive_time=1000) == "bbb"
    assert resolve_run_commit(150, TIMELINE, archive_time=1000) == "aaa"


def test_an_artifact_written_by_the_commit_itself_resolves_to_that_commit():
    assert resolve_run_commit(200, TIMELINE, archive_time=1000) == "bbb"


def test_an_mtime_from_a_fresh_checkout_resolves_to_nothing():
    # After a clone every file's mtime is checkout time. Returning the newest
    # commit there would confidently name the wrong one.
    assert resolve_run_commit(1200, TIMELINE, archive_time=1000) is None


def test_an_artifact_older_than_every_commit_resolves_to_nothing():
    assert resolve_run_commit(50, TIMELINE, archive_time=1000) is None


def test_every_frozen_run_commit_agrees_with_the_schema_its_report_carries():
    # A report carrying a field cannot predate the commit that added it, and a
    # report missing that field cannot postdate it. This is the second,
    # mtime-independent signal -- it must agree with the frozen mapping.
    import subprocess

    def commit_time(short):
        return int(subprocess.run(
            ["git", "show", "-s", "--format=%ct", short],
            capture_output=True, text=True, check=True).stdout.strip())

    directory = Path(__file__).resolve().parents[1] / "results/dag_patching"
    for name, commit in RUN_COMMITS.items():
        report = json.loads((directory / f"{name}.json").read_text())
        run_at = commit_time(commit)
        for field, introduced in SCHEMA_INTRODUCED.items():
            added_at = commit_time(introduced)
            if field in report:
                assert run_at >= added_at, f"{name} has {field} but predates it"
            else:
                assert run_at < added_at, f"{name} lacks {field} but postdates it"


def test_sha256_is_stable_and_content_addressed(tmp_path):
    one = tmp_path / "a.json"
    two = tmp_path / "b.json"
    one.write_text('{"x": 1}')
    two.write_text('{"x": 1}')
    assert sha256_file(one) == sha256_file(two)
    two.write_text('{"x": 2}')
    assert sha256_file(one) != sha256_file(two)


def test_verdict_table_reports_both_policies_and_the_reason_for_a_change():
    report = archived_report()
    # Make the surface edit louder than every null edit at a scoring layer, so
    # the active policy flips the archived verdict.
    for row in report["rows"]:
        if row["kind"] == "surface_null":
            row["tv"] = 0.9
        elif row["kind"] == "null":
            row["tv"] = 0.01
    row_table = verdict_table({"loud": report})
    entry = row_table[0]
    assert entry["name"] == "loud"
    assert entry["original_verdict"] == "positive"
    assert entry["v2_one_sided"] == "invalid test"
    assert entry["changed"] is True
    assert "surface_above_null" in entry["reason"]
