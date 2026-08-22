"""Pooling the per-item patching outcomes across arms.

Every arm holds five items and every README reports its arm alone, so the
strongest count written down anywhere is ``5/5``. Pooling is only worth doing if
it counts *measurements* rather than *files*: the same item is stored in up to
four arms, and a pool that double-counts those would inflate the one result the
project still stands on. These tests pin the identity rule, the argmax
criterion, and what happens to a measurement the arm cannot support.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dag.dag_pooling import (
    CONFIDENCE_BANDS,
    DONOR_KINDS,
    band_table,
    measurement_id,
    outcomes,
    pool,
    summarize,
)

BINS = (6, 13, 20, 27)


def spread(top, value, n=10):
    """A ten-digit readout whose argmax is ``value`` with share ``top``."""
    rest = (1.0 - top) / (n - 1)
    return [top if digit == value else rest for digit in range(n)]


def tied(low, high, top=0.45, n=10):
    """A readout where two digits hold the maximum at exactly equal shares.

    Not a contrived case: the model runs in bfloat16, so the digit logits sit
    on a 0.125-nat grid and eight of the fifty-three depth-1 patched readouts
    have two digits at bit-identical probability.
    """
    rest = (1.0 - 2 * top) / (n - 2)
    return [top if digit in (low, high) else rest for digit in range(n)]


def block(kind, *, implied, raw, target, tv=0.9, toward=5.0, toward_raw=1.0,
          probs=None, layers=BINS):
    return [{
        "kind": kind, "node": "a", "layer": layer, "tv": tv,
        "implied_value": implied, "raw_value": raw,
        "delta_toward": toward, "delta_toward_raw": toward_raw,
        "delta_away": -3.0, "distance_to_read": 24,
        "probs_patched": probs if probs is not None else spread(0.9, implied),
    } for layer in layers]


def report(*items, depth=1, seed=0, generator="v3_distinct", kinds=("ancestor",),
           **item_kwargs):
    """One arm: ``items`` is a list of (target, clean_top, clean_share) tuples."""
    summaries, rows = [], []
    for target, clean_top, clean_share in items:
        summaries.append({
            "target_value": target, "clean_top_digit": clean_top,
            "clean_probs": spread(clean_share, clean_top),
        })
        for kind in kinds:
            rows.extend(block(kind, target=target, **item_kwargs))
    return {
        "generator": generator, "depth": depth, "seed": seed,
        "n_items": len(summaries), "layer_bins": list(BINS),
        "items": summaries, "rows": rows,
    }


ONE = ((3, 3, 0.7),)


# --------------------------------------------------------------------------
# One measurement is one measurement, however many files hold it
# --------------------------------------------------------------------------


def test_the_same_arm_stored_twice_pools_to_one_measurement():
    arm = report(*ONE, implied=5, raw=8)
    pooled = pool({"a": arm, "b": dict(arm)}, layer=13)
    assert len(pooled) == 1


def test_pooling_records_how_many_arms_carried_the_measurement():
    arm = report(*ONE, implied=5, raw=8)
    (record,) = pool({"a": arm, "b": dict(arm)}, layer=13)
    assert record["n_arms"] == 2
    assert record["arms"] == ["a", "b"]


def test_two_arms_measuring_different_items_both_survive():
    pooled = pool({"a": report(*ONE, implied=5, raw=8),
                   "b": report((4, 4, 0.7), implied=6, raw=9)}, layer=13)
    assert len(pooled) == 2


def test_the_identity_is_the_measurement_not_the_arm_metadata():
    """Same numbers under a different seed is the same measurement.

    The seed is metadata; two runs that produced identical clean readouts and
    identical patched rows to float precision ran the same forward passes.
    """
    left = report(*ONE, implied=5, raw=8, seed=0)
    right = report(*ONE, implied=5, raw=8, seed=1)
    assert measurement_id(left, 0, "ancestor") == \
        measurement_id(right, 0, "ancestor")


def test_a_changed_measurement_is_a_different_measurement():
    left = report(*ONE, implied=5, raw=8, tv=0.90)
    right = report(*ONE, implied=5, raw=8, tv=0.91)
    assert measurement_id(left, 0, "ancestor") != \
        measurement_id(right, 0, "ancestor")


# --------------------------------------------------------------------------
# What is being counted: the argmax, and the two rivals it beat
# --------------------------------------------------------------------------


def test_landing_on_the_implied_digit_is_an_argmax_test():
    (record,) = outcomes(report(*ONE, implied=5, raw=8), layer=13)
    assert record["on_implied"] is True
    assert record["on_raw"] is False
    assert record["on_clean"] is False


def test_the_argmax_and_the_margin_are_reported_separately():
    """A patch can put the implied digit on top while moving the raw one more.

    Both are informative and they are not the same claim, so neither is allowed
    to stand in for the other.
    """
    (record,) = outcomes(
        report(*ONE, implied=5, raw=8, toward=2.0, toward_raw=6.0), layer=13)
    assert record["on_implied"] is True
    assert record["toward_implied_over_raw"] is False


def test_a_patch_that_copies_the_donor_digit_is_counted_as_raw():
    (record,) = outcomes(
        report(*ONE, implied=5, raw=8, probs=spread(0.9, 8)), layer=13)
    assert (record["on_implied"], record["on_raw"]) == (False, True)


def test_a_patch_that_left_the_clean_answer_on_top_is_counted_as_clean():
    (record,) = outcomes(
        report((3, 3, 0.7), implied=5, raw=8, probs=spread(0.9, 3)), layer=13)
    assert (record["on_implied"], record["on_clean"]) == (False, True)


def test_only_the_requested_layer_is_read():
    arm = report(*ONE, implied=5, raw=8)
    for row in arm["rows"]:
        if row["layer"] != 20:
            row["probs_patched"] = spread(0.9, 8)
    (record,) = outcomes(arm, layer=20)
    assert record["layer"] == 20 and record["on_implied"] is True


# --------------------------------------------------------------------------
# Ties, which the recorded precision cannot break and neither may we
# --------------------------------------------------------------------------


def test_a_patched_tie_between_implied_and_raw_is_not_a_win_for_either():
    """Digit order decided five of these in the first pooled table.

    ``probs.index(max(probs))`` returns the lowest tying digit, so the implied
    digit beat the raw one whenever it happened to be the smaller numeral. That
    is a fact about ``list.index``, not about the model.
    """
    (record,) = outcomes(
        report(*ONE, implied=2, raw=4, probs=tied(2, 4)), layer=13)
    assert record["implied_top_unique"] is False
    assert record["raw_top_unique"] is False
    assert record["implied_top_tied"] is True
    assert record["on_implied"] is True  # the legacy reading, kept visible


def test_a_tie_is_counted_and_not_folded_into_the_miss_column():
    pooled = pool({"a": report(*ONE, implied=2, raw=4, probs=tied(2, 4))},
                  layer=13)
    (group,) = summarize(pooled)
    assert group["n_clean_correct_unique"] == 1
    assert group["n_implied_top_unique"] == 0
    assert group["n_implied_tied"] == 1


def test_an_item_whose_clean_answer_is_tied_is_not_a_clean_correct_item():
    """A clean readout with two digits on top is no answer to move the model off.

    The stored ``clean_top_digit`` resolves it by digit order, so two of the
    thirty-three depth-1 items counted as clean-correct on a coin the model
    never flipped.
    """
    arm = report((3, 3, 0.7), implied=5, raw=8)
    arm["items"][0]["clean_probs"] = tied(3, 5)
    (record,) = outcomes(arm, layer=13)
    assert record["clean_correct"] is True  # by stored digit order
    assert record["clean_correct_unique"] is False


def test_a_tied_clean_item_is_reported_rather_than_quietly_dropped():
    arm = report((3, 3, 0.7), implied=5, raw=8)
    arm["items"][0]["clean_probs"] = tied(3, 5)
    (group,) = summarize(pool({"a": arm}, layer=13))
    assert (group["n_items"], group["n_clean_correct_unique"]) == (1, 0)
    assert group["n_clean_tied"] == 1


def test_a_tied_clean_item_is_outside_the_confidence_bands():
    arm = report((3, 3, 0.7), implied=5, raw=8)
    arm["items"][0]["clean_probs"] = tied(3, 5)
    assert sum(row["n_items"] for row in band_table(pool({"a": arm},
                                                         layer=13))) == 0


# --------------------------------------------------------------------------
# What the arm cannot support, it does not get to answer
# --------------------------------------------------------------------------


def test_an_item_without_a_stored_distribution_is_unmeasured_not_a_miss():
    arm = report(*ONE, implied=5, raw=8)
    for row in arm["rows"]:
        row.pop("probs_patched")
    (record,) = outcomes(arm, layer=13)
    assert record["measured"] is False
    assert record["on_implied"] is None


def test_unmeasured_items_are_excluded_from_the_rates_they_cannot_inform():
    arm = report(*ONE, implied=5, raw=8)
    for row in arm["rows"]:
        row.pop("probs_patched")
    (group,) = summarize(pool({"a": arm}, layer=13))
    assert (group["n_items"], group["n_measured"]) == (1, 0)
    assert group["n_on_implied"] == 0


def test_a_donor_kind_the_arm_never_ran_produces_no_record():
    pooled = pool({"a": report(*ONE, implied=5, raw=8)}, layer=13)
    assert {record["kind"] for record in pooled} == {"ancestor"}


def test_both_donor_kinds_are_pooled_when_the_arm_ran_both():
    pooled = pool({"a": report(*ONE, implied=5, raw=8, kinds=DONOR_KINDS)},
                  layer=13)
    assert {record["kind"] for record in pooled} == set(DONOR_KINDS)


def test_a_different_generator_is_a_different_family_and_is_left_out():
    pooled = pool({"a": report(*ONE, implied=5, raw=8, generator="v1_unpaired")},
                  layer=13, generator="v3_distinct")
    assert pooled == []


# --------------------------------------------------------------------------
# Grouping: clean correctness is a split, never a filter applied in silence
# --------------------------------------------------------------------------


def test_clean_correctness_compares_the_stored_top_digit_to_the_target():
    pooled = pool({"a": report((3, 3, 0.7), (4, 9, 0.7), implied=5, raw=8)},
                  layer=13)
    assert sorted(record["clean_correct"] for record in pooled) == [False, True]


def test_the_summary_splits_by_kind_and_depth_and_clean_correctness():
    pooled = pool({"a": report((3, 3, 0.7), (4, 9, 0.7), implied=5, raw=8),
                   "b": report((6, 6, 0.7), implied=2, raw=1, depth=2)},
                  layer=13)
    rates = {(g["kind"], g["depth"]): g for g in summarize(pooled)}
    assert rates[("ancestor", 1)]["n_items"] == 2
    assert rates[("ancestor", 1)]["n_clean_correct"] == 1
    assert rates[("ancestor", 1)]["n_on_implied_clean_correct"] == 1
    assert rates[("ancestor", 2)]["n_items"] == 1


def test_the_summary_reports_the_seeds_a_group_was_pooled_over():
    """Thirty-three rows over four seeds is not thirty-three independent draws.

    The count is meaningless without the clustering beside it, so the clustering
    travels in the same record.
    """
    pooled = pool({"a": report(*ONE, implied=5, raw=8, seed=0),
                   "b": report((4, 4, 0.7), implied=6, raw=9, seed=1)},
                  layer=13)
    (group,) = summarize(pooled)
    assert group["seeds"] == [0, 1]


# --------------------------------------------------------------------------
# The confidence bands, which are what the pooling was for
# --------------------------------------------------------------------------


def test_the_bands_cover_the_unit_interval_without_overlapping():
    assert CONFIDENCE_BANDS[0][0] == 0.0
    assert CONFIDENCE_BANDS[-1][1] == 1.0
    for (_, upper), (lower, _) in zip(CONFIDENCE_BANDS, CONFIDENCE_BANDS[1:]):
        assert upper == lower


def test_an_item_on_a_band_edge_falls_in_the_upper_band():
    lower, _ = CONFIDENCE_BANDS[1]
    pooled = pool({"a": report((3, 3, lower), implied=5, raw=8)}, layer=13)
    rows = [row for row in band_table(pooled) if row["n_items"]]
    assert [row["band"] for row in rows] == [CONFIDENCE_BANDS[1]]


def test_the_band_table_counts_only_clean_correct_items():
    """On a clean-incorrect item the patch has no clean answer to move off.

    Banding those by the target's share would be reading a confidence the model
    never expressed, so they are excluded here and counted in ``summarize``.
    """
    pooled = pool({"a": report((3, 9, 0.7), implied=5, raw=8)}, layer=13)
    assert sum(row["n_items"] for row in band_table(pooled)) == 0


def test_the_bands_are_taken_within_a_depth_and_never_across_depths():
    """Depth and clean confidence are collinear, so a pooled band lies.

    Depth-1 items top out below where the depth-2 items start. Band the two
    together and the top band is almost entirely depth-2 misses, which reads as
    the depth-1 effect decaying with confidence -- the exact artefact this table
    exists to rule out.
    """
    pooled = pool({"a": report((3, 3, 0.85), implied=5, raw=8, depth=1),
                   "b": report((4, 4, 0.95), implied=6, raw=9, depth=2,
                               probs=spread(0.9, 4))}, layer=13)
    top = {(row["depth"]): row for row in band_table(pooled)
           if row["band"] == CONFIDENCE_BANDS[-1]}
    assert top[1]["n_implied_top_unique"] == 1 and top[1]["n_items"] == 1
    assert top[2]["n_implied_top_unique"] == 0 and top[2]["n_items"] == 1


def test_a_flat_rate_across_bands_is_visible_as_such():
    pooled = pool({"a": report((3, 3, 0.3), (4, 4, 0.6), (5, 5, 0.9),
                               implied=7, raw=8)}, layer=13)
    rows = [row for row in band_table(pooled) if row["n_items"]]
    assert len(rows) == len(CONFIDENCE_BANDS)
    assert all(row["n_implied_top_unique"] == row["n_items"] for row in rows)


def test_pooling_an_empty_set_of_arms_is_empty_not_an_error():
    assert pool({}, layer=13) == []
    assert summarize([]) == []


def test_a_layer_no_arm_measured_is_refused_rather_than_silently_empty():
    with pytest.raises(ValueError, match="layer 99"):
        pool({"a": report(*ONE, implied=5, raw=8)}, layer=99)


def test_an_omitted_arm_is_never_pooled_into_the_written_rate():
    """The chain-omitted arms are a clean-behaviour ablation, not a patch test.

    They hit the pre-registered stop condition, so their implied-hit rate is not
    evidence about propagation at that depth. Merging them into the written
    arm's row would launder an invalid test into the headline number.
    """
    written = report((3, 3, 0.9), implied=5, raw=8, depth=2,
                     probs=spread(0.9, 3))
    omitted = {**report((4, 4, 0.2), implied=6, raw=9, depth=2), "omit": "chain"}
    rows = {row["omit"]: row for row in summarize(pool(
        {"a": written, "b": omitted}, layer=13))}
    assert rows["none"]["n_on_implied"] == 0
    assert rows["chain"]["n_on_implied"] == 1


# --------------------------------------------------------------------------
# Two readout precisions are two measurements, and must not share a rate
# --------------------------------------------------------------------------


def test_arms_recorded_at_one_precision_pool_normally():
    arms = {"a": {**report(*ONE, implied=5, raw=8), "readout_dtype": "float32"},
            "b": {**report((4, 4, 0.7), implied=6, raw=9), "readout_dtype": "float32"}}
    assert len(pool(arms, layer=13)) == 2


def test_the_archived_arms_pool_although_none_of_them_records_a_precision():
    """Absent is a state, not a mismatch.

    The eight archived runs predate the field entirely. They were all bfloat16,
    so they are one precision and one rate; refusing them would make the fix
    retroactively unusable on the only data that exists.
    """
    assert len(pool({"a": report(*ONE, implied=5, raw=8)}, layer=13)) == 1


def test_a_float32_arm_and_an_archived_arm_are_refused_rather_than_merged():
    """The whole point of rerunning in float32 is that it is a different readout.

    A bfloat16 digit logit is on a 0.125-nat grid and ties at that resolution
    are what forced the tie policy. Pooling a float32 arm into the archived
    counts would compare a rate that can tie against one that mostly cannot,
    and the difference would read as an effect.
    """
    arms = {"archived": report(*ONE, implied=5, raw=8),
            "fresh": {**report((4, 4, 0.7), implied=6, raw=9), "readout_dtype": "float32"}}
    with pytest.raises(ValueError, match="readout_dtype"):
        pool(arms, layer=13)


def test_two_recorded_precisions_are_refused_as_well():
    arms = {"a": {**report(*ONE, implied=5, raw=8), "readout_dtype": "bfloat16"},
            "b": {**report((4, 4, 0.7), implied=6, raw=9), "readout_dtype": "float32"}}
    with pytest.raises(ValueError, match="readout_dtype"):
        pool(arms, layer=13)


def test_a_precision_mismatch_outside_the_pooled_generator_is_not_a_mismatch():
    """The filter runs first: an arm that is not in the family cannot conflict."""
    arms = {"a": report(*ONE, implied=5, raw=8),
            "b": {**report((4, 4, 0.7), implied=6, raw=9, generator="v1_unpaired"),
                  "readout_dtype": "float32"}}
    assert len(pool(arms, layer=13, generator="v3_distinct")) == 1
