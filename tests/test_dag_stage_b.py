"""Stage B of E2: the patched run, restricted to the pairs stage A chose.

Stage A is the half that decides *which* items are comparable, and it ran
before any patch existed. Stage B is the half that patches them. The seam
between the two is the part worth testing: stage B has to measure the selected
items and no others, and it has to be able to prove that the item it just
regenerated is the item that was screened -- the screening file records
`(depth, seed, index, gap)` but not `n_decoys`, so regeneration is a claim, not
a guarantee, until it is checked against the archived measurement.

The outcome definitions below are the registered ones (`EXPERIMENT_LOG.md`,
2026-08-15): the implied digit *uniquely* on top under the ancestor patch, a
1,000-replicate cluster bootstrap for the depth difference, the four-way level
split reported beside it, and a null-flip rate of 20% or more making the arm an
invalid test rather than a negative one.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dag.dag_stage_b as dag_stage_b
from dag.dag_stage_b import (
    LAYER,
    NULL_FLIP_LIMIT,
    batches,
    check_clean,
    check_regenerated,
    flip_rates,
    level_split,
    primary,
    rates,
    validity,
)

BINS = [6, 13, 20, 27]


def probs(**named):
    """A ten-digit readout, the unnamed mass spread over the other digits."""
    values = [0.0] * 10
    for digit, value in named.items():
        values[int(digit[1:])] = value
    spare = [d for d in range(10) if d not in {int(k[1:]) for k in named}]
    left = 1.0 - sum(values)
    for digit in spare:
        values[digit] = left / len(spare)
    return values


def block(kind, patched, *, implied=6, raw=9, clean=3):
    """One edit's rows: one per layer bin, only layer 13 carrying the readout."""
    return [{"kind": kind, "layer": layer, "node": "a",
             "implied_value": implied, "raw_value": raw, "clean_value": clean,
             "distance_to_read": 24, "tv": 0.5,
             "delta_toward": 1.0, "delta_toward_raw": 0.5,
             "clean_target_logodds": 0.0,
             "probs_patched": patched if layer == LAYER else probs(d3=0.9)}
            for layer in BINS]


def item(patched, *, clean=None, implied=6, raw=9, target=3, nulls=()):
    """One item: an ancestor edit, plus whatever null edits a test needs."""
    clean = clean if clean is not None else probs(d3=0.9)
    rows = block("ancestor", patched, implied=implied, raw=raw, clean=target)
    for null in nulls:
        rows += block("null", null, implied=implied, raw=raw, clean=target)
    return rows, {"target_value": target, "clean_probs": clean,
                  "clean_top_digit": clean.index(max(clean)),
                  "clean_target_logodds": 0.0, "clean_digit_mass": 0.99}


def report(built, *, depth=1, pairs=None):
    """A stage-B arm in the shape `dag_patching` writes and `dag_pooling` reads."""
    rows = [row for block_rows, _ in built for row in block_rows]
    items = [summary for _, summary in built]
    pairs = list(range(len(built))) if pairs is None else pairs
    return {
        "model": "test", "generator": "v3_distinct", "readout_dtype": "float32",
        "condition": "both", "omit": "none", "depth": depth, "seed": None,
        "n_items": len(items), "layer_bins": list(BINS), "n_layers": 28,
        "items": items, "rows": rows,
        "selected": [{"depth": depth, "seed": 10 + n, "index": n, "gap": 0,
                      "pair": pair} for n, pair in enumerate(pairs)],
    }


def selection(pairs):
    """A stage-A selection payload: `(depth, seed, index, gap)` per side."""
    return {"selection": {"depths": [1, 2], "proceed": True, "pairs": [
        [{"depth": depth, "seed": seed, "index": index, "gap": gap,
          "generator": "v3_distinct", "ancestor_distance": 24,
          "target_value": 3, "clean_target_share": 0.9}
         for depth, seed, index, gap in sides]
        for sides in pairs]}}


# --------------------------------------------------------------------------
# Which items get patched, and nothing else does
# --------------------------------------------------------------------------


def test_selected_items_are_grouped_by_the_batch_that_generates_them():
    """Regenerating a batch is the expensive part; two selected items in one
    batch must not cost two regenerations."""
    chosen = selection([[(1, 11, 3, 0), (2, 21, 0, 0)],
                        [(1, 11, 7, 0), (2, 22, 0, 0)]])
    grouped = {(batch["depth"], batch["seed"], batch["gap"]): batch
               for batch in batches(chosen)}
    assert len(grouped) == 3
    assert [entry["index"] for entry in grouped[(1, 11, 0)]["items"]] == [3, 7]


def test_no_item_outside_the_selection_is_reached():
    """Stage A picked index 3 of a twenty-item batch. The other nineteen were
    screened and rejected, and patching them would be a different experiment."""
    chosen = selection([[(1, 11, 3, 0), (2, 21, 0, 0)]])
    reached = {(batch["depth"], batch["seed"], batch["gap"], entry["index"])
               for batch in batches(chosen) for entry in batch["items"]}
    assert reached == {(1, 11, 0, 3), (2, 21, 0, 0)}


def test_the_pair_an_item_was_matched_in_survives_the_grouping():
    """The bootstrap resamples pairs, so the pairing has to reach the analysis.
    Grouping by batch cuts across it -- one batch can hold items from several
    pairs -- which is exactly how it would get lost."""
    chosen = selection([[(1, 11, 3, 0), (2, 21, 0, 0)],
                        [(1, 11, 7, 0), (2, 22, 0, 0)]])
    tagged = {(batch["seed"], entry["index"]): entry["pair"]
              for batch in batches(chosen) for entry in batch["items"]}
    assert tagged == {(11, 3): 0, (21, 0): 0, (11, 7): 1, (22, 0): 1}


def test_the_batches_come_out_in_a_fixed_order():
    forward = selection([[(1, 12, 0, 0), (2, 21, 0, 0)],
                         [(1, 11, 0, 1), (2, 22, 0, 0)]])
    backward = selection([[(1, 11, 0, 1), (2, 22, 0, 0)],
                          [(1, 12, 0, 0), (2, 21, 0, 0)]])
    keys = lambda chosen: [(b["depth"], b["seed"], b["gap"])
                           for b in batches(chosen)]
    assert keys(forward) == sorted(keys(forward))
    assert set(keys(forward)) == set(keys(backward))


# --------------------------------------------------------------------------
# The regenerated item has to be the screened item
# --------------------------------------------------------------------------


def test_an_item_that_regenerates_to_a_different_distance_is_refused():
    """The screening file records seed, index and gap but not `n_decoys`, and
    `n_decoys` changes the trace. So regeneration is checked against what stage
    A measured rather than trusted because the knobs looked right."""
    record = {"ancestor_distance": 24, "target_value": 3, "gap": 0}
    with pytest.raises(ValueError, match="ancestor_distance"):
        check_regenerated(record, {"ancestor_distance": 25, "target_value": 3,
                                   "gap": 0})


def test_an_item_that_regenerates_to_a_different_answer_is_refused():
    record = {"ancestor_distance": 24, "target_value": 3, "gap": 0}
    with pytest.raises(ValueError, match="target_value"):
        check_regenerated(record, {"ancestor_distance": 24, "target_value": 4,
                                   "gap": 0})


def test_an_item_that_regenerates_to_the_screened_one_passes():
    record = {"ancestor_distance": 24, "target_value": 3, "gap": 0}
    assert check_regenerated(record, dict(record)) is None


def test_a_clean_readout_that_does_not_reproduce_is_refused():
    """Same weights, same tokens, same precision: the clean share should come
    back bit-for-bit. A drift here means the item is not the screened item,
    whatever its distance says."""
    record = {"target_value": 3, "clean_target_share": 0.9}
    with pytest.raises(ValueError, match="clean"):
        check_clean(record, {"target_value": 3, "clean_probs": probs(d3=0.5)})


def test_a_clean_readout_that_reproduces_passes():
    record = {"target_value": 3, "clean_target_share": probs(d3=0.9)[3]}
    assert check_clean(record, {"target_value": 3,
                                "clean_probs": probs(d3=0.9)}) is None


# --------------------------------------------------------------------------
# The validity gate, which keeps the verdict space three-valued
# --------------------------------------------------------------------------


def test_an_arm_whose_nulls_flip_the_answer_is_an_invalid_test():
    """`depth2_chain` is the precedent: nulls flipped 23/40 and every relative
    gate passed anyway, because every gate is relative. A background that moves
    as much as the ancestor does makes the arm unreadable, not negative."""
    flipped = probs(d6=0.8)
    built = [item(probs(d6=0.8), nulls=[flipped, flipped]) for _ in range(4)]
    outcome = validity(report(built))
    assert outcome["null"]["flipped"] == 8
    assert outcome["invalid_test"] is True


def test_an_arm_whose_nulls_leave_the_answer_alone_is_readable():
    quiet = probs(d3=0.9)
    built = [item(probs(d6=0.8), nulls=[quiet, quiet]) for _ in range(4)]
    outcome = validity(report(built))
    assert outcome["null"]["rate"] == 0.0
    assert outcome["invalid_test"] is False


def test_the_gate_fires_exactly_at_the_registered_limit():
    """20% *or more*, as registered. One flip in five is the boundary case and
    it is inside the gate, not outside it."""
    quiet, flipped = probs(d3=0.9), probs(d6=0.8)
    built = [item(probs(d6=0.8), nulls=[flipped, quiet, quiet, quiet, quiet])]
    outcome = validity(report(built))
    assert outcome["null"]["rate"] == pytest.approx(NULL_FLIP_LIMIT)
    assert outcome["invalid_test"] is True


def test_the_flip_rate_is_read_at_the_registered_layer_only():
    """`block` puts a quiet readout at every bin but 13. If the gate pooled
    layers it would see four times the rows and a quarter of the rate."""
    flipped = probs(d6=0.8)
    built = [item(probs(d6=0.8), nulls=[flipped])]
    assert flip_rates(report(built))["null"] == {"flipped": 1, "n": 1,
                                                 "rate": 1.0}


# --------------------------------------------------------------------------
# The primary outcome: the implied digit, uniquely, on top
# --------------------------------------------------------------------------


def test_the_rate_counts_the_implied_digit_alone_on_top():
    built = [item(probs(d6=0.8)), item(probs(d9=0.8))]
    outcome = rates(report(built))
    assert (outcome["hits"], outcome["n"]) == (1, 2)


def test_a_tie_between_the_implied_digit_and_another_is_not_a_win():
    """The tie policy the bfloat16 correction forced. float32 makes ties rare
    rather than impossible, and a bare argmax would resolve this one by digit
    order -- which is a property of `list.index`, not of the model."""
    built = [item(probs(d6=0.4, d9=0.4))]
    assert rates(report(built))["hits"] == 0


def test_an_item_whose_clean_readout_ties_is_out_of_the_denominator():
    """No unique clean answer means nothing for the patch to move off."""
    built = [item(probs(d6=0.8), clean=probs(d3=0.4, d4=0.4)),
             item(probs(d6=0.8))]
    outcome = rates(report(built))
    assert (outcome["hits"], outcome["n"]) == (1, 1)


def test_the_difference_is_depth_one_minus_depth_two():
    win, lose = probs(d6=0.8), probs(d3=0.9)
    arms = {1: report([item(win)] * 4, depth=1),
            2: report([item(win)] + [item(lose)] * 3, depth=2)}
    outcome = primary(arms, replicates=200)
    assert outcome["rate"] == {1: 1.0, 2: 0.25}
    assert outcome["difference"] == pytest.approx(0.75)


# --------------------------------------------------------------------------
# The interval, resampled over the unit the design pairs
# --------------------------------------------------------------------------


def test_a_difference_with_no_variation_gets_a_degenerate_interval():
    win, lose = probs(d6=0.8), probs(d3=0.9)
    arms = {1: report([item(win)] * 8, depth=1),
            2: report([item(lose)] * 8, depth=2)}
    outcome = primary(arms, replicates=200)
    assert outcome["interval"] == [1.0, 1.0]


def test_the_interval_brackets_the_point_estimate():
    win, lose = probs(d6=0.8), probs(d3=0.9)
    arms = {1: report([item(win)] * 6 + [item(lose)] * 2, depth=1),
            2: report([item(win)] * 2 + [item(lose)] * 6, depth=2)}
    outcome = primary(arms, replicates=500)
    low, high = outcome["interval"]
    assert low <= outcome["difference"] <= high
    assert low < high


def test_the_same_seed_gives_the_same_interval():
    win, lose = probs(d6=0.8), probs(d3=0.9)
    arms = lambda: {1: report([item(win)] * 5 + [item(lose)] * 3, depth=1),
                    2: report([item(win)] * 3 + [item(lose)] * 5, depth=2)}
    assert (primary(arms(), replicates=200)["interval"]
            == primary(arms(), replicates=200)["interval"])


def test_resampling_takes_whole_pairs_rather_than_the_two_arms_separately():
    """A replicate draws a pair index and takes both of its sides, so it can
    never hold half a pair -- an unmatched comparison inside a procedure whose
    whole purpose is the matching.

    Both sides of every pair here agree, so a pairwise resample cancels exactly
    and the difference is 0.0 in every replicate however lopsided the draw. Two
    arms resampled independently would not cancel: they would draw different
    numbers of wins and put spread on a difference that has none. The degenerate
    interval is the assertion.
    """
    win, lose = probs(d6=0.8), probs(d3=0.9)
    sides = [item(win)] * 4 + [item(lose)] * 4
    arms = {1: report(sides, depth=1), 2: report(sides, depth=2)}
    outcome = primary(arms, replicates=500)
    assert outcome["difference"] == 0.0
    assert outcome["interval"] == [0.0, 0.0]


# --------------------------------------------------------------------------
# The secondary reading: where the mass actually went
# --------------------------------------------------------------------------


def test_the_level_split_reports_clean_and_patched_medians():
    """Reported and not gated. It is the reading that says what `21/23` means:
    at depth 1 the transplanted state promotes the donor's *literal* digit too,
    0.0005 to 0.373 under a foreign donor, and a reader will ask."""
    built = [item(probs(d6=0.7, d9=0.2, d3=0.05), clean=probs(d3=0.9, d6=0.01))]
    split = level_split(report(built))
    assert split["implied"]["clean"] == pytest.approx(0.01)
    assert split["implied"]["patched"] == pytest.approx(0.7)
    assert split["target"]["clean"] == pytest.approx(0.9)
    assert split["target"]["patched"] == pytest.approx(0.05)


def test_the_remaining_mass_is_what_the_three_named_digits_leave():
    built = [item(probs(d6=0.7, d9=0.2, d3=0.05))]
    split = level_split(report(built))
    assert split["other"]["patched"] == pytest.approx(0.05)


def test_the_level_split_counts_ties_apart_rather_than_resolving_them():
    built = [item(probs(d6=0.4, d9=0.4)), item(probs(d6=0.8))]
    assert level_split(report(built))["patched_tied"] == 1


# --------------------------------------------------------------------------
# What was registered, and what this module is allowed to reach
# --------------------------------------------------------------------------


def test_the_layer_is_the_one_inherited_from_the_discovery_table():
    """Not re-searched here. That is what makes stage B confirmatory for the
    depth contrast and not for the layer."""
    from dag.dag_pooling import POOLED_LAYER

    assert LAYER == POOLED_LAYER == 13


def test_the_analysis_cannot_reach_a_layer_the_arm_did_not_measure():
    built = [item(probs(d6=0.8))]
    with pytest.raises(ValueError, match="layer"):
        rates(report(built), layer=99)


def test_the_cross_item_row_kind_is_not_claimed_to_be_here():
    """The registered row-kind list names five. A cross-item batch is selected
    for mutual donatability, so it is a different batch from a plain run at the
    same seed -- the selected items do not exist in one. Four kinds are
    reachable on matched pairs and the fifth is not, which is a limit of the
    matched design and is recorded as one rather than quietly dropped.
    """
    assert "cross_item" not in dag_stage_b.ROW_KINDS
    assert dag_stage_b.UNREACHABLE_ROW_KINDS == ("cross_item",)
