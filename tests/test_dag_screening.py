"""Stage A of E2: choosing which items are comparable across depth.

The depth ladder's collapse after depth 1 is confounded. Eligible clean
p(target) runs 0.666-0.961 at depth 1 and 0.966-0.999 at depth 2, so the two
supports do not touch: every depth-1 success is on an item the model was unsure
of, and every depth-2 failure on one it was sure of. `ancestor_distance` is
{11, 24} against {23, 36}. A comparison that does not match on both is a
comparison of confidence and position, whatever it says about depth.

The rule below is registered in `EXPERIMENT_LOG.md`, 2026-08-15, and its one
load-bearing property is that it reads clean measurements only. There is no
patched outcome in existence when it runs, so it cannot be tuned toward one --
which is what these tests are mostly here to pin.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dag_screening import (
    DISTANCE_TOLERANCE,
    MAX_PAIRS,
    MIN_PAIRS,
    confidence_window,
    eligible,
    select,
)


def item(depth, share, distance, *, seed=0, index=0, gap=0, tied=False):
    """One screened item: what a clean forward pass alone can say about it."""
    return {
        "depth": depth, "seed": seed, "index": index, "gap": gap,
        "clean_target_share": share, "ancestor_distance": distance,
        "clean_correct_unique": not tied,
    }


def spines(depth, shares, distance, *, start=0):
    """One item per spine, so nothing is dropped by the one-per-spine rule."""
    return [item(depth, share, distance, index=start + n)
            for n, share in enumerate(shares)]


def anchors(depth, distance):
    """Two items that widen the depth's support without being matchable.

    The window is the intersection of the two supports, so with three or four
    items in a test it collapses onto whichever endpoint is innermost and there
    is no room left to observe the matching rule. These sit at the far ends and
    at a distance no partner comes near, which is what a real 200-item screen
    would supply on its own.
    """
    return [item(depth, share, distance, index=900 + n)
            for n, share in enumerate((0.80, 0.99))]


def identities(pairs):
    return [tuple((record["depth"], record["seed"], record["index"],
                   record["gap"]) for record in pair) for pair in pairs]


# --------------------------------------------------------------------------
# Eligibility: an item with no clean answer has nothing for a patch to move off
# --------------------------------------------------------------------------


def test_an_item_whose_clean_readout_ties_is_not_screened_in():
    records = [item(1, 0.7, 24), item(1, 0.7, 24, index=1, tied=True)]
    assert len(eligible(records)) == 1


def test_an_item_with_no_recorded_share_is_not_screened_in():
    """Unmeasured is not the same as ineligible, but it cannot be matched."""
    assert eligible([item(1, None, 24)]) == []


# --------------------------------------------------------------------------
# The window is where the two depths overlap, and it is allowed to be empty
# --------------------------------------------------------------------------


def test_the_window_is_the_intersection_of_the_two_supports():
    records = spines(1, [0.60, 0.90], 24) + spines(2, [0.80, 0.99], 23)
    assert confidence_window(records, depths=(1, 2)) == (0.80, 0.90)


def test_supports_that_do_not_touch_give_no_window_rather_than_an_error():
    """This is the archived situation exactly: 0.961 against 0.966.

    An empty window is the registered stop condition, so it has to be a value
    the caller can act on, not an exception it has to catch.
    """
    records = spines(1, [0.66, 0.96], 24) + spines(2, [0.97, 0.99], 23)
    assert confidence_window(records, depths=(1, 2)) is None


def test_a_depth_with_no_eligible_item_gives_no_window():
    assert confidence_window(spines(1, [0.7], 24), depths=(1, 2)) is None


# --------------------------------------------------------------------------
# Matching: confidence, distance, and one item used once
# --------------------------------------------------------------------------


def test_a_pair_outside_the_distance_tolerance_is_not_matched():
    records = spines(1, [0.90], 24) + spines(2, [0.90], 24 + DISTANCE_TOLERANCE + 1)
    assert select(records, depths=(1, 2))["pairs"] == []


def test_a_pair_inside_the_distance_tolerance_is_matched():
    records = spines(1, [0.90], 24) + spines(2, [0.90], 24 + DISTANCE_TOLERANCE)
    assert len(select(records, depths=(1, 2))["pairs"]) == 1


def test_the_closer_confidence_match_wins_the_partner():
    """Greedy on |delta share|, so the near miss does not consume the good pair."""
    records = (anchors(1, 60) + anchors(2, 90)
               + spines(1, [0.900, 0.930], 24)
               + [item(2, 0.902, 23, index=9)])
    pairs = select(records, depths=(1, 2))["pairs"]
    assert [pair[0]["clean_target_share"] for pair in pairs] == [0.900]


def test_an_item_is_not_matched_twice():
    records = (anchors(1, 60) + anchors(2, 90)
               + spines(1, [0.90], 24) + spines(2, [0.90, 0.91], 23))
    assert len(select(records, depths=(1, 2))["pairs"]) == 1


def test_only_one_placement_of_a_spine_survives():
    """`depth1_gap{0,1,2}` are the same five spines at three ancestor distances.

    Pooling counted 33 depth-1 observations over 17 spines for exactly this
    reason. Three placements of one spine are one item's worth of evidence, and
    the gap sweep here is a distance sampler, not a way to triple n.
    """
    placements = [item(1, 0.90, distance, seed=0, index=0, gap=gap)
                  for gap, distance in enumerate((22, 23, 24))]
    partners = spines(2, [0.90, 0.90, 0.90], 23, start=100)
    records = anchors(1, 60) + anchors(2, 90) + placements + partners
    # Three placements against three distinct partners: without the rule this
    # is three pairs, and every one of them is the same spine's evidence.
    assert len(select(records, depths=(1, 2))["pairs"]) == 1


def test_the_surviving_placement_is_the_best_matching_one():
    placements = [item(1, share, 23, seed=0, index=0, gap=gap)
                  for gap, share in enumerate((0.85, 0.90))]
    records = (anchors(1, 60) + anchors(2, 90) + placements
               + [item(2, 0.90, 23, index=100)])
    pairs = select(records, depths=(1, 2))["pairs"]
    assert pairs[0][0]["gap"] == 1


# --------------------------------------------------------------------------
# The registered floor, ceiling, and the three outcomes
# --------------------------------------------------------------------------


def test_too_few_pairs_stops_the_experiment_rather_than_running_it_small():
    records = spines(1, [0.90] * 3, 24) + spines(2, [0.90] * 3, 23, start=100)
    outcome = select(records, depths=(1, 2))
    assert outcome["proceed"] is False
    assert outcome["n_pairs"] == 3


def test_the_floor_is_the_registered_one():
    shares = [0.90] * MIN_PAIRS
    records = spines(1, shares, 24) + spines(2, shares, 23, start=100)
    assert select(records, depths=(1, 2))["proceed"] is True


def test_no_more_than_the_registered_ceiling_is_taken():
    shares = [0.90] * (MAX_PAIRS + 5)
    records = spines(1, shares, 24) + spines(2, shares, 23, start=100)
    assert select(records, depths=(1, 2))["n_pairs"] == MAX_PAIRS


def test_an_empty_window_stops_without_matching_anything():
    records = spines(1, [0.66, 0.96], 24) + spines(2, [0.97, 0.99], 23, start=100)
    outcome = select(records, depths=(1, 2))
    assert outcome["proceed"] is False
    assert outcome["window"] is None
    assert outcome["pairs"] == []


# --------------------------------------------------------------------------
# The property the whole design rests on
# --------------------------------------------------------------------------


def test_the_selection_ignores_anything_a_patch_would_have_produced():
    """If a patched outcome could reach this, it would not be a selection rule.

    Stage A runs before any patch, so in practice these fields are absent. The
    test adds them anyway, with values that would flip the result if they were
    read, because "we did not look" is a claim that should be checkable rather
    than promised.
    """
    shares = [0.90] * MIN_PAIRS
    plain = spines(1, shares, 24) + spines(2, shares, 23, start=100)
    tainted = [{**record,
                "implied_top_unique": record["depth"] == 1,
                "patched_tops": [record["depth"]],
                "tv": 0.9 / record["depth"]}
               for record in plain]
    chosen = select(tainted, depths=(1, 2))
    assert identities(chosen["pairs"]) == identities(select(plain,
                                                            depths=(1, 2))["pairs"])
    assert chosen["window"] == select(plain, depths=(1, 2))["window"]
    assert chosen["proceed"] is True


def test_selecting_twice_gives_the_same_answer():
    records = spines(1, [0.90, 0.91, 0.92], 24) + spines(2, [0.905, 0.915], 23,
                                                         start=100)
    assert select(records, depths=(1, 2)) == select(records, depths=(1, 2))


def test_the_order_the_items_arrive_in_does_not_change_the_selection():
    records = spines(1, [0.90, 0.91, 0.92], 24) + spines(2, [0.905, 0.915], 23,
                                                         start=100)
    forward = select(records, depths=(1, 2))["pairs"]
    backward = select(list(reversed(records)), depths=(1, 2))["pairs"]
    assert forward == backward
