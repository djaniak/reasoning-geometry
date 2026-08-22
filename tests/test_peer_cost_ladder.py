import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controls.peer_cost_ladder import (
    AGREE_PREFIX,
    PEER_PREFIX,
    ladder_rungs,
    peer_agreement_rates,
    peer_sample_rates,
    rung_cost,
    saturation_flags,
    target_winners,
    tokens_per_sibling,
)
from controls.peer_difficulty_control import peer_pass_rates


def _rows(prompt_id, answers, correct, *, gold="7", length=100):
    return [
        {
            "prompt_id": prompt_id,
            "trace_id": prompt_id * 100 + index,
            "sample_id": index,
            "predicted_answer": answer,
            "is_correct": float(flag),
            "gold_answer": gold,
            "trace_length": length,
        }
        for index, (answer, flag) in enumerate(zip(answers, correct))
    ]


def test_buying_every_sibling_reproduces_the_frozen_pass_rate():
    """The m=8 rung must be the frozen control's feature, not a near-copy.

    The continuity check against `peer_difficulty_control` is the only thing
    tying this ladder to the locked artifact; if the full-cache rung drifted,
    that check would pass vacuously.
    """
    rows = _rows(0, ["7", "9", "7", "7"], [1, 0, 1, 1])

    assert peer_sample_rates(rows, 4, seed=42, draw=0) == peer_pass_rates(rows)


def test_a_one_sample_peer_returns_one_siblings_outcome():
    rows = _rows(0, ["7", "9", "7", "7"], [1, 0, 1, 1])

    rate = peer_sample_rates(rows, 1, seed=42, draw=0)[0]

    assert rate in (0.0, 1.0)


def test_redrawing_changes_which_siblings_were_bought():
    """Sub-sampled rungs have a second variance source the bootstrap omits."""
    rows = _rows(0, ["7"] * 4 + ["9"] * 4, [1] * 4 + [0] * 4)

    rates = {peer_sample_rates(rows, 1, seed=42, draw=draw)[0] for draw in range(25)}

    assert rates == {0.0, 1.0}


def test_agreement_never_consults_the_gold_answer():
    """The deployable rung must be computable when no gold exists.

    `graded` reads `is_correct`, which is only defined once the problem has been
    marked. If agreement quietly depended on the same column it would be the
    same non-deployable feature under a new name.
    """
    winners = {0: "7"}
    right = _rows(0, ["7", "7", "7", "9"], [1, 1, 1, 0], gold="7")
    # Same answers, graded against a different gold: every is_correct flips.
    wrong = _rows(0, ["7", "7", "7", "9"], [0, 0, 0, 1], gold="9")

    assert peer_agreement_rates(right, winners, 4, seed=42, draw=0) == (
        peer_agreement_rates(wrong, winners, 4, seed=42, draw=0)
    )
    assert peer_sample_rates(right, 4, seed=42, draw=0) != (
        peer_sample_rates(wrong, 4, seed=42, draw=0)
    )


def test_agreement_is_measured_against_the_targets_answer():
    rows = _rows(0, ["7", "7", "9", "9"], [1, 1, 0, 0])

    assert peer_agreement_rates(rows, {0: "7"}, 4, seed=42, draw=0)[0] == 0.5
    assert peer_agreement_rates(rows, {0: "9"}, 4, seed=42, draw=0)[0] == 0.5
    assert peer_agreement_rates(rows, {0: "42"}, 4, seed=42, draw=0)[0] == 0.0


def test_an_unparsed_peer_trace_counts_as_disagreement():
    """It did not confirm the answer, and you paid for it either way."""
    rows = _rows(0, ["7", None, "7", None], [1, 0, 1, 0])

    assert peer_agreement_rates(rows, {0: "7"}, 4, seed=42, draw=0)[0] == 0.5


def test_a_target_with_no_answer_of_its_own_scores_zero_agreement():
    """Matching `vote_agreement`: there is nothing for a peer to confirm."""
    rows = _rows(0, ["7", "7", "7", "7"], [1, 1, 1, 1])

    assert peer_agreement_rates(rows, {0: None}, 4, seed=42, draw=0)[0] == 0.0


def test_the_two_readouts_buy_the_same_generations():
    """Same draw, same siblings -- only what is read off them differs.

    If the two rungs drew independently they would not be cost comparable: one
    would be reading a different purchase from the other at the same price.
    """
    # Answer and correctness carry the same information here, so an identical
    # draw must produce identical rates and any divergence is a different draw.
    rows = _rows(0, ["7", "9", "7", "9", "7", "9", "7", "9"], [1, 0, 1, 0, 1, 0, 1, 0])
    winners = {0: "7"}

    for draw in range(10):
        graded = peer_sample_rates(rows, 3, seed=42, draw=draw)[0]
        agree = peer_agreement_rates(rows, winners, 3, seed=42, draw=draw)[0]
        assert graded == agree


def test_target_winners_uses_the_plurality_answer():
    rows = _rows(0, ["7", "7", "9", None], [1, 1, 0, 0])

    assert target_winners(rows)[0] == "7"


def test_the_free_rung_buys_nothing_and_the_peer_rungs_scale_with_what_they_buy():
    rungs = ladder_rungs(("a", "b"), (1, 8))
    cost = lambda name: rung_cost(
        rungs[name], target_calls=8, target_tokens=1000.0, peer_tokens={"a": 10.0, "b": 20.0}
    )

    assert cost("B1")["extra_calls"] == 0
    assert cost("B1")["extra_tokens"] == 0
    # B1 is not cost-free in absolute terms; it is free at the margin.
    assert cost("B1")["total_calls"] == 8
    assert cost("B0_graded_a_m1")["extra_calls"] == 1
    assert cost("B0_graded_a_m8")["extra_tokens"] == 80.0
    assert cost("B0_graded_both_m8")["extra_calls"] == 16
    assert cost("B0_graded_both_m8")["extra_tokens"] == 240.0


def test_the_two_readouts_of_one_purchase_cost_the_same():
    """Otherwise the ladder would rank them on price rather than on content."""
    rungs = ladder_rungs(("a", "b"), (1, 2, 4, 8))
    kwargs = dict(target_calls=8, target_tokens=1000.0, peer_tokens={"a": 10.0, "b": 20.0})

    for name, rung in rungs.items():
        if rung["kind"] != "graded":
            continue
        twin = name.replace("_graded_", "_agree_")
        assert rung_cost(rung, **kwargs) == rung_cost(rungs[twin], **kwargs)


def test_the_ladder_names_its_two_readouts_apart():
    rungs = ladder_rungs(("a", "b"), (8,))

    assert any(PEER_PREFIX + "a" in rungs["B0_graded_a_m8"]["features"] for _ in (0,))
    assert AGREE_PREFIX + "a" in rungs["B0_agree_a_m8"]["features"]
    assert PEER_PREFIX + "a" not in rungs["B0_agree_a_m8"]["features"]


def test_capped_traces_are_charged_the_budget_they_spent():
    rows = _rows(0, ["7", None], [1, 0], length=1024)

    assert tokens_per_sibling(rows) == 1024


def test_a_rung_on_the_oracle_floor_is_flagged_saturated():
    """A delta against a saturated rung cannot say which readout is better."""
    body = {
        "populations": {
            "full_population": {
                "oracle_aurc": 0.03,
                "rungs": {
                    "B0": {"aurc_mean": 0.16},
                    "B1": {"aurc_mean": 0.13},
                    "B0_graded_both_m8": {"aurc_mean": 0.04},
                },
            }
        }
    }

    flags = saturation_flags(body, "full_population", threshold=0.9)

    assert flags["saturated_rungs"] == ["B0_graded_both_m8"]
    assert flags["headroom_fraction_removed"]["B1"] == pytest.approx(0.03 / 0.13)
