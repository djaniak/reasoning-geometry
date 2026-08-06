import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepconf_weighted_vote import (
    DEEPCONF_STATISTICS,
    FILTER_KEEP,
    _readout_specs,
    load_trace_confidence,
    prompt_vote_features,
    selection_accuracy,
    weight_dispersion,
    weighted_vote,
)


def _row(trace_id, answer, *, gold="A", correct=None):
    return {
        "prompt_id": 0,
        "trace_id": trace_id,
        "predicted_answer": answer,
        "gold_answer": gold,
        "is_correct": int(answer == gold) if correct is None else int(correct),
        "logprob_score": -1.0,
    }


def test_weighted_vote_can_overturn_the_plurality_winner():
    """The whole point of weighting: a confident minority outvotes the majority."""
    rows = [_row(0, "A"), _row(1, "A"), _row(2, "B")]
    weights = {0: 1.0, 1: 1.0, 2: 10.0}

    result = weighted_vote(rows, weights)

    assert result["winner"] == "B"
    assert result["own_share"] == pytest.approx(10 / 12)
    # The plurality winner is still "A", and its weighted share is what replaces
    # `vote_agreement` -- the target is plurality correctness either way.
    assert result["plurality_share"] == pytest.approx(2 / 12)


def test_uniform_weights_reproduce_plain_vote_agreement():
    rows = [_row(0, "A"), _row(1, "A"), _row(2, "B")]

    result = weighted_vote(rows, {0: 3.0, 1: 3.0, 2: 3.0})

    assert result["winner"] == "A"
    assert result["plurality_share"] == pytest.approx(2 / 3)


def test_filtering_keeps_the_most_confident_traces():
    rows = [_row(0, "A"), _row(1, "B"), _row(2, "B"), _row(3, "B")]
    weights = {0: 9.0, 1: 4.0, 2: 4.0, 3: 4.0}

    # Keeping only the most confident trace elects its answer outright; keeping
    # every trace hands it back to the (less confident) majority.
    assert weighted_vote(rows, weights, keep=1)["winner"] == "A"
    assert weighted_vote(rows, weights, keep=1)["n_traces"] == 1
    assert weighted_vote(rows, weights, keep=4)["winner"] == "B"


def test_the_filtered_share_is_measured_against_the_unfiltered_plurality_winner():
    """Otherwise keep=1 scores 1.0 for every prompt and the feature is constant."""
    rows = [_row(0, "A"), _row(1, "B"), _row(2, "B")]
    weights = {0: 9.0, 1: 1.0, 2: 1.0}

    # "B" is the prompt's plurality winner; filtering to the single most confident
    # trace leaves only an "A", so B's surviving share is zero, not one.
    assert weighted_vote(rows, weights, keep=1)["plurality_share"] == pytest.approx(0.0)
    assert weighted_vote(rows, weights, keep=1)["own_share"] == pytest.approx(1.0)


def test_keeping_two_traces_selects_the_same_answer_as_keeping_one():
    """Structural, not a coincidence in the results: with two survivors the
    heavier one wins any disagreement, so top-2 selection *is* top-1 selection.
    Only the vote share the two produce differs."""
    rows = [_row(0, "A"), _row(1, "B"), _row(2, "B")]
    weights = {0: 5.0, 1: 4.0, 2: 4.0}

    one, two = weighted_vote(rows, weights, keep=1), weighted_vote(rows, weights, keep=2)

    assert one["winner"] == two["winner"] == "A"
    assert one["own_share"] != two["own_share"]


def test_survivor_set_does_not_depend_on_row_order():
    rows = [_row(0, "A"), _row(1, "B"), _row(2, "C")]
    weights = {0: 5.0, 1: 5.0, 2: 1.0}

    forward = weighted_vote(rows, weights, keep=2)
    reversed_ = weighted_vote(list(reversed(rows)), weights, keep=2)

    assert forward["winner"] == reversed_["winner"]
    assert forward["own_share"] == pytest.approx(reversed_["own_share"])


def test_unparseable_and_unscored_traces_are_dropped():
    rows = [_row(0, "A"), _row(1, ""), _row(2, "B")]

    result = weighted_vote(rows, {0: 1.0, 1: 1.0, 2: 1.0})
    assert result["n_traces"] == 2

    # A trace with no exact confidence cannot be weighted, so it leaves the vote.
    assert weighted_vote(rows, {0: 1.0, 2: 1.0})["n_traces"] == 2
    assert weighted_vote(rows, {0: 1.0})["n_traces"] == 1


def test_non_positive_weights_raise_rather_than_produce_a_bogus_share():
    """DeepConf's C is strictly positive; anything else means the wrong column."""
    rows = [_row(0, "A"), _row(1, "B")]

    with pytest.raises(ValueError, match="strictly positive"):
        weighted_vote(rows, {0: 1.0, 1: -3.0})


def test_a_prompt_with_no_usable_trace_scores_nan_not_zero():
    result = weighted_vote([_row(0, "")], {0: 1.0})

    assert result["winner"] is None
    assert np.isnan(result["plurality_share"])


def test_trace_confidence_is_loaded_per_trace_not_averaged(tmp_path):
    rows = [
        {"prompt_id": 3, "trace_id": 0, **{key: 1.0 for key in DEEPCONF_STATISTICS}},
        {"prompt_id": 3, "trace_id": 1, **{key: 5.0 for key in DEEPCONF_STATISTICS}},
    ]
    path = tmp_path / "exact.npz"
    np.savez(path, trace_summaries=np.array(rows, dtype=object))

    confidence = load_trace_confidence(path)

    assert set(confidence) == {(3, 0), (3, 1)}
    assert confidence[(3, 0)]["bottom10_group_confidence"] == pytest.approx(1.0)
    assert confidence[(3, 1)]["bottom10_group_confidence"] == pytest.approx(5.0)


def test_prompt_features_cover_every_statistic_and_every_filter_level():
    rows = {0: [_row(0, "A"), _row(1, "B")]}
    confidence = {
        (0, 0): {key: 2.0 for key in DEEPCONF_STATISTICS},
        (0, 1): {key: 1.0 for key in DEEPCONF_STATISTICS},
    }

    features = prompt_vote_features(rows, confidence)[0]

    for statistic in DEEPCONF_STATISTICS:
        assert f"dcvote_{statistic}" in features
        assert f"dcvote_own_{statistic}" in features
        for keep in FILTER_KEEP:
            assert f"dcfilter{keep}_{statistic}" in features


def test_selection_accuracy_scores_the_selected_answer_against_gold():
    """Weighting selects a different answer, so its accuracy can differ."""
    rows = {0: [_row(0, "A"), _row(1, "A"), _row(2, "B")]}
    confidence = {
        (0, 0): {key: 1.0 for key in DEEPCONF_STATISTICS},
        (0, 1): {key: 1.0 for key in DEEPCONF_STATISTICS},
        (0, 2): {key: 50.0 for key in DEEPCONF_STATISTICS},
    }

    table = selection_accuracy(rows, confidence, [0])

    assert table["accuracy"]["plurality"] == pytest.approx(1.0)
    assert table["accuracy"]["weighted_bottom10_group_confidence"] == pytest.approx(0.0)
    # An accuracy that matches plurality means nothing without knowing whether the
    # rule ever picked a different answer.
    assert table["disagreement_with_plurality"]["weighted_bottom10_group_confidence"] == 1.0
    assert table["disagreement_with_plurality"]["plurality"] == 0.0
    assert table["n_prompts"] == 1


def test_weight_dispersion_reports_the_ceiling_on_any_reweighting():
    rows = {0: [_row(0, "A"), _row(1, "B")]}
    confidence = {
        (0, 0): {key: 4.0 for key in DEEPCONF_STATISTICS},
        (0, 1): {key: 2.0 for key in DEEPCONF_STATISTICS},
    }

    body = weight_dispersion(rows, confidence, [0])["bottom10_group_confidence"]

    assert body["median_max_over_min"] == pytest.approx(2.0)
    assert body["median_within_prompt_cv"] == pytest.approx(1 / 3)
    assert body["n_prompts"] == 1


def test_selection_accuracy_ignores_prompts_outside_the_population():
    rows = {0: [_row(0, "A")], 1: [_row(0, "B")]}
    confidence = {
        (0, 0): {key: 1.0 for key in DEEPCONF_STATISTICS},
        (1, 0): {key: 1.0 for key in DEEPCONF_STATISTICS},
    }

    assert selection_accuracy(rows, confidence, [0])["n_prompts"] == 1


def test_the_weighted_baseline_replaces_the_vote_rather_than_only_adding_to_it():
    """A baseline that keeps plain agreement alongside is not the strengthened one."""
    specs = _readout_specs(("bottom10_group_confidence",))

    swapped = specs["B0_dcvote_bottom10_group_confidence"]
    added = specs["B0_plus_dcvote_bottom10_group_confidence"]

    assert "vote_agreement" not in swapped
    assert "dcvote_bottom10_group_confidence" in swapped
    assert len(swapped) == len(specs["B0"])
    assert "vote_agreement" in added and len(added) == len(specs["B0"]) + 1
    assert specs["B1_dcvote_bottom10_group_confidence"] == swapped + ("rmd_tail_q20",)
