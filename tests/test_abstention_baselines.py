import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.abstention_baselines import (
    calibration_metrics,
    decision_values,
    group_confidences,
    holm_correction,
    prompt_auacc,
    prompt_aurc,
    prompt_vote_scores,
    trace_confidence_scores,
)
from applications.prompt_decomposition import region_indices


def _ranked_toy(n=100, base=0.6):
    """Outcomes ordered so that a higher score means a likelier-correct prompt."""
    outcomes = {i: float(i >= n * (1 - base)) for i in range(n)}
    perfect = {i: float(i) for i in range(n)}  # ranks every correct prompt first
    inverted = {i: -float(i) for i in range(n)}
    uninformative = {i: 0.0 for i in range(n)}
    return outcomes, perfect, inverted, uninformative


def test_auacc_is_higher_better_and_aurc_is_lower_better():
    outcomes, perfect, inverted, _ = _ranked_toy()
    assert prompt_auacc(perfect, outcomes) > prompt_auacc(inverted, outcomes)
    assert prompt_aurc(perfect, outcomes) < prompt_aurc(inverted, outcomes)


def test_perfect_ranking_saturates_auacc_and_floors_aurc():
    """A perfect ranker holds accuracy at 1.0 until coverage reaches base rate."""
    outcomes, perfect, _, _ = _ranked_toy(n=100, base=0.6)
    span = 1.0 - 1.0 / 100
    # Accuracy is 1.0 up to coverage 0.6 then decays as 0.6/c, so the closed form
    # is 0.6 + 0.6*ln(1/0.6) minus the unswept first 1/n of the grid.
    expected = 0.6 + 0.6 * np.log(1 / 0.6) - 0.01
    assert prompt_auacc(perfect, outcomes) == pytest.approx(expected, abs=0.01)
    assert prompt_auacc(perfect, outcomes) < span  # cannot reach the full span
    assert prompt_aurc(perfect, outcomes) == pytest.approx(span - expected, abs=0.01)


def test_constant_scores_are_ranked_by_prompt_id_not_averaged():
    """Ties are broken deterministically by prompt id, so a constant score is an
    arbitrary ranking rather than an average-case one.

    This matters for the discrete baselines: vote_agreement takes only nine
    distinct values over eight traces, so a large share of its ordering is decided
    by prompt id. The behaviour is inherited from wave1's sort key; the test pins
    it so the arbitrariness is visible rather than mistaken for a null result.
    """
    outcomes, _, _, uninformative = _ranked_toy(n=100, base=0.6)
    # Correct prompts have the *higher* ids here, so id-ascending ties put them
    # last and the constant scorer lands well below base accuracy, not at it.
    assert prompt_auacc(uninformative, outcomes) < 0.4
    flipped = {i: float(i < 40) for i in range(100)}
    assert prompt_auacc({i: 0.0 for i in range(100)}, flipped) > 0.6


def test_auacc_and_aurc_sum_to_the_coverage_span():
    outcomes, perfect, _, _ = _ranked_toy(n=40)
    span = 1.0 - 1.0 / 40
    assert prompt_auacc(perfect, outcomes) + prompt_aurc(perfect, outcomes) == pytest.approx(span)


def test_the_reported_auacc_exceeds_base_accuracy_only_when_ranking_helps():
    """Guards the mislabel that motivated this test: 0.83 against base 0.62 is
    only sensible for an accuracy integral, never for a risk integral."""
    outcomes, perfect, _, uninformative = _ranked_toy(n=100, base=0.62)
    assert prompt_auacc(perfect, outcomes) > 0.62
    assert prompt_aurc(perfect, outcomes) < 1 - 0.62
    assert prompt_auacc(uninformative, outcomes) < prompt_auacc(perfect, outcomes)


def test_group_confidences_are_sliding_window_means():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(group_confidences(values, 2), [1.5, 2.5, 3.5])
    # A window at least as long as the series collapses to the global mean.
    assert np.allclose(group_confidences(values, 4), [2.5])
    assert np.allclose(group_confidences(values, 99), [2.5])


def test_tail_scores_use_the_same_mask_as_the_geometry_scorers():
    """The whole point of the baseline is mask parity with rmd_tail_q20."""
    rng = np.random.default_rng(0)
    entropies = rng.random(50)
    logprobs = rng.normal(size=50)
    scores = trace_confidence_scores(entropies, logprobs)

    tail = region_indices(entropies, "tail_q20")
    high_entropy = region_indices(entropies, "high_entropy_q20")
    assert scores["conf_tail_q20_ent"] == pytest.approx(-entropies[tail].mean())
    assert scores["conf_tail_q20_lp"] == pytest.approx(logprobs[tail].mean())
    assert scores["conf_he_q20_ent"] == pytest.approx(-entropies[high_entropy].mean())
    assert scores["conf_he_q20_lp"] == pytest.approx(logprobs[high_entropy].mean())


def test_bottom10_group_is_no_greater_than_the_trace_mean():
    rng = np.random.default_rng(1)
    entropies = rng.random(80)
    scores = trace_confidence_scores(entropies, None)
    assert scores["conf_lowest_group_ent"] <= scores["conf_bottom10_group_ent"]
    assert scores["conf_bottom10_group_ent"] <= -entropies.mean() + 1e-9


def test_missing_or_nonfinite_token_series_yields_the_sentinel():
    assert all(np.isneginf(v) for v in trace_confidence_scores(None, None).values())
    bad = np.array([0.1, np.nan, 0.3])
    assert all(np.isneginf(v) for v in trace_confidence_scores(bad, None).values())


def test_logprob_scores_drop_out_when_only_entropies_are_available():
    """A missing logprob series must not silently fall back to the entropy form."""
    scores = trace_confidence_scores(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), None)
    assert np.isfinite(scores["conf_tail_q20_ent"])
    assert np.isneginf(scores["conf_tail_q20_lp"])


def _row(prompt_id, answer, conf):
    return {
        "prompt_id": prompt_id,
        "predicted_answer": answer,
        "logprob_score": conf,
        "conf_tail_q20_ent_score": conf,
        "conf_tail_q20_lp_score": conf,
    }


def test_vote_agreement_counts_only_parseable_traces():
    rows = {
        1: [_row(1, "a", -0.1), _row(1, "a", -0.2), _row(1, "b", -0.3), _row(1, "", -0.4)],
    }
    scores = prompt_vote_scores(rows)
    # Three parseable traces, two of which agree with the winner.
    assert scores[1]["vote_agreement"] == pytest.approx(2 / 3)


def test_weighted_vote_favours_the_confident_side():
    """Equal vote counts, unequal confidence -> the weighted score breaks the tie."""
    rows = {1: [_row(1, "a", -0.01), _row(1, "b", -5.0)]}
    scores = prompt_vote_scores(rows)
    assert scores[1]["vote_agreement"] == pytest.approx(0.5)
    assert scores[1]["conf_weighted_vote_lp"] > 0.9


def test_prompts_with_no_parseable_trace_get_the_sentinel():
    scores = prompt_vote_scores({1: [_row(1, "", -0.1), _row(1, None, -0.2)]})
    assert all(np.isneginf(value) for value in scores[1].values())


def test_decision_values_track_coverage_and_spend():
    outcomes = {i: float(i >= 5) for i in range(10)}
    scores = {i: float(i) for i in range(10)}  # perfectly ranked
    tokens = {i: 100.0 for i in range(10)}
    table = decision_values(scores, outcomes, tokens, rates=(0.0, 0.5))
    assert table["0.0"]["coverage"] == pytest.approx(1.0)
    assert table["0.0"]["selective_accuracy"] == pytest.approx(0.5)
    assert table["0.0"]["token_savings"] == pytest.approx(0.0)
    assert table["0.5"]["coverage"] == pytest.approx(0.5)
    assert table["0.5"]["selective_accuracy"] == pytest.approx(1.0)
    assert table["0.5"]["token_savings"] == pytest.approx(0.5)


def test_token_savings_reflect_which_prompts_are_dropped():
    """Savings are score-dependent: dropping long prompts saves more than short."""
    outcomes = {i: 1.0 for i in range(4)}
    tokens = {0: 10.0, 1: 10.0, 2: 1000.0, 3: 1000.0}
    drops_long = decision_values({0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0}, outcomes, tokens, rates=(0.5,))
    drops_short = decision_values({0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0}, outcomes, tokens, rates=(0.5,))
    assert drops_long["0.5"]["token_savings"] > drops_short["0.5"]["token_savings"]


def test_holm_is_step_down_and_monotone():
    adjusted = holm_correction({"a": 0.01, "b": 0.02, "c": 0.04})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] == pytest.approx(0.04)
    assert adjusted["c"] == pytest.approx(0.04)  # enforced monotone, not 0.04*1
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]


def test_holm_passes_through_missing_pvalues():
    adjusted = holm_correction({"a": 0.01, "b": None})
    assert adjusted["b"] is None


def test_calibration_is_zero_for_a_perfectly_calibrated_scorer():
    rng = np.random.default_rng(3)
    probabilities = rng.random(4000)
    outcomes = (rng.random(4000) < probabilities).astype(float)
    metrics = calibration_metrics(probabilities, outcomes)
    assert metrics["ece"] < 0.03
    assert metrics["n_calibrated"] == 4000


def test_calibration_penalises_a_confidently_wrong_scorer():
    outcomes = np.zeros(200)
    metrics = calibration_metrics(np.full(200, 0.95), outcomes)
    assert metrics["ece"] == pytest.approx(0.95)
    assert metrics["brier"] == pytest.approx(0.95**2)


def test_calibration_does_not_leak_labels_across_folds():
    """A score carrying no information must not calibrate better than the base rate.

    If the logistic were fitted on all prompts before scoring them, a noise score
    would pick up sample-specific structure and beat p*(1-p). Fitting strictly
    inside the training folds leaves it at the base rate.
    """
    from baselines.abstention_baselines import _oof_probabilities

    rng = np.random.default_rng(7)
    n, base = 500, 0.62
    outcomes = (rng.random(n) < base).astype(float)
    noise = rng.normal(size=n)
    probabilities = _oof_probabilities(noise, outcomes, seed=0)
    brier = calibration_metrics(probabilities, outcomes)["brier"]
    assert brier == pytest.approx(base * (1 - base), abs=0.02)


def test_calibration_rewards_an_informative_score():
    from baselines.abstention_baselines import _oof_probabilities

    rng = np.random.default_rng(8)
    n = 500
    signal = rng.normal(size=n)
    outcomes = (rng.random(n) < 1 / (1 + np.exp(-2 * signal))).astype(float)
    probabilities = _oof_probabilities(signal, outcomes, seed=0)
    assert calibration_metrics(probabilities, outcomes)["brier"] < 0.20


def test_calibration_ignores_uncalibrated_prompts():
    probabilities = np.array([0.5, np.nan, 0.5])
    metrics = calibration_metrics(probabilities, np.array([1.0, 0.0, 0.0]))
    assert metrics["n_calibrated"] == 2
