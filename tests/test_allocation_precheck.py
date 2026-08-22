"""Tests for the allocation precheck (step 2).

The load-bearing claims are: the exhaustive subset accuracy really is exhaustive
and really uses the frozen majority-vote convention; the gain target is
non-monotone in difficulty; the constant baseline is held out; and the gate reads
the medians it says it reads.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest

from applications.allocation_precheck import (
    DIFFICULTY_NOT_GAIN_NEAR_ZERO,
    DIFFICULTY_NOT_GAIN_STRONG,
    FEATURE_SETS,
    GAIN_FROM,
    GAIN_TO,
    GATE_MIN_MODELS,
    SUBSET_SIZES,
    accuracy_curves,
    cached_pass_rates,
    crossfit_constant_predictions,
    crossfit_ridge_predictions,
    failure_mode_flags,
    gain_targets,
    gate_verdict,
    harness_checks,
    r2_against_constant,
    stage1_rows,
    subset_accuracy,
)
from applications.incremental_abstention import _plurality_outcome


def make_row(prompt_id, sample_id, answer, gold="7", **extra):
    row = {
        "prompt_id": prompt_id,
        "trace_id": sample_id,
        "sample_id": sample_id,
        "predicted_answer": answer,
        "gold_answer": gold,
        "logprob_score": -1.0,
        "is_correct": 1.0 if answer == gold else 0.0,
        "fold": 0,
        "layer": 3,
        "trace_length": 100,
        "length_score": -1.0,
        "entropy_score": -0.1,
        "rmd_tail_q20_score": 0.5,
    }
    row.update(extra)
    return row


def group(answers, prompt_id=0, gold="7"):
    return [make_row(prompt_id, i, answer, gold) for i, answer in enumerate(answers)]


# --- the exhaustive target ------------------------------------------------


def test_all_correct_prompt_has_accuracy_one_at_every_k():
    rows = group(["7"] * 8)
    for k in SUBSET_SIZES:
        assert subset_accuracy(rows, k) == 1.0


def test_all_wrong_prompt_has_accuracy_zero_at_every_k():
    rows = group(["3"] * 8)
    for k in SUBSET_SIZES:
        assert subset_accuracy(rows, k) == 0.0


def test_k_equal_one_is_the_share_of_correct_siblings():
    rows = group(["7", "7", "7", "3", "3", "3", "3", "3"])
    assert subset_accuracy(rows, 1) == pytest.approx(3 / 8)


def test_k_equal_n_is_the_frozen_plurality_outcome():
    """C(8,8) is the single subset containing every sibling -- the identity the
    harness check asserts at run time."""
    rows = group(["7", "7", "7", "7", "7", "3", "3", "3"])
    assert subset_accuracy(rows, 8) == _plurality_outcome(rows)


def test_subset_accuracy_enumerates_every_subset_not_a_sample():
    rows = group(["7", "7", "7", "3", "3", "3", "3", "3"])
    expected = np.mean(
        [_plurality_outcome(list(subset)) for subset in combinations(rows, 4)]
    )
    assert subset_accuracy(rows, 4) == pytest.approx(expected)


def test_majority_of_five_correct_beats_the_pass_rate():
    """The point of buying more samples: voting rescues a prompt below 8/8."""
    rows = group(["7"] * 5 + ["3"] * 3)
    assert subset_accuracy(rows, 1) == pytest.approx(5 / 8)
    assert subset_accuracy(rows, 8) == 1.0


def test_wrong_majority_makes_more_samples_worse():
    """Gain is signed: with a coherent wrong majority, voting destroys the prompt."""
    rows = group(["3", "3", "3", "3", "3", "7", "7", "7"])
    curves = accuracy_curves(rows)
    assert gain_targets(curves)[0] < 0.0


def test_unparsed_subset_scores_zero_not_nan():
    """The frozen automatic-failure rule, not a dropped subset."""
    rows = group(["", "", "7", "7", "7", "7", "7", "7"])
    assert subset_accuracy(rows, 2) == pytest.approx(
        np.mean([_plurality_outcome(list(s)) for s in combinations(rows, 2)])
    )
    assert subset_accuracy([make_row(0, 0, ""), make_row(0, 1, "")], 2) == 0.0


def test_subset_accuracy_is_nan_when_k_exceeds_the_cached_siblings():
    assert math.isnan(subset_accuracy(group(["7"] * 4), 8))


def test_subset_accuracy_rejects_k_below_one():
    assert math.isnan(subset_accuracy(group(["7"] * 4), 0))


def test_gain_target_is_a8_minus_a1():
    rows = group(["7", "7", "7", "7", "7", "3", "3", "3"])
    curves = accuracy_curves(rows)
    assert gain_targets(curves)[0] == pytest.approx(
        curves[0][GAIN_TO] - curves[0][GAIN_FROM]
    )


def test_gain_is_zero_at_both_extremes_of_difficulty():
    """The non-monotonicity the whole precheck rests on: 0/8 and 8/8 both gain nothing."""
    easy = gain_targets(accuracy_curves(group(["7"] * 8, prompt_id=0)))[0]
    hard = gain_targets(accuracy_curves(group(["3"] * 8, prompt_id=0)))[0]
    middling = gain_targets(accuracy_curves(group(["7"] * 5 + ["3"] * 3)))[0]
    assert easy == 0.0
    assert hard == 0.0
    assert middling > 0.0


# --- stage-1 selection ----------------------------------------------------


def test_stage1_takes_one_trace_per_prompt_at_the_requested_position():
    rows = group(["7", "3", "7", "3", "7", "3", "7", "3"], prompt_id=0) + group(
        ["3", "7", "3", "7", "3", "7", "3", "7"], prompt_id=1
    )
    selected = stage1_rows(rows, 1)
    assert len(selected) == 2
    assert [row["prompt_id"] for row in selected] == [0, 1]
    assert [row["sample_id"] for row in selected] == [1, 1]


def test_stage1_order_follows_sample_id_not_file_order():
    rows = list(reversed(group(["7", "3", "7", "3", "7", "3", "7", "3"])))
    assert stage1_rows(rows, 0)[0]["sample_id"] == 0


def test_stage1_draws_differ_from_each_other():
    rows = group(["7", "3", "7", "3", "7", "3", "7", "3"])
    answers = [stage1_rows(rows, position)[0]["predicted_answer"] for position in range(8)]
    assert answers == ["7", "3", "7", "3", "7", "3", "7", "3"]


def test_vote_agreement_is_not_a_stage1_feature():
    """A single sample has no siblings to agree with."""
    for spec in FEATURE_SETS.values():
        assert "vote_agreement" not in spec


def test_both_is_exactly_output_plus_geometry():
    assert set(FEATURE_SETS["both"]) == set(FEATURE_SETS["output"]) | set(
        FEATURE_SETS["geometry"]
    )


# --- readouts and the constant baseline -----------------------------------


def test_constant_baseline_uses_the_training_fold_mean_not_the_overall_mean():
    target = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    folds = np.array([0, 0, 0, 1, 1, 1])
    constant = crossfit_constant_predictions(target, folds)
    # Fold 0 is predicted from fold 1's mean (2/3), fold 1 from fold 0's (0.0).
    assert constant[:3] == pytest.approx([2 / 3] * 3)
    assert constant[3:] == pytest.approx([0.0] * 3)


def test_ridge_predictions_are_out_of_fold():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 1))
    target = np.where(np.arange(40) < 20, 5.0, -5.0)
    folds = np.where(np.arange(40) < 20, 0, 1)
    predictions = crossfit_ridge_predictions(x, target, folds)
    # Each fold is predicted by a model that only saw the other fold's constant.
    assert np.all(predictions[:20] < 0)
    assert np.all(predictions[20:] > 0)


def test_ridge_recovers_a_clean_linear_signal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 1))
    target = (2.0 * x[:, 0]).astype(float)
    folds = np.arange(200) % 5
    predictions = crossfit_ridge_predictions(x, target, folds)
    assert np.corrcoef(predictions, target)[0, 1] > 0.99


def test_ridge_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        crossfit_ridge_predictions(np.zeros((3, 1)), np.zeros(2), np.zeros(3))


def test_r2_is_zero_when_the_readout_equals_the_constant():
    target = np.array([0.0, 1.0, 2.0, 3.0])
    constant = np.full(4, 1.5)
    assert r2_against_constant(constant, constant, target) == pytest.approx(0.0)


def test_r2_is_negative_when_the_readout_is_worse_than_the_constant():
    target = np.array([0.0, 1.0, 2.0, 3.0])
    constant = np.full(4, 1.5)
    worse = np.array([3.0, 2.0, 1.0, 0.0])
    assert r2_against_constant(worse, constant, target) < 0.0


def test_r2_is_nan_when_the_constant_is_already_perfect():
    target = np.full(5, 2.0)
    constant = np.full(5, 2.0)
    assert math.isnan(r2_against_constant(constant, constant, target))


# --- harness checks -------------------------------------------------------


def test_harness_check_passes_when_a8_matches_the_frozen_outcome():
    rows = group(["7", "7", "7", "7", "7", "3", "3", "3"])
    curves = accuracy_curves(rows)
    features = {0: {"outcome": _plurality_outcome(rows)}}
    checks = harness_checks(curves, features, cached_pass_rates(rows), [0])
    assert checks["a8_equals_frozen_outcome"] is True
    assert checks["a1_vs_cached_is_correct"]["n_prompts_differing"] == 0


def test_harness_check_raises_when_a8_disagrees_with_the_frozen_outcome():
    rows = group(["7"] * 8)
    curves = accuracy_curves(rows)
    with pytest.raises(AssertionError, match="frozen prompt outcome"):
        harness_checks(curves, {0: {"outcome": 0.0}}, cached_pass_rates(rows), [0])


def test_a1_disagreement_with_the_cached_column_is_reported_not_raised():
    """`a(p,1)` re-derives the match through the frozen parser; `is_correct` is
    the collector's stored verdict, and drift between them is a finding, not a crash."""
    rows = group(["7", "7", "3", "3", "3", "3", "3", "3"])
    for row in rows:
        row["is_correct"] = 1.0
    curves = accuracy_curves(rows)
    features = {0: {"outcome": _plurality_outcome(rows)}}
    checks = harness_checks(curves, features, cached_pass_rates(rows), [0])
    assert checks["a1_vs_cached_is_correct"]["n_prompts_differing"] == 1
    assert checks["a1_vs_cached_is_correct"]["max_abs_difference"] == pytest.approx(0.75)


def test_cached_pass_rate_counts_unparsed_traces_in_the_denominator():
    rows = group(["7", "7", "", "", "", "", "", ""])
    assert cached_pass_rates(rows)[0] == pytest.approx(2 / 8)


# --- the gate -------------------------------------------------------------


def body(label, r2_geometry, spearman_gain, *, pass_rate=0.5, gain=0.0):
    def spread(value):
        return {"median": value, "min": value, "max": value, "n": 8}

    return {
        "label": label,
        "across_draws": {
            "spearman": {name: spread(0.1) for name in FEATURE_SETS},
            "r2_vs_constant": {
                "geometry": spread(r2_geometry),
                "output": spread(0.0),
                "both": spread(r2_geometry),
            },
            "spearman_gain_from_geometry": spread(spearman_gain),
            "auroc_vs_prompt_outcome": spread(0.7),
            "auroc_vs_own_trace_correct": spread(0.65),
            "spearman_vs_pass_rate": spread(pass_rate),
            "spearman_vs_gain": spread(gain),
        },
    }


def test_gate_needs_both_conditions_on_one_model():
    only_r2 = gate_verdict([body("a", 0.05, -0.01)])["per_model"]["a"]
    only_spearman = gate_verdict([body("a", -0.05, 0.01)])["per_model"]["a"]
    both = gate_verdict([body("a", 0.05, 0.01)])["per_model"]["a"]
    assert only_r2["passes"] is False
    assert only_spearman["passes"] is False
    assert both["passes"] is True


def test_gate_opens_at_the_declared_number_of_models():
    passing = [body(f"m{i}", 0.05, 0.01) for i in range(GATE_MIN_MODELS)]
    failing = [body("z", -0.05, -0.01)]
    assert gate_verdict(passing + failing)["passes"] is True
    assert gate_verdict(passing[:-1] + failing)["passes"] is False


def test_gate_is_strict_at_zero():
    """R^2 of exactly zero means the readout matched the constant, which is not
    beating it."""
    assert gate_verdict([body("a", 0.0, 0.01)])["per_model"]["a"]["passes"] is False
    assert gate_verdict([body("a", 0.05, 0.0)])["per_model"]["a"]["passes"] is False


def test_failing_gate_says_step_three_is_not_run():
    verdict = gate_verdict([body("a", -0.1, -0.1), body("b", -0.1, -0.1)])
    assert verdict["passes"] is False
    assert "not run" in verdict["consequence"]


def test_gate_reads_the_paired_spearman_difference_not_the_two_medians():
    """The paired difference is a separate summary, so a model can be flagged as
    adding nothing even when both marginal medians look similar."""
    entry = body("a", 0.05, -0.02)
    entry["across_draws"]["spearman"]["both"] = {
        "median": 0.5, "min": 0.5, "max": 0.5, "n": 8
    }
    entry["across_draws"]["spearman"]["output"] = {
        "median": 0.1, "min": 0.1, "max": 0.1, "n": 8
    }
    assert gate_verdict([entry])["per_model"]["a"]["geometry_adds_over_output"] is False


# --- the named failure mode ----------------------------------------------


def test_difficulty_not_gain_fires_on_strong_difficulty_and_flat_gain():
    flags = failure_mode_flags(
        [body("a", -0.1, -0.1, pass_rate=DIFFICULTY_NOT_GAIN_STRONG + 0.1, gain=0.0)]
    )
    assert flags["a"] is True


def test_difficulty_not_gain_is_silent_when_geometry_tracks_the_gain():
    flags = failure_mode_flags(
        [
            body(
                "a",
                0.1,
                0.1,
                pass_rate=0.5,
                gain=DIFFICULTY_NOT_GAIN_NEAR_ZERO + 0.1,
            )
        ]
    )
    assert flags["a"] is False


def test_difficulty_not_gain_is_silent_when_geometry_reads_neither():
    flags = failure_mode_flags([body("a", -0.1, -0.1, pass_rate=0.01, gain=0.0)])
    assert flags["a"] is False


def test_difficulty_not_gain_uses_magnitude_so_sign_does_not_matter():
    flags = failure_mode_flags(
        [body("a", -0.1, -0.1, pass_rate=-(DIFFICULTY_NOT_GAIN_STRONG + 0.1), gain=0.0)]
    )
    assert flags["a"] is True


def test_failure_mode_flag_does_not_feed_the_gate():
    entry = body("a", 0.05, 0.01, pass_rate=0.9, gain=0.0)
    assert failure_mode_flags([entry])["a"] is True
    assert gate_verdict([entry])["per_model"]["a"]["passes"] is True
