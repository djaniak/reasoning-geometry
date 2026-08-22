import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controls.orgad_agreement_control import (
    MIN_STRATUM_CLASS,
    VOTELESS_FEATURE_NAMES,
    _readout_specs,
    agreement_level_counts,
    agreement_strata,
    crossfit_residuals,
    spearman,
    stratum_readout,
)


def test_spearman_sees_a_monotone_relation_pearson_understates():
    """The features are bounded and lumpy, so the rank version is the honest one."""
    left = np.asarray([1.0, 2.0, 3.0, 4.0, 100.0])
    right = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])

    assert spearman(left, right) == pytest.approx(1.0)


def test_spearman_ignores_pairs_where_either_side_is_missing():
    left = np.asarray([1.0, 2.0, np.nan, 4.0])
    right = np.asarray([1.0, 2.0, 9.0, 4.0])

    assert spearman(left, right) == pytest.approx(1.0)


def test_residuals_are_computed_out_of_fold():
    """An in-sample fit would absorb on these very prompts what we are testing for."""
    covariate = np.asarray([0.0, 1.0, 2.0, 3.0])
    # Held-out folds each see the *other* fold's fit, so a relation that holds only
    # within one fold cannot be residualized away.
    target = 2.0 * covariate
    folds = np.asarray([0, 0, 1, 1])

    residuals = crossfit_residuals(target, covariate, folds)

    assert np.all(np.isfinite(residuals))
    assert residuals == pytest.approx(np.zeros(4))


def test_a_fold_with_a_constant_covariate_yields_no_residual_rather_than_a_crash():
    covariate = np.asarray([1.0, 1.0, 0.0, 3.0])
    target = np.asarray([5.0, 6.0, 0.0, 3.0])
    folds = np.asarray([1, 1, 0, 0])

    residuals = crossfit_residuals(target, covariate, folds)

    # Fold 1's training half (fold 0) varies, so those two are residualized; fold 0's
    # training half is constant, so it is left missing instead of silently zeroed.
    assert np.all(np.isfinite(residuals[folds == 1]))
    assert np.all(np.isnan(residuals[folds == 0]))


def test_the_unanimous_stratum_is_exactly_the_prompts_with_no_disagreement():
    agreement = np.asarray([1.0, 0.875, 0.5, 1.0, np.nan])

    strata = agreement_strata(agreement)

    assert strata["unanimous"].tolist() == [True, False, False, True, False]
    assert strata["split"].tolist() == [False, True, True, False, False]


def test_a_stratum_with_too_few_of_one_class_is_refused_not_estimated():
    """A point estimate on three wrong prompts reads as a null; it is no data."""
    outcomes = np.concatenate([np.ones(40), np.zeros(3)])
    scores = np.arange(43, dtype=float)
    mask = np.ones(43, dtype=bool)

    body = stratum_readout(scores, outcomes, mask, n_bootstrap=50)

    assert body["reported"] is False
    assert body["auroc"] is None
    assert body["n_wrong"] == 3 < MIN_STRATUM_CLASS


def test_a_stratum_with_both_classes_present_reports_an_interval():
    outcomes = np.concatenate([np.ones(20), np.zeros(20)])
    scores = np.concatenate([np.arange(20, 40), np.arange(20)]).astype(float)

    body = stratum_readout(scores, outcomes, np.ones(40, dtype=bool), n_bootstrap=100)

    assert body["reported"] is True
    assert body["auroc"]["point_estimate"] == pytest.approx(1.0)
    assert body["base_accuracy"] == pytest.approx(0.5)


def test_stratum_readout_scores_only_the_masked_prompts():
    outcomes = np.asarray([1.0] * 10 + [0.0] * 10 + [1.0] * 100)
    scores = np.concatenate([np.ones(20), np.zeros(100)])
    mask = np.arange(120) < 20

    body = stratum_readout(scores, outcomes, mask, n_bootstrap=50)

    assert body["n_prompts"] == 20
    assert body["n_correct"] == 10 and body["n_wrong"] == 10


def test_agreement_levels_are_reported_with_their_own_accuracy():
    agreement = np.asarray([1.0, 1.0, 0.5])
    outcomes = np.asarray([1.0, 0.0, 1.0])

    table = agreement_level_counts(agreement, outcomes)

    assert [row["agreement"] for row in table] == [0.5, 1.0]
    assert table[1] == {"agreement": 1.0, "n_prompts": 2, "accuracy": pytest.approx(0.5)}


def test_the_substitution_readout_puts_geometry_where_the_vote_was():
    """The proxy reading predicts the swap is free; it can only be checked if the
    swapped baseline really has the vote removed rather than sitting alongside it."""
    specs = _readout_specs()

    assert "vote_agreement" not in specs["B0_rmd_for_vote"]
    assert "rmd_tail_q20" in specs["B0_rmd_for_vote"]
    assert len(specs["B0_rmd_for_vote"]) == len(specs["B0"])
    assert specs["B0_voteless"] == VOTELESS_FEATURE_NAMES
    assert specs["B1"] == specs["B0"] + ("rmd_tail_q20",)
