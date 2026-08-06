import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepconf_asymmetry import (
    DEEPCONF_FEATURES,
    auroc,
    bootstrap_auroc,
    cohens_d,
    load_all_deepconf_scores,
    trace_length_summary,
)
from incremental_abstention import _auacc


def test_auroc_ignores_the_base_rate_but_auacc_does_not():
    """The reason the module exists: AUACC credits a readout for the base rate."""
    scores = np.array([3.0, 4.0, 1.0, 2.0])
    outcomes = np.array([1.0, 1.0, 0.0, 0.0])
    # Same ranking structure, positives duplicated so the base rate rises to 0.75.
    skewed_scores = np.array([3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 1.0, 2.0])
    skewed_outcomes = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    assert auroc(scores, outcomes) == pytest.approx(auroc(skewed_scores, skewed_outcomes))
    assert _auacc(skewed_scores, skewed_outcomes) > _auacc(scores, outcomes)


def test_auroc_endpoints_and_chance():
    outcomes = np.array([1.0, 1.0, 0.0, 0.0])
    assert auroc(np.array([4.0, 3.0, 2.0, 1.0]), outcomes) == pytest.approx(1.0)
    assert auroc(np.array([1.0, 2.0, 3.0, 4.0]), outcomes) == pytest.approx(0.0)
    # A feature carrying no information sits at chance rather than at an endpoint.
    assert auroc(np.full(4, 7.0), outcomes) == pytest.approx(0.5)


def test_tied_scores_do_not_depend_on_input_order():
    outcomes = np.array([1.0, 0.0, 1.0, 0.0])
    scores = np.array([2.0, 2.0, 1.0, 3.0])
    reordered = np.array([3, 2, 1, 0])

    assert auroc(scores, outcomes) == pytest.approx(auroc(scores[reordered], outcomes[reordered]))


def test_auroc_is_nan_when_one_class_is_missing():
    assert np.isnan(auroc(np.array([1.0, 2.0]), np.array([1.0, 1.0])))


def test_cohens_d_is_positive_when_correct_prompts_score_higher():
    scores = np.array([5.0, 6.0, 1.0, 2.0])
    outcomes = np.array([1.0, 1.0, 0.0, 0.0])
    assert cohens_d(scores, outcomes) > 0
    assert cohens_d(-scores, outcomes) < 0


def test_bootstrap_interval_brackets_the_point_estimate_and_is_seeded():
    rng = np.random.default_rng(0)
    outcomes = (rng.random(200) < 0.6).astype(float)
    scores = outcomes + rng.normal(0, 1.0, 200)

    first = bootstrap_auroc(scores, outcomes, n_bootstrap=200, seed=42)
    second = bootstrap_auroc(scores, outcomes, n_bootstrap=200, seed=42)

    assert first == second
    assert first["ci_low"] <= first["point_estimate"] <= first["ci_high"]


def test_all_four_deepconf_statistics_are_read_and_averaged_per_prompt(tmp_path):
    """The comparison on record read only two of the four; the omission is checkable."""
    rows = [
        {"prompt_id": 7, "trace_id": 0, **{key: 1.0 for key in DEEPCONF_FEATURES}},
        {"prompt_id": 7, "trace_id": 1, **{key: 3.0 for key in DEEPCONF_FEATURES}},
        {"prompt_id": 9, "trace_id": 0, **{key: 5.0 for key in DEEPCONF_FEATURES}},
    ]
    path = tmp_path / "exact.npz"
    np.savez(path, trace_summaries=np.array(rows, dtype=object))

    scores = load_all_deepconf_scores(path)

    assert set(scores) == {7, 9}
    assert set(scores[7]) == set(DEEPCONF_FEATURES)
    assert "bottom10_group_confidence" in scores[7]
    assert scores[7]["deepconf_global"] == pytest.approx(2.0)
    assert scores[9]["deepconf_global"] == pytest.approx(5.0)


def test_tail_window_is_reported_in_tokens_not_as_a_fraction():
    """Two models collected at different budgets average over different windows."""
    rows = [{"prompt_id": 1, "trace_length": 1000.0}, {"prompt_id": 1, "trace_length": 2000.0}]

    summary = trace_length_summary(rows, [1])

    assert summary["n_traces"] == 2
    assert summary["median_trace_length"] == pytest.approx(1500.0)
    assert summary["median_tail_window_tokens"] == pytest.approx(300.0)


def test_traces_outside_the_population_are_excluded():
    rows = [{"prompt_id": 1, "trace_length": 100.0}, {"prompt_id": 2, "trace_length": 9999.0}]

    assert trace_length_summary(rows, [1])["n_traces"] == 1
