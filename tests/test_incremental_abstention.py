import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from incremental_abstention import (
    FEATURE_NAMES,
    _load_prompt_states,
    aggregate_prompt_features,
    _load_exact_prompt_scores,
    _model_specs,
    paired_bootstrap_delta,
    prompt_accounting,
    prompt_metrics,
    crossfit_logistic_predictions,
)


def _row(prompt_id, trace_id, answer, *, fold, length=100, entropy=-1.0, logprob=-0.2, rmd=0.0):
    return {
        "prompt_id": prompt_id,
        "trace_id": trace_id,
        "fold": fold,
        "predicted_answer": answer,
        "gold_answer": "a",
        "trace_length": length,
        "length_score": -np.log1p(length),
        "entropy_score": entropy,
        "logprob_score": logprob,
        "rmd_tail_q20_score": rmd,
    }


def test_accounting_marks_all_unparsed_as_automatic_failure_and_counts_caps():
    rows = [
        _row(0, 0, "", fold=0, length=8),
        _row(0, 1, "", fold=0, length=10),
        _row(1, 0, "a", fold=1, length=10),
        _row(1, 1, "b", fold=1, length=11),
    ]
    accounting = prompt_accounting(rows, max_new_tokens=10, expected_traces=2)
    assert accounting[0]["automatic_failure"] is True
    assert accounting[0]["outcome"] == 0.0
    assert accounting[0]["unparsed_count"] == 2
    assert accounting[1]["automatic_failure"] is False
    assert accounting[1]["cap_count"] == 2
    assert accounting[1]["valid_plurality"] is True


def test_feature_aggregation_uses_parseable_vote_and_keeps_failure_counts():
    rows = [
        _row(0, 0, "a", fold=0, length=100, entropy=-1, logprob=-0.1, rmd=0.4),
        _row(0, 1, "a", fold=0, length=110, entropy=-2, logprob=-0.2, rmd=0.2),
        _row(0, 2, "b", fold=0, length=120, entropy=-3, logprob=-0.3, rmd=0.0),
        _row(0, 3, "", fold=0, length=130, entropy=-4, logprob=-0.4, rmd=-0.2),
    ]
    features = aggregate_prompt_features(rows, max_new_tokens=130, expected_traces=4)[0]
    assert tuple(features[name] for name in FEATURE_NAMES[:4]) == pytest.approx(
        (-np.mean(np.log1p([100, 110, 120, 130])), -2.5, -0.25, 2 / 3)
    )
    assert features["rmd_tail_q20"] == pytest.approx(0.1)
    assert features["cap_count"] == 1
    assert features["unparsed_count"] == 1


def test_crossfit_predictions_respect_explicit_prompt_folds():
    features = np.asarray([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]])
    outcomes = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    folds = np.asarray([0, 1, 2, 0, 1, 2])
    probabilities = crossfit_logistic_predictions(features, outcomes, folds, seed=3)
    assert np.all(np.isfinite(probabilities))
    assert probabilities[0] < probabilities[3]
    assert probabilities[1] < probabilities[4]


def test_prompt_metrics_report_auacc_and_conventional_aurc():
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
    outcomes = np.asarray([0.0, 0.0, 1.0, 1.0])
    metrics = prompt_metrics(probabilities, outcomes)
    assert metrics["auacc"] > 0.5
    assert metrics["aurc"] < 0.5
    assert metrics["brier"] < 0.1
    assert metrics["log_loss"] < 0.25


def test_paired_bootstrap_delta_is_reproducible_and_positive_for_better_ranker():
    left = np.asarray([0.1, 0.2, 0.8, 0.9])
    right = np.asarray([0.4, 0.6, 0.5, 0.7])
    outcomes = np.asarray([0.0, 0.0, 1.0, 1.0])
    first = paired_bootstrap_delta(left, right, outcomes, n_bootstrap=300, seed=9)
    second = paired_bootstrap_delta(left, right, outcomes, n_bootstrap=300, seed=9)
    assert first == second
    assert first["point_estimate"] > 0


def test_prompt_state_loader_averages_exact_pilot_trace_states(tmp_path):
    path = tmp_path / "pilot.npz"
    np.savez(
        path,
        prompt_ids=np.asarray([4]),
        prompt_hidden_L21=np.asarray([[[1.0, 3.0], [3.0, 5.0]]], dtype=np.float16),
    )
    states = _load_prompt_states(path, 21)
    assert np.allclose(states[4], [2.0, 4.0])


def test_exact_scores_are_aggregated_and_exposed_as_incremental_models(tmp_path):
    path = tmp_path / "exact.npz"
    rows = np.asarray(
        [
            {"prompt_id": 0, "deepconf_global": 10.0, "deepconf_tail_q20": 11.0},
            {"prompt_id": 0, "deepconf_global": 12.0, "deepconf_tail_q20": 13.0},
        ],
        dtype=object,
    )
    np.savez(path, trace_summaries=rows)
    scores = _load_exact_prompt_scores(path)
    assert scores[0]["deepconf_global"] == pytest.approx(11.0)
    specs = _model_specs({0: {"prompt_only_geometry": np.nan, "deepconf_global": 11.0}})
    assert "B0_plus_DeepConf_tail_q20" in specs
