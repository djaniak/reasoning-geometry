import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from one_class_sweep import (
    evaluate_one_class_sweep,
    fit_projected_references,
    score_projected,
)


def test_projected_estimators_score_target_points_above_far_points():
    target = np.array(
        [[-0.2, 0.0], [0.0, 0.1], [0.2, -0.1], [0.1, 0.2]],
        dtype=float,
    )
    background = np.vstack([target, np.array([[4.0, 4.0], [5.0, 5.0]])])
    references = fit_projected_references(
        target,
        background,
        dimensions=[1, 2],
        ridge_scale=1e-3,
    )

    for method in (
        "centroid",
        "diagonal_mahalanobis",
        "empirical_ridge_mahalanobis",
        "ledoit_wolf_mahalanobis",
        "rmd_ledoit_wolf",
    ):
        near = score_projected(target[:1], references["2"], method)
        far = score_projected(np.array([[8.0, 8.0]]), references["2"], method)
        assert near > far


def test_one_class_sweep_keeps_incorrect_traces_out_of_target_fit_and_in_background():
    traces = []
    for index, correct in enumerate([True, False, True, False, True, False]):
        traces.append(
            {
                "idx": index,
                "trace_id": index,
                "is_correct": correct,
                "hiddens": {7: np.array([[index, index + 1]], dtype=float)},
            }
        )

    calls = []

    def fake_fit(correct_traces, background_traces, layer, dimensions, ridge_scale):
        calls.append(
            (
                {trace["trace_id"] for trace in correct_traces},
                {trace["trace_id"] for trace in background_traces},
            )
        )
        assert all(trace["is_correct"] for trace in correct_traces)
        assert {trace["is_correct"] for trace in background_traces} == {True, False}
        return {"dimensions": dimensions}

    def fake_score(trace, layer, references):
        base = 1.0 if trace["is_correct"] else 0.0
        return {
            str(dimension): {
                method: base
                for method in (
                    "centroid",
                    "diagonal_mahalanobis",
                    "empirical_ridge_mahalanobis",
                    "ledoit_wolf_mahalanobis",
                    "rmd_ledoit_wolf",
                    "normalized_rmd_ledoit_wolf",
                )
            }
            for dimension in references["dimensions"]
        }

    result = evaluate_one_class_sweep(
        traces,
        model="qwen",
        dataset="math500",
        layers=[7],
        dimensions=[1, 2],
        n_splits=3,
        seed=42,
        ridge_scale=1e-3,
        reference_fitter=fake_fit,
        trace_scorer=fake_score,
    )

    assert len(calls) == 3
    method = result["layers"]["7"]["dimensions"]["2"]["methods"]["centroid"]
    assert method["pooled_roc_auc"] == pytest.approx(1.0)
    assert method["n_eval"] == 6
    assert result["settings"]["no_dimension_selection"] is True
