import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import selective_prediction


def _traces():
    traces = []
    for index, correct in enumerate([True, True, False, False, True, False]):
        traces.append(
            {
                "idx": index,
                "trace_id": index,
                "is_correct": correct,
                "entropies": np.array([0.2, 0.8], dtype=float),
                "mean_logprob": -0.5,
                "hiddens": {7: np.array([[index], [index + 1]], dtype=float)},
            }
        )
    return traces


def test_selective_prediction_fits_distinct_raw_rmd_and_norm_rmd_probes(monkeypatch):
    fitted_feature_markers = []

    monkeypatch.setattr(
        selective_prediction,
        "fit_mahalanobis_reference_safe",
        lambda *args, **kwargs: ("raw",),
    )

    def fake_relative_ref(*args, normalize_input=False, **kwargs):
        return ("norm" if normalize_input else "rmd", None, None, None, None)

    monkeypatch.setattr(
        selective_prediction,
        "fit_relative_mahalanobis_reference_safe",
        fake_relative_ref,
    )
    monkeypatch.setattr(
        selective_prediction,
        "compute_mahal_distances",
        lambda hiddens, marker: np.full(len(hiddens), 1.0),
    )

    def fake_relative_distance(hiddens, marker, *unused, normalize_input=False):
        value = 20.0 if normalize_input or marker == "norm" else 10.0
        return np.full(len(hiddens), value)

    monkeypatch.setattr(
        selective_prediction,
        "compute_relative_mahal_distances",
        fake_relative_distance,
    )

    def fake_fit(X, y):
        fitted_feature_markers.append(float(np.mean(X[:, 5])))
        return object()

    monkeypatch.setattr(selective_prediction, "_fit_logistic", fake_fit)
    monkeypatch.setattr(
        selective_prediction,
        "_predict_proba",
        lambda model, X: np.linspace(0.1, 0.9, len(X)),
    )

    result = selective_prediction.evaluate_selective_prediction(
        traces=_traces(),
        layers=[7],
        pca_dim=2,
        n_splits=3,
        cv_random_state=42,
        min_coverage=0.3,
        operating_points=[0.6],
        dataset_label="math500",
    )

    assert "combined_lr_L7" in result["scorers"]
    assert "rmd_combined_lr_L7" in result["scorers"]
    assert "norm_rmd_combined_lr_L7" in result["scorers"]
    assert result["scorers"]["rmd_combined_lr_L7"]["n_eval"] == 6
    assert result["scorer_definitions"]["combined_lr"].endswith(
        "raw Mahalanobis features"
    )
    rounded_markers = {round(value) for value in fitted_feature_markers}
    assert {1, 10, 20}.issubset(rounded_markers)
