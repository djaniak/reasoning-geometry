from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.analyze import (  # noqa: E402
    compute_relative_mahal_distances,
    evaluate_foldwise_low_rank_subspace_sweep,
    fit_mahalanobis_reference,
    fit_relative_mahalanobis_reference,
)


def make_trace(tokens: np.ndarray, is_correct: bool, idx: int) -> dict:
    n_tokens = tokens.shape[0]
    return {
        "trace_id": idx,
        "idx": idx,
        "sample_id": idx,
        "is_correct": is_correct,
        "entropies": np.linspace(0.2, 0.8, n_tokens, dtype=np.float64),
        "hiddens": {0: tokens.astype(np.float64)},
    }


def make_synthetic_traces(n_per_class: int = 12) -> list[dict]:
    rng = np.random.default_rng(7)
    traces = []
    for idx in range(n_per_class):
        tokens = rng.normal(loc=[2.0, 0.0], scale=[0.2, 0.1], size=(6, 2))
        traces.append(make_trace(tokens, True, idx))
    for idx in range(n_per_class):
        tokens = rng.normal(loc=[0.0, 2.0], scale=[0.1, 0.2], size=(6, 2))
        traces.append(make_trace(tokens, False, n_per_class + idx))
    return traces


def test_normalized_mahalanobis_is_scale_invariant():
    correct_traces = [
        make_trace(np.array([[1.0, 0.0], [1.1, 0.1], [0.9, -0.1]]), True, 0),
        make_trace(np.array([[2.0, 0.0], [2.1, 0.1], [1.9, -0.1]]), True, 1),
        make_trace(np.array([[3.0, 0.0], [3.1, 0.1], [2.9, -0.1]]), True, 2),
    ]

    raw_ref = fit_mahalanobis_reference(correct_traces, layer=0, pca_dim=2, normalize_input=False)
    norm_ref = fit_mahalanobis_reference(correct_traces, layer=0, pca_dim=2, normalize_input=True)

    same_direction_small = np.array([[1.0, 0.0], [1.0, 0.0]])
    same_direction_large = np.array([[10.0, 0.0], [10.0, 0.0]])

    raw_small = raw_ref[0].transform(same_direction_small)
    raw_large = raw_ref[0].transform(same_direction_large)
    assert not np.allclose(raw_small, raw_large)

    from analysis.analyze import compute_mahal_distances  # noqa: E402

    norm_small = compute_mahal_distances(
        same_direction_small, *norm_ref, normalize_input=True
    )
    norm_large = compute_mahal_distances(
        same_direction_large, *norm_ref, normalize_input=True
    )
    assert np.allclose(norm_small, norm_large, atol=1e-6)


def test_relative_mahalanobis_scores_incorrect_higher_than_correct():
    traces = make_synthetic_traces()
    correct_traces = [trace for trace in traces if trace["is_correct"]]

    ref = fit_relative_mahalanobis_reference(
        correct_traces,
        traces,
        layer=0,
        pca_dim=2,
        normalize_input=False,
    )

    correct_probe = np.array([[2.1, 0.0], [1.9, 0.1]])
    incorrect_probe = np.array([[0.1, 2.1], [0.0, 1.9]])

    correct_score = compute_relative_mahal_distances(correct_probe, *ref).mean()
    incorrect_score = compute_relative_mahal_distances(incorrect_probe, *ref).mean()

    assert incorrect_score > correct_score


def test_low_rank_subspace_sweep_returns_centroid_and_mahal_results():
    traces = make_synthetic_traces()
    y = np.array([1 if trace["is_correct"] else 0 for trace in traces], dtype=int)
    dummy_X = np.zeros((len(traces), 1), dtype=np.float64)
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(dummy_X, y))

    sweep = evaluate_foldwise_low_rank_subspace_sweep(
        traces,
        layer=0,
        pca_dim=2,
        subspace_ranks=[1, 2],
        y=y,
        fold_indices=folds,
    )

    assert set(sweep) == {"1", "2"}
    for rank in ("1", "2"):
        rank_result = sweep[rank]
        assert rank_result["requested_rank"] == int(rank)
        assert rank_result["actual_dims"]
        for key in ("centroid_combined", "mahalanobis_combined"):
            auc = rank_result[key]["roc_auc_mean"]
            assert not math.isnan(auc)
            assert auc >= 0.5
