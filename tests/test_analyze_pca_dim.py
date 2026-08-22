import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.analyze import (
    _concatenate_hidden_tokens,
    _fit_lw_precision,
    _prepare_hidden_tokens,
    _project_trace_hiddens,
    extend_reference_with_background,
    parse_pca_dim_arg,
    resolve_pca_n_components,
)


def test_parse_pca_dim_arg_accepts_max_and_all_aliases():
    assert parse_pca_dim_arg("max") == "max"
    assert parse_pca_dim_arg("all") == "max"


def test_resolve_pca_n_components_uses_full_rank_for_max():
    hiddens = np.zeros((12, 7))
    assert resolve_pca_n_components(hiddens, "max") == 7


def test_resolve_pca_n_components_keeps_numeric_request():
    hiddens = np.zeros((12, 7))
    assert resolve_pca_n_components(hiddens, 5) == 5


def test_memory_bounded_hidden_helpers_match_explicit_concatenation():
    traces = [
        {"hiddens": {7: np.arange(12, dtype=np.float32).reshape(3, 4)}},
        {"hiddens": {7: np.arange(8, dtype=np.float32).reshape(2, 4) + 10}},
    ]
    expected = np.asarray(
        np.concatenate([trace["hiddens"][7] for trace in traces], axis=0),
        dtype=np.float64,
    )

    concatenated = _concatenate_hidden_tokens(traces, layer=7)

    class IdentityProjection:
        def transform(self, values):
            return values

    projected = _project_trace_hiddens(
        IdentityProjection(), traces, layer=7
    )

    assert concatenated.dtype == np.float64
    np.testing.assert_array_equal(concatenated, expected)
    np.testing.assert_array_equal(projected, expected)


def test_bounded_background_reference_matches_full_matrix_reference():
    rng = np.random.default_rng(42)
    correct = rng.normal(size=(20, 6)).astype(np.float32)
    background_traces = [
        {"hiddens": {7: rng.normal(size=(7, 6)).astype(np.float32)}},
        {"hiddens": {7: rng.normal(size=(9, 6)).astype(np.float32)}},
    ]
    pca = PCA(n_components=3, random_state=42).fit(correct.astype(np.float64))
    base_ref = (pca, np.zeros(3), np.eye(3))

    explicit_hiddens = _prepare_hidden_tokens(
        np.concatenate(
            [trace["hiddens"][7] for trace in background_traces], axis=0
        )
    )
    explicit_projected = pca.transform(explicit_hiddens)
    expected_mu = explicit_projected.mean(axis=0)
    expected_precision = _fit_lw_precision(explicit_projected - expected_mu)

    _, _, _, actual_mu, actual_precision = extend_reference_with_background(
        base_ref, background_traces, layer=7
    )

    np.testing.assert_allclose(actual_mu, expected_mu, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        actual_precision, expected_precision, rtol=1e-12, atol=1e-12
    )
