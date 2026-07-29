import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze import get_hidden_dim, resolve_cross_model_layer_map


def test_resolve_cross_model_layer_map_uses_sparse_layer_order_for_different_depths():
    assert resolve_cross_model_layer_map([8, 16, 24], [7, 14, 21]) == {
        8: 7,
        16: 14,
        24: 21,
    }


def test_resolve_cross_model_layer_map_prefers_exact_layer_matches():
    assert resolve_cross_model_layer_map([7, 14, 21], [0, 7, 14, 21, 27]) == {
        7: 7,
        14: 14,
        21: 21,
    }


def test_get_hidden_dim_reports_missing_layer():
    traces = [{"hiddens": {7: np.zeros((2, 3584))}}]

    assert get_hidden_dim(traces, 8) is None


def test_get_hidden_dim_returns_hidden_width():
    traces = [{"hiddens": {8: np.zeros((2, 4096))}}]

    assert get_hidden_dim(traces, 8) == 4096
