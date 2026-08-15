import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry.pca_ablation import aggregate_pca_ablation, parse_pca_dims


def _write_result(path: Path, pca_dim: int, layer_metrics: dict):
    payload = {
        "dataset": "gsm8k",
        "n_correct": 100,
        "n_incorrect": 40,
        "settings": {
            "pca_dim": pca_dim,
            "cv_random_state": 42,
            "analysis_family": "base",
        },
        "entropy_baseline": {
            "roc_auc_mean": 0.75,
            "roc_auc_std": 0.01,
        },
        "layers": layer_metrics,
    }
    path.write_text(json.dumps(payload))


def test_parse_pca_dims_sorts_and_deduplicates():
    assert parse_pca_dims("128,32,128,max,all") == ["128", "32", "max"]


def test_parse_pca_dims_rejects_non_positive_values():
    with pytest.raises(ValueError):
        parse_pca_dims("32,0,64")


def test_aggregate_pca_ablation_builds_layer_dim_table(tmp_path: Path):
    in_32 = tmp_path / "gsm8k_pca32_base_results.json"
    in_64 = tmp_path / "gsm8k_pca64_base_results.json"

    _write_result(
        in_32,
        32,
        {
            "7": {
                "mahalanobis_only": {"roc_auc_mean": 0.61, "roc_auc_std": 0.02},
                "combined": {"roc_auc_mean": 0.78, "roc_auc_std": 0.02},
                "delta_vs_entropy": 0.03,
                "length_controlled_delta": 0.01,
            }
        },
    )
    _write_result(
        in_64,
        64,
        {
            "7": {
                "mahalanobis_only": {"roc_auc_mean": 0.62, "roc_auc_std": 0.02},
                "combined": {"roc_auc_mean": 0.80, "roc_auc_std": 0.01},
                "delta_vs_entropy": 0.05,
                "length_controlled_delta": 0.02,
            }
        },
    )

    result = aggregate_pca_ablation([in_32, in_64], expected_dims=[32, 64], model="qwen")

    layer7 = result["layers"]["7"]
    assert result["settings"]["pca_dims"] == [32, 64]
    assert result["model"] == "qwen"
    assert layer7["best_dim_by_combined_auc"] == 64
    assert layer7["combined_auc_monotone_non_decreasing"] is True
    assert layer7["dims"]["32"]["combined"]["roc_auc_mean"] == 0.78
    assert layer7["dims"]["64"]["combined"]["roc_auc_mean"] == 0.80


def test_aggregate_pca_ablation_fails_if_requested_dim_is_missing(tmp_path: Path):
    in_32 = tmp_path / "gsm8k_pca32_base_results.json"
    _write_result(
        in_32,
        32,
        {
            "7": {
                "mahalanobis_only": {"roc_auc_mean": 0.61, "roc_auc_std": 0.02},
                "combined": {"roc_auc_mean": 0.78, "roc_auc_std": 0.02},
                "delta_vs_entropy": 0.03,
                "length_controlled_delta": 0.01,
            }
        },
    )

    with pytest.raises(ValueError):
        aggregate_pca_ablation([in_32], expected_dims=[32, 64], model="qwen")


def test_aggregate_pca_ablation_supports_max_alias(tmp_path: Path):
    in_32 = tmp_path / "gsm8k_pca32_base_results.json"
    in_all = tmp_path / "gsm8k_pcamax_base_results.json"

    _write_result(
        in_32,
        32,
        {
            "7": {
                "mahalanobis_only": {"roc_auc_mean": 0.60, "roc_auc_std": 0.02},
                "combined": {"roc_auc_mean": 0.76, "roc_auc_std": 0.02},
                "delta_vs_entropy": 0.01,
                "length_controlled_delta": 0.00,
            }
        },
    )

    payload_all = {
        "dataset": "gsm8k",
        "n_correct": 100,
        "n_incorrect": 40,
        "settings": {
            "pca_dim": "all",
            "cv_random_state": 42,
            "analysis_family": "base",
        },
        "entropy_baseline": {
            "roc_auc_mean": 0.75,
            "roc_auc_std": 0.01,
        },
        "layers": {
            "7": {
                "mahalanobis_only": {"roc_auc_mean": 0.63, "roc_auc_std": 0.01},
                "combined": {"roc_auc_mean": 0.79, "roc_auc_std": 0.01},
                "delta_vs_entropy": 0.04,
                "length_controlled_delta": 0.02,
            }
        },
    }
    in_all.write_text(json.dumps(payload_all))

    result = aggregate_pca_ablation(
        [in_32, in_all],
        expected_dims=["32", "max"],
        model="qwen",
    )

    layer7 = result["layers"]["7"]
    assert result["settings"]["pca_dims"] == [32, "max"]
    assert layer7["best_dim_by_combined_auc"] == "max"
    assert layer7["dims"]["max"]["combined"]["roc_auc_mean"] == 0.79
