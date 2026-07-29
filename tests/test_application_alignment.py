import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application_alignment import build_application_alignment


def _decomposition():
    return {
        "model": "qwen",
        "dataset": "math500",
        "layers": {
            "7": {
                "methods": {
                    "raw": {
                        "metrics": {
                            "within_prompt_pair_weighted": 0.45,
                            "prompt_centered_auc": 0.47,
                            "score_icc": 0.90,
                            "prompt_score_pass_rate_spearman": -0.20,
                        }
                    },
                    "rmd": {
                        "metrics": {
                            "within_prompt_pair_weighted": 0.60,
                            "prompt_centered_auc": 0.59,
                            "score_icc": 0.96,
                            "prompt_score_pass_rate_spearman": 0.55,
                        }
                    },
                }
            }
        },
    }


def _selection():
    def selector(value):
        return {"pass_at_1": value}

    return {
        "model": "qwen",
        "dataset": "math500",
        "layers": {
            "7": {
                "selectors": {
                    "random": selector(0.55),
                    "majority_vote": selector(0.62),
                    "top1_raw": selector(0.57),
                    "top1_rmd": selector(0.61),
                }
            }
        },
    }


def _selective():
    return {
        "dataset": "math500",
        "scorers": {
            "entropy_mean": {"ausc": 0.62},
            "raw_mahal_L7": {"ausc": 0.52},
            "raw_rmd_L7": {"ausc": 0.72},
        },
    }


def test_application_alignment_joins_metrics_and_computes_gains():
    result = build_application_alignment(
        {"qwen": _decomposition()},
        {"qwen": _selection()},
        {"qwen": _selective()},
    )

    rows = {row["method"]: row for row in result["conditions"]}
    assert rows["raw"]["top1_gain_over_random"] == pytest.approx(0.02)
    assert rows["rmd"]["top1_gap_to_majority"] == pytest.approx(-0.01)
    assert rows["rmd"]["selective_ausc_gain_over_entropy"] == pytest.approx(0.10)
    assert result["warning"].startswith("Exploratory")


def test_application_alignment_rejects_missing_required_selector():
    selection = _selection()
    del selection["layers"]["7"]["selectors"]["top1_rmd"]

    with pytest.raises(ValueError, match="qwen L7 rmd"):
        build_application_alignment(
            {"qwen": _decomposition()},
            {"qwen": selection},
            {"qwen": _selective()},
        )
