from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.rmd_window_sensitivity import (
    DETECTORS,
    FROZEN_DETECTOR,
    ROBUSTNESS_SENTENCE,
    add_detector_features,
    robustness_verdict,
)


def _rows():
    rows = []
    for prompt_id in range(2):
        for trace_id in range(2):
            base = prompt_id * 10 + trace_id
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "trace_id": base,
                    "sample_id": trace_id,
                    "fold": prompt_id,
                    "is_correct": trace_id == 0,
                    "predicted_answer": "1" if trace_id == 0 else "2",
                    "gold_answer": "1",
                    "trace_length": 10,
                    "length_score": -1.0,
                    "entropy_score": -0.2,
                    "logprob_score": -0.3,
                    "rmd_score": float(base),
                    "rmd_tail_q10_score": float(base + 1),
                    "rmd_tail_q20_score": float(base + 2),
                    "rmd_tail_q50_score": float(base + 3),
                    "rmd_high_entropy_q20_score": float(base + 4),
                    "rmd_random_q20_score": float(base + 5),
                }
            )
    return rows


def test_detector_features_are_sibling_means_and_keep_q20_frozen():
    features = add_detector_features(
        _rows(), expected_traces=2, max_new_tokens=10
    )

    assert FROZEN_DETECTOR == "rmd_tail_q20"
    assert DETECTORS == (
        "rmd_tail_q10",
        "rmd_tail_q20",
        "rmd_tail_q50",
        "rmd_full",
        "rmd_high_entropy_q20",
        "rmd_random_q20",
    )
    assert features[0]["rmd_full"] == pytest.approx(0.5)
    assert features[0]["rmd_tail_q10"] == pytest.approx(1.5)
    assert features[0]["rmd_tail_q50"] == pytest.approx(3.5)


def _delta(point, low, high):
    return {"point_estimate": point, "ci_low": low, "ci_high": high}


def test_robustness_sentence_is_conditional_on_both_adjacent_windows():
    models = [
        {
            "label": label,
            "paired_deltas_aurc": {
                "rmd_tail_q10_over_B0": _delta(-0.03, -0.05, -0.01),
                "rmd_tail_q20_over_B0": _delta(-0.04, -0.06, -0.02),
                "rmd_tail_q50_over_B0": _delta(-0.02, -0.04, -0.001),
            },
        }
        for label in ("qwen", "deepseek", "deepseek_llama")
    ]

    passed = robustness_verdict(models)
    assert passed["passed"] is True
    assert passed["paper_sentence"] == ROBUSTNESS_SENTENCE

    models[1]["paired_deltas_aurc"]["rmd_tail_q50_over_B0"] = _delta(
        -0.01, -0.03, 0.01
    )
    failed = robustness_verdict(models)
    assert failed["passed"] is False
    assert ROBUSTNESS_SENTENCE not in failed["paper_sentence"]
