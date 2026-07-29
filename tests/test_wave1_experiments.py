import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wave1_experiments import (
    aggregate_prompt_scores,
    answer_cluster_eligibility,
    crossfit_incremental_probes,
    entropy_trajectory_features,
    event_locked_profile,
    lve_features,
    position_matched_events,
    prompt_abstention_bootstrap,
    separated_entropy_events,
)
from cross_model_confirmation import summarize_model


def _row(prompt_id, trace_id, correct, answer, score, length=4):
    entropy = np.asarray([0.1, 0.9, 0.2, 0.8][:length], dtype=float)
    hidden = np.stack(
        [np.full(2, float(trace_id + token)) for token in range(length)], axis=0
    )
    return {
        "prompt_id": prompt_id,
        "trace_id": trace_id,
        "sample_id": trace_id,
        "is_correct": int(correct),
        "predicted_answer": answer,
        "gold_answer": "A",
        "trace_length": length,
        "entropy_score": -float(entropy.mean()),
        "logprob_score": float(score),
        "length_score": -float(np.log1p(length)),
        "rmd_tail_q20_score": float(score),
        "rmd_high_entropy_q20_score": float(score),
        "entropies": entropy,
        "hiddens": {21: hidden},
        "token_logprobs": -entropy,
    }


def test_entropy_trajectory_features_are_fixed_and_finite():
    values = entropy_trajectory_features(np.asarray([0.0, 1.0, 0.0, 1.0]))

    assert set(values) == {
        "upper_tail_mass",
        "peak_rate",
        "mean_peak_position",
        "post_peak_decay",
    }
    assert all(np.isfinite(value) for value in values.values())
    assert values["peak_rate"] == 0.5


def test_event_selection_is_separated_and_position_matched_is_reproducible():
    entropy = np.zeros(40, dtype=float)
    entropy[[3, 5, 22, 35]] = [4.0, 3.0, 5.0, 2.0]

    events = separated_entropy_events(entropy, min_separation=8, quantile=0.8)
    assert np.array_equal(events, np.asarray([3, 22, 35]))

    matched = position_matched_events(
        entropy, events, trace_id=7, seed=42, min_separation=8
    )
    repeated = position_matched_events(
        entropy, events, trace_id=7, seed=42, min_separation=8
    )
    assert len(matched) == len(events)
    assert np.array_equal(matched, repeated)
    assert np.all(np.diff(matched) >= 8)


def test_event_locked_profile_has_rmd_entropy_and_random_controls():
    entropy = np.zeros(40, dtype=float)
    entropy[[10, 30]] = [4.0, 5.0]
    rmd = np.linspace(-1.0, 1.0, 40)

    result = event_locked_profile(
        rmd,
        entropy,
        trace_id=3,
        seed=42,
        window=4,
        min_separation=8,
    )

    assert result["rmd"].shape == (9,)
    assert result["entropy"].shape == (9,)
    assert result["random_rmd"].shape == (9,)
    assert result["n_events"] == 2


def test_lve_features_include_order_sensitive_and_shuffle_control():
    hidden = np.asarray([[0.0], [1.0], [3.0], [6.0], [10.0]])
    entropy = np.asarray([0.1, 0.9, 0.2, 0.8, 0.1])

    values = lve_features(hidden, entropy, seed=42)

    assert {"lve_mean", "lve_slope", "lve_he", "lve_mean_shuffle", "lve_slope_shuffle"} <= set(values)
    assert all(np.isfinite(value) for value in values.values())


def test_incremental_lve_probe_is_prompt_cross_fitted():
    rows = []
    trace_id = 0
    for prompt_id in range(6):
        for label in (1, 0):
            rows.append({
                "prompt_id": prompt_id,
                "trace_id": trace_id,
                "fold": prompt_id % 3,
                "is_correct": label,
                "length_score": -1.0,
                "logprob_score": 0.0,
                "lve_mean_score": float(label),
                "lve_slope_score": float(label),
                "lve_he_score": float(label),
            })
            trace_id += 1
    result = crossfit_incremental_probes(
        rows,
        {
            "probe_outputs_lve": ("length", "logprob"),
            "probe_lve_mean": ("length", "logprob", "lve_mean"),
        },
        n_bootstrap=10,
    )
    assert result["diagnostics"]
    assert "probe_lve_mean_minus_probe_outputs_lve" in result["paired_deltas"]


def test_prompt_abstention_bootstrap_reports_deltas_and_coverage():
    rows = [
        _row(0, 0, True, "A", 0.9),
        _row(0, 1, False, "B", 0.1),
        _row(1, 2, False, "B", 0.8),
        _row(1, 3, False, "B", 0.2),
        _row(2, 4, True, "A", 0.7),
        _row(2, 5, False, "B", 0.3),
        _row(3, 6, True, "A", 0.6),
        _row(3, 7, True, "A", 0.4),
    ]
    scores = aggregate_prompt_scores(rows, methods=("rmd_tail_q20", "length"))
    outcomes = {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0}

    result = prompt_abstention_bootstrap(
        scores, outcomes, coverages=(0.5, 0.8), n_bootstrap=20, seed=42
    )

    assert "rmd_tail_q20" in result["point"]
    assert len(result["point"]["rmd_tail_q20"]["curve"]["coverages"]) == 4
    assert "rmd_tail_q20_minus_length" in result["deltas"]
    assert result["deltas"]["rmd_tail_q20_minus_length"]["0.5"]["n_valid"] == 20


def test_answer_cluster_eligibility_counts_invalid_and_censored_clusters():
    rows = [
        _row(0, 0, True, "A", 0.9),
        _row(0, 1, True, "A", 0.8),
        _row(0, 2, False, "B", 0.1),
        _row(0, 3, False, None, 0.0),
        _row(1, 4, False, "B", 0.1),
        _row(1, 5, False, "B", 0.2),
    ]
    result = answer_cluster_eligibility(rows, max_new_tokens=4)

    assert result["prompts_with_correct_cluster_ge2"] == 1
    assert result["prompts_with_wrong_cluster_ge2"] == 1
    assert result["prompts_with_both_ge2"] == 0
    assert "eligible_clusters_after_censoring" in result


def test_cross_model_summary_fixes_deepest_layer_and_only_two_contrasts(tmp_path):
    path = tmp_path / "decomposition.json"
    path.write_text(json.dumps({
        "settings": {"layers": [7, 21]},
        "truncation": {"capped_rate": 0.1},
        "layers": {"21": {"parseable_only": {
            "n_parseable_traces": 4,
            "methods": {"rmd": {"n_mixed_prompts": 2}},
            "paired_score_deltas": {
                "rmd_high_entropy_q20_minus_rmd": {
                    "prompt_centered_auc": {"point_estimate": 0.1}
                }
            },
        }}},
    }))
    result = summarize_model(path, model_label="qwen", min_mixed_prompts=3)

    assert result["deepest_layer"] == 21
    assert result["underpowered"] is True
    assert set(result["contrasts"]) == {
        "rmd_high_entropy_q20_minus_rmd",
        "rmd_high_entropy_q20_minus_rmd_random_q20",
    }
