import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompt_decomposition import (
    HIDDEN_PROBE_METHODS,
    HIDDEN_PROBE_REGIONS,
    SUPERVISED_METHODS,
    available_score_methods,
    fit_hidden_state_probe,
    length_collapse_diagnostics,
    score_hidden_state_probe,
)


def _trace(prompt_id, trace_id, is_correct, predicted_answer="42", n_tokens=6):
    rng = np.random.default_rng(trace_id)
    return {
        "idx": prompt_id,
        "trace_id": trace_id,
        "sample_id": trace_id % 2,
        "is_correct": is_correct,
        "predicted_answer": predicted_answer,
        "gold_answer": "42",
        "entropies": rng.uniform(0.1, 0.9, size=n_tokens),
        "mean_logprob": -0.5,
    }


def _separable_groups(n_prompts=6):
    """Correct traces centre on +1 along axis 0, incorrect on -1.

    Noise is deliberate: LDA needs a non-singular within-class covariance, and a
    fixture with identical members per class is degenerate in a way real
    activations never are.
    """
    groups = {}
    projections = {}
    trace_id = 0
    for prompt_id in range(n_prompts):
        traces = []
        for is_correct in (True, False):
            trace = _trace(prompt_id, trace_id, is_correct)
            offset = 1.0 if is_correct else -1.0
            rng = np.random.default_rng(1000 + trace_id)
            n_tokens = len(trace["entropies"])
            projections[trace_id] = np.asarray(
                [offset, 0.0]
            ) + rng.normal(scale=0.15, size=(n_tokens, 2))
            traces.append(trace)
            trace_id += 1
        groups[prompt_id] = traces
    return groups, projections


def test_regions_and_methods_line_up():
    assert HIDDEN_PROBE_METHODS == tuple(
        f"probe_hidden_{region}" for region in HIDDEN_PROBE_REGIONS
    )


def test_hidden_probe_methods_are_treated_as_supervised():
    rows = [
        {f"{method}_score": 0.5 for method in HIDDEN_PROBE_METHODS} | {"rmd_score": 1.0}
    ]
    assert set(available_score_methods(rows)).isdisjoint(HIDDEN_PROBE_METHODS)
    assert set(HIDDEN_PROBE_METHODS).issubset(SUPERVISED_METHODS)
    supervised = available_score_methods(rows, include_supervised=True)
    assert set(HIDDEN_PROBE_METHODS).issubset(supervised)


@pytest.mark.parametrize("region", HIDDEN_PROBE_REGIONS)
def test_probe_separates_a_separable_set(region):
    groups, projections = _separable_groups()
    fit = fit_hidden_state_probe(
        groups, list(groups), projections, region=region
    )
    assert fit["classifier"] is not None
    assert fit["n_train"] == 12
    assert fit["n_correct"] == 6

    scores = {}
    for prompt_id, traces in groups.items():
        for trace in traces:
            trace_id = int(trace["trace_id"])
            scores[bool(trace["is_correct"])] = scores.get(
                bool(trace["is_correct"]), []
            ) + [
                score_hidden_state_probe(
                    projections[trace_id],
                    trace["entropies"],
                    fit,
                    region,
                    trace_id=trace_id,
                )
            ]
    # Higher score must mean more likely correct, matching every other scorer.
    assert min(scores[True]) > max(scores[False])


def test_unparsed_traces_are_excluded_from_training():
    """Unparsed traces are auto-labeled incorrect upstream; training on them
    would let the probe win by detecting truncation instead of correctness."""
    groups, projections = _separable_groups(n_prompts=4)
    for trace in groups[0]:
        trace["predicted_answer"] = ""
    fit = fit_hidden_state_probe(groups, list(groups), projections, region="full")
    assert fit["n_train"] == 6
    assert fit["n_correct"] == 3


def test_single_class_training_set_is_reported_not_raised():
    groups, projections = _separable_groups(n_prompts=2)
    for traces in groups.values():
        for trace in traces:
            trace["is_correct"] = True
    fit = fit_hidden_state_probe(groups, list(groups), projections, region="full")
    assert fit["classifier"] is None
    assert fit["skipped"] == "single_class"
    with pytest.raises(ValueError, match="no usable hidden-state probe"):
        score_hidden_state_probe(
            projections[0], groups[0][0]["entropies"], fit, "full", trace_id=0
        )


def test_degenerate_training_set_is_reported_not_raised():
    """One trace per class is below LDA's floor; report it rather than let
    sklearn raise an opaque error mid-fold."""
    groups, projections = _separable_groups(n_prompts=1)
    fit = fit_hidden_state_probe(groups, list(groups), projections, region="full")
    assert fit["classifier"] is None
    assert fit["skipped"] == "insufficient_samples"


def test_length_collapse_detects_a_pure_length_scorer():
    rows = [
        {
            "predicted_answer": "42",
            "length_score": -float(np.log1p(length)),
            "probe_hidden_full_score": -float(np.log1p(length)),
            "rmd_score": float(length % 3),
        }
        for length in range(10, 40)
    ]
    summary = length_collapse_diagnostics(
        rows, ("probe_hidden_full", "rmd")
    )
    assert summary["probe_hidden_full"]["spearman"] == pytest.approx(1.0)
    assert summary["probe_hidden_full"]["n"] == 30
    assert abs(summary["rmd"]["spearman"]) < 0.5


def test_length_collapse_ignores_unparsed_rows():
    rows = [
        {"predicted_answer": "42", "length_score": -1.0, "probe_hidden_full_score": 1.0},
        {"predicted_answer": "", "length_score": -9.0, "probe_hidden_full_score": 9.0},
        {"predicted_answer": "7", "length_score": -2.0, "probe_hidden_full_score": 2.0},
        {"predicted_answer": "8", "length_score": -3.0, "probe_hidden_full_score": 3.0},
    ]
    summary = length_collapse_diagnostics(rows, ("probe_hidden_full",))
    assert summary["probe_hidden_full"]["n"] == 3
