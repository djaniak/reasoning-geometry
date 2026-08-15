import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from applications.incremental_abstention import (
    FEATURE_NAMES,
    _load_prompt_states,
    aggregate_prompt_features,
    _load_exact_prompt_scores,
    _model_specs,
    paired_bootstrap_delta,
    prompt_accounting,
    prompt_metrics,
    crossfit_logistic_predictions,
    select_layer_rows,
)


def _row(prompt_id, trace_id, answer, *, fold, length=100, entropy=-1.0, logprob=-0.2, rmd=0.0):
    return {
        "prompt_id": prompt_id,
        "trace_id": trace_id,
        "fold": fold,
        "predicted_answer": answer,
        "gold_answer": "a",
        "trace_length": length,
        "length_score": -np.log1p(length),
        "entropy_score": entropy,
        "logprob_score": logprob,
        "rmd_tail_q20_score": rmd,
    }


def test_accounting_marks_all_unparsed_as_automatic_failure_and_counts_caps():
    rows = [
        _row(0, 0, "", fold=0, length=8),
        _row(0, 1, "", fold=0, length=10),
        _row(1, 0, "a", fold=1, length=10),
        _row(1, 1, "b", fold=1, length=11),
    ]
    accounting = prompt_accounting(rows, max_new_tokens=10, expected_traces=2)
    assert accounting[0]["automatic_failure"] is True
    assert accounting[0]["outcome"] == 0.0
    assert accounting[0]["unparsed_count"] == 2
    assert accounting[1]["automatic_failure"] is False
    assert accounting[1]["cap_count"] == 2
    assert accounting[1]["valid_plurality"] is True


def test_feature_aggregation_uses_parseable_vote_and_keeps_failure_counts():
    rows = [
        _row(0, 0, "a", fold=0, length=100, entropy=-1, logprob=-0.1, rmd=0.4),
        _row(0, 1, "a", fold=0, length=110, entropy=-2, logprob=-0.2, rmd=0.2),
        _row(0, 2, "b", fold=0, length=120, entropy=-3, logprob=-0.3, rmd=0.0),
        _row(0, 3, "", fold=0, length=130, entropy=-4, logprob=-0.4, rmd=-0.2),
    ]
    features = aggregate_prompt_features(rows, max_new_tokens=130, expected_traces=4)[0]
    assert tuple(features[name] for name in FEATURE_NAMES[:4]) == pytest.approx(
        (-np.mean(np.log1p([100, 110, 120, 130])), -2.5, -0.25, 2 / 3)
    )
    assert features["rmd_tail_q20"] == pytest.approx(0.1)
    assert features["cap_count"] == 1
    assert features["unparsed_count"] == 1


def test_crossfit_predictions_respect_explicit_prompt_folds():
    features = np.asarray([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]])
    outcomes = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    folds = np.asarray([0, 1, 2, 0, 1, 2])
    probabilities = crossfit_logistic_predictions(features, outcomes, folds, seed=3)
    assert np.all(np.isfinite(probabilities))
    assert probabilities[0] < probabilities[3]
    assert probabilities[1] < probabilities[4]


def test_prompt_metrics_report_auacc_and_conventional_aurc():
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
    outcomes = np.asarray([0.0, 0.0, 1.0, 1.0])
    metrics = prompt_metrics(probabilities, outcomes)
    assert metrics["auacc"] > 0.5
    assert metrics["aurc"] < 0.5
    assert metrics["brier"] < 0.1
    assert metrics["log_loss"] < 0.25


def test_paired_bootstrap_delta_is_reproducible_and_positive_for_better_ranker():
    left = np.asarray([0.1, 0.2, 0.8, 0.9])
    right = np.asarray([0.4, 0.6, 0.5, 0.7])
    outcomes = np.asarray([0.0, 0.0, 1.0, 1.0])
    first = paired_bootstrap_delta(left, right, outcomes, n_bootstrap=300, seed=9)
    second = paired_bootstrap_delta(left, right, outcomes, n_bootstrap=300, seed=9)
    assert first == second
    assert first["point_estimate"] > 0


def test_prompt_state_loader_averages_exact_pilot_trace_states(tmp_path):
    path = tmp_path / "pilot.npz"
    np.savez(
        path,
        prompt_ids=np.asarray([4]),
        prompt_hidden_L21=np.asarray([[[1.0, 3.0], [3.0, 5.0]]], dtype=np.float16),
    )
    states = _load_prompt_states(path, 21)
    assert np.allclose(states[4], [2.0, 4.0])


def _write_trace_batch(path, prompt_id, n_traces, n_tokens, layer, hidden=4):
    """One batch file shaped like `collect_data.py` writes them."""
    arrays = {}
    metadata = []
    for trace_id in range(n_traces):
        block = np.zeros((n_tokens, hidden), dtype=np.float32)
        # Row zero is the last prompt token, so it is identical across siblings;
        # the rest of the trace diverges. Only row zero should survive the load.
        block[0] = np.arange(hidden, dtype=np.float32)
        block[1:] = float(trace_id + 1)
        arrays[f"hidden_L{layer}_{trace_id}"] = block
        metadata.append({"trace_id": trace_id, "idx": prompt_id, "sample_id": trace_id})
    np.savez(path, metadata=np.asarray(metadata, dtype=object), **arrays)


def test_prompt_state_loader_reads_row_zero_from_a_trace_directory(tmp_path):
    _write_trace_batch(tmp_path / "batch_0000.npz", prompt_id=7, n_traces=8,
                       n_tokens=32, layer=21)

    states = _load_prompt_states(tmp_path, 21)

    # Row zero is identical across the eight siblings, so the mean is a no-op.
    assert np.allclose(states[7], np.arange(4, dtype=np.float32))


def test_prompt_state_loader_does_not_retain_whole_traces(tmp_path):
    """The cached blocks are already float32, so `np.asarray` on row zero hands
    back a *view* and the returned dict pins every trace the loader ever read --
    104 GiB on DeepSeek against the ~50 MiB the rows themselves need. Owning the
    data is what the docstring's "without retaining tokens" actually requires."""
    _write_trace_batch(tmp_path / "batch_0000.npz", prompt_id=3, n_traces=2,
                       n_tokens=4096, layer=21)

    states = _load_prompt_states(tmp_path, 21)

    row = states[3]
    assert row.base is None, "row zero is a view onto the full trace block"
    assert row.nbytes < 4096 * 4 * np.dtype(np.float32).itemsize


def test_exact_scores_are_aggregated_and_exposed_as_incremental_models(tmp_path):
    path = tmp_path / "exact.npz"
    rows = np.asarray(
        [
            {"prompt_id": 0, "deepconf_global": 10.0, "deepconf_tail_q20": 11.0},
            {"prompt_id": 0, "deepconf_global": 12.0, "deepconf_tail_q20": 13.0},
        ],
        dtype=object,
    )
    np.savez(path, trace_summaries=rows)
    scores = _load_exact_prompt_scores(path)
    assert scores[0]["deepconf_global"] == pytest.approx(11.0)
    specs = _model_specs({0: {"prompt_only_geometry": np.nan, "deepconf_global": 11.0}})
    assert "B0_plus_DeepConf_tail_q20" in specs


def test_layer_selection_defaults_to_the_deepest_layer_present():
    rows = [
        {"prompt_id": 0, "layer": 7, "rmd_tail_q20_score": 0.0},
        {"prompt_id": 0, "layer": 21, "rmd_tail_q20_score": 9.0},
    ]

    kept, layer = select_layer_rows(rows)

    assert layer == 21
    assert [row["rmd_tail_q20_score"] for row in kept] == [9.0]


def test_layer_selection_honours_an_explicit_layer_and_rejects_an_absent_one():
    rows = [
        {"prompt_id": 0, "layer": 7, "rmd_tail_q20_score": 0.0},
        {"prompt_id": 0, "layer": 21, "rmd_tail_q20_score": 9.0},
    ]

    assert select_layer_rows(rows, 7)[1] == 7
    with pytest.raises(ValueError, match="no rows at layer 14"):
        select_layer_rows(rows, 14)
    with pytest.raises(ValueError, match="no OOF rows"):
        select_layer_rows([])


def test_leaving_the_layer_sweep_in_would_change_geometry_but_not_the_baseline():
    """Why the helper exists: the defect it prevents is silent in every B0 column.

    Output-side scores repeat unchanged at every layer, so an analysis that forgets
    to select one still reproduces the frozen baseline exactly -- and quietly scores
    a cross-layer average of `rmd_tail_q20` that no frozen result was computed from.
    """
    rows = [
        _row(0, 0, "a", fold=0, rmd=0.0) | {"layer": 7},
        _row(0, 0, "a", fold=0, rmd=4.0) | {"layer": 21},
    ]

    swept = aggregate_prompt_features(rows, max_new_tokens=100)[0]
    selected = aggregate_prompt_features(select_layer_rows(rows)[0], max_new_tokens=100)[0]

    assert swept["logprob"] == pytest.approx(selected["logprob"])
    assert swept["vote_agreement"] == pytest.approx(selected["vote_agreement"])
    assert swept["rmd_tail_q20"] == pytest.approx(2.0)
    assert selected["rmd_tail_q20"] == pytest.approx(4.0)
