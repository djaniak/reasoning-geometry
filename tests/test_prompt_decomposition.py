import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.analyze import load_all_traces
from applications.prompt_decomposition import (
    _status,
    analyze_oof_scores,
    bootstrap_metrics,
    compute_scalar_metrics,
    compute_weighted_bootstrap_metrics,
    generate_oof_scores,
    generate_oof_scores_layerwise,
    is_unparsed,
    make_prompt_folds,
    parse_args,
    parseable_within_prompt_metrics,
    prompt_local_rmd_scores,
    prepare_bootstrap_arrays,
    prompt_centered_auc,
    prompt_score_pass_rate_correlation,
    resample_prompt_rows,
    score_icc,
    truncation_report,
    validate_groups,
    within_prompt_concordance,
    write_json,
    write_markdown,
    write_trace_csv,
)


def _write_trace_batch(path: Path, trace_id: int, value: float) -> None:
    metadata = np.array(
        [
            {
                "trace_id": trace_id,
                "idx": trace_id,
                "sample_id": 0,
                "is_correct": True,
            }
        ],
        dtype=object,
    )
    np.savez_compressed(
        path,
        metadata=metadata,
        **{
            f"entropies_{trace_id}": np.array([value], dtype=np.float32),
            f"token_logprobs_{trace_id}": np.array([-value], dtype=np.float32),
            f"tokens_{trace_id}": np.array(["token"], dtype=object),
            f"hidden_L7_{trace_id}": np.array([[value]], dtype=np.float32),
        },
    )


def test_parallel_trace_loading_preserves_order_and_can_skip_auxiliary_arrays(
    tmp_path: Path,
):
    _write_trace_batch(tmp_path / "batch_0001.npz", trace_id=1, value=2.0)
    _write_trace_batch(tmp_path / "batch_0000.npz", trace_id=0, value=1.0)

    traces = load_all_traces(
        str(tmp_path),
        [7],
        max_workers=2,
        include_auxiliary=False,
    )

    assert [trace["trace_id"] for trace in traces] == [0, 1]
    assert [trace["hiddens"][7].item() for trace in traces] == [1.0, 2.0]
    assert all(trace["entropies"] is None for trace in traces)


def test_trace_loading_can_select_only_required_auxiliary_arrays(tmp_path: Path):
    _write_trace_batch(tmp_path / "batch_0000.npz", trace_id=0, value=1.0)

    traces = load_all_traces(
        str(tmp_path),
        [7],
        auxiliary_fields={"entropies"},
    )

    assert traces[0]["entropies"].tolist() == [1.0]
    assert traces[0]["token_logprobs"] is None
    assert traces[0]["tokens"] is None


def test_status_is_immediate_and_visible(monkeypatch):
    writes = []
    flushes = []

    class Stream:
        def write(self, text):
            writes.append(text)

        def flush(self):
            flushes.append(True)

    monkeypatch.setattr(sys, "stderr", Stream())

    _status("Loading traces")

    assert "".join(writes) == "[prompt-decomposition] Loading traces\n"
    assert flushes


def _rows(prompt_scores):
    rows = []
    trace_id = 0
    for prompt_id, values in prompt_scores.items():
        for is_correct, score in values:
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "trace_id": trace_id,
                    "sample_id": len(rows),
                    "is_correct": int(is_correct),
                    "fold": 0,
                    "layer": 7,
                    "score": float(score),
                    "raw_score": float(score),
                    "rmd_score": float(score) + 0.1,
                }
            )
            trace_id += 1
    return rows


def test_within_prompt_concordance_handles_perfect_reversed_and_tied_scores():
    perfect = _rows({0: [(1, 2.0), (0, 1.0)], 1: [(1, 4.0), (0, 3.0)]})
    reversed_rows = _rows({0: [(1, 1.0), (0, 2.0)], 1: [(1, 3.0), (0, 4.0)]})
    tied = _rows({0: [(1, 1.0), (0, 1.0)], 1: [(1, 2.0), (0, 2.0)]})

    assert within_prompt_concordance(perfect)["macro"] == 1.0
    assert within_prompt_concordance(perfect)["pair_weighted"] == 1.0
    assert within_prompt_concordance(reversed_rows)["macro"] == 0.0
    assert within_prompt_concordance(tied)["macro"] == 0.5


def test_prompt_centered_auc_excludes_homogeneous_prompts():
    rows = _rows(
        {
            0: [(1, 3.0), (0, 1.0)],
            1: [(1, 4.0), (0, 2.0)],
            2: [(1, 10.0), (1, 11.0)],
            3: [(0, -10.0), (0, -11.0)],
        }
    )

    result = prompt_centered_auc(rows)

    assert result["auc"] == 1.0
    assert result["n_mixed_prompts"] == 2
    assert result["n_traces"] == 4


def test_score_icc_is_high_for_prompt_offsets_and_signed_when_between_variance_is_zero():
    high = _rows(
        {
            0: [(0, 0.0), (1, 0.0)],
            1: [(0, 10.0), (1, 10.0)],
            2: [(0, 20.0), (1, 20.0)],
        }
    )
    non_positive = _rows(
        {
            0: [(0, -1.0), (1, 1.0)],
            1: [(0, -1.0), (1, 1.0)],
            2: [(0, -1.0), (1, 1.0)],
        }
    )

    assert score_icc(high)["icc"] == pytest.approx(1.0)
    assert score_icc(non_positive)["icc"] < 0.0


def test_prompt_score_pass_rate_correlation_uses_prompt_means():
    rows = _rows(
        {
            0: [(0, 0.0), (0, 0.0)],
            1: [(1, 1.0), (0, 1.0)],
            2: [(1, 2.0), (1, 2.0)],
        }
    )

    result = prompt_score_pass_rate_correlation(rows)

    assert result["spearman"] == pytest.approx(1.0)
    assert result["pearson"] == pytest.approx(1.0)
    assert result["n_prompts"] == 3


def test_prompt_score_pass_rate_correlation_handles_constant_inputs_without_warning():
    rows = _rows(
        {
            0: [(1, 1.0), (0, 1.0)],
            1: [(1, 1.0), (0, 1.0)],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = prompt_score_pass_rate_correlation(rows)

    assert result["spearman"] is None
    assert result["pearson"] is None


def test_prompt_local_rmd_scores_use_leave_one_trace_out_prompt_background():
    scores = prompt_local_rmd_scores(
        {
            1: np.array([[0.0]], dtype=float),
            2: np.array([[10.0]], dtype=float),
            3: np.array([[10.0]], dtype=float),
        },
        {
            1: np.array([0.0], dtype=float),
            2: np.array([0.0], dtype=float),
            3: np.array([0.0], dtype=float),
        },
        variance_floor=1.0,
    )

    assert scores[1] == pytest.approx(10.0)
    assert scores[2] == pytest.approx(1.0)
    assert scores[3] == pytest.approx(1.0)


def _groups(n_prompts=4, n=8):
    groups = {}
    trace_id = 0
    for prompt_id in range(n_prompts):
        group = []
        for sample_id in range(n):
            group.append(
                {
                    "idx": prompt_id,
                    "trace_id": trace_id,
                    "sample_id": sample_id,
                    "is_correct": sample_id % 2 == 0,
                    "hiddens": {
                        7: np.array(
                            [[prompt_id, sample_id], [prompt_id, sample_id]],
                            dtype=float,
                        )
                    },
                }
            )
            trace_id += 1
        groups[prompt_id] = group
    return groups


def test_validate_groups_is_strict_by_default_and_partial_mode_excludes_incomplete_groups():
    groups = _groups()
    groups[3] = groups[3][:-1]

    with pytest.raises(ValueError, match="expected 4 complete prompts"):
        validate_groups(groups, expected_prompts=4, n=8, allow_partial=False)

    validated, report = validate_groups(
        groups, expected_prompts=4, n=8, allow_partial=True
    )

    assert sorted(validated) == [0, 1, 2]
    assert report["partial_data"] is True
    assert report["excluded_prompt_ids"] == [3]


def test_validate_groups_rejects_duplicate_sample_and_trace_ids():
    groups = _groups(n_prompts=2, n=2)
    groups[1][1]["sample_id"] = 0
    groups[1][1]["trace_id"] = groups[0][0]["trace_id"]

    with pytest.raises(ValueError, match="invalid sample IDs"):
        validate_groups(groups, expected_prompts=2, n=2, allow_partial=False)


def test_make_prompt_folds_has_no_overlap_and_scores_every_prompt_once():
    prompt_ids = list(range(10))
    folds = make_prompt_folds(prompt_ids, n_splits=5, seed=42)

    seen = []
    for train_ids, test_ids in folds:
        assert set(train_ids).isdisjoint(test_ids)
        seen.extend(test_ids)

    assert sorted(seen) == prompt_ids


def test_generate_oof_scores_never_fits_on_held_out_prompt():
    groups = _groups(n_prompts=4, n=2)
    fit_prompt_sets = []

    class FakePca:
        def transform(self, values):
            return np.asarray(values, dtype=float)

    def fake_fit(correct_traces, layer, pca_dim):
        train_prompt_ids = frozenset(int(trace["idx"]) for trace in correct_traces)
        fit_prompt_sets.append(train_prompt_ids)
        return (FakePca(), np.zeros(2), train_prompt_ids)

    def fake_extend(ref, background_traces, layer):
        background_ids = frozenset(int(trace["idx"]) for trace in background_traces)
        assert background_ids == ref[2]
        return (*ref, "background")

    def fake_raw_distance(hiddens, pca, mu, train_prompt_ids):
        prompt_id = int(hiddens[0, 0])
        assert prompt_id not in train_prompt_ids
        return np.full(hiddens.shape[0], hiddens[0, 1] + 1.0)

    def fake_rmd_distance(hiddens, pca, mu, train_prompt_ids, background):
        prompt_id = int(hiddens[0, 0])
        assert prompt_id not in train_prompt_ids
        return np.full(hiddens.shape[0], hiddens[0, 1] - 0.5)

    for group in groups.values():
        for trace in group:
            trace["entropies"] = np.array([0.25, 0.75])
            trace["mean_logprob"] = -0.5
            trace["predicted_answer"] = str(trace["sample_id"])
            trace["gold_answer"] = "0"

    rows = generate_oof_scores(
        groups,
        layers=[7],
        pca_dim=2,
        n_splits=2,
        seed=42,
        fit_reference=fake_fit,
        extend_reference=fake_extend,
        raw_distance=fake_raw_distance,
        relative_distance=fake_rmd_distance,
    )

    assert len(rows) == 8
    assert len({row["trace_id"] for row in rows}) == 8
    assert {row["fold"] for row in rows} == {0, 1}
    assert all(row["raw_score"] <= -1.0 for row in rows)
    assert all(row["entropy_score"] == pytest.approx(-0.5) for row in rows)
    assert all(row["logprob_score"] == pytest.approx(-0.5) for row in rows)
    assert all(row["length_score"] == pytest.approx(-np.log1p(2)) for row in rows)
    assert all(row["activation_norm_score"] <= 0.0 for row in rows)
    assert all(row["centroid_score"] <= 0.0 for row in rows)
    assert all(np.isfinite(row["prompt_local_rmd_score"]) for row in rows)
    # Constant per-token distances: localized RMD must equal full-trace RMD.
    assert all(
        row["rmd_high_entropy_q20_score"] == pytest.approx(row["rmd_score"])
        for row in rows
    )
    assert all(
        row["rmd_tail_q20_score"] == pytest.approx(row["rmd_score"])
        for row in rows
    )
    assert rows[0]["predicted_answer"] in {"0", "1"}
    assert rows[0]["gold_answer"] == "0"
    assert len(fit_prompt_sets) == 2


def test_localized_rmd_scores_average_only_region_tokens():
    groups = _groups(n_prompts=4, n=2)
    entropies = np.array([0.1, 0.9, 0.2, 0.3, 0.8])
    for group in groups.values():
        for trace in group:
            trace["hiddens"] = {7: np.tile(trace["hiddens"][7][:1], (5, 1))}
            trace["entropies"] = entropies
            trace["mean_logprob"] = -0.5
            trace["predicted_answer"] = str(trace["sample_id"])
            trace["gold_answer"] = "0"

    class FakePca:
        def transform(self, values):
            return np.asarray(values, dtype=float)

    per_token_rmd = np.arange(5, dtype=float)

    rows = generate_oof_scores(
        groups,
        layers=[7],
        pca_dim=2,
        n_splits=2,
        seed=42,
        fit_reference=lambda correct, layer, dim: (FakePca(), np.zeros(2)),
        extend_reference=lambda ref, traces, layer: (*ref, "background"),
        raw_distance=lambda hiddens, *ref: np.ones(hiddens.shape[0]),
        relative_distance=lambda hiddens, *ref: per_token_rmd.copy(),
    )

    # 20% of 5 tokens = 1 token: highest entropy is index 1, tail is index 4.
    assert all(row["rmd_score"] == pytest.approx(-2.0) for row in rows)
    assert all(
        row["rmd_high_entropy_q20_score"] == pytest.approx(-1.0) for row in rows
    )
    assert all(row["rmd_tail_q20_score"] == pytest.approx(-4.0) for row in rows)


def test_generate_oof_scores_layerwise_loads_only_one_layer_at_a_time():
    load_calls = []
    score_calls = []

    def fake_load(data_dir, layers, **kwargs):
        load_calls.append(list(layers))
        layer = layers[0]
        traces = []
        trace_id = 0
        for prompt_id in range(2):
            for sample_id in range(2):
                traces.append(
                    {
                        "idx": prompt_id,
                        "trace_id": trace_id,
                        "sample_id": sample_id,
                        "is_correct": sample_id == 0,
                        "entropies": np.array([0.5], dtype=np.float32),
                        "hiddens": {
                            layer: np.array(
                                [[prompt_id, sample_id]], dtype=np.float32
                            )
                        },
                    }
                )
                trace_id += 1
        return traces

    def fake_score(groups, layers, **kwargs):
        score_calls.append(list(layers))
        return [{"layer": layers[0], "n_prompts": len(groups)}]

    rows, report = generate_oof_scores_layerwise(
        data_dir="unused",
        layers=[7, 14, 21],
        expected_prompts=2,
        n=2,
        allow_partial=False,
        pca_dim=2,
        n_splits=2,
        seed=42,
        load_workers=1,
        show_progress=False,
        load_traces=fake_load,
        score_groups=fake_score,
    )

    assert load_calls == [[7], [14], [21]]
    assert score_calls == [[7], [14], [21]]
    assert [row["layer"] for row in rows] == [7, 14, 21]
    assert report["observed_complete_prompts"] == 2


def test_generic_bootstrap_reports_all_complete_methods_and_rmd_pairs():
    rows = _rows(
        {
            0: [(1, 3.0), (0, 1.0)],
            1: [(1, 4.0), (0, 2.0)],
            2: [(1, 2.0), (0, 0.0)],
            3: [(1, 5.0), (0, 1.0)],
        }
    )
    for row in rows:
        row["entropy_score"] = row["raw_score"] - 0.25
        row["logprob_score"] = None

    result = bootstrap_metrics(rows, n_bootstrap=10, seed=7)

    assert set(result["methods"]) == {"raw", "rmd", "entropy"}
    assert "logprob" not in result["methods"]
    assert "entropy" in result["paired_rmd_minus_baseline"]
    assert result["paired_rmd_minus_raw"] == result["paired_rmd_minus_baseline"]["raw"]


def test_resample_prompt_rows_preserves_duplicate_prompt_draws_as_distinct_clusters():
    rows = _rows({0: [(1, 2.0), (0, 1.0)], 1: [(1, 3.0), (0, 0.0)]})

    sampled = resample_prompt_rows(rows, [0, 0, 1])

    assert len(sampled) == 6
    assert {row["prompt_id"] for row in sampled} == {0, 1, 2}
    assert [row["source_prompt_id"] for row in sampled[::2]] == [0, 0, 1]


def test_bootstrap_metrics_is_reproducible_and_returns_paired_differences():
    rows = _rows(
        {
            0: [(1, 3.0), (0, 1.0)],
            1: [(1, 4.0), (0, 2.0)],
            2: [(1, 2.0), (0, 0.0)],
            3: [(1, 5.0), (0, 1.0)],
        }
    )

    first = bootstrap_metrics(rows, n_bootstrap=20, seed=7)
    second = bootstrap_metrics(rows, n_bootstrap=20, seed=7)

    assert first == second
    assert "pooled_auc" in first["methods"]["raw"]
    assert "pooled_auc" in first["paired_rmd_minus_raw"]


def test_weighted_bootstrap_metrics_match_explicit_prompt_duplication():
    rows = _rows(
        {
            0: [(1, 3.0), (0, 1.0)],
            1: [(1, 4.0), (0, 2.0)],
            2: [(1, 2.0), (0, 0.0)],
            3: [(1, 5.0), (0, 1.0)],
        }
    )
    sampled_prompt_ids = [0, 0, 2, 3]
    counts = np.bincount(sampled_prompt_ids, minlength=4)
    explicit_rows = resample_prompt_rows(rows, sampled_prompt_ids)
    prepared = prepare_bootstrap_arrays(rows)

    for method in ("raw", "rmd"):
        expected = compute_scalar_metrics(
            explicit_rows, score_key=f"{method}_score"
        )
        actual = compute_weighted_bootstrap_metrics(
            prepared, counts=counts, method=method
        )
        for metric in (
            "pooled_auc",
            "prompt_centered_auc",
            "within_prompt_macro",
            "within_prompt_pair_weighted",
            "score_icc",
            "prompt_score_pass_rate_spearman",
            "prompt_score_pass_rate_pearson",
        ):
            assert actual[metric] == pytest.approx(expected[metric])


def test_bootstrap_progress_can_be_disabled(capsys):
    rows = _rows(
        {
            0: [(1, 3.0), (0, 1.0)],
            1: [(1, 4.0), (0, 2.0)],
        }
    )

    bootstrap_metrics(rows, n_bootstrap=2, seed=7, show_progress=False)

    assert capsys.readouterr().err == ""


def test_parse_args_accepts_no_progress(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prompt_decomposition.py",
            "--data_dir",
            "data",
            "--output_dir",
            "results",
            "--no_progress",
            "--load_workers",
            "3",
        ],
    )

    assert parse_args().no_progress is True
    assert parse_args().load_workers == 3


def test_result_schema_and_writers(tmp_path: Path):
    rows = _rows(
        {
            0: [(1, 3.0), (0, 1.0)],
            1: [(1, 4.0), (0, 2.0)],
            2: [(1, 2.0), (0, 0.0)],
            3: [(1, 5.0), (0, 1.0)],
        }
    )
    config = {
        "dataset": "math500",
        "model": "qwen",
        "layers": [7],
        "pca_dim": 128,
        "n": 2,
        "expected_prompts": 4,
        "n_splits": 2,
        "seed": 42,
        "data_report": {
            "partial_data": False,
            "observed_complete_prompts": 4,
            "expected_prompts": 4,
            "n": 2,
            "excluded_prompt_ids": [],
        },
    }

    result = analyze_oof_scores(rows, config=config, n_bootstrap=10, seed=42, max_new_tokens=2048)

    assert result["settings"]["no_layer_selection"] is True
    assert result["settings"]["raw_score"].startswith("-mean(raw")
    assert result["settings"]["rmd_score"].startswith("-mean(relative")
    assert set(result["layers"]["7"]["methods"]) == {"raw", "rmd"}
    assert result["data"]["partial_data"] is False
    assert result["incremental_readouts"]["supervised"] is True
    assert result["incremental_readouts"]["cross_fitted_by_prompt_fold"] is True
    assert (
        result["layers"]["7"]["paired_rmd_minus_raw"]["pooled_auc"][
            "point_estimate"
        ]
        == pytest.approx(0.0)
    )

    csv_path = tmp_path / "scores.csv"
    json_path = tmp_path / "results.json"
    markdown_path = tmp_path / "report.md"
    write_trace_csv(rows, csv_path)
    write_json(result, json_path)
    write_markdown(result, markdown_path)

    with csv_path.open() as fh:
        written_rows = list(csv.DictReader(fh))
    assert len(written_rows) == len(rows)
    assert json.loads(json_path.read_text())["dataset"] == "math500"
    markdown = markdown_path.read_text()
    assert "Supervised cross-fitted incremental readouts" in markdown
    assert "No layer was selected" in markdown


def test_is_unparsed_flags_missing_or_blank_predicted_answer():
    assert is_unparsed({"predicted_answer": None})
    assert is_unparsed({"predicted_answer": ""})
    assert is_unparsed({"predicted_answer": "   "})
    assert is_unparsed({})
    assert not is_unparsed({"predicted_answer": "42"})


def test_truncation_report_quantifies_unparsed_and_capped_traces():
    rows = [
        {"is_correct": 1, "predicted_answer": "42", "trace_length": 100},
        {"is_correct": 0, "predicted_answer": "7", "trace_length": 120},
        {"is_correct": 0, "predicted_answer": "", "trace_length": 2048},
        {"is_correct": 0, "predicted_answer": None, "trace_length": 2048},
    ]

    report = truncation_report(rows, max_new_tokens=2048)

    assert report["n_unparsed"] == 2
    assert report["unparsed_rate"] == pytest.approx(0.5)
    assert report["n_capped"] == 2
    assert report["n_unparsed_and_incorrect"] == 2
    # two of the three incorrect traces are unparsed non-answers
    assert report["unparsed_share_of_incorrect"] == pytest.approx(2 / 3)


def test_truncation_report_requires_an_explicit_cap():
    """Inferring the cap marks only the longest trace as capped, understating truncation."""
    rows = [
        {"is_correct": 1, "predicted_answer": "1", "trace_length": 500},
        {"is_correct": 0, "predicted_answer": "", "trace_length": 1024},
    ]

    with pytest.raises(ValueError, match="max_new_tokens is required"):
        truncation_report(rows)


def _artifact_rows():
    """Mixedness driven by unparsed (truncated) incorrect traces, not wrong answers.

    Every prompt's only 'incorrect' trace is an unparsed non-answer that RMD scores
    as anomalous. Dropping unparsed traces leaves homogeneous (all-correct) prompts,
    so the within-prompt contrast must vanish.
    """
    rows = []
    trace_id = 0
    for prompt_id in range(4):
        specs = [
            (1, "ok", 5.0, 100),
            (1, "ok", 5.1, 110),
            (0, "", -5.0, 2048),  # unparsed, anomalous score, length-capped
        ]
        for is_correct, predicted, score, length in specs:
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "trace_id": trace_id,
                    "sample_id": len(rows),
                    "is_correct": is_correct,
                    "fold": 0,
                    "layer": 7,
                    "predicted_answer": predicted,
                    "trace_length": length,
                    "rmd_score": float(score),
                    "entropy_score": float(score) - 0.1,
                }
            )
            trace_id += 1
    return rows


def test_parseable_within_prompt_metrics_collapses_truncation_driven_mixedness():
    rows = _artifact_rows()

    full = within_prompt_concordance(rows, score_key="rmd_score")
    parseable = parseable_within_prompt_metrics(rows)

    # Full set: every prompt is mixed and RMD perfectly ranks correct over unparsed.
    assert full["n_mixed_prompts"] == 4
    assert full["macro"] == pytest.approx(1.0)
    # Parseable-only: no incorrect parseable traces remain, so no mixed prompts.
    assert parseable["methods"]["rmd"]["n_mixed_prompts"] == 0
    assert parseable["methods"]["rmd"]["within_prompt_macro"] is None
    assert parseable["n_parseable_traces"] == 8


def test_analyze_oof_scores_emits_truncation_and_parseable_blocks():
    rows = _artifact_rows()
    config = {
        "dataset": "math500",
        "model": "deepseek",
        "layers": [7],
        "pca_dim": 128,
        "n": 3,
        "expected_prompts": 4,
        "n_splits": 2,
        "seed": 42,
        "data_report": {"partial_data": False, "observed_complete_prompts": 4},
    }

    result = analyze_oof_scores(
        rows, config=config, n_bootstrap=5, seed=1, max_new_tokens=2048
    )

    assert result["truncation"]["n_unparsed"] == 4
    assert result["truncation"]["n_capped"] == 4
    assert "truncation" in result["layers"]["7"]
    assert (
        result["layers"]["7"]["parseable_only"]["methods"]["rmd"]["n_mixed_prompts"]
        == 0
    )


def test_paired_rmd_minus_length_is_surfaced_and_reported(tmp_path: Path):
    rows = []
    trace_id = 0
    for prompt_id in range(4):
        for is_correct, score in [(1, 3.0 + prompt_id), (0, 1.0 + prompt_id)]:
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "trace_id": trace_id,
                    "sample_id": len(rows),
                    "is_correct": is_correct,
                    "fold": 0,
                    "layer": 7,
                    "predicted_answer": "x",
                    "trace_length": 100,
                    "rmd_score": float(score),
                    "length_score": float(score) - 0.5,
                }
            )
            trace_id += 1
    config = {
        "dataset": "math500",
        "model": "deepseek",
        "layers": [7],
        "pca_dim": 128,
        "n": 2,
        "expected_prompts": 4,
        "n_splits": 2,
        "seed": 42,
        "data_report": {"partial_data": False, "observed_complete_prompts": 4},
    }

    result = analyze_oof_scores(rows, config=config, n_bootstrap=20, seed=1, max_new_tokens=100)

    paired_length = result["layers"]["7"]["paired_rmd_minus_length"]
    assert "pooled_auc" in paired_length
    assert "point_estimate" in paired_length["pooled_auc"]

    markdown_path = tmp_path / "report.md"
    write_markdown(result, markdown_path)
    text = markdown_path.read_text()
    assert "Primary contrast: RMD − length" in text


def test_trace_csv_writes_enriched_schema_and_blank_missing_values(tmp_path: Path):
    row = _rows({0: [(1, 1.0)]})[0]
    row.update(
        {
            "predicted_answer": "42",
            "gold_answer": "42",
            "mean_logprob": None,
            "trace_length": 3,
            "entropy_score": -0.2,
            "logprob_score": None,
            "length_score": -np.log1p(3),
            "activation_norm_score": -2.0,
            "centroid_score": -1.0,
        }
    )
    path = tmp_path / "scores.csv"

    write_trace_csv([row], path)

    written = next(csv.DictReader(path.open()))
    assert written["predicted_answer"] == "42"
    assert written["trace_length"] == "3"
    assert written["logprob_score"] == ""
