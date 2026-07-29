import sys
import csv
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prompt_selection
from prompt_selection import (
    INVALID_ANSWER,
    evaluate_prompt_selection,
    majority_cluster_mean_tiebreak_answer,
    majority_answer,
    majority_rmd_tiebreak_answer,
    rank_weights,
    read_oof_csv,
    rmd_weighted_answer,
    select_top_trace,
)


def _row(
    prompt_id,
    sample_id,
    correct,
    answer,
    rmd,
    entropy,
    logprob,
    layer=7,
):
    return {
        "prompt_id": prompt_id,
        "trace_id": prompt_id * 10 + sample_id,
        "sample_id": sample_id,
        "is_correct": int(correct),
        "fold": prompt_id % 2,
        "layer": layer,
        "predicted_answer": answer,
        "gold_answer": "A",
        "mean_logprob": logprob,
        "entropy_score": entropy,
        "rmd_score": rmd,
        "rmd_high_entropy_q20_score": rmd,
        "contrast_high_entropy_q20_score": rmd,
        "raw_score": -rmd,
    }


def test_rank_weights_use_average_ranks_with_larger_weight_for_higher_score():
    assert rank_weights([0.5, 0.5, 2.0]) == pytest.approx([1.5, 1.5, 3.0])


def test_top_trace_ties_are_resolved_by_logprob_then_sample_id():
    rows = [
        _row(0, 1, False, "B", 1.0, 0.5, -0.2),
        _row(0, 0, True, "A", 1.0, 0.5, -0.2),
    ]

    selected = select_top_trace(rows, "rmd_score")

    assert selected["sample_id"] == 0


def test_rmd_weighted_and_hybrid_voting_follow_locked_tie_rules():
    rows = [
        _row(0, 0, True, "A", 4.0, 0.1, -0.4),
        _row(0, 1, True, "A", 1.0, 0.2, -0.3),
        _row(0, 2, False, "B", 3.0, 0.3, -0.1),
        _row(0, 3, False, "B", 2.0, 0.4, -0.2),
    ]

    assert rmd_weighted_answer(rows) == "B"
    assert majority_rmd_tiebreak_answer(rows) == "B"


def test_cluster_mean_tiebreak_never_overturns_strict_majority():
    rows = [
        _row(0, 0, True, "A", 0.0, 0.1, -0.5),
        _row(0, 1, True, "A", 0.0, 0.1, -0.5),
        _row(0, 2, True, "A", 0.0, 0.1, -0.5),
        _row(0, 3, False, "B", 10.0, 0.1, -0.1),
        _row(0, 4, False, "B", 10.0, 0.1, -0.1),
    ]

    assert (
        majority_cluster_mean_tiebreak_answer(
            rows, "contrast_high_entropy_q20_score"
        )
        == "A"
    )


def test_cluster_mean_tiebreak_uses_named_mean_then_deterministic_fallback():
    rows = [
        _row(0, 0, True, "A", 1.0, 0.1, -0.4),
        _row(0, 1, True, "A", 1.0, 0.1, -0.1),
        _row(0, 2, False, "B", 3.0, 0.1, -0.3),
        _row(0, 3, False, "B", 3.0, 0.1, -0.3),
    ]

    assert (
        majority_cluster_mean_tiebreak_answer(
            rows, "contrast_high_entropy_q20_score"
        )
        == "B"
    )
    assert majority_cluster_mean_tiebreak_answer(rows, "mean_logprob") == "A"

    invalid_tie = [
        _row(0, 0, True, "A", 1.0, 0.1, -0.4),
        _row(0, 1, False, None, 4.0, 0.1, -0.1),
    ]
    assert (
        majority_cluster_mean_tiebreak_answer(
            invalid_tie, "contrast_high_entropy_q20_score"
        )
        == INVALID_ANSWER
    )


def test_rmd_voting_ranks_all_traces_before_discarding_unparsed_answers(
    monkeypatch,
):
    rows = [
        _row(0, 0, True, "A", 4.0, 0.1, -0.4),
        _row(0, 1, False, None, 3.0, 0.2, -0.3),
        _row(0, 2, False, "B", 2.0, 0.3, -0.2),
        _row(0, 3, True, "A", 1.0, 0.4, -0.1),
    ]
    ranked_inputs = []

    def recording_rank_weights(scores):
        ranked_inputs.append(scores)
        return rank_weights(scores)

    monkeypatch.setattr(prompt_selection, "rank_weights", recording_rank_weights)

    rmd_weighted_answer(rows)
    majority_rmd_tiebreak_answer(rows)

    assert ranked_inputs == [[4.0, 3.0, 2.0, 1.0]] * 2


def test_unparsed_answers_are_counted_as_invalid_votes():
    rows = [
        _row(0, 0, True, "A", 8.0, 0.1, -0.1),
        _row(0, 1, True, "A", 7.0, 0.2, -0.2),
        _row(0, 2, True, "A", 6.0, 0.3, -0.3),
        _row(0, 3, False, None, 5.0, 0.4, -0.4),
        _row(0, 4, False, None, 4.0, 0.5, -0.5),
        _row(0, 5, False, None, 3.0, 0.6, -0.6),
        _row(0, 6, False, None, 2.0, 0.7, -0.7),
        _row(0, 7, False, None, 1.0, 0.8, -0.8),
    ]

    assert majority_answer(rows) == INVALID_ANSWER
    assert rmd_weighted_answer(rows) == "A"
    assert majority_rmd_tiebreak_answer(rows) == INVALID_ANSWER


def test_prompt_selection_reports_answer_parse_diagnostics():
    rows = [
        _row(0, 0, True, "A", 4.0, 0.1, -0.1),
        _row(0, 1, False, None, 1.0, 0.8, -0.2),
        _row(1, 0, False, None, 2.0, 0.9, -0.1),
        _row(1, 1, False, None, 3.0, 0.1, -0.2),
    ]

    result = evaluate_prompt_selection(
        rows,
        model="qwen",
        dataset="math500",
        n_bootstrap=0,
        seed=42,
    )

    layer = result["layers"]["7"]
    assert result["settings"]["invalid_answer_policy"] == "count as failure"
    assert layer["answer_parsing"] == {
        "n_traces": 4,
        "n_parsed": 1,
        "parse_rate": pytest.approx(0.25),
        "correct_parse_rate": pytest.approx(1.0),
        "incorrect_parse_rate": pytest.approx(0.0),
        "n_prompts_without_parsed_answer": 1,
    }
    assert layer["selectors"]["majority_vote"]["pass_at_1"] == pytest.approx(0.5)


def test_prompt_selection_reports_top1_and_vote_outcomes_with_bootstrap():
    rows = [
        _row(0, 0, True, "A", 4.0, 0.1, -0.1),
        _row(0, 1, False, "B", 1.0, 0.8, -0.2),
        _row(1, 0, False, "B", 2.0, 0.9, -0.1),
        _row(1, 1, True, "A", 3.0, 0.1, -0.2),
    ]

    result = evaluate_prompt_selection(
        rows,
        model="qwen",
        dataset="math500",
        n_bootstrap=20,
        seed=42,
    )

    selectors = result["layers"]["7"]["selectors"]
    assert selectors["random"]["pass_at_1"] == pytest.approx(0.5)
    assert selectors["oracle_pass_at_n"]["pass_at_1"] == pytest.approx(1.0)
    assert selectors["top1_rmd"]["pass_at_1"] == pytest.approx(1.0)
    assert selectors["top1_entropy"]["pass_at_1"] == pytest.approx(0.0)
    assert selectors["rmd_rank_weighted_vote"]["n_prompts"] == 2
    assert selectors["top1_rmd"]["confidence_interval"]["n_valid"] == 20
    assert set(selectors["top1_rmd"]["prompt_outcomes"]) == {"0", "1"}
    assert "majority_mean_logprob_tiebreak" in selectors
    assert "majority_rmd_high_entropy_q20_tiebreak" in selectors
    assert "majority_contrast_high_entropy_q20_tiebreak" in selectors
    paired = result["layers"]["7"]["paired_selector_deltas"]
    geometry_delta = paired[
        "majority_contrast_high_entropy_q20_tiebreak_minus_majority_vote"
    ]
    assert geometry_delta["n_valid"] == 20
    assert geometry_delta["wins"] + geometry_delta["losses"] + geometry_delta[
        "ties"
    ] == 2


def test_prompt_selection_report_surfaces_paired_selector_deltas(tmp_path: Path):
    rows = [
        _row(0, 0, True, "A", 4.0, 0.1, -0.1),
        _row(0, 1, False, "B", 1.0, 0.8, -0.2),
        _row(1, 0, False, "B", 2.0, 0.9, -0.1),
        _row(1, 1, True, "A", 3.0, 0.1, -0.2),
    ]
    result = evaluate_prompt_selection(
        rows, model="qwen", dataset="math500", n_bootstrap=10, seed=42
    )
    report = tmp_path / "selection.md"

    prompt_selection.write_markdown(result, report)

    text = report.read_text()
    assert "Paired selector deltas" in text
    assert "majority_contrast_high_entropy_q20_tiebreak_minus_majority_vote" in text


def test_prompt_selection_report_handles_zero_bootstrap_intervals(tmp_path: Path):
    rows = [
        _row(0, 0, True, "A", 4.0, 0.1, -0.1),
        _row(0, 1, False, "B", 1.0, 0.8, -0.2),
    ]
    result = evaluate_prompt_selection(
        rows, model="qwen", dataset="math500", n_bootstrap=0, seed=42
    )
    report = tmp_path / "selection.md"

    prompt_selection.write_markdown(result, report)

    assert "| NA | NA |" in report.read_text()


def test_read_oof_csv_rejects_legacy_schema(tmp_path: Path):
    path = tmp_path / "legacy.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_id",
                "trace_id",
                "sample_id",
                "is_correct",
                "fold",
                "layer",
                "raw_score",
                "rmd_score",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "prompt_id": 0,
                "trace_id": 0,
                "sample_id": 0,
                "is_correct": 1,
                "fold": 0,
                "layer": 7,
                "raw_score": 0.1,
                "rmd_score": 0.2,
            }
        )

    with pytest.raises(ValueError, match="enriched prompt decomposition"):
        read_oof_csv(path)


def test_read_oof_csv_accepts_enriched_csv_without_new_optional_scores(tmp_path: Path):
    path = tmp_path / "enriched_without_prompt_local.csv"
    fieldnames = [
        "prompt_id",
        "trace_id",
        "sample_id",
        "is_correct",
        "fold",
        "layer",
        "predicted_answer",
        "gold_answer",
        "mean_logprob",
        "trace_length",
        "entropy_score",
        "logprob_score",
        "length_score",
        "activation_norm_score",
        "centroid_score",
        "raw_score",
        "rmd_score",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "prompt_id": 0,
                "trace_id": 0,
                "sample_id": 0,
                "is_correct": 1,
                "fold": 0,
                "layer": 7,
                "predicted_answer": "42",
                "gold_answer": "42",
                "mean_logprob": -0.1,
                "trace_length": 10,
                "entropy_score": -0.2,
                "logprob_score": -0.1,
                "length_score": -2.4,
                "activation_norm_score": -1.0,
                "centroid_score": -0.3,
                "raw_score": -0.4,
                "rmd_score": 0.5,
            }
        )

    rows = read_oof_csv(path)

    assert rows[0]["rmd_score"] == pytest.approx(0.5)
    assert "prompt_local_rmd_score" not in rows[0]
