import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posttraining_error_recovery import (
    _score_continuation,
    load_existing_results,
    prepare_cases,
    summarize_results,
)


def _gold(problem: str, answer: int) -> dict:
    return {"question": problem, "answer": f"work\n#### {answer}"}


def _process_row(
    case_id: str,
    problem: str,
    *,
    label: int = 1,
    steps: list[str] | None = None,
) -> dict:
    return {
        "id": case_id,
        "generator": "generator-a",
        "problem": problem,
        "steps": steps or ["correct step", "first wrong step", "later step"],
        "final_answer_correct": label == -1,
        "label": label,
    }


def test_prepare_cases_builds_pre_and_err_from_the_expert_boundary():
    cases, excluded = prepare_cases(
        [_process_row("gsm8k-7", "How many?", label=1)],
        [_gold("How many?", 24)],
        limit=50,
        seed=42,
    )

    assert excluded == {}
    assert len(cases) == 1
    case = cases[0]
    assert case["gold"] == "24"
    assert case["pre_prefix"] == "correct step"
    assert case["err_prefix"] == "correct step\n\nfirst wrong step"
    assert case["prompts"]["PRE"].encode() == (
        "Solve the problem step by step. End with `#### <answer>`.\n\n"
        "Problem:\nHow many?\n\nSolution:\ncorrect step\n"
    ).encode()
    assert case["prompts"]["ERR"].endswith(
        "Solution:\ncorrect step\n\nfirst wrong step\n"
    )


def test_prepare_cases_excludes_non_continuable_boundaries_with_counts():
    process_rows = [
        _process_row("correct", "p0", label=-1),
        _process_row("final", "p1", label=2),
        _process_row("answer", "p2", label=1, steps=["ok", r"wrong \\boxed{9}", "later"]),
        _process_row("invalid", "p3", label=9),
    ]
    gold_rows = [_gold(f"p{i}", i) for i in range(4)]

    cases, excluded = prepare_cases(process_rows, gold_rows, limit=50, seed=42)

    assert cases == []
    assert excluded == {
        "answer_in_err_prefix": 1,
        "error_is_final_step": 1,
        "invalid_error_label": 1,
        "no_error": 1,
    }


def test_prepare_cases_fails_when_processbench_problem_has_no_gold_join():
    with pytest.raises(ValueError, match="no GSM8K gold answer"):
        prepare_cases(
            [_process_row("missing", "unmatched problem")],
            [_gold("another problem", 1)],
            limit=50,
            seed=42,
        )


def _result(case_id: str, model: str, condition: str, correct: bool) -> dict:
    return {
        "case_id": case_id,
        "model_name": model,
        "condition": condition,
        "seed": 42,
        "correct": correct,
    }


def test_summary_reports_paired_difference_in_differences():
    rows = [
        _result("a", "parent", "PRE", True),
        _result("a", "parent", "ERR", False),
        _result("a", "distilled", "PRE", True),
        _result("a", "distilled", "ERR", True),
        _result("b", "parent", "PRE", True),
        _result("b", "parent", "ERR", True),
        _result("b", "distilled", "PRE", True),
        _result("b", "distilled", "ERR", False),
    ]

    summary = summarize_results(
        rows,
        parent_model="parent",
        distilled_model="distilled",
        n_bootstrap=20,
        seed=0,
    )

    assert summary["n_complete_cases"] == 2
    assert summary["models"]["parent"]["error_damage"] == 0.5
    assert summary["models"]["distilled"]["error_damage"] == 0.5
    assert summary["distilled_minus_parent_error_damage"]["point"] == 0.0
    assert len(summary["distilled_minus_parent_error_damage"]["ci95"]) == 2


def test_summary_drops_cases_without_all_four_paired_cells():
    rows = [
        _result("complete", "parent", "PRE", True),
        _result("complete", "parent", "ERR", False),
        _result("complete", "distilled", "PRE", True),
        _result("complete", "distilled", "ERR", True),
        _result("partial", "parent", "PRE", True),
    ]

    summary = summarize_results(
        rows,
        parent_model="parent",
        distilled_model="distilled",
        n_bootstrap=0,
        seed=0,
    )

    assert summary["n_complete_cases"] == 1
    assert summary["n_incomplete_cases"] == 1
    assert summary["distilled_minus_parent_error_damage"]["ci95"] is None


def test_resume_refuses_to_mix_results_from_another_protocol(tmp_path):
    path = tmp_path / "results.jsonl"
    row = {
        "case_id": "a",
        "model_name": "parent",
        "condition": "PRE",
        "seed": 42,
        "correct": True,
        "settings": {"protocol_version": 1, "max_new_tokens": 256},
    }
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(FileExistsError, match="--resume"):
        load_existing_results(path, row["settings"], resume=False)
    with pytest.raises(ValueError, match="incompatible settings"):
        load_existing_results(
            path,
            {"protocol_version": 1, "max_new_tokens": 512},
            resume=True,
        )

    assert load_existing_results(path, row["settings"], resume=True) == [row]


def test_continuation_scoring_stops_at_eos_and_distinguishes_a_cap():
    decoded = {1: "reasoning ", 2: "#### 24", 3: "ignored"}

    completed = _score_continuation(
        [1, 2, 99, 3],
        eos_ids={99},
        max_new_tokens=4,
        decode=lambda ids: "".join(decoded[token] for token in ids),
        gold="24",
    )
    capped = _score_continuation(
        [1, 1, 1, 1],
        eos_ids={99},
        max_new_tokens=4,
        decode=lambda ids: "".join(decoded[token] for token in ids),
        gold="24",
    )

    assert completed == {
        "ids": [1, 2],
        "continuation_text": "reasoning #### 24",
        "predicted": "24",
        "correct": True,
        "explicit_answer": True,
        "terminated": True,
        "truncated": False,
        "censored": False,
    }
    assert not capped["terminated"]
    assert capped["truncated"]
    assert capped["censored"]


def test_summary_reports_uncensored_sensitivity_separately():
    rows = [
        _result("usable", "parent", "PRE", True),
        _result("usable", "parent", "ERR", False),
        _result("usable", "distilled", "PRE", True),
        _result("usable", "distilled", "ERR", True),
        _result("capped", "parent", "PRE", True),
        _result("capped", "parent", "ERR", False),
        _result("capped", "distilled", "PRE", True),
        {**_result("capped", "distilled", "ERR", False), "censored": True},
    ]

    summary = summarize_results(
        rows,
        parent_model="parent",
        distilled_model="distilled",
        n_bootstrap=0,
        seed=0,
    )

    assert summary["n_complete_cases"] == 2
    assert summary["n_uncensored_complete_cases"] == 1
    assert summary["uncensored_distilled_minus_parent_error_damage"]["point"] == -1.0
