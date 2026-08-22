import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.budget_outcomes import cap_accounting, continuation_case, population_table


def _write_oof(directory: Path, rows: list[dict], layers=(7, 14, 21)) -> None:
    """Write an OOF CSV in the real shape: one row per (trace, layer)."""
    path = directory / "qwen_bestofn_full" / "math500"
    path.mkdir(parents=True, exist_ok=True)
    expanded = [{**row, "layer": layer} for layer in layers for row in rows]
    pd.DataFrame(expanded).to_csv(path / "math500_prompt_decomposition_oof.csv", index=False)


def _trace(trace_id: int, length: int, answer, correct: float) -> dict:
    return {
        "prompt_id": trace_id // 8,
        "trace_id": trace_id,
        "trace_length": length,
        "predicted_answer": answer,
        "is_correct": correct,
    }


def test_missing_answer_is_not_parseable(tmp_path):
    """A capped trace with no answer must not be counted as having answered.

    Through pandas an absent answer arrives as NaN, and ``is_parseable_answer``
    is True for NaN because ``str(nan).strip()`` is the non-empty ``"nan"``.
    Reading the column without screening NaN first reports every capped trace as
    parseable and silently zeroes the unparsed count.
    """
    _write_oof(tmp_path, [
        _trace(0, 1024, None, 0.0),      # capped, no answer
        _trace(1, 1024, "42", 1.0),      # capped, answered
        _trace(2, 512, "17", 1.0),       # finished, answered
        _trace(3, 512, None, 0.0),       # finished, no answer
    ])
    accounting = cap_accounting("qwen", tmp_path, cap=1024)

    assert accounting["n_capped"] == 2
    assert accounting["n_capped_parseable"] == 1
    assert accounting["n_capped_unparsed"] == 1
    assert accounting["n_uncapped_unparsed"] == 1


def test_layer_sweep_does_not_multiply_counts(tmp_path):
    """OOF rows are per (trace, layer); counting rows would triple every total."""
    rows = [_trace(index, 1024, None, 0.0) for index in range(5)]
    _write_oof(tmp_path, rows, layers=(7, 14, 21))

    accounting = cap_accounting("qwen", tmp_path, cap=1024)

    assert accounting["n_traces"] == 5
    assert accounting["n_capped"] == 5


def test_empty_string_answer_is_not_parseable(tmp_path):
    _write_oof(tmp_path, [_trace(0, 1024, "   ", 0.0)])

    assert cap_accounting("qwen", tmp_path, cap=1024)["n_capped_parseable"] == 0


def test_population_table_flags_intervals_that_span_zero():
    results = {
        "qwen": {
            "populations": {
                "full_population": {
                    "n_prompts": 500,
                    "base_accuracy": 0.62,
                    "n_capped_prompts": 108,
                    "n_automatic_failures": 2,
                    "n_unparsed_traces": 328,
                    "models": {},
                    "paired_deltas": {
                        "B1_minus_B0_aurc": {
                            "point_estimate": -0.05,
                            "ci_low": -0.08,
                            "ci_high": -0.02,
                            "p_two_sided": 0.0,
                        }
                    },
                },
                "valid_plurality": {
                    "n_prompts": 498,
                    "base_accuracy": 0.622,
                    "n_capped_prompts": 106,
                    "n_automatic_failures": 0,
                    "n_unparsed_traces": 326,
                    "models": {},
                    "paired_deltas": {
                        "B1_minus_B0_aurc": {
                            "point_estimate": -0.01,
                            "ci_low": -0.03,
                            "ci_high": 0.01,
                            "p_two_sided": 0.4,
                        }
                    },
                },
            }
        }
    }

    table = population_table(results).set_index("population")

    assert bool(table.loc["full_population", "excludes_zero"]) is True
    assert bool(table.loc["valid_plurality", "excludes_zero"]) is False
    # Retention is measured against the full population, not against the widest
    # row present, so a missing population cannot inflate it.
    assert table.loc["valid_plurality", "retained"] == pytest.approx(498 / 500)


def test_continuation_accuracy_discrepancy_is_carried_not_hidden(tmp_path):
    """The stored share and the outcome counts disagree; both must survive."""
    path = tmp_path / "deepseek_bestofn_full" / "math500"
    path.mkdir(parents=True)
    (path / "math500_continue_capped_results.json").write_text(
        '{"n_continued": 50, "accuracy_of_completions": 0.4571, "outcomes": '
        '{"completed_correct": 16, "completed_incorrect": 18, '
        '"still_unfinished": 13, "degenerate_loop": 3}, "settings": {}}'
    )

    case = continuation_case(tmp_path)

    assert case["n_completed"] == 34
    assert case["accuracy_of_completions_recomputed"] == pytest.approx(16 / 34)
    assert case["accuracy_of_completions_as_stored"] == pytest.approx(0.4571)
