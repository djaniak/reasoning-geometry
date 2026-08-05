import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sibling_structure import (
    _regime,
    load_rows,
    prompt_table,
    sibling_report,
)


def _trace(prompt_id, length, correct, answer="7", gold="7"):
    return {
        "prompt_id": prompt_id,
        "trace_length": length,
        "is_correct": correct,
        "predicted_answer": answer,
        "gold_answer": gold,
    }


def _group(prompt_id, lengths, correctness, cap=100):
    return [
        _trace(prompt_id, length, correct, answer="7" if correct else "9")
        for length, correct in zip(lengths, correctness)
    ]


def test_capped_and_finished_siblings_are_counted_separately():
    rows = _group(0, [100, 100, 40, 50], [False, False, True, False])

    table = prompt_table(rows, cap=100)

    assert table[0]["n_capped"] == 2
    assert table[0]["n_finished"] == 2
    assert table[0]["n_finished_correct"] == 1


def test_a_trace_exactly_at_the_budget_is_capped():
    """Generation stops after `max_new_tokens` steps, so length == cap is a hit."""
    table = prompt_table(_group(0, [100], [False]), cap=100)

    assert table[0]["n_capped"] == 1


def test_plurality_ignores_unparsed_siblings():
    rows = [
        _trace(0, 10, True, answer="7"),
        _trace(0, 10, False, answer=""),
        _trace(0, 10, False, answer=""),
    ]

    assert prompt_table(rows, cap=100)[0]["plurality_correct"] is True


def test_plurality_is_none_when_nothing_parses():
    rows = [_trace(0, 10, False, answer=""), _trace(0, 10, False, answer="")]

    assert prompt_table(rows, cap=100)[0]["plurality_correct"] is None


def test_finished_plurality_excludes_capped_siblings():
    """Two capped traces agreeing on a wrong answer must not carry the vote."""
    rows = [
        _trace(0, 100, False, answer="9"),
        _trace(0, 100, False, answer="9"),
        _trace(0, 20, True, answer="7"),
    ]

    prompt = prompt_table(rows, cap=100)[0]

    assert prompt["plurality_correct"] is False
    assert prompt["finished_plurality_correct"] is True


def test_most_siblings_capping_is_prompt_limited():
    prompt = prompt_table(
        _group(0, [100] * 6 + [10, 12], [False] * 6 + [True, True]), cap=100
    )[0]

    assert _regime(prompt) == "prompt_limited"


def test_a_lone_cap_beside_correct_finishers_is_trajectory_limited():
    prompt = prompt_table(
        _group(0, [100, 10, 12, 14], [False, True, True, True]), cap=100
    )[0]

    assert _regime(prompt) == "trajectory_limited"


def test_finishers_pressed_against_the_budget_are_borderline():
    """A correct finisher at 95% of budget is still a borderline prompt."""
    prompt = prompt_table(_group(0, [100, 95], [False, True]), cap=100)[0]

    assert prompt["longest_finisher_fraction"] == pytest.approx(0.95)
    assert _regime(prompt) == "budget_borderline"


def test_a_prompt_nobody_solves_is_not_called_trajectory_limited():
    prompt = prompt_table(_group(0, [100, 10, 12, 14], [False] * 4), cap=100)[0]

    assert _regime(prompt) == "unresolved"


def test_report_counts_prompts_where_every_sibling_caps():
    rows = _group(0, [100, 100], [False, False]) + _group(1, [100, 10], [False, True])

    report = sibling_report(rows, cap=100)

    assert report["n_prompts_all_capped"] == 1
    assert report["n_prompts_with_a_capped_sibling"] == 2
    assert report["p_a_sibling_finishes_given_a_cap"] == 0.5
    assert report["p_a_finisher_is_correct_given_a_cap"] == 0.5


def test_report_is_empty_of_rates_when_nothing_caps():
    report = sibling_report(_group(0, [10, 12], [True, True]), cap=100)

    assert report["n_prompts_with_a_capped_sibling"] == 0
    assert report["p_a_sibling_finishes_given_a_cap"] is None
    assert report["capped_trace_accuracy"] is None
    assert report["longest_finisher_fraction_percentiles"] is None


def test_load_rows_takes_one_layer_so_traces_are_not_multiplied(tmp_path):
    csv_path = tmp_path / "oof.csv"
    header = "prompt_id,trace_id,is_correct,layer,predicted_answer,gold_answer,trace_length"
    lines = [header]
    for layer in (7, 21):
        for trace_id in range(3):
            lines.append(f"0,{trace_id},1,{layer},7,7,{100 + trace_id}")
    csv_path.write_text("\n".join(lines) + "\n")

    assert len(load_rows(csv_path)) == 3
    assert len(load_rows(csv_path, layer=7)) == 3


def test_load_rows_rejects_a_layer_that_was_not_probed(tmp_path):
    csv_path = tmp_path / "oof.csv"
    csv_path.write_text(
        "prompt_id,trace_id,is_correct,layer,predicted_answer,gold_answer,trace_length\n"
        "0,0,1,21,7,7,100\n"
    )

    with pytest.raises(ValueError, match="layer 3 not in"):
        load_rows(csv_path, layer=3)
