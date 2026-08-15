import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controls.continue_capped import (
    capped_traces,
    classify,
    select_traces,
    summarize,
)


def _extract(text):
    """Stand-in for collect_data.extract_math_answer."""
    if "\\boxed{" not in text:
        return None
    start = text.index("\\boxed{") + len("\\boxed{")
    return text[start:text.index("}", start)]


def _normalize(text):
    return text.strip().lower().replace(" ", "") if text else text


def _classify(tokens, text, terminated, gold="7"):
    return classify(
        tokens, text, terminated, gold, extract_answer=_extract, normalize=_normalize
    )


def test_a_trace_that_terminates_with_the_gold_answer_completed_correct():
    result = _classify(["so", "the", "answer"], "so \\boxed{7}", terminated=True)

    assert result["outcome"] == "completed_correct"
    assert result["correct"]


def test_a_trace_that_terminates_with_a_wrong_answer_completed_incorrect():
    result = _classify(["so"], "so \\boxed{9}", terminated=True)

    assert result["outcome"] == "completed_incorrect"
    assert result["predicted"] == "9"


def test_a_trace_that_terminates_without_any_answer_is_still_a_completion():
    """Termination is about stopping, not about being right."""
    result = _classify(["hmm"], "hmm, I give up", terminated=True)

    assert result["outcome"] == "completed_incorrect"
    assert result["predicted"] is None


def test_a_trace_that_never_terminates_is_still_unfinished():
    result = _classify([f"tok{i}" for i in range(900)], "reasoning", terminated=False)

    assert result["outcome"] == "still_unfinished"


def test_repetition_outranks_non_termination():
    """A looping trace is a termination failure, not slow convergence."""
    result = _classify(["a", "b"] * 500, "a b " * 500, terminated=False)

    assert result["outcome"] == "degenerate_loop"


def test_an_answer_followed_by_pages_more_is_flagged_as_not_stopping():
    text = "\\boxed{7}" + " wait" * 400

    result = _classify(["tok"] * 500, text, terminated=True)

    assert result["answered_then_continued"]


def test_a_short_wrap_up_after_the_answer_is_not_flagged():
    result = _classify(["tok"] * 20, "\\boxed{7}. That is the final answer.", True)

    assert not result["answered_then_continued"]


def test_summary_reports_shares_and_the_extra_budget_completions_needed():
    results = [
        {"outcome": "completed_correct", "terminated": True, "correct": True,
         "continuation_tokens": 300, "answered_then_continued": False},
        {"outcome": "completed_incorrect", "terminated": True, "correct": False,
         "continuation_tokens": 900, "answered_then_continued": True},
        {"outcome": "still_unfinished", "terminated": False, "correct": False,
         "continuation_tokens": 8192, "answered_then_continued": False},
        {"outcome": "degenerate_loop", "terminated": False, "correct": False,
         "continuation_tokens": 8192, "answered_then_continued": False},
    ]

    report = summarize(results, excluded=[{"trace_id": 1}])

    assert report["n_continued"] == 4
    assert report["n_excluded_as_already_degenerate"] == 1
    assert report["outcome_shares"]["completed_correct"] == 0.25
    assert report["accuracy_of_completions"] == 0.5
    assert report["n_answered_then_continued"] == 1
    # Only the traces that actually finished contribute a budget requirement.
    assert report["extra_tokens_to_finish_percentiles"]["90"] == 900


def test_summary_of_an_empty_run_reports_none_rather_than_zero():
    report = summarize([], excluded=[])

    assert report["outcome_shares"]["completed_correct"] is None
    assert report["accuracy_of_completions"] is None
    assert report["extra_tokens_to_finish_percentiles"] is None


def _write_batch(path, metas, tokens):
    np.savez(
        path,
        metadata=np.array(metas, dtype=object),
        **{f"tokens_{m['trace_id']}": np.array(t, dtype=object)
           for m, t in zip(metas, tokens)},
    )


def test_capped_traces_indexes_only_traces_at_the_budget(tmp_path):
    _write_batch(
        tmp_path / "batch_0000.npz",
        [
            {"n_tokens": 100, "trace_id": 0, "idx": 0, "sample_id": 0,
             "gold": "7", "is_correct": False},
            {"n_tokens": 40, "trace_id": 1, "idx": 0, "sample_id": 1,
             "gold": "7", "is_correct": True},
        ],
        [["a"] * 100, ["b"] * 40],
    )

    found = capped_traces(tmp_path, cap=100)

    assert [entry["trace_id"] for entry in found] == [0]


def test_already_looping_traces_are_excluded_from_the_sample(tmp_path):
    _write_batch(
        tmp_path / "batch_0000.npz",
        [
            {"n_tokens": 800, "trace_id": 0, "idx": 0, "sample_id": 0,
             "gold": "7", "is_correct": False},
            {"n_tokens": 800, "trace_id": 1, "idx": 1, "sample_id": 0,
             "gold": "7", "is_correct": False},
        ],
        [["x", "y"] * 400, [f"tok{i}" for i in range(800)]],
    )

    selected, degenerate = select_traces(capped_traces(tmp_path, cap=800), n=5, seed=0)

    assert [entry["trace_id"] for entry in degenerate] == [0]
    assert [entry["trace_id"] for entry in selected] == [1]


def test_selection_is_reproducible_under_a_seed(tmp_path):
    _write_batch(
        tmp_path / "batch_0000.npz",
        [
            {"n_tokens": 600, "trace_id": i, "idx": i, "sample_id": 0,
             "gold": "7", "is_correct": False}
            for i in range(6)
        ],
        [[f"t{i}_{j}" for j in range(600)] for i in range(6)],
    )
    records = capped_traces(tmp_path, cap=600)

    first, _ = select_traces(records, n=3, seed=7)
    second, _ = select_traces(records, n=3, seed=7)

    assert [entry["trace_id"] for entry in first] == [
        entry["trace_id"] for entry in second
    ]
    assert len(first) == 3


def test_merged_shards_cover_the_population_once(tmp_path):
    from controls.continue_capped import merge_shards

    entries = [
        {"trace_id": i, "outcome": "completed_correct", "terminated": True,
         "correct": True, "continuation_tokens": 100 * i,
         "answered_then_continued": False}
        for i in range(1, 5)
    ]
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(entries[::2]))
    second.write_text(json.dumps(entries[1::2]))

    report, merged = merge_shards([first, second], excluded=[])

    assert len(merged) == 4
    assert report["n_continued"] == 4


def test_overlapping_shards_are_rejected_rather_than_double_counted(tmp_path):
    from controls.continue_capped import merge_shards

    entry = {"trace_id": 1, "outcome": "still_unfinished", "terminated": False,
             "correct": False, "continuation_tokens": 8192,
             "answered_then_continued": False}
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps([entry]))
    second.write_text(json.dumps([entry]))

    with pytest.raises(ValueError, match="shards overlap"):
        merge_shards([first, second], excluded=[])
