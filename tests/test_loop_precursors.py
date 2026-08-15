import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controls.loop_precursors import (
    tail_periodicity,
    onset_report,
    repetition_flags,
    repetition_onset,
)


def test_novel_text_flags_nothing():
    tokens = [f"tok{i}" for i in range(100)]

    assert not repetition_flags(tokens, n=4).any()


def test_repeated_block_is_flagged_on_its_second_occurrence():
    block = [f"tok{i}" for i in range(20)]
    flags = repetition_flags(block + block, n=4)

    # The first copy is novel; the second re-treads n-grams already seen.
    assert not flags[:17].any()
    assert flags[20:].all()


def test_novelty_is_measured_against_the_whole_prefix_not_a_window():
    block = [f"tok{i}" for i in range(10)]
    filler = [f"pad{i}" for i in range(400)]
    flags = repetition_flags(block + filler + block, n=4)

    # The recycled block sits 400 tokens after its first appearance.
    assert flags[-7:].all()


def test_onset_is_none_for_a_trace_that_never_repeats():
    tokens = [f"tok{i}" for i in range(1000)]

    assert repetition_onset(tokens, n=4, window=100, threshold=0.5) is None


def test_onset_is_none_when_the_trace_is_shorter_than_the_window():
    assert repetition_onset(["a", "b", "c"], n=2, window=100) is None


def test_onset_lands_after_the_loop_starts_and_before_it_ends():
    prefix = [f"tok{i}" for i in range(600)]
    loop = [f"cycle{i}" for i in range(20)] * 40  # 800 tokens of tight repetition
    onset = repetition_onset(prefix + loop, n=4, window=100, threshold=0.5)

    assert onset is not None
    assert 600 <= onset < len(prefix) + len(loop)


def test_onset_reports_the_end_of_the_qualifying_window():
    """An online rule can only act once the window's evidence exists."""
    prefix = [f"tok{i}" for i in range(300)]
    loop = [f"cycle{i}" for i in range(10)] * 60
    window = 100
    onset = repetition_onset(prefix + loop, n=4, window=window, threshold=0.5)

    assert onset >= window


def test_repetition_flags_rejects_a_degenerate_ngram():
    with pytest.raises(ValueError):
        repetition_flags(["a", "b"], n=0)


def test_repetition_flags_handles_a_trace_shorter_than_the_ngram():
    assert repetition_flags(["a", "b"], n=8).size == 0


def _row(capped, onset, is_correct, cap=8192, periodicity=0.05):
    return {
        "capped": capped,
        "onset": onset,
        "onset_frac": None if onset is None else onset / cap,
        "is_correct": is_correct,
        "tail_periodicity": periodicity,
    }


def test_onset_report_separates_flagged_uncapped_accuracy():
    rows = [
        _row(True, 3000, False),
        _row(True, 5000, False),
        _row(True, None, True),
        _row(False, 4000, False),   # flagged but finished: wrong
        _row(False, None, True),
        _row(False, None, True),
    ]

    report = onset_report(rows)

    assert report["n_capped"] == 3
    assert report["capped_detected"] == 2
    assert report["uncapped_flagged"] == 1
    assert report["accuracy_uncapped_flagged"] == 0.0
    assert report["accuracy_uncapped_unflagged"] == 1.0


def test_onset_report_handles_an_empty_capped_population():
    report = onset_report([_row(False, None, True)])

    assert report["n_capped"] == 0
    assert report["capped_detection_rate"] is None
    assert report["onset_frac_percentiles"] is None


def test_tail_periodicity_finds_a_tight_cycle():
    tokens = [f"tok{i}" for i in range(300)] + ["a", "b", "c"] * 200
    period, score = tail_periodicity(tokens, tail=500)

    assert period == 3
    assert score > 0.95


def test_tail_periodicity_is_low_for_non_repeating_text():
    _, score = tail_periodicity([f"tok{i}" for i in range(800)], tail=500)

    assert score < 0.05


def test_tail_periodicity_separates_a_loop_from_reused_vocabulary():
    """Ordinary math reuses tokens heavily without cycling a fixed block."""
    rng = np.random.default_rng(0)
    vocabulary = ["=", "x", "+", "2", "(", ")", "the", "so"]
    prose = [str(rng.choice(vocabulary)) for _ in range(800)]
    loop = ["1", ","] * 400

    _, prose_score = tail_periodicity(prose, tail=500)
    _, loop_score = tail_periodicity(loop, tail=500)

    assert loop_score > 0.95
    assert prose_score < loop_score / 2


def test_tail_periodicity_handles_a_trace_too_short_to_score():
    assert tail_periodicity(["a", "b"], tail=500) == (0, 0.0)


def test_degenerate_share_is_reported_against_the_calibrated_threshold():
    rows = [
        _row(True, 100, False, periodicity=0.95),   # stuck loop
        _row(True, 100, False, periodicity=0.10),   # unfinished but coherent
        _row(True, None, False, periodicity=0.05),
        _row(False, None, True, periodicity=0.02),
    ]

    degenerate = onset_report(rows)["degenerate"]

    assert degenerate["n_capped_degenerate"] == 1
    assert degenerate["share_of_capped"] == pytest.approx(1 / 3, abs=1e-4)
    assert degenerate["n_uncapped_degenerate"] == 0
