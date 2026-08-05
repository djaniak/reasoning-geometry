import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from difficulty_control import _delta_seed, sibling_difficulty


def _trace(prompt_id: int, length: int) -> dict:
    return {"prompt_id": prompt_id, "trace_length": length}


def test_the_longest_finisher_ignores_capped_siblings():
    # Two siblings cap at 1024; the longest that actually terminated ran 512.
    rows = [_trace(0, 1024), _trace(0, 1024), _trace(0, 512), _trace(0, 100)]

    entry = sibling_difficulty(rows, cap=1024)[0]

    assert entry["longest_finisher_frac"] == pytest.approx(0.5)
    assert entry["capped_fraction"] == pytest.approx(0.5)


def test_a_fully_capped_prompt_has_no_finisher_to_measure():
    """NaN, not zero: there is no evidence, and zero would read as 'finished instantly'."""
    rows = [_trace(0, 1024), _trace(0, 1024)]

    entry = sibling_difficulty(rows, cap=1024)[0]

    assert np.isnan(entry["longest_finisher_frac"])
    assert entry["capped_fraction"] == pytest.approx(1.0)


def test_budget_edge_pressure_separates_the_two_regimes():
    """The statistic the sibling-structure report is built on: 88% against 35%."""
    edge = [_trace(0, 900), _trace(0, 1024)]
    slack = [_trace(1, 350), _trace(1, 200)]

    difficulty = sibling_difficulty(edge + slack, cap=1024)

    assert difficulty[0]["longest_finisher_frac"] > difficulty[1]["longest_finisher_frac"]


def test_traces_over_the_cap_still_count_as_capped():
    # A stored length can exceed the budget by a token or two; >= is the test.
    rows = [_trace(0, 1030), _trace(0, 400)]

    assert sibling_difficulty(rows, cap=1024)[0]["capped_fraction"] == pytest.approx(0.5)


def test_missing_lengths_do_not_crash_the_accounting():
    rows = [{"prompt_id": 0, "trace_length": None}, _trace(0, 512)]

    entry = sibling_difficulty(rows, cap=1024)[0]

    assert entry["longest_finisher_frac"] == pytest.approx(0.5)
    assert entry["capped_fraction"] == pytest.approx(0.0)


def test_dispersion_is_zero_when_siblings_agree_on_length():
    rows = [_trace(0, 500), _trace(0, 500), _trace(0, 500)]

    assert sibling_difficulty(rows, cap=1024)[0]["length_dispersion"] == pytest.approx(0.0)


def test_the_delta_seed_matches_the_frozen_analysis_convention():
    """B1-B0 must reproduce the locked artifact, which requires the same seed.

    `incremental_abstention.run_incremental_analysis` seeds each paired bootstrap
    with `seed + 1000 + len(metric) + len(label)`. If that convention drifts, the
    harness check in this module silently stops being a check.
    """
    assert _delta_seed(42, "B1_minus_B0", "auacc") == 42 + 1000 + 5 + 11
