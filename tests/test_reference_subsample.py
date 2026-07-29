"""Tests for the reference-fit token cap.

The cap exists to keep the reference-fit concatenation inside RAM for
long-trace models. It must (a) hit the requested total exactly, (b) never ask a
trace for more rows than it has, (c) leave short-trace runs untouched so their
results stay bit-identical, and (d) be deterministic across runs.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze
from analyze import (
    _concatenate_hidden_tokens,
    _reference_subsample_plan,
    get_max_reference_tokens,
    set_max_reference_tokens,
)


@pytest.fixture(autouse=True)
def restore_cap():
    original = get_max_reference_tokens()
    yield
    set_max_reference_tokens(original)


def _traces(row_counts, layer=7, dim=4):
    """Traces whose rows encode (trace_index, row_index) so we can track them."""
    traces = []
    for trace_index, count in enumerate(row_counts):
        block = np.zeros((count, dim), dtype=np.float32)
        block[:, 0] = trace_index
        block[:, 1] = np.arange(count)
        traces.append({"hiddens": {layer: block}})
    return traces


def test_no_plan_when_under_cap():
    assert _reference_subsample_plan([100, 200], 1000, seed=0) is None


def test_no_plan_when_cap_disabled():
    assert _reference_subsample_plan([10**6, 10**6], None, seed=0) is None


def test_plan_hits_cap_exactly():
    counts = [1000, 5000, 250, 12345, 7]
    plan = _reference_subsample_plan(counts, 3000, seed=0)
    taken = sum(c if idx is None else len(idx) for c, idx in zip(counts, plan))
    assert taken == 3000


def _taken(counts, plan):
    return [c if idx is None else len(idx) for c, idx in zip(counts, plan)]


def test_dominant_trace_does_not_starve_short_ones():
    # Plain proportional allocation would give the tiny traces zero rows and fit
    # the reference on the long trace alone.
    counts = [1_000_000, 3, 2, 1]
    plan = _reference_subsample_plan(counts, 5000, seed=0)
    taken = _taken(counts, plan)
    assert sum(taken) == 5000
    assert all(t >= 1 for t in taken)
    for count, indices in zip(counts, plan):
        if indices is not None:
            assert len(indices) <= count
            assert indices.max() < count


def test_allocation_is_proportional_and_keeps_every_trace():
    counts = [9000, 1000]
    plan = _reference_subsample_plan(counts, 1000, seed=0)
    taken = _taken(counts, plan)
    assert sum(taken) == 1000
    assert all(t > 0 for t in taken)
    # Roughly the 90/10 split of the source lengths; exact rounding is an
    # implementation detail, so allow a little slack.
    assert 880 <= taken[0] <= 910
    assert 90 <= taken[1] <= 120


def test_budget_smaller_than_trace_count():
    counts = [500, 400, 300, 200]
    plan = _reference_subsample_plan(counts, 2, seed=0)
    taken = _taken(counts, plan)
    assert sum(taken) == 2
    # Spent on the longest traces.
    assert taken[0] == 1 and taken[1] == 1


def test_empty_traces_are_skipped_not_allocated():
    counts = [1000, 0, 500]
    plan = _reference_subsample_plan(counts, 300, seed=0)
    taken = _taken(counts, plan)
    assert sum(taken) == 300
    assert taken[1] == 0
    assert taken[0] > 0 and taken[2] > 0


def test_plan_is_deterministic():
    counts = [500, 700, 900]
    first = _reference_subsample_plan(counts, 600, seed=11)
    second = _reference_subsample_plan(counts, 600, seed=11)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_different_seeds_pick_different_rows():
    counts = [10_000]
    a = _reference_subsample_plan(counts, 500, seed=1)[0]
    b = _reference_subsample_plan(counts, 500, seed=2)[0]
    assert not np.array_equal(a, b)


def test_concatenate_respects_cap_and_draws_from_all_traces():
    traces = _traces([600, 400])
    set_max_reference_tokens(200)
    combined = _concatenate_hidden_tokens(traces, layer=7)
    assert combined.shape[0] == 200
    # Both traces contribute; rows are real rows, not fabricated.
    trace_ids = set(combined[:, 0].astype(int).tolist())
    assert trace_ids == {0, 1}
    for trace_index, count in ((0, 600), (1, 400)):
        rows = combined[combined[:, 0] == trace_index][:, 1]
        assert rows.max() < count
        assert len(set(rows.tolist())) == len(rows)  # sampled without replacement


def test_concatenate_unchanged_when_cap_not_binding():
    """The guarantee that short-trace runs (Qwen) stay bit-identical."""
    traces = _traces([300, 200])
    set_max_reference_tokens(None)
    uncapped = _concatenate_hidden_tokens(traces, layer=7)
    set_max_reference_tokens(2_000_000)
    capped = _concatenate_hidden_tokens(traces, layer=7)
    assert np.array_equal(uncapped, capped)
    assert capped.shape[0] == 500


def test_concatenate_uses_compute_dtype():
    traces = _traces([50, 50])
    set_max_reference_tokens(40)
    original = analyze.get_compute_dtype()
    try:
        analyze.set_compute_dtype(np.float32)
        assert _concatenate_hidden_tokens(traces, layer=7).dtype == np.float32
    finally:
        analyze.set_compute_dtype(original)


def test_rejects_nonpositive_cap():
    with pytest.raises(ValueError):
        set_max_reference_tokens(0)
    with pytest.raises(ValueError):
        set_max_reference_tokens(-5)
