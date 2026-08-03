import pytest

from trace_caps import resolve_cap


def test_missing_cap_raises_instead_of_counting_nothing():
    with pytest.raises(ValueError, match="max_new_tokens is required"):
        resolve_cap(None, [100, 1024])


def test_cap_from_another_model_raises():
    # Qwen traces (budget 1024) handed DeepSeek's 8192: no trace can reach it.
    with pytest.raises(ValueError, match="exceeds every observed trace length"):
        resolve_cap(8192, [79, 500, 1024])


def test_binding_cap_is_returned():
    assert resolve_cap(1024, [79, 500, 1024]) == 1024


def test_none_lengths_are_ignored():
    assert resolve_cap(1024, [None, 1024, None]) == 1024


def test_no_observed_lengths_cannot_be_checked():
    assert resolve_cap(1024, []) == 1024
