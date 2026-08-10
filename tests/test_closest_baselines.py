import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closest_baselines import (
    CONTRASTS,
    EXTRA_FEATURE_NAMES,
    READOUT_SPECS,
    STOP_RULE_1A_CONTRAST,
    TAIL_VERSUS_FULL_CONTRAST,
    answer_entropy,
    extra_prompt_columns,
    holm_adjusted,
    stop_rule_verdicts,
    tail_window_sizes,
    window_strata,
)
from incremental_abstention import BASE_FEATURE_NAMES


def _row(prompt_id, answer, rmd):
    return {
        "prompt_id": prompt_id,
        "predicted_answer": answer,
        "rmd_score": rmd,
    }


def test_entropy_separates_two_vote_shares_that_agreement_calls_identical():
    """The whole point of 1a: `vote_agreement` is 0.625 for both of these."""
    concentrated = [_row(0, a, 0.0) for a in "aaaaabbb"]
    scattered = [_row(0, a, 0.0) for a in "aaaaabcd"]

    assert answer_entropy(concentrated) < answer_entropy(scattered)


def test_unanimous_answers_have_zero_entropy():
    assert answer_entropy([_row(0, "7", 0.0) for _ in range(8)]) == pytest.approx(0.0)


def test_uniform_answers_reach_log_n():
    rows = [_row(0, str(index), 0.0) for index in range(4)]

    assert answer_entropy(rows) == pytest.approx(math.log(4))


def test_a_prompt_with_nothing_parseable_is_missing_rather_than_unanimous():
    """Zero entropy means one cluster -- the opposite state from no clusters at all."""
    rows = [_row(0, "", 0.0), _row(0, None, 0.0)]

    assert math.isnan(answer_entropy(rows))


def test_unparsed_siblings_do_not_dilute_the_histogram():
    """The denominator matches `vote_agreement`: parseable siblings only."""
    rows = [_row(0, "7", 0.0), _row(0, "7", 0.0), _row(0, "", 0.0)]

    assert answer_entropy(rows) == pytest.approx(0.0)


def test_extra_columns_are_negated_like_every_other_frozen_score():
    """Higher is better across the design matrix, so entropy enters negated."""
    rows = [_row(0, str(index), 0.0) for index in range(4)]

    assert extra_prompt_columns(rows)[0]["neg_answer_entropy"] == pytest.approx(
        -math.log(4)
    )


def test_rmd_full_is_the_sibling_mean_of_the_whole_trace_score():
    rows = [_row(0, "1", -2.0), _row(0, "2", -4.0), _row(1, "3", -1.0)]

    columns = extra_prompt_columns(rows)

    assert columns[0]["rmd_full"] == pytest.approx(-3.0)
    assert columns[1]["rmd_full"] == pytest.approx(-1.0)


def test_every_readout_is_b0_plus_something():
    """The contrasts are only interpretable if the baseline is held fixed."""
    for spec in READOUT_SPECS.values():
        assert spec[: len(BASE_FEATURE_NAMES)] == BASE_FEATURE_NAMES


def test_contrasts_only_reference_defined_readouts():
    for left, right, _ in CONTRASTS:
        assert left in READOUT_SPECS
        assert right in READOUT_SPECS


def test_the_two_added_features_are_the_only_new_columns():
    used = {column for spec in READOUT_SPECS.values() for column in spec}

    assert used - set(BASE_FEATURE_NAMES) - {"rmd_tail_q20"} == set(EXTRA_FEATURE_NAMES)


def _model(label, contrast, low, high):
    return {
        "label": label,
        "populations": {
            "p": {
                "paired_deltas_aurc": {
                    contrast: {"point_estimate": 0.5 * (low + high), "ci_low": low, "ci_high": high}
                }
            }
        },
    }


def test_stop_rule_1a_fires_only_at_two_or_more_overlapping_models():
    contrast = STOP_RULE_1A_CONTRAST
    one_overlap = [
        _model("a", contrast, -0.06, -0.02),
        _model("b", contrast, -0.05, -0.01),
        _model("c", contrast, -0.03, 0.01),
    ]
    two_overlap = [
        _model("a", contrast, -0.06, -0.02),
        _model("b", contrast, -0.04, 0.02),
        _model("c", contrast, -0.03, 0.01),
    ]

    assert not stop_rule_verdicts(one_overlap, "p")["1a"]["triggered"]
    verdict = stop_rule_verdicts(two_overlap, "p")["1a"]
    assert verdict["triggered"]
    assert verdict["models_with_interval_overlapping_zero"] == ["b", "c"]


def test_stop_rule_ignores_a_population_a_model_does_not_have():
    results = [_model("a", STOP_RULE_1A_CONTRAST, -0.04, 0.02)]
    results[0]["populations"] = {}

    verdict = stop_rule_verdicts(results, "p")["1a"]

    assert verdict["models_with_interval_overlapping_zero"] == []
    assert not verdict["triggered"]


def test_the_tail_window_is_a_fifth_of_each_trace_rounded_up():
    """"The final 20%" is a different number of tokens at different trace lengths."""
    rows = [
        {"prompt_id": 0, "trace_length": 100},
        {"prompt_id": 0, "trace_length": 401},
    ]

    # ceil(20) and ceil(80.2) -> 21, averaged over the two siblings.
    assert tail_window_sizes(rows)[0] == pytest.approx((20 + 81) / 2)


def test_window_strata_partition_the_terciles_and_the_median_split_separately():
    windows = np.arange(1.0, 10.0)

    strata = window_strata(windows)

    terciles = ["window_short", "window_mid", "window_long"]
    assert sum(strata[name].sum() for name in terciles) == len(windows)
    for left in terciles:
        for right in terciles:
            if left != right:
                assert not (strata[left] & strata[right]).any()
    halves = ["window_below_median", "window_above_median"]
    assert sum(strata[name].sum() for name in halves) == len(windows)
    assert not (strata[halves[0]] & strata[halves[1]]).any()


def test_an_absolute_threshold_adds_a_stratum_without_disturbing_the_others():
    windows = np.arange(1.0, 10.0)

    strata = window_strata(windows, absolute_threshold=4.0)

    assert strata["window_le_4"].sum() == 4
    assert window_strata(windows).keys() < strata.keys()


def test_holm_is_step_down_and_monotone_over_the_declared_family():
    results = [
        _model("a", STOP_RULE_1A_CONTRAST, -0.06, -0.02),
        _model("b", STOP_RULE_1A_CONTRAST, -0.05, -0.01),
    ]
    for body, p in zip(results, (0.01, 0.04)):
        body["populations"]["p"]["paired_deltas_aurc"][STOP_RULE_1A_CONTRAST][
            "p_two_sided"
        ] = p

    holm = holm_adjusted(results, "p")

    assert holm["family_size"] == 2
    # Smallest p is multiplied by the family size, the next by one fewer, and the
    # running maximum keeps the sequence non-decreasing.
    assert holm["tests"]["a:" + STOP_RULE_1A_CONTRAST]["p_holm"] == pytest.approx(0.02)
    assert holm["tests"]["b:" + STOP_RULE_1A_CONTRAST]["p_holm"] == pytest.approx(0.04)


def test_holm_never_lets_an_adjusted_p_fall_below_an_earlier_one():
    """Without the running maximum a later test can report a smaller adjusted p."""
    results = [
        _model("a", STOP_RULE_1A_CONTRAST, -0.06, -0.02),
        _model("b", STOP_RULE_1A_CONTRAST, -0.05, -0.01),
    ]
    for body, p in zip(results, (0.03, 0.031)):
        body["populations"]["p"]["paired_deltas_aurc"][STOP_RULE_1A_CONTRAST][
            "p_two_sided"
        ] = p

    holm = holm_adjusted(results, "p")

    assert holm["tests"]["a:" + STOP_RULE_1A_CONTRAST]["p_holm"] == pytest.approx(0.06)
    assert holm["tests"]["b:" + STOP_RULE_1A_CONTRAST]["p_holm"] == pytest.approx(0.06)


def test_1b_reports_a_branch_per_model_and_never_a_trigger():
    """1b forbids a follow-up sweep either way, so a 'triggered' flag would misread it."""
    contrast = TAIL_VERSUS_FULL_CONTRAST
    results = [
        _model("wins", contrast, -0.09, -0.03),
        _model("ties", contrast, -0.016, 0.007),
    ]

    verdict = stop_rule_verdicts(results, "p")["1b"]

    assert verdict["branch_by_model"] == {"wins": "tail_wins", "ties": "tie_or_full_wins"}
    assert verdict["n_tail_wins"] == 1
    assert "triggered" not in verdict
