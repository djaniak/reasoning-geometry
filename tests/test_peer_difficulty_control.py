import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from incremental_abstention import BASE_FEATURE_NAMES
from peer_difficulty_control import (
    CONTRASTS,
    METRICS,
    NEAR_ORACLE_SPEARMAN,
    PEER_PREFIX,
    PRE_DECLARED_CONTRAST,
    assert_shared_prompt_ids,
    holm_adjusted,
    near_oracle_flags,
    peer_pass_rates,
    prompt_golds,
    readout_specs,
    stop_rule_verdict,
)


def _row(prompt_id, is_correct, gold="7"):
    return {"prompt_id": prompt_id, "is_correct": is_correct, "gold_answer": gold}


def test_pass_rate_is_the_share_of_correct_siblings():
    rows = [_row(0, c) for c in (1, 1, 0, 0, 1, 0, 0, 0)]

    assert peer_pass_rates(rows)[0] == pytest.approx(3 / 8)


def test_an_unparsed_sibling_counts_as_not_solved_rather_than_as_missing():
    """`is_correct` is 0 for a trace with no extractable answer, and that is the
    right denominator: empirical difficulty asks whether the problem got solved."""
    rows = [_row(0, 1), _row(0, 0)]

    assert peer_pass_rates(rows)[0] == pytest.approx(0.5)


def test_a_prompt_with_no_usable_flags_is_missing_rather_than_all_wrong():
    rows = [_row(0, ""), _row(0, None)]

    assert np.isnan(peer_pass_rates(rows)[0])


def test_pass_rates_are_computed_per_prompt():
    rows = [_row(0, 1), _row(0, 1), _row(1, 0), _row(1, 1)]

    rates = peer_pass_rates(rows)

    assert rates[0] == pytest.approx(1.0)
    assert rates[1] == pytest.approx(0.5)


def test_alignment_check_accepts_ids_denoting_the_same_problem():
    golds = {
        "a": {0: "12", 1: "p-q"},
        "b": {0: "12", 1: "P - Q"},
    }

    assert assert_shared_prompt_ids(golds) == {0, 1}


def test_alignment_check_rejects_a_shifted_prompt_id():
    """A silent misalignment would look exactly like a control that does not work."""
    golds = {"a": {0: "12", 1: "13"}, "b": {0: "13", 1: "14"}}

    with pytest.raises(ValueError, match="same problem"):
        assert_shared_prompt_ids(golds)


def test_alignment_check_only_compares_ids_every_model_has():
    golds = {"a": {0: "12", 1: "13"}, "b": {0: "12"}}

    assert assert_shared_prompt_ids(golds) == {0}


def test_prompt_golds_takes_one_answer_per_prompt():
    rows = [_row(0, 1, "12"), _row(0, 0, "12"), _row(1, 1, "13")]

    assert prompt_golds(rows) == {0: "12", 1: "13"}


def test_every_readout_is_b0_plus_something():
    """The contrasts are only interpretable if the baseline is held fixed."""
    for spec in readout_specs(("peer_a", "peer_b")).values():
        assert spec[: len(BASE_FEATURE_NAMES)] == BASE_FEATURE_NAMES


def test_the_two_peer_readouts_differ_by_exactly_the_tail_feature():
    specs = readout_specs(("peer_a", "peer_b"))

    assert set(specs["B1_plus_peer"]) - set(specs["B0_plus_peer"]) == {"rmd_tail_q20"}


def test_the_peer_block_is_added_as_a_unit():
    """`B1_minus_B0_given_peer` means nothing if the two sides carry different controls."""
    specs = readout_specs(("peer_a", "peer_b"))

    assert set(specs["B0_plus_peer"]) - set(specs["B0"]) == {"peer_a", "peer_b"}
    assert set(specs["B1_plus_peer"]) - set(specs["B1"]) == {"peer_a", "peer_b"}


def test_contrasts_only_reference_defined_readouts():
    specs = readout_specs(("peer_a", "peer_b"))

    for left, right, _ in CONTRASTS:
        assert left in specs
        assert right in specs


def _model(label, low, high, *, metric="aurc", p=0.01, spearman=0.1):
    column = PEER_PREFIX + "other"
    return {
        "label": label,
        "peer_columns": [column],
        "populations": {
            "p": {
                "paired_deltas": {
                    f"{PRE_DECLARED_CONTRAST}_{metric}": {
                        "point_estimate": 0.5 * (low + high),
                        "ci_low": low,
                        "ci_high": high,
                        "p_two_sided": p,
                    }
                },
                "peer_association": {
                    f"{column}_vs_outcome": {"spearman": spearman}
                },
            }
        },
    }


def test_rule_fires_only_at_two_or_more_overlapping_models():
    one = [_model("a", -0.06, -0.02), _model("b", -0.05, -0.01), _model("c", -0.03, 0.01)]
    two = [_model("a", -0.06, -0.02), _model("b", -0.04, 0.02), _model("c", -0.03, 0.01)]

    assert not stop_rule_verdict(one, "p")["triggered"]
    verdict = stop_rule_verdict(two, "p")
    assert verdict["triggered"]
    assert verdict["models_with_interval_overlapping_zero"] == ["b", "c"]


def test_rule_ignores_a_population_a_model_does_not_have():
    results = [_model("a", -0.04, 0.02)]
    results[0]["populations"] = {}

    verdict = stop_rule_verdict(results, "p")

    assert verdict["models_with_interval_overlapping_zero"] == []
    assert not verdict["triggered"]


def test_the_rule_is_evaluated_on_aurc_not_the_secondary_metric():
    """AUACC is reported for continuity with the locked artifact, not for deciding."""
    results = [_model("a", -0.04, 0.02, metric="auacc")]

    assert stop_rule_verdict(results, "p")["models_with_interval_overlapping_zero"] == []
    assert METRICS[0] == "aurc"


def test_near_oracle_flags_only_fire_above_the_threshold():
    below = [_model("a", -0.06, -0.02, spearman=NEAR_ORACLE_SPEARMAN - 0.01)]
    above = [_model("a", -0.06, -0.02, spearman=NEAR_ORACLE_SPEARMAN + 0.01)]

    assert near_oracle_flags(below, "p") == {}
    assert list(near_oracle_flags(above, "p")) == ["a"]


def test_near_oracle_flags_are_symmetric_in_sign():
    """The control enters however the readout wants it; only its strength matters."""
    negative = [_model("a", -0.06, -0.02, spearman=-(NEAR_ORACLE_SPEARMAN + 0.01))]

    assert list(near_oracle_flags(negative, "p")) == ["a"]


def test_holm_is_step_down_and_monotone_over_the_three_model_family():
    results = [
        _model("a", -0.06, -0.02, p=0.01),
        _model("b", -0.05, -0.01, p=0.03),
        _model("c", -0.04, 0.00, p=0.04),
    ]

    holm = holm_adjusted(results, "p")

    assert holm["family_size"] == 3
    assert holm["tests"]["a"]["p_holm"] == pytest.approx(0.03)
    assert holm["tests"]["b"]["p_holm"] == pytest.approx(0.06)
    assert holm["tests"]["c"]["p_holm"] == pytest.approx(0.06)


def test_holm_never_lets_an_adjusted_p_fall_below_an_earlier_one():
    results = [_model("a", -0.06, -0.02, p=0.03), _model("b", -0.05, -0.01, p=0.031)]

    holm = holm_adjusted(results, "p")

    assert holm["tests"]["a"]["p_holm"] == pytest.approx(0.06)
    assert holm["tests"]["b"]["p_holm"] == pytest.approx(0.06)


def test_oracle_aurc_is_the_floor_a_perfect_ranker_reaches():
    """AURC does not bottom out at zero, and the floor rises as accuracy falls."""
    from peer_difficulty_control import oracle_aurc

    easy = np.array([1.0] * 9 + [0.0])
    hard = np.array([1.0] * 5 + [0.0] * 5)

    assert oracle_aurc(easy) > 0.0
    assert oracle_aurc(hard) > oracle_aurc(easy)


def test_a_perfectly_ranked_readout_has_no_headroom_left():
    from incremental_abstention import prompt_metrics
    from peer_difficulty_control import oracle_aurc

    outcomes = np.array([1.0, 1.0, 1.0, 0.0, 0.0])

    assert prompt_metrics(outcomes, outcomes)["aurc"] == pytest.approx(oracle_aurc(outcomes))


def test_share_of_headroom_is_undefined_rather_than_huge_when_none_is_left():
    """A readout at the floor has nothing to give up; any ratio there is denominator noise."""
    from peer_difficulty_control import _share

    assert _share(0.01, 0.04) == pytest.approx(0.25)
    assert np.isnan(_share(0.001, 0.0))
    assert np.isnan(_share(0.001, -0.002))
