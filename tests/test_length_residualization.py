import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from applications.wave1_experiments import (
    length_residualized_abstention,
    rank_residualize,
)


def _scores(n=40, seed=0):
    """A length proxy, a length-independent oracle, and pure noise.

    `length` orders prompts; `proxy` is a monotone function of it, so nothing
    should survive residualization. `oracle` ranks by outcome and is built to be
    rank-uncorrelated with length, so all of it should survive.
    """
    rng = np.random.default_rng(seed)
    outcomes = {}
    prompt_scores = {}
    for prompt_id in range(n):
        length = float(prompt_id)
        # Outcome alternates so it is orthogonal to the length ordering.
        outcome = float(prompt_id % 2)
        outcomes[prompt_id] = outcome
        prompt_scores[prompt_id] = {
            "length": length,
            "proxy": float(np.log1p(length)),
            "oracle": outcome + 1e-3 * rng.normal(),
            "noise": float(rng.normal()),
        }
    return prompt_scores, outcomes


def test_residual_is_rank_uncorrelated_with_the_control():
    prompt_scores, _ = _scores()
    prompt_ids = sorted(prompt_scores)
    residual = rank_residualize(
        prompt_scores, prompt_ids, methods=("proxy", "oracle", "noise"), control="length"
    )
    control = [prompt_scores[p]["length"] for p in prompt_ids]
    # `proxy` is a strictly monotone transform of the control, so its ranks match
    # exactly and the residual collapses to a constant -- total removal, which
    # Spearman cannot express (it is undefined on a constant input).
    proxy = np.asarray([residual[p]["proxy"] for p in prompt_ids])
    assert np.allclose(proxy, 0.0, atol=1e-12)
    for method in ("oracle", "noise"):
        values = [residual[p][method] for p in prompt_ids]
        assert abs(float(spearmanr(control, values).statistic)) < 0.05


def test_a_pure_length_proxy_loses_its_abstention_signal():
    prompt_scores, outcomes = _scores()
    result = length_residualized_abstention(
        prompt_scores,
        outcomes,
        methods=("length", "proxy", "oracle"),
        n_bootstrap=200,
        seed=1,
    )
    # The control itself is dropped: residualizing it on itself is degenerate.
    assert "length" not in result["point"]
    proxy = result["vs_uninformative"]["proxy"]["aurc"]
    assert abs(proxy["point_estimate"]) < 0.05
    assert proxy["ci_low"] < 0.0 < proxy["ci_high"]


def test_a_length_independent_scorer_survives_residualization():
    prompt_scores, outcomes = _scores()
    result = length_residualized_abstention(
        prompt_scores,
        outcomes,
        methods=("length", "proxy", "oracle"),
        n_bootstrap=200,
        seed=1,
    )
    oracle = result["vs_uninformative"]["oracle"]["aurc"]
    assert oracle["point_estimate"] > 0.1
    assert oracle["ci_low"] > 0.0
    assert result["base_accuracy"] == pytest.approx(0.5)


def test_head_to_head_deltas_are_reported_for_requested_pairs():
    prompt_scores, outcomes = _scores()
    result = length_residualized_abstention(
        prompt_scores,
        outcomes,
        methods=("length", "proxy", "oracle"),
        n_bootstrap=200,
        seed=1,
        comparisons=(("oracle", "proxy"),),
    )
    delta = result["deltas"]["oracle_minus_proxy"]["aurc"]
    assert delta["point_estimate"] > 0.1
    assert delta["ci_low"] > 0.0
    assert "proxy_minus_oracle" not in result["deltas"]


def test_non_finite_scores_keep_the_sentinel_and_sort_last():
    prompt_scores, _ = _scores(n=10)
    prompt_scores[3]["oracle"] = -np.inf
    prompt_ids = sorted(prompt_scores)
    residual = rank_residualize(
        prompt_scores, prompt_ids, methods=("oracle",), control="length"
    )
    assert residual[3]["oracle"] == -np.inf
    assert all(np.isfinite(residual[p]["oracle"]) for p in prompt_ids if p != 3)


def test_too_few_usable_prompts_is_reported_not_raised():
    prompt_scores = {
        prompt_id: {"length": float(prompt_id), "oracle": -np.inf}
        for prompt_id in range(5)
    }
    prompt_scores[0]["oracle"] = 1.0
    residual = rank_residualize(
        prompt_scores, sorted(prompt_scores), methods=("oracle",), control="length"
    )
    assert all(value["oracle"] == -np.inf for value in residual.values())
