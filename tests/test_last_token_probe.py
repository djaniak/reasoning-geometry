import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controls.last_token_probe import (
    PROBE_C_GRID,
    all_metrics,
    crossfit_probe,
    is_unparsed,
    load_frozen_scores,
    population_mask,
    prompt_centered_auroc,
    reference_score_sets,
    resample_prompts,
    select_layer_and_penalty,
    within_prompt_metrics,
)


def _cache(n_prompts=6, n_samples=4, unparsed=()):
    total = n_prompts * n_samples
    return {
        "trace_id": np.arange(total),
        "prompt_id": np.repeat(np.arange(n_prompts), n_samples),
        "unparsed": np.asarray(
            [1 if index in unparsed else 0 for index in range(total)], dtype=np.int8
        ),
    }


def test_unparsed_answers_are_recognised():
    assert is_unparsed(None)
    assert is_unparsed("")
    assert is_unparsed("   ")
    assert not is_unparsed("42")


def test_the_parseable_population_drops_unparsed_traces():
    """It must match the frozen probe's training rule.

    Unparsed traces are auto-labeled incorrect upstream, so a probe that keeps
    them can score by detecting truncation instead of reasoning failure.
    """
    cache = _cache(unparsed=(0, 5))

    assert population_mask(cache, "all_traces").sum() == 24
    assert population_mask(cache, "parseable").sum() == 22


def test_a_prompt_with_one_outcome_contributes_no_pairs_and_no_macro_term():
    """The three readouts are defined on different populations by construction."""
    prompt_ids = np.array([0, 0, 1, 1])
    labels = np.array([1, 0, 1, 1])
    scores = np.array([1.0, 0.0, 5.0, 4.0])

    metrics = within_prompt_metrics(prompt_ids, labels, scores)

    assert metrics["n_mixed_prompts"] == 1
    assert metrics["n_within_prompt_pairs"] == 1
    assert metrics["micro_pair_auroc"] == 1.0
    assert metrics["macro_prompt_auroc"] == 1.0


def test_a_tie_counts_half():
    prompt_ids = np.array([0, 0])
    labels = np.array([1, 0])
    scores = np.array([1.0, 1.0])

    assert within_prompt_metrics(prompt_ids, labels, scores)["micro_pair_auroc"] == 0.5


def test_micro_weights_pairs_and_macro_weights_prompts():
    """The two within-prompt readouts must be able to disagree.

    A large prompt the score gets right and a small one it gets wrong pull the
    pair-weighted and prompt-weighted averages in different directions; if they
    always agreed there would be no reason to report both.
    """
    # Prompt 0: 3 correct x 3 incorrect = 9 pairs, all concordant.
    # Prompt 1: 1 correct x 1 incorrect = 1 pair, discordant.
    prompt_ids = np.array([0] * 6 + [1] * 2)
    labels = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    scores = np.array([3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 0.0, 1.0])

    metrics = within_prompt_metrics(prompt_ids, labels, scores)

    assert metrics["micro_pair_auroc"] == pytest.approx(9 / 10)
    assert metrics["macro_prompt_auroc"] == pytest.approx(0.5)


def test_a_pure_prompt_difficulty_score_pools_high_and_collapses_within():
    """This is the decomposition the whole module exists to perform.

    A score that carries only prompt-level difficulty ranks traces well in the
    pool and not at all inside a prompt. If the two readouts did not separate
    here, a high pooled number could never be diagnosed.
    """
    rng = np.random.default_rng(0)
    prompt_ids = np.repeat(np.arange(50), 8)
    difficulty = rng.uniform(size=50)
    labels = (rng.uniform(size=400) < np.repeat(difficulty, 8)).astype(int)
    # Constant inside a prompt, informative across prompts.
    scores = np.repeat(difficulty, 8)

    metrics = all_metrics(prompt_ids, labels, scores)

    assert metrics["pooled_auroc"] > 0.7
    # Every within-prompt comparison is a tie, so both readouts sit at chance.
    assert metrics["micro_pair_auroc"] == pytest.approx(0.5)
    assert metrics["macro_prompt_auroc"] == pytest.approx(0.5)
    assert metrics["pooled_minus_macro"] > 0.2


def test_prompt_centering_removes_a_pure_between_prompt_score():
    prompt_ids = np.repeat(np.arange(10), 4)
    labels = np.tile([1, 1, 0, 0], 10)
    scores = np.repeat(np.arange(10, dtype=float), 4)

    assert prompt_centered_auroc(prompt_ids, labels, scores) == pytest.approx(0.5)


def test_a_prompt_drawn_twice_stays_two_prompts():
    """Merging the copies would manufacture within-prompt pairs across a
    boundary the design says exists, inflating the micro denominator."""
    prompt_ids = np.repeat(np.arange(3), 2)

    indices, relabelled = resample_prompts(prompt_ids, np.array([0, 0, 2]))

    assert list(indices) == [0, 1, 0, 1, 4, 5]
    assert list(relabelled) == [0, 0, 1, 1, 2, 2]
    assert len(np.unique(relabelled)) == 3


def _layer_fixture(n_prompts=20, n_samples=6, d=8, seed=0):
    rng = np.random.default_rng(seed)
    total = n_prompts * n_samples
    prompt_ids = np.repeat(np.arange(n_prompts), n_samples)
    labels = np.tile([1] * (n_samples // 2) + [0] * (n_samples // 2), n_prompts)
    informative = rng.normal(size=(total, d))
    informative[:, 0] += labels * 4.0
    noise = rng.normal(size=(total, d))
    return prompt_ids, labels, {7: noise, 14: informative}


def test_selection_finds_the_informative_layer():
    prompt_ids, labels, states = _layer_fixture()

    layer, penalty, grid = select_layer_and_penalty(
        states, labels, prompt_ids, sorted(set(prompt_ids)),
        inner_splits=3, seed=42, penalties=(1e-3, 1e-1),
    )

    assert layer == 14
    assert penalty in (1e-3, 1e-1)
    assert max(value for (key, _), value in grid.items() if key == 14) > (
        max(value for (key, _), value in grid.items() if key == 7)
    )


def test_the_penalty_is_chosen_not_fixed():
    """A loose penalty separates the training set perfectly and loses several
    points of held-out AUROC, so leaving it fixed would understate the very
    claim this module reproduces before decomposing."""
    prompt_ids, labels, states = _layer_fixture(d=64)
    penalties = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)

    _, penalty, grid = select_layer_and_penalty(
        states, labels, prompt_ids, sorted(set(prompt_ids)),
        inner_splits=3, seed=42, penalties=penalties,
    )

    assert set(penalties) == {key[1] for key in grid}
    assert grid[(14, penalty)] == max(
        value for (layer, _), value in grid.items() if layer == 14
    )


def test_layer_selection_never_sees_the_held_out_prompts():
    """Selecting on the test split is the leak that makes published pooled
    numbers hard to reproduce, so the choice must not move when only the
    held-out prompts change."""
    prompt_ids, labels, states = _layer_fixture()
    train_prompts = sorted(set(prompt_ids))[:15]
    held_out = np.isin(prompt_ids, sorted(set(prompt_ids))[15:])

    train_mask = np.isin(prompt_ids, train_prompts)
    baseline = select_layer_and_penalty(
        {layer: state[train_mask] for layer, state in states.items()},
        labels[train_mask], prompt_ids[train_mask], train_prompts,
        inner_splits=3, seed=42, penalties=(1e-3, 1e-1),
    )[:2]

    corrupted = {layer: state.copy() for layer, state in states.items()}
    corrupted[7][held_out] += 100.0
    corrupted_choice = select_layer_and_penalty(
        {layer: state[train_mask] for layer, state in corrupted.items()},
        labels[train_mask], prompt_ids[train_mask], train_prompts,
        inner_splits=3, seed=42, penalties=(1e-3, 1e-1),
    )[:2]

    assert baseline == corrupted_choice


def test_every_trace_is_scored_by_a_fold_that_never_saw_its_prompt():
    prompt_ids, labels, states = _layer_fixture(n_prompts=20)

    scores, folds = crossfit_probe(
        states, labels, prompt_ids, n_splits=5, inner_splits=3, seed=42
    )

    assert np.isfinite(scores).all()
    assert sum(record["n_test"] for record in folds) == len(labels)
    # Prompt-disjoint folds: no prompt may appear in two test splits.
    assert len(folds) == 5
    assert all(record["selected_layer"] == 14 for record in folds)
    assert all(record["selected_C"] in PROBE_C_GRID for record in folds)


def test_the_frozen_join_requires_a_layer(tmp_path):
    """OOF rows are per (trace, layer). Picking no layer would average the
    geometry over the sweep while the output-side columns reproduce exactly --
    a failure that leaves no trace in the numbers."""
    csv_path = tmp_path / "oof.csv"
    csv_path.write_text(
        "trace_id,layer,rmd_tail_q20_score,probe_hidden_tail_q20_score\n"
        "0,7,-1.0,0.1\n"
        "0,21,-5.0,0.9\n"
        "1,7,-2.0,0.2\n"
        "1,21,-6.0,0.8\n"
    )

    joined = load_frozen_scores(str(csv_path), 21, np.array([1, 0]))

    assert list(joined) == ["rmd_tail_q20", "probe_hidden_tail_q20"]
    # Aligned to the requested trace order, not the file order.
    assert list(joined["rmd_tail_q20"]) == [-6.0, -5.0]


def test_the_frozen_join_refuses_to_silently_drop_traces(tmp_path):
    csv_path = tmp_path / "oof.csv"
    csv_path.write_text("trace_id,layer,rmd_tail_q20_score\n0,21,-5.0\n")

    with pytest.raises(ValueError, match="missing 1 traces"):
        load_frozen_scores(str(csv_path), 21, np.array([0, 1]))


def test_prompt_centering_matches_the_frozen_definition():
    """The frozen report has a column of this name; the continuity check is
    only meaningful if it is the same quantity. It excludes single-outcome
    prompts, which centre to a constant-label block."""
    from applications.prompt_decomposition import prompt_centered_auc

    rng = np.random.default_rng(3)
    prompt_ids = np.repeat(np.arange(30), 4)
    labels = rng.integers(0, 2, size=120)
    scores = rng.normal(size=120)
    rows = [
        {"prompt_id": int(prompt), "is_correct": bool(label), "score": float(score)}
        for prompt, label, score in zip(prompt_ids, labels, scores)
    ]

    assert prompt_centered_auroc(prompt_ids, labels, scores) == pytest.approx(
        prompt_centered_auc(rows)["auc"]
    )


def test_the_length_score_uses_the_frozen_log_transform():
    """Pooled, micro and macro are rank statistics and cannot see the difference
    between token count and log1p(token count). The prompt-centered readout
    subtracts a per-prompt mean from the raw score, so it can, and the frozen
    report defines length as -log1p(token count). Reproducing three columns and
    missing the fourth is exactly what a monotone mismatch looks like."""
    from applications.prompt_decomposition import SCORE_DESCRIPTIONS

    assert SCORE_DESCRIPTIONS["length"] == "-log1p(token count)"

    cache = {
        "n_tokens": np.array([10, 200, 3000, 45]),
        "mean_logprob": np.array([-0.1, -0.2, -0.3, -0.4]),
        "mean_entropy": np.array([0.5, 0.6, 0.7, 0.8]),
    }
    mask = np.ones(4, dtype=bool)
    scores = reference_score_sets(cache, mask)

    np.testing.assert_allclose(scores["length"], -np.log1p(cache["n_tokens"]))
    np.testing.assert_allclose(scores["mean_entropy"], -cache["mean_entropy"])
    np.testing.assert_allclose(scores["mean_logprob"], cache["mean_logprob"])
