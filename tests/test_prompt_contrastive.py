import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from applications.prompt_decomposition import (
    CONTRASTIVE_METHODS,
    E2_PROBE_METHODS,
    add_crossfit_probe_scores,
    analyze_oof_scores,
    bootstrap_parseable_paired_deltas,
    fit_prompt_contrastive_direction,
    generate_oof_scores,
    region_indices,
    prompt_class_balanced_weights,
    score_localized_rmd,
    score_contrastive_trace,
    write_markdown,
)
from applications.prompt_selection import evaluate_prompt_selection


def _trace(prompt_id, trace_id, is_correct, predicted_answer="42"):
    return {
        "idx": prompt_id,
        "trace_id": trace_id,
        "sample_id": trace_id % 2,
        "is_correct": is_correct,
        "predicted_answer": predicted_answer,
        "gold_answer": "42",
        "entropies": np.asarray([0.1, 0.8, 0.2, 0.4]),
        "hiddens": {7: np.zeros((4, 2), dtype=float)},
        "mean_logprob": -0.5,
    }


def _mixed_groups():
    groups = {}
    projections = {}
    trace_id = 0
    for prompt_id in range(2):
        correct = _trace(prompt_id, trace_id, True)
        correct["hiddens"][7] = np.tile(
            np.asarray([[1.0, 0.0]]), (4, 1)
        )
        projections[trace_id] = np.tile(
            np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
            (1, 1),
        )
        trace_id += 1
        incorrect = _trace(prompt_id, trace_id, False)
        incorrect["hiddens"][7] = np.tile(
            np.asarray([[0.0, -1.0]]), (4, 1)
        )
        projections[trace_id] = np.tile(
            np.asarray([[0.0, -1.0], [0.0, -1.0], [0.0, -1.0], [0.0, -1.0]]),
            (1, 1),
        )
        trace_id += 1
        groups[prompt_id] = [correct, incorrect]
    return groups, projections


def test_region_indices_are_deterministic_and_keep_at_least_one_token():
    assert np.array_equal(
        region_indices(np.asarray([0.1, 0.8, 0.2, 0.4]), "full"),
        np.arange(4),
    )
    assert np.array_equal(
        region_indices(np.asarray([0.1, 0.8, 0.2, 0.4]), "high_entropy_q20"),
        np.asarray([1]),
    )
    assert np.array_equal(
        region_indices(np.asarray([0.1, 0.8, 0.2, 0.4]), "tail_q20"),
        np.asarray([3]),
    )
    assert np.array_equal(
        region_indices(np.asarray([0.5]), "high_entropy_q20"),
        np.asarray([0]),
    )


def test_region_indices_support_fixed_tail_sensitivity_windows():
    entropies = np.arange(10, dtype=float)

    assert np.array_equal(region_indices(entropies, "tail_q10"), np.asarray([9]))
    assert np.array_equal(
        region_indices(entropies, "tail_q50"), np.arange(5, 10)
    )
    with pytest.raises(ValueError, match="tail percentage"):
        region_indices(entropies, "tail_q0")
    with pytest.raises(ValueError, match="tail percentage"):
        region_indices(entropies, "tail_q101")


def test_random_region_is_exact_trace_specific_and_reproducible():
    entropies = np.linspace(0.0, 1.0, 21)

    first = region_indices(
        entropies, "random_q20", trace_id=17, region_seed=42
    )
    repeated = region_indices(
        entropies, "random_q20", trace_id=17, region_seed=42
    )
    other_trace = region_indices(
        entropies, "random_q20", trace_id=18, region_seed=42
    )

    assert len(first) == 5
    assert len(np.unique(first)) == 5
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, other_trace)
    assert np.array_equal(
        region_indices(
            np.asarray([0.5]), "random_q20", trace_id=17, region_seed=42
        ),
        np.asarray([0]),
    )


def test_localized_rmd_and_contrast_share_random_region_indices():
    entropies = np.linspace(0.0, 1.0, 10)
    projected = np.arange(20, dtype=float).reshape(10, 2)
    distances = np.arange(10, dtype=float)
    indices = region_indices(
        entropies, "random_q20", trace_id=9, region_seed=42
    )

    rmd_score = score_localized_rmd(
        distances,
        entropies,
        "random_q20",
        trace_id=9,
        region_seed=42,
    )
    contrast_score = score_contrastive_trace(
        projected,
        entropies,
        np.asarray([1.0, 0.0]),
        "random_q20",
        trace_id=9,
        region_seed=42,
    )

    assert rmd_score == pytest.approx(-distances[indices].mean())
    assert contrast_score == pytest.approx(projected[indices, 0].mean())


def test_region_indices_reject_invalid_entropy_sequences():
    with pytest.raises(ValueError, match="non-empty 1D"):
        region_indices(np.asarray([]), "full")
    with pytest.raises(ValueError, match="non-empty 1D"):
        region_indices(np.asarray([[0.1]]), "full")
    with pytest.raises(ValueError, match="unknown region"):
        region_indices(np.asarray([0.1]), "middle")


def test_prompt_direction_weights_prompts_equally_and_orients_correctly():
    groups, projections = _mixed_groups()
    fitted = fit_prompt_contrastive_direction(
        groups,
        sorted(groups),
        projections,
        region="full",
        seed=42,
        n_alignment_shuffles=20,
    )

    assert fitted["n_prompt_vectors"] == 2
    assert fitted["direction"][0] > 0
    assert fitted["direction"][1] > 0
    assert np.linalg.norm(fitted["direction"]) == pytest.approx(1.0)
    assert fitted["observed_alignment"] == pytest.approx(1.0)
    assert fitted["null"]["p_value"] <= 1.0

    correct_score = score_contrastive_trace(
        projections[0], groups[0][0]["entropies"], fitted["direction"], "full"
    )
    incorrect_score = score_contrastive_trace(
        projections[1], groups[0][1]["entropies"], fitted["direction"], "full"
    )
    assert correct_score > incorrect_score


def test_prompt_direction_excludes_unparsed_and_homogeneous_prompts():
    groups, projections = _mixed_groups()
    groups[0][1]["predicted_answer"] = None
    groups[1] = [groups[1][0]]

    fitted = fit_prompt_contrastive_direction(
        groups,
        sorted(groups),
        projections,
        region="full",
        n_alignment_shuffles=0,
    )

    assert fitted["n_prompt_vectors"] == 0
    assert fitted["skipped_prompts"] == {0: "no_parseable_incorrect", 1: "no_incorrect"}
    with pytest.raises(ValueError, match="no usable prompt-contrastive direction"):
        score_contrastive_trace(
            projections[0], groups[0][0]["entropies"], fitted["direction"], "full"
        )


def test_oof_runner_adds_all_contrastive_scores_without_changing_row_count():
    groups, _ = _mixed_groups()

    class FakePca:
        def transform(self, values):
            return np.asarray(values, dtype=float)

    def fake_fit(correct_traces, layer, pca_dim):
        return FakePca(), np.zeros(2), frozenset(
            int(trace["idx"]) for trace in correct_traces
        )

    def fake_extend(ref, background_traces, layer):
        return (*ref, "background")

    def fake_raw(hiddens, pca, mu, train_ids):
        return np.zeros(len(hiddens), dtype=float)

    def fake_rmd(hiddens, pca, mu, train_ids, background):
        return np.zeros(len(hiddens), dtype=float)

    rows = generate_oof_scores(
        groups,
        layers=[7],
        pca_dim=2,
        n_splits=2,
        seed=42,
        fit_reference=fake_fit,
        extend_reference=fake_extend,
        raw_distance=fake_raw,
        relative_distance=fake_rmd,
        contrastive_regions=(
            "full",
            "high_entropy_q20",
            "tail_q20",
            "random_q20",
        ),
        n_alignment_shuffles=3,
        region_seed=42,
    )

    assert len(rows) == 4
    assert all(
        f"{method}_score" in rows[0]
        for method in CONTRASTIVE_METHODS
    )
    assert all(
        np.isfinite(float(row[f"{method}_score"]))
        for row in rows
        for method in CONTRASTIVE_METHODS
    )
    assert all(
        np.isfinite(float(row[f"{method}_score"]))
        for row in rows
        for method in (
            "rmd_high_entropy_q20",
            "rmd_tail_q20",
            "rmd_random_q20",
        )
    )

    selection = evaluate_prompt_selection(
        rows, model="qwen", dataset="math500", n_bootstrap=0, seed=42
    )
    selectors = selection["layers"]["7"]["selectors"]
    assert "top1_contrast_full" in selectors
    assert "top1_contrast_high_entropy_q20" in selectors
    assert "top1_contrast_tail_q20" in selectors
    assert "top1_contrast_random_q20" in selectors
    assert "top1_rmd_high_entropy_q20" in selectors


def test_parseable_paired_bootstrap_accepts_unequal_prompt_sizes():
    rows = []
    trace_id = 0
    for prompt_id, labels in enumerate(([1, 0], [1, 1, 0], [1, 0, 0, 0])):
        for sample_id, label in enumerate(labels):
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "trace_id": trace_id,
                    "sample_id": sample_id,
                    "is_correct": label,
                    "contrast_full_score": 2.0 if label else -1.0,
                    "rmd_score": 1.0 if label else 0.0,
                    "logprob_score": 0.5 if label else 0.0,
                }
            )
            trace_id += 1

    result = bootstrap_parseable_paired_deltas(
        rows,
        methods=["contrast_full"],
        baselines=("rmd", "logprob"),
        n_bootstrap=20,
        seed=42,
    )

    assert set(result) == {
        "contrast_full_minus_rmd",
        "contrast_full_minus_logprob",
    }
    assert result["contrast_full_minus_rmd"]["within_prompt_macro"][
        "n_valid"
    ] == 20


def test_parseable_paired_bootstrap_accepts_explicit_score_pairs():
    rows = []
    for prompt_id in range(3):
        for sample_id, label in enumerate((1, 0)):
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "is_correct": label,
                    "rmd_high_entropy_q20_score": 2.0 if label else -1.0,
                    "rmd_score": 1.0 if label else 0.0,
                    "logprob_score": 0.5 if label else 0.0,
                }
            )

    result = bootstrap_parseable_paired_deltas(
        rows,
        methods=[],
        baselines=(),
        n_bootstrap=20,
        seed=42,
        pairs=(
            ("rmd_high_entropy_q20", "rmd"),
            ("rmd_high_entropy_q20", "logprob"),
        ),
    )

    assert set(result) == {
        "rmd_high_entropy_q20_minus_rmd",
        "rmd_high_entropy_q20_minus_logprob",
    }
    assert result["rmd_high_entropy_q20_minus_rmd"]["within_prompt_macro"][
        "n_valid"
    ] == 20


def _probe_rows():
    rows = []
    trace_id = 0
    for prompt_id in range(6):
        for sample_id, label in enumerate((1, 0)):
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "trace_id": trace_id,
                    "sample_id": sample_id,
                    "layer": 7,
                    "fold": prompt_id % 3,
                    "is_correct": label,
                    "predicted_answer": "A" if label else "B",
                    "logprob_score": 0.0,
                    "entropy_score": 0.0,
                    "length_score": 0.0,
                    "rmd_high_entropy_q20_score": 0.25 if label else -0.25,
                    "contrast_high_entropy_q20_score": 2.0 if label else -2.0,
                }
            )
            trace_id += 1
    return rows


def test_prompt_class_weights_equalize_prompts_and_classes():
    rows = _probe_rows()[:4]
    weights = prompt_class_balanced_weights(rows)

    for prompt_id in (0, 1):
        prompt_indices = [
            index for index, row in enumerate(rows) if row["prompt_id"] == prompt_id
        ]
        assert sum(weights[index] for index in prompt_indices) == pytest.approx(1.0)
        for label in (0, 1):
            class_indices = [
                index
                for index in prompt_indices
                if rows[index]["is_correct"] == label
            ]
            assert sum(weights[index] for index in class_indices) == pytest.approx(0.5)


def test_crossfit_probes_exclude_heldout_labels_and_unparsed_training_prompts():
    original = _probe_rows()
    original[-1]["predicted_answer"] = None
    changed = [dict(row) for row in original]
    for row in changed:
        if row["fold"] == 0:
            row["is_correct"] = 1 - row["is_correct"]

    original_diagnostics = add_crossfit_probe_scores(original)
    changed_diagnostics = add_crossfit_probe_scores(changed)

    probe_methods = (
        "probe_outputs",
        "probe_outputs_plus_rmd_high_entropy_q20",
        "probe_outputs_plus_contrast_high_entropy_q20",
    )
    for left, right in zip(original, changed):
        if left["fold"] == 0 and left["predicted_answer"] is not None:
            for method in probe_methods:
                assert left[f"{method}_score"] == pytest.approx(
                    right[f"{method}_score"]
                )
    assert all(
        original[-1][f"{method}_score"] is None for method in probe_methods
    )
    assert all(
        diagnostic["n_train_prompts"] < 4
        for diagnostic in original_diagnostics
        if diagnostic["fold"] != 2
    )


def test_crossfit_contrast_probe_recovers_complementary_signal():
    rows = _probe_rows()
    diagnostics = add_crossfit_probe_scores(rows)

    output_scores = np.asarray(
        [row["probe_outputs_score"] for row in rows], dtype=float
    )
    contrast_scores = np.asarray(
        [row["probe_outputs_plus_contrast_high_entropy_q20_score"] for row in rows],
        dtype=float,
    )
    labels = np.asarray([row["is_correct"] for row in rows], dtype=int)

    assert np.ptp(output_scores) == pytest.approx(0.0)
    assert contrast_scores[labels == 1].min() > contrast_scores[labels == 0].max()
    assert diagnostics
    assert all("coefficients" in diagnostic for diagnostic in diagnostics)


def test_same_token_output_autopsy_emits_four_nested_probe_scores():
    rows = _probe_rows()
    for row in rows:
        row.update(
            {
                "entropy_he_score": -0.2 if row["is_correct"] else -0.8,
                "logprob_he_score": -0.2 if row["is_correct"] else -0.8,
                "rmd_random_q20_score": 0.1 if row["is_correct"] else -0.1,
            }
        )

    diagnostics = add_crossfit_probe_scores(rows)

    assert set(E2_PROBE_METHODS) == {
        "probe_b0",
        "probe_b1",
        "probe_g_he",
        "probe_g_random",
    }
    assert all(row["probe_g_he_score"] is not None for row in rows)
    assert all(row["probe_g_random_score"] is not None for row in rows)
    assert all(
        set(diagnostic["features"]) <= {
            "length",
            "entropy",
            "logprob",
            "entropy_he",
            "logprob_he",
            "rmd_high_entropy_q20",
            "rmd_random_q20",
        }
        for diagnostic in diagnostics
        if diagnostic["method"] in E2_PROBE_METHODS
    )


def test_analysis_reports_prespecified_pairs_and_legacy_view(tmp_path: Path):
    rows = _probe_rows()
    for row in rows:
        label_score = 1.0 if row["is_correct"] else -1.0
        row.update(
            {
                "trace_length": 10,
                "raw_score": label_score,
                "rmd_score": label_score,
                "rmd_tail_q20_score": label_score,
                "rmd_random_q20_score": 0.5 * label_score,
                "contrast_full_score": label_score,
                "contrast_tail_q20_score": label_score,
                "contrast_random_q20_score": 0.5 * label_score,
            }
        )
    diagnostics = add_crossfit_probe_scores(rows)
    result = analyze_oof_scores(
        rows,
        config={
            "dataset": "math500",
            "model": "qwen",
            "layers": [7],
            "pca_dim": 2,
            "n": 2,
            "expected_prompts": 6,
            "n_splits": 3,
            "seed": 42,
            "data_report": {
                "partial_data": False,
                "observed_complete_prompts": 6,
            },
            "probe_diagnostics": diagnostics,
            "contrastive_regions": [
                "full",
                "high_entropy_q20",
                "tail_q20",
                "random_q20",
            ],
        },
        n_bootstrap=5,
        seed=42,
        max_new_tokens=10,
    )

    parseable = result["layers"]["7"]["parseable_only"]
    assert "probe_outputs" not in result["layers"]["7"]["methods"]
    assert "probe_outputs" in parseable["methods"]
    assert "rmd_high_entropy_q20_minus_rmd" in parseable["paired_score_deltas"]
    assert (
        "probe_outputs_plus_contrast_high_entropy_q20_minus_probe_outputs"
        in parseable["paired_score_deltas"]
    )
    assert "contrast_full_minus_rmd" in parseable["paired_contrastive_deltas"]

    report = tmp_path / "report.md"
    write_markdown(result, report)
    text = report.read_text()
    assert "Prespecified parseable score contrasts" in text
    assert "rmd_high_entropy_q20_minus_rmd" in text
