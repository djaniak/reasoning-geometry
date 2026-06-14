"""Within-prompt versus between-prompt decomposition for Best-of-N traces."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm

from analyze import (
    compute_mahal_distances,
    compute_relative_mahal_distances,
    detect_layers,
    extend_reference_with_background_safe,
    fit_mahalanobis_reference_safe,
    load_all_traces,
)
from best_of_n import group_traces_by_problem


SCALAR_METRICS = (
    "pooled_auc",
    "prompt_centered_auc",
    "within_prompt_macro",
    "within_prompt_pair_weighted",
    "score_icc",
    "prompt_score_pass_rate_spearman",
    "prompt_score_pass_rate_pearson",
)

SCORE_METHODS = (
    "entropy",
    "logprob",
    "length",
    "activation_norm",
    "centroid",
    "raw",
    "rmd",
)

SCORE_DESCRIPTIONS = {
    "entropy": "-mean(token entropy)",
    "logprob": "mean token log-probability",
    "length": "-log1p(token count)",
    "activation_norm": "-mean(token hidden-state L2 norm)",
    "centroid": "-mean Euclidean distance to the correct-trace PCA centroid",
    "raw": "-mean(raw Mahalanobis token distance)",
    "rmd": "-mean(relative Mahalanobis token distance)",
}


def _status(message: str) -> None:
    print(f"[prompt-decomposition] {message}", file=sys.stderr, flush=True)


def parse_int_list(raw: str | None) -> list[int]:
    if not raw:
        return []
    return sorted({int(part.strip()) for part in raw.split(",") if part.strip()})


def _group_rows(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["prompt_id"])].append(row)
    return dict(grouped)


def _safe_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def available_score_methods(rows: list[dict]) -> list[str]:
    methods = []
    for method in SCORE_METHODS:
        key = f"{method}_score"
        values = [row.get(key) for row in rows]
        if values and all(
            value is not None and math.isfinite(float(value)) for value in values
        ):
            methods.append(method)
    return methods


def within_prompt_concordance(
    rows: list[dict], score_key: str = "score"
) -> dict:
    per_prompt = []
    concordant_total = 0.0
    pair_total = 0

    for prompt_id, group in sorted(_group_rows(rows).items()):
        correct = [float(row[score_key]) for row in group if row["is_correct"]]
        incorrect = [float(row[score_key]) for row in group if not row["is_correct"]]
        if not correct or not incorrect:
            continue

        concordant = 0.0
        for correct_score in correct:
            for incorrect_score in incorrect:
                if correct_score > incorrect_score:
                    concordant += 1.0
                elif correct_score == incorrect_score:
                    concordant += 0.5
        n_pairs = len(correct) * len(incorrect)
        value = concordant / n_pairs
        per_prompt.append(
            {
                "prompt_id": int(prompt_id),
                "concordance": float(value),
                "n_pairs": int(n_pairs),
            }
        )
        concordant_total += concordant
        pair_total += n_pairs

    return {
        "macro": float(np.mean([item["concordance"] for item in per_prompt]))
        if per_prompt
        else None,
        "pair_weighted": float(concordant_total / pair_total)
        if pair_total
        else None,
        "n_mixed_prompts": len(per_prompt),
        "n_pairs": int(pair_total),
        "per_prompt": per_prompt,
    }


def prompt_centered_auc(rows: list[dict], score_key: str = "score") -> dict:
    labels = []
    centered_scores = []
    mixed_prompts = 0

    for group in _group_rows(rows).values():
        group_labels = [int(row["is_correct"]) for row in group]
        if len(set(group_labels)) < 2:
            continue
        mixed_prompts += 1
        mean_score = float(np.mean([float(row[score_key]) for row in group]))
        labels.extend(group_labels)
        centered_scores.extend(float(row[score_key]) - mean_score for row in group)

    return {
        "auc": _safe_auc(labels, centered_scores),
        "n_mixed_prompts": int(mixed_prompts),
        "n_traces": len(labels),
    }


def score_icc(rows: list[dict], score_key: str = "score") -> dict:
    groups = list(_group_rows(rows).values())
    if len(groups) < 2:
        return {
            "icc": None,
            "ms_between": None,
            "ms_within": None,
            "n_prompts": len(groups),
            "group_size": None,
        }

    group_sizes = {len(group) for group in groups}
    if len(group_sizes) != 1:
        raise ValueError("ICC requires the same number of traces for every prompt")
    k = group_sizes.pop()
    if k < 2:
        raise ValueError("ICC requires at least two traces per prompt")

    values = np.array(
        [[float(row[score_key]) for row in group] for group in groups],
        dtype=float,
    )
    grand_mean = float(values.mean())
    prompt_means = values.mean(axis=1)
    ms_between = float(k * np.sum((prompt_means - grand_mean) ** 2) / (len(groups) - 1))
    ms_within = float(np.sum((values - prompt_means[:, None]) ** 2) / (len(groups) * (k - 1)))
    denominator = ms_between + (k - 1) * ms_within
    icc = (ms_between - ms_within) / denominator if denominator > 0 else None

    return {
        "icc": float(icc) if icc is not None else None,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "n_prompts": len(groups),
        "group_size": int(k),
    }


def _finite_correlation(result) -> float | None:
    value = float(result.statistic)
    return value if math.isfinite(value) else None


def prompt_score_pass_rate_correlation(
    rows: list[dict], score_key: str = "score"
) -> dict:
    prompt_means = []
    pass_rates = []
    for group in _group_rows(rows).values():
        prompt_means.append(float(np.mean([float(row[score_key]) for row in group])))
        pass_rates.append(float(np.mean([int(row["is_correct"]) for row in group])))

    if (
        len(prompt_means) < 2
        or np.ptp(prompt_means) == 0
        or np.ptp(pass_rates) == 0
    ):
        spearman = pearson = None
    else:
        spearman = _finite_correlation(spearmanr(prompt_means, pass_rates))
        pearson = _finite_correlation(pearsonr(prompt_means, pass_rates))

    return {
        "spearman": spearman,
        "pearson": pearson,
        "n_prompts": len(prompt_means),
    }


def compute_scalar_metrics(
    rows: list[dict], score_key: str = "score"
) -> dict:
    labels = [int(row["is_correct"]) for row in rows]
    scores = [float(row[score_key]) for row in rows]
    centered = prompt_centered_auc(rows, score_key=score_key)
    concordance = within_prompt_concordance(rows, score_key=score_key)
    icc = score_icc(rows, score_key=score_key)
    correlation = prompt_score_pass_rate_correlation(rows, score_key=score_key)

    return {
        "pooled_auc": _safe_auc(labels, scores),
        "prompt_centered_auc": centered["auc"],
        "within_prompt_macro": concordance["macro"],
        "within_prompt_pair_weighted": concordance["pair_weighted"],
        "score_icc": icc["icc"],
        "prompt_score_pass_rate_spearman": correlation["spearman"],
        "prompt_score_pass_rate_pearson": correlation["pearson"],
        "counts": {
            "n_traces": len(rows),
            "n_prompts": len(_group_rows(rows)),
            "n_mixed_prompts": centered["n_mixed_prompts"],
            "n_within_prompt_pairs": concordance["n_pairs"],
        },
        "icc_components": {
            "ms_between": icc["ms_between"],
            "ms_within": icc["ms_within"],
            "group_size": icc["group_size"],
        },
    }


def validate_groups(
    groups: dict[int, list[dict]],
    expected_prompts: int,
    n: int,
    allow_partial: bool = False,
) -> tuple[dict[int, list[dict]], dict]:
    invalid_sample_ids = sorted(
        int(idx)
        for idx, group in groups.items()
        if len(group) == n
        and sorted(int(trace.get("sample_id", 0)) for trace in group)
        != list(range(n))
    )
    trace_id_to_prompts: dict[int, set[int]] = defaultdict(set)
    trace_id_counts: dict[int, int] = defaultdict(int)
    for idx, group in groups.items():
        for trace in group:
            trace_id = int(trace["trace_id"])
            trace_id_to_prompts[trace_id].add(int(idx))
            trace_id_counts[trace_id] += 1
    duplicate_trace_ids = sorted(
        trace_id for trace_id, count in trace_id_counts.items() if count > 1
    )
    duplicate_trace_prompts = {
        prompt_id
        for trace_id in duplicate_trace_ids
        for prompt_id in trace_id_to_prompts[trace_id]
    }
    excluded = set(invalid_sample_ids) | duplicate_trace_prompts
    complete = {
        int(idx): group
        for idx, group in groups.items()
        if len(group) == n and int(idx) not in excluded
    }
    incomplete = sorted(int(idx) for idx, group in groups.items() if len(group) != n)
    expected_ids = set(range(expected_prompts))
    observed_ids = set(int(idx) for idx in groups)
    missing_ids = sorted(expected_ids - observed_ids)
    unexpected_ids = sorted(observed_ids - expected_ids)

    is_complete = (
        len(complete) == expected_prompts
        and not incomplete
        and not invalid_sample_ids
        and not duplicate_trace_ids
        and not missing_ids
        and not unexpected_ids
    )
    if not is_complete and not allow_partial:
        raise ValueError(
            f"expected {expected_prompts} complete prompts with {n} traces each; "
            f"found {len(complete)} complete prompts, {len(incomplete)} incomplete, "
            f"{len(invalid_sample_ids)} with invalid sample IDs, "
            f"{len(duplicate_trace_ids)} duplicate trace IDs, "
            f"{len(missing_ids)} missing, and {len(unexpected_ids)} unexpected"
        )
    if len(complete) < 2:
        raise ValueError("Need at least two complete prompts for decomposition")

    report = {
        "partial_data": not is_complete,
        "expected_prompts": int(expected_prompts),
        "observed_prompts": len(groups),
        "observed_complete_prompts": len(complete),
        "n": int(n),
        "excluded_prompt_ids": sorted(set(incomplete) | excluded),
        "invalid_sample_id_prompt_ids": invalid_sample_ids,
        "duplicate_trace_ids": duplicate_trace_ids,
        "missing_prompt_ids": missing_ids,
        "unexpected_prompt_ids": unexpected_ids,
    }
    return dict(sorted(complete.items())), report


def make_prompt_folds(
    prompt_ids: list[int], n_splits: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    if len(prompt_ids) < 2:
        raise ValueError("Need at least two prompts")
    n_splits = min(int(n_splits), len(prompt_ids))
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")

    ids = list(prompt_ids)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (
            [ids[index] for index in train_indices],
            [ids[index] for index in test_indices],
        )
        for train_indices, test_indices in splitter.split(ids)
    ]


def _flatten_groups(
    groups: dict[int, list[dict]], prompt_ids: list[int]
) -> list[dict]:
    return [trace for prompt_id in prompt_ids for trace in groups[prompt_id]]


def generate_oof_scores(
    groups: dict[int, list[dict]],
    layers: list[int],
    pca_dim: int,
    n_splits: int,
    seed: int,
    fit_reference: Callable = fit_mahalanobis_reference_safe,
    extend_reference: Callable = extend_reference_with_background_safe,
    raw_distance: Callable = compute_mahal_distances,
    relative_distance: Callable = compute_relative_mahal_distances,
    show_progress: bool = False,
) -> list[dict]:
    rows = []
    prompt_ids = sorted(groups)
    folds = make_prompt_folds(prompt_ids, n_splits=n_splits, seed=seed)

    with tqdm(
        total=len(folds) * len(layers),
        desc="OOF scoring",
        unit="fold-layer",
        disable=not show_progress,
        dynamic_ncols=True,
    ) as progress:
        for fold_index, (train_ids, test_ids) in enumerate(folds):
            train_traces = _flatten_groups(groups, train_ids)
            correct_train = [trace for trace in train_traces if trace["is_correct"]]

            for layer in layers:
                context = f"fold {fold_index + 1}/{len(folds)}, layer {layer}"
                progress.set_description_str(
                    f"OOF {context}: fitting correct reference", refresh=True
                )
                ref = fit_reference(correct_train, layer, pca_dim)
                if ref is None:
                    raise RuntimeError(
                        f"Could not fit Mahalanobis reference for fold {fold_index}, layer {layer}"
                    )
                progress.set_description_str(
                    f"OOF {context}: fitting RMD background", refresh=True
                )
                rmd_ref = extend_reference(ref, train_traces, layer)
                if rmd_ref is None:
                    raise RuntimeError(
                        f"Could not fit RMD background for fold {fold_index}, layer {layer}"
                    )

                progress.set_description_str(
                    f"OOF {context}: scoring held-out prompts", refresh=True
                )
                for prompt_id in test_ids:
                    for trace in groups[prompt_id]:
                        if layer not in trace["hiddens"]:
                            raise ValueError(
                                f"Trace {trace['trace_id']} is missing layer {layer}"
                            )
                        raw = np.asarray(
                            raw_distance(trace["hiddens"][layer], *ref), dtype=float
                        )
                        rmd = np.asarray(
                            relative_distance(trace["hiddens"][layer], *rmd_ref),
                            dtype=float,
                        )
                        entropies = trace.get("entropies")
                        if entropies is None:
                            raise ValueError(
                                f"Trace {trace['trace_id']} is missing entropies"
                            )
                        entropies = np.asarray(entropies, dtype=float)
                        hiddens = np.asarray(trace["hiddens"][layer], dtype=float)
                        projected = ref[0].transform(hiddens)
                        centroid = np.linalg.norm(projected - ref[1], axis=1)
                        mean_logprob = trace.get("mean_logprob")
                        if mean_logprob is not None:
                            mean_logprob = float(mean_logprob)
                        rows.append(
                            {
                                "prompt_id": int(prompt_id),
                                "trace_id": int(trace["trace_id"]),
                                "sample_id": int(trace.get("sample_id", 0)),
                                "is_correct": int(bool(trace["is_correct"])),
                                "fold": int(fold_index),
                                "layer": int(layer),
                                "predicted_answer": trace.get("predicted_answer"),
                                "gold_answer": trace.get("gold_answer"),
                                "mean_logprob": mean_logprob,
                                "trace_length": int(len(entropies)),
                                "entropy_score": -float(entropies.mean()),
                                "logprob_score": mean_logprob,
                                "length_score": -float(np.log1p(len(entropies))),
                                "activation_norm_score": -float(
                                    np.linalg.norm(hiddens, axis=1).mean()
                                ),
                                "centroid_score": -float(centroid.mean()),
                                "raw_score": -float(raw.mean()),
                                "rmd_score": -float(rmd.mean()),
                            }
                        )
                progress.update()

    return sorted(
        rows,
        key=lambda row: (
            row["layer"],
            row["prompt_id"],
            row["sample_id"],
            row["trace_id"],
        ),
    )


def generate_oof_scores_layerwise(
    data_dir: str,
    layers: list[int],
    expected_prompts: int,
    n: int,
    allow_partial: bool,
    pca_dim: int,
    n_splits: int,
    seed: int,
    load_workers: int,
    show_progress: bool,
    load_traces: Callable = load_all_traces,
    score_groups: Callable = generate_oof_scores,
) -> tuple[list[dict], dict]:
    """Load and score one layer at a time to bound peak hidden-state memory."""
    all_rows = []
    data_report = None

    for layer_index, layer in enumerate(layers, start=1):
        _status(
            f"[2/7] Loading layer {layer} ({layer_index}/{len(layers)}) "
            f"with {load_workers} parallel workers"
        )
        traces = load_traces(
            data_dir,
            [layer],
            max_workers=load_workers,
            show_progress=show_progress,
            include_auxiliary=True,
            auxiliary_fields={"entropies"},
        )
        hidden_gib = sum(
            np.asarray(trace["hiddens"][layer]).nbytes
            for trace in traces
            if layer in trace["hiddens"]
        ) / (1024**3)
        _status(
            f"[3/7] Validating {len(traces)} traces for layer {layer} "
            f"({hidden_gib:.1f} GiB hidden states in memory)"
        )
        groups, layer_report = validate_groups(
            group_traces_by_problem(traces),
            expected_prompts=expected_prompts,
            n=n,
            allow_partial=allow_partial,
        )
        if data_report is None:
            data_report = layer_report
        elif layer_report != data_report:
            raise ValueError(
                f"Layer {layer} produced a different data-validation report"
            )

        _status(
            f"[4/7] Generating out-of-fold scores for layer {layer} "
            f"({layer_index}/{len(layers)})"
        )
        all_rows.extend(
            score_groups(
                groups,
                layers=[layer],
                pca_dim=pca_dim,
                n_splits=n_splits,
                seed=seed,
                show_progress=show_progress,
            )
        )

        del groups
        del traces
        gc.collect()

    if data_report is None:
        raise ValueError("No layers were provided")
    return all_rows, data_report


def resample_prompt_rows(rows: list[dict], sampled_prompt_ids: list[int]) -> list[dict]:
    grouped = _group_rows(rows)
    sampled = []
    for draw_index, source_prompt_id in enumerate(sampled_prompt_ids):
        for row in grouped[int(source_prompt_id)]:
            copied = dict(row)
            copied["source_prompt_id"] = int(source_prompt_id)
            copied["prompt_id"] = int(draw_index)
            sampled.append(copied)
    return sampled


def prepare_bootstrap_arrays(rows: list[dict]) -> dict:
    grouped = _group_rows(rows)
    prompt_ids = sorted(grouped)
    group_sizes = {len(grouped[prompt_id]) for prompt_id in prompt_ids}
    if len(group_sizes) != 1:
        raise ValueError("Bootstrap arrays require balanced prompt groups")

    labels = []
    methods = available_score_methods(rows)
    score_rows = {method: [] for method in methods}
    for prompt_id in prompt_ids:
        group = sorted(
            grouped[prompt_id],
            key=lambda row: (int(row["sample_id"]), int(row["trace_id"])),
        )
        labels.append([int(row["is_correct"]) for row in group])
        for method in methods:
            score_rows[method].append(
                [float(row[f"{method}_score"]) for row in group]
            )

    label_array = np.asarray(labels, dtype=np.int8)
    arrays = {
        "prompt_ids": np.asarray(prompt_ids, dtype=int),
        "labels": label_array,
        "pass_rates": label_array.mean(axis=1),
        "mixed_prompts": (label_array.min(axis=1) != label_array.max(axis=1)),
        "methods": {},
    }
    for method in methods:
        scores = np.asarray(score_rows[method], dtype=float)
        prompt_means = scores.mean(axis=1)
        centered_scores = scores - prompt_means[:, None]
        within_ss = np.sum(centered_scores**2, axis=1)
        concordance = np.full(len(prompt_ids), np.nan, dtype=float)
        concordant_pairs = np.zeros(len(prompt_ids), dtype=float)
        pair_counts = np.zeros(len(prompt_ids), dtype=int)

        for prompt_index in np.flatnonzero(arrays["mixed_prompts"]):
            correct = scores[prompt_index][label_array[prompt_index] == 1]
            incorrect = scores[prompt_index][label_array[prompt_index] == 0]
            comparisons = correct[:, None] - incorrect[None, :]
            concordant = float(
                np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)
            )
            n_pairs = int(comparisons.size)
            concordant_pairs[prompt_index] = concordant
            pair_counts[prompt_index] = n_pairs
            concordance[prompt_index] = concordant / n_pairs

        arrays["methods"][method] = {
            "scores": scores,
            "centered_scores": centered_scores,
            "prompt_means": prompt_means,
            "within_ss": within_ss,
            "concordance": concordance,
            "concordant_pairs": concordant_pairs,
            "pair_counts": pair_counts,
        }
    return arrays


def _safe_weighted_auc(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> float | None:
    positive_weight = weights > 0
    labels = labels[positive_weight]
    scores = scores[positive_weight]
    weights = weights[positive_weight]
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def compute_weighted_bootstrap_metrics(
    prepared: dict,
    counts: np.ndarray,
    method: str,
) -> dict:
    labels = prepared["labels"]
    method_arrays = prepared["methods"][method]
    counts = np.asarray(counts, dtype=int)
    if counts.shape != (labels.shape[0],):
        raise ValueError("Bootstrap counts must have one entry per prompt")
    n_draws = int(counts.sum())
    if n_draws < 2:
        raise ValueError("Bootstrap replicate must contain at least two prompts")

    trace_weights = np.repeat(counts, labels.shape[1])
    pooled_auc = _safe_weighted_auc(
        labels.ravel(),
        method_arrays["scores"].ravel(),
        trace_weights,
    )

    mixed = prepared["mixed_prompts"]
    centered_auc = _safe_weighted_auc(
        labels[mixed].ravel(),
        method_arrays["centered_scores"][mixed].ravel(),
        np.repeat(counts[mixed], labels.shape[1]),
    )

    mixed_counts = counts[mixed]
    n_mixed_draws = int(mixed_counts.sum())
    if n_mixed_draws:
        within_macro = float(
            np.sum(mixed_counts * method_arrays["concordance"][mixed])
            / n_mixed_draws
        )
        weighted_pairs = int(
            np.sum(mixed_counts * method_arrays["pair_counts"][mixed])
        )
        within_pair_weighted = (
            float(
                np.sum(
                    mixed_counts * method_arrays["concordant_pairs"][mixed]
                )
                / weighted_pairs
            )
            if weighted_pairs
            else None
        )
    else:
        within_macro = None
        within_pair_weighted = None

    prompt_means = method_arrays["prompt_means"]
    grand_mean = float(np.sum(counts * prompt_means) / n_draws)
    group_size = labels.shape[1]
    ms_between = float(
        group_size
        * np.sum(counts * (prompt_means - grand_mean) ** 2)
        / (n_draws - 1)
    )
    ms_within = float(
        np.sum(counts * method_arrays["within_ss"])
        / (n_draws * (group_size - 1))
    )
    icc_denominator = ms_between + (group_size - 1) * ms_within
    icc = (
        float((ms_between - ms_within) / icc_denominator)
        if icc_denominator > 0
        else None
    )

    repeated_means = np.repeat(prompt_means, counts)
    repeated_pass_rates = np.repeat(prepared["pass_rates"], counts)
    if (
        np.ptp(repeated_means) == 0
        or np.ptp(repeated_pass_rates) == 0
    ):
        spearman = pearson = None
    else:
        spearman = _finite_correlation(
            spearmanr(repeated_means, repeated_pass_rates)
        )
        pearson = _finite_correlation(
            pearsonr(repeated_means, repeated_pass_rates)
        )

    return {
        "pooled_auc": pooled_auc,
        "prompt_centered_auc": centered_auc,
        "within_prompt_macro": within_macro,
        "within_prompt_pair_weighted": within_pair_weighted,
        "score_icc": icc,
        "prompt_score_pass_rate_spearman": spearman,
        "prompt_score_pass_rate_pearson": pearson,
    }


def _interval(draws: list[float]) -> dict:
    finite = np.asarray([value for value in draws if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {"ci_low": None, "ci_high": None, "n_valid": 0}
    low, high = np.percentile(finite, [2.5, 97.5])
    return {
        "ci_low": float(low),
        "ci_high": float(high),
        "n_valid": int(finite.size),
    }


def bootstrap_metrics(
    rows: list[dict],
    n_bootstrap: int,
    seed: int,
    show_progress: bool = False,
    progress_desc: str = "Bootstrap",
) -> dict:
    prepared = prepare_bootstrap_arrays(rows)
    methods = list(prepared["methods"])
    method_draws = {
        method: {metric: [] for metric in SCALAR_METRICS}
        for method in methods
    }
    paired_draws = {
        baseline: {metric: [] for metric in SCALAR_METRICS}
        for baseline in methods
        if baseline != "rmd" and "rmd" in methods
    }
    n_prompts = len(prepared["prompt_ids"])
    rng = np.random.default_rng(seed)

    for _ in tqdm(
        range(n_bootstrap),
        desc=progress_desc,
        unit="replicate",
        disable=not show_progress,
    ):
        sampled_indices = rng.integers(0, n_prompts, size=n_prompts)
        counts = np.bincount(sampled_indices, minlength=n_prompts)
        replicate = {
            method: compute_weighted_bootstrap_metrics(
                prepared, counts, method=method
            )
            for method in methods
        }
        for method in methods:
            for metric in SCALAR_METRICS:
                value = replicate[method][metric]
                if value is not None and math.isfinite(value):
                    method_draws[method][metric].append(float(value))
        if "rmd" in replicate:
            for baseline, metric_draws in paired_draws.items():
                for metric in SCALAR_METRICS:
                    baseline_value = replicate[baseline][metric]
                    rmd_value = replicate["rmd"][metric]
                    if (
                        baseline_value is not None
                        and rmd_value is not None
                        and math.isfinite(baseline_value)
                        and math.isfinite(rmd_value)
                    ):
                        metric_draws[metric].append(
                            float(rmd_value - baseline_value)
                        )

    methods = {
        method: {
            metric: _interval(draws)
            for metric, draws in metrics.items()
        }
        for method, metrics in method_draws.items()
    }
    paired = {}
    for baseline, metric_draws in paired_draws.items():
        paired[baseline] = {}
        for metric, draws in metric_draws.items():
            summary = _interval(draws)
            if draws:
                values = np.asarray(draws)
                p_value = min(
                    1.0,
                    2.0
                    * min(
                        float(np.mean(values <= 0.0)),
                        float(np.mean(values >= 0.0)),
                    ),
                )
            else:
                p_value = None
            paired[baseline][metric] = {
                **summary,
                "p_two_sided": p_value,
            }

    return {
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "methods": methods,
        "paired_rmd_minus_baseline": paired,
        "paired_rmd_minus_raw": paired.get("raw", {}),
    }


def analyze_oof_scores(
    rows: list[dict],
    config: dict,
    n_bootstrap: int,
    seed: int,
    show_progress: bool = False,
) -> dict:
    result = {
        "dataset": config["dataset"],
        "model": config["model"],
        "data": dict(config["data_report"]),
        "settings": {
            "layers": [int(layer) for layer in config["layers"]],
            "pca_dim": config["pca_dim"],
            "n": int(config["n"]),
            "expected_prompts": int(config["expected_prompts"]),
            "n_splits": int(config["n_splits"]),
            "seed": int(config["seed"]),
            "n_bootstrap": int(n_bootstrap),
            "score_orientation": "higher predicts correctness",
            "raw_score": SCORE_DESCRIPTIONS["raw"],
            "rmd_score": SCORE_DESCRIPTIONS["rmd"],
            "score_definitions": dict(SCORE_DESCRIPTIONS),
            "no_layer_selection": True,
        },
        "leakage_protocol": (
            "All trace scores are out-of-fold by prompt. PCA, correct-manifold, "
            "and background-manifold parameters use training prompts only."
        ),
        "layers": {},
    }

    for layer in config["layers"]:
        layer_rows = [row for row in rows if int(row["layer"]) == int(layer)]
        bootstrap = bootstrap_metrics(
            layer_rows,
            n_bootstrap=n_bootstrap,
            seed=seed + int(layer),
            show_progress=show_progress,
            progress_desc=f"Bootstrap L{layer}",
        )
        methods = {}
        method_metrics = {}
        available_methods = available_score_methods(layer_rows)
        for method in available_methods:
            score_key = f"{method}_score"
            metrics = compute_scalar_metrics(layer_rows, score_key=score_key)
            method_metrics[method] = metrics
            methods[method] = {
                "metrics": metrics,
                "confidence_intervals": bootstrap["methods"][method],
            }
        paired_by_baseline = {}
        for baseline, summaries in bootstrap[
            "paired_rmd_minus_baseline"
        ].items():
            paired_by_baseline[baseline] = {}
            for metric, summary in summaries.items():
                baseline_value = method_metrics[baseline][metric]
                rmd_value = method_metrics["rmd"][metric]
                point_estimate = (
                    float(rmd_value - baseline_value)
                    if baseline_value is not None and rmd_value is not None
                    else None
                )
                paired_by_baseline[baseline][metric] = {
                    **summary,
                    "point_estimate": point_estimate,
                }
        result["layers"][str(layer)] = {
            "methods": methods,
            "paired_rmd_minus_baseline": paired_by_baseline,
            "paired_rmd_minus_raw": paired_by_baseline.get("raw", {}),
        }

    return result


def write_trace_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_id",
        "trace_id",
        "sample_id",
        "is_correct",
        "fold",
        "layer",
        "predicted_answer",
        "gold_answer",
        "mean_logprob",
        "trace_length",
        "entropy_score",
        "logprob_score",
        "length_score",
        "activation_norm_score",
        "centroid_score",
        "raw_score",
        "rmd_score",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)


def write_json(result: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def _format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def write_markdown(result: dict, path: str | Path) -> None:
    lines = [
        f"# {result['model']} {result['dataset']} prompt decomposition",
        "",
        (
            f"Data: {result['data']['observed_complete_prompts']} complete prompts "
            f"with N={result['settings']['n']}; "
            f"partial_data={str(result['data']['partial_data']).lower()}."
        ),
        "",
        "| Layer | Method | Pooled AUC | Centered AUC | Within macro | Within pair | ICC | Spearman difficulty |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, layer_result in result["layers"].items():
        for method in SCORE_METHODS:
            if method not in layer_result["methods"]:
                continue
            method_result = layer_result["methods"][method]
            metrics = method_result["metrics"]
            lines.append(
                f"| {layer} | {method} | "
                f"{_format_metric(metrics['pooled_auc'])} | "
                f"{_format_metric(metrics['prompt_centered_auc'])} | "
                f"{_format_metric(metrics['within_prompt_macro'])} | "
                f"{_format_metric(metrics['within_prompt_pair_weighted'])} | "
                f"{_format_metric(metrics['score_icc'])} | "
                f"{_format_metric(metrics['prompt_score_pass_rate_spearman'])} |"
            )
    lines.extend(
        [
            "",
            "No layer was selected after observing these results.",
            "",
            (
                "Confidence intervals use a prompt-cluster bootstrap over fixed "
                "out-of-fold predictions; reference fitting is not repeated."
            ),
            "",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_label", default="math500")
    parser.add_argument("--model_label", default=None)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--expected_prompts", type=int, default=500)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--load_workers", type=int, default=4)
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()

    _status("[1/7] Detecting hidden-state layers")
    layers = parse_int_list(args.layers) if args.layers else detect_layers(args.data_dir)

    if args.load_workers < 1:
        raise ValueError("--load_workers must be at least 1")
    model_label = args.model_label or os.path.basename(
        os.path.dirname(os.path.normpath(args.data_dir))
    )
    _status(
        f"Processing {model_label} one layer at a time to limit peak memory: "
        f"{args.expected_prompts} prompts x {args.n} traces, "
        f"{args.n_splits} folds, layers {layers}"
    )
    rows, data_report = generate_oof_scores_layerwise(
        data_dir=args.data_dir,
        layers=layers,
        expected_prompts=args.expected_prompts,
        n=args.n,
        allow_partial=args.allow_partial,
        pca_dim=args.pca_dim,
        n_splits=args.n_splits,
        seed=args.seed,
        load_workers=args.load_workers,
        show_progress=not args.no_progress,
    )
    config = {
        "dataset": args.dataset_label,
        "model": model_label,
        "layers": layers,
        "pca_dim": args.pca_dim,
        "n": args.n,
        "expected_prompts": args.expected_prompts,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "data_report": data_report,
    }

    _status(
        f"[5/7] Computing decomposition metrics and "
        f"{args.n_bootstrap} prompt-bootstrap resamples per layer"
    )
    result = analyze_oof_scores(
        rows,
        config=config,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        show_progress=not args.no_progress,
    )

    output_dir = Path(args.output_dir)
    prefix = f"{args.dataset_label}_prompt_decomposition"
    _status(f"[6/7] Writing CSV, JSON, and Markdown outputs to {output_dir}")
    write_trace_csv(rows, output_dir / f"{prefix}_oof.csv")
    write_json(result, output_dir / f"{prefix}_results.json")
    write_markdown(result, output_dir / f"{prefix}_report.md")
    _status(f"[7/7] Complete in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
