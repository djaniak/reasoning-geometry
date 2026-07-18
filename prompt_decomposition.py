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
    "prompt_local_rmd",
    "contrast_full",
    "contrast_high_entropy_q20",
    "contrast_tail_q20",
)

CONTRASTIVE_METHODS = (
    "contrast_full",
    "contrast_high_entropy_q20",
    "contrast_tail_q20",
)

CONTRASTIVE_REGION_NAMES = {
    "contrast_full": "full",
    "contrast_high_entropy_q20": "high_entropy_q20",
    "contrast_tail_q20": "tail_q20",
}

SCORE_DESCRIPTIONS = {
    "entropy": "-mean(token entropy)",
    "logprob": "mean token log-probability",
    "length": "-log1p(token count)",
    "activation_norm": "-mean(token hidden-state L2 norm)",
    "centroid": "-mean Euclidean distance to the correct-trace PCA centroid",
    "raw": "-mean(raw Mahalanobis token distance)",
    "rmd": "-mean(relative Mahalanobis token distance)",
    "prompt_local_rmd": (
        "mean(leave-one-trace-out prompt-local background distance "
        "- raw Mahalanobis distance)"
    ),
    "contrast_full": "projection onto the prompt-contrastive full-trace direction",
    "contrast_high_entropy_q20": (
        "projection onto the prompt-contrastive highest-entropy 20% direction"
    ),
    "contrast_tail_q20": "projection onto the prompt-contrastive final-20% direction",
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


def is_unparsed(row: dict) -> bool:
    """A trace with no parseable final answer.

    Such traces are auto-labeled is_correct=False upstream (collect_data.py),
    so they are not evidence about reasoning *correctness* -- only about whether
    a final answer was emitted. They confound the within-prompt decomposition.
    """
    value = row.get("predicted_answer")
    return value is None or str(value).strip() == ""


def region_indices(entropies: np.ndarray, region: str) -> np.ndarray:
    """Return deterministic token indices for a fixed contrastive region."""
    values = np.asarray(entropies, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("entropy sequence must be a non-empty 1D array")
    if region == "full":
        return np.arange(values.size, dtype=int)
    count = max(1, int(np.ceil(0.20 * values.size)))
    if region == "tail_q20":
        return np.arange(values.size - count, values.size, dtype=int)
    if region == "high_entropy_q20":
        order = np.argsort(values, kind="stable")
        return np.sort(order[-count:]).astype(int)
    raise ValueError(f"unknown region: {region}")


def trace_region_mean(
    projected: np.ndarray, entropies: np.ndarray, region: str
) -> np.ndarray:
    """Average projected hidden states over one fixed token region."""
    projected = np.asarray(projected, dtype=float)
    entropies = np.asarray(entropies, dtype=float)
    if projected.ndim != 2 or projected.shape[0] == 0:
        raise ValueError("projected hidden states must be a non-empty 2D array")
    if entropies.ndim != 1 or entropies.shape[0] != projected.shape[0]:
        raise ValueError("entropy sequence must match projected hidden-state length")
    indices = region_indices(entropies, region)
    return projected[indices].mean(axis=0)


def _unit_prompt_difference(
    traces: list[dict],
    projected_by_trace: dict[int, np.ndarray],
    region: str,
    shuffled_correct_indices: np.ndarray | None = None,
    region_means_by_trace: dict[int, np.ndarray] | None = None,
) -> np.ndarray | None:
    parseable = [trace for trace in traces if not is_unparsed(trace)]
    if shuffled_correct_indices is None:
        correct = [trace for trace in parseable if bool(trace["is_correct"])]
        incorrect = [trace for trace in parseable if not bool(trace["is_correct"])]
    else:
        correct = [parseable[index] for index in shuffled_correct_indices]
        correct_ids = {id(trace) for trace in correct}
        incorrect = [trace for trace in parseable if id(trace) not in correct_ids]
    if not correct or not incorrect:
        return None
    if region_means_by_trace is None:
        region_means_by_trace = {
            int(trace["trace_id"]): trace_region_mean(
                projected_by_trace[int(trace["trace_id"])],
                trace.get("entropies"),
                region,
            )
            for trace in parseable
        }
    correct_means = [region_means_by_trace[int(trace["trace_id"])] for trace in correct]
    incorrect_means = [
        region_means_by_trace[int(trace["trace_id"])] for trace in incorrect
    ]
    difference = np.mean(correct_means, axis=0) - np.mean(incorrect_means, axis=0)
    norm = float(np.linalg.norm(difference))
    return difference / norm if norm > 0 else None


def _alignment_summary(vectors: list[np.ndarray]) -> dict:
    if not vectors:
        return {"resultant_length": None, "pairwise_cosine_mean": None}
    array = np.asarray(vectors, dtype=float)
    mean_vector = np.mean(array, axis=0)
    pairwise = []
    for left in range(len(array)):
        for right in range(left + 1, len(array)):
            pairwise.append(float(np.dot(array[left], array[right])))
    return {
        "resultant_length": float(np.linalg.norm(mean_vector)),
        "pairwise_cosine_mean": float(np.mean(pairwise)) if pairwise else None,
    }


def fit_prompt_contrastive_direction(
    groups: dict[int, list[dict]],
    prompt_ids: list[int],
    projected_by_trace: dict[int, np.ndarray],
    region: str,
    seed: int = 42,
    n_alignment_shuffles: int = 0,
) -> dict:
    """Fit an equal-prompt-weighted correctness direction on training prompts."""
    prompt_vectors = []
    eligible = []
    skipped = {}
    region_means_by_trace = {
        int(trace["trace_id"]): trace_region_mean(
            projected_by_trace[int(trace["trace_id"])],
            trace.get("entropies"),
            region,
        )
        for prompt_id in prompt_ids
        for trace in groups[int(prompt_id)]
        if not is_unparsed(trace)
    }
    for prompt_id in prompt_ids:
        traces = groups[int(prompt_id)]
        parseable = [trace for trace in traces if not is_unparsed(trace)]
        if not any(bool(trace["is_correct"]) for trace in parseable):
            skipped[int(prompt_id)] = "no_correct"
            continue
        if not any(not bool(trace["is_correct"]) for trace in parseable):
            skipped[int(prompt_id)] = (
                "no_parseable_incorrect" if any(is_unparsed(trace) for trace in traces)
                else "no_incorrect"
            )
            continue
        vector = _unit_prompt_difference(
            traces,
            projected_by_trace,
            region,
            region_means_by_trace=region_means_by_trace,
        )
        if vector is None:
            skipped[int(prompt_id)] = "zero_norm_difference"
            continue
        prompt_vectors.append(vector)
        eligible.append(int(prompt_id))

    direction = None
    if prompt_vectors:
        mean_vector = np.mean(prompt_vectors, axis=0)
        norm = float(np.linalg.norm(mean_vector))
        if norm > 0:
            direction = mean_vector / norm

    observed = _alignment_summary(prompt_vectors)
    null_values = []
    rng = np.random.default_rng(seed)
    for _ in range(max(0, int(n_alignment_shuffles))):
        shuffled_vectors = []
        for prompt_id in eligible:
            parseable = [
                trace for trace in groups[prompt_id] if not is_unparsed(trace)
            ]
            n_correct = sum(bool(trace["is_correct"]) for trace in parseable)
            shuffled_indices = rng.choice(
                len(parseable), size=n_correct, replace=False
            )
            vector = _unit_prompt_difference(
                groups[prompt_id],
                projected_by_trace,
                region,
                shuffled_correct_indices=np.asarray(shuffled_indices, dtype=int),
                region_means_by_trace=region_means_by_trace,
            )
            if vector is not None:
                shuffled_vectors.append(vector)
        null_values.append(
            float(np.linalg.norm(np.mean(shuffled_vectors, axis=0)))
            if shuffled_vectors
            else None
        )

    finite_null = np.asarray(
        [value for value in null_values if value is not None], dtype=float
    )
    observed_alignment = observed["resultant_length"]
    if finite_null.size:
        null_summary = {
            "mean": float(np.mean(finite_null)),
            "ci_low": float(np.percentile(finite_null, 2.5)),
            "ci_high": float(np.percentile(finite_null, 97.5)),
            "p_value": (
                float((1 + np.sum(finite_null >= observed_alignment)) / (finite_null.size + 1))
                if observed_alignment is not None
                else None
            ),
            "n_valid": int(finite_null.size),
        }
    else:
        null_summary = {
            "mean": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
            "n_valid": 0,
        }

    return {
        "direction": direction,
        "prompt_vectors": prompt_vectors,
        "prompt_ids": eligible,
        "n_prompt_vectors": len(prompt_vectors),
        "skipped_prompts": skipped,
        "observed_alignment": observed_alignment,
        "observed_pairwise_cosine": observed["pairwise_cosine_mean"],
        "null": null_summary,
    }


def score_contrastive_trace(
    projected: np.ndarray,
    entropies: np.ndarray,
    direction: np.ndarray | None,
    region: str,
) -> float:
    """Score a trace by projection onto a fitted correctness direction."""
    if direction is None:
        raise ValueError("no usable prompt-contrastive direction was fitted")
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if direction.ndim != 1 or norm == 0:
        raise ValueError("prompt-contrastive direction must be nonzero and 1D")
    return float(np.dot(trace_region_mean(projected, entropies, region), direction / norm))


def truncation_report(rows: list[dict], max_new_tokens: int | None = None) -> dict:
    """Diagnose how much of the 'incorrect' class is non-answers vs wrong answers.

    A high unparsed/capped rate means within-prompt metrics are largely driven by
    correct-vs-truncated contrasts (a generation-length/termination signal) rather
    than correct-vs-wrong reasoning. ``max_new_tokens`` defines the length cap; if
    omitted it is inferred as the maximum observed trace length.
    """
    lengths = [
        int(row["trace_length"])
        for row in rows
        if row.get("trace_length") is not None
    ]
    cap = int(max_new_tokens) if max_new_tokens else (max(lengths) if lengths else None)
    n_traces = len(rows)
    n_unparsed = sum(1 for row in rows if is_unparsed(row))
    n_unparsed_incorrect = sum(
        1 for row in rows if is_unparsed(row) and not int(row["is_correct"])
    )
    n_capped = (
        sum(1 for row in rows if int(row.get("trace_length", 0)) >= cap)
        if cap is not None
        else 0
    )
    n_incorrect = sum(1 for row in rows if not int(row["is_correct"]))
    return {
        "n_traces": int(n_traces),
        "max_new_tokens": cap,
        "n_unparsed": int(n_unparsed),
        "unparsed_rate": float(n_unparsed / n_traces) if n_traces else None,
        "n_capped": int(n_capped),
        "capped_rate": float(n_capped / n_traces) if n_traces else None,
        "n_unparsed_and_incorrect": int(n_unparsed_incorrect),
        "unparsed_share_of_incorrect": (
            float(n_unparsed_incorrect / n_incorrect) if n_incorrect else None
        ),
    }


def parseable_within_prompt_metrics(rows: list[dict]) -> dict:
    """Within-prompt metrics recomputed after dropping unparsed (non-answer) traces.

    This isolates the genuine correct-vs-wrong-answer contrast from the
    correct-vs-truncated contrast. ``n_mixed_prompts`` collapsing relative to the
    full set is the tell that the headline within-prompt signal was driven by
    non-answers rather than wrong answers.
    """
    parseable = [row for row in rows if not is_unparsed(row)]
    methods = {}
    for method in available_score_methods(parseable):
        score_key = f"{method}_score"
        concordance = within_prompt_concordance(parseable, score_key=score_key)
        centered = prompt_centered_auc(parseable, score_key=score_key)
        methods[method] = {
            "within_prompt_macro": concordance["macro"],
            "within_prompt_pair_weighted": concordance["pair_weighted"],
            "prompt_centered_auc": centered["auc"],
            "n_mixed_prompts": concordance["n_mixed_prompts"],
            "n_within_prompt_pairs": concordance["n_pairs"],
        }
    return {"n_parseable_traces": len(parseable), "methods": methods}


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


def prompt_local_rmd_scores(
    projected_by_trace: dict[int, np.ndarray],
    raw_distances_by_trace: dict[int, np.ndarray],
    variance_floor: float = 1e-6,
) -> dict[int, float]:
    """Score traces against a same-prompt leave-one-trace-out background.

    The target distance is the global correct-manifold Mahalanobis distance
    already used by ``raw_score``. The background is the diagonal Gaussian over
    sibling attempts for the same prompt, excluding the trace being scored.
    Higher scores mean the trace is closer to the global correct manifold than
    expected relative to sibling attempts from the same prompt.
    """
    if len(projected_by_trace) < 2:
        raise ValueError("Prompt-local RMD requires at least two traces per prompt")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive")

    prepared: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    total_count = 0
    total_sum = None
    total_sumsq = None

    for trace_id, projected in projected_by_trace.items():
        projected = np.asarray(projected, dtype=float)
        if projected.ndim != 2 or projected.shape[0] == 0:
            raise ValueError("Projected hidden states must be non-empty 2D arrays")
        raw = np.asarray(raw_distances_by_trace[trace_id], dtype=float)
        if raw.shape != (projected.shape[0],):
            raise ValueError(
                "Raw distances must have one value per projected hidden-state row"
            )
        prepared[int(trace_id)] = (projected, raw)
        trace_sum = projected.sum(axis=0)
        trace_sumsq = np.square(projected).sum(axis=0)
        total_sum = trace_sum if total_sum is None else total_sum + trace_sum
        total_sumsq = (
            trace_sumsq if total_sumsq is None else total_sumsq + trace_sumsq
        )
        total_count += projected.shape[0]

    scores = {}
    for trace_id, (projected, raw) in prepared.items():
        background_count = total_count - projected.shape[0]
        if background_count <= 0:
            raise ValueError("Prompt-local background has no sibling tokens")
        background_sum = total_sum - projected.sum(axis=0)
        background_sumsq = total_sumsq - np.square(projected).sum(axis=0)
        mean = background_sum / background_count
        variance = background_sumsq / background_count - np.square(mean)
        variance = np.maximum(variance, variance_floor)
        diff = projected - mean
        local_dist = np.sqrt(np.maximum(np.sum(np.square(diff) / variance, axis=1), 0))
        scores[int(trace_id)] = float(np.mean(local_dist - raw))

    return scores


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
    contrastive_regions: tuple[str, ...] = (),
    n_alignment_shuffles: int = 0,
    alignment_seed: int = 42,
    alignment_diagnostics: list[dict] | None = None,
) -> list[dict]:
    contrastive_regions = tuple(dict.fromkeys(contrastive_regions))
    unknown_regions = set(contrastive_regions) - set(CONTRASTIVE_REGION_NAMES.values())
    if unknown_regions:
        raise ValueError(f"unknown contrastive regions: {sorted(unknown_regions)}")
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

                contrastive_fits = {}
                if contrastive_regions:
                    training_projected = {}
                    for trace in train_traces:
                        if layer not in trace["hiddens"]:
                            raise ValueError(
                                f"Trace {trace['trace_id']} is missing layer {layer}"
                            )
                        entropies = trace.get("entropies")
                        if entropies is None:
                            raise ValueError(
                                f"Trace {trace['trace_id']} is missing entropies"
                            )
                        training_projected[int(trace["trace_id"])] = ref[0].transform(
                            np.asarray(trace["hiddens"][layer], dtype=float)
                        )
                    for region in contrastive_regions:
                        fit = fit_prompt_contrastive_direction(
                            groups,
                            train_ids,
                            training_projected,
                            region=region,
                            seed=alignment_seed + fold_index * 1000 + int(layer),
                            n_alignment_shuffles=n_alignment_shuffles,
                        )
                        if fit["direction"] is None:
                            raise RuntimeError(
                                f"No usable prompt-contrastive direction for fold "
                                f"{fold_index}, layer {layer}, region {region}"
                            )
                        contrastive_fits[region] = fit
                        if alignment_diagnostics is not None:
                            alignment_diagnostics.append(
                                {
                                    "fold": int(fold_index),
                                    "layer": int(layer),
                                    "region": region,
                                    "n_prompt_vectors": int(fit["n_prompt_vectors"]),
                                    "skipped_prompts": fit["skipped_prompts"],
                                    "observed_alignment": fit["observed_alignment"],
                                    "observed_pairwise_cosine": fit[
                                        "observed_pairwise_cosine"
                                    ],
                                    "null": fit["null"],
                                }
                            )

                progress.set_description_str(
                    f"OOF {context}: scoring held-out prompts", refresh=True
                )
                for prompt_id in test_ids:
                    prompt_payloads = []
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
                        prompt_payloads.append(
                            {
                                "trace": trace,
                                "hiddens": hiddens,
                                "projected": projected,
                                "entropies": entropies,
                                "raw": raw,
                                "rmd": rmd,
                                "centroid": centroid,
                            }
                        )

                    prompt_local_scores = prompt_local_rmd_scores(
                        {
                            int(payload["trace"]["trace_id"]): payload["projected"]
                            for payload in prompt_payloads
                        },
                        {
                            int(payload["trace"]["trace_id"]): payload["raw"]
                            for payload in prompt_payloads
                        },
                    )

                    for payload in prompt_payloads:
                        trace = payload["trace"]
                        hiddens = payload["hiddens"]
                        raw = payload["raw"]
                        rmd = payload["rmd"]
                        centroid = payload["centroid"]
                        entropies = payload["entropies"]
                        mean_logprob = trace.get("mean_logprob")
                        if mean_logprob is not None:
                            mean_logprob = float(mean_logprob)
                        row = {
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
                            "prompt_local_rmd_score": prompt_local_scores[
                                int(trace["trace_id"])
                            ],
                        }
                        for region in contrastive_regions:
                            method = next(
                                method
                                for method, mapped_region in CONTRASTIVE_REGION_NAMES.items()
                                if mapped_region == region
                            )
                            row[f"{method}_score"] = score_contrastive_trace(
                                payload["projected"],
                                entropies,
                                contrastive_fits[region]["direction"],
                                region,
                            )
                        rows.append(row)
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
    contrastive_regions: tuple[str, ...] = (),
    n_alignment_shuffles: int = 0,
    alignment_seed: int = 42,
    return_diagnostics: bool = False,
) -> tuple[list[dict], dict] | tuple[list[dict], dict, list[dict]]:
    """Load and score one layer at a time to bound peak hidden-state memory."""
    all_rows = []
    data_report = None
    contrastive_diagnostics: list[dict] = []

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
        score_kwargs = {
            "layers": [layer],
            "pca_dim": pca_dim,
            "n_splits": n_splits,
            "seed": seed,
            "show_progress": show_progress,
        }
        if contrastive_regions:
            score_kwargs.update(
                {
                    "contrastive_regions": contrastive_regions,
                    "n_alignment_shuffles": n_alignment_shuffles,
                    "alignment_seed": alignment_seed,
                    "alignment_diagnostics": contrastive_diagnostics,
                }
            )
        all_rows.extend(score_groups(groups, **score_kwargs))

        del groups
        del traces
        gc.collect()

    if data_report is None:
        raise ValueError("No layers were provided")
    if return_diagnostics:
        return all_rows, data_report, contrastive_diagnostics
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


def bootstrap_parseable_paired_deltas(
    rows: list[dict],
    methods: list[str],
    baselines: tuple[str, ...],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Prompt-cluster bootstrap for paired deltas on variable-size groups."""
    grouped = _group_rows(rows)
    prompt_ids = sorted(grouped)
    metrics = ("prompt_centered_auc", "within_prompt_macro")
    pairs = [
        (method, baseline)
        for method in methods
        for baseline in baselines
        if method != baseline
        and all(
            all(
                row.get(f"{candidate}_score") is not None
                and math.isfinite(float(row[f"{candidate}_score"]))
                for row in rows
            )
            for candidate in (method, baseline)
        )
    ]
    draws = {
        pair: {metric: [] for metric in metrics}
        for pair in pairs
    }
    rng = np.random.default_rng(seed)

    def paired_metrics(sample_rows: list[dict], method: str) -> dict:
        """Compute only the two metrics used by this variable-size bootstrap.

        Parseability filtering leaves prompts with unequal numbers of traces, so
        balanced-group statistics such as ICC are intentionally not computed here.
        """
        score_key = f"{method}_score"
        return {
            "prompt_centered_auc": prompt_centered_auc(
                sample_rows, score_key=score_key
            )["auc"],
            "within_prompt_macro": within_prompt_concordance(
                sample_rows, score_key=score_key
            )["macro"],
        }

    if prompt_ids and n_bootstrap > 0:
        for _ in range(int(n_bootstrap)):
            sampled_ids = rng.choice(prompt_ids, size=len(prompt_ids), replace=True)
            replicate = resample_prompt_rows(rows, [int(value) for value in sampled_ids])
            values = {
                method: paired_metrics(replicate, method)
                for method in sorted(set(methods).union(baselines))
            }
            for method, baseline in pairs:
                for metric in metrics:
                    left = values[method][metric]
                    right = values[baseline][metric]
                    if left is not None and right is not None:
                        draws[(method, baseline)][metric].append(float(left - right))

    result = {}
    for pair, pair_draws in draws.items():
        method, baseline = pair
        result[f"{method}_minus_{baseline}"] = {}
        point_values = {
            candidate: paired_metrics(rows, candidate)
            for candidate in (method, baseline)
        }
        for metric, values in pair_draws.items():
            summary = _interval(values)
            if values:
                array = np.asarray(values, dtype=float)
                p_value = min(
                    1.0,
                    2.0
                    * min(
                        float(np.mean(array <= 0.0)),
                        float(np.mean(array >= 0.0)),
                    ),
                )
            else:
                p_value = None
            point_left = point_values[method][metric]
            point_right = point_values[baseline][metric]
            result[f"{method}_minus_{baseline}"][metric] = {
                **summary,
                "point_estimate": (
                    float(point_left - point_right)
                    if point_left is not None and point_right is not None
                    else None
                ),
                "p_two_sided": p_value,
            }
    return result


def analyze_oof_scores(
    rows: list[dict],
    config: dict,
    n_bootstrap: int,
    seed: int,
    show_progress: bool = False,
    max_new_tokens: int | None = None,
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
        "truncation": truncation_report(rows, max_new_tokens=max_new_tokens),
        "truncation_caveat": (
            "Unparsed traces (no final answer) are auto-labeled incorrect upstream "
            "and are typically length-capped, not wrong-answer. They confound the "
            "within-prompt decomposition. See each layer's parseable_only block for "
            "within-prompt metrics restricted to traces that emitted an answer."
        ),
        "layers": {},
        "contrastive": {
            "regions": list(config.get("contrastive_regions", [])),
            "n_alignment_shuffles": int(config.get("n_alignment_shuffles", 0)),
            "alignment_seed": int(config.get("alignment_seed", config["seed"])),
            "alignment_diagnostics": config.get("contrastive_diagnostics", []),
        },
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
            # Trace length is the strong baseline that confounds geometry claims:
            # wrong/hard/truncated traces ramble, so length alone predicts correctness.
            # RMD's contribution is its margin OVER length, not over entropy.
            "paired_rmd_minus_length": paired_by_baseline.get("length", {}),
            "truncation": truncation_report(layer_rows, max_new_tokens=max_new_tokens),
            "parseable_only": parseable_within_prompt_metrics(layer_rows),
        }
        parseable_rows = [row for row in layer_rows if not is_unparsed(row)]
        contrastive_methods = [
            method
            for method in CONTRASTIVE_METHODS
            if method in result["layers"][str(layer)]["parseable_only"]["methods"]
        ]
        if contrastive_methods:
            result["layers"][str(layer)]["parseable_only"][
                "paired_contrastive_deltas"
            ] = bootstrap_parseable_paired_deltas(
                parseable_rows,
                contrastive_methods,
                baselines=("rmd", "logprob"),
                n_bootstrap=n_bootstrap,
                seed=seed + int(layer) + 10000,
            )

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
        "prompt_local_rmd_score",
        *(f"{method}_score" for method in CONTRASTIVE_METHODS),
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
    def _paired(entry: dict, metric: str) -> str:
        cell = entry.get(metric) if entry else None
        if not cell or cell.get("point_estimate") is None:
            return "NA"
        point = _format_metric(cell["point_estimate"])
        low = _format_metric(cell.get("ci_low"))
        high = _format_metric(cell.get("ci_high"))
        p = cell.get("p_two_sided")
        p_str = "NA" if p is None else f"{p:.3f}"
        return f"{point} [{low}, {high}] p={p_str}"

    if any(
        layer_result.get("paired_rmd_minus_length")
        for layer_result in result["layers"].values()
    ):
        lines.extend(
            [
                "",
                "## Primary contrast: RMD − length",
                "",
                (
                    "Trace length is the strong baseline for correctness (wrong/hard/"
                    "truncated traces ramble), so RMD's contribution is its margin OVER "
                    "length, not over entropy. Point estimate with 95% prompt-bootstrap CI "
                    "and two-sided p; a contribution requires the CI to exclude zero."
                ),
                "",
                "| Layer | RMD−length pooled AUC | RMD−length centered AUC | RMD−length within macro |",
                "|---:|:---|:---|:---|",
            ]
        )
        for layer, layer_result in result["layers"].items():
            paired_length = layer_result.get("paired_rmd_minus_length", {})
            lines.append(
                f"| {layer} | "
                f"{_paired(paired_length, 'pooled_auc')} | "
                f"{_paired(paired_length, 'prompt_centered_auc')} | "
                f"{_paired(paired_length, 'within_prompt_macro')} |"
            )

    truncation = result.get("truncation")
    if truncation:
        lines.extend(
            [
                "",
                "## Truncation / parseability diagnostic",
                "",
                (
                    f"Unparsed (no final answer): {truncation['n_unparsed']}/"
                    f"{truncation['n_traces']} "
                    f"({_format_metric(truncation['unparsed_rate'])}); "
                    f"length-capped at {truncation['max_new_tokens']}: "
                    f"{truncation['n_capped']} "
                    f"({_format_metric(truncation['capped_rate'])}); "
                    f"unparsed share of the incorrect class: "
                    f"{_format_metric(truncation['unparsed_share_of_incorrect'])}."
                ),
                "",
                (
                    "Unparsed traces are auto-labeled incorrect upstream and are "
                    "usually truncated, not wrong-answer. The within-prompt metrics "
                    "below restrict to traces that emitted a parseable answer; a large "
                    "drop in mixed-prompt count or in the RMD-minus-entropy gap means "
                    "the headline within-prompt signal was a truncation detector."
                ),
                "",
                "| Layer | Method | Parseable within macro | Parseable centered AUC | Mixed prompts |",
                "|---:|:---|---:|---:|---:|",
            ]
        )
        for layer, layer_result in result["layers"].items():
            parseable = layer_result.get("parseable_only", {}).get("methods", {})
            for method in SCORE_METHODS:
                if method not in parseable:
                    continue
                pm = parseable[method]
                lines.append(
                    f"| {layer} | {method} | "
                    f"{_format_metric(pm['within_prompt_macro'])} | "
                    f"{_format_metric(pm['prompt_centered_auc'])} | "
                    f"{pm['n_mixed_prompts']} |"
                )
    contrastive = result.get("contrastive", {})
    if contrastive.get("regions"):
        lines.extend(
            [
                "",
                "## Prompt-contrastive direction diagnostics",
                "",
                (
                    "Directions are fit out-of-fold from parseable mixed training prompts. "
                    "Each prompt contributes one normalized difference vector; alignment "
                    "nulls shuffle labels within prompts while preserving class counts."
                ),
                "",
                "| Layer | Region | Prompt vectors | Observed alignment | Pairwise cosine | Null mean | Null 95% interval | p |",
                "|---:|:---|---:|---:|---:|---:|:---|---:|",
            ]
        )
        for diagnostic in contrastive.get("alignment_diagnostics", []):
            null = diagnostic.get("null", {})
            lines.append(
                f"| {diagnostic['layer']} | {diagnostic['region']} | "
                f"{diagnostic['n_prompt_vectors']} | "
                f"{_format_metric(diagnostic.get('observed_alignment'))} | "
                f"{_format_metric(diagnostic.get('observed_pairwise_cosine'))} | "
                f"{_format_metric(null.get('mean'))} | "
                f"[{_format_metric(null.get('ci_low'))}, { _format_metric(null.get('ci_high')) }] | "
                f"{_format_metric(null.get('p_value'))} |"
            )
        lines.extend(
            [
                "",
                "## Parseable paired contrasts",
                "",
                "Contrastive score minus baseline, using prompt-cluster bootstrap intervals.",
                "",
                "| Layer | Contrast | Centered AUC delta | Within macro delta |",
                "|---:|:---|:---|:---|",
            ]
        )
        for layer, layer_result in result["layers"].items():
            paired = layer_result.get("parseable_only", {}).get(
                "paired_contrastive_deltas", {}
            )
            for name, values in paired.items():
                lines.append(
                    f"| {layer} | {name} | "
                    f"{_paired(values, 'prompt_centered_auc')} | "
                    f"{_paired(values, 'within_prompt_macro')} |"
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
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Generation cap used at collection time; enables exact capped-trace "
        "diagnostics. Inferred from the max observed trace length if omitted.",
    )
    parser.add_argument("--expected_prompts", type=int, default=500)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--contrastive_regions",
        default="",
        help="Comma-separated fixed regions: full,high_entropy_q20,tail_q20",
    )
    parser.add_argument("--n_alignment_shuffles", type=int, default=0)
    parser.add_argument("--alignment_seed", type=int, default=42)
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
    contrastive_regions = tuple(
        part.strip()
        for part in args.contrastive_regions.split(",")
        if part.strip()
    )
    rows, data_report, contrastive_diagnostics = generate_oof_scores_layerwise(
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
        contrastive_regions=contrastive_regions,
        n_alignment_shuffles=args.n_alignment_shuffles,
        alignment_seed=args.alignment_seed,
        return_diagnostics=True,
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
        "contrastive_regions": list(contrastive_regions),
        "n_alignment_shuffles": int(args.n_alignment_shuffles),
        "alignment_seed": int(args.alignment_seed),
        "data_report": data_report,
        "contrastive_diagnostics": contrastive_diagnostics,
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
        max_new_tokens=args.max_new_tokens,
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
