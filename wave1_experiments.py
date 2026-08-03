"""Wave 1 CPU experiments on the saved Best-of-N traces.

This module deliberately keeps the Wave 1 additions in one small, auditable
stage.  It consumes the existing OOF CSV and NPZ hidden-state artifacts; it
does not collect traces or require a GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from analyze import (
    compute_relative_mahal_distances,
    extend_reference_with_background_safe,
    fit_mahalanobis_reference_safe,
    load_all_traces,
    set_compute_dtype,
    set_max_reference_tokens,
)
from best_of_n import group_traces_by_problem
from trace_caps import resolve_cap
from prompt_decomposition import (
    HIDDEN_PROBE_METHODS,
    bootstrap_parseable_paired_deltas,
    make_prompt_folds,
    prompt_class_balanced_weights,
    region_indices,
)


WAVE1_SCORE_METHODS = (
    "entropy",
    "logprob",
    "length",
    "rmd_tail_q20",
    "rmd_high_entropy_q20",
)
WAVE1_PRIMARY_LAYERS = (21,)


def _finite(value: float | None) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _parseable(trace: dict) -> bool:
    value = trace.get("predicted_answer")
    return value is not None and str(value).strip() != ""


def _safe_slope(values: np.ndarray) -> float:
    if len(values) < 2 or np.ptp(values) == 0:
        return 0.0
    x = np.linspace(0.0, 1.0, len(values))
    return float(np.polyfit(x, values, 1)[0])


def entropy_trajectory_features(
    entropies: np.ndarray,
    *,
    absolute_threshold: float | None = None,
    decay_window: int = 16,
) -> dict[str, float]:
    """Return the four fixed, label-free entropy-trajectory features."""
    values = np.asarray(entropies, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("entropies must be a non-empty 1D array")
    q80 = float(np.quantile(values, 0.8))
    tail = values[values >= q80]
    threshold = q80 if absolute_threshold is None else float(absolute_threshold)
    peaks = np.flatnonzero(values >= threshold)
    positions = peaks / max(1, values.size - 1)
    slopes = []
    for peak in peaks:
        after = values[peak + 1 : peak + 1 + int(decay_window)]
        if len(after) >= 2:
            slopes.append(_safe_slope(after))
    return {
        "upper_tail_mass": float(np.mean(tail)),
        "peak_rate": float(np.mean(values >= threshold)),
        "mean_peak_position": float(np.mean(positions)) if len(positions) else 0.0,
        "post_peak_decay": float(np.mean(slopes)) if slopes else 0.0,
    }


def separated_entropy_events(
    entropies: np.ndarray,
    *,
    min_separation: int = 16,
    quantile: float = 0.8,
) -> np.ndarray:
    """Greedily select high-entropy events, keeping the highest value per window."""
    values = np.asarray(entropies, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("entropies must be a non-empty 1D array")
    count = max(1, int(np.ceil(0.20 * values.size)))
    candidates = region_indices(values, "high_entropy_q20")
    candidates = candidates[np.argsort(values[candidates])[::-1]]
    selected: list[int] = []
    # The event definition is the same top-20% entropy mask used by the
    # decomposition stage; ``quantile`` is retained only for API clarity.
    del quantile
    for index in candidates[:count]:
        index = int(index)
        if all(abs(index - other) >= int(min_separation) for other in selected):
            selected.append(index)
    return np.asarray(sorted(selected), dtype=int)


def position_matched_events(
    entropies: np.ndarray,
    events: np.ndarray,
    *,
    trace_id: int,
    seed: int = 42,
    min_separation: int = 16,
) -> np.ndarray:
    """Sample random event centers near the empirical event positions.

    Jittering the observed normalized positions preserves the position
    distribution while removing the entropy value at the event.  The seed and
    trace id make the control reproducible across folds and reruns.
    """
    values = np.asarray(entropies, dtype=float)
    events = np.asarray(events, dtype=int)
    if not len(events):
        return events.copy()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(trace_id), len(values)]))
    jitter = max(1, int(min_separation) // 2)
    selected: list[int] = []
    for event in events:
        candidates = np.arange(
            max(0, int(event) - jitter),
            min(values.size, int(event) + jitter + 1),
            dtype=int,
        )
        candidates = rng.permutation(candidates)
        candidates = [
            int(candidate)
            for candidate in candidates
            if all(abs(int(candidate) - other) >= int(min_separation) for other in selected)
        ]
        if candidates:
            selected.append(candidates[0])
        elif all(abs(int(event) - other) >= int(min_separation) for other in selected):
            selected.append(int(event))
    if len(selected) < len(events):
        for candidate in rng.permutation(np.arange(values.size, dtype=int)):
            candidate = int(candidate)
            if all(abs(candidate - other) >= int(min_separation) for other in selected):
                selected.append(candidate)
            if len(selected) == len(events):
                break
    return np.asarray(sorted(selected), dtype=int)


def event_locked_profile(
    rmd_scores: np.ndarray,
    entropies: np.ndarray,
    *,
    trace_id: int,
    seed: int = 42,
    window: int = 16,
    min_separation: int = 16,
) -> dict[str, np.ndarray | int]:
    """Extract mean RMD/entropy profiles around high-entropy and matched events."""
    rmd = np.asarray(rmd_scores, dtype=float)
    entropy = np.asarray(entropies, dtype=float)
    if rmd.shape != entropy.shape or rmd.ndim != 1:
        raise ValueError("RMD scores and entropies must be matching 1D arrays")
    events = separated_entropy_events(entropy, min_separation=min_separation)
    random_events = position_matched_events(
        entropy, events, trace_id=trace_id, seed=seed, min_separation=min_separation
    )
    offsets = np.arange(-int(window), int(window) + 1)

    def collect(series: np.ndarray, centers: np.ndarray) -> np.ndarray:
        profiles = [
            series[int(center) + offsets]
            for center in centers
            if int(center) - window >= 0 and int(center) + window < len(series)
        ]
        return np.mean(profiles, axis=0) if profiles else np.full(len(offsets), np.nan)

    valid_events = [
        int(center)
        for center in events
        if int(center) - window >= 0 and int(center) + window < len(rmd)
    ]
    valid_random = [
        int(center)
        for center in random_events
        if int(center) - window >= 0 and int(center) + window < len(rmd)
    ]
    return {
        "offsets": offsets,
        "rmd": collect(rmd, np.asarray(valid_events, dtype=int)),
        "entropy": collect(entropy, np.asarray(valid_events, dtype=int)),
        "random_rmd": collect(rmd, np.asarray(valid_random, dtype=int)),
        "random_entropy": collect(entropy, np.asarray(valid_random, dtype=int)),
        "n_events": int(len(valid_events)),
    }


def lve_features(
    hiddens: np.ndarray,
    entropies: np.ndarray,
    *,
    seed: int = 42,
) -> dict[str, float]:
    """Compute LogNorm-LVE features and the order-destroying shuffle control."""
    hidden = np.asarray(hiddens, dtype=float)
    entropy = np.asarray(entropies, dtype=float)
    if hidden.ndim != 2 or hidden.shape[0] < 2:
        raise ValueError("hiddens must contain at least two token states")
    if entropy.shape != (hidden.shape[0],):
        raise ValueError("entropy length must match hidden-state length")
    log_updates = np.log(np.maximum(np.linalg.norm(np.diff(hidden, axis=0), axis=1), 1e-12))
    he_indices = region_indices(entropy[:-1], "high_entropy_q20")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), hidden.shape[0]]))
    shuffled = rng.permutation(log_updates)
    return {
        "lve_mean": float(np.mean(log_updates)),
        "lve_slope": _safe_slope(log_updates),
        "lve_he": float(np.mean(log_updates[he_indices])),
        "lve_mean_shuffle": float(np.mean(shuffled)),
        "lve_slope_shuffle": _safe_slope(shuffled),
    }


def _prompt_outcome(rows: list[dict]) -> float:
    parsed = [row for row in rows if row.get("predicted_answer") not in (None, "")]
    if not parsed:
        return 0.0
    counts = Counter(str(row["predicted_answer"]) for row in parsed)
    max_count = max(counts.values())
    candidates = [answer for answer, count in counts.items() if count == max_count]
    candidates.sort(key=lambda answer: (-max(float(row.get("logprob_score", -np.inf)) for row in parsed if str(row["predicted_answer"]) == answer), answer))
    return float(candidates[0] == str(parsed[0].get("gold_answer")))


def aggregate_prompt_scores(
    rows: list[dict],
    *,
    methods: tuple[str, ...],
) -> dict[int, dict[str, float]]:
    """Aggregate trace scores per prompt, excluding unparsed traces."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("predicted_answer") not in (None, ""):
            grouped[int(row["prompt_id"])].append(row)
    prompt_ids = sorted({int(row["prompt_id"]) for row in rows})
    return {
        prompt_id: {
            method: (
                float(np.mean([float(row[f"{method}_score"]) for row in grouped[prompt_id]]))
                if grouped.get(prompt_id)
                and all(_finite(row.get(f"{method}_score")) for row in grouped[prompt_id])
                else float("-inf")
            )
            for method in methods
        }
        for prompt_id in prompt_ids
    }


def _prompt_coverage_accuracy(
    scores: dict[int, float], outcomes: dict[int, float], coverage: float
) -> float:
    prompt_ids = sorted(outcomes)
    order = sorted(prompt_ids, key=lambda prompt_id: (-float(scores.get(prompt_id, -np.inf)), prompt_id))
    n = max(1, min(len(order), int(np.ceil(float(coverage) * len(order)))))
    return float(np.mean([outcomes[prompt_id] for prompt_id in order[:n]]))


def _prompt_aurc(scores: dict[int, float], outcomes: dict[int, float]) -> float:
    prompt_ids = sorted(outcomes)
    order = sorted(prompt_ids, key=lambda prompt_id: (-float(scores.get(prompt_id, -np.inf)), prompt_id))
    values = np.asarray([outcomes[prompt_id] for prompt_id in order], dtype=float)
    coverages = np.arange(1, len(values) + 1, dtype=float) / max(1, len(values))
    accuracies = np.cumsum(values) / np.arange(1, len(values) + 1)
    return float(np.trapezoid(accuracies, coverages)) if len(values) > 1 else float(accuracies[0])


def _prompt_curve(scores: dict[int, float], outcomes: dict[int, float]) -> dict[str, list[float]]:
    prompt_ids = sorted(outcomes)
    order = sorted(prompt_ids, key=lambda prompt_id: (-float(scores.get(prompt_id, -np.inf)), prompt_id))
    values = np.asarray([outcomes[prompt_id] for prompt_id in order], dtype=float)
    return {
        "coverages": (np.arange(1, len(values) + 1, dtype=float) / max(1, len(values))).tolist(),
        "accuracies": (np.cumsum(values) / np.arange(1, len(values) + 1)).tolist(),
    }


def _ci_pvalue(values: list[float]) -> dict:
    if not values:
        return {"point_estimate": None, "ci_low": None, "ci_high": None, "p_two_sided": None, "n_valid": 0}
    array = np.asarray(values, dtype=float)
    return {
        "point_estimate": None,
        "ci_low": float(np.percentile(array, 2.5)),
        "ci_high": float(np.percentile(array, 97.5)),
        "p_two_sided": float(min(1.0, 2 * min(np.mean(array <= 0), np.mean(array >= 0)))),
        "n_valid": int(len(array)),
    }


def prompt_abstention_bootstrap(
    prompt_scores: dict[int, dict[str, float]],
    outcomes: dict[int, float],
    *,
    coverages: tuple[float, ...] = (0.5, 0.8),
    n_bootstrap: int = 1000,
    seed: int = 42,
    baseline_methods: tuple[str, ...] = ("length", "logprob", "entropy"),
) -> dict:
    """Report prompt-level risk/coverage values and paired bootstrap deltas."""
    prompt_ids = sorted(outcomes)
    methods = sorted(next(iter(prompt_scores.values())).keys()) if prompt_scores else []
    point = {
        method: {
            "aurc": _prompt_aurc(
                {prompt_id: prompt_scores[prompt_id].get(method, -np.inf) for prompt_id in prompt_ids},
                outcomes,
            ),
            "accuracy_at_coverage": {
                str(coverage): _prompt_coverage_accuracy(
                    {prompt_id: prompt_scores[prompt_id].get(method, -np.inf) for prompt_id in prompt_ids},
                    outcomes,
                    coverage,
                )
                for coverage in coverages
            },
            "curve": _prompt_curve(
                {prompt_id: prompt_scores[prompt_id].get(method, -np.inf) for prompt_id in prompt_ids},
                outcomes,
            ),
        }
        for method in methods
    }
    baselines = tuple(method for method in methods if method in set(baseline_methods))
    draws = {
        (method, baseline, "aurc"): []
        for method in methods
        for baseline in baselines
        if method != baseline
    }
    draws.update(
        {
            (method, baseline, str(coverage)): []
            for method in methods
            for baseline in baselines
            if method != baseline
            for coverage in coverages
        }
    )
    rng = np.random.default_rng(seed)
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(prompt_ids, size=len(prompt_ids), replace=True)
        sampled_outcomes = {index: outcomes[int(prompt_id)] for index, prompt_id in enumerate(sampled)}
        sampled_scores = {
            index: prompt_scores[int(prompt_id)] for index, prompt_id in enumerate(sampled)
        }
        for method, baseline, metric in draws:
            if metric == "aurc":
                left = _prompt_aurc({i: row.get(method, -np.inf) for i, row in sampled_scores.items()}, sampled_outcomes)
                right = _prompt_aurc({i: row.get(baseline, -np.inf) for i, row in sampled_scores.items()}, sampled_outcomes)
            else:
                coverage = float(metric)
                left = _prompt_coverage_accuracy({i: row.get(method, -np.inf) for i, row in sampled_scores.items()}, sampled_outcomes, coverage)
                right = _prompt_coverage_accuracy({i: row.get(baseline, -np.inf) for i, row in sampled_scores.items()}, sampled_outcomes, coverage)
            draws[(method, baseline, metric)].append(float(left - right))
    deltas = {}
    for (method, baseline, metric), values in draws.items():
        key = f"{method}_minus_{baseline}"
        deltas.setdefault(key, {})[metric] = {
            **_ci_pvalue(values),
            "point_estimate": float(point[method]["aurc"] - point[baseline]["aurc"])
            if metric == "aurc"
            else float(point[method]["accuracy_at_coverage"][metric] - point[baseline]["accuracy_at_coverage"][metric]),
        }
    return {"n_prompts": len(prompt_ids), "coverages": list(coverages), "point": point, "deltas": deltas}


def _normalized_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks mapped to (0, 1]; ties share a rank, as in Spearman."""
    return rankdata(values) / float(len(values))


def rank_residualize(
    prompt_scores: dict[int, dict[str, float]],
    prompt_ids: list[int],
    *,
    methods: tuple[str, ...],
    control: str = "length",
) -> dict[int, dict[str, float]]:
    """Strip the monotone `control` component out of each prompt-level score.

    Rank space, not raw values. The collapse this is testing was measured with
    Spearman (rho +0.82 for DeepSeek RMD vs length), so an OLS fit on raw scores
    would leave any monotone-but-nonlinear length dependence in the residual and
    understate the collapse. Regressing normalized ranks on normalized ranks
    removes exactly the component Spearman sees, and the residual has ~zero rank
    correlation with the control by construction.

    The fit uses no labels, so running it on the evaluation prompts leaks nothing
    about correctness -- the same argument that licenses the label-free
    residualization in the L21 analysis. It costs one degree of freedom per
    method, which the bootstrap pays for by refitting inside every draw.

    Prompts whose score or control is non-finite keep the -inf sentinel, so they
    sort last exactly as they do in the unresidualized E1.
    """
    control_raw = np.asarray(
        [prompt_scores[prompt_id].get(control, -np.inf) for prompt_id in prompt_ids],
        dtype=float,
    )
    residualized: dict[int, dict[str, float]] = {
        prompt_id: {} for prompt_id in prompt_ids
    }
    for method in methods:
        raw = np.asarray(
            [prompt_scores[prompt_id].get(method, -np.inf) for prompt_id in prompt_ids],
            dtype=float,
        )
        usable = np.isfinite(raw) & np.isfinite(control_raw)
        if int(usable.sum()) < 3:
            for prompt_id in prompt_ids:
                residualized[prompt_id][method] = -np.inf
            continue
        y = _normalized_ranks(raw[usable])
        x = _normalized_ranks(control_raw[usable])
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denominator = float(x_centered @ x_centered)
        slope = float(x_centered @ y_centered) / denominator if denominator > 0 else 0.0
        residual = y_centered - slope * x_centered
        positions = np.flatnonzero(usable)
        for offset, index in enumerate(positions):
            residualized[prompt_ids[index]][method] = float(residual[offset])
        for index in np.flatnonzero(~usable):
            residualized[prompt_ids[index]][method] = -np.inf
    return residualized


def length_residualized_abstention(
    prompt_scores: dict[int, dict[str, float]],
    outcomes: dict[int, float],
    *,
    methods: tuple[str, ...],
    control: str = "length",
    coverages: tuple[float, ...] = (0.5, 0.8),
    n_bootstrap: int = 1000,
    seed: int = 42,
    comparisons: tuple[tuple[str, str], ...] = (),
) -> dict:
    """Re-run the E1 abstention metrics after partialling `control` out.

    Answers the question E1 cannot: once no scorer is allowed to use trace
    length, does any of them still rank prompts by solvability? The reference is
    an uninformative scorer, whose expected accuracy at every coverage -- and
    therefore whose expected AURC -- is the base accuracy, so each method is
    tested against that rather than against another scorer.

    `comparisons` additionally reports head-to-head residual deltas, e.g. probe
    vs RMD once neither can lean on length.
    """
    prompt_ids = sorted(outcomes)
    scored = tuple(method for method in methods if method != control)
    base_accuracy = float(np.mean([outcomes[prompt_id] for prompt_id in prompt_ids]))
    residual = rank_residualize(
        prompt_scores, prompt_ids, methods=scored, control=control
    )
    point = {
        method: {
            "aurc": _prompt_aurc(
                {prompt_id: residual[prompt_id][method] for prompt_id in prompt_ids},
                outcomes,
            ),
            "accuracy_at_coverage": {
                str(coverage): _prompt_coverage_accuracy(
                    {prompt_id: residual[prompt_id][method] for prompt_id in prompt_ids},
                    outcomes,
                    coverage,
                )
                for coverage in coverages
            },
        }
        for method in scored
    }
    metrics = ("aurc", *(str(coverage) for coverage in coverages))
    vs_base_draws = {(method, metric): [] for method in scored for metric in metrics}
    pair_draws = {
        (method, baseline, metric): []
        for method, baseline in comparisons
        for metric in metrics
    }
    rng = np.random.default_rng(seed)
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(prompt_ids, size=len(prompt_ids), replace=True)
        indices = list(range(len(sampled)))
        sampled_outcomes = {
            index: outcomes[int(prompt_id)] for index, prompt_id in enumerate(sampled)
        }
        sampled_scores = {
            index: prompt_scores[int(prompt_id)] for index, prompt_id in enumerate(sampled)
        }
        # Refit inside the draw: the residualization is estimated, and a CI that
        # conditioned on the full-sample slope would be too narrow.
        sampled_residual = rank_residualize(
            sampled_scores, indices, methods=scored, control=control
        )
        sampled_base = float(np.mean([sampled_outcomes[index] for index in indices]))
        values = {}
        for method in scored:
            column = {index: sampled_residual[index][method] for index in indices}
            values[(method, "aurc")] = _prompt_aurc(column, sampled_outcomes)
            for coverage in coverages:
                values[(method, str(coverage))] = _prompt_coverage_accuracy(
                    column, sampled_outcomes, coverage
                )
        for key, value in values.items():
            if key in vs_base_draws:
                vs_base_draws[key].append(float(value - sampled_base))
        for method, baseline, metric in pair_draws:
            pair_draws[(method, baseline, metric)].append(
                float(values[(method, metric)] - values[(baseline, metric)])
            )
    vs_uninformative = {}
    for (method, metric), draws in vs_base_draws.items():
        observed = (
            point[method]["aurc"]
            if metric == "aurc"
            else point[method]["accuracy_at_coverage"][metric]
        )
        vs_uninformative.setdefault(method, {})[metric] = {
            **_ci_pvalue(draws),
            "point_estimate": float(observed - base_accuracy),
        }
    deltas = {}
    for (method, baseline, metric), draws in pair_draws.items():
        left = (
            point[method]["aurc"]
            if metric == "aurc"
            else point[method]["accuracy_at_coverage"][metric]
        )
        right = (
            point[baseline]["aurc"]
            if metric == "aurc"
            else point[baseline]["accuracy_at_coverage"][metric]
        )
        deltas.setdefault(f"{method}_minus_{baseline}", {})[metric] = {
            **_ci_pvalue(draws),
            "point_estimate": float(left - right),
        }
    return {
        "control": control,
        "n_prompts": len(prompt_ids),
        "base_accuracy": base_accuracy,
        "coverages": list(coverages),
        "prespecified": False,
        "point": point,
        "vs_uninformative": vs_uninformative,
        "deltas": deltas,
    }


def answer_cluster_eligibility(rows: list[dict], max_new_tokens: int | None = None) -> dict:
    """Count sibling answer clusters, retaining an explicit invalid cluster."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["prompt_id"])].append(row)
    cap = resolve_cap(
        max_new_tokens,
        (row.get("trace_length") for row in rows),
        context="answer_cluster_eligibility",
    )
    correct_ge2 = wrong_ge2 = both_ge2 = eligible_clusters = 0
    for prompt_rows in grouped.values():
        gold = str(prompt_rows[0].get("gold_answer"))
        clusters: dict[str, list[dict]] = defaultdict(list)
        for row in prompt_rows:
            answer = row.get("predicted_answer")
            key = "__INVALID__" if answer in (None, "") else str(answer)
            clusters[key].append(row)
        correct = any(answer != "__INVALID__" and answer == gold and len(values) >= 2 for answer, values in clusters.items())
        wrong = any(
            len(values) >= 2 and (answer == "__INVALID__" or answer != gold)
            for answer, values in clusters.items()
        )
        correct_ge2 += int(correct)
        wrong_ge2 += int(wrong)
        both_ge2 += int(correct and wrong)
        for values in clusters.values():
            eligible = [
                row
                for row in values
                if row.get("predicted_answer") not in (None, "")
                and int(row.get("trace_length", 0)) < cap
            ]
            eligible_clusters += int(len(eligible) >= 2)
    return {
        "n_prompts": len(grouped),
        "prompts_with_correct_cluster_ge2": correct_ge2,
        "prompts_with_wrong_cluster_ge2": wrong_ge2,
        "prompts_with_both_ge2": both_ge2,
        "eligible_clusters_after_censoring": eligible_clusters,
    }


def _center_features(rows: list[dict], features: tuple[str, ...]) -> np.ndarray:
    matrix = np.asarray(
        [[float(row[f"{feature}_score"]) for feature in features] for row in rows],
        dtype=float,
    )
    centered = matrix.copy()
    by_prompt: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_prompt[int(row["prompt_id"])].append(index)
    for indices in by_prompt.values():
        centered[indices] -= centered[indices].mean(axis=0, keepdims=True)
    return centered


def crossfit_incremental_probes(
    rows: list[dict],
    feature_sets: dict[str, tuple[str, ...]],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Fit fixed prompt-grouped OOF probes for E6 and return paired deltas."""
    for method in feature_sets:
        for row in rows:
            row[f"{method}_score"] = None
    folds = sorted({int(row["fold"]) for row in rows})
    diagnostics = []
    for fold in folds:
        train_rows = [row for row in rows if int(row["fold"]) != fold]
        test_rows = [row for row in rows if int(row["fold"]) == fold]
        weights = prompt_class_balanced_weights(train_rows)
        labels = np.asarray([int(row["is_correct"]) for row in train_rows], dtype=int)
        for method, features in feature_sets.items():
            scaler = StandardScaler().fit(_center_features(train_rows, features))
            classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(
                scaler.transform(_center_features(train_rows, features)),
                labels,
                sample_weight=weights,
            )
            scores = classifier.predict_proba(
                scaler.transform(_center_features(test_rows, features))
            )[:, 1]
            for row, score in zip(test_rows, scores):
                row[f"{method}_score"] = float(score)
            diagnostics.append({
                "fold": int(fold),
                "method": method,
                "features": list(features),
                "n_train_prompts": len({int(row["prompt_id"]) for row in train_rows}),
                "n_test_traces": len(test_rows),
                "coefficients": {
                    feature: float(value)
                    for feature, value in zip(features, classifier.coef_[0])
                },
            })
    pairs = tuple(
        (method, "probe_outputs_lve")
        for method in feature_sets
        if method != "probe_outputs_lve"
    )
    return {
        "diagnostics": diagnostics,
        "paired_deltas": bootstrap_parseable_paired_deltas(
            rows,
            methods=[],
            baselines=(),
            pairs=pairs,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
    }


def _load_oof_csv(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            converted = dict(row)
            for key in ("prompt_id", "trace_id", "sample_id", "is_correct", "fold", "layer", "trace_length"):
                if converted.get(key) not in (None, ""):
                    converted[key] = int(float(converted[key]))
            for key, value in list(converted.items()):
                if key.endswith("_score") and value not in (None, ""):
                    converted[key] = float(value)
            rows.append(converted)
    return rows


def _majority_outcomes(rows: list[dict]) -> dict[int, float]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["prompt_id"])].append(row)
    return {prompt_id: _prompt_outcome(group) for prompt_id, group in grouped.items()}


def _event_summary(profiles: list[dict], *, key: str) -> tuple[np.ndarray, dict]:
    grouped: dict[int, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for profile in profiles:
        if profile["n_events"] <= 0:
            continue
        grouped[int(profile["prompt_id"])][int(profile["is_correct"])].append(profile[key])
    differences = []
    for values in grouped.values():
        if values.get(1) and values.get(0):
            differences.append(np.mean(values[1], axis=0) - np.mean(values[0], axis=0))
    if not differences:
        return np.empty(0), {"n_prompts": 0, "profile": None}
    return np.mean(differences, axis=0), {"n_prompts": len(differences), "profile": np.mean(differences, axis=0).tolist()}


def _bootstrap_profile(profiles: list[dict], key: str, n_bootstrap: int, seed: int) -> dict:
    prompt_ids = sorted({int(profile["prompt_id"]) for profile in profiles})
    by_prompt = defaultdict(list)
    for profile in profiles:
        by_prompt[int(profile["prompt_id"])].append(profile)
    point, summary = _event_summary(profiles, key=key)
    draws = []
    rng = np.random.default_rng(seed)
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(prompt_ids, size=len(prompt_ids), replace=True)
        sampled_profiles = [dict(profile, prompt_id=index) for index, prompt_id in enumerate(sampled) for profile in by_prompt[int(prompt_id)]]
        value, _ = _event_summary(sampled_profiles, key=key)
        if len(value):
            draws.append(value)
    if not len(point):
        return {
            **summary,
            "ci_low": None,
            "ci_high": None,
            "pre_event_mean": None,
            "post_event_slope": None,
            "pre_event_ci": None,
            "post_event_slope_ci": None,
            "n_valid": 0,
        }
    array = np.asarray(draws, dtype=float)
    center = len(point) // 2
    pre = float(np.nanmean(point[:center]))
    post = _safe_slope(np.asarray(point[center + 1 :], dtype=float))
    draw_pre = np.nanmean(array[:, :center], axis=1) if len(array) else np.asarray([])
    draw_post = np.asarray(
        [_safe_slope(draw[center + 1 :]) for draw in array], dtype=float
    )
    return {
        **summary,
        "ci_low": np.percentile(array, 2.5, axis=0).tolist() if len(array) else None,
        "ci_high": np.percentile(array, 97.5, axis=0).tolist() if len(array) else None,
        "pre_event_mean": pre,
        "post_event_slope": post,
        "pre_event_ci": (
            [float(np.percentile(draw_pre, 2.5)), float(np.percentile(draw_pre, 97.5))]
            if len(draw_pre)
            else None
        ),
        "post_event_slope_ci": (
            [float(np.percentile(draw_post, 2.5)), float(np.percentile(draw_post, 97.5))]
            if len(draw_post)
            else None
        ),
        "n_valid": int(len(array)),
    }


def _compute_rmd_series(
    traces: list[dict], layer: int, pca_dim: int, seed: int = 42
) -> list[dict]:
    groups = group_traces_by_problem(traces)
    parseable_groups = {
        prompt_id: [trace for trace in group if _parseable(trace)]
        for prompt_id, group in groups.items()
    }
    mixed = {
        prompt_id: group
        for prompt_id, group in parseable_groups.items()
        if {int(trace["is_correct"]) for trace in group} == {0, 1}
    }
    folds = make_prompt_folds(sorted(groups), n_splits=5, seed=seed)
    train_ids = folds[0][0]
    train_traces = [trace for prompt_id in train_ids for trace in groups[prompt_id]]
    correct_train = [trace for trace in train_traces if trace["is_correct"]]
    ref = fit_mahalanobis_reference_safe(correct_train, layer, pca_dim)
    rmd_ref = extend_reference_with_background_safe(ref, train_traces, layer)
    output = []
    for prompt_id, group in mixed.items():
        for trace in group:
            distances = compute_relative_mahal_distances(trace["hiddens"][layer], *rmd_ref)
            output.append(
                {
                    "prompt_id": int(prompt_id),
                    "trace_id": int(trace["trace_id"]),
                    "is_correct": int(trace["is_correct"]),
                    "entropies": np.asarray(trace["entropies"], dtype=float),
                    "rmd": -np.asarray(distances, dtype=float),
                }
            )
    return output


def run_wave1(
    *,
    data_dir: str,
    oof_csv: str,
    output_dir: str,
    dataset_label: str,
    model_label: str,
    layers: list[int],
    pca_dim: int = 128,
    max_new_tokens: int | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    load_workers: int = 4,
    hidden_dtype: np.dtype | None = None,
) -> dict:
    """Run E1 and E4–E7; E2 is emitted by prompt_decomposition.py."""
    rows = _load_oof_csv(oof_csv)
    layer = max(layers)
    layer_rows = [row for row in rows if int(row["layer"]) == layer]
    fold_by_trace = {int(row["trace_id"]): int(row["fold"]) for row in layer_rows}
    # The supervised hidden-state probe joins E1 only when prompt_decomposition
    # was run with --hidden_probe_regions; older OOF CSVs lack the columns.
    present = set(layer_rows[0]) if layer_rows else set()
    e1_methods = (
        "rmd_tail_q20",
        "rmd_high_entropy_q20",
        "length",
        "logprob",
        "entropy",
        *(
            method
            for method in HIDDEN_PROBE_METHODS
            if f"{method}_score" in present
        ),
    )
    prompt_scores = aggregate_prompt_scores(layer_rows, methods=e1_methods)
    outcomes = _majority_outcomes(layer_rows)
    hidden_probes_present = any(
        method in e1_methods for method in HIDDEN_PROBE_METHODS
    )
    e1 = prompt_abstention_bootstrap(
        prompt_scores,
        outcomes,
        n_bootstrap=n_bootstrap,
        seed=seed,
        # Adding rmd_tail_q20 as a baseline yields probe-minus-RMD deltas: does
        # supervision on the same activations beat the unsupervised scorer?
        # Only when probes ran, so probe-free runs stay byte-identical.
        baseline_methods=(
            ("length", "logprob", "entropy", "rmd_tail_q20")
            if hidden_probes_present
            else ("length", "logprob", "entropy")
        ),
    )
    # E1R: the same abstention metrics with the monotone length component removed
    # from every scorer. E1 shows RMD beating length; it cannot show whether RMD
    # carries anything length does not already supply. Exploratory, not
    # pre-registered.
    e1_residual = length_residualized_abstention(
        prompt_scores,
        outcomes,
        methods=e1_methods,
        control="length",
        n_bootstrap=n_bootstrap,
        seed=seed + 50000,
        comparisons=tuple(
            (method, "rmd_tail_q20")
            for method in HIDDEN_PROBE_METHODS
            if method in e1_methods
        ),
    )
    eligibility = answer_cluster_eligibility(layer_rows, max_new_tokens=max_new_tokens)

    base_traces = load_all_traces(
        data_dir,
        [layer],
        max_workers=load_workers,
        include_auxiliary=True,
        auxiliary_fields={"entropies", "token_logprobs"},
        hidden_dtype=hidden_dtype,
    )
    parseable_traces = [trace for trace in base_traces if _parseable(trace)]
    mixed_ids = {
        prompt_id
        for prompt_id, group in group_traces_by_problem(parseable_traces).items()
        if {int(trace["is_correct"]) for trace in group} == {0, 1}
    }
    threshold = float(np.quantile(np.concatenate([np.asarray(trace["entropies"], dtype=float) for trace in parseable_traces]), 0.8))
    trajectory_rows = []
    for trace in parseable_traces:
        if int(trace["idx"]) not in mixed_ids:
            continue
        features = entropy_trajectory_features(trace["entropies"], absolute_threshold=threshold)
        trajectory_rows.append({
            "prompt_id": int(trace["idx"]),
            "is_correct": int(trace["is_correct"]),
            "entropy_score": -float(np.mean(trace["entropies"])),
            "logprob_score": float(trace["mean_logprob"]),
            "upper_tail_mass_score": -features["upper_tail_mass"],
            "peak_rate_score": -features["peak_rate"],
            "mean_peak_position_score": features["mean_peak_position"],
            "post_peak_decay_score": features["post_peak_decay"],
        })
    e4_pairs = bootstrap_parseable_paired_deltas(
        trajectory_rows,
        methods=["upper_tail_mass", "peak_rate", "mean_peak_position", "post_peak_decay"],
        baselines=("entropy",),
        n_bootstrap=n_bootstrap,
        seed=seed + 40000,
    )
    e4 = {
        "absolute_threshold": threshold,
        "n_mixed_prompts": len(mixed_ids),
        "features": e4_pairs,
    }

    e5 = {"layers": {}}
    e6 = {"layers": {}}
    for current_layer in (layer, *[value for value in layers if value != layer]):
        traces = (
            base_traces
            if current_layer == layer
            else load_all_traces(
                data_dir,
                [current_layer],
                max_workers=load_workers,
                include_auxiliary=True,
                auxiliary_fields={"entropies", "token_logprobs"},
                hidden_dtype=hidden_dtype,
            )
        )
        parseable_layer_traces = [trace for trace in traces if _parseable(trace)]
        profiles = []
        for item in _compute_rmd_series(traces, current_layer, pca_dim, seed=seed):
            trace = next(trace for trace in parseable_layer_traces if int(trace["trace_id"]) == item["trace_id"])
            profile = event_locked_profile(
                item["rmd"], item["entropies"], trace_id=item["trace_id"], seed=seed
            )
            profiles.append({**profile, "prompt_id": item["prompt_id"], "is_correct": item["is_correct"]})
        e5["layers"][str(current_layer)] = {
            "rmd": _bootstrap_profile(profiles, "rmd", n_bootstrap, seed + current_layer),
            "random_rmd": _bootstrap_profile(profiles, "random_rmd", n_bootstrap, seed + current_layer + 1000),
            "entropy": _bootstrap_profile(profiles, "entropy", n_bootstrap, seed + current_layer + 2000),
        }
        lve_rows = []
        for trace in parseable_layer_traces:
            if int(trace["idx"]) not in mixed_ids or current_layer not in trace["hiddens"]:
                continue
            features = lve_features(trace["hiddens"][current_layer], trace["entropies"], seed=seed)
            lve_rows.append({
                "prompt_id": int(trace["idx"]),
                "trace_id": int(trace["trace_id"]),
                "fold": fold_by_trace.get(int(trace["trace_id"]), 0),
                "is_correct": int(trace["is_correct"]),
                "length_score": -float(np.log1p(len(trace["entropies"]))),
                "logprob_score": float(trace["mean_logprob"]),
                **{f"{key}_score": value for key, value in features.items()},
            })
        lve_methods = ["lve_mean", "lve_slope", "lve_he", "lve_mean_shuffle", "lve_slope_shuffle"]
        lve_probe_features = {
            "probe_outputs_lve": ("length", "logprob"),
            "probe_lve_mean": ("length", "logprob", "lve_mean"),
            "probe_lve_slope": ("length", "logprob", "lve_slope"),
            "probe_lve_he": ("length", "logprob", "lve_he"),
        }
        probe_result = crossfit_incremental_probes(
            lve_rows,
            lve_probe_features,
            n_bootstrap=n_bootstrap,
            seed=seed + current_layer + 5000,
        )
        shuffle_result = bootstrap_parseable_paired_deltas(
            lve_rows,
            methods=[],
            baselines=(),
            pairs=(("lve_mean", "lve_mean_shuffle"), ("lve_slope", "lve_slope_shuffle")),
            n_bootstrap=n_bootstrap,
            seed=seed + current_layer + 6000,
        )
        e6["layers"][str(current_layer)] = {
            "metrics": bootstrap_parseable_paired_deltas(
                lve_rows,
                methods=lve_methods,
                baselines=("logprob",),
                n_bootstrap=n_bootstrap,
                seed=seed + current_layer + 3000,
            ),
            "incremental_probes": probe_result,
            "shuffle_controls": shuffle_result,
        }
        if current_layer != layer:
            del traces

    result = {
        "dataset": dataset_label,
        "model": model_label,
        "settings": {
            "layers": layers,
            "deepest_layer": layer,
            "pca_dim": pca_dim,
            "max_new_tokens": max_new_tokens,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "censoring": "unparsed and cap-hit traces excluded from correctness comparisons",
        },
        "e1_prompt_abstention": e1,
        "e1r_length_residualized_abstention": e1_residual,
        "e4_entropy_trajectory": e4,
        "e5_event_locked_rmd": e5,
        "e6_log_norm_lve": e6,
        "e7_sibling_eligibility": eligibility,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset_label}_wave1"
    (output / f"{prefix}_results.json").write_text(json.dumps(result, indent=2) + "\n")
    write_wave1_report(result, output / f"{prefix}_report.md")
    plot_wave1(result, output / f"{prefix}.png")
    return result


def write_wave1_report(result: dict, path: str | Path) -> None:
    e1 = result["e1_prompt_abstention"]
    e7 = result["e7_sibling_eligibility"]
    lines = [
        f"# Wave 1 experiments — {result['model']} / {result['dataset']}",
        "",
        "All Wave 1 claims use prompt-cluster bootstrap intervals; E1 is prompt-level and E4–E7 are CPU-only diagnostics.",
        "",
        "## E1 — prompt abstention",
        "",
        "| Method | AURC | Acc@50% | Acc@80% |",
        "|---|---:|---:|---:|",
    ]
    for method, values in e1["point"].items():
        lines.append(
            f"| {method} | {values['aurc']:.3f} | {values['accuracy_at_coverage'].get('0.5', float('nan')):.3f} | {values['accuracy_at_coverage'].get('0.8', float('nan')):.3f} |"
        )
    lines.extend([
        "",
        "### E1 paired deltas",
        "",
        "| Contrast | Metric | Delta [95% CI] | p | n |",
        "|---|---|---|---:|---:|",
    ])
    for contrast, metrics in e1["deltas"].items():
        for metric, values in metrics.items():
            lines.append(
                f"| {contrast} | {metric} | {values['point_estimate']:+.3f} "
                f"[{values['ci_low']:+.3f}, {values['ci_high']:+.3f}] | "
                f"{values['p_two_sided'] if values['p_two_sided'] is not None else 'NA'} | {values['n_valid']} |"
            )
    e1r = result.get("e1r_length_residualized_abstention")
    if e1r:
        lines.extend([
            "",
            "## E1R — abstention with length partialled out (exploratory)",
            "",
            f"Control: `{e1r['control']}`, removed in rank space. Reference is an "
            f"uninformative scorer, whose expected AURC and accuracy at every "
            f"coverage equal the base accuracy {e1r['base_accuracy']:.3f}. "
            "A method with no length-independent signal lands at zero here.",
            "",
            "| Method | resid AURC | Δ vs uninformative [95% CI] | p | resid Acc@50% | Δ@50% [95% CI] | p |",
            "|---|---:|---|---:|---:|---|---:|",
        ])
        for method, values in e1r["point"].items():
            aurc = e1r["vs_uninformative"][method]["aurc"]
            half = e1r["vs_uninformative"][method].get("0.5", {})
            lines.append(
                f"| {method} | {values['aurc']:.3f} | "
                f"{aurc['point_estimate']:+.3f} [{aurc['ci_low']:+.3f}, {aurc['ci_high']:+.3f}] | "
                f"{aurc['p_two_sided']} | "
                f"{values['accuracy_at_coverage'].get('0.5', float('nan')):.3f} | "
                f"{half.get('point_estimate', float('nan')):+.3f} "
                f"[{half.get('ci_low', float('nan')):+.3f}, {half.get('ci_high', float('nan')):+.3f}] | "
                f"{half.get('p_two_sided')} |"
            )
        if e1r["deltas"]:
            lines.extend([
                "",
                "### E1R head-to-head (neither scorer may use length)",
                "",
                "| Contrast | Metric | Delta [95% CI] | p |",
                "|---|---|---|---:|",
            ])
            for contrast, metrics in e1r["deltas"].items():
                for metric, values in metrics.items():
                    lines.append(
                        f"| {contrast} | {metric} | {values['point_estimate']:+.3f} "
                        f"[{values['ci_low']:+.3f}, {values['ci_high']:+.3f}] | "
                        f"{values['p_two_sided']} |"
                    )
    lines.extend(["", "## E7 — sibling eligibility", "", "```json", json.dumps(e7, indent=2), "```", ""])
    e4 = result.get("e4_entropy_trajectory", {})
    lines.extend(
        [
            "",
            "## E4 — entropy-trajectory autopsy",
            "",
            f"Fixed absolute threshold: {e4.get('absolute_threshold')}",
            "",
            "| Feature | Centered-AUC delta vs mean entropy |",
            "|---|---|",
        ]
    )
    for name, value in e4.get("features", {}).items():
        metric = value.get("prompt_centered_auc", {})
        lines.append(
            f"| {name} | {metric.get('point_estimate')} "
            f"[{metric.get('ci_low')}, {metric.get('ci_high')}] "
            f"p={metric.get('p_two_sided')} |"
        )
    lines.extend(["", "## E5 — event-locked RMD", ""])
    for layer, payload in result.get("e5_event_locked_rmd", {}).get("layers", {}).items():
        rmd = payload.get("rmd", {})
        random_rmd = payload.get("random_rmd", {})
        lines.append(
            f"- L{layer}: RMD pre={rmd.get('pre_event_mean')} "
            f"CI={rmd.get('pre_event_ci')}; post-slope={rmd.get('post_event_slope')} "
            f"CI={rmd.get('post_event_slope_ci')}; random slope={random_rmd.get('post_event_slope')} "
            f"CI={random_rmd.get('post_event_slope_ci')}"
        )
    lines.extend(["", "## E6 — LogNorm-LVE", ""])
    for layer in result.get("e6_log_norm_lve", {}).get("layers", {}):
        lines.append(f"- L{layer}: incremental probes and token-order shuffle controls are in the JSON.")
    Path(path).write_text("\n".join(lines))


def plot_wave1(result: dict, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    point = result["e1_prompt_abstention"]["point"]
    for method, values in point.items():
        axes[0].plot(values["curve"]["coverages"], values["curve"]["accuracies"], label=method)
    axes[0].set(xlabel="Coverage", ylabel="Prompt accuracy", title="E1 risk–coverage curves")
    layers = result["e5_event_locked_rmd"]["layers"]
    primary = layers[str(result["settings"]["deepest_layer"])]
    profile = primary["rmd"].get("profile")
    if profile:
        axes[1].plot(np.arange(len(profile)) - len(profile) // 2, profile, label="RMD correct − incorrect")
    random_profile = primary["random_rmd"].get("profile")
    if random_profile:
        axes[1].plot(np.arange(len(random_profile)) - len(random_profile) // 2, random_profile, label="position-matched random")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Event-relative token", ylabel="Score difference", title="E5 event-locked profile")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--oof_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_label", default="math500")
    parser.add_argument("--model_label", default="qwen")
    parser.add_argument("--layers", default="21")
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_workers", type=int, default=4)
    # Memory controls, mirroring prompt_decomposition.py. wave1 fits the same
    # reference manifolds on the same traces, so it has the same peak-RAM
    # profile and needs the same three levers.
    parser.add_argument(
        "--hidden_dtype", type=str, default="float32",
        choices=["float16", "float32"],
        help="Resident dtype for hidden states. Stored data is float32 but comes from a "
             "bf16 forward pass, so float16 round-trips losslessly and halves memory.",
    )
    parser.add_argument(
        "--compute_dtype", type=str, default="float64",
        choices=["float32", "float64"],
        help="Working precision for PCA/Mahalanobis.",
    )
    parser.add_argument(
        "--max_reference_tokens", type=int, default=0,
        help="Cap on tokens concatenated to fit each reference manifold (0 = no cap).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_compute_dtype(args.compute_dtype)
    set_max_reference_tokens(args.max_reference_tokens or None)
    run_wave1(
        hidden_dtype=np.dtype(args.hidden_dtype).type,
        data_dir=args.data_dir,
        oof_csv=args.oof_csv,
        output_dir=args.output_dir,
        dataset_label=args.dataset_label,
        model_label=args.model_label,
        layers=[int(value) for value in args.layers.split(",") if value.strip()],
        pca_dim=args.pca_dim,
        max_new_tokens=args.max_new_tokens,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        load_workers=args.load_workers,
    )


if __name__ == "__main__":
    main()
