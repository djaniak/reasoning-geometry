"""Prompt-held-out incremental abstention analysis.

This module is deliberately separate from the historical Wave-1 report.  It
uses the existing prompt folds to fit multivariate B0/B1 readouts, keeps
all-unparsed prompts visible as automatic failures, and reports calibrated
probabilities separately from ranking metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from trace_caps import resolve_cap


BASE_FEATURE_NAMES = ("length", "entropy", "logprob", "vote_agreement")
FEATURE_NAMES = BASE_FEATURE_NAMES + (
    "rmd_tail_q20",
    "prompt_only_geometry",
    "cap_count",
    "unparsed_count",
    "deepconf_global",
    "deepconf_tail_q20",
)
METRIC_NAMES = ("auacc", "aurc", "brier", "log_loss")


def _finite(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def is_parseable_answer(value) -> bool:
    return value is not None and bool(str(value).strip())


def _group_rows(rows: Iterable[Mapping]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["prompt_id"])].append(dict(row))
    return dict(grouped)


def _row_logprob(row: Mapping) -> float:
    for key in ("logprob_score", "mean_logprob"):
        value = _finite(row.get(key))
        if value is not None:
            return value
    return float("-inf")


def _winning_answer(rows: list[Mapping]) -> str | None:
    parsed = [row for row in rows if is_parseable_answer(row.get("predicted_answer"))]
    if not parsed:
        return None
    counts = Counter(str(row["predicted_answer"]) for row in parsed)
    best = max(counts.values())
    candidates = sorted(answer for answer, count in counts.items() if count == best)
    if len(candidates) == 1:
        return candidates[0]
    return max(
        candidates,
        key=lambda answer: max(
            _row_logprob(row)
            for row in parsed
            if str(row["predicted_answer"]) == answer
        ),
    )


def _plurality_outcome(rows: list[Mapping]) -> float:
    winner = _winning_answer(rows)
    if winner is None:
        return 0.0
    gold = next(
        (str(row.get("gold_answer")) for row in rows if row.get("gold_answer") is not None),
        "",
    )
    return float(winner == gold)


def prompt_accounting(
    rows: Iterable[Mapping],
    *,
    max_new_tokens: int | None = None,
    data_dir: str | None = None,
    expected_traces: int = 8,
) -> dict[int, dict]:
    """Return prompt-level parseability/cap accounting and plurality outcomes.

    An all-unparsed prompt is assigned outcome zero for the full-population
    analysis, but ``automatic_failure`` records that this is a generation
    failure rather than an ordinary incorrect answer.  The valid-plurality
    populations exclude those prompts.
    """
    grouped = _group_rows(rows)
    # Pool every observed length: the unvalidated-cap warning is about the run,
    # and a per-prompt view would fire on any prompt whose siblings all
    # terminated early under a perfectly valid cap.
    cap = resolve_cap(
        max_new_tokens,
        data_dir=data_dir,
        lengths=(
            _finite(row.get("trace_length"))
            for group in grouped.values()
            for row in group
        ),
        context="prompt_accounting",
    )
    result: dict[int, dict] = {}
    for prompt_id, group in sorted(grouped.items()):
        parseable_count = sum(
            is_parseable_answer(row.get("predicted_answer")) for row in group
        )
        lengths = [
            int(value)
            for row in group
            if (value := _finite(row.get("trace_length"))) is not None
        ]
        cap_count = sum(length >= cap.value for length in lengths)
        result[prompt_id] = {
            "prompt_id": prompt_id,
            "n_traces": len(group),
            "expected_traces": int(expected_traces),
            "parseable_count": int(parseable_count),
            "unparsed_count": int(len(group) - parseable_count),
            "cap_count": int(cap_count),
            "automatic_failure": bool(parseable_count == 0),
            "valid_plurality": bool(parseable_count > 0),
            "all_eight_parseable": bool(
                len(group) == int(expected_traces)
                and parseable_count == int(expected_traces)
            ),
            "complete_traces": bool(len(group) == int(expected_traces)),
            "outcome": _plurality_outcome(group),
            "fold": int(group[0]["fold"]) if group[0].get("fold") not in (None, "") else None,
        }
    return result


def _mean_field(group: list[Mapping], key: str) -> float:
    values = [_finite(row.get(key)) for row in group]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else float("nan")


def aggregate_prompt_features(
    rows: Iterable[Mapping],
    *,
    max_new_tokens: int | None = None,
    data_dir: str | None = None,
    expected_traces: int = 8,
    prompt_geometry: Mapping[int, float] | None = None,
    exact_scores: Mapping[int, Mapping[str, float]] | None = None,
) -> dict[int, dict]:
    """Aggregate the requested output and geometry scores once per prompt.

    Length, entropy, log-probability, and tail RMD are averaged over all cached
    traces (including unparsed traces).  Vote agreement uses only parseable
    traces; an all-unparsed prompt receives zero agreement and is separately
    marked as an automatic failure by :func:`prompt_accounting`.
    """
    grouped = _group_rows(rows)
    accounting = prompt_accounting(
        rows,
        max_new_tokens=max_new_tokens,
        data_dir=data_dir,
        expected_traces=expected_traces,
    )
    features: dict[int, dict] = {}
    for prompt_id, group in sorted(grouped.items()):
        parsed = [row for row in group if is_parseable_answer(row.get("predicted_answer"))]
        winner = _winning_answer(group)
        vote = (
            sum(str(row.get("predicted_answer")) == winner for row in parsed)
            / len(parsed)
            if parsed and winner is not None
            else 0.0
        )
        entry = {
            "prompt_id": prompt_id,
            "length": _mean_field(group, "length_score"),
            "entropy": _mean_field(group, "entropy_score"),
            "logprob": _mean_field(group, "logprob_score"),
            "vote_agreement": float(vote),
            "rmd_tail_q20": _mean_field(group, "rmd_tail_q20_score"),
            "prompt_only_geometry": float("nan"),
            "cap_count": accounting[prompt_id]["cap_count"],
            "unparsed_count": accounting[prompt_id]["unparsed_count"],
            "outcome": accounting[prompt_id]["outcome"],
            "automatic_failure": accounting[prompt_id]["automatic_failure"],
            "valid_plurality": accounting[prompt_id]["valid_plurality"],
            "all_eight_parseable": accounting[prompt_id]["all_eight_parseable"],
            "complete_traces": accounting[prompt_id]["complete_traces"],
            "fold": accounting[prompt_id]["fold"],
            "deepconf_global": float("nan"),
            "deepconf_tail_q20": float("nan"),
        }
        if prompt_geometry is not None and prompt_id in prompt_geometry:
            entry["prompt_only_geometry"] = float(prompt_geometry[prompt_id])
        if exact_scores is not None and prompt_id in exact_scores:
            for key in ("deepconf_global", "deepconf_tail_q20"):
                value = _finite(exact_scores[prompt_id].get(key))
                if value is not None:
                    entry[key] = value
        features[prompt_id] = entry
    return features


def _impute_and_scale(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    means = np.nanmean(np.where(np.isfinite(train), train, np.nan), axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    train = np.where(np.isfinite(train), train, means)
    test = np.where(np.isfinite(test), test, means)
    scale = np.std(train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return (train - means) / scale, (test - means) / scale


def crossfit_logistic_predictions(
    features: np.ndarray,
    outcomes: np.ndarray,
    folds: np.ndarray,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Fit a logistic readout inside each supplied prompt fold."""
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(outcomes, dtype=float)
    fold_values = np.asarray(folds)
    if len(x) != len(y) or len(y) != len(fold_values):
        raise ValueError("features, outcomes, and folds must have the same length")
    probabilities = np.full(len(y), np.nan, dtype=float)
    usable = np.isfinite(y) & np.isfinite(fold_values.astype(float))
    if not usable.any():
        return probabilities
    for fold in sorted(set(fold_values[usable].tolist())):
        test_mask = usable & (fold_values == fold)
        train_mask = usable & (fold_values != fold)
        if not test_mask.any() or not train_mask.any():
            continue
        train_x, test_x = _impute_and_scale(x[train_mask], x[test_mask])
        train_y = y[train_mask].astype(int)
        if len(np.unique(train_y)) < 2:
            probabilities[test_mask] = float(np.mean(train_y))
            continue
        model = LogisticRegression(max_iter=2000, random_state=seed)
        try:
            model.fit(train_x, train_y)
            probabilities[test_mask] = model.predict_proba(test_x)[:, 1]
        except ValueError:
            probabilities[test_mask] = float(np.mean(train_y))
    return probabilities


def _auacc(scores: np.ndarray, outcomes: np.ndarray) -> float:
    usable = np.isfinite(scores) & np.isfinite(outcomes)
    if not usable.any():
        return float("nan")
    scores = np.asarray(scores[usable], dtype=float)
    outcomes = np.asarray(outcomes[usable], dtype=float)
    order = np.argsort(-scores, kind="stable")
    values = outcomes[order]
    if len(values) == 1:
        return float(values[0])
    coverages = np.arange(1, len(values) + 1, dtype=float) / len(values)
    accuracies = np.cumsum(values) / np.arange(1, len(values) + 1, dtype=float)
    return float(np.trapezoid(accuracies, coverages))


def prompt_metrics(probabilities: np.ndarray, outcomes: np.ndarray) -> dict[str, float | int | None]:
    """Return ranking and strictly proper scores for calibrated OOF probabilities."""
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    usable = np.isfinite(probabilities) & np.isfinite(outcomes)
    n = int(usable.sum())
    if n == 0:
        return {"auacc": None, "aurc": None, "brier": None, "log_loss": None, "n": 0}
    p = np.clip(probabilities[usable], 1e-12, 1 - 1e-12)
    y = outcomes[usable]
    auacc = _auacc(p, y)
    aurc = float((1.0 - 1.0 / n) - auacc)
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return {
        "auacc": float(auacc),
        "aurc": aurc,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": log_loss,
        "n": n,
    }


def paired_bootstrap_delta(
    left: np.ndarray,
    right: np.ndarray,
    outcomes: np.ndarray,
    *,
    metric: str = "auacc",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired prompt-bootstrap delta ``left - right`` for one metric."""
    if metric not in METRIC_NAMES:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRIC_NAMES}")
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    usable = np.isfinite(left) & np.isfinite(right) & np.isfinite(outcomes)
    left, right, outcomes = left[usable], right[usable], outcomes[usable]
    if not len(outcomes):
        return {
            "metric": metric,
            "point_estimate": None,
            "ci_low": None,
            "ci_high": None,
            "p_two_sided": None,
            "n_valid": 0,
        }

    def value(scores: np.ndarray, labels: np.ndarray) -> float:
        return float(prompt_metrics(scores, labels)[metric])

    point = value(left, outcomes) - value(right, outcomes)
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        sampled = rng.integers(0, len(outcomes), size=len(outcomes))
        draws[index] = value(left[sampled], outcomes[sampled]) - value(
            right[sampled], outcomes[sampled]
        )
    return {
        "metric": metric,
        "point_estimate": float(point),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_two_sided": float(
            min(1.0, 2.0 * min(np.mean(draws <= 0), np.mean(draws >= 0)))
        ),
        "n_valid": int(len(draws)),
    }


def _read_oof(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in ("prompt_id", "trace_id", "sample_id", "fold", "layer", "trace_length"):
                if row.get(key) not in (None, ""):
                    row[key] = int(float(row[key]))
            for key, value in list(row.items()):
                if key.endswith("_score") and value not in (None, ""):
                    row[key] = float(value)
            rows.append(row)
    return rows


def _load_prompt_states(data_dir: str | Path, layer: int) -> dict[int, np.ndarray]:
    """Extract row-zero (last prompt position) states without retaining tokens."""
    source = Path(data_dir)
    if source.is_file():
        with np.load(source, allow_pickle=True) as data:
            ids = np.asarray(data["prompt_ids"], dtype=int)
            key = f"prompt_hidden_L{layer}"
            if key not in data.files:
                raise ValueError(f"{source} has no {key} array")
            values = np.asarray(data[key], dtype=np.float32)
            # The exact pilot stores [prompt, trace, hidden]; average the eight
            # traces so the downstream prompt-level feature has one state.
            if values.ndim == 3:
                values = values.mean(axis=1)
            if values.ndim != 2 or len(values) != len(ids):
                raise ValueError(f"invalid prompt-state shape in {source}: {values.shape}")
            return {int(prompt_id): values[index] for index, prompt_id in enumerate(ids)}
    states: dict[int, list[np.ndarray]] = defaultdict(list)
    for path in sorted(Path(data_dir).glob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            available = set(data.files)
            for metadata in data["metadata"]:
                trace_id = int(metadata.get("trace_id", metadata.get("idx", 0)))
                prompt_id = int(metadata["idx"])
                key = f"hidden_L{layer}_{trace_id}"
                if key not in available:
                    key = f"hidden_L{layer}_{int(metadata.get('idx', 0))}"
                if key not in available or data[key].ndim != 2 or not len(data[key]):
                    continue
                states[prompt_id].append(np.asarray(data[key][0], dtype=np.float32))
    return {
        prompt_id: np.mean(np.stack(values, axis=0), axis=0)
        for prompt_id, values in states.items()
        if values
    }


def crossfit_prompt_geometry_scores(
    states: Mapping[int, np.ndarray],
    outcomes: Mapping[int, float],
    folds: Mapping[int, int],
    *,
    pca_dim: int = 32,
) -> dict[int, float]:
    """OOF negative Mahalanobis distance to the training correct prompt states."""
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA

    prompt_ids = sorted(set(states) & set(outcomes) & set(folds))
    result = {prompt_id: float("nan") for prompt_id in prompt_ids}
    for fold in sorted({folds[prompt_id] for prompt_id in prompt_ids}):
        test_ids = [prompt_id for prompt_id in prompt_ids if folds[prompt_id] == fold]
        train_ids = [
            prompt_id
            for prompt_id in prompt_ids
            if folds[prompt_id] != fold and float(outcomes[prompt_id]) > 0.5
        ]
        train = np.asarray([states[prompt_id] for prompt_id in train_ids], dtype=np.float32)
        if len(train) < 3:
            continue
        try:
            n_components = max(1, min(int(pca_dim), train.shape[0] - 1, train.shape[1]))
            pca = PCA(n_components=n_components, random_state=42, svd_solver="randomized")
            projected = pca.fit_transform(train)
            covariance = LedoitWolf().fit(projected)
            test = np.asarray([states[prompt_id] for prompt_id in test_ids], dtype=np.float32)
            test_projected = pca.transform(test)
            diff = test_projected - covariance.location_
            distances = np.sqrt(
                np.maximum(np.einsum("ij,jk,ik->i", diff, covariance.precision_, diff), 0.0)
            )
            for prompt_id, distance in zip(test_ids, distances):
                result[prompt_id] = -float(distance)
        except (ValueError, np.linalg.LinAlgError):
            continue
    return result


def _population_ids(features: Mapping[int, Mapping]) -> dict[str, list[int]]:
    all_ids = sorted(features)
    valid = [prompt_id for prompt_id in all_ids if features[prompt_id]["valid_plurality"]]
    cap_free = [prompt_id for prompt_id in valid if features[prompt_id]["cap_count"] == 0]
    return {
        "full_population": all_ids,
        "valid_plurality": valid,
        "cap_free_valid_plurality": cap_free,
        "cap_free_full_population": [
            prompt_id for prompt_id in all_ids if features[prompt_id]["cap_count"] == 0
        ],
        "all_eight_parseable": [
            prompt_id for prompt_id in all_ids if features[prompt_id]["all_eight_parseable"]
        ],
    }


def _model_specs(features: Mapping[int, Mapping]) -> dict[str, tuple[str, ...]]:
    specs = {
        "B0": BASE_FEATURE_NAMES,
        "B1": BASE_FEATURE_NAMES + ("rmd_tail_q20",),
    }
    if any(_finite(row.get("prompt_only_geometry")) is not None for row in features.values()):
        specs.update(
            {
                "B0_prompt_only_geometry": BASE_FEATURE_NAMES + ("prompt_only_geometry",),
                "B1_prompt_only_geometry": BASE_FEATURE_NAMES
                + ("prompt_only_geometry", "rmd_tail_q20"),
            }
        )
    if any(_finite(row.get("deepconf_global")) is not None for row in features.values()):
        specs.update(
            {
                "DeepConf_global": ("deepconf_global",),
                "DeepConf_tail_q20": ("deepconf_tail_q20",),
                "B0_plus_DeepConf_global": BASE_FEATURE_NAMES + ("deepconf_global",),
                "B0_plus_DeepConf_tail_q20": BASE_FEATURE_NAMES + ("deepconf_tail_q20",),
                "B0_plus_DeepConf_tail_q20_plus_RMD": BASE_FEATURE_NAMES
                + ("deepconf_tail_q20", "rmd_tail_q20"),
            }
        )
    return specs


def _load_exact_prompt_scores(path: str | Path) -> dict[int, dict[str, float]]:
    """Aggregate exact trace summaries into one raw DeepConf score per prompt."""
    source = Path(path)
    with np.load(source, allow_pickle=True) as data:
        rows = data["trace_summaries"].tolist()
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["prompt_id"])].append(row)
    return {
        prompt_id: {
            key: float(np.mean([float(row[key]) for row in group]))
            for key in ("deepconf_global", "deepconf_tail_q20")
        }
        for prompt_id, group in grouped.items()
    }


def run_incremental_analysis(
    *,
    oof_csv: str,
    output_dir: str,
    model_label: str,
    dataset_label: str = "math500",
    layer: int | None = None,
    max_new_tokens: int | None = None,
    data_dir: str | None = None,
    expected_traces: int = 8,
    n_bootstrap: int = 1000,
    seed: int = 42,
    prompt_states_dir: str | None = None,
    prompt_states_layer: int | None = None,
    prompt_states_pca_dim: int = 32,
    exact_scores_npz: str | None = None,
) -> dict:
    rows = _read_oof(oof_csv)
    if not rows:
        raise ValueError(f"no OOF rows in {oof_csv}")
    layer = max(int(row["layer"]) for row in rows) if layer is None else int(layer)
    rows = [row for row in rows if int(row["layer"]) == layer]
    if not rows:
        raise ValueError(f"no rows at layer {layer} in {oof_csv}")
    states = None
    geometry = None
    exact_scores = _load_exact_prompt_scores(exact_scores_npz) if exact_scores_npz else None
    if prompt_states_dir:
        state_layer = layer if prompt_states_layer is None else int(prompt_states_layer)
        states = _load_prompt_states(prompt_states_dir, state_layer)
        base_features = aggregate_prompt_features(
            rows,
            max_new_tokens=max_new_tokens,
            data_dir=data_dir,
            expected_traces=expected_traces,
            exact_scores=exact_scores,
        )
        outcomes = {prompt_id: entry["outcome"] for prompt_id, entry in base_features.items()}
        folds = {prompt_id: int(entry["fold"]) for prompt_id, entry in base_features.items() if entry["fold"] is not None}
        geometry = crossfit_prompt_geometry_scores(
            states, outcomes, folds, pca_dim=prompt_states_pca_dim
        )
    features = aggregate_prompt_features(
        rows,
        max_new_tokens=max_new_tokens,
        data_dir=data_dir,
        expected_traces=expected_traces,
        prompt_geometry=geometry,
        exact_scores=exact_scores,
    )
    cap = resolve_cap(
        max_new_tokens,
        data_dir=data_dir,
        lengths=(_finite(row.get("trace_length")) for row in rows),
        context="run_incremental_analysis",
    )
    populations = _population_ids(features)
    specs = _model_specs(features)
    result = {
        "model": model_label,
        "dataset": dataset_label,
        "layer": layer,
        "target": {
            "name": "plurality_vote_correctness",
            "definition": "majority parseable answer equals gold; ties use highest log-probability",
            "all_unparsed": "automatic failure, retained only in full-population metrics",
        },
        "feature_definitions": {
            "B0": list(BASE_FEATURE_NAMES),
            "B1": list(BASE_FEATURE_NAMES + ("rmd_tail_q20",)),
            "prompt_only_geometry": "OOF negative Mahalanobis distance from row-zero prompt states to training correct prompt states",
            "cap_count": "number of sibling traces with trace_length >= max_new_tokens",
            "unparsed_count": "number of sibling traces without a parseable final answer",
            "deepconf_global": "raw DeepConf C = mean(-log p) over top-20 candidates, higher reported confidence",
            "deepconf_tail_q20": "raw DeepConf C averaged over the final 20% of generated tokens",
        },
        "max_new_tokens": cap.value,
        "cap_provenance": cap.provenance,
        "expected_traces": expected_traces,
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "exact_scores_npz": exact_scores_npz,
        "populations": {},
    }

    for population, prompt_ids in populations.items():
        prompt_ids = [prompt_id for prompt_id in prompt_ids if features[prompt_id]["fold"] is not None]
        if len(prompt_ids) < 2:
            continue
        y = np.asarray([features[prompt_id]["outcome"] for prompt_id in prompt_ids], dtype=float)
        folds = np.asarray([features[prompt_id]["fold"] for prompt_id in prompt_ids])
        entry = {
            "n_prompts": len(prompt_ids),
            "n_automatic_failures": int(sum(features[prompt_id]["automatic_failure"] for prompt_id in prompt_ids)),
            "n_capped_prompts": int(sum(features[prompt_id]["cap_count"] > 0 for prompt_id in prompt_ids)),
            "n_unparsed_traces": int(sum(features[prompt_id]["unparsed_count"] for prompt_id in prompt_ids)),
            "base_accuracy": float(np.mean(y)),
            "models": {},
            "paired_deltas": {},
        }
        predictions: dict[str, np.ndarray] = {}
        for name, columns in specs.items():
            x = np.asarray(
                [[features[prompt_id].get(column, np.nan) for column in columns] for prompt_id in prompt_ids],
                dtype=float,
            )
            predictions[name] = crossfit_logistic_predictions(x, y, folds, seed=seed)
            entry["models"][name] = {
                "features": list(columns),
                "metrics": prompt_metrics(predictions[name], y),
            }
        if population in {"full_population", "cap_free_full_population"}:
            for name, column in (
                ("dumb_cap_count", "cap_count"),
                ("dumb_unparsed_count", "unparsed_count"),
            ):
                x = -np.asarray([features[prompt_id][column] for prompt_id in prompt_ids], dtype=float)
                predictions[name] = crossfit_logistic_predictions(x[:, None], y, folds, seed=seed)
                entry["models"][name] = {
                    "features": [f"-{column}"],
                    "metrics": prompt_metrics(predictions[name], y),
                }
        if "B0" in predictions and "B1" in predictions:
            for left, right, label in (
                ("B1", "B0", "B1_minus_B0"),
                ("B1", "B0_plus_DeepConf_global", "B1_minus_B0_plus_DeepConf_global"),
                ("B1", "B0_plus_DeepConf_tail_q20", "B1_minus_B0_plus_DeepConf_tail_q20"),
            ):
                if right not in predictions:
                    continue
                for metric in METRIC_NAMES:
                    entry["paired_deltas"][label + "_" + metric] = paired_bootstrap_delta(
                        predictions[left], predictions[right], y,
                        metric=metric, n_bootstrap=n_bootstrap,
                        seed=seed + 1000 + len(metric) + len(label),
                    )
        if "B0_prompt_only_geometry" in predictions:
            for left, right in (
                ("B0_prompt_only_geometry", "B0"),
                ("B1_prompt_only_geometry", "B1"),
            ):
                for metric in METRIC_NAMES:
                    entry["paired_deltas"][f"{left}_minus_{right}_{metric}"] = paired_bootstrap_delta(
                        predictions[left], predictions[right], y,
                        metric=metric, n_bootstrap=n_bootstrap,
                        seed=seed + 2000 + len(metric),
                    )
        result["populations"][population] = entry

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset_label}_incremental_abstention"
    (output / f"{prefix}_results.json").write_text(json.dumps(result, indent=2))
    write_report(result, output / f"{prefix}_report.md")
    return result


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def _interval(entry: Mapping | None) -> str:
    if not entry or entry.get("point_estimate") is None:
        return "n/a"
    return (
        f"{_fmt(entry['point_estimate'])} [{_fmt(entry['ci_low'])}, "
        f"{_fmt(entry['ci_high'])}] p={_fmt(entry['p_two_sided'])}"
    )


def write_report(result: Mapping, path: str | Path) -> None:
    lines = [
        f"# Incremental abstention analysis — {result['model']} / {result['dataset']} (L{result['layer']})",
        "",
        "Brier and log loss are OOF probabilistic-forecast scores after logistic calibration; they do not isolate calibration from discrimination/resolution.",
        "",
        "`B0` = length + global entropy + global log-probability + vote agreement. `B1` adds tail RMD. All deltas below use paired prompt bootstrap.",
        "",
    ]
    for population, entry in result["populations"].items():
        lines += [
            f"## {population}",
            "",
            f"{entry['n_prompts']} prompts; base accuracy {_fmt(entry['base_accuracy'])}; automatic failures {entry['n_automatic_failures']}; capped prompts {entry['n_capped_prompts']}; unparsed traces {entry['n_unparsed_traces']}.",
            "",
            "| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, model in entry["models"].items():
            metrics = model["metrics"]
            lines.append(
                f"| {name} | {_fmt(metrics.get('auacc'))} | {_fmt(metrics.get('aurc'))} | "
                f"{_fmt(metrics.get('brier'))} | {_fmt(metrics.get('log_loss'))} |"
            )
        lines += ["", "### Paired increments", "", "| comparison | AUACC | AURC | Brier | log loss |", "| --- | --- | --- | --- | --- |"]
        for prefix in (
            "B1_minus_B0",
            "B1_minus_B0_plus_DeepConf_global",
            "B1_minus_B0_plus_DeepConf_tail_q20",
            "B0_prompt_only_geometry_minus_B0",
            "B1_prompt_only_geometry_minus_B1",
        ):
            if not any(key.startswith(prefix + "_") for key in entry["paired_deltas"]):
                continue
            lines.append(
                f"| {prefix} | "
                + " | ".join(
                    _interval(entry["paired_deltas"].get(f"{prefix}_{metric}"))
                    for metric in METRIC_NAMES
                )
                + " |"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--dataset_label", default="math500")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument(
        "--data_dir",
        default=None,
        help=(
            "Trace directory these OOF scores came from. Used to look the "
            "generation budget up in dvc.lock/params.yaml; without it "
            "--max_new_tokens cannot be validated."
        ),
    )
    parser.add_argument("--expected_traces", type=int, default=8)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_states_dir", default=None)
    parser.add_argument("--prompt_states_layer", type=int, default=None)
    parser.add_argument("--prompt_states_pca_dim", type=int, default=32)
    parser.add_argument("--exact_scores_npz", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_incremental_analysis(
        oof_csv=args.oof_csv,
        output_dir=args.output_dir,
        model_label=args.model_label,
        dataset_label=args.dataset_label,
        layer=args.layer,
        max_new_tokens=args.max_new_tokens,
        data_dir=args.data_dir,
        expected_traces=args.expected_traces,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        prompt_states_dir=args.prompt_states_dir,
        prompt_states_layer=args.prompt_states_layer,
        prompt_states_pca_dim=args.prompt_states_pca_dim,
        exact_scores_npz=args.exact_scores_npz,
    )
    for population, entry in result["populations"].items():
        delta = entry["paired_deltas"].get("B1_minus_B0_auacc")
        print(f"{population}: n={entry['n_prompts']} B1-B0 AUACC {_interval(delta)}")


if __name__ == "__main__":
    main()
