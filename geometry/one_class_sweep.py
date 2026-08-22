"""One-class PCA-dimension and covariance-estimator mechanism sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from analysis.analyze import detect_layers, load_all_traces


METHODS = (
    "centroid",
    "diagonal_mahalanobis",
    "empirical_ridge_mahalanobis",
    "ledoit_wolf_mahalanobis",
    "rmd_ledoit_wolf",
    "normalized_rmd_ledoit_wolf",
)


def _l2_normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms > eps, norms, 1.0)


def _precision_ledoit(centered: np.ndarray) -> np.ndarray:
    return LedoitWolf(assume_centered=True).fit(centered).precision_


def _precision_empirical_ridge(
    centered: np.ndarray, ridge_scale: float
) -> np.ndarray:
    covariance = np.atleast_2d(np.cov(centered, rowvar=False, bias=False))
    dimension = covariance.shape[0]
    scale = float(np.trace(covariance) / dimension)
    ridge = max(float(ridge_scale) * scale, np.finfo(float).eps)
    return np.linalg.pinv(covariance + ridge * np.eye(dimension))


def fit_projected_references(
    target_projected: np.ndarray,
    background_projected: np.ndarray,
    dimensions: list[int],
    ridge_scale: float,
) -> dict:
    target_projected = np.asarray(target_projected, dtype=np.float64)
    background_projected = np.asarray(background_projected, dtype=np.float64)
    if not dimensions:
        raise ValueError("At least one dimension is required")
    max_available = target_projected.shape[1]
    if max(dimensions) > max_available:
        raise ValueError(
            f"Requested dimension {max(dimensions)} exceeds PCA width {max_available}"
        )

    references = {}
    for dimension in dimensions:
        target = target_projected[:, :dimension]
        background = background_projected[:, :dimension]
        mu = target.mean(axis=0)
        centered = target - mu
        variance = np.var(centered, axis=0, ddof=1)
        variance_floor = max(
            float(np.mean(variance)) * 1e-8, np.finfo(float).eps
        )
        background_mu = background.mean(axis=0)
        references[str(dimension)] = {
            "mu": mu,
            "diagonal_precision": 1.0 / np.maximum(variance, variance_floor),
            "empirical_ridge_precision": _precision_empirical_ridge(
                centered, ridge_scale
            ),
            "ledoit_wolf_precision": _precision_ledoit(centered),
            "background_mu": background_mu,
            "background_ledoit_wolf_precision": _precision_ledoit(
                background - background_mu
            ),
        }
    return references


def _mahalanobis(
    projected: np.ndarray, mu: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    diff = projected - mu
    squared = np.sum((diff @ precision) * diff, axis=1)
    return np.sqrt(np.maximum(squared, 0.0))


def score_projected(
    projected: np.ndarray,
    reference: dict,
    method: str,
) -> float:
    projected = np.asarray(projected, dtype=np.float64)
    dimension = len(reference["mu"])
    projected = projected[:, :dimension]
    diff = projected - reference["mu"]
    if method == "centroid":
        distances = np.linalg.norm(diff, axis=1)
    elif method == "diagonal_mahalanobis":
        distances = np.sqrt(
            np.maximum(
                np.sum(diff**2 * reference["diagonal_precision"], axis=1),
                0.0,
            )
        )
    elif method == "empirical_ridge_mahalanobis":
        distances = _mahalanobis(
            projected,
            reference["mu"],
            reference["empirical_ridge_precision"],
        )
    elif method == "ledoit_wolf_mahalanobis":
        distances = _mahalanobis(
            projected,
            reference["mu"],
            reference["ledoit_wolf_precision"],
        )
    elif method in {"rmd_ledoit_wolf", "normalized_rmd_ledoit_wolf"}:
        target = _mahalanobis(
            projected,
            reference["mu"],
            reference["ledoit_wolf_precision"],
        )
        background = _mahalanobis(
            projected,
            reference["background_mu"],
            reference["background_ledoit_wolf_precision"],
        )
        distances = target - background
    else:
        raise ValueError(f"Unknown one-class method: {method}")
    return -float(np.mean(distances))


def _concatenate_tokens(
    traces: list[dict],
    layer: int,
    normalize: bool,
) -> np.ndarray:
    values = np.concatenate(
        [np.asarray(trace["hiddens"][layer], dtype=np.float64) for trace in traces],
        axis=0,
    )
    return _l2_normalize_rows(values) if normalize else values


def fit_fold_references(
    correct_traces: list[dict],
    background_traces: list[dict],
    layer: int,
    dimensions: list[int],
    ridge_scale: float,
) -> dict:
    max_dimension = max(dimensions)
    raw_target = _concatenate_tokens(correct_traces, layer, normalize=False)
    raw_background = _concatenate_tokens(
        background_traces, layer, normalize=False
    )
    normalized_target = _concatenate_tokens(
        correct_traces, layer, normalize=True
    )
    normalized_background = _concatenate_tokens(
        background_traces, layer, normalize=True
    )
    for family, values in (
        ("raw", raw_target),
        ("normalized", normalized_target),
    ):
        if min(values.shape) < max_dimension:
            raise ValueError(
                f"{family} target matrix cannot support PCA dimension {max_dimension}"
            )

    solver = "randomized" if raw_target.shape[0] > 200_000 else "full"
    raw_pca = PCA(
        n_components=max_dimension, random_state=42, svd_solver=solver
    ).fit(raw_target)
    normalized_solver = (
        "randomized" if normalized_target.shape[0] > 200_000 else "full"
    )
    normalized_pca = PCA(
        n_components=max_dimension,
        random_state=42,
        svd_solver=normalized_solver,
    ).fit(normalized_target)

    raw_references = fit_projected_references(
        raw_pca.transform(raw_target),
        raw_pca.transform(raw_background),
        dimensions,
        ridge_scale,
    )
    normalized_references = fit_projected_references(
        normalized_pca.transform(normalized_target),
        normalized_pca.transform(normalized_background),
        dimensions,
        ridge_scale,
    )
    return {
        "dimensions": [int(value) for value in dimensions],
        "raw_pca": raw_pca,
        "normalized_pca": normalized_pca,
        "raw_references": raw_references,
        "normalized_references": normalized_references,
    }


def score_trace_references(
    trace: dict,
    layer: int,
    references: dict,
) -> dict:
    hiddens = np.asarray(trace["hiddens"][layer], dtype=np.float64)
    raw_projected = references["raw_pca"].transform(hiddens)
    normalized_projected = references["normalized_pca"].transform(
        _l2_normalize_rows(hiddens)
    )
    scores = {}
    for dimension in references["dimensions"]:
        key = str(dimension)
        scores[key] = {}
        for method in METHODS:
            if method == "normalized_rmd_ledoit_wolf":
                projected = normalized_projected
                reference = references["normalized_references"][key]
            else:
                projected = raw_projected
                reference = references["raw_references"][key]
            scores[key][method] = score_projected(
                projected, reference, method
            )
    return scores


def _pack_metrics(entries: list[tuple[float, int, int]]) -> dict:
    scores = np.asarray([entry[0] for entry in entries], dtype=float)
    labels = np.asarray([entry[1] for entry in entries], dtype=int)
    folds = np.asarray([entry[2] for entry in entries], dtype=int)
    fold_roc = []
    fold_pr = []
    for fold in sorted(set(folds.tolist())):
        mask = folds == fold
        if len(np.unique(labels[mask])) < 2:
            continue
        fold_roc.append(float(roc_auc_score(labels[mask], scores[mask])))
        fold_pr.append(float(average_precision_score(labels[mask], scores[mask])))
    return {
        "pooled_roc_auc": float(roc_auc_score(labels, scores)),
        "pooled_pr_auc": float(average_precision_score(labels, scores)),
        "fold_roc_auc_mean": float(np.mean(fold_roc)),
        "fold_roc_auc_std": float(np.std(fold_roc)),
        "fold_pr_auc_mean": float(np.mean(fold_pr)),
        "fold_pr_auc_std": float(np.std(fold_pr)),
        "fold_roc_aucs": fold_roc,
        "fold_pr_aucs": fold_pr,
        "n_eval": len(entries),
        "n_correct": int(labels.sum()),
        "n_incorrect": int(len(labels) - labels.sum()),
    }


def evaluate_one_class_sweep(
    traces: list[dict],
    model: str,
    dataset: str,
    layers: list[int],
    dimensions: list[int],
    n_splits: int,
    seed: int,
    ridge_scale: float,
    reference_fitter: Callable = fit_fold_references,
    trace_scorer: Callable = score_trace_references,
) -> dict:
    labels = np.asarray([int(trace["is_correct"]) for trace in traces])
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    folds = list(splitter.split(np.arange(len(traces)), labels))
    accumulators = {
        str(layer): {
            str(dimension): {method: [] for method in METHODS}
            for dimension in dimensions
        }
        for layer in layers
    }

    for fold_index, (train_idx, test_idx) in enumerate(folds):
        background = [traces[index] for index in train_idx]
        correct = [trace for trace in background if trace["is_correct"]]
        for layer in layers:
            references = reference_fitter(
                correct,
                background,
                layer,
                dimensions,
                ridge_scale,
            )
            for index in test_idx:
                scores = trace_scorer(traces[index], layer, references)
                for dimension in dimensions:
                    for method in METHODS:
                        accumulators[str(layer)][str(dimension)][method].append(
                            (
                                float(scores[str(dimension)][method]),
                                int(labels[index]),
                                int(fold_index),
                            )
                        )

    return {
        "model": model,
        "dataset": dataset,
        "settings": {
            "layers": [int(value) for value in layers],
            "dimensions": [int(value) for value in dimensions],
            "n_splits": int(n_splits),
            "seed": int(seed),
            "ridge_scale": float(ridge_scale),
            "target_fit": "correct training traces only",
            "background_fit": "all training traces without using negative labels",
            "no_dimension_selection": True,
        },
        "layers": {
            layer: {
                "dimensions": {
                    dimension: {
                        "methods": {
                            method: _pack_metrics(entries)
                            for method, entries in methods.items()
                        }
                    }
                    for dimension, methods in dimensions_data.items()
                }
            }
            for layer, dimensions_data in accumulators.items()
        },
    }


def write_markdown(result: dict, path: str | Path) -> None:
    lines = [
        f"# {result['model']} {result['dataset']} one-class mechanism sweep",
        "",
        "All dimensions are reported; no post-hoc dimension is selected.",
        "",
        "| Layer | Dimension | Method | Pooled ROC | Fold ROC | Pooled PR | N |",
        "|---:|---:|:---|---:|:---|---:|---:|",
    ]
    for layer, layer_result in result["layers"].items():
        for dimension, dimension_result in layer_result["dimensions"].items():
            for method, metrics in dimension_result["methods"].items():
                lines.append(
                    f"| {layer} | {dimension} | {method} "
                    f"| {metrics['pooled_roc_auc']:.3f} "
                    f"| {metrics['fold_roc_auc_mean']:.3f} ± "
                    f"{metrics['fold_roc_auc_std']:.3f} "
                    f"| {metrics['pooled_pr_auc']:.3f} | {metrics['n_eval']} |"
                )
    Path(path).write_text("\n".join(lines) + "\n")


def parse_int_list(raw: str) -> list[int]:
    return sorted({int(value.strip()) for value in raw.split(",") if value.strip()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_label", default="math500")
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--dimensions", required=True)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge_scale", type=float, default=1e-3)
    parser.add_argument(
        "--exclude_unparsed",
        action="store_true",
        help="Drop traces with no parseable final answer (truncated/non-terminating, "
        "auto-labeled incorrect upstream). Use to check whether the low-dimensional "
        "plateau survives once the 'incorrect' class is genuine wrong answers, not "
        "non-answers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = parse_int_list(args.layers)
    dimensions = parse_int_list(args.dimensions)
    if not layers:
        layers = detect_layers(args.data_dir)
    traces = load_all_traces(
        args.data_dir,
        layers,
        include_auxiliary=False,
    )
    dataset_label = args.dataset_label
    if args.exclude_unparsed:
        before = len(traces)
        traces = [
            t for t in traces
            if (t.get("predicted_answer") is not None
                and str(t["predicted_answer"]).strip() != "")
        ]
        dataset_label = f"{args.dataset_label}_parseable"
        print(
            f"Excluded {before - len(traces)} unparsed traces; {len(traces)} parseable "
            f"remain ({sum(t['is_correct'] for t in traces)} correct)",
            flush=True,
        )
    result = evaluate_one_class_sweep(
        traces,
        model=args.model_label,
        dataset=dataset_label,
        layers=layers,
        dimensions=dimensions,
        n_splits=args.n_splits,
        seed=args.seed,
        ridge_scale=args.ridge_scale,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset_label}_one_class_sweep"
    (output_dir / f"{prefix}_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_markdown(result, output_dir / f"{prefix}_report.md")


if __name__ == "__main__":
    main()
