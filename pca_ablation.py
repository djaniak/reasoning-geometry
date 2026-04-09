"""
pca_ablation.py - Aggregate per-dimension base analyses into one PCA sweep report.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MAX_PCA_DIM_ALIASES = {"max", "all"}


def normalize_pca_dim_label(raw_dim: object) -> str:
    token = str(raw_dim).strip().lower()
    if token in MAX_PCA_DIM_ALIASES:
        return "max"
    dim = int(token)
    if dim <= 0:
        raise ValueError("PCA dimensions must be positive integers or max/all")
    return str(dim)


def pca_dim_label_sort_key(label: str) -> tuple[int, int]:
    if label == "max":
        return (1, 0)
    return (0, int(label))


def pca_dim_label_to_output(label: str) -> int | str:
    return "max" if label == "max" else int(label)


def parse_pca_dims(raw_dims: str) -> list[str]:
    dims = []
    seen = set()
    for part in raw_dims.split(","):
        if not part.strip():
            continue
        label = normalize_pca_dim_label(part)
        if label in seen:
            continue
        seen.add(label)
        dims.append(label)
    if not dims:
        raise ValueError("No valid PCA dimensions were provided")
    return dims


def _load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _extract_dim(payload: dict, path: Path) -> str:
    settings = payload.get("settings", {})
    pca_dim = settings.get("pca_dim")
    if pca_dim is None:
        raise ValueError(f"Missing settings.pca_dim in {path}")
    return normalize_pca_dim_label(pca_dim)


def _extract_layer_metrics(layer_payload: dict, layer: str, dim: str) -> dict:
    if "mahalanobis_only" not in layer_payload or "combined" not in layer_payload:
        raise ValueError(f"Layer {layer} in dim={dim} is missing base metrics")

    entry = {
        "mahalanobis_only": {
            "roc_auc_mean": float(layer_payload["mahalanobis_only"]["roc_auc_mean"]),
            "roc_auc_std": float(layer_payload["mahalanobis_only"]["roc_auc_std"]),
        },
        "combined": {
            "roc_auc_mean": float(layer_payload["combined"]["roc_auc_mean"]),
            "roc_auc_std": float(layer_payload["combined"]["roc_auc_std"]),
        },
        "delta_vs_entropy": float(layer_payload["delta_vs_entropy"]),
    }
    if "length_controlled_delta" in layer_payload:
        entry["length_controlled_delta"] = float(layer_payload["length_controlled_delta"])
    return entry


def aggregate_pca_ablation(
    input_paths: list[str | Path],
    expected_dims: list[str | int] | None = None,
    model: str | None = None,
    dataset: str | None = None,
) -> dict:
    if not input_paths:
        raise ValueError("No input result files were provided")

    by_dim: dict[str, dict] = {}
    dataset_name: str | None = dataset
    n_correct: int | None = None
    n_incorrect: int | None = None
    cv_random_state: int | None = None
    entropy_by_dim: dict[str, dict] = {}
    source_files: dict[str, str] = {}

    for raw_path in input_paths:
        path = Path(raw_path)
        payload = _load_json(path)
        dim = _extract_dim(payload, path)
        if dim in by_dim:
            raise ValueError(f"Duplicate PCA dimension {dim} in inputs")

        current_dataset = payload.get("dataset")
        if dataset_name is None:
            dataset_name = current_dataset
        elif current_dataset != dataset_name:
            raise ValueError(
                f"Mismatched dataset in {path}: {current_dataset!r} != {dataset_name!r}"
            )

        current_n_correct = int(payload.get("n_correct", -1))
        current_n_incorrect = int(payload.get("n_incorrect", -1))
        if n_correct is None:
            n_correct = current_n_correct
            n_incorrect = current_n_incorrect
        elif current_n_correct != n_correct or current_n_incorrect != n_incorrect:
            raise ValueError(
                f"Mismatched class counts in {path}: "
                f"({current_n_correct}, {current_n_incorrect}) != ({n_correct}, {n_incorrect})"
            )

        current_cv_state = payload.get("settings", {}).get("cv_random_state")
        if cv_random_state is None:
            cv_random_state = int(current_cv_state) if current_cv_state is not None else None
        elif current_cv_state is not None and int(current_cv_state) != cv_random_state:
            raise ValueError(
                f"Mismatched cv_random_state in {path}: {current_cv_state} != {cv_random_state}"
            )

        if "layers" not in payload or not isinstance(payload["layers"], dict):
            raise ValueError(f"Missing layers map in {path}")

        by_dim[dim] = payload
        entropy_by_dim[dim] = payload.get("entropy_baseline", {})
        source_files[dim] = str(path)

    if dataset_name is None:
        raise ValueError("Failed to resolve dataset name from inputs")
    if n_correct is None or n_incorrect is None:
        raise ValueError("Failed to resolve class counts from inputs")

    normalized_expected_dims = None
    if expected_dims is not None:
        normalized_expected_dims = []
        seen_expected = set()
        for raw_dim in expected_dims:
            label = normalize_pca_dim_label(raw_dim)
            if label in seen_expected:
                continue
            seen_expected.add(label)
            normalized_expected_dims.append(label)

    found_dims = sorted(by_dim, key=pca_dim_label_sort_key)
    if normalized_expected_dims is not None:
        missing = sorted(set(normalized_expected_dims) - set(found_dims), key=pca_dim_label_sort_key)
        extra = sorted(set(found_dims) - set(normalized_expected_dims), key=pca_dim_label_sort_key)
        if missing:
            raise ValueError(f"Missing expected PCA dimensions: {missing}")
        if extra:
            raise ValueError(f"Found unexpected PCA dimensions: {extra}")
        dims = list(normalized_expected_dims)
    else:
        dims = found_dims

    layer_keys = set()
    for payload in by_dim.values():
        layer_keys.update(payload["layers"].keys())
    sorted_layers = sorted(layer_keys, key=int)

    layers_out = {}
    for layer in sorted_layers:
        layer_by_dim = {}
        combined_by_dim: dict[str, float] = {}
        for dim in dims:
            layer_payload = by_dim[dim]["layers"].get(layer)
            if layer_payload is None:
                raise ValueError(f"Layer {layer} is missing for dim={dim}")
            metrics = _extract_layer_metrics(layer_payload, layer=layer, dim=dim)
            layer_by_dim[dim] = metrics
            combined_by_dim[dim] = metrics["combined"]["roc_auc_mean"]

        combined_curve = [combined_by_dim[dim] for dim in dims]
        is_monotone = all(
            combined_curve[i] <= combined_curve[i + 1] + 1e-12
            for i in range(len(combined_curve) - 1)
        )
        best_dim = max(dims, key=lambda dim: combined_by_dim[dim])

        layers_out[layer] = {
            "dims": layer_by_dim,
            "best_dim_by_combined_auc": pca_dim_label_to_output(best_dim),
            "combined_auc_monotone_non_decreasing": bool(is_monotone),
        }

    return {
        "dataset": dataset_name,
        "model": model,
        "n_correct": int(n_correct),
        "n_incorrect": int(n_incorrect),
        "settings": {
            "analysis_family": "base",
            "pca_dims": [pca_dim_label_to_output(dim) for dim in dims],
            "cv_random_state": cv_random_state,
            "expected_dims": [
                pca_dim_label_to_output(dim)
                for dim in (normalized_expected_dims if normalized_expected_dims is not None else dims)
            ],
        },
        "entropy_baseline_by_dim": entropy_by_dim,
        "layers": layers_out,
        "source_files": source_files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate PCA-dimension ablation outputs")
    parser.add_argument("--inputs", nargs="+", required=True, help="Per-dim base result JSON files")
    parser.add_argument("--output_path", required=True, help="Path to aggregated output JSON")
    parser.add_argument(
        "--expected_dims",
        default=None,
        help="Optional comma-separated PCA dims to enforce (e.g. 32,128,512,max)",
    )
    parser.add_argument("--model", default=None, help="Optional model label for output metadata")
    parser.add_argument("--dataset", default=None, help="Optional dataset label for output metadata")
    return parser.parse_args()


def main():
    args = parse_args()
    expected_dims = parse_pca_dims(args.expected_dims) if args.expected_dims else None
    aggregated = aggregate_pca_ablation(
        args.inputs,
        expected_dims=expected_dims,
        model=args.model,
        dataset=args.dataset,
    )
    output_path = Path(args.output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(aggregated, fh, indent=2)
    print(f"PCA ablation summary written to {output_path}")


if __name__ == "__main__":
    main()
