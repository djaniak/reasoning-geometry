from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def repo_root(start: Path | None = None) -> Path:
    candidate = (start or Path.cwd()).resolve()
    for path in [candidate, *candidate.parents]:
        if (path / ".git").exists():
            return path
    return candidate


def results_root(start: Path | None = None) -> Path:
    return repo_root(start) / "results"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def load_all_results(start: Path | None = None) -> dict[str, Any]:
    return load_json(results_root(start) / "all_results.json")


def load_result_json(model: str, dataset: str, *, cross: bool = False, start: Path | None = None) -> dict[str, Any]:
    dataset_dir = f"{dataset}_cross" if cross else dataset
    file_path = results_root(start) / model / dataset_dir / f"{dataset}_results.json"
    return load_json(file_path)


def sort_layers(layer_values: list[int] | tuple[int, ...] | set[int]) -> list[int]:
    return sorted(layer_values)


def condition_label(model: str, dataset: str) -> str:
    pretty_model = model.replace("_", " ")
    return f"{pretty_model} / {dataset}"


def collect_layer_rows(all_results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, model_payload in all_results.items():
        if not isinstance(model_payload, dict):
            continue
        for dataset_key, run in model_payload.items():
            if not isinstance(run, dict) or dataset_key.endswith("_cross"):
                continue
            layers = run.get("layers", {})
            entropy = run.get("entropy_baseline", {}).get("roc_auc_mean")
            if not isinstance(layers, dict):
                continue
            for layer_key, layer_data in layers.items():
                if not isinstance(layer_data, dict):
                    continue
                try:
                    layer = int(layer_key)
                except ValueError:
                    continue
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset_key,
                        "layer": layer,
                        "entropy_auc": float(entropy) if entropy is not None else float("nan"),
                        "mahal_auc": float(layer_data.get("mahalanobis_only", {}).get("roc_auc_mean", float("nan"))),
                        "combined_auc": float(layer_data.get("combined", {}).get("roc_auc_mean", float("nan"))),
                        "delta_raw": float(layer_data.get("delta_vs_entropy", float("nan"))),
                        "delta_len_ctrl": float(layer_data.get("length_controlled_delta", float("nan"))),
                        "confident_wrong_p": float(
                            layer_data.get("confident_wrong", {}).get("mannwhitney_pvalue", float("nan"))
                        ),
                        "n_correct": int(run.get("n_correct", 0)),
                        "n_incorrect": int(run.get("n_incorrect", 0)),
                    }
                )
    return rows


def collect_best_layer_rows(layer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in layer_rows:
        key = (row["model"], row["dataset"])
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        if row["combined_auc"] > current["combined_auc"]:
            best[key] = row
        elif row["combined_auc"] == current["combined_auc"] and row["layer"] < current["layer"]:
            best[key] = row
    return list(best.values())


def collect_cross_rows(all_results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, model_payload in all_results.items():
        if not isinstance(model_payload, dict):
            continue
        for dataset_key, run in model_payload.items():
            if not isinstance(run, dict) or not dataset_key.endswith("_cross"):
                continue
            dataset = dataset_key[: -len("_cross")]
            base_run = model_payload.get(dataset)
            if not isinstance(base_run, dict):
                continue
            base_layers = base_run.get("layers", {})
            transfers = run.get("cross_model_transfer", {})
            if not isinstance(base_layers, dict) or not isinstance(transfers, dict):
                continue
            for layer_key, transfer in transfers.items():
                if not isinstance(transfer, dict):
                    continue
                base_layer = base_layers.get(layer_key, {})
                native = base_layer.get("mahalanobis_only", {}).get("roc_auc_mean")
                if native is None:
                    continue
                try:
                    native_value = float(native)
                    cross_value = float(transfer.get("cross_mahal_only", {}).get("roc_auc_mean", float("nan")))
                    cross_combined = float(transfer.get("cross_combined", {}).get("roc_auc_mean", float("nan")))
                    clf_mahal = float(transfer.get("clf_transfer_mahal_only", {}).get("roc_auc", float("nan")))
                    clf_combined = float(transfer.get("clf_transfer_combined", {}).get("roc_auc", float("nan")))
                    layer = int(layer_key)
                except ValueError:
                    continue
                transfer_pct = (cross_value / native_value * 100.0) if native_value else float("nan")
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "layer": layer,
                        "native_mahal_auc": native_value,
                        "cross_mahal_auc": cross_value,
                        "cross_combined_auc": cross_combined,
                        "clf_transfer_mahal_auc": clf_mahal,
                        "clf_transfer_combined_auc": clf_combined,
                        "transfer_pct": transfer_pct,
                    }
                )
    return rows


def collect_difficulty_rows(model: str, record: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    strata = record.get("difficulty_stratification", {})
    if not isinstance(strata, dict):
        return rows
    for bucket, values in strata.items():
        if not isinstance(values, dict):
            continue
        bootstrap = values.get("bootstrap_ci", {})
        entropy_obj = values.get("entropy_only", float("nan"))
        mahal_obj = values.get("mahalanobis_only", float("nan"))
        combined_obj = values.get("combined", float("nan"))
        entropy_auc = float(entropy_obj.get("roc_auc_mean", float("nan"))) if isinstance(entropy_obj, dict) else float(entropy_obj)
        mahal_auc = float(mahal_obj.get("roc_auc_mean", float("nan"))) if isinstance(mahal_obj, dict) else float(mahal_obj)
        combined_auc = (
            float(combined_obj.get("roc_auc_mean", float("nan"))) if isinstance(combined_obj, dict) else float(combined_obj)
        )
        rows.append(
            {
                "model": model,
                "bucket": bucket,
                "n_total": int(values.get("n_total", 0)),
                "n_correct": int(values.get("n_correct", 0)),
                "n_incorrect": int(values.get("n_incorrect", 0)),
                "entropy_auc": entropy_auc,
                "mahal_auc": mahal_auc,
                "combined_auc": combined_auc,
                "delta": float(values.get("delta", float("nan"))),
                "entropy_ci_low": float(bootstrap.get("entropy_ci95", [float("nan"), float("nan")])[0]),
                "entropy_ci_high": float(bootstrap.get("entropy_ci95", [float("nan"), float("nan")])[1]),
                "combined_ci_low": float(bootstrap.get("combined_ci95", [float("nan"), float("nan")])[0]),
                "combined_ci_high": float(bootstrap.get("combined_ci95", [float("nan"), float("nan")])[1]),
                "bootstrap_n_valid": int(bootstrap.get("n_valid", 0)),
            }
        )
    return rows


def collect_subject_rows(model: str, record: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    strata = record.get("subject_stratification", {})
    if not isinstance(strata, dict):
        return rows
    for subject, values in strata.items():
        if not isinstance(values, dict):
            continue
        bootstrap = values.get("bootstrap_ci", {})
        entropy_obj = values.get("entropy_only", float("nan"))
        mahal_obj = values.get("mahalanobis_only", float("nan"))
        combined_obj = values.get("combined", float("nan"))
        entropy_auc = float(entropy_obj.get("roc_auc_mean", float("nan"))) if isinstance(entropy_obj, dict) else float(entropy_obj)
        mahal_auc = float(mahal_obj.get("roc_auc_mean", float("nan"))) if isinstance(mahal_obj, dict) else float(mahal_obj)
        combined_auc = (
            float(combined_obj.get("roc_auc_mean", float("nan"))) if isinstance(combined_obj, dict) else float(combined_obj)
        )
        rows.append(
            {
                "model": model,
                "subject": subject,
                "n_total": int(values.get("n_total", 0)),
                "n_correct": int(values.get("n_correct", 0)),
                "n_incorrect": int(values.get("n_incorrect", 0)),
                "entropy_auc": entropy_auc,
                "mahal_auc": mahal_auc,
                "combined_auc": combined_auc,
                "delta": float(values.get("delta", float("nan"))),
                "entropy_ci_low": float(bootstrap.get("entropy_ci95", [float("nan"), float("nan")])[0]),
                "entropy_ci_high": float(bootstrap.get("entropy_ci95", [float("nan"), float("nan")])[1]),
                "combined_ci_low": float(bootstrap.get("combined_ci95", [float("nan"), float("nan")])[0]),
                "combined_ci_high": float(bootstrap.get("combined_ci95", [float("nan"), float("nan")])[1]),
                "bootstrap_n_valid": int(bootstrap.get("n_valid", 0)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = sorted(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
