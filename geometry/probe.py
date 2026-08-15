from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import shutil
import tempfile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

print("[trajectory_probe] importing analyze helpers...", flush=True)

from analysis.analyze import (
    build_feature_matrix,
    compute_mahal_distances,
    detect_layers,
    fit_mahalanobis_reference_safe,
    load_all_traces,
)
from data.trajectory_encoders import FunctionalPCAEncoder, LinearGaussianSequenceModel
from data.trajectory_preprocessing import resample_1d_sequence, stack_trace_channels

CV_RANDOM_STATE = 42
CV_SPLITS = 5
DEFAULT_METHODS = ("fpca_mahal", "fpca_combined", "probseq_joint")


def log_progress(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[trajectory_probe][{stamp}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_label", default="math500")
    parser.add_argument("--layers", default=None)
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--target_len", type=int, default=64)
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional limit on number of .npz files to load from data_dir for smoke tests",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods to run",
    )
    return parser.parse_args()


def parse_layers_arg(raw_layers: str | None, data_dir: str) -> list[int]:
    if raw_layers is None:
        return detect_layers(data_dir)
    layers = [int(part.strip()) for part in raw_layers.split(",") if part.strip()]
    if not layers:
        raise ValueError("No valid layers were provided")
    return layers


def parse_methods_arg(raw_methods: str) -> list[str]:
    methods = [part.strip() for part in raw_methods.split(",") if part.strip()]
    if not methods:
        raise ValueError("No methods were provided")
    invalid = sorted(set(methods) - set(DEFAULT_METHODS))
    if invalid:
        raise ValueError(f"Unsupported methods: {', '.join(invalid)}")
    return methods


def prepare_subset_data_dir(data_dir: str, max_files: int | None):
    if max_files is None:
        return data_dir, None
    if max_files <= 0:
        raise ValueError("--max_files must be positive")

    npz_files = sorted(fname for fname in os.listdir(data_dir) if fname.endswith(".npz"))
    if not npz_files:
        raise RuntimeError(f"No .npz files found in {data_dir}")

    temp_dir = tempfile.TemporaryDirectory(prefix="trajectory_probe_subset_")
    for fname in npz_files[:max_files]:
        src = os.path.join(data_dir, fname)
        dst = os.path.join(temp_dir.name, fname)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return temp_dir.name, temp_dir


def pack_metric_dict(fold_roc_aucs: list[float], fold_pr_aucs: list[float]) -> dict:
    if not fold_roc_aucs:
        return {
            "roc_auc_mean": None,
            "roc_auc_std": None,
            "pr_auc_mean": None,
            "pr_auc_std": None,
            "fold_roc_aucs": [],
            "fold_pr_aucs": [],
            "n_valid_folds": 0,
        }
    return {
        "roc_auc_mean": float(np.mean(fold_roc_aucs)),
        "roc_auc_std": float(np.std(fold_roc_aucs)),
        "pr_auc_mean": float(np.mean(fold_pr_aucs)),
        "pr_auc_std": float(np.std(fold_pr_aucs)),
        "fold_roc_aucs": [float(v) for v in fold_roc_aucs],
        "fold_pr_aucs": [float(v) for v in fold_pr_aucs],
        "n_valid_folds": len(fold_roc_aucs),
    }


def fit_balanced_logreg_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, random_state=CV_RANDOM_STATE, class_weight="balanced")
    clf.fit(X_train_scaled, y_train)
    return clf.predict_proba(X_test_scaled)[:, 1]


def score_auc_metrics(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float] | None:
    if len(np.unique(y_true)) < 2:
        return None
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    return float(roc_auc), float(pr_auc)


def build_fold_local_sequences(
    traces: list[dict],
    indices: np.ndarray,
    layer: int,
    pca,
    mu: np.ndarray,
    cov_inv: np.ndarray,
    target_len: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    mahal_only = []
    combined = []
    probseq = []

    for idx in indices:
        trace = traces[int(idx)]
        mahal = compute_mahal_distances(trace["hiddens"][layer], pca, mu, cov_inv)
        mahal_resampled = resample_1d_sequence(mahal, target_len)
        combined_resampled = stack_trace_channels(
            entropies=trace["entropies"],
            mahal_distances=mahal,
            target_len=target_len,
            include_relpos=False,
        )
        mahal_only.append(mahal_resampled[:, None])
        combined.append(combined_resampled)
        probseq.append(combined_resampled)

    return np.asarray(mahal_only), np.asarray(combined), probseq


def evaluate_foldwise_trajectory_models(
    traces: list[dict],
    layers: list[int],
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    pca_dim: int,
    target_len: int,
    methods: list[str],
) -> dict:
    results: dict[str, dict[str, dict]] = {}
    n_folds = len(fold_indices)

    for layer in layers:
        layer_results = {method: {"fold_roc_aucs": [], "fold_pr_aucs": []} for method in methods}
        log_progress(f"layer_start layer={layer}")

        for fold_idx, (train_idx, test_idx) in enumerate(fold_indices, start=1):
            log_progress(
                f"fold_start layer={layer} fold={fold_idx}/{n_folds} "
                f"n_train={len(train_idx)} n_test={len(test_idx)}"
            )
            correct_train = [traces[i] for i in train_idx if traces[i]["is_correct"]]
            ref = fit_mahalanobis_reference_safe(correct_train, layer, pca_dim)
            if ref is None:
                log_progress(
                    f"fold_skip layer={layer} fold={fold_idx}/{n_folds} reason=ref_fit_failed"
                )
                continue
            pca, mu, cov_inv = ref
            log_progress(f"ref_fit_done layer={layer} fold={fold_idx}/{n_folds}")

            train_mahal, train_combined, train_probseq = build_fold_local_sequences(
                traces=traces,
                indices=train_idx,
                layer=layer,
                pca=pca,
                mu=mu,
                cov_inv=cov_inv,
                target_len=target_len,
            )
            test_mahal, test_combined, test_probseq = build_fold_local_sequences(
                traces=traces,
                indices=test_idx,
                layer=layer,
                pca=pca,
                mu=mu,
                cov_inv=cov_inv,
                target_len=target_len,
            )
            log_progress(f"sequence_build_done layer={layer} fold={fold_idx}/{n_folds}")

            if "fpca_mahal" in methods:
                log_progress(f"method_start layer={layer} fold={fold_idx}/{n_folds} method=fpca_mahal")
                n_components = min(
                    pca_dim,
                    train_mahal.shape[0],
                    train_mahal.shape[1] * train_mahal.shape[2],
                )
                encoder = FunctionalPCAEncoder(n_components=n_components, random_state=CV_RANDOM_STATE)
                X_train = encoder.fit_transform(train_mahal)
                X_test = encoder.transform(test_mahal)
                scores = fit_balanced_logreg_scores(X_train, y[train_idx], X_test)
                metrics = score_auc_metrics(y[test_idx], scores)
                if metrics is not None:
                    layer_results["fpca_mahal"]["fold_roc_aucs"].append(metrics[0])
                    layer_results["fpca_mahal"]["fold_pr_aucs"].append(metrics[1])
                    log_progress(
                        f"method_done layer={layer} fold={fold_idx}/{n_folds} method=fpca_mahal "
                        f"roc_auc={metrics[0]:.4f} pr_auc={metrics[1]:.4f}"
                    )
                else:
                    log_progress(
                        f"method_skip layer={layer} fold={fold_idx}/{n_folds} method=fpca_mahal "
                        f"reason=single_class_test_fold"
                    )

            if "fpca_combined" in methods:
                log_progress(
                    f"method_start layer={layer} fold={fold_idx}/{n_folds} method=fpca_combined"
                )
                n_components = min(
                    pca_dim,
                    train_combined.shape[0],
                    train_combined.shape[1] * train_combined.shape[2],
                )
                encoder = FunctionalPCAEncoder(n_components=n_components, random_state=CV_RANDOM_STATE)
                X_train = encoder.fit_transform(train_combined)
                X_test = encoder.transform(test_combined)
                scores = fit_balanced_logreg_scores(X_train, y[train_idx], X_test)
                metrics = score_auc_metrics(y[test_idx], scores)
                if metrics is not None:
                    layer_results["fpca_combined"]["fold_roc_aucs"].append(metrics[0])
                    layer_results["fpca_combined"]["fold_pr_aucs"].append(metrics[1])
                    log_progress(
                        f"method_done layer={layer} fold={fold_idx}/{n_folds} method=fpca_combined "
                        f"roc_auc={metrics[0]:.4f} pr_auc={metrics[1]:.4f}"
                    )
                else:
                    log_progress(
                        f"method_skip layer={layer} fold={fold_idx}/{n_folds} method=fpca_combined "
                        f"reason=single_class_test_fold"
                    )

            if "probseq_joint" in methods:
                log_progress(
                    f"method_start layer={layer} fold={fold_idx}/{n_folds} method=probseq_joint"
                )
                correct_probseq_train = [
                    train_probseq[i]
                    for i in range(len(train_probseq))
                    if y[train_idx][i] == 1
                ]
                if correct_probseq_train:
                    model = LinearGaussianSequenceModel()
                    model.fit(correct_probseq_train)
                    scores = model.score_sequences(test_probseq)
                    metrics = score_auc_metrics(y[test_idx], scores)
                    if metrics is not None:
                        layer_results["probseq_joint"]["fold_roc_aucs"].append(metrics[0])
                        layer_results["probseq_joint"]["fold_pr_aucs"].append(metrics[1])
                        log_progress(
                            f"method_done layer={layer} fold={fold_idx}/{n_folds} method=probseq_joint "
                            f"roc_auc={metrics[0]:.4f} pr_auc={metrics[1]:.4f}"
                        )
                    else:
                        log_progress(
                            f"method_skip layer={layer} fold={fold_idx}/{n_folds} method=probseq_joint "
                            f"reason=single_class_test_fold"
                        )
                else:
                    log_progress(
                        f"method_skip layer={layer} fold={fold_idx}/{n_folds} method=probseq_joint "
                        f"reason=no_correct_train_traces"
                    )

        results[str(layer)] = {
            method: pack_metric_dict(
                fold_roc_aucs=layer_results[method]["fold_roc_aucs"],
                fold_pr_aucs=layer_results[method]["fold_pr_aucs"],
            )
            for method in methods
        }
        log_progress(f"layer_done layer={layer}")

    return results


def main():
    args = parse_args()
    load_data_dir, temp_dir = prepare_subset_data_dir(args.data_dir, args.max_files)
    layers = parse_layers_arg(args.layers, load_data_dir)
    methods = parse_methods_arg(args.methods)
    log_progress(
        f"start dataset_label={args.dataset_label} data_dir={args.data_dir} "
        f"load_data_dir={load_data_dir} layers={layers} methods={methods} "
        f"target_len={args.target_len} pca_dim={args.pca_dim} max_files={args.max_files}"
    )

    traces = load_all_traces(load_data_dir, layers)
    _, y = build_feature_matrix(traces)
    log_progress(
        f"loaded_traces n_traces={len(traces)} n_correct={int(y.sum())} "
        f"n_incorrect={int((y == 0).sum())}"
    )

    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    fold_indices = list(skf.split(np.zeros(len(y)), y))
    log_progress(f"folds_ready n_folds={len(fold_indices)}")

    results = {
        "metadata": {
            "data_dir": args.data_dir,
            "loaded_data_dir": load_data_dir,
            "dataset_label": args.dataset_label,
            "layers": layers,
            "pca_dim": args.pca_dim,
            "target_len": args.target_len,
            "methods": methods,
            "max_files": args.max_files,
            "n_traces": len(traces),
            "cv_splits": CV_SPLITS,
            "cv_random_state": CV_RANDOM_STATE,
        },
        "layer_results": evaluate_foldwise_trajectory_models(
            traces=traces,
            layers=layers,
            fold_indices=fold_indices,
            y=y,
            pca_dim=args.pca_dim,
            target_len=args.target_len,
            methods=methods,
        ),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{args.dataset_label}_trajectory_probe_results.json")
    log_progress(f"write_results path={json_path}")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    log_progress(f"done path={json_path}")
    if temp_dir is not None:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
