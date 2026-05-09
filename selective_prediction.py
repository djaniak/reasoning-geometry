"""
selective_prediction.py — Binary trust/abstain downstream task for reasoning traces.

Given one completed trace, uses geometry-based confidence scores to decide whether
to trust the answer or abstain. Evaluates an unsupervised-first scorer set:
entropy, log-prob, raw Mahalanobis, RMD, norm-RMD, plus combined-LR as a labeled
supervised upper bound.

Evaluation uses pooled out-of-fold (OOF) coverage-accuracy curves, not averaged
fold curves. CV protocol matches analyze.py: StratifiedKFold on trace indices,
references fit on train-fold correct traces only. For single-sample data (one trace
per problem), trace indices and problem indices coincide.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from analyze import (
    compute_mahal_distances,
    compute_relative_mahal_distances,
    detect_layers,
    entropy_features,
    fit_mahalanobis_reference_safe,
    fit_relative_mahalanobis_reference_safe,
    load_all_traces,
    mahal_features,
    parse_pca_dim_arg,
)


# ---------------------------------------------------------------------------
# Coverage-accuracy curve utilities
# ---------------------------------------------------------------------------

def coverage_accuracy_curve(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute coverage-accuracy curve by sorting traces by confidence descending.

    For k = 1..n: accept the top-k highest-scoring traces.
      coverage = k / n   (guaranteed strictly increasing)
      accuracy = mean(labels[top-k])  (may wiggle empirically)

    Args:
        scores: confidence values (higher = more confident = trust)
        labels: binary correctness labels (1 = correct, 0 = incorrect)

    Returns:
        (coverages, accuracies): both np.ndarray of length n.
    """
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    n = len(sorted_labels)
    coverages = np.arange(1, n + 1) / n
    accuracies = np.cumsum(sorted_labels.astype(float)) / np.arange(1, n + 1)
    return coverages, accuracies


def ausc(
    coverages: np.ndarray, accuracies: np.ndarray, min_coverage: float = 0.3
) -> float | None:
    """Area under the selective prediction curve, restricted to [min_coverage, 1.0].

    Normalised by the coverage range so the value is comparable across settings.
    Returns None if fewer than two points fall in range.
    """
    mask = coverages >= min_coverage
    if mask.sum() < 2:
        return None
    c = coverages[mask]
    a = accuracies[mask]
    coverage_range = float(c[-1] - c[0])
    if coverage_range < 1e-9:
        return None
    return float(np.trapezoid(a, c) / coverage_range)


def read_at_coverage(
    coverages: np.ndarray, accuracies: np.ndarray, targets: list[float]
) -> dict[str, float | None]:
    """Interpolate accuracy at each target coverage level.

    Returns the accuracy at the smallest coverage >= target (i.e. the most
    selective operating point that still meets that coverage requirement).
    """
    result = {}
    for t in targets:
        above = coverages >= t
        if not above.any():
            result[str(t)] = None
        else:
            idx = int(np.argmax(above))
            result[str(t)] = float(accuracies[idx])
    return result


def _pack_scorer(
    coverages: np.ndarray,
    accuracies: np.ndarray,
    n_eval: int,
    min_coverage: float,
    operating_points: list[float],
) -> dict:
    return {
        "n_eval": n_eval,
        "ausc": ausc(coverages, accuracies, min_coverage),
        "operating_points": read_at_coverage(coverages, accuracies, operating_points),
        "curve": {
            "coverages": coverages.tolist(),
            "accuracies": accuracies.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Logistic regression helpers
# ---------------------------------------------------------------------------

def _fit_logistic(X_train: np.ndarray, y_train: np.ndarray):
    """Fit StandardScaler + LogisticRegression. Returns (scaler, clf) or None."""
    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        return None
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    clf.fit(X_tr, y_train)
    return scaler, clf


def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    scaler, clf = model
    return clf.predict_proba(scaler.transform(X))[:, 1]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_selective_prediction(
    traces: list[dict],
    layers: list[int],
    pca_dim,
    n_splits: int,
    cv_random_state: int,
    min_coverage: float,
    operating_points: list[float],
    dataset_label: str,
) -> dict:
    """Evaluate all scorers and return the full results dict."""
    n = len(traces)
    labels_all = np.array([int(t["is_correct"]) for t in traces])

    # ── Unsupervised scorers (no CV needed) ───────────────────────────────

    # entropy_mean: negate mean token entropy (high entropy → abstain)
    entropy_scores = np.array([-t["entropies"].mean() for t in traces])

    # mean_logprob: high log-prob → trust; skip None traces
    logprob_scores_list: list[float] = []
    logprob_valid: list[bool] = []
    for t in traces:
        lp = t.get("mean_logprob")
        if lp is None and t.get("token_logprobs") is not None:
            arr = t["token_logprobs"]
            if arr is not None and len(arr) > 0:
                lp = float(np.mean(arr))
        if lp is not None:
            logprob_scores_list.append(float(lp))
            logprob_valid.append(True)
        else:
            logprob_scores_list.append(0.0)
            logprob_valid.append(False)
    logprob_mask = np.array(logprob_valid)

    # ── CV fold structure ─────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_random_state)
    fold_indices = list(skf.split(np.arange(n), labels_all))

    # OOF accumulators: {layer: [(score, label), ...]}
    oof: dict[str, list[tuple[float, int]]] = {
        f"raw_mahal_L{layer}": [] for layer in layers
    }
    for layer in layers:
        oof[f"raw_rmd_L{layer}"] = []
        oof[f"norm_rmd_L{layer}"] = []
        oof[f"combined_lr_L{layer}"] = []

    for fi, (train_idx, test_idx) in enumerate(fold_indices):
        correct_train = [traces[i] for i in train_idx if traces[i]["is_correct"]]
        all_train = [traces[i] for i in train_idx]

        for layer in layers:
            raw_ref = fit_mahalanobis_reference_safe(correct_train, layer, pca_dim)
            rmd_ref = fit_relative_mahalanobis_reference_safe(
                correct_train, all_train, layer, pca_dim
            )
            norm_rmd_ref = fit_relative_mahalanobis_reference_safe(
                correct_train, all_train, layer, pca_dim, normalize_input=True
            )

            # Build combined-LR training features using raw_ref
            X_train_lr: list[list[float]] = []
            y_train_lr: list[int] = []
            if raw_ref is not None:
                for i in train_idx:
                    t = traces[i]
                    dists = compute_mahal_distances(t["hiddens"][layer], *raw_ref)
                    feats = entropy_features(t["entropies"]) + mahal_features(t["entropies"], dists)
                    X_train_lr.append(feats)
                    y_train_lr.append(int(t["is_correct"]))

            lr_model = (
                _fit_logistic(np.array(X_train_lr), np.array(y_train_lr))
                if X_train_lr
                else None
            )

            # Score test traces
            X_test_lr: list[list[float]] = []
            test_labels_for_lr: list[int] = []

            for i in test_idx:
                t = traces[i]
                hiddens = t["hiddens"].get(layer)
                if hiddens is None:
                    continue
                e = t["entropies"]
                label = int(t["is_correct"])

                if raw_ref is not None:
                    dists = compute_mahal_distances(hiddens, *raw_ref)
                    # negate: high distance → abstain
                    oof[f"raw_mahal_L{layer}"].append((-float(dists.mean()), label))
                    feats = entropy_features(e) + mahal_features(e, dists)
                    X_test_lr.append(feats)
                    test_labels_for_lr.append(label)

                if rmd_ref is not None:
                    rmd = compute_relative_mahal_distances(hiddens, *rmd_ref)
                    oof[f"raw_rmd_L{layer}"].append((-float(rmd.mean()), label))

                if norm_rmd_ref is not None:
                    pca, mu, cov_inv, bg_mu, bg_cov_inv = norm_rmd_ref
                    norm_rmd = compute_relative_mahal_distances(
                        hiddens, pca, mu, cov_inv, bg_mu, bg_cov_inv, normalize_input=True
                    )
                    oof[f"norm_rmd_L{layer}"].append((-float(norm_rmd.mean()), label))

            if lr_model is not None and X_test_lr:
                probs = _predict_proba(lr_model, np.array(X_test_lr))
                for prob, label in zip(probs, test_labels_for_lr):
                    oof[f"combined_lr_L{layer}"].append((float(prob), label))

    # ── Build scorer results ──────────────────────────────────────────────
    scorers: dict[str, dict] = {}

    # no_abstain: single operating point, no curve
    base_acc = float(labels_all.mean())
    scorers["no_abstain"] = {
        "n_eval": n,
        "ausc": None,
        "operating_points": {str(op): base_acc for op in operating_points},
    }

    # entropy_mean
    cov, acc = coverage_accuracy_curve(entropy_scores, labels_all)
    scorers["entropy_mean"] = _pack_scorer(cov, acc, n, min_coverage, operating_points)

    # mean_logprob
    if logprob_mask.any():
        lp_arr = np.array(logprob_scores_list)
        valid_scores = lp_arr[logprob_mask]
        valid_labels = labels_all[logprob_mask]
        cov, acc = coverage_accuracy_curve(valid_scores, valid_labels)
        scorers["mean_logprob"] = _pack_scorer(
            cov, acc, int(logprob_mask.sum()), min_coverage, operating_points
        )
    else:
        scorers["mean_logprob"] = {"n_eval": 0, "ausc": None, "operating_points": {}}

    # Reference-based scorers (OOF)
    for key, pairs in oof.items():
        if not pairs:
            scorers[key] = {"n_eval": 0, "ausc": None, "operating_points": {}}
            continue
        sc = np.array([p[0] for p in pairs])
        lb = np.array([p[1] for p in pairs])
        cov, acc = coverage_accuracy_curve(sc, lb)
        scorers[key] = _pack_scorer(cov, acc, len(pairs), min_coverage, operating_points)

    return {
        "dataset": dataset_label,
        "n_traces": n,
        "n_correct": int(labels_all.sum()),
        "settings": {
            "pca_dim": str(pca_dim),
            "n_splits": n_splits,
            "min_coverage": min_coverage,
            "cv_random_state": cv_random_state,
            "operating_points": operating_points,
        },
        "scorers": scorers,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_coverage_accuracy(results: dict, output_path: str) -> None:
    scorers = results["scorers"]
    dataset = results.get("dataset", "")
    base_acc = results["n_correct"] / results["n_traces"]
    min_cov = results["settings"].get("min_coverage", 0.3)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Fixed styles for non-geometry scorers
    fixed_styles: dict[str, tuple[str, str]] = {
        "entropy_mean": ("--", "gray"),
        "mean_logprob": ("--", "steelblue"),
    }
    for key, (ls, color) in fixed_styles.items():
        s = scorers.get(key, {})
        curve = s.get("curve")
        if not curve or s.get("n_eval", 0) == 0:
            continue
        ausc_val = s.get("ausc")
        lbl = f"{key} (AUSC={ausc_val:.3f})" if ausc_val is not None else key
        ax.plot(curve["coverages"], curve["accuracies"],
                linestyle=ls, color=color, alpha=0.75, linewidth=1.5, label=lbl)

    # Geometry scorers
    colors = list(plt.cm.tab10.colors)
    ci = 0
    for key, s in scorers.items():
        if key in {"no_abstain", "entropy_mean", "mean_logprob"}:
            continue
        curve = s.get("curve")
        if not curve or s.get("n_eval", 0) == 0:
            continue
        ausc_val = s.get("ausc")
        lbl = f"{key} (AUSC={ausc_val:.3f})" if ausc_val is not None else key
        ax.plot(curve["coverages"], curve["accuracies"],
                color=colors[ci % len(colors)], linewidth=1.5, alpha=0.85, label=lbl)
        ci += 1

    ax.axhline(base_acc, color="black", linestyle=":", linewidth=1,
               label=f"No abstain ({base_acc:.3f})")

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Selective Prediction — {dataset}")
    ax.set_xlim(min_cov, 1.02)
    ax.set_ylim(bottom=max(0.0, base_acc - 0.15))
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def write_summary(results: dict, output_path: str) -> None:
    dataset = results.get("dataset", "")
    n_traces = results["n_traces"]
    n_correct = results["n_correct"]
    base_acc = n_correct / n_traces
    ops = results["settings"].get("operating_points", [0.6, 0.7, 0.8, 0.9])

    lines = [
        f"# Selective Prediction: {dataset}",
        "",
        f"- Traces: {n_traces} ({n_correct} correct, base accuracy {base_acc:.3f})",
        f"- Min coverage for AUSC: {results['settings']['min_coverage']}",
        f"- CV splits: {results['settings']['n_splits']}",
        "",
        "## Results",
        "",
    ]

    op_header = " | ".join(f"Acc@{op}" for op in ops)
    lines.append(f"| Scorer | n_eval | AUSC | {op_header} |")
    lines.append("| --- | --- | --- | " + " | ".join(["---"] * len(ops)) + " |")

    for name, s in results["scorers"].items():
        n_eval = s.get("n_eval", 0)
        ausc_val = s.get("ausc")
        ausc_str = f"{ausc_val:.4f}" if ausc_val is not None else "—"
        op_dict = s.get("operating_points", {})
        op_cells = " | ".join(
            f"{op_dict[str(op)]:.3f}" if op_dict.get(str(op)) is not None else "—"
            for op in ops
        )
        lines.append(f"| {name} | {n_eval} | {ausc_str} | {op_cells} |")

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Summary saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Binary trust/abstain downstream evaluation for reasoning traces."
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_label", type=str, default="math500")
    parser.add_argument("--pca_dim", type=parse_pca_dim_arg, default="128")
    parser.add_argument("--cv_random_state", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (auto-detected if omitted)",
    )
    parser.add_argument("--min_coverage", type=float, default=0.3)
    parser.add_argument(
        "--operating_points",
        type=str,
        default="0.6,0.7,0.8,0.9",
        help="Comma-separated coverage levels to report accuracy at",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    operating_points = [
        float(x) for x in args.operating_points.split(",") if x.strip()
    ]
    layers = (
        [int(x) for x in args.layers.split(",") if x.strip()]
        if args.layers
        else None
    )

    if layers is None:
        layers = detect_layers(args.data_dir)

    print(f"Loading traces from {args.data_dir} (layers: {layers})")
    traces = load_all_traces(args.data_dir, layers)
    print(f"Loaded {len(traces)} traces ({sum(t['is_correct'] for t in traces)} correct)")

    results = evaluate_selective_prediction(
        traces=traces,
        layers=layers,
        pca_dim=args.pca_dim,
        n_splits=args.n_splits,
        cv_random_state=args.cv_random_state,
        min_coverage=args.min_coverage,
        operating_points=operating_points,
        dataset_label=args.dataset_label,
    )

    prefix = args.dataset_label
    results_path = os.path.join(args.output_dir, f"{prefix}_selective_prediction_results.json")
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved to {results_path}")

    plot_coverage_accuracy(
        results, os.path.join(args.output_dir, f"{prefix}_selective_prediction.png")
    )
    write_summary(
        results, os.path.join(args.output_dir, f"{prefix}_selective_prediction_summary.md")
    )


if __name__ == "__main__":
    main()
