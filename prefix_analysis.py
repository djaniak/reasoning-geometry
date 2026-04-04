"""
prefix_analysis.py — Leakage-safe prefix detection on reasoning traces.

Evaluates whether short prefixes of a trace already predict final correctness.
For each prefix length k, entropy-only features are compared against
entropy+Mahalanobis features where the Mahalanobis reference is fit inside each
CV train fold using only the first k tokens of correct traces.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

from analyze import (
    _fold_clf_auc,
    compute_mahal_distances,
    detect_layers,
    entropy_features,
    evaluate_features,
    load_all_traces,
    mahal_features,
)

DEFAULT_PREFIX_LENGTHS = [5, 10, 20, 40]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_label", type=str, default="math500")
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--cv_random_state", type=int, default=42)
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to analyze (auto-detected if omitted)",
    )
    parser.add_argument(
        "--prefix_lengths",
        type=str,
        default=",".join(str(v) for v in DEFAULT_PREFIX_LENGTHS),
        help="Comma-separated prefix lengths in generated tokens",
    )
    return parser.parse_args()


def parse_int_list(raw: str | None, default: list[int] | None = None) -> list[int]:
    if raw is None:
        return list(default or [])
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return sorted(dict.fromkeys(values))


def pack_scores(roc_aucs: list[float], pr_aucs: list[float], label: str) -> dict:
    if not roc_aucs:
        print(f"    {label:30s}: no valid folds")
        return {
            "roc_auc_mean": float("nan"),
            "roc_auc_std": float("nan"),
            "pr_auc_mean": float("nan"),
            "pr_auc_std": float("nan"),
            "fold_roc_aucs": [],
        }

    result = {
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs)),
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs)),
        "fold_roc_aucs": [float(v) for v in roc_aucs],
    }
    print(
        f"    {label:30s}: ROC-AUC = {result['roc_auc_mean']:.4f} ± "
        f"{result['roc_auc_std']:.4f} | PR-AUC = {result['pr_auc_mean']:.4f} ± "
        f"{result['pr_auc_std']:.4f}"
    )
    return result


def build_entropy_prefix_matrix(
    traces: list[dict], prefix_len: int
) -> tuple[np.ndarray, np.ndarray]:
    rows, labels = [], []
    for trace in traces:
        ent = trace["entropies"][:prefix_len]
        rows.append(entropy_features(ent))
        labels.append(1 if trace["is_correct"] else 0)
    return np.array(rows), np.array(labels)


def fit_prefix_reference_safe(
    correct_traces: list[dict], layer: int, prefix_len: int, pca_dim: int
):
    if not correct_traces:
        return None

    try:
        correct_hiddens = np.concatenate(
            [trace["hiddens"][layer][:prefix_len] for trace in correct_traces], axis=0
        )
        effective_dim = min(pca_dim, correct_hiddens.shape[0], correct_hiddens.shape[1])
        if effective_dim < 2:
            return None

        svd_solver = "randomized" if correct_hiddens.shape[0] > 200_000 else "full"
        pca = PCA(n_components=effective_dim, random_state=42, svd_solver=svd_solver)
        pca.fit(correct_hiddens)

        projected = pca.transform(correct_hiddens)
        mu = projected.mean(axis=0)
        cov = np.cov(projected - mu, rowvar=False)
        cov_inv = np.linalg.inv(cov + 1e-4 * np.eye(effective_dim))
        return pca, mu, cov_inv
    except Exception:
        return None


def evaluate_prefix_mahalanobis(
    traces: list[dict],
    layer: int,
    prefix_len: int,
    pca_dim: int,
    y: np.ndarray,
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    roc_mah, pr_mah = [], []
    roc_comb, pr_comb = [], []

    for fi, (train_idx, test_idx) in enumerate(fold_indices):
        correct_train = [traces[i] for i in train_idx if traces[i]["is_correct"]]
        ref = fit_prefix_reference_safe(correct_train, layer, prefix_len, pca_dim)
        if ref is None:
            print(
                f"    WARNING: skip fold {fi} — prefix Mahalanobis ref failed "
                f"(layer {layer}, prefix {prefix_len})"
            )
            continue
        pca, mu, cov_inv = ref

        x_mah_rows, x_comb_rows = [], []
        for trace in traces:
            ent = trace["entropies"][:prefix_len]
            dists = compute_mahal_distances(trace["hiddens"][layer][:prefix_len], pca, mu, cov_inv)
            mah = mahal_features(ent, dists)
            x_mah_rows.append(mah)
            x_comb_rows.append(entropy_features(ent) + mah)

        x_mah = np.array(x_mah_rows)
        x_comb = np.array(x_comb_rows)

        out_mah = _fold_clf_auc(x_mah, y, train_idx, test_idx)
        if out_mah is not None:
            roc_mah.append(out_mah[0])
            pr_mah.append(out_mah[1])
        out_comb = _fold_clf_auc(x_comb, y, train_idx, test_idx)
        if out_comb is not None:
            roc_comb.append(out_comb[0])
            pr_comb.append(out_comb[1])

    return {
        "mahalanobis_only": pack_scores(roc_mah, pr_mah, f"L{layer} mahal-only k={prefix_len}"),
        "combined": pack_scores(roc_comb, pr_comb, f"L{layer} combined k={prefix_len}"),
    }


def eligible_traces(traces: list[dict], layers: list[int], prefix_len: int) -> list[dict]:
    subset = []
    for trace in traces:
        if len(trace["entropies"]) < prefix_len:
            continue
        if any(layer not in trace["hiddens"] or len(trace["hiddens"][layer]) < prefix_len for layer in layers):
            continue
        subset.append(trace)
    return subset


def plot_prefix_auc(results: dict, output_path: str) -> None:
    prefix_keys = [
        key for key in sorted(results["prefixes"], key=int)
        if not results["prefixes"][key].get("skipped")
    ]
    if not prefix_keys:
        print("No prefix results to plot.")
        return

    prefix_lengths = [int(k) for k in prefix_keys]
    layers = sorted({int(layer) for p in prefix_keys for layer in results["prefixes"][p]["layers"]})

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    auc_ax, delta_ax = axes

    entropy_curve = [
        results["prefixes"][str(k)]["entropy_only"]["roc_auc_mean"] for k in prefix_lengths
    ]
    auc_ax.plot(
        prefix_lengths,
        entropy_curve,
        "o--",
        color="#222222",
        linewidth=2,
        markersize=6,
        label="Entropy-only",
    )

    palette = ["#1b9e77", "#d95f02", "#7570b3", "#66a61e", "#e7298a"]
    for idx, layer in enumerate(layers):
        xs, aucs, deltas = [], [], []
        for prefix_len in prefix_lengths:
            lr = results["prefixes"][str(prefix_len)]["layers"].get(str(layer))
            if not lr:
                continue
            xs.append(prefix_len)
            aucs.append(lr["combined"]["roc_auc_mean"])
            deltas.append(lr["delta_vs_entropy"])
        color = palette[idx % len(palette)]
        auc_ax.plot(xs, aucs, "o-", color=color, linewidth=2, markersize=6, label=f"L{layer} combined")
        delta_ax.plot(xs, deltas, "o-", color=color, linewidth=2, markersize=6, label=f"L{layer}")

    auc_ax.set_xlabel("Prefix Length (generated tokens)")
    auc_ax.set_ylabel("ROC-AUC")
    auc_ax.set_title("Prefix Correctness Detection")
    auc_ax.set_ylim(0.45, 1.0)
    auc_ax.legend(fontsize=9)

    delta_ax.axhline(0.0, color="#444444", linewidth=1, linestyle="--")
    delta_ax.set_xlabel("Prefix Length (generated tokens)")
    delta_ax.set_ylabel("Combined - Entropy ROC-AUC")
    delta_ax.set_title("Geometry Gain by Prefix Length")
    delta_ax.legend(fontsize=9)

    fig.suptitle(
        f"Prefix analysis for {results['dataset']} ({results['model_label']})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot written to {output_path}")


def write_summary_markdown(results: dict, output_path: str) -> None:
    prefix_keys = [
        key for key in sorted(results["prefixes"], key=int)
        if not results["prefixes"][key].get("skipped")
    ]
    layers = sorted({int(layer) for p in prefix_keys for layer in results["prefixes"][p]["layers"]})

    lines = [
        "# Prefix Analysis Summary",
        "",
        f"*Dataset:* `{results['dataset']}`  ",
        f"*Trace source:* `{results['data_dir']}`",
        "",
        "| Prefix | Eligible N | Correct | Incorrect | Entropy AUC | "
        + " | ".join(f"L{layer} Combined | L{layer} Δ" for layer in layers)
        + " |",
        "|---|---|---|---|---|"
        + "|".join(["---|---"] * len(layers))
        + "|",
    ]

    for prefix_len in prefix_keys:
        row = results["prefixes"][prefix_len]
        cells = [
            prefix_len,
            str(row["n_total"]),
            str(row["n_correct"]),
            str(row["n_incorrect"]),
            f"{row['entropy_only']['roc_auc_mean']:.3f}",
        ]
        for layer in layers:
            layer_result = row["layers"].get(str(layer))
            if layer_result:
                cells.append(f"{layer_result['combined']['roc_auc_mean']:.3f}")
                cells.append(f"{layer_result['delta_vs_entropy']:+.3f}")
            else:
                cells.extend(["—", "—"])
        lines.append("| " + " | ".join(cells) + " |")

    best = results.get("best_combined")
    if best:
        lines.extend(
            [
                "",
                "## Best Combined Setting",
                "",
                f"- Layer `L{best['layer']}`, prefix `{best['prefix_len']}`",
                f"- Combined ROC-AUC `{best['combined_auc']:.3f}`",
                f"- Entropy baseline `{best['entropy_auc']:.3f}`",
                f"- Gain `{best['delta_vs_entropy']:+.3f}`",
            ]
        )

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Summary written to {output_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    layers = parse_int_list(args.layers) if args.layers else detect_layers(args.data_dir)
    prefix_lengths = parse_int_list(args.prefix_lengths, DEFAULT_PREFIX_LENGTHS)
    traces = load_all_traces(args.data_dir, layers)

    results: dict[str, object] = {
        "dataset": args.dataset_label,
        "data_dir": args.data_dir,
        "model_label": os.path.basename(os.path.dirname(args.data_dir)) or args.data_dir,
        "prefix_lengths": prefix_lengths,
        "layers": layers,
        "n_total_traces": len(traces),
        "prefixes": {},
    }

    best_combined = None

    for prefix_len in prefix_lengths:
        subset = eligible_traces(traces, layers, prefix_len)
        n_correct = sum(1 for trace in subset if trace["is_correct"])
        n_incorrect = len(subset) - n_correct

        prefix_result = {
            "n_total": len(subset),
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_dropped_short": len(traces) - len(subset),
            "layers": {},
        }

        print(
            f"\nPrefix {prefix_len}: {len(subset)} eligible traces "
            f"({n_correct} correct / {n_incorrect} incorrect)"
        )
        if min(n_correct, n_incorrect) < 5:
            prefix_result["skipped"] = True
            results["prefixes"][str(prefix_len)] = prefix_result
            print("  Skipping: insufficient class balance for CV.")
            continue

        x_ent, y = build_entropy_prefix_matrix(subset, prefix_len)
        n_splits = min(5, min(n_correct, n_incorrect))
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=args.cv_random_state
        )
        fold_indices = list(skf.split(x_ent, y))
        ent_res = evaluate_features(x_ent, y, fold_indices, f"entropy k={prefix_len}")
        prefix_result["entropy_only"] = ent_res

        for layer in layers:
            print(f"  Layer {layer}:")
            fw = evaluate_prefix_mahalanobis(
                subset, layer, prefix_len, args.pca_dim, y, fold_indices
            )
            delta = fw["combined"]["roc_auc_mean"] - ent_res["roc_auc_mean"]
            fw["delta_vs_entropy"] = float(delta)
            prefix_result["layers"][str(layer)] = fw

            candidate = {
                "layer": layer,
                "prefix_len": prefix_len,
                "combined_auc": fw["combined"]["roc_auc_mean"],
                "entropy_auc": ent_res["roc_auc_mean"],
                "delta_vs_entropy": float(delta),
            }
            if best_combined is None or candidate["combined_auc"] > best_combined["combined_auc"]:
                best_combined = candidate

        results["prefixes"][str(prefix_len)] = prefix_result

    if best_combined is not None:
        results["best_combined"] = best_combined

    json_path = os.path.join(args.output_dir, f"{args.dataset_label}_prefix_results.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results written to {json_path}")

    plot_prefix_auc(results, os.path.join(args.output_dir, f"{args.dataset_label}_prefix_auc.png"))
    write_summary_markdown(
        results, os.path.join(args.output_dir, f"{args.dataset_label}_prefix_summary.md")
    )


if __name__ == "__main__":
    main()
