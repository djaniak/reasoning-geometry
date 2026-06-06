"""
prefix_filter.py - Leakage-safe prefix filtering simulation on multi-sample traces.

This script simulates an online "abort and retry" policy:
1) Observe first k tokens of a trace.
2) Score trace correctness from prefix features.
3) Abort if score is below a threshold and restart from a new sample.

Thresholds are calibrated on train-fold traces only. Reported metrics are computed
on held-out test-fold problems.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from analyze import (
    compute_mahal_distances,
    detect_layers,
    entropy_features,
    load_all_traces,
    mahal_features,
    mahal_trajectory_features,
)

DEFAULT_PREFIX_LENGTHS = [5, 10, 20]
DEFAULT_SCORE_KINDS = ["entropy_only", "mahalanobis_only", "combined", "mahal_raw", "mahal_cumtraj"]

# Score kinds that use raw distance directly — no logistic classifier trained.
_RAW_SCORE_KINDS = {"mahal_raw"}
DEFAULT_THRESHOLD_QUANTILES = [0.2, 0.35, 0.5, 0.65, 0.8]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_label", type=str, default="math500")
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--cv_random_state", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layers for mahal/combined scoring (auto-detected if omitted)",
    )
    parser.add_argument(
        "--prefix_lengths",
        type=str,
        default=",".join(str(v) for v in DEFAULT_PREFIX_LENGTHS),
        help="Comma-separated prefix lengths in generated tokens",
    )
    parser.add_argument(
        "--score_kinds",
        type=str,
        default=",".join(DEFAULT_SCORE_KINDS),
        help="Comma-separated score types: entropy_only, mahalanobis_only, combined",
    )
    parser.add_argument(
        "--threshold_quantiles",
        type=str,
        default=",".join(str(v) for v in DEFAULT_THRESHOLD_QUANTILES),
        help="Comma-separated quantiles on train scores used as abort thresholds",
    )
    parser.add_argument(
        "--max_restarts",
        type=int,
        default=3,
        help="Maximum restarts after abort. Attempts per problem = max_restarts + 1. Use -1 for unlimited.",
    )
    return parser.parse_args()


def parse_int_list(raw: str | None, default: list[int] | None = None) -> list[int]:
    if raw is None:
        return list(default or [])
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return sorted(dict.fromkeys(values))


def parse_float_list(raw: str | None, default: list[float] | None = None) -> list[float]:
    if raw is None:
        return list(default or [])
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    unique = sorted(dict.fromkeys(values))
    return [val for val in unique if 0.0 <= val <= 1.0]


def parse_str_list(raw: str | None, default: list[str] | None = None) -> list[str]:
    if raw is None:
        return list(default or [])
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return list(dict.fromkeys(values))


def trace_key(trace: dict) -> tuple[int, int, int]:
    return (
        int(trace["idx"]),
        int(trace.get("sample_id", 0)),
        int(trace.get("trace_id", trace["idx"])),
    )


def sort_group(group: list[dict]) -> list[dict]:
    return sorted(
        group,
        key=lambda trace: (
            int(trace.get("sample_id", 0)),
            int(trace.get("trace_id", trace["idx"])),
        ),
    )


def group_traces_by_problem(traces: list[dict]) -> dict[int, list[dict]]:
    grouped = defaultdict(list)
    for trace in traces:
        grouped[int(trace["idx"])].append(trace)
    return {idx: sort_group(group) for idx, group in grouped.items()}


def flatten_groups(groups: dict[int, list[dict]], problem_ids: list[int]) -> list[dict]:
    traces = []
    for idx in problem_ids:
        traces.extend(groups[idx])
    return traces


def choose_problem_folds(problem_ids: list[int], n_splits: int, random_state: int):
    n_splits = min(n_splits, len(problem_ids))
    if n_splits < 2:
        raise ValueError("Need at least two problems to run CV.")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    problem_ids = np.array(sorted(problem_ids), dtype=int)
    folds = []
    for train_idx, test_idx in kf.split(problem_ids):
        folds.append((problem_ids[train_idx].tolist(), problem_ids[test_idx].tolist()))
    return folds


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


def fit_logistic_selector(X_train: np.ndarray, y_train: np.ndarray):
    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        return None
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    clf.fit(X_tr, y_train)
    return scaler, clf


def predict_selector_scores(model, X: np.ndarray) -> np.ndarray:
    scaler, clf = model
    return clf.predict_proba(scaler.transform(X))[:, 1]


def raw_mahal_scores(
    traces: list[dict], prefix_len: int, layer: int, ref
) -> np.ndarray:
    """Negative cumulative Mahalanobis distance — no classifier, higher = closer to correct geometry."""
    pca, mu, cov_inv = ref
    scores = []
    for trace in traces:
        dists = compute_mahal_distances(trace["hiddens"][layer][:prefix_len], pca, mu, cov_inv)
        scores.append(-float(dists.sum()))
    return np.array(scores)


def trace_is_eligible(trace: dict, prefix_len: int, layer: int | None, score_kind: str) -> bool:
    if len(trace["entropies"]) < prefix_len:
        return False
    if score_kind == "entropy_only":
        return True
    if layer is None:
        return False
    if layer not in trace["hiddens"]:
        return False
    return len(trace["hiddens"][layer]) >= prefix_len


def build_setting_groups(
    grouped: dict[int, list[dict]], prefix_len: int, layer: int | None, score_kind: str
) -> dict[int, list[dict]]:
    out = {}
    for idx, group in grouped.items():
        eligible = [
            trace
            for trace in group
            if trace_is_eligible(trace, prefix_len, layer, score_kind)
        ]
        if eligible:
            out[idx] = sort_group(eligible)
    return out


def build_feature_matrix(
    traces: list[dict],
    score_kind: str,
    prefix_len: int,
    layer: int | None,
    ref=None,
) -> tuple[np.ndarray, np.ndarray]:
    rows, labels = [], []
    for trace in traces:
        ent = trace["entropies"][:prefix_len]
        ent_feats = entropy_features(ent)
        if score_kind == "entropy_only":
            feats = ent_feats
        else:
            if layer is None or ref is None:
                raise ValueError("Layer and Mahalanobis reference are required.")
            pca, mu, cov_inv = ref
            dists = compute_mahal_distances(trace["hiddens"][layer][:prefix_len], pca, mu, cov_inv)
            if score_kind == "mahal_cumtraj":
                feats = mahal_trajectory_features(dists)
            else:
                mah_feats = mahal_features(ent, dists)
                feats = mah_feats if score_kind == "mahalanobis_only" else (ent_feats + mah_feats)
        rows.append(feats)
        labels.append(1 if trace["is_correct"] else 0)
    return np.array(rows), np.array(labels)


def baseline_fold_metrics(groups: dict[int, list[dict]], problem_ids: list[int]) -> tuple[float, float]:
    successes, tokens = [], []
    for idx in problem_ids:
        first = groups[idx][0]
        successes.append(float(first["is_correct"]))
        tokens.append(float(len(first["entropies"])))
    return float(np.mean(successes)), float(np.mean(tokens))


def simulate_policy_for_problem(
    group: list[dict],
    score_map: dict[tuple[int, int, int], float],
    threshold: float,
    prefix_len: int,
    max_restarts: int,
) -> dict:
    if not group:
        return {
            "success": 0.0,
            "tokens_used": 0.0,
            "aborts": 0,
            "correct_aborts": 0,
            "first_try_accept": 0,
        }

    max_attempts = len(group) if max_restarts < 0 else min(len(group), max_restarts + 1)
    max_attempts = max(max_attempts, 1)

    aborts = 0
    correct_aborts = 0
    tokens_used = 0.0
    accepted = None

    for attempt_idx in range(max_attempts):
        trace = group[attempt_idx]
        score = score_map[trace_key(trace)]
        prefix_cost = float(min(prefix_len, len(trace["entropies"])))
        should_abort = score < threshold and attempt_idx < max_attempts - 1

        if should_abort:
            tokens_used += prefix_cost
            aborts += 1
            correct_aborts += int(trace["is_correct"])
            continue

        accepted = trace
        tokens_used += float(len(trace["entropies"]))
        break

    if accepted is None:
        accepted = group[max_attempts - 1]
        tokens_used += float(len(accepted["entropies"]))

    return {
        "success": float(accepted["is_correct"]),
        "tokens_used": tokens_used,
        "aborts": aborts,
        "correct_aborts": correct_aborts,
        "first_try_accept": 1 if aborts == 0 else 0,
    }


def evaluate_setting(
    groups: dict[int, list[dict]],
    folds,
    score_kind: str,
    prefix_len: int,
    layer: int | None,
    threshold_quantiles: list[float],
    max_restarts: int,
    pca_dim: int,
):
    baseline_fold_pass = []
    baseline_fold_tokens = []
    threshold_state = {
        q: {
            "threshold_values": [],
            "fold_pass": [],
            "fold_tokens": [],
            "n_problems": 0,
            "n_aborted": 0,
            "n_correct_aborted": 0,
            "n_first_try_accept": 0,
        }
        for q in threshold_quantiles
    }
    skipped_folds = 0

    for train_problem_ids, test_problem_ids in folds:
        train_traces = flatten_groups(groups, train_problem_ids)
        test_traces = flatten_groups(groups, test_problem_ids)
        if not train_traces or not test_traces:
            skipped_folds += 1
            continue

        if score_kind == "entropy_only":
            ref = None
        else:
            correct_train = [trace for trace in train_traces if trace["is_correct"]]
            ref = fit_prefix_reference_safe(correct_train, int(layer), prefix_len, pca_dim)
            if ref is None:
                skipped_folds += 1
                continue

        if score_kind in _RAW_SCORE_KINDS:
            train_scores = raw_mahal_scores(train_traces, prefix_len, int(layer), ref)
            test_scores = raw_mahal_scores(test_traces, prefix_len, int(layer), ref)
        else:
            X_train, y_train = build_feature_matrix(
                train_traces, score_kind, prefix_len, layer, ref=ref
            )
            model = fit_logistic_selector(X_train, y_train)
            if model is None:
                skipped_folds += 1
                continue
            X_test, _ = build_feature_matrix(test_traces, score_kind, prefix_len, layer, ref=ref)
            train_scores = predict_selector_scores(model, X_train)
            test_scores = predict_selector_scores(model, X_test)

        test_score_map = {
            trace_key(trace): float(score)
            for trace, score in zip(test_traces, test_scores)
        }

        baseline_pass, baseline_tokens = baseline_fold_metrics(groups, test_problem_ids)
        baseline_fold_pass.append(baseline_pass)
        baseline_fold_tokens.append(baseline_tokens)

        for quantile in threshold_quantiles:
            threshold = float(np.quantile(train_scores, quantile))
            state = threshold_state[quantile]
            state["threshold_values"].append(threshold)

            per_problem = [
                simulate_policy_for_problem(
                    groups[idx], test_score_map, threshold, prefix_len, max_restarts
                )
                for idx in test_problem_ids
            ]
            if not per_problem:
                continue

            state["fold_pass"].append(float(np.mean([entry["success"] for entry in per_problem])))
            state["fold_tokens"].append(float(np.mean([entry["tokens_used"] for entry in per_problem])))
            state["n_problems"] += len(per_problem)
            state["n_aborted"] += int(sum(entry["aborts"] for entry in per_problem))
            state["n_correct_aborted"] += int(sum(entry["correct_aborts"] for entry in per_problem))
            state["n_first_try_accept"] += int(sum(entry["first_try_accept"] for entry in per_problem))

    if not baseline_fold_pass or not baseline_fold_tokens:
        return {"skipped": True, "reason": "no valid folds for setting"}

    baseline_pass_mean = float(np.mean(baseline_fold_pass))
    baseline_pass_std = float(np.std(baseline_fold_pass))
    baseline_tokens_mean = float(np.mean(baseline_fold_tokens))
    baseline_tokens_std = float(np.std(baseline_fold_tokens))

    out_thresholds = {}
    for quantile in threshold_quantiles:
        state = threshold_state[quantile]
        if not state["fold_pass"]:
            out_thresholds[f"{quantile:.2f}"] = {
                "skipped": True,
                "reason": "no valid folds at quantile",
            }
            continue

        pass_mean = float(np.mean(state["fold_pass"]))
        pass_std = float(np.std(state["fold_pass"]))
        tokens_mean = float(np.mean(state["fold_tokens"]))
        tokens_std = float(np.std(state["fold_tokens"]))
        pass_delta = pass_mean - baseline_pass_mean
        token_savings = 1.0 - (tokens_mean / baseline_tokens_mean if baseline_tokens_mean > 0 else 1.0)
        false_abort_rate = (
            float(state["n_correct_aborted"] / state["n_aborted"]) if state["n_aborted"] > 0 else 0.0
        )
        avg_aborts_per_problem = (
            float(state["n_aborted"] / state["n_problems"]) if state["n_problems"] > 0 else 0.0
        )
        first_try_accept_rate = (
            float(state["n_first_try_accept"] / state["n_problems"]) if state["n_problems"] > 0 else 0.0
        )

        out_thresholds[f"{quantile:.2f}"] = {
            "quantile": float(quantile),
            "threshold_mean": float(np.mean(state["threshold_values"])),
            "threshold_std": float(np.std(state["threshold_values"])),
            "pass_at_1_mean": pass_mean,
            "pass_at_1_std": pass_std,
            "pass_delta_vs_baseline": pass_delta,
            "expected_tokens_mean": tokens_mean,
            "expected_tokens_std": tokens_std,
            "token_savings_mean": token_savings,
            "false_abort_rate": false_abort_rate,
            "avg_aborts_per_problem": avg_aborts_per_problem,
            "first_try_accept_rate": first_try_accept_rate,
            "n_problems_eval": int(state["n_problems"]),
            "n_aborted": int(state["n_aborted"]),
            "n_correct_aborted": int(state["n_correct_aborted"]),
            "n_valid_folds": len(state["fold_pass"]),
        }

    return {
        "score_kind": score_kind,
        "prefix_len": int(prefix_len),
        "layer": None if layer is None else int(layer),
        "n_problems": int(len(groups)),
        "baseline": {
            "pass_at_1_mean": baseline_pass_mean,
            "pass_at_1_std": baseline_pass_std,
            "expected_tokens_mean": baseline_tokens_mean,
            "expected_tokens_std": baseline_tokens_std,
            "policy": "take first sample without filtering",
        },
        "thresholds": out_thresholds,
        "n_skipped_folds": int(skipped_folds),
    }


def setting_label(setting: dict) -> str:
    layer = setting.get("layer")
    layer_label = "-" if layer is None else f"L{layer}"
    return f"{setting['score_kind']} {layer_label} k={setting['prefix_len']}"


def plot_tradeoff(results: dict, output_path: str) -> None:
    settings = [setting for setting in results["settings"].values() if not setting.get("skipped")]
    if not settings:
        print("No valid settings to plot.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_pass, ax_delta) = plt.subplots(1, 2, figsize=(13, 5.2))
    palette = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#66a61e",
        "#e7298a",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ]

    for idx, setting in enumerate(settings):
        rows = []
        for key, payload in setting["thresholds"].items():
            if payload.get("skipped"):
                continue
            rows.append((float(key), payload))
        if not rows:
            continue
        rows.sort(key=lambda item: item[0])

        xs = [item[1]["token_savings_mean"] for item in rows]
        y_pass = [item[1]["pass_at_1_mean"] for item in rows]
        y_delta = [item[1]["pass_delta_vs_baseline"] for item in rows]
        color = palette[idx % len(palette)]
        label = setting_label(setting)

        ax_pass.plot(xs, y_pass, "o-", color=color, linewidth=2, markersize=5, label=label)
        ax_delta.plot(xs, y_delta, "o-", color=color, linewidth=2, markersize=5, label=label)

    ax_pass.set_xlabel("Token Savings (fraction)")
    ax_pass.set_ylabel("Pass@1")
    ax_pass.set_title("Pass@1 vs Token Savings")

    ax_delta.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax_delta.set_xlabel("Token Savings (fraction)")
    ax_delta.set_ylabel("Pass@1 Delta vs Baseline")
    ax_delta.set_title("Quality Change vs Token Savings")

    ax_pass.legend(fontsize=8)
    ax_delta.legend(fontsize=8)
    fig.suptitle(
        f"Prefix filtering trade-off for {results['dataset']} ({results['model_label']})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot written to {output_path}")


def write_summary_markdown(results: dict, output_path: str) -> None:
    lines = [
        "# Prefix Filtering Summary",
        "",
        f"*Dataset:* `{results['dataset']}`  ",
        f"*Trace source:* `{results['data_dir']}`",
        "",
        "Policy:",
        "- Train classifier on train-fold traces only.",
        "- Calibrate abort threshold from train scores only.",
        "- On each test problem, consume prefix k, abort if score < threshold, and retry until accept or restart cap.",
        "",
        f"*Max restarts:* `{results['max_restarts']}`",
        "",
        "| Score Kind | Layer | Prefix | Quantile | Pass@1 | Pass Delta | Tokens / Problem | Token Savings | False Abort Rate | Avg Aborts / Problem | First-Try Accept |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def _row(setting: dict, payload: dict) -> list[str]:
        return [
            setting["score_kind"],
            "-" if setting["layer"] is None else str(setting["layer"]),
            str(setting["prefix_len"]),
            f"{payload['quantile']:.2f}",
            f"{payload['pass_at_1_mean']:.3f}",
            f"{payload['pass_delta_vs_baseline']:+.3f}",
            f"{payload['expected_tokens_mean']:.1f}",
            f"{payload['token_savings_mean']:+.3f}",
            f"{payload['false_abort_rate']:.3f}",
            f"{payload['avg_aborts_per_problem']:.3f}",
            f"{payload['first_try_accept_rate']:.3f}",
        ]

    best_candidate = None
    for setting in results["settings"].values():
        if setting.get("skipped"):
            continue
        for payload in setting["thresholds"].values():
            if payload.get("skipped"):
                continue
            lines.append("| " + " | ".join(_row(setting, payload)) + " |")
            candidate = {
                "setting": setting,
                "payload": payload,
            }
            if best_candidate is None:
                best_candidate = candidate
            else:
                prev = best_candidate["payload"]
                if payload["pass_delta_vs_baseline"] > prev["pass_delta_vs_baseline"]:
                    best_candidate = candidate
                elif (
                    payload["pass_delta_vs_baseline"] == prev["pass_delta_vs_baseline"]
                    and payload["token_savings_mean"] > prev["token_savings_mean"]
                ):
                    best_candidate = candidate

    if best_candidate is not None:
        setting = best_candidate["setting"]
        payload = best_candidate["payload"]
        baseline = setting["baseline"]
        lines.extend(
            [
                "",
                "## Highest Delta Operating Point",
                "",
                f"- Setting: `{setting_label(setting)}`",
                f"- Threshold quantile: `{payload['quantile']:.2f}` (score threshold ~{payload['threshold_mean']:.3f})",
                f"- Pass@1: `{payload['pass_at_1_mean']:.3f}` vs baseline `{baseline['pass_at_1_mean']:.3f}` "
                f"(delta `{payload['pass_delta_vs_baseline']:+.3f}`)",
                f"- Expected tokens/problem: `{payload['expected_tokens_mean']:.1f}` vs baseline "
                f"`{baseline['expected_tokens_mean']:.1f}` (savings `{payload['token_savings_mean']:+.3f}`)",
                f"- False abort rate: `{payload['false_abort_rate']:.3f}`",
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
    score_kinds = parse_str_list(args.score_kinds, DEFAULT_SCORE_KINDS)
    threshold_quantiles = parse_float_list(
        args.threshold_quantiles, DEFAULT_THRESHOLD_QUANTILES
    )

    valid_score_kinds = {"entropy_only", "mahalanobis_only", "combined", "mahal_raw", "mahal_cumtraj"}
    score_kinds = [kind for kind in score_kinds if kind in valid_score_kinds]
    if not score_kinds:
        raise ValueError("No valid score kinds requested.")
    if not threshold_quantiles:
        raise ValueError("No valid threshold quantiles requested.")

    traces = load_all_traces(args.data_dir, layers)
    grouped_all = group_traces_by_problem(traces)

    print(f"Loaded {len(traces)} traces across {len(grouped_all)} problems")
    print(f"Layers available: {layers}")
    print(f"Prefix lengths: {prefix_lengths}")
    print(f"Score kinds: {score_kinds}")

    results: dict[str, object] = {
        "dataset": args.dataset_label,
        "data_dir": args.data_dir,
        "model_label": os.path.basename(os.path.dirname(args.data_dir)) or args.data_dir,
        "layers": [int(layer) for layer in layers],
        "prefix_lengths": prefix_lengths,
        "score_kinds": score_kinds,
        "threshold_quantiles": threshold_quantiles,
        "max_restarts": int(args.max_restarts),
        "n_total_traces": len(traces),
        "n_total_problems": len(grouped_all),
        "leakage_protocol": (
            "per fold: model fit on train traces only; mahal reference fit on train-fold "
            "correct traces only; abort threshold calibrated on train scores only"
        ),
        "settings": {},
    }

    for prefix_len in prefix_lengths:
        for score_kind in score_kinds:
            layer_candidates = [None] if score_kind == "entropy_only" else layers
            for layer in layer_candidates:
                setting_key = f"{score_kind}|k={prefix_len}|L{layer if layer is not None else 'na'}"
                setting_groups = build_setting_groups(grouped_all, prefix_len, layer, score_kind)
                if len(setting_groups) < 2:
                    results["settings"][setting_key] = {
                        "score_kind": score_kind,
                        "prefix_len": int(prefix_len),
                        "layer": None if layer is None else int(layer),
                        "skipped": True,
                        "reason": "not enough eligible problems",
                    }
                    continue

                folds = choose_problem_folds(
                    list(setting_groups.keys()), args.n_splits, args.cv_random_state
                )
                setting_result = evaluate_setting(
                    groups=setting_groups,
                    folds=folds,
                    score_kind=score_kind,
                    prefix_len=prefix_len,
                    layer=layer,
                    threshold_quantiles=threshold_quantiles,
                    max_restarts=args.max_restarts,
                    pca_dim=args.pca_dim,
                )
                results["settings"][setting_key] = setting_result

                if setting_result.get("skipped"):
                    print(f"Skip {setting_key}: {setting_result.get('reason', 'unknown')}")
                    continue

                best_quant = None
                best_payload = None
                for quantile_key, payload in setting_result["thresholds"].items():
                    if payload.get("skipped"):
                        continue
                    if best_payload is None or payload["pass_delta_vs_baseline"] > best_payload["pass_delta_vs_baseline"]:
                        best_quant = quantile_key
                        best_payload = payload
                if best_payload is not None:
                    print(
                        f"{setting_key:30s} best q={best_quant} "
                        f"pass={best_payload['pass_at_1_mean']:.3f} "
                        f"delta={best_payload['pass_delta_vs_baseline']:+.3f} "
                        f"savings={best_payload['token_savings_mean']:+.3f}"
                    )

    json_path = os.path.join(args.output_dir, f"{args.dataset_label}_prefix_filter_results.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results written to {json_path}")

    plot_tradeoff(
        results, os.path.join(args.output_dir, f"{args.dataset_label}_prefix_filter_tradeoff.png")
    )
    write_summary_markdown(
        results, os.path.join(args.output_dir, f"{args.dataset_label}_prefix_filter_summary.md")
    )


if __name__ == "__main__":
    main()
