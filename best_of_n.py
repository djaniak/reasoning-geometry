"""
best_of_n.py — Leakage-safe Best-of-N evaluation for reasoning traces.

Requires multi-sample traces grouped by problem index. For trained selectors,
the scoring model is fit on traces from train problems only, then used to select
one trace among the N candidates for each held-out problem.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from analyze import (
    compute_mahal_distances,
    detect_layers,
    entropy_features,
    fit_mahalanobis_reference_safe,
    load_all_traces,
    load_math500_levels,
    load_math500_subjects,
    mahal_features,
)


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
        help="Comma-separated layer indices to evaluate (auto-detected if omitted)",
    )
    parser.add_argument(
        "--n_values",
        type=str,
        default="1,8,16",
        help="Comma-separated Best-of-N values to evaluate",
    )
    return parser.parse_args()


def parse_int_list(raw: str | None) -> list[int]:
    if not raw:
        return []
    return sorted(dict.fromkeys(int(part.strip()) for part in raw.split(",") if part.strip()))


def trace_mean_logprob(trace: dict) -> float:
    if trace.get("mean_logprob") is not None:
        return float(trace["mean_logprob"])
    token_logprobs = trace.get("token_logprobs")
    if token_logprobs is None or len(token_logprobs) == 0:
        return float("-inf")
    return float(np.mean(token_logprobs))


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


def subset_groups_for_n(groups: dict[int, list[dict]], n: int) -> dict[int, list[dict]]:
    subset = {}
    for idx, group in groups.items():
        if len(group) >= n:
            subset[idx] = group[:n]
    return subset


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


def choose_trace_by_scores(group: list[dict], scores: list[float]) -> dict:
    best_trace = None
    best_key = None
    for trace, score in zip(group, scores):
        key = (
            float(score),
            trace_mean_logprob(trace),
            -int(trace.get("sample_id", 0)),
            -int(trace.get("trace_id", trace["idx"])),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_trace = trace
    return best_trace


def majority_vote_success(group: list[dict]) -> float:
    parsed = [trace for trace in group if trace.get("predicted_answer") not in (None, "")]
    if not parsed:
        chosen = choose_trace_by_scores(group, [trace_mean_logprob(trace) for trace in group])
        return float(chosen["is_correct"])

    counts = Counter(trace["predicted_answer"] for trace in parsed)
    answer_stats = []
    for answer, count in counts.items():
        traces = [trace for trace in parsed if trace["predicted_answer"] == answer]
        best_lp = max(trace_mean_logprob(trace) for trace in traces)
        answer_stats.append((count, best_lp, answer))

    _, _, winning_answer = max(answer_stats)
    gold = group[0].get("gold_answer")
    return float(winning_answer == gold)


def random_selector_success(group: list[dict]) -> float:
    return float(np.mean([trace["is_correct"] for trace in group]))


def oracle_selector_success(group: list[dict]) -> float:
    return float(any(trace["is_correct"] for trace in group))


def mean_logprob_available(groups: dict[int, list[dict]]) -> bool:
    for group in groups.values():
        for trace in group:
            if trace_mean_logprob(trace) == float("-inf"):
                return False
    return True


def compute_subject_breakdown(problem_scores: dict[int, float], subject_map: dict[int, str] | None) -> dict:
    if not subject_map:
        return {}
    grouped = defaultdict(list)
    for idx, score in problem_scores.items():
        subject = subject_map.get(int(idx))
        if subject is not None:
            grouped[subject].append(float(score))
    return {
        subject: {
            "pass_at_1": float(np.mean(scores)),
            "n_problems": len(scores),
        }
        for subject, scores in sorted(grouped.items())
    }


def compute_difficulty_breakdown(problem_scores: dict[int, float], level_map: dict[int, int] | None) -> dict:
    if not level_map:
        return {}

    by_level = defaultdict(list)
    for idx, score in problem_scores.items():
        level = level_map.get(int(idx))
        if level in (1, 2, 3, 4, 5):
            by_level[int(level)].append(float(score))

    out = {}
    for level in range(1, 6):
        scores = by_level.get(level, [])
        if scores:
            out[f"level_{level}"] = {
                "pass_at_1": float(np.mean(scores)),
                "n_problems": len(scores),
            }

    grouped = {
        "easy_1-2": by_level.get(1, []) + by_level.get(2, []),
        "medium_3": by_level.get(3, []),
        "hard_4-5": by_level.get(4, []) + by_level.get(5, []),
    }
    for key, scores in grouped.items():
        if scores:
            out[key] = {
                "pass_at_1": float(np.mean(scores)),
                "n_problems": len(scores),
            }
    return out


def pack_selector_result(
    fold_scores: list[float],
    problem_scores: dict[int, float],
    subject_map: dict[int, str] | None = None,
    difficulty_map: dict[int, int] | None = None,
    skipped: bool = False,
    reason: str | None = None,
) -> dict:
    if skipped:
        return {"skipped": True, "reason": reason}
    return {
        "pass_at_1_mean": float(np.mean(fold_scores)),
        "pass_at_1_std": float(np.std(fold_scores)),
        "fold_pass_at_1": [float(score) for score in fold_scores],
        "n_problems": len(problem_scores),
        "subject_breakdown": compute_subject_breakdown(problem_scores, subject_map),
        "difficulty_breakdown": compute_difficulty_breakdown(problem_scores, difficulty_map),
    }


def build_entropy_matrix(traces: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([entropy_features(trace["entropies"]) for trace in traces])
    y = np.array([1 if trace["is_correct"] else 0 for trace in traces])
    return X, y


def build_layer_feature_matrices(
    traces: list[dict], layer: int, ref
) -> tuple[np.ndarray, np.ndarray]:
    pca, mu, cov_inv = ref
    x_mah_rows, x_comb_rows = [], []
    for trace in traces:
        dists = compute_mahal_distances(trace["hiddens"][layer], pca, mu, cov_inv)
        mah = mahal_features(trace["entropies"], dists)
        x_mah_rows.append(mah)
        x_comb_rows.append(entropy_features(trace["entropies"]) + mah)
    return np.array(x_mah_rows), np.array(x_comb_rows)


def flatten_groups(groups: dict[int, list[dict]], problem_ids: list[int]) -> list[dict]:
    traces = []
    for idx in problem_ids:
        traces.extend(groups[idx])
    return traces


def select_problem_ids(problem_ids: list[int], n_splits: int, random_state: int):
    n_splits = min(n_splits, len(problem_ids))
    if n_splits < 2:
        raise ValueError("Need at least 2 eligible problems for Best-of-N evaluation.")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(problem_ids))


def summarize_best_layer(layer_results: dict) -> dict | None:
    best = None
    for layer, result in layer_results.items():
        if result.get("combined", {}).get("skipped"):
            continue
        candidate = {
            "layer": int(layer),
            "combined_pass_at_1": result["combined"]["pass_at_1_mean"],
            "mahal_pass_at_1": result["mahalanobis_only"]["pass_at_1_mean"],
        }
        if best is None or candidate["combined_pass_at_1"] > best["combined_pass_at_1"]:
            best = candidate
    return best


def evaluate_best_of_n(
    groups: dict[int, list[dict]],
    layers: list[int],
    n_value: int,
    pca_dim: int,
    n_splits: int,
    random_state: int,
    subject_map: dict[int, str] | None = None,
    difficulty_map: dict[int, int] | None = None,
) -> dict:
    problem_ids = sorted(groups)
    folds = select_problem_ids(problem_ids, n_splits=n_splits, random_state=random_state)

    base_acc = {
        "random": {"fold_scores": [], "problem_scores": {}},
        "oracle_pass_at_n": {"fold_scores": [], "problem_scores": {}},
        "majority_vote": {"fold_scores": [], "problem_scores": {}},
        "mean_logprob": {"fold_scores": [], "problem_scores": {}},
        "entropy_only": {"fold_scores": [], "problem_scores": {}},
    }
    layer_acc = {
        str(layer): {
            "mahalanobis_only": {"fold_scores": [], "problem_scores": {}},
            "combined": {"fold_scores": [], "problem_scores": {}},
        }
        for layer in layers
    }

    has_mean_logprob = mean_logprob_available(groups)

    for train_index, test_index in folds:
        train_problem_ids = [problem_ids[i] for i in train_index]
        test_problem_ids = [problem_ids[i] for i in test_index]

        train_traces = flatten_groups(groups, train_problem_ids)
        y_train = np.array([1 if trace["is_correct"] else 0 for trace in train_traces])

        X_ent_train, _ = build_entropy_matrix(train_traces)
        ent_model = fit_logistic_selector(X_ent_train, y_train)

        layer_models = {}
        for layer in layers:
            correct_train = [trace for trace in train_traces if trace["is_correct"]]
            ref = fit_mahalanobis_reference_safe(correct_train, layer, pca_dim)
            if ref is None:
                layer_models[str(layer)] = None
                continue
            X_mah_train, X_comb_train = build_layer_feature_matrices(train_traces, layer, ref)
            mah_model = fit_logistic_selector(X_mah_train, y_train)
            comb_model = fit_logistic_selector(X_comb_train, y_train)
            layer_models[str(layer)] = {
                "ref": ref,
                "mah_model": mah_model,
                "comb_model": comb_model,
            }

        fold_scores = {name: [] for name in base_acc}
        layer_fold_scores = {
            str(layer): {"mahalanobis_only": [], "combined": []}
            for layer in layers
        }

        for idx in test_problem_ids:
            group = groups[idx]

            random_score = random_selector_success(group)
            oracle_score = oracle_selector_success(group)
            vote_score = majority_vote_success(group)
            base_acc["random"]["problem_scores"][idx] = random_score
            base_acc["oracle_pass_at_n"]["problem_scores"][idx] = oracle_score
            base_acc["majority_vote"]["problem_scores"][idx] = vote_score
            fold_scores["random"].append(random_score)
            fold_scores["oracle_pass_at_n"].append(oracle_score)
            fold_scores["majority_vote"].append(vote_score)

            if has_mean_logprob:
                chosen = choose_trace_by_scores(group, [trace_mean_logprob(trace) for trace in group])
                score = float(chosen["is_correct"])
                base_acc["mean_logprob"]["problem_scores"][idx] = score
                fold_scores["mean_logprob"].append(score)

            if ent_model is not None:
                X_group = np.array([entropy_features(trace["entropies"]) for trace in group])
                chosen = choose_trace_by_scores(group, predict_selector_scores(ent_model, X_group))
                score = float(chosen["is_correct"])
                base_acc["entropy_only"]["problem_scores"][idx] = score
                fold_scores["entropy_only"].append(score)

            for layer in layers:
                layer_model = layer_models[str(layer)]
                if layer_model is None:
                    continue
                ref = layer_model["ref"]
                X_mah_group, X_comb_group = build_layer_feature_matrices(group, layer, ref)

                mah_model = layer_model["mah_model"]
                if mah_model is not None:
                    chosen = choose_trace_by_scores(group, predict_selector_scores(mah_model, X_mah_group))
                    score = float(chosen["is_correct"])
                    layer_acc[str(layer)]["mahalanobis_only"]["problem_scores"][idx] = score
                    layer_fold_scores[str(layer)]["mahalanobis_only"].append(score)

                comb_model = layer_model["comb_model"]
                if comb_model is not None:
                    chosen = choose_trace_by_scores(group, predict_selector_scores(comb_model, X_comb_group))
                    score = float(chosen["is_correct"])
                    layer_acc[str(layer)]["combined"]["problem_scores"][idx] = score
                    layer_fold_scores[str(layer)]["combined"].append(score)

        for name, values in fold_scores.items():
            if values:
                base_acc[name]["fold_scores"].append(float(np.mean(values)))
        for layer in layers:
            for selector in ("mahalanobis_only", "combined"):
                values = layer_fold_scores[str(layer)][selector]
                if values:
                    layer_acc[str(layer)][selector]["fold_scores"].append(float(np.mean(values)))

    result = {
        "n": n_value,
        "n_problems": len(problem_ids),
        "selectors": {
            "random": pack_selector_result(
                base_acc["random"]["fold_scores"],
                base_acc["random"]["problem_scores"],
                subject_map,
                difficulty_map,
            ),
            "oracle_pass_at_n": pack_selector_result(
                base_acc["oracle_pass_at_n"]["fold_scores"],
                base_acc["oracle_pass_at_n"]["problem_scores"],
                subject_map,
                difficulty_map,
            ),
            "majority_vote": pack_selector_result(
                base_acc["majority_vote"]["fold_scores"],
                base_acc["majority_vote"]["problem_scores"],
                subject_map,
                difficulty_map,
            ),
            "mean_logprob": pack_selector_result(
                base_acc["mean_logprob"]["fold_scores"],
                base_acc["mean_logprob"]["problem_scores"],
                subject_map,
                difficulty_map,
                skipped=not has_mean_logprob,
                reason="token log-probs unavailable in data",
            ),
            "entropy_only": pack_selector_result(
                base_acc["entropy_only"]["fold_scores"],
                base_acc["entropy_only"]["problem_scores"],
                subject_map,
                difficulty_map,
                skipped=len(base_acc["entropy_only"]["fold_scores"]) == 0,
                reason="entropy selector could not be fit",
            ),
        },
        "layers": {},
    }

    for layer in layers:
        result["layers"][str(layer)] = {
            "mahalanobis_only": pack_selector_result(
                layer_acc[str(layer)]["mahalanobis_only"]["fold_scores"],
                layer_acc[str(layer)]["mahalanobis_only"]["problem_scores"],
                subject_map,
                difficulty_map,
                skipped=len(layer_acc[str(layer)]["mahalanobis_only"]["fold_scores"]) == 0,
                reason="Mahalanobis selector could not be fit",
            ),
            "combined": pack_selector_result(
                layer_acc[str(layer)]["combined"]["fold_scores"],
                layer_acc[str(layer)]["combined"]["problem_scores"],
                subject_map,
                difficulty_map,
                skipped=len(layer_acc[str(layer)]["combined"]["fold_scores"]) == 0,
                reason="combined selector could not be fit",
            ),
        }

    best = summarize_best_layer(result["layers"])
    if best is not None:
        result["best_combined_layer"] = best
    return result


def plot_best_of_n(results: dict, output_path: str) -> None:
    n_keys = [
        key for key in sorted(results["n_values"], key=int)
        if not results["n_values"][key].get("skipped")
    ]
    if not n_keys:
        print("No Best-of-N results to plot.")
        return

    ns = [int(key) for key in n_keys]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    def series(selector_name):
        ys = []
        for key in n_keys:
            res = results["n_values"][key]["selectors"][selector_name]
            ys.append(None if res.get("skipped") else res["pass_at_1_mean"])
        return ys

    ax.plot(ns, series("random"), "o--", color="#555555", label="Random")
    ax.plot(ns, series("oracle_pass_at_n"), "o-", color="#111111", linewidth=2, label="Oracle Pass@N")
    ax.plot(ns, series("majority_vote"), "o-", color="#4c78a8", linewidth=2, label="Majority vote")

    logprob = series("mean_logprob")
    if any(v is not None for v in logprob):
        ax.plot(ns, logprob, "o-", color="#f58518", linewidth=2, label="Mean log-prob")

    entropy = series("entropy_only")
    if any(v is not None for v in entropy):
        ax.plot(ns, entropy, "o-", color="#54a24b", linewidth=2, label="Entropy-only")

    layers = sorted(results.get("layers", []), key=int)
    palette = ["#e45756", "#72b7b2", "#b279a2", "#ff9da6", "#9d755d"]
    for idx, layer in enumerate(layers):
        ys = []
        for key in n_keys:
            lr = results["n_values"][key]["layers"][str(layer)]["combined"]
            ys.append(None if lr.get("skipped") else lr["pass_at_1_mean"])
        if any(v is not None for v in ys):
            ax.plot(ns, ys, "o-", color=palette[idx % len(palette)], linewidth=2, label=f"Combined L{layer}")

    ax.set_xlabel("N")
    ax.set_ylabel("Pass@1")
    ax.set_title(f"Best-of-N selection on {results['dataset']} ({results['model_label']})")
    ax.set_xticks(ns)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot written to {output_path}")


def write_summary_markdown(results: dict, output_path: str) -> None:
    n_keys = sorted(results["n_values"], key=int)
    layers = sorted(results.get("layers", []), key=int)

    lines = [
        "# Best-of-N Summary",
        "",
        f"*Dataset:* `{results['dataset']}`  ",
        f"*Trace source:* `{results['data_dir']}`",
        "",
        "| N | Problems | Random | Oracle Pass@N | Majority Vote | Mean Log-Prob | Entropy | "
        + " | ".join(f"L{layer} Mahal | L{layer} Combined" for layer in layers)
        + " |",
        "|---|---|---|---|---|---|---|"
        + "|".join(["---|---"] * len(layers))
        + "|",
    ]

    def fmt_selector(res: dict) -> str:
        return "—" if res.get("skipped") else f"{res['pass_at_1_mean']:.3f}"

    for key in n_keys:
        row = results["n_values"][key]
        if row.get("skipped"):
            lines.append(f"| {key} | — | — | — | — | — | — |" + " — | — |" * len(layers))
            continue
        cells = [
            key,
            str(row["n_problems"]),
            fmt_selector(row["selectors"]["random"]),
            fmt_selector(row["selectors"]["oracle_pass_at_n"]),
            fmt_selector(row["selectors"]["majority_vote"]),
            fmt_selector(row["selectors"]["mean_logprob"]),
            fmt_selector(row["selectors"]["entropy_only"]),
        ]
        for layer in layers:
            cells.append(fmt_selector(row["layers"][str(layer)]["mahalanobis_only"]))
            cells.append(fmt_selector(row["layers"][str(layer)]["combined"]))
        lines.append("| " + " | ".join(cells) + " |")

    has_difficulty = False
    for key in n_keys:
        row = results["n_values"][key]
        if row.get("skipped"):
            continue
        breakdown = row["selectors"]["random"].get("difficulty_breakdown", {})
        if any(f"level_{i}" in breakdown for i in range(1, 6)):
            has_difficulty = True
            break

    if has_difficulty:
        lines.extend([
            "",
            "## Difficulty-Stratified Best-of-N (MATH-500)",
            "",
            "| N | Level | Problems | Random | Oracle Pass@N | Majority Vote | Entropy | Best Mahal | Best Combined |",
            "|---|---|---|---|---|---|---|---|---|",
        ])

        def level_value(selector_dict: dict, level_key: str) -> float | None:
            if selector_dict.get("skipped"):
                return None
            diff = selector_dict.get("difficulty_breakdown", {})
            if level_key not in diff:
                return None
            return float(diff[level_key]["pass_at_1"])

        def level_score(selector_dict: dict, level_key: str) -> str:
            value = level_value(selector_dict, level_key)
            return "—" if value is None else f"{value:.3f}"

        for key in n_keys:
            row = results["n_values"][key]
            if row.get("skipped"):
                continue
            for level in range(1, 6):
                level_key = f"level_{level}"
                base_diff = row["selectors"]["random"].get("difficulty_breakdown", {})
                if level_key not in base_diff:
                    continue
                n_problems = base_diff[level_key]["n_problems"]

                best_mah = None
                best_comb = None
                for layer in layers:
                    layer_res = row["layers"][str(layer)]
                    mah_val = level_value(layer_res["mahalanobis_only"], level_key)
                    comb_val = level_value(layer_res["combined"], level_key)
                    if mah_val is not None and (best_mah is None or mah_val > best_mah):
                        best_mah = mah_val
                    if comb_val is not None and (best_comb is None or comb_val > best_comb):
                        best_comb = comb_val

                best_mah_str = "—" if best_mah is None else f"{best_mah:.3f}"
                best_comb_str = "—" if best_comb is None else f"{best_comb:.3f}"

                lines.append(
                    "| "
                    + " | ".join(
                        [
                            key,
                            str(level),
                            str(n_problems),
                            level_score(row["selectors"]["random"], level_key),
                            level_score(row["selectors"]["oracle_pass_at_n"], level_key),
                            level_score(row["selectors"]["majority_vote"], level_key),
                            level_score(row["selectors"]["entropy_only"], level_key),
                            best_mah_str,
                            best_comb_str,
                        ]
                    )
                    + " |"
                )

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Summary written to {output_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    layers = parse_int_list(args.layers) if args.layers else detect_layers(args.data_dir)
    n_values = parse_int_list(args.n_values)
    traces = load_all_traces(args.data_dir, layers)
    grouped = group_traces_by_problem(traces)
    subject_map = None
    difficulty_map = None
    if args.dataset_label == "math500":
        try:
            subject_map = load_math500_subjects()
        except Exception:
            print("Warning: could not load MATH-500 subjects; subject breakdown will be skipped.")
        try:
            difficulty_map = load_math500_levels()
        except Exception:
            print("Warning: could not load MATH-500 levels; difficulty breakdown will be skipped.")

    results = {
        "dataset": args.dataset_label,
        "data_dir": args.data_dir,
        "model_label": os.path.basename(os.path.dirname(args.data_dir)) or args.data_dir,
        "layers": [int(layer) for layer in layers],
        "leakage_protocol": "selectors and Mahalanobis references are fit on train-fold traces only; Mahalanobis reference uses train-fold correct traces only",
        "n_values": {},
    }

    print(f"Loaded {len(traces)} traces across {len(grouped)} problems")
    print(f"Evaluating layers: {layers}")

    for n_value in n_values:
        subset = subset_groups_for_n(grouped, n_value)
        if len(subset) < 2:
            print(f"Skipping N={n_value}: only {len(subset)} eligible problems")
            results["n_values"][str(n_value)] = {
                "skipped": True,
                "reason": "not enough problems with at least N traces",
            }
            continue
        print(f"\nEvaluating Best-of-{n_value} on {len(subset)} problems")
        results["n_values"][str(n_value)] = evaluate_best_of_n(
            subset,
            layers=layers,
            n_value=n_value,
            pca_dim=args.pca_dim,
            n_splits=args.n_splits,
            random_state=args.cv_random_state,
            subject_map=subject_map,
            difficulty_map=difficulty_map,
        )

    json_path = os.path.join(args.output_dir, f"{args.dataset_label}_best_of_n_results.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results written to {json_path}")

    plot_best_of_n(results, os.path.join(args.output_dir, f"{args.dataset_label}_best_of_n.png"))
    write_summary_markdown(
        results, os.path.join(args.output_dir, f"{args.dataset_label}_best_of_n_summary.md")
    )


if __name__ == "__main__":
    main()
