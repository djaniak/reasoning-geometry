"""
analyze.py — Mahalanobis geometry vs entropy for reasoning trace correctness.

Key design: entropy-only AUC is computed ONCE with fixed CV folds and reused as
the baseline for all layer comparisons. This ensures the delta is meaningful.

Mahalanobis reference (PCA + Gaussian on correct traces) is fit **inside each
CV train fold** on correct training traces only, then applied to all traces for
that fold's classifier. This avoids label leakage from test-fold correct traces
into the geometry features.
"""
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu

OUTPUT_DIR = "collected_data"
PCA_DIM = 128
CV_RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for results JSON and plots (defaults to data_dir)")
    parser.add_argument("--pca_dim", type=int, default=PCA_DIM)
    parser.add_argument("--dataset_label", type=str, default="gsm8k",
                        help="Label for output files (gsm8k or math500)")
    parser.add_argument("--post_fork", action="store_true")
    parser.add_argument("--narrow_ref", action="store_true")
    parser.add_argument("--difficulty", action="store_true")
    parser.add_argument("--all_analyses", action="store_true")
    parser.add_argument("--n_bootstrap", type=int, default=200,
                        help="Bootstrap iterations for difficulty CI (0=skip)")
    parser.add_argument("--cv_random_state", type=int, default=CV_RANDOM_STATE)
    parser.add_argument("--cross_model_ref", type=str, default=None,
                        help="Data dir of another model (same arch) to fit cross-model Mahalanobis reference")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to analyze (auto-detected from data if omitted)")
    parser.add_argument("--subject", action="store_true",
                        help="Run subject-category stratification (MATH-500 only)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading — load all layers in one pass
# ---------------------------------------------------------------------------

def detect_layers(data_dir: str) -> list[int]:
    """Auto-detect available layer indices from the first .npz file."""
    import re as _re
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(data_dir, fname)
        try:
            data = np.load(path, allow_pickle=True)
        except Exception:
            continue
        layers = sorted({int(m.group(1)) for k in data.files
                         if (m := _re.match(r"hidden_L(\d+)_", k))})
        if layers:
            return layers
    raise RuntimeError(f"No hidden state layers found in {data_dir}")


def load_all_traces(data_dir: str, layers: list[int]) -> list[dict]:
    """Load traces with hidden states for the specified layers."""
    traces = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(data_dir, fname)
        try:
            data = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"  Skipping {fname} ({e})")
            continue
        for m in data["metadata"]:
            idx = m["idx"]
            trace_id = m["trace_id"] if "trace_id" in m else idx
            ent_key = f"entropies_{trace_id}" if f"entropies_{trace_id}" in data else f"entropies_{idx}"
            lp_key = f"token_logprobs_{trace_id}" if f"token_logprobs_{trace_id}" in data else None
            tok_key = f"tokens_{trace_id}" if f"tokens_{trace_id}" in data else (f"tokens_{idx}" if f"tokens_{idx}" in data else None)
            trace = {
                "trace_id": trace_id,
                "idx": idx,
                "sample_id": m["sample_id"] if "sample_id" in m else 0,
                "is_correct": bool(m["is_correct"]),
                "gold_answer": m["gold"] if "gold" in m else None,
                "predicted_answer": m["predicted"] if "predicted" in m else None,
                "mean_logprob": m["mean_logprob"] if "mean_logprob" in m else None,
                "generation_seed": m["seed"] if "seed" in m else None,
                "entropies": data[ent_key],
                "hiddens": {},
            }
            trace["token_logprobs"] = data[lp_key] if lp_key and lp_key in data else None
            trace["tokens"] = [str(x) for x in data[tok_key].tolist()] if tok_key and tok_key in data else None
            for layer in layers:
                key = f"hidden_L{layer}_{trace_id}"
                if key not in data:
                    key = f"hidden_L{layer}_{idx}"
                if key in data:
                    trace["hiddens"][layer] = data[key]
            traces.append(trace)
    return traces


# ---------------------------------------------------------------------------
# Mahalanobis fitting
# ---------------------------------------------------------------------------

def fit_mahalanobis_reference(correct_traces: list[dict], layer: int, pca_dim: int):
    correct_hiddens = np.concatenate([t["hiddens"][layer] for t in correct_traces], axis=0)
    svd_solver = "randomized" if correct_hiddens.shape[0] > 200_000 else "full"
    pca = PCA(n_components=pca_dim, random_state=42, svd_solver=svd_solver)
    pca.fit(correct_hiddens)

    projected = pca.transform(correct_hiddens)
    mu = projected.mean(axis=0)
    cov = np.cov(projected - mu, rowvar=False)
    cov_reg = cov + 1e-4 * np.eye(pca_dim)
    cov_inv = np.linalg.inv(cov_reg)

    print(f"  Layer {layer}: PCA var={pca.explained_variance_ratio_.sum():.3f}, "
          f"cov cond={np.linalg.cond(cov_reg):.1f}, "
          f"fit on {correct_hiddens.shape[0]} tokens")
    return pca, mu, cov_inv


def fit_mahalanobis_reference_narrow(
    correct_traces: list[dict], layer: int, pca_dim: int, ent_percentile: int = 50
):
    """Fit Mahalanobis reference using only high-entropy tokens from correct traces."""
    all_ent = np.concatenate([t["entropies"] for t in correct_traces])
    threshold = np.percentile(all_ent, ent_percentile)

    filtered = []
    for t in correct_traces:
        mask = t["entropies"] > threshold
        if mask.any():
            filtered.append(t["hiddens"][layer][mask])
    filtered_hiddens = np.concatenate(filtered, axis=0)

    svd_solver = "randomized" if filtered_hiddens.shape[0] > 200_000 else "full"
    pca = PCA(n_components=pca_dim, random_state=42, svd_solver=svd_solver)
    pca.fit(filtered_hiddens)

    projected = pca.transform(filtered_hiddens)
    mu = projected.mean(axis=0)
    cov = np.cov(projected - mu, rowvar=False)
    cov_reg = cov + 1e-4 * np.eye(pca_dim)
    cov_inv = np.linalg.inv(cov_reg)

    print(f"  Layer {layer} (narrow): PCA var={pca.explained_variance_ratio_.sum():.3f}, "
          f"cov cond={np.linalg.cond(cov_reg):.1f}, "
          f"fit on {filtered_hiddens.shape[0]} tokens (>{ent_percentile}th pctl entropy)")
    return pca, mu, cov_inv


def compute_mahal_distances(hiddens: np.ndarray, pca, mu, cov_inv) -> np.ndarray:
    projected = pca.transform(hiddens)
    diff = projected - mu
    dists_sq = np.sum((diff @ cov_inv) * diff, axis=1)
    return np.sqrt(np.maximum(dists_sq, 0))


def fit_mahalanobis_reference_safe(correct_traces: list[dict], layer: int, pca_dim: int):
    """Like fit_mahalanobis_reference but returns None on failure (e.g. too few tokens)."""
    if not correct_traces:
        return None
    try:
        return fit_mahalanobis_reference(correct_traces, layer, pca_dim)
    except Exception:
        return None


def fit_mahalanobis_reference_narrow_safe(
    correct_traces: list[dict], layer: int, pca_dim: int, ent_percentile: int = 50
):
    if not correct_traces:
        return None
    try:
        return fit_mahalanobis_reference_narrow(correct_traces, layer, pca_dim, ent_percentile)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

ENTROPY_FEATURE_NAMES = [
    "ent_mean", "ent_max", "ent_std", "ent_q90", "ent_frac_high",
]
MAHAL_FEATURE_NAMES = [
    "mah_mean", "mah_max", "mah_std", "mah_q90",
    "mah_at_high_ent_mean", "mah_at_high_ent_max",
    "ent_mah_corr",
]


def entropy_features(e: np.ndarray) -> list:
    median_e = np.median(e)
    return [
        e.mean(),
        e.max(),
        e.std(),
        np.percentile(e, 90),
        (e > median_e).mean(),
    ]


def mahal_features(e: np.ndarray, m: np.ndarray) -> list:
    median_e = np.median(e)
    high_mask = e > median_e
    return [
        m.mean(),
        m.max(),
        m.std(),
        np.percentile(m, 90),
        m[high_mask].mean() if high_mask.any() else 0.0,
        m[high_mask].max() if high_mask.any() else 0.0,
        np.corrcoef(e, m)[0, 1] if len(e) > 2 else 0.0,
    ]


def build_feature_matrix(traces: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract entropy-only features. Returns X_ent [N, 5] and y [N]."""
    rows, labels = [], []
    for t in traces:
        rows.append(entropy_features(t["entropies"]))
        labels.append(1 if t["is_correct"] else 0)
    return np.array(rows), np.array(labels)


# ---------------------------------------------------------------------------
# CV evaluation — shared fold indices
# ---------------------------------------------------------------------------

def evaluate_transfer(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    label: str,
) -> dict:
    """Train on one dataset, evaluate on another (no CV — full train/test split)."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    clf.fit(X_tr, y_train)
    probs = clf.predict_proba(X_te)[:, 1]
    roc = roc_auc_score(y_test, probs)
    pr = average_precision_score(y_test, probs)
    result = {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
    }
    print(f"    {label:30s}: ROC-AUC = {roc:.4f} | PR-AUC = {pr:.4f}")
    return result


def evaluate_features(
    X: np.ndarray,
    y: np.ndarray,
    fold_indices: list[tuple],
    label: str,
) -> dict:
    roc_aucs, pr_aucs = [], []
    for train_idx, test_idx in fold_indices:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        clf.fit(X_train, y[train_idx])
        probs = clf.predict_proba(X_test)[:, 1]
        roc_aucs.append(roc_auc_score(y[test_idx], probs))
        pr_aucs.append(average_precision_score(y[test_idx], probs))

    result = {
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs)),
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs)),
        "fold_roc_aucs": [float(v) for v in roc_aucs],
    }
    print(f"    {label:30s}: ROC-AUC = {result['roc_auc_mean']:.4f} ± {result['roc_auc_std']:.4f} | "
          f"PR-AUC = {result['pr_auc_mean']:.4f} ± {result['pr_auc_std']:.4f}")
    return result


def _fold_clf_auc(X: np.ndarray, y: np.ndarray, train_idx, test_idx):
    """Train scaler + balanced LR on train fold; return ROC-AUC, PR-AUC on test, or None."""
    if len(np.unique(y[test_idx])) < 2:
        return None
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    clf.fit(X_train, y[train_idx])
    probs = clf.predict_proba(X_test)[:, 1]
    return (
        roc_auc_score(y[test_idx], probs),
        average_precision_score(y[test_idx], probs),
    )


def evaluate_foldwise_mahalanobis(
    traces: list[dict],
    layer: int,
    pca_dim: int,
    y: np.ndarray,
    fold_indices: list[tuple],
    label_prefix: str = "",
) -> dict:
    """CV where each train fold fits PCA+Gaussian on **train-fold correct** traces only."""
    roc_mah, pr_mah = [], []
    roc_comb, pr_comb = [], []
    roc_lcomb, pr_lcomb = [], []

    for fi, (train_idx, test_idx) in enumerate(fold_indices):
        correct_train = [traces[i] for i in train_idx if traces[i]["is_correct"]]
        ref = fit_mahalanobis_reference_safe(correct_train, layer, pca_dim)
        if ref is None:
            print(f"    WARNING: skip fold {fi} — Mahalanobis ref failed (layer {layer})")
            continue
        pca, mu, cov_inv = ref

        X_mah, X_comb, X_lcomb = [], [], []
        for trace in traces:
            m = compute_mahal_distances(trace["hiddens"][layer], pca, mu, cov_inv)
            e = trace["entropies"]
            ef = entropy_features(e)
            mf = mahal_features(e, m)
            log_len = [np.log1p(len(e))]
            X_mah.append(mf)
            X_comb.append(ef + mf)
            X_lcomb.append(ef + mf + log_len)
        X_mah = np.array(X_mah)
        X_comb = np.array(X_comb)
        X_lcomb = np.array(X_lcomb)

        for X, roc_list, pr_list in (
            (X_mah, roc_mah, pr_mah),
            (X_comb, roc_comb, pr_comb),
            (X_lcomb, roc_lcomb, pr_lcomb),
        ):
            out = _fold_clf_auc(X, y, train_idx, test_idx)
            if out is None:
                continue
            roc, pr = out
            roc_list.append(roc)
            pr_list.append(pr)

    def pack(roc_list, pr_list, name: str) -> dict:
        if not roc_list:
            print(f"    {label_prefix}{name}: no valid folds")
            return {
                "roc_auc_mean": float("nan"),
                "roc_auc_std": float("nan"),
                "pr_auc_mean": float("nan"),
                "pr_auc_std": float("nan"),
                "fold_roc_aucs": [],
            }
        r = {
            "roc_auc_mean": float(np.mean(roc_list)),
            "roc_auc_std": float(np.std(roc_list)),
            "pr_auc_mean": float(np.mean(pr_list)),
            "pr_auc_std": float(np.std(pr_list)),
            "fold_roc_aucs": [float(v) for v in roc_list],
        }
        print(f"    {label_prefix}{name:30s}: ROC-AUC = {r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f} | "
              f"PR-AUC = {r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f}")
        return r

    return {
        "mahalanobis_only": pack(roc_mah, pr_mah, "mahalanobis_only"),
        "combined": pack(roc_comb, pr_comb, "combined"),
        "combined_with_length": pack(roc_lcomb, pr_lcomb, "combined+length"),
    }


def evaluate_foldwise_narrow_mahalanobis(
    traces: list[dict],
    layer: int,
    pca_dim: int,
    y: np.ndarray,
    fold_indices: list[tuple],
    label_prefix: str = "",
    ent_percentile: int = 50,
) -> dict:
    """Same as evaluate_foldwise_mahalanobis but reference uses high-entropy tokens on train correct."""
    roc_nmah, pr_nmah = [], []
    roc_ncomb, pr_ncomb = [], []

    for fi, (train_idx, test_idx) in enumerate(fold_indices):
        correct_train = [traces[i] for i in train_idx if traces[i]["is_correct"]]
        ref = fit_mahalanobis_reference_narrow_safe(
            correct_train, layer, pca_dim, ent_percentile
        )
        if ref is None:
            print(f"    WARNING: skip fold {fi} — narrow Mahalanobis ref failed (layer {layer})")
            continue
        npca, nmu, ncov_inv = ref

        X_nmah, X_ncomb = [], []
        for trace in traces:
            m = compute_mahal_distances(trace["hiddens"][layer], npca, nmu, ncov_inv)
            e = trace["entropies"]
            mf = mahal_features(e, m)
            X_nmah.append(mf)
            X_ncomb.append(entropy_features(e) + mf)
        X_nmah = np.array(X_nmah)
        X_ncomb = np.array(X_ncomb)

        out_m = _fold_clf_auc(X_nmah, y, train_idx, test_idx)
        if out_m:
            roc_nmah.append(out_m[0])
            pr_nmah.append(out_m[1])
        out_c = _fold_clf_auc(X_ncomb, y, train_idx, test_idx)
        if out_c:
            roc_ncomb.append(out_c[0])
            pr_ncomb.append(out_c[1])

    def pack(roc_list, pr_list, name: str) -> dict:
        if not roc_list:
            print(f"    {label_prefix}{name}: no valid folds")
            return {
                "roc_auc_mean": float("nan"),
                "roc_auc_std": float("nan"),
                "pr_auc_mean": float("nan"),
                "pr_auc_std": float("nan"),
                "fold_roc_aucs": [],
            }
        r = {
            "roc_auc_mean": float(np.mean(roc_list)),
            "roc_auc_std": float(np.std(roc_list)),
            "pr_auc_mean": float(np.mean(pr_list)),
            "pr_auc_std": float(np.std(pr_list)),
            "fold_roc_aucs": [float(v) for v in roc_list],
        }
        print(f"    {label_prefix}{name:30s}: ROC-AUC = {r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f} | "
              f"PR-AUC = {r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f}")
        return r

    return {
        "narrow_mahal_only": pack(roc_nmah, pr_nmah, "narrow_mahal_only"),
        "narrow_combined": pack(roc_ncomb, pr_ncomb, "narrow_combined"),
    }


def assign_oof_mahalanobis_distances(
    traces: list[dict], layer: int, pca_dim: int, fold_indices: list[tuple]
) -> None:
    """Set trace['mahal_dists'] using the CV fold where the trace is in the test set (no label leakage)."""
    n = len(traces)
    oof = [None] * n
    for train_idx, test_idx in fold_indices:
        correct_train = [traces[i] for i in train_idx if traces[i]["is_correct"]]
        ref = fit_mahalanobis_reference_safe(correct_train, layer, pca_dim)
        if ref is None:
            continue
        pca, mu, cov_inv = ref
        for i in test_idx:
            oof[i] = compute_mahal_distances(traces[i]["hiddens"][layer], pca, mu, cov_inv)
    # Fallback: any index never scored (skipped fold) — use full-train ref from first fold's train correct
    missing = [i for i in range(n) if oof[i] is None]
    if missing:
        train_idx, _ = fold_indices[0]
        correct_train = [traces[i] for i in train_idx if traces[i]["is_correct"]]
        ref = fit_mahalanobis_reference_safe(correct_train, layer, pca_dim)
        if ref:
            pca, mu, cov_inv = ref
            for i in missing:
                oof[i] = compute_mahal_distances(traces[i]["hiddens"][layer], pca, mu, cov_inv)
    for i, t in enumerate(traces):
        t["mahal_dists"] = oof[i] if oof[i] is not None else np.array([], dtype=np.float64)


# ---------------------------------------------------------------------------
# Stratified analysis: confidently wrong tokens
# ---------------------------------------------------------------------------

def analyze_confident_wrong(
    traces: list[dict],
    layer: int,
    pca_dim: int,
    fold_indices: list[tuple],
) -> dict:
    """
    Are incorrect traces geometrically off-manifold even at low-entropy tokens?
    Uses train-fold-only reference and train-derived low-entropy threshold per fold;
    pools statistics from **test-fold** traces only (each trace once).
    """
    correct_vals, incorrect_vals = [], []

    for train_idx, test_idx in fold_indices:
        train_ent = np.concatenate([traces[i]["entropies"] for i in train_idx])
        if train_ent.size == 0:
            continue
        low_ent_threshold = np.percentile(train_ent, 25)

        correct_train = [traces[i] for i in train_idx if traces[i]["is_correct"]]
        ref = fit_mahalanobis_reference_safe(correct_train, layer, pca_dim)
        if ref is None:
            continue
        pca, mu, cov_inv = ref

        for i in test_idx:
            trace = traces[i]
            m = compute_mahal_distances(trace["hiddens"][layer], pca, mu, cov_inv)
            low_mask = trace["entropies"] < low_ent_threshold
            if not low_mask.any():
                continue
            avg = float(m[low_mask].mean())
            (correct_vals if trace["is_correct"] else incorrect_vals).append(avg)

    if len(correct_vals) < 2 or len(incorrect_vals) < 2:
        return {
            "correct_mean": None,
            "correct_std": None,
            "incorrect_mean": None,
            "incorrect_std": None,
            "n_correct": len(correct_vals),
            "n_incorrect": len(incorrect_vals),
            "mannwhitney_pvalue": None,
            "mannwhitney_statistic": None,
            "skipped": True,
        }

    stat, pval = mannwhitneyu(incorrect_vals, correct_vals, alternative="greater")

    result = {
        "correct_mean": float(np.mean(correct_vals)),
        "correct_std": float(np.std(correct_vals)),
        "incorrect_mean": float(np.mean(incorrect_vals)),
        "incorrect_std": float(np.std(incorrect_vals)),
        "n_correct": len(correct_vals),
        "n_incorrect": len(incorrect_vals),
        "mannwhitney_pvalue": float(pval),
        "mannwhitney_statistic": float(stat),
    }

    print(f"\n    Confident-wrong analysis (low-entropy tokens, bottom-25%):")
    print(f"      Correct   traces: {result['correct_mean']:.3f} ± {result['correct_std']:.3f} (n={len(correct_vals)})")
    print(f"      Incorrect traces: {result['incorrect_mean']:.3f} ± {result['incorrect_std']:.3f} (n={len(incorrect_vals)})")
    print(f"      Mann-Whitney U p={pval:.2e}", end="  ")
    if pval < 0.01:
        print("*** STRONG: geometry detects off-manifold states where entropy is blind")
    elif pval < 0.05:
        print("*   MARGINAL: p<0.05, verify with more data")
    else:
        print("    NOT SIGNIFICANT")

    return result


def analyze_post_fork(
    traces: list[dict],
    layer: int,
    pca_dim: int,
    fold_indices: list[tuple],
    window: int = 5,
    percentile: int = 75,
) -> dict:
    """
    After high-entropy 'fork' tokens, do incorrect traces diverge geometrically?
    Train-fold threshold and Mahalanobis ref; pools **test-fold** traces only.
    """
    correct_vals, incorrect_vals = [], []

    for train_idx, test_idx in fold_indices:
        train_ent = np.concatenate([traces[i]["entropies"] for i in train_idx])
        if train_ent.size == 0:
            continue
        threshold = float(np.percentile(train_ent, percentile))

        correct_train = [traces[i] for i in train_idx if traces[i]["is_correct"]]
        ref = fit_mahalanobis_reference_safe(correct_train, layer, pca_dim)
        if ref is None:
            continue
        pca, mu, cov_inv = ref

        for i in test_idx:
            trace = traces[i]
            ent = trace["entropies"]
            m = compute_mahal_distances(trace["hiddens"][layer], pca, mu, cov_inv)
            fork_indices = np.where(ent > threshold)[0]
            if len(fork_indices) == 0:
                continue
            post_vals = []
            for idx in fork_indices:
                start = idx + 1
                end = min(start + window, len(m))
                if start < end:
                    post_vals.append(m[start:end].mean())
            if not post_vals:
                continue
            avg = float(np.mean(post_vals))
            (correct_vals if trace["is_correct"] else incorrect_vals).append(avg)

    if len(correct_vals) < 2 or len(incorrect_vals) < 2:
        print(f"\n    Post-fork analysis: insufficient data (correct={len(correct_vals)}, incorrect={len(incorrect_vals)})")
        return {"skipped": True}

    stat, pval = mannwhitneyu(incorrect_vals, correct_vals, alternative="greater")

    result = {
        "correct_mean": float(np.mean(correct_vals)),
        "correct_std": float(np.std(correct_vals)),
        "incorrect_mean": float(np.mean(incorrect_vals)),
        "incorrect_std": float(np.std(incorrect_vals)),
        "n_correct": len(correct_vals),
        "n_incorrect": len(incorrect_vals),
        "mannwhitney_pvalue": float(pval),
        "mannwhitney_statistic": float(stat),
        "window": window,
        "percentile": percentile,
        "threshold": float("nan"),
    }

    print(f"\n    Post-fork analysis (window={window} after >{percentile}th pctl entropy, train-fold threshold):")
    print(f"      Correct   traces: {result['correct_mean']:.3f} ± {result['correct_std']:.3f} (n={len(correct_vals)})")
    print(f"      Incorrect traces: {result['incorrect_mean']:.3f} ± {result['incorrect_std']:.3f} (n={len(incorrect_vals)})")
    print(f"      Mann-Whitney U p={pval:.2e}", end="  ")
    if pval < 0.01:
        print("*** STRONG: geometry diverges after fork points")
    elif pval < 0.05:
        print("*   MARGINAL")
    else:
        print("    NOT SIGNIFICANT")

    return result


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _compute_mahal_features_from_projected(
    projected_tokens: np.ndarray, entropies: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray,
) -> list:
    """Mahalanobis features from already-PCA-projected token hidden states."""
    diff = projected_tokens - mu
    dists_sq = np.sum((diff @ cov_inv) * diff, axis=1)
    m = np.sqrt(np.maximum(dists_sq, 0))
    return mahal_features(entropies, m)


def bootstrap_auc_ci(
    subset: list[dict],
    layer: int,
    pca_dim: int,
    n_bootstrap: int = 200,
    cv_splits: int = 3,
    random_state: int = 42,
    shared_pca=None,
) -> dict:
    """95% bootstrap CIs for entropy-only, mahal-only, combined AUC.

    Stratified bootstrap (resample within each class) keeps class proportions.
    Within each bootstrap draw, AUC is estimated by stratified k-fold CV where
    the Gaussian reference (mean + cov) is fit on train-fold correct projected
    tokens only (no label leakage).

    PCA is fitted once on all correct traces upfront — it is a deterministic
    dimensionality reduction step that does not use labels and is stable across
    resamples.  Only the Gaussian (mu, cov_inv) is refit per fold, which is the
    component that could overfit to the reference set.
    """
    del shared_pca  # kept for call-site compatibility

    correct = [t for t in subset if t["is_correct"]]
    incorrect = [t for t in subset if not t["is_correct"]]

    # --- Fit PCA once on all correct traces ---
    correct_hiddens = np.concatenate([t["hiddens"][layer] for t in correct], axis=0)
    svd_solver = "randomized" if correct_hiddens.shape[0] > 200_000 else "full"
    pca = PCA(n_components=pca_dim, random_state=42, svd_solver=svd_solver)
    pca.fit(correct_hiddens)
    del correct_hiddens

    # --- Precompute per-trace projected tokens & entropy features ---
    projected_per_trace = [pca.transform(t["hiddens"][layer]) for t in subset]
    ent_feats_per_trace = [entropy_features(t["entropies"]) for t in subset]
    is_correct = np.array([t["is_correct"] for t in subset])
    entropies_per_trace = [t["entropies"] for t in subset]

    rng = np.random.default_rng(random_state)
    correct_idx = np.where(is_correct)[0]
    incorrect_idx = np.where(~is_correct)[0]
    ent_aucs, mah_aucs, comb_aucs = [], [], []

    for _ in range(n_bootstrap):
        c_sel = rng.integers(0, len(correct_idx), size=len(correct_idx))
        i_sel = rng.integers(0, len(incorrect_idx), size=len(incorrect_idx))
        boot_idx = np.concatenate([correct_idx[c_sel], incorrect_idx[i_sel]])
        y_boot = is_correct[boot_idx].astype(int)

        X_ent_b = np.array([ent_feats_per_trace[i] for i in boot_idx])
        n_cv = min(cv_splits, int(min((y_boot == 0).sum(), (y_boot == 1).sum())))
        if n_cv < 2:
            continue
        skf_b = StratifiedKFold(
            n_splits=n_cv, shuffle=True, random_state=int(rng.integers(0, 10_000))
        )
        folds_b = list(skf_b.split(X_ent_b, y_boot))

        ent_fold_aucs, mah_fold_aucs, comb_fold_aucs = [], [], []

        for tr, te in folds_b:
            if len(np.unique(y_boot[te])) < 2:
                continue

            # Fit Gaussian on train-fold correct projected tokens only
            train_correct_mask = y_boot[tr] == 1
            train_correct_projected = np.concatenate(
                [projected_per_trace[boot_idx[j]] for j in tr[train_correct_mask]]
            )
            mu = train_correct_projected.mean(axis=0)
            cov = np.cov(train_correct_projected - mu, rowvar=False)
            cov_inv = np.linalg.inv(cov + 1e-4 * np.eye(pca_dim))

            # Compute Mahalanobis features for all boot traces
            X_mah_b = np.array([
                _compute_mahal_features_from_projected(
                    projected_per_trace[boot_idx[j]], entropies_per_trace[boot_idx[j]], mu, cov_inv,
                ) for j in range(len(boot_idx))
            ])
            X_comb_b = np.hstack([X_ent_b, X_mah_b])

            out_e = _fold_clf_auc(X_ent_b, y_boot, tr, te)
            if out_e:
                ent_fold_aucs.append(out_e[0])
            out_m = _fold_clf_auc(X_mah_b, y_boot, tr, te)
            if out_m:
                mah_fold_aucs.append(out_m[0])
            out_c = _fold_clf_auc(X_comb_b, y_boot, tr, te)
            if out_c:
                comb_fold_aucs.append(out_c[0])

        if ent_fold_aucs:
            ent_aucs.append(float(np.mean(ent_fold_aucs)))
        if mah_fold_aucs:
            mah_aucs.append(float(np.mean(mah_fold_aucs)))
        if comb_fold_aucs:
            comb_aucs.append(float(np.mean(comb_fold_aucs)))

    def ci95(vals):
        if len(vals) < 20:
            return (float("nan"), float("nan"))
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    return {
        "entropy_ci95": ci95(ent_aucs),
        "mahal_ci95": ci95(mah_aucs),
        "combined_ci95": ci95(comb_aucs),
        "n_valid": len(ent_aucs),
    }


# ---------------------------------------------------------------------------
# Difficulty stratification
# ---------------------------------------------------------------------------

def load_math500_levels() -> dict[int, int]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return {i: ex["level"] for i, ex in enumerate(ds)}


def load_math500_subjects() -> dict[int, str]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return {i: ex["subject"] for i, ex in enumerate(ds)}


def analyze_by_difficulty(
    traces, idx_to_level, pca_dim, layer: int,
    n_bootstrap: int = 0, cv_random_state: int = CV_RANDOM_STATE,
):
    """Stratify traces by difficulty level and evaluate each stratum.

    Args:
        layer: which layer's hiddens to use (pass best_layer from the main loop)
        n_bootstrap: bootstrap iterations for 95% CI (0 = skip)
    """
    for t in traces:
        t["level"] = idx_to_level.get(t["idx"], None)

    labeled = [t for t in traces if t["level"] is not None]
    if not labeled:
        print("  No traces matched difficulty levels.")
        return {}

    # Individual levels + grouped buckets
    buckets = {}
    for lv in range(1, 6):
        subset = [t for t in labeled if t["level"] == lv]
        if subset:
            buckets[f"level_{lv}"] = subset
    buckets["easy_1-2"] = [t for t in labeled if t["level"] in (1, 2)]
    buckets["medium_3"] = [t for t in labeled if t["level"] == 3]
    buckets["hard_4-5"] = [t for t in labeled if t["level"] in (4, 5)]

    results = {}
    for bucket_name, subset in buckets.items():
        n_correct = sum(1 for t in subset if t["is_correct"])
        n_incorrect = len(subset) - n_correct
        if n_correct < 5 or n_incorrect < 5:
            print(f"    {bucket_name}: skipped (correct={n_correct}, incorrect={n_incorrect})")
            results[bucket_name] = {"skipped": True, "n_correct": n_correct, "n_incorrect": n_incorrect}
            continue

        y_sub = np.array([1 if t["is_correct"] else 0 for t in subset])
        X_ent_sub = np.array([entropy_features(t["entropies"]) for t in subset])

        n_splits = max(3, min(5, min(n_correct, n_incorrect)))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_random_state)
        folds = list(skf.split(X_ent_sub, y_sub))

        print(f"\n    {bucket_name} (n={len(subset)}, correct={n_correct}, incorrect={n_incorrect}, layer={layer}):")
        ent_res = evaluate_features(X_ent_sub, y_sub, folds, f"{bucket_name} entropy")
        fw = evaluate_foldwise_mahalanobis(
            subset, layer, pca_dim, y_sub, folds, label_prefix=f"{bucket_name} "
        )
        mah_res = fw["mahalanobis_only"]
        comb_res = fw["combined"]

        delta = comb_res["roc_auc_mean"] - ent_res["roc_auc_mean"]
        print(f"      Delta: {delta:+.4f}", end="")

        bucket_result = {
            "n_total": len(subset),
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "entropy_only": ent_res,
            "mahalanobis_only": mah_res,
            "combined": comb_res,
            "delta": float(delta),
        }

        # Bootstrap CIs
        if n_bootstrap > 0:
            print(f"  [bootstrapping n={n_bootstrap}...]", end="", flush=True)
            ci = bootstrap_auc_ci(subset, layer, pca_dim, n_bootstrap)
            bucket_result["bootstrap_ci"] = ci
            print(f"\r      Delta: {delta:+.4f}  "
                  f"combined 95% CI [{ci['combined_ci95'][0]:.3f}, {ci['combined_ci95'][1]:.3f}]  "
                  f"entropy 95% CI [{ci['entropy_ci95'][0]:.3f}, {ci['entropy_ci95'][1]:.3f}]")
        else:
            print()

        results[bucket_name] = bucket_result

    return results


def plot_difficulty_stratification(difficulty_results, output_path):
    """Figure 1: AUC by difficulty level with 95% bootstrap CI error bars."""
    levels = [
        k for k in sorted(difficulty_results)
        if k.startswith("level_") and not difficulty_results[k].get("skipped")
    ]
    if not levels:
        print("  No difficulty levels to plot.")
        return

    has_ci = all("bootstrap_ci" in difficulty_results[l] for l in levels)

    x = np.arange(len(levels))
    ent_means = np.array([difficulty_results[l]["entropy_only"]["roc_auc_mean"] for l in levels])
    comb_means = np.array([difficulty_results[l]["combined"]["roc_auc_mean"] for l in levels])
    deltas = np.array([difficulty_results[l]["delta"] for l in levels])
    ns = [difficulty_results[l]["n_total"] for l in levels]

    # --- asymmetric error bars from CI ---
    def err_bars(key, means):
        lo_err = np.zeros(len(levels))
        hi_err = np.zeros(len(levels))
        for i, lv in enumerate(levels):
            ci = difficulty_results[lv].get("bootstrap_ci", {}).get(key, (float("nan"), float("nan")))
            lo, hi = ci
            lo_err[i] = max(0, means[i] - lo) if not np.isnan(lo) else 0
            hi_err[i] = max(0, hi - means[i]) if not np.isnan(hi) else 0
        return np.array([lo_err, hi_err])

    COLOR_ENT = "#4878CF"   # muted blue
    COLOR_COMB = "#2ca02c"  # green

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    offset = 0.12
    if has_ci:
        ent_err = err_bars("entropy_ci95", ent_means)
        comb_err = err_bars("combined_ci95", comb_means)
        ax.errorbar(x - offset, ent_means, yerr=ent_err, fmt="o-", color=COLOR_ENT,
                    capsize=5, capthick=1.5, linewidth=2, markersize=7, label="Entropy-only")
        ax.errorbar(x + offset, comb_means, yerr=comb_err, fmt="s-", color=COLOR_COMB,
                    capsize=5, capthick=1.5, linewidth=2, markersize=7, label="Combined (entropy + geometry)")
    else:
        ax.plot(x, ent_means, "o-", color=COLOR_ENT, linewidth=2, markersize=7, label="Entropy-only")
        ax.plot(x, comb_means, "s-", color=COLOR_COMB, linewidth=2, markersize=7, label="Combined (entropy + geometry)")

    # Annotate delta above combined line
    for i, (d, n) in enumerate(zip(deltas, ns)):
        ax.annotate(
            f"Δ{d:+.2f}\n(n={n})",
            xy=(x[i] + offset, comb_means[i]),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center", fontsize=8.5, color=COLOR_COMB, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Level {l.split('_')[1]}" for l in levels], fontsize=11)
    ax.set_xlabel("MATH-500 Difficulty Level  (1 = easiest, 5 = hardest)", fontsize=11)
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ci_note = " with 95% bootstrap CI" if has_ci else ""
    ax.set_title(
        f"Hidden-state geometry vs. entropy{ci_note}\nby MATH-500 difficulty level",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(0.4, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Subject stratification
# ---------------------------------------------------------------------------

def analyze_by_subject(
    traces, idx_to_subject, pca_dim, layer: int,
    n_bootstrap: int = 0, cv_random_state: int = CV_RANDOM_STATE,
):
    """Stratify traces by MATH-500 subject and evaluate each stratum."""
    for t in traces:
        t["subject"] = idx_to_subject.get(t["idx"], None)

    labeled = [t for t in traces if t["subject"] is not None]
    if not labeled:
        print("  No traces matched subjects.")
        return {}

    subjects = sorted({t["subject"] for t in labeled})
    buckets = {s: [t for t in labeled if t["subject"] == s] for s in subjects}

    results = {}
    for subject, subset in buckets.items():
        n_correct = sum(1 for t in subset if t["is_correct"])
        n_incorrect = len(subset) - n_correct
        if n_correct < 5 or n_incorrect < 5:
            print(f"    {subject}: skipped (correct={n_correct}, incorrect={n_incorrect})")
            results[subject] = {"skipped": True, "n_correct": n_correct, "n_incorrect": n_incorrect}
            continue

        y_sub = np.array([1 if t["is_correct"] else 0 for t in subset])
        X_ent_sub = np.array([entropy_features(t["entropies"]) for t in subset])

        n_splits = max(3, min(5, min(n_correct, n_incorrect)))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_random_state)
        folds = list(skf.split(X_ent_sub, y_sub))

        print(f"\n    {subject} (n={len(subset)}, correct={n_correct}, incorrect={n_incorrect}, layer={layer}):")
        ent_res = evaluate_features(X_ent_sub, y_sub, folds, f"{subject} entropy")
        fw = evaluate_foldwise_mahalanobis(
            subset, layer, pca_dim, y_sub, folds, label_prefix=f"{subject[:20]} "
        )
        mah_res = fw["mahalanobis_only"]
        comb_res = fw["combined"]

        delta = comb_res["roc_auc_mean"] - ent_res["roc_auc_mean"]
        print(f"      Delta: {delta:+.4f}", end="")

        bucket_result = {
            "n_total": len(subset),
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "entropy_only": ent_res,
            "mahalanobis_only": mah_res,
            "combined": comb_res,
            "delta": float(delta),
        }

        if n_bootstrap > 0:
            print(f"  [bootstrapping n={n_bootstrap}...]", end="", flush=True)
            ci = bootstrap_auc_ci(subset, layer, pca_dim, n_bootstrap)
            bucket_result["bootstrap_ci"] = ci
            print(f"\r      Delta: {delta:+.4f}  "
                  f"combined 95% CI [{ci['combined_ci95'][0]:.3f}, {ci['combined_ci95'][1]:.3f}]  "
                  f"entropy 95% CI [{ci['entropy_ci95'][0]:.3f}, {ci['entropy_ci95'][1]:.3f}]")
        else:
            print()

        results[subject] = bucket_result

    return results


def plot_subject_stratification(subject_results, output_path):
    """Bar chart: AUC by MATH-500 subject category."""
    subjects = [
        k for k in sorted(subject_results)
        if not subject_results[k].get("skipped")
    ]
    if not subjects:
        print("  No subjects to plot.")
        return

    x = np.arange(len(subjects))
    ent_means = np.array([subject_results[s]["entropy_only"]["roc_auc_mean"] for s in subjects])
    mah_means = np.array([subject_results[s]["mahalanobis_only"]["roc_auc_mean"] for s in subjects])
    comb_means = np.array([subject_results[s]["combined"]["roc_auc_mean"] for s in subjects])
    deltas = np.array([subject_results[s]["delta"] for s in subjects])

    COLOR_ENT = "#4878CF"
    COLOR_MAH = "#D4A03C"
    COLOR_COMB = "#2ca02c"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    w = 0.25
    ax.bar(x - w, ent_means, w, color=COLOR_ENT, alpha=0.8, label="Entropy-only")
    ax.bar(x, mah_means, w, color=COLOR_MAH, alpha=0.8, label="Mahalanobis-only")
    ax.bar(x + w, comb_means, w, color=COLOR_COMB, alpha=0.8, label="Combined")

    for i, d in enumerate(deltas):
        ax.annotate(f"Δ{d:+.2f}", xy=(x[i] + w, comb_means[i]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7.5, color=COLOR_COMB, fontweight="bold")

    # Shorten long subject names for display
    short = [s.replace("Intermediate ", "Int. ").replace("Counting & Probability", "Count & Prob")
             for s in subjects]
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_title("AUC by MATH-500 subject category", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0.3, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_geometry(traces: list[dict], pca, layer: int, output_path: str):
    all_entropies = np.concatenate([t["entropies"] for t in traces])
    high_ent_threshold = np.percentile(all_entropies, 75)

    points_correct, points_incorrect = [], []
    for trace in traces:
        high_mask = trace["entropies"] > high_ent_threshold
        if not high_mask.any():
            continue
        projected = pca.transform(trace["hiddens"][layer][high_mask])
        (points_correct if trace["is_correct"] else points_incorrect).append(projected.mean(axis=0))

    if not points_correct or not points_incorrect:
        print(f"  Skipping 2D plot for layer {layer} (insufficient data).")
        return

    pc, pi = np.array(points_correct), np.array(points_incorrect)
    pts_2d = PCA(n_components=2, random_state=42).fit_transform(np.concatenate([pc, pi]))
    n_c = len(pc)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pts_2d[:n_c, 0], pts_2d[:n_c, 1], alpha=0.4, s=20, c="green", label=f"Correct ({n_c})")
    ax.scatter(pts_2d[n_c:, 0], pts_2d[n_c:, 1], alpha=0.4, s=20, c="red", label=f"Incorrect ({len(pi)})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"Layer {layer} hidden states at high-entropy tokens\n(per-trace mean, projected to 2D)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


def plot_mahal_distributions(traces: list[dict], layer: int, output_path: str):
    correct_means = [t["mahal_dists"].mean() for t in traces if t["is_correct"]]
    incorrect_means = [t["mahal_dists"].mean() for t in traces if not t["is_correct"]]
    lo = min(min(correct_means), min(incorrect_means))
    hi = max(max(correct_means), max(incorrect_means))

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(lo, hi, 40)
    ax.hist(correct_means, bins=bins, alpha=0.6, color="green",
            label=f"Correct (n={len(correct_means)})", density=True)
    ax.hist(incorrect_means, bins=bins, alpha=0.6, color="red",
            label=f"Incorrect (n={len(incorrect_means)})", density=True)
    ax.set_xlabel("Mean Mahalanobis distance")
    ax.set_ylabel("Density")
    ax.set_title(f"Layer {layer} — per-trace mean Mahalanobis distance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


def plot_layer_comparison(layer_results: dict, entropy_baseline: dict, output_path: str):
    layers = sorted(layer_results.keys())
    ent_auc = entropy_baseline["roc_auc_mean"]
    ent_std = entropy_baseline["roc_auc_std"]

    mahal_means = [layer_results[l]["mahalanobis_only"]["roc_auc_mean"] for l in layers]
    mahal_stds = [layer_results[l]["mahalanobis_only"]["roc_auc_std"] for l in layers]
    comb_means = [layer_results[l]["combined"]["roc_auc_mean"] for l in layers]
    comb_stds = [layer_results[l]["combined"]["roc_auc_std"] for l in layers]

    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(ent_auc, color="blue", linestyle="--", label=f"Entropy-only ({ent_auc:.3f} ± {ent_std:.3f})")
    ax.fill_between([-0.5, len(layers) - 0.5],
                    ent_auc - ent_std, ent_auc + ent_std, alpha=0.1, color="blue")
    ax.errorbar(x - 0.15, mahal_means, yerr=mahal_stds, fmt="o-", color="orange",
                capsize=4, label="Mahalanobis-only")
    ax.errorbar(x + 0.15, comb_means, yerr=comb_stds, fmt="s-", color="green",
                capsize=4, label="Combined")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_ylabel("ROC-AUC")
    ax.set_title("AUC by layer (entropy baseline fixed; Mahalanobis ref fit per train fold)")
    ax.legend()
    ax.set_xlim(-0.5, len(layers) - 0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    prefix = args.dataset_label
    out_dir = args.output_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)

    # Determine layers to analyze
    if args.layers:
        layers_to_analyze = [int(x) for x in args.layers.split(",")]
    else:
        layers_to_analyze = detect_layers(args.data_dir)
    print(f"Layers to analyze: {layers_to_analyze}")

    print("Loading all traces...")
    traces = load_all_traces(args.data_dir, layers_to_analyze)
    correct = [t for t in traces if t["is_correct"]]
    incorrect = [t for t in traces if not t["is_correct"]]
    print(f"Loaded {len(traces)} traces: {len(correct)} correct, {len(incorrect)} incorrect")
    if len(incorrect) < 10:
        print("WARNING: Fewer than 10 incorrect traces — statistics will be unreliable.")

    # --- Single entropy baseline with fixed folds ---
    print("\n--- Entropy-only baseline (computed once) ---")
    X_ent, y = build_feature_matrix(traces)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.cv_random_state)
    fold_indices = list(skf.split(X_ent, y))  # fixed for all subsequent comparisons
    entropy_baseline = evaluate_features(X_ent, y, fold_indices, "entropy_only")

    # Length control baseline (independent of layer / Mahalanobis)
    X_lent = np.array([
        entropy_features(t["entropies"]) + [np.log1p(len(t["entropies"]))] for t in traces
    ])
    lent_result_global = evaluate_features(X_lent, y, fold_indices, "entropy+length")

    # --- Per-layer analysis: Mahalanobis ref fit on train-fold correct only ---
    layer_results = {}

    for layer in layers_to_analyze:
        print(f"\n{'='*60}")
        print(f"LAYER {layer}")
        print(f"{'='*60}")

        fw = evaluate_foldwise_mahalanobis(
            traces, layer, args.pca_dim, y, fold_indices, label_prefix=""
        )
        mahal_result = fw["mahalanobis_only"]
        comb_result = fw["combined"]
        lcomb_result = fw["combined_with_length"]

        delta = comb_result["roc_auc_mean"] - entropy_baseline["roc_auc_mean"]
        len_delta = lcomb_result["roc_auc_mean"] - lent_result_global["roc_auc_mean"]
        print(f"\n  Delta (combined - entropy_only):          {delta:+.4f}")
        print(f"  Delta (combined+len - entropy+len):       {len_delta:+.4f}  [length-controlled]")

        confident_wrong = analyze_confident_wrong(traces, layer, args.pca_dim, fold_indices)

        layer_results[layer] = {
            "mahalanobis_only": mahal_result,
            "combined": comb_result,
            "delta_vs_entropy": float(delta),
            "entropy_with_length": lent_result_global,
            "combined_with_length": lcomb_result,
            "length_controlled_delta": float(len_delta),
            "confident_wrong": confident_wrong,
        }

        # --- Post-fork analysis ---
        if args.post_fork or args.all_analyses:
            post_fork = analyze_post_fork(traces, layer, args.pca_dim, fold_indices)
            layer_results[layer]["post_fork"] = post_fork

        # --- Narrow reference distribution ---
        if args.narrow_ref or args.all_analyses:
            print(f"\n  Narrow reference distribution (fold-wise)...")
            nw = evaluate_foldwise_narrow_mahalanobis(
                traces, layer, args.pca_dim, y, fold_indices, label_prefix=""
            )
            narrow_mah_result = nw["narrow_mahal_only"]
            narrow_comb_result = nw["narrow_combined"]
            narrow_delta = narrow_comb_result["roc_auc_mean"] - entropy_baseline["roc_auc_mean"]
            print(f"  Narrow delta (narrow_combined - entropy_only): {narrow_delta:+.4f}")

            layer_results[layer]["narrow_mahal_only"] = narrow_mah_result
            layer_results[layer]["narrow_combined"] = narrow_comb_result
            layer_results[layer]["narrow_delta_vs_entropy"] = float(narrow_delta)

        assign_oof_mahalanobis_distances(traces, layer, args.pca_dim, fold_indices)
        plot_mahal_distributions(traces, layer, os.path.join(out_dir, f"{prefix}_mahal_dist_L{layer}.png"))

    # Best layer plots
    best_layer = max(layer_results, key=lambda l: layer_results[l]["combined"]["roc_auc_mean"])
    print(f"\nBest layer by combined AUC: {best_layer}")
    # Exploratory 2D figure: PCA basis from all correct traces (visualization only)
    pca_vis, _, _ = fit_mahalanobis_reference(correct, best_layer, args.pca_dim)
    assign_oof_mahalanobis_distances(traces, best_layer, args.pca_dim, fold_indices)
    plot_geometry(traces, pca_vis, best_layer, os.path.join(out_dir, f"{prefix}_geometry_L{best_layer}.png"))

    plot_layer_comparison(layer_results, entropy_baseline, os.path.join(out_dir, f"{prefix}_layer_comparison.png"))

    # --- Difficulty stratification ---
    difficulty_results = None
    if prefix == "math500" and (args.difficulty or args.all_analyses):
        print(f"\n{'='*60}")
        print("DIFFICULTY STRATIFICATION (best layer)")
        print(f"{'='*60}")
        idx_to_level = load_math500_levels()
        difficulty_results = analyze_by_difficulty(
            traces, idx_to_level, args.pca_dim,
            layer=best_layer,
            n_bootstrap=args.n_bootstrap,
            cv_random_state=args.cv_random_state,
        )
        if difficulty_results:
            plot_difficulty_stratification(difficulty_results, os.path.join(out_dir, f"{prefix}_difficulty.png"))
    elif (args.difficulty or args.all_analyses) and prefix != "math500":
        print("\nSkipping difficulty stratification (only available for math500 dataset).")

    # --- Subject stratification ---
    subject_results = None
    if prefix == "math500" and (args.subject or args.all_analyses):
        print(f"\n{'='*60}")
        print("SUBJECT STRATIFICATION (best layer)")
        print(f"{'='*60}")
        idx_to_subject = load_math500_subjects()
        subject_results = analyze_by_subject(
            traces, idx_to_subject, args.pca_dim,
            layer=best_layer,
            n_bootstrap=args.n_bootstrap,
            cv_random_state=args.cv_random_state,
        )
        if subject_results:
            plot_subject_stratification(subject_results, os.path.join(out_dir, f"{prefix}_subject.png"))
    elif (args.subject or args.all_analyses) and prefix != "math500":
        print("\nSkipping subject stratification (only available for math500 dataset).")

    # --- Summary table ---
    print(f"\n\n{'='*70}")
    print(f"SUMMARY ({prefix})")
    print(f"{'='*70}")
    print(f"  Entropy-only baseline: {entropy_baseline['roc_auc_mean']:.4f} ± {entropy_baseline['roc_auc_std']:.4f}  [FIXED ACROSS ALL LAYERS]")
    print()

    # Build header
    has_post_fork = any("post_fork" in layer_results[l] for l in layer_results)
    has_narrow = any("narrow_combined" in layer_results[l] for l in layer_results)

    header = f"  {'Layer':>6}  {'Mahal-only':>12}  {'Combined':>10}  {'Delta':>8}  {'LenCtrlΔ':>10}  {'ConfWrong p':>12}"
    if has_post_fork:
        header += f"  {'PostFork p':>12}"
    if has_narrow:
        header += f"  {'NarrowComb':>12}  {'NarrowΔ':>8}"
    print(header)

    for layer in sorted(layer_results):
        r = layer_results[layer]
        cw = r["confident_wrong"]
        cw_p = cw.get("mannwhitney_pvalue")
        if cw.get("skipped") or cw_p is None or (isinstance(cw_p, float) and np.isnan(cw_p)):
            cw_cell = f"{'N/A':>10}    "
        else:
            sig = "***" if cw_p < 0.01 else ("*" if cw_p < 0.05 else "")
            cw_cell = f"{cw_p:>10.2e} {sig}"
        lc_delta = r.get("length_controlled_delta", float("nan"))
        line = (
            f"  {layer:>6}  "
            f"{r['mahalanobis_only']['roc_auc_mean']:>10.4f}  "
            f"{r['combined']['roc_auc_mean']:>10.4f}  "
            f"{r['delta_vs_entropy']:>+8.4f}  "
            f"{lc_delta:>+10.4f}  "
            f"{cw_cell}"
        )
        if has_post_fork:
            pf = r.get("post_fork", {})
            if pf.get("skipped"):
                line += f"  {'N/A':>12}"
            else:
                pf_p = pf.get("mannwhitney_pvalue", float("nan"))
                pf_sig = "***" if pf_p < 0.01 else ("*" if pf_p < 0.05 else "")
                line += f"  {pf_p:>10.2e} {pf_sig}"
        if has_narrow:
            nc = r.get("narrow_combined", {})
            nd = r.get("narrow_delta_vs_entropy", float("nan"))
            if nc:
                line += f"  {nc['roc_auc_mean']:>10.4f}  {nd:>+8.4f}"
            else:
                line += f"  {'N/A':>12}  {'N/A':>8}"
        print(line)

    # Difficulty summary
    if difficulty_results:
        print(f"\n  Difficulty stratification (best layer = {best_layer}):")
        print(f"    {'Bucket':>12}  {'Ent AUC':>10}  {'Comb AUC':>10}  {'Delta':>8}  {'N':>5}")
        for bucket in sorted(difficulty_results):
            dr = difficulty_results[bucket]
            if dr.get("skipped"):
                print(f"    {bucket:>12}  {'skipped':>10}  {'':>10}  {'':>8}  {dr.get('n_correct', 0) + dr.get('n_incorrect', 0):>5}")
            else:
                print(
                    f"    {bucket:>12}  "
                    f"{dr['entropy_only']['roc_auc_mean']:>10.4f}  "
                    f"{dr['combined']['roc_auc_mean']:>10.4f}  "
                    f"{dr['delta']:>+8.4f}  "
                    f"{dr['n_total']:>5}"
                )

    # Subject summary
    if subject_results:
        print(f"\n  Subject stratification (best layer = {best_layer}):")
        print(f"    {'Subject':>25}  {'Ent AUC':>10}  {'Mahal AUC':>10}  {'Comb AUC':>10}  {'Delta':>8}  {'N':>5}")
        for subject in sorted(subject_results):
            sr = subject_results[subject]
            if sr.get("skipped"):
                print(f"    {subject:>25}  {'skipped':>10}  {'':>10}  {'':>10}  {'':>8}  {sr.get('n_correct', 0) + sr.get('n_incorrect', 0):>5}")
            else:
                print(
                    f"    {subject:>25}  "
                    f"{sr['entropy_only']['roc_auc_mean']:>10.4f}  "
                    f"{sr['mahalanobis_only']['roc_auc_mean']:>10.4f}  "
                    f"{sr['combined']['roc_auc_mean']:>10.4f}  "
                    f"{sr['delta']:>+8.4f}  "
                    f"{sr['n_total']:>5}"
                )

    # --- Cross-model Mahalanobis transfer ---
    cross_model_results = None
    if args.cross_model_ref:
        print(f"\n{'='*60}")
        print(f"CROSS-MODEL MAHALANOBIS TRANSFER")
        print(f"{'='*60}")
        print(f"  Reference data: {args.cross_model_ref}")

        ref_traces = load_all_traces(args.cross_model_ref, layers_to_analyze)
        ref_correct = [t for t in ref_traces if t["is_correct"]]
        print(f"  Loaded {len(ref_traces)} reference traces ({len(ref_correct)} correct)")

        cross_model_results = {}

        # Build ref model labels
        y_ref = np.array([1 if t["is_correct"] else 0 for t in ref_traces])

        for layer in layers_to_analyze:
            print(f"\n  Layer {layer}:")
            # Fit PCA + Gaussian on other model's correct traces
            ref_pca, ref_mu, ref_cov_inv = fit_mahalanobis_reference(
                ref_correct, layer, args.pca_dim
            )

            # Compute Mahalanobis distances for this model's traces using the other model's reference
            X_xmah_rows, X_xcomb_rows = [], []
            for trace in traces:
                xm = compute_mahal_distances(trace["hiddens"][layer], ref_pca, ref_mu, ref_cov_inv)
                xmf = mahal_features(trace["entropies"], xm)
                X_xmah_rows.append(xmf)
                X_xcomb_rows.append(entropy_features(trace["entropies"]) + xmf)
            X_xmah = np.array(X_xmah_rows)
            X_xcomb = np.array(X_xcomb_rows)

            xmah_result = evaluate_features(X_xmah, y, fold_indices, "cross_mahal_only")
            xcomb_result = evaluate_features(X_xcomb, y, fold_indices, "cross_combined")
            native = layer_results[layer]["mahalanobis_only"]["roc_auc_mean"]
            cross = xmah_result["roc_auc_mean"]
            print(f"    Native Mahal AUC: {native:.4f}  →  Cross-model Mahal AUC: {cross:.4f}  "
                  f"(transfer: {cross/native:.0%})" if native > 0 else "")

            # --- Classifier transfer: train on ref model, evaluate on this model ---
            # Ref model's features use ref model's own native reference
            # (same ref_pca/ref_mu/ref_cov_inv since that IS the ref model's geometry)
            X_ref_mah_rows, X_ref_comb_rows = [], []
            for rt in ref_traces:
                rm = compute_mahal_distances(rt["hiddens"][layer], ref_pca, ref_mu, ref_cov_inv)
                rmf = mahal_features(rt["entropies"], rm)
                X_ref_mah_rows.append(rmf)
                X_ref_comb_rows.append(entropy_features(rt["entropies"]) + rmf)
            X_ref_mah = np.array(X_ref_mah_rows)
            X_ref_comb = np.array(X_ref_comb_rows)

            print("    Classifier transfer (train on ref model → eval on this model):")
            clf_mah = evaluate_transfer(X_ref_mah, y_ref, X_xmah, y, "  clf_transfer_mahal_only")
            clf_comb = evaluate_transfer(X_ref_comb, y_ref, X_xcomb, y, "  clf_transfer_combined")

            cross_model_results[str(layer)] = {
                "cross_mahal_only": xmah_result,
                "cross_combined": xcomb_result,
                "clf_transfer_mahal_only": clf_mah,
                "clf_transfer_combined": clf_comb,
            }

    # Save results
    output = {
        "dataset": prefix,
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "entropy_baseline": entropy_baseline,
        "layers": {str(l): layer_results[l] for l in layer_results},
    }
    if difficulty_results:
        output["difficulty_stratification"] = difficulty_results
    if subject_results:
        output["subject_stratification"] = subject_results
    if cross_model_results:
        output["cross_model_transfer"] = cross_model_results
    out_path = os.path.join(out_dir, f"{prefix}_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
