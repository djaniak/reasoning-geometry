"""Published-style last-token probe, reproduced and then decomposed.

The workshop review's fourth blocker: the within-prompt decomposition has so far
been applied to this project's own scores. It has not been applied to the object
whose claim shape it corrects -- a supervised probe on the *last token* hidden
state, of the kind Yuan et al. report a high pooled trace AUROC for.

The repository already has a supervised hidden-state probe
(``probe_hidden_*`` in ``applications/prompt_decomposition``), but that probe
pools a *region mean* over PCA-projected states and selects its layer outside
the training data. Neither is the published protocol. Reproducing the claim
before contesting it means matching it: the final token's raw hidden state, a
plain regularized logistic readout, prompt-disjoint folds, and layer selection
performed *inside* each training split.

The decomposition then asks the only question that matters for the paper: how
much of that pooled number survives conditioning on prompt identity. Three
readouts, and the population each one is defined on:

  pooled          every held-out trace, prompt identity ignored. This is the
                  published number. A prompt with one outcome still contributes
                  its traces.
  micro (pairs)   every (correct, incorrect) pair drawn *within* one prompt,
                  weighted by pair count. Single-outcome prompts contribute no
                  pairs and so drop out entirely.
  macro           per-prompt AUROC averaged over prompts, each prompt counting
                  once. Single-outcome prompts are undefined and excluded.

Reporting all three is the point: pooled and micro/macro are computed on
different populations, and a gap between them is the between-prompt component,
not a defect.

Not a DVC stage. It reads the collected trace batches once to cache last-token
states, then fits on CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from applications.prompt_decomposition import make_prompt_folds

# The published protocol is a plain linear readout on the raw final-token state.
# The model width exceeds the trace count, so the fit separates the training set
# perfectly at any loose penalty and the strength of the penalty is not a detail:
# on Qwen layer 21 the held-out pooled AUROC runs 0.828 at C=1 and 0.895 at
# C=1e-3. Reproducing a published claim means reproducing it at its strength, so
# C is selected inside the training data along with the layer -- never fixed, and
# never chosen on the split it is scored on.
PROBE_C_GRID = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
# liblinear's dual formulation solves in the sample count rather than the feature
# count. It is the same L2 logistic objective as lbfgs, an order of magnitude
# faster in this regime, and what makes in-fold selection affordable at all.
PROBE_SOLVER = "liblinear"
PROBE_MAX_ITER = 5000

N_SPLITS = 5
INNER_SPLITS = 5
N_BOOTSTRAP = 1000
SEED = 42

POPULATIONS = ("parseable", "all_traces")

# Output-side readouts computed from the same cached metadata, at no extra cost.
# They are here because a probe that merely rediscovers "long traces are wrong"
# must be visible as such, and because the decomposition is only interesting if
# the collapse is specific to the probe rather than shared by everything.
REFERENCE_SCORES = ("length", "mean_logprob", "mean_entropy")

# Optional join against the frozen pipeline's out-of-fold scores, so this
# project's own geometry and its own supervised hidden probe pass through the
# same three readouts on the same traces. Without it the decomposition is a
# statement about someone else's method shape; with it, it is a table.
FROZEN_SCORE_COLUMNS = ("rmd_tail_q20_score", "probe_hidden_tail_q20_score")


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def is_unparsed(predicted: str | None) -> bool:
    """Mirror of ``prompt_decomposition.is_unparsed`` on a bare answer string."""
    return predicted is None or str(predicted).strip() == ""


def extract_batch(path: str, layers: tuple[int, ...]) -> list[dict]:
    """Pull the final generated token's hidden state out of one trace batch.

    The stored arrays are DEFLATE members, so the tail cannot be seeked to; the
    whole array is inflated and all but its last row discarded. That is the
    entire cost of this script, and it is paid once.
    """
    records = []
    with np.load(path, allow_pickle=True) as data:
        available = set(data.files)
        for meta in data["metadata"]:
            trace_id = int(meta["trace_id"] if "trace_id" in meta else meta["idx"])
            idx = int(meta["idx"])

            def _key(prefix: str) -> str | None:
                for candidate in (f"{prefix}_{trace_id}", f"{prefix}_{idx}"):
                    if candidate in available:
                        return candidate
                return None

            states = {}
            for layer in layers:
                key = _key(f"hidden_L{layer}")
                if key is None:
                    raise KeyError(f"{path}: trace {trace_id} has no layer {layer}")
                hidden = data[key]
                if hidden.ndim != 2 or hidden.shape[0] == 0:
                    raise ValueError(f"{path}: trace {trace_id} layer {layer} is empty")
                states[layer] = hidden[-1].astype(np.float16)

            entropy_key = _key("entropies")
            logprob_key = _key("token_logprobs")
            entropies = data[entropy_key] if entropy_key else None
            logprobs = data[logprob_key] if logprob_key else None

            records.append(
                {
                    "trace_id": trace_id,
                    "prompt_id": idx,
                    "sample_id": int(meta["sample_id"] if "sample_id" in meta else 0),
                    "is_correct": bool(meta["is_correct"]),
                    "predicted_answer": (
                        str(meta["predicted"]) if "predicted" in meta and meta["predicted"] is not None else ""
                    ),
                    "n_tokens": int(
                        meta["n_tokens"]
                        if "n_tokens" in meta
                        else len(entropies) if entropies is not None else 0
                    ),
                    "mean_entropy": (
                        float(np.mean(entropies)) if entropies is not None and len(entropies) else float("nan")
                    ),
                    "mean_logprob": (
                        float(np.mean(logprobs))
                        if logprobs is not None and len(logprobs)
                        else float(meta["mean_logprob"]) if "mean_logprob" in meta else float("nan")
                    ),
                    "states": states,
                }
            )
    return records


def extract_last_token_states(
    data_dir: str, layers: tuple[int, ...], *, max_workers: int = 8
) -> dict:
    paths = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(".npz")
    )
    if not paths:
        raise ValueError(f"no trace batches under {data_dir}")

    records: list[dict] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_batch, path, layers): path for path in paths}
        for done, future in enumerate(as_completed(futures), start=1):
            records.extend(future.result())
            if done % 10 == 0 or done == len(paths):
                _status(
                    f"  extracted {done}/{len(paths)} batches "
                    f"({time.perf_counter() - started:.0f}s)"
                )

    records.sort(key=lambda record: (record["prompt_id"], record["sample_id"], record["trace_id"]))
    cache = {
        "trace_id": np.asarray([r["trace_id"] for r in records], dtype=np.int64),
        "prompt_id": np.asarray([r["prompt_id"] for r in records], dtype=np.int64),
        "sample_id": np.asarray([r["sample_id"] for r in records], dtype=np.int64),
        "is_correct": np.asarray([r["is_correct"] for r in records], dtype=np.int8),
        "unparsed": np.asarray(
            [is_unparsed(r["predicted_answer"]) for r in records], dtype=np.int8
        ),
        "n_tokens": np.asarray([r["n_tokens"] for r in records], dtype=np.int64),
        "mean_entropy": np.asarray([r["mean_entropy"] for r in records], dtype=np.float64),
        "mean_logprob": np.asarray([r["mean_logprob"] for r in records], dtype=np.float64),
        "layers": np.asarray(layers, dtype=np.int64),
    }
    for layer in layers:
        cache[f"states_L{layer}"] = np.stack([r["states"][layer] for r in records])
    return cache


def load_or_build_cache(
    data_dir: str, layers: tuple[int, ...], cache_path: Path, *, max_workers: int
) -> dict:
    if cache_path.exists():
        with np.load(cache_path) as stored:
            cached_layers = tuple(int(value) for value in stored["layers"])
            if cached_layers == tuple(layers):
                _status(f"  reusing {cache_path}")
                return {key: stored[key] for key in stored.files}
        _status(f"  {cache_path} holds layers {cached_layers}; re-extracting")
    cache = extract_last_token_states(data_dir, layers, max_workers=max_workers)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **cache)
    return cache


def population_mask(cache: dict, population: str) -> np.ndarray:
    """Which traces the analysis is defined on.

    ``parseable`` matches the frozen hidden-state probe's training rule.
    Unparsed traces are auto-labeled incorrect upstream, so a probe that keeps
    them can win by detecting truncation rather than reasoning failure --
    ``all_traces`` is reported precisely so that inflation is visible.
    """
    if population == "all_traces":
        return np.ones(len(cache["trace_id"]), dtype=bool)
    if population == "parseable":
        return cache["unparsed"] == 0
    raise ValueError(f"unknown population: {population}")


def fit_probe(
    features: np.ndarray, labels: np.ndarray, penalty: float
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(
        C=penalty,
        solver=PROBE_SOLVER,
        dual=True,
        max_iter=PROBE_MAX_ITER,
        random_state=SEED,
    ).fit(scaler.transform(features), labels)
    return scaler, classifier


def score_probe(fit: tuple[StandardScaler, LogisticRegression], features: np.ndarray) -> np.ndarray:
    scaler, classifier = fit
    # classes_ is [0, 1], so the decision function points at is_correct=1 and
    # every score in this module reads "higher = more likely correct".
    return classifier.decision_function(scaler.transform(features))


def _pooled_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def select_layer_and_penalty(
    states_by_layer: dict[int, np.ndarray],
    labels: np.ndarray,
    prompt_ids: np.ndarray,
    train_prompts: list[int],
    *,
    inner_splits: int,
    seed: int,
    penalties: tuple[float, ...] = PROBE_C_GRID,
) -> tuple[int, float, dict[tuple[int, float], float]]:
    """Choose the layer and penalty using only the outer training prompts.

    Selecting either on the test split is the quiet leak that makes published
    pooled numbers hard to reproduce, so the inner folds are prompt-disjoint
    too and every candidate is scored the same way the outer fold will be.
    """
    inner_folds = make_prompt_folds(
        sorted(train_prompts), n_splits=inner_splits, seed=seed
    )
    grid: dict[tuple[int, float], float] = {}
    for layer, states in states_by_layer.items():
        for penalty in penalties:
            fold_aurocs = []
            for inner_train, inner_test in inner_folds:
                train_mask = np.isin(prompt_ids, inner_train)
                test_mask = np.isin(prompt_ids, inner_test)
                if len(np.unique(labels[train_mask])) < 2:
                    continue
                fit = fit_probe(
                    states[train_mask].astype(np.float32), labels[train_mask], penalty
                )
                auroc = _pooled_auroc(
                    labels[test_mask],
                    score_probe(fit, states[test_mask].astype(np.float32)),
                )
                if auroc is not None:
                    fold_aurocs.append(auroc)
            grid[(int(layer), float(penalty))] = (
                float(np.mean(fold_aurocs)) if fold_aurocs else float("nan")
            )
    layer, penalty = max(grid, key=lambda key: grid[key])
    return int(layer), float(penalty), grid


def crossfit_probe(
    states_by_layer: dict[int, np.ndarray],
    labels: np.ndarray,
    prompt_ids: np.ndarray,
    *,
    n_splits: int,
    inner_splits: int,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """Out-of-fold probe scores with per-fold layer selection."""
    folds = make_prompt_folds(sorted(set(int(p) for p in prompt_ids)), n_splits=n_splits, seed=seed)
    scores = np.full(len(labels), np.nan)
    fold_records = []
    for fold_index, (train_prompts, test_prompts) in enumerate(folds):
        train_mask = np.isin(prompt_ids, train_prompts)
        test_mask = np.isin(prompt_ids, test_prompts)
        layer, penalty, grid = select_layer_and_penalty(
            {layer: states[train_mask] for layer, states in states_by_layer.items()},
            labels[train_mask],
            prompt_ids[train_mask],
            train_prompts,
            inner_splits=inner_splits,
            seed=seed,
        )
        states = states_by_layer[layer]
        fit = fit_probe(states[train_mask].astype(np.float32), labels[train_mask], penalty)
        scores[test_mask] = score_probe(fit, states[test_mask].astype(np.float32))
        best_by_layer = {
            str(candidate_layer): max(
                value for (grid_layer, _), value in grid.items()
                if grid_layer == candidate_layer
            )
            for candidate_layer in states_by_layer
        }
        fold_records.append(
            {
                "fold": fold_index,
                "selected_layer": layer,
                "selected_C": penalty,
                "inner_pooled_auroc_by_layer": best_by_layer,
                "inner_pooled_auroc_grid": {
                    f"L{grid_layer}_C{grid_penalty:g}": value
                    for (grid_layer, grid_penalty), value in grid.items()
                },
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "train_base_rate": float(labels[train_mask].mean()),
            }
        )
    if np.isnan(scores).any():
        raise ValueError("some traces were never scored out of fold")
    return scores, fold_records


def within_prompt_metrics(
    prompt_ids: np.ndarray, labels: np.ndarray, scores: np.ndarray
) -> dict:
    """Micro (pair-weighted) and macro (per-prompt) within-prompt AUROC.

    A tie counts half, matching the AUROC convention. Prompts without both
    outcomes contribute no pairs and no macro term; their trace count is
    reported so the populations stay legible.
    """
    concordant = 0.0
    total_pairs = 0
    per_prompt = []
    n_mixed = 0
    for prompt in np.unique(prompt_ids):
        mask = prompt_ids == prompt
        prompt_labels = labels[mask]
        prompt_scores = scores[mask]
        correct = prompt_scores[prompt_labels == 1]
        incorrect = prompt_scores[prompt_labels == 0]
        if len(correct) == 0 or len(incorrect) == 0:
            continue
        n_mixed += 1
        comparison = correct[:, None] - incorrect[None, :]
        wins = float((comparison > 0).sum()) + 0.5 * float((comparison == 0).sum())
        pairs = comparison.size
        concordant += wins
        total_pairs += pairs
        per_prompt.append(wins / pairs)
    return {
        "micro_pair_auroc": concordant / total_pairs if total_pairs else None,
        "macro_prompt_auroc": float(np.mean(per_prompt)) if per_prompt else None,
        "n_mixed_prompts": n_mixed,
        "n_within_prompt_pairs": total_pairs,
    }


def prompt_centered_auroc(
    prompt_ids: np.ndarray, labels: np.ndarray, scores: np.ndarray
) -> float | None:
    """Pooled AUROC after removing each prompt's mean score.

    Restricted to mixed prompts, matching `prompt_decomposition.prompt_centered_auc`
    exactly -- a single-outcome prompt centres to a constant-label block that can
    only dilute the result, and the frozen report's column of this name excludes
    them. The continuity check against that report is only meaningful if the two
    are the same quantity.
    """
    kept_labels = []
    centered_scores = []
    for prompt in np.unique(prompt_ids):
        mask = prompt_ids == prompt
        group_labels = labels[mask]
        if len(np.unique(group_labels)) < 2:
            continue
        group_scores = np.asarray(scores, dtype=float)[mask]
        kept_labels.extend(group_labels.tolist())
        centered_scores.extend((group_scores - group_scores.mean()).tolist())
    if not kept_labels:
        return None
    return _pooled_auroc(np.asarray(kept_labels), np.asarray(centered_scores))


def all_metrics(prompt_ids: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict:
    metrics = {
        "pooled_auroc": _pooled_auroc(labels, scores),
        "prompt_centered_auroc": prompt_centered_auroc(prompt_ids, labels, scores),
    }
    metrics.update(within_prompt_metrics(prompt_ids, labels, scores))
    pooled = metrics["pooled_auroc"]
    macro = metrics["macro_prompt_auroc"]
    micro = metrics["micro_pair_auroc"]
    metrics["pooled_minus_macro"] = (
        pooled - macro if pooled is not None and macro is not None else None
    )
    metrics["pooled_minus_micro"] = (
        pooled - micro if pooled is not None and micro is not None else None
    )
    return metrics


def load_frozen_scores(
    csv_path: str, layer: int, trace_ids: np.ndarray,
    columns: tuple[str, ...] = FROZEN_SCORE_COLUMNS,
) -> dict[str, np.ndarray]:
    """Read frozen out-of-fold scores at one layer, aligned to ``trace_ids``.

    The OOF table has one row per (trace, layer). Forgetting to pick a layer
    silently averages the geometry over the sweep while the output-side columns
    reproduce exactly, so the layer is required rather than inferred.
    """
    import csv as _csv

    by_trace: dict[int, dict[str, float]] = {}
    with open(csv_path, newline="") as handle:
        for row in _csv.DictReader(handle):
            if int(row["layer"]) != int(layer):
                continue
            trace_id = int(row["trace_id"])
            if trace_id in by_trace:
                raise ValueError(
                    f"{csv_path}: trace {trace_id} appears twice at layer {layer}"
                )
            by_trace[trace_id] = {
                column: float(row[column]) for column in columns if column in row
            }

    missing = [int(trace_id) for trace_id in trace_ids if int(trace_id) not in by_trace]
    if missing:
        raise ValueError(
            f"{csv_path}: layer {layer} is missing {len(missing)} traces "
            f"(first: {missing[:3]})"
        )
    available = [column for column in columns if column in next(iter(by_trace.values()))]
    return {
        column.removesuffix("_score"): np.asarray(
            [by_trace[int(trace_id)][column] for trace_id in trace_ids], dtype=float
        )
        for column in available
    }


def resample_prompts(
    prompt_ids: np.ndarray, sampled_prompts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Expand a prompt-level draw into trace indices and fresh prompt keys.

    The fresh keys matter. A prompt drawn twice is two prompts of eight traces,
    not one prompt of sixteen: keeping the original id would merge the copies
    and manufacture within-prompt pairs that cross a boundary the design says
    exists, inflating the pair count the micro readout divides by.
    """
    index_by_prompt = {
        int(prompt): np.flatnonzero(prompt_ids == prompt)
        for prompt in np.unique(prompt_ids)
    }
    blocks = [index_by_prompt[int(prompt)] for prompt in sampled_prompts]
    indices = np.concatenate(blocks) if blocks else np.array([], dtype=int)
    relabelled = np.concatenate(
        [np.full(len(block), position) for position, block in enumerate(blocks)]
    ) if blocks else np.array([], dtype=int)
    return indices, relabelled


def bootstrap_metrics(
    prompt_ids: np.ndarray,
    labels: np.ndarray,
    score_sets: dict[str, np.ndarray],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Prompt-level resampling of fixed out-of-fold scores.

    This holds the fitted pipeline -- folds, layer choice, coefficients --
    constant, exactly as every other interval in this project does. It is the
    sampling uncertainty of the prompt set, not of the fit; the outer-refit
    blocker is what addresses the latter.
    """
    unique_prompts = np.unique(prompt_ids)
    rng = np.random.default_rng(seed)
    draws: dict[str, dict[str, list[float]]] = {
        name: {key: [] for key in
               ("pooled_auroc", "prompt_centered_auroc", "micro_pair_auroc",
                "macro_prompt_auroc", "pooled_minus_macro", "pooled_minus_micro")}
        for name in score_sets
    }
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_prompts, size=len(unique_prompts), replace=True)
        indices, relabelled = resample_prompts(prompt_ids, sampled)
        sample_labels = labels[indices]
        for name, scores in score_sets.items():
            metrics = all_metrics(relabelled, sample_labels, scores[indices])
            for key, bucket in draws[name].items():
                value = metrics.get(key)
                if value is not None:
                    bucket.append(value)
    return {
        name: {key: _interval(values) for key, values in buckets.items()}
        for name, buckets in draws.items()
    }


def _interval(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_draws": 0}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.percentile(array, 2.5)),
        "ci_high": float(np.percentile(array, 97.5)),
        "n_draws": int(array.size),
    }


def reference_score_sets(cache: dict, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Output-side readouts, oriented so higher means more likely correct."""
    return {
        # matches the frozen SCORE_DESCRIPTIONS["length"] = "-log1p(token count)".
        # The transform is monotone, so pooled/micro/macro are unchanged by it, but
        # the prompt-centered readout is scale-sensitive and only reproduces under log1p.
        "length": -np.log1p(np.asarray(cache["n_tokens"], dtype=float))[mask],
        "mean_logprob": np.asarray(cache["mean_logprob"], dtype=float)[mask],
        "mean_entropy": -np.asarray(cache["mean_entropy"], dtype=float)[mask],
    }


def analyze_population(
    cache: dict, population: str, *, n_splits: int, inner_splits: int,
    n_bootstrap: int, seed: int, frozen: dict[str, np.ndarray] | None = None,
) -> dict:
    mask = population_mask(cache, population)
    prompt_ids = np.asarray(cache["prompt_id"])[mask]
    labels = np.asarray(cache["is_correct"], dtype=int)[mask]
    layers = tuple(int(value) for value in cache["layers"])
    states_by_layer = {layer: np.asarray(cache[f"states_L{layer}"])[mask] for layer in layers}

    _status(f"  [{population}] {mask.sum()} traces, {len(np.unique(prompt_ids))} prompts")
    probe_scores, fold_records = crossfit_probe(
        states_by_layer, labels, prompt_ids,
        n_splits=n_splits, inner_splits=inner_splits, seed=seed,
    )

    score_sets = {"last_token_probe": probe_scores}
    score_sets.update(reference_score_sets(cache, mask))
    for name, values in (frozen or {}).items():
        score_sets[name] = np.asarray(values, dtype=float)[mask]

    point = {name: all_metrics(prompt_ids, labels, scores) for name, scores in score_sets.items()}
    _status(f"  [{population}] bootstrapping {n_bootstrap} prompt resamples")
    intervals = bootstrap_metrics(
        prompt_ids, labels, score_sets, n_bootstrap=n_bootstrap, seed=seed
    )

    return {
        "n_traces": int(mask.sum()),
        "n_prompts": int(len(np.unique(prompt_ids))),
        "n_correct": int(labels.sum()),
        "base_rate": float(labels.mean()),
        "n_single_outcome_prompts": int(
            len(np.unique(prompt_ids)) - point["last_token_probe"]["n_mixed_prompts"]
        ),
        "folds": fold_records,
        "selected_layers": sorted({record["selected_layer"] for record in fold_records}),
        "selected_penalties": sorted({record["selected_C"] for record in fold_records}),
        "scores": {
            name: {"point": point[name], "bootstrap": intervals[name]}
            for name in score_sets
        },
    }


def analyze_model(
    label: str, data_dir: str, layers: tuple[int, ...], cache_path: Path, *,
    populations: tuple[str, ...], n_splits: int, inner_splits: int,
    n_bootstrap: int, seed: int, max_workers: int,
    oof_csv: str | None = None, oof_layer: int | None = None,
) -> dict:
    _status(f"[{label}] last-token states from {data_dir}")
    cache = load_or_build_cache(data_dir, layers, cache_path, max_workers=max_workers)
    frozen = None
    if oof_csv:
        frozen = load_frozen_scores(oof_csv, int(oof_layer), cache["trace_id"])
        _status(f"  joined {sorted(frozen)} from {oof_csv} at layer {oof_layer}")
    return {
        "data_dir": data_dir,
        "layers": list(layers),
        "frozen_oof_csv": oof_csv,
        "frozen_oof_layer": oof_layer,
        "cache": str(cache_path),
        "hidden_dim": int(cache[f"states_L{layers[0]}"].shape[1]),
        "n_traces_collected": int(len(cache["trace_id"])),
        "n_unparsed": int(cache["unparsed"].sum()),
        "populations": {
            population: analyze_population(
                cache, population, n_splits=n_splits, inner_splits=inner_splits,
                n_bootstrap=n_bootstrap, seed=seed, frozen=frozen,
            )
            for population in populations
        },
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _ci(entry: dict | None) -> str:
    if not entry or entry.get("ci_low") is None:
        return "--"
    return f"[{entry['ci_low']:.4f}, {entry['ci_high']:.4f}]"


def write_report(body: dict, path: Path) -> None:
    lines = [
        "# The last-token probe, reproduced and decomposed",
        "",
        "Built by `controls/last_token_probe.py`. Last-token hidden state, "
        "L2 logistic readout, prompt-disjoint outer folds; layer *and* penalty "
        "chosen inside each training split by prompt-disjoint inner folds.",
        "",
        "Three readouts, reported on each population. `pooled` counts every "
        "held-out "
        "trace and ignores prompt identity -- it is the published claim shape. "
        "`micro` weights every within-prompt (correct, incorrect) pair; `macro` "
        "averages per-prompt AUROC with each prompt counting once. Prompts with "
        "a single outcome contribute traces to `pooled` but no pairs to `micro` "
        "and no term to `macro`, so the three numbers are not defined on the "
        "same population. That is the finding, not a caveat.",
        "",
    ]

    lines += ["## 1. What was fitted", "",
              "| Model | Layers offered | Layers chosen | C chosen | Hidden dim | Traces | Unparsed |",
              "|---|---|---|---|---:|---:|---:|"]
    for label, entry in body["models"].items():
        chosen = sorted({
            layer
            for population in entry["populations"].values()
            for layer in population["selected_layers"]
        })
        penalties = sorted({
            value
            for population in entry["populations"].values()
            for value in population["selected_penalties"]
        })
        lines.append(
            f"| {label} | {', '.join(str(l) for l in entry['layers'])} | "
            f"{', '.join(str(l) for l in chosen)} | "
            f"{', '.join(f'{value:g}' for value in penalties)} | "
            f"{entry['hidden_dim']} | "
            f"{entry['n_traces_collected']} | {entry['n_unparsed']} |"
        )
    lines.append("")

    for section, population in enumerate(body["populations"], start=2):
        lines += [f"## {section}. Population `{population}`", ""]
        first = next(iter(body["models"].values()))["populations"][population]
        lines += [
            "| Model | Traces | Prompts | Mixed | Single-outcome | Pairs | Base acc. |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label, entry in body["models"].items():
            stats = entry["populations"][population]
            probe = stats["scores"]["last_token_probe"]["point"]
            lines.append(
                f"| {label} | {stats['n_traces']} | {stats['n_prompts']} | "
                f"{probe['n_mixed_prompts']} | {stats['n_single_outcome_prompts']} | "
                f"{probe['n_within_prompt_pairs']} | {stats['base_rate']:.3f} |"
            )
        lines += ["", "### Pooled versus within-prompt", "",
                  "| Model | Score | Pooled | 95% CI | Micro pair | Macro prompt | "
                  "Prompt-centered | Pooled - macro | 95% CI |",
                  "|---|---|---:|---|---:|---:|---:|---:|---|"]
        for label, entry in body["models"].items():
            stats = entry["populations"][population]
            for name, scored in stats["scores"].items():
                point = scored["point"]
                boot = scored["bootstrap"]
                lines.append(
                    f"| {label} | `{name}` | {_fmt(point['pooled_auroc'])} | "
                    f"{_ci(boot['pooled_auroc'])} | {_fmt(point['micro_pair_auroc'])} | "
                    f"{_fmt(point['macro_prompt_auroc'])} | "
                    f"{_fmt(point['prompt_centered_auroc'])} | "
                    f"{_fmt(point['pooled_minus_macro'])} | "
                    f"{_ci(boot['pooled_minus_macro'])} |"
                )
        lines.append("")

    lines += [f"## {section + 1}. Layer selection inside training data", "",
              "Mean inner-fold pooled AUROC per layer, per outer fold. Selection "
              "never sees the outer test prompts.", "",
              "| Model | Population | Fold | Layer | C | Best inner pooled AUROC by layer |",
              "|---|---|---:|---:|---:|---|"]
    for label, entry in body["models"].items():
        for population, stats in entry["populations"].items():
            for record in stats["folds"]:
                by_layer = ", ".join(
                    f"L{key} {value:.4f}"
                    for key, value in sorted(record["inner_pooled_auroc_by_layer"].items(),
                                             key=lambda item: int(item[0]))
                )
                lines.append(
                    f"| {label} | `{population}` | {record['fold']} | "
                    f"{record['selected_layer']} | {record['selected_C']:g} | {by_layer} |"
                )
    lines += ["", "## What this does and does not establish", "",
              "The probe is fitted at the strength in-fold selection gives "
              "it, not at a fixed penalty. That matters: a loose penalty "
              "separates the training set perfectly and costs several points of "
              "held-out AUROC, which would understate the very claim this is "
              "meant to reproduce before decomposing.",
              "",
              "It establishes what happens to a published-style pooled trace "
              "AUROC when prompt identity is conditioned on, under the same "
              "protocol that produced it. It does not establish that any "
              "particular published number is wrong: the models, datasets, and "
              "training populations differ. The claim is about the claim shape.",
              "",
              "The intervals resample prompts with the fit held fixed. They do "
              "not carry the uncertainty of fold assignment, layer choice, or "
              "coefficients; that is the outer-refit blocker.", ""]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merge", action="append", default=None,
        help="Result JSONs from per-model runs to combine into one report. "
             "Models fit independently, so running them in parallel and "
             "merging is the same computation as one sequential pass.",
    )
    parser.add_argument(
        "--model", action="append", default=None,
        help="LABEL:DATA_DIR:LAYERS, e.g. qwen:data/qwen_bestofn_full/math500:7,14,21",
    )
    parser.add_argument(
        "--oof", action="append", default=None,
        help="LABEL:OOF_CSV:LAYER -- join the frozen pipeline's out-of-fold "
             "scores at one layer so they pass through the same three readouts.",
    )
    parser.add_argument("--population", action="append", default=None,
                        choices=list(POPULATIONS))
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    parser.add_argument("--inner_splits", type=int, default=INNER_SPLITS)
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--extract_workers", type=int, default=8)
    parser.add_argument("--cache_dir", default="results/last_token_probe/cache")
    parser.add_argument("--output_dir", default="results/last_token_probe")
    return parser.parse_args()


def merge_results(paths: list[str]) -> dict:
    """Combine per-model result JSONs, refusing a mismatched analysis."""
    merged: dict | None = None
    for path in paths:
        body = json.loads(Path(path).read_text())
        if merged is None:
            merged = {key: value for key, value in body.items() if key != "models"}
            merged["models"] = {}
            merged["runtime_seconds"] = 0.0
        elif body["config"] != merged["config"] or body["populations"] != merged["populations"]:
            raise ValueError(f"{path} was produced by a different analysis")
        overlap = set(body["models"]) & set(merged["models"])
        if overlap:
            raise ValueError(f"{path} repeats model(s) {sorted(overlap)}")
        merged["models"].update(body["models"])
        merged["runtime_seconds"] += body.get("runtime_seconds", 0.0)
    if merged is None:
        raise ValueError("nothing to merge")
    merged["runtime_seconds"] = round(merged["runtime_seconds"], 1)
    return merged


def main() -> None:
    args = parse_args()
    populations = tuple(args.population) if args.population else POPULATIONS
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        body = merge_results(args.merge)
        (output_dir / "last_token_probe_results.json").write_text(json.dumps(body, indent=2))
        write_report(body, output_dir / "last_token_probe_report.md")
        _status(f"merged {len(args.merge)} runs into {output_dir}")
        return
    if not args.model:
        raise SystemExit("--model is required unless --merge is given")

    frozen_specs = {}
    for spec in args.oof or []:
        label, csv_path, layer = spec.rsplit(":", 2)
        frozen_specs[label] = (csv_path, int(layer))

    started = time.perf_counter()
    models = {}
    for spec in args.model:
        label, data_dir, raw_layers = spec.split(":", 2)
        layers = tuple(int(value) for value in raw_layers.split(","))
        oof_csv, oof_layer = frozen_specs.get(label, (None, None))
        models[label] = analyze_model(
            label, data_dir, layers, cache_dir / f"{label}_last_token.npz",
            populations=populations, n_splits=args.n_splits,
            inner_splits=args.inner_splits, n_bootstrap=args.n_bootstrap,
            seed=args.seed, max_workers=args.extract_workers,
            oof_csv=oof_csv, oof_layer=oof_layer,
        )

    body = {
        "config": {
            "probe": {
                "readout": "logistic",
                "solver": PROBE_SOLVER,
                "dual": True,
                "C_grid": list(PROBE_C_GRID),
                "C_selected_in_fold": True,
                "max_iter": PROBE_MAX_ITER,
            },
            "n_splits": args.n_splits,
            "inner_splits": args.inner_splits,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
        },
        "populations": list(populations),
        "models": models,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    (output_dir / "last_token_probe_results.json").write_text(json.dumps(body, indent=2))
    write_report(body, output_dir / "last_token_probe_report.md")
    _status(f"wrote {output_dir} in {body['runtime_seconds']}s")


if __name__ == "__main__":
    main()
