"""Why does DeepConf look strong on DeepSeek-Qwen and useless on Llama?

The 2026-08-05 entry reports `DeepConf_tail_q20` scoring AUACC 0.799 on
DeepSeek-Qwen, and the 2026-08-06 entry reports 0.625 on Llama.  Both entries
read that gap as a fact about the baseline: DeepConf is a strong competitor on
one model and a weak one on the other, so clearing it on Llama is the easier
test.  That reading has never been checked, and it is the first thing a reviewer
asks after seeing two entries that disagree.

There is an obvious confound.  **AUACC is not zero-based.**  A score that ranks
prompts at chance still integrates to the base accuracy, so a model that answers
80% of prompts correctly hands every readout a free 0.80 and a model at 67%
hands it 0.67.  The two AUACC numbers above are therefore not comparable across
models, and neither is any "DeepConf against B0" gap read off them.

This module recomputes the comparison in two forms that do not move with the
base rate: **excess AUACC** (`auacc - base_accuracy`, the discrimination the
readout adds over ranking at chance) and **AUROC** over the raw feature, which
is invariant to class balance by construction.  It then asks what else differs
between the models -- effect size on the outcome, redundancy against B0's four
features, and the absolute size of the 20% tail window the statistic averages
over, since the two models were collected at different budgets (8192 against
12288).

Not a DVC stage and not a new abstention variant: it re-reads two artifacts that
already exist.  It imports the frozen aggregation from `incremental_abstention`
rather than copying it, so the prompt features and populations here are the same
objects the locked results were computed from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from incremental_abstention import (
    BASE_FEATURE_NAMES,
    _auacc,
    _population_ids,
    _read_oof,
    aggregate_prompt_features,
)

# `_load_exact_prompt_scores` reads only the first two, so the comparison on
# record never saw `bottom10_group_confidence` -- which is DeepConf's own
# headline statistic. All four are in the stored artifact, so all four are read
# here and the omission can be checked rather than assumed harmless.
DEEPCONF_FEATURES = (
    "deepconf_global",
    "deepconf_tail_q20",
    "bottom10_group_confidence",
    "lowest_group_confidence",
)
RAW_FEATURES = BASE_FEATURE_NAMES + ("rmd_tail_q20",) + DEEPCONF_FEATURES


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, so tied scores cannot bias AUROC either way."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_values = values[order]
    start = 0
    for stop in range(1, len(values) + 1):
        if stop == len(values) or sorted_values[stop] != sorted_values[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
    return ranks


def auroc(scores: np.ndarray, outcomes: np.ndarray) -> float:
    """Probability a correct prompt outranks an incorrect one; 0.5 is chance.

    Unlike AUACC this does not inherit the base accuracy, which is the whole
    reason the module exists.
    """
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    usable = np.isfinite(scores) & np.isfinite(outcomes)
    scores, outcomes = scores[usable], outcomes[usable]
    positives = outcomes > 0.5
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = _average_ranks(scores)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cohens_d(scores: np.ndarray, outcomes: np.ndarray) -> float:
    """Standardized mean difference between correct and incorrect prompts."""
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    usable = np.isfinite(scores) & np.isfinite(outcomes)
    scores, outcomes = scores[usable], outcomes[usable]
    right, wrong = scores[outcomes > 0.5], scores[outcomes <= 0.5]
    if len(right) < 2 or len(wrong) < 2:
        return float("nan")
    pooled = np.sqrt(((len(right) - 1) * right.var(ddof=1)
                      + (len(wrong) - 1) * wrong.var(ddof=1))
                     / (len(right) + len(wrong) - 2))
    return float((right.mean() - wrong.mean()) / pooled) if pooled > 1e-12 else float("nan")


def bootstrap_auroc(
    scores: np.ndarray,
    outcomes: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Percentile interval over prompt resamples, matching the frozen convention."""
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    point = auroc(scores, outcomes)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(n_bootstrap)):
        index = rng.integers(0, len(outcomes), len(outcomes))
        value = auroc(scores[index], outcomes[index])
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return {"point_estimate": point, "ci_low": None, "ci_high": None, "n_valid": 0}
    return {
        "point_estimate": point,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "n_valid": len(draws),
    }


def load_all_deepconf_scores(path: str | Path) -> dict[int, dict[str, float]]:
    """Every DeepConf statistic in the artifact, averaged over a prompt's traces.

    The same sibling-mean aggregation `incremental_abstention` uses, widened to
    the statistics that module does not read.
    """
    with np.load(Path(path), allow_pickle=True) as data:
        rows = data["trace_summaries"].tolist()
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["prompt_id"]), []).append(row)
    return {
        prompt_id: {
            key: float(np.mean([float(row[key]) for row in group]))
            for key in DEEPCONF_FEATURES
        }
        for prompt_id, group in grouped.items()
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    usable = np.isfinite(left) & np.isfinite(right)
    left, right = left[usable], right[usable]
    if len(left) < 3 or left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def trace_length_summary(rows: list[dict], prompt_ids: list[int]) -> dict:
    """Absolute size of the window `deepconf_tail_q20` averages over.

    The statistic is defined on the final 20% of a trace, so the same definition
    covers a different number of tokens on models collected at different budgets.
    """
    keep = set(prompt_ids)
    lengths = np.asarray(
        [float(row["trace_length"]) for row in rows if int(row["prompt_id"]) in keep],
        dtype=float,
    )
    lengths = lengths[np.isfinite(lengths)]
    if not len(lengths):
        return {"n_traces": 0}
    return {
        "n_traces": int(len(lengths)),
        "median_trace_length": float(np.median(lengths)),
        "median_tail_window_tokens": float(np.median(lengths) * 0.2),
        "iqr_trace_length": [float(np.percentile(lengths, 25)), float(np.percentile(lengths, 75))],
    }


def analyze_model(
    *,
    label: str,
    oof_csv: str | Path,
    data_dir: str | Path,
    exact_scores_npz: str | Path,
    results_json: str | Path,
    population: str = "cap_free_valid_plurality",
    expected_traces: int = 8,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    rows = _read_oof(oof_csv)
    deepconf = load_all_deepconf_scores(exact_scores_npz)
    features = aggregate_prompt_features(
        rows,
        data_dir=str(data_dir),
        expected_traces=expected_traces,
    )
    prompt_ids = _population_ids(features)[population]
    outcomes = np.asarray([features[i]["outcome"] for i in prompt_ids], dtype=float)
    base_accuracy = float(np.nanmean(outcomes))

    columns = {}
    for name in RAW_FEATURES:
        if name in DEEPCONF_FEATURES:
            values = [deepconf.get(i, {}).get(name, float("nan")) for i in prompt_ids]
        else:
            values = [features[i][name] for i in prompt_ids]
        columns[name] = np.asarray(values, dtype=float)

    raw = {}
    for name, values in columns.items():
        interval = bootstrap_auroc(values, outcomes, n_bootstrap=n_bootstrap, seed=seed)
        raw[name] = {
            "auroc": interval["point_estimate"],
            "auroc_ci": [interval["ci_low"], interval["ci_high"]],
            "cohens_d": cohens_d(values, outcomes),
            "auacc": _auacc(values, outcomes),
            "excess_auacc": _auacc(values, outcomes) - base_accuracy,
        }

    stored = json.loads(Path(results_json).read_text())["populations"][population]
    fitted = {
        name: {
            "auacc": body["metrics"]["auacc"],
            "excess_auacc": body["metrics"]["auacc"] - stored["base_accuracy"],
        }
        for name, body in stored["models"].items()
    }

    redundancy = {
        name: {
            other: _pearson(columns[name], columns[other])
            for other in BASE_FEATURE_NAMES + ("rmd_tail_q20",)
        }
        for name in DEEPCONF_FEATURES
    }
    spread = {
        name: {"mean": float(np.nanmean(columns[name])), "sd": float(np.nanstd(columns[name]))}
        for name in DEEPCONF_FEATURES
    }

    return {
        "label": label,
        "population": population,
        "n_prompts": len(prompt_ids),
        "base_accuracy": base_accuracy,
        "stored_base_accuracy": stored["base_accuracy"],
        "raw_features": raw,
        "fitted_readouts": fitted,
        "deepconf_redundancy_pearson": redundancy,
        "deepconf_spread": spread,
        "trace_lengths": trace_length_summary(rows, prompt_ids),
    }


def write_report(result: dict, path: str | Path) -> None:
    lines = [
        "# DeepConf asymmetry between DeepSeek-Qwen and Llama",
        "",
        f"Population: `{result['population']}`. Seed {result['seed']},"
        f" {result['n_bootstrap']} bootstrap draws.",
        "",
        "## Base rate first",
        "",
        "| model | n | base accuracy |",
        "|---|---:|---:|",
    ]
    for model in result["models"]:
        lines.append(f"| {model['label']} | {model['n_prompts']} | {model['base_accuracy']:.4f} |")

    lines += ["", "## Fitted readouts, as excess over the base rate", "",
              "| readout | " + " | ".join(
                  f"{m['label']} AUACC | {m['label']} excess" for m in result["models"]
              ) + " |",
              "|---" * (1 + 2 * len(result["models"])) + "|"]
    names = [n for n in result["models"][0]["fitted_readouts"] if
             all(n in m["fitted_readouts"] for m in result["models"])]
    for name in names:
        cells = []
        for model in result["models"]:
            body = model["fitted_readouts"][name]
            cells += [f"{body['auacc']:.4f}", f"{body['excess_auacc']:+.4f}"]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    lines += ["", "## Raw features, AUROC (base-rate invariant; 0.5 is chance)", "",
              "| feature | " + " | ".join(
                  f"{m['label']} AUROC | {m['label']} d" for m in result["models"]
              ) + " |",
              "|---" * (1 + 2 * len(result["models"])) + "|"]
    for name in RAW_FEATURES:
        cells = []
        for model in result["models"]:
            body = model["raw_features"][name]
            low, high = body["auroc_ci"]
            cells += [f"{body['auroc']:.3f} [{low:.3f}, {high:.3f}]", f"{body['cohens_d']:+.3f}"]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    lines += ["", "## DeepConf statistics against B0's features (Pearson)", "",
              "| model | statistic | " + " | ".join(BASE_FEATURE_NAMES + ("rmd_tail_q20",)) + " |",
              "|---" * (3 + len(BASE_FEATURE_NAMES)) + "|"]
    for model in result["models"]:
        for name in DEEPCONF_FEATURES:
            body = model["deepconf_redundancy_pearson"][name]
            cells = [f"{body[k]:+.3f}" for k in BASE_FEATURE_NAMES + ("rmd_tail_q20",)]
            lines.append(f"| {model['label']} | `{name}` | " + " | ".join(cells) + " |")

    lines += ["", "## Tail window size", "",
              "| model | median trace length | median tail window (tokens) |",
              "|---|---:|---:|"]
    for model in result["models"]:
        body = model["trace_lengths"]
        lines.append(
            f"| {model['label']} | {body['median_trace_length']:.0f} |"
            f" {body['median_tail_window_tokens']:.0f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL:OOF_CSV:DATA_DIR:EXACT_NPZ:RESULTS_JSON",
        help="one colon-separated model spec; pass twice to compare two models",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--population", default="cap_free_valid_plurality")
    parser.add_argument("--expected_traces", type=int, default=8)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = []
    for spec in args.model:
        label, oof_csv, data_dir, exact_npz, results_json = spec.split(":")
        models.append(
            analyze_model(
                label=label,
                oof_csv=oof_csv,
                data_dir=data_dir,
                exact_scores_npz=exact_npz,
                results_json=results_json,
                population=args.population,
                expected_traces=args.expected_traces,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
        )
    result = {
        "population": args.population,
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "models": models,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "deepconf_asymmetry_results.json").write_text(json.dumps(result, indent=2))
    write_report(result, output_dir / "deepconf_asymmetry_report.md")
    for model in models:
        tail = model["raw_features"]["deepconf_tail_q20"]
        print(
            f"{model['label']}: n={model['n_prompts']} base={model['base_accuracy']:.4f} "
            f"deepconf_tail AUROC={tail['auroc']:.3f} excess_auacc={tail['excess_auacc']:+.4f}"
        )


if __name__ == "__main__":
    main()
