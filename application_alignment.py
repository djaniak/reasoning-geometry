"""Relate confidence variance structure to downstream application performance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


METHOD_SCORERS = {
    "raw": ("top1_raw", "raw_mahal_L{layer}"),
    "rmd": ("top1_rmd", "raw_rmd_L{layer}"),
}

CORRELATION_SPECS = {
    "within_auc_vs_top1_gain": (
        "within_prompt_pair_weighted",
        "top1_gain_over_random",
    ),
    "prompt_correlation_vs_selective_gain": (
        "prompt_score_pass_rate_spearman",
        "selective_ausc_gain_over_entropy",
    ),
    "icc_vs_selective_gain": (
        "score_icc",
        "selective_ausc_gain_over_entropy",
    ),
}


def _require(mapping: dict, key: str, context: str):
    if key not in mapping:
        raise ValueError(f"Missing {context}: {key}")
    return mapping[key]


def _correlation(rows: list[dict], x_key: str, y_key: str) -> dict:
    valid = [
        row
        for row in rows
        if row.get(x_key) is not None
        and row.get(y_key) is not None
        and math.isfinite(float(row[x_key]))
        and math.isfinite(float(row[y_key]))
    ]
    if len(valid) < 2:
        return {"spearman": None, "n": len(valid)}
    x = np.asarray([float(row[x_key]) for row in valid])
    y = np.asarray([float(row[y_key]) for row in valid])
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"spearman": None, "n": len(valid)}
    return {
        "spearman": float(spearmanr(x, y).statistic),
        "n": len(valid),
    }


def _correlation_block(rows: list[dict]) -> dict:
    return {
        name: _correlation(rows, x_key, y_key)
        for name, (x_key, y_key) in CORRELATION_SPECS.items()
    }


def build_application_alignment(
    decompositions: dict[str, dict],
    selections: dict[str, dict],
    selective_results: dict[str, dict],
) -> dict:
    conditions = []
    for model in sorted(decompositions):
        decomposition = decompositions[model]
        selection = _require(selections, model, f"selection result for {model}")
        selective = _require(
            selective_results, model, f"selective-prediction result for {model}"
        )
        entropy_ausc = _require(
            _require(selective, "scorers", f"{model} selective scorers"),
            "entropy_mean",
            f"{model} selective entropy scorer",
        ).get("ausc")
        if entropy_ausc is None:
            raise ValueError(f"Missing {model} entropy AUSC")

        for layer, layer_result in sorted(
            decomposition.get("layers", {}).items(),
            key=lambda item: int(item[0]),
        ):
            selection_layer = _require(
                selection.get("layers", {}),
                layer,
                f"{model} L{layer} selection layer",
            )
            selectors = _require(
                selection_layer, "selectors", f"{model} L{layer} selectors"
            )
            random_score = _require(
                selectors, "random", f"{model} L{layer} random selector"
            )["pass_at_1"]
            majority_score = _require(
                selectors,
                "majority_vote",
                f"{model} L{layer} majority selector",
            )["pass_at_1"]

            for method, (selector_name, scorer_template) in METHOD_SCORERS.items():
                context = f"{model} L{layer} {method}"
                method_result = _require(
                    layer_result.get("methods", {}),
                    method,
                    f"{context} decomposition method",
                )
                metrics = _require(
                    method_result, "metrics", f"{context} decomposition metrics"
                )
                top1 = _require(selectors, selector_name, f"{context} selector")[
                    "pass_at_1"
                ]
                scorer_name = scorer_template.format(layer=layer)
                selective_scorer = _require(
                    selective["scorers"], scorer_name, f"{context} selective scorer"
                )
                scorer_ausc = selective_scorer.get("ausc")
                if scorer_ausc is None:
                    raise ValueError(f"Missing {context} selective AUSC")

                conditions.append(
                    {
                        "model": model,
                        "dataset": decomposition.get("dataset"),
                        "layer": int(layer),
                        "method": method,
                        "within_prompt_pair_weighted": metrics.get(
                            "within_prompt_pair_weighted"
                        ),
                        "prompt_centered_auc": metrics.get(
                            "prompt_centered_auc"
                        ),
                        "score_icc": metrics.get("score_icc"),
                        "prompt_score_pass_rate_spearman": metrics.get(
                            "prompt_score_pass_rate_spearman"
                        ),
                        "top1_pass_at_1": float(top1),
                        "random_pass_at_1": float(random_score),
                        "majority_pass_at_1": float(majority_score),
                        "top1_gain_over_random": float(top1 - random_score),
                        "top1_gap_to_majority": float(top1 - majority_score),
                        "selective_ausc": float(scorer_ausc),
                        "entropy_ausc": float(entropy_ausc),
                        "selective_ausc_gain_over_entropy": float(
                            scorer_ausc - entropy_ausc
                        ),
                    }
                )

    correlations = {
        "pooled": _correlation_block(conditions),
        "by_method": {
            method: _correlation_block(
                [row for row in conditions if row["method"] == method]
            )
            for method in METHOD_SCORERS
        },
    }
    return {
        "warning": (
            "Exploratory descriptive correlations only; the number of independent "
            "model-layer conditions is too small for significance claims."
        ),
        "conditions": conditions,
        "correlations": correlations,
    }


def _parse_model_paths(values: list[str]) -> dict[str, dict]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected MODEL=PATH, received {value!r}")
        model, path = value.split("=", 1)
        result[model] = json.loads(Path(path).read_text())
    return result


def write_markdown(result: dict, path: str | Path) -> None:
    lines = [
        "# Application alignment",
        "",
        result["warning"],
        "",
        (
            "| Model | Layer | Method | Within AUC | Centered AUC | ICC | "
            "Prompt corr | Top-1 gain | Gap to majority | Selective gain |"
        ),
        "|:---|---:|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["conditions"]:
        lines.append(
            f"| {row['model']} | {row['layer']} | {row['method']} "
            f"| {row['within_prompt_pair_weighted']:.3f} "
            f"| {row['prompt_centered_auc']:.3f} "
            f"| {row['score_icc']:.3f} "
            f"| {row['prompt_score_pass_rate_spearman']:.3f} "
            f"| {row['top1_gain_over_random']:+.3f} "
            f"| {row['top1_gap_to_majority']:+.3f} "
            f"| {row['selective_ausc_gain_over_entropy']:+.3f} |"
        )
    lines.extend(["", "## Descriptive correlations", ""])
    for group, block in result["correlations"].items():
        if group == "by_method":
            for method, method_block in block.items():
                for name, payload in method_block.items():
                    value = payload["spearman"]
                    rendered = "NA" if value is None else f"{value:.3f}"
                    lines.append(
                        f"- `{method}/{name}`: Spearman={rendered}, n={payload['n']}"
                    )
        else:
            for name, payload in block.items():
                value = payload["spearman"]
                rendered = "NA" if value is None else f"{value:.3f}"
                lines.append(
                    f"- `pooled/{name}`: Spearman={rendered}, n={payload['n']}"
                )
    Path(path).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomposition", action="append", required=True)
    parser.add_argument("--selection", action="append", required=True)
    parser.add_argument("--selective", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_label", default="math500")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_application_alignment(
        _parse_model_paths(args.decomposition),
        _parse_model_paths(args.selection),
        _parse_model_paths(args.selective),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.dataset_label}_application_alignment"
    (output_dir / f"{prefix}_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_markdown(result, output_dir / f"{prefix}_report.md")


if __name__ == "__main__":
    main()
