"""Pre-specified Wave 1 cross-model confirmation summary (E3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PRIMARY_CONTRASTS = (
    "rmd_high_entropy_q20_minus_rmd",
    "rmd_high_entropy_q20_minus_rmd_random_q20",
)


def summarize_model(
    path: str | Path,
    *,
    model_label: str,
    min_mixed_prompts: int = 30,
) -> dict:
    data = json.loads(Path(path).read_text())
    settings = data.get("settings", {})
    layers = [int(layer) for layer in settings.get("layers", [])]
    deepest_layer = max(layers) if layers else None
    layer = data.get("layers", {}).get(str(deepest_layer), {}) if deepest_layer is not None else {}
    parseable = layer.get("parseable_only", {})
    deltas = parseable.get("paired_score_deltas", {})
    truncation = data.get("truncation", {})
    n_mixed = int(parseable.get("methods", {}).get("rmd", {}).get("n_mixed_prompts", 0) or 0)
    return {
        "model": model_label,
        "artifact": str(path),
        "deepest_layer": deepest_layer,
        "n_parseable_traces": parseable.get("n_parseable_traces"),
        "n_mixed_prompts": n_mixed,
        "cap_hit_rate": truncation.get("capped_rate"),
        "underpowered": n_mixed < int(min_mixed_prompts),
        "contrasts": {
            name: deltas.get(name)
            for name in PRIMARY_CONTRASTS
        },
    }


def write_confirmation_report(result: dict, path: str | Path) -> None:
    lines = [
        "# Wave 1 E3 — cross-model confirmation",
        "",
        "The deepest extracted layer was fixed before reading the contrast values. Only localization and entropy-specificity were requested.",
        "",
        "| Model | Layer | Mixed prompts | Cap-hit rate | Status | Localization | Entropy-specificity |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for model in result["models"]:
        status = "underpowered" if model["underpowered"] else "adequate"
        values = []
        for contrast in PRIMARY_CONTRASTS:
            item = model["contrasts"].get(contrast)
            if not item:
                values.append("NA")
                continue
            metric = item.get("prompt_centered_auc", {})
            def _fmt(value):
                return "NA" if value is None else f"{float(value):+.3f}"
            values.append(
                f"{_fmt(metric.get('point_estimate'))} "
                f"[{_fmt(metric.get('ci_low'))}, {_fmt(metric.get('ci_high'))}], "
                f"p={_fmt(metric.get('p_two_sided'))}"
            )
        lines.append(
            f"| {model['model']} | {model['deepest_layer']} | {model['n_mixed_prompts']} | "
            f"{model['cap_hit_rate']} | {status} | {values[0]} | {values[1]} |"
        )
    lines.extend(["", "No multiplicity-adjusted or post-hoc layer claims are made.", ""])
    Path(path).write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--decomposition", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_mixed_prompts", type=int, default=30)
    args = parser.parse_args()
    result = {
        "experiment": "E3_cross_model_confirmation",
        "prespecified_contrasts": list(PRIMARY_CONTRASTS),
        "models": [
            summarize_model(
                args.decomposition,
                model_label=args.model_label,
                min_mixed_prompts=args.min_mixed_prompts,
            )
        ],
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.model_label}_math500_wave1_cross_model"
    (output / f"{prefix}_results.json").write_text(json.dumps(result, indent=2) + "\n")
    write_confirmation_report(result, output / f"{prefix}_report.md")


if __name__ == "__main__":
    main()
