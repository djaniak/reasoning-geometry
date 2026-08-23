"""Post-hoc sensitivity of the frozen RMD tail-q20 prompt feature."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from analysis.analyze import set_compute_dtype, set_max_reference_tokens
from applications.incremental_abstention import (
    BASE_FEATURE_NAMES,
    _group_rows,
    _mean_field,
    _population_ids,
    aggregate_prompt_features,
    crossfit_logistic_predictions,
    paired_bootstrap_delta,
    prompt_metrics,
)
from applications.prompt_decomposition import generate_oof_scores_layerwise


DETECTOR_SCORE_KEYS = {
    "rmd_tail_q10": "rmd_tail_q10_score",
    "rmd_tail_q20": "rmd_tail_q20_score",
    "rmd_tail_q50": "rmd_tail_q50_score",
    "rmd_full": "rmd_score",
    "rmd_high_entropy_q20": "rmd_high_entropy_q20_score",
    "rmd_random_q20": "rmd_random_q20_score",
}
DETECTORS = tuple(DETECTOR_SCORE_KEYS)
FROZEN_DETECTOR = "rmd_tail_q20"
LOCALIZED_REGIONS = (
    "tail_q10",
    "tail_q20",
    "tail_q50",
    "high_entropy_q20",
    "random_q20",
)
ROBUSTNESS_SENTENCE = (
    "We fixed 20% as a simple localized window; sensitivity analysis shows that "
    "the result is not specific to this exact cutoff."
)
SENSITIVITY_SENTENCE = (
    "The RMD increment is sensitive to the tail-window cutoff; tail-q20 remains "
    "the frozen feature, but the exact cutoff is exploratory."
)


def add_detector_features(
    rows: Iterable[Mapping],
    *,
    max_new_tokens: int | None = None,
    data_dir: str | None = None,
    expected_traces: int = 8,
) -> dict[int, dict]:
    """Aggregate every detector over sibling traces using the frozen convention."""
    rows = list(rows)
    features = aggregate_prompt_features(
        rows,
        max_new_tokens=max_new_tokens,
        data_dir=data_dir,
        expected_traces=expected_traces,
    )
    for prompt_id, group in _group_rows(rows).items():
        features[prompt_id].update(
            {
                detector: _mean_field(group, score_key)
                for detector, score_key in DETECTOR_SCORE_KEYS.items()
            }
        )
    return features


def analyze_population(
    features: Mapping[int, Mapping],
    prompt_ids: Iterable[int],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    prompt_ids = list(prompt_ids)
    outcomes = np.asarray([features[i]["outcome"] for i in prompt_ids], dtype=float)
    folds = np.asarray([features[i]["fold"] for i in prompt_ids], dtype=int)
    columns = {
        name: np.asarray([features[i][name] for i in prompt_ids], dtype=float)
        for name in BASE_FEATURE_NAMES + DETECTORS
    }
    predictions = {
        "B0": crossfit_logistic_predictions(
            np.column_stack([columns[name] for name in BASE_FEATURE_NAMES]),
            outcomes,
            folds,
            seed=seed,
        )
    }
    for detector in DETECTORS:
        predictions[f"B0_plus_{detector}"] = crossfit_logistic_predictions(
            np.column_stack(
                [columns[name] for name in BASE_FEATURE_NAMES + (detector,)]
            ),
            outcomes,
            folds,
            seed=seed,
        )

    deltas = {
        f"{detector}_over_B0": paired_bootstrap_delta(
            predictions[f"B0_plus_{detector}"],
            predictions["B0"],
            outcomes,
            metric="aurc",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        for detector in DETECTORS
    }
    for detector in DETECTORS:
        if detector == FROZEN_DETECTOR:
            continue
        deltas[f"{detector}_over_{FROZEN_DETECTOR}"] = paired_bootstrap_delta(
            predictions[f"B0_plus_{detector}"],
            predictions[f"B0_plus_{FROZEN_DETECTOR}"],
            outcomes,
            metric="aurc",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

    return {
        "n_prompts": len(prompt_ids),
        "base_accuracy": float(np.mean(outcomes)),
        "readouts": {
            name: prompt_metrics(values, outcomes)
            for name, values in predictions.items()
        },
        "paired_deltas_aurc": deltas,
    }


def robustness_verdict(models: Iterable[Mapping]) -> dict:
    """Require q10, q20, and q50 to improve over B0 with 95% intervals."""
    checks = {}
    for model in models:
        deltas = model["paired_deltas_aurc"]
        checks[str(model["label"])] = {
            detector: bool(deltas[f"{detector}_over_B0"]["ci_high"] < 0.0)
            for detector in ("rmd_tail_q10", "rmd_tail_q20", "rmd_tail_q50")
        }
    passed = bool(checks) and all(all(values.values()) for values in checks.values())
    return {
        "rule": "q10, q20, and q50 each improve AURC over B0 with a 95% interval below zero on every checkpoint",
        "by_model": checks,
        "passed": passed,
        "paper_sentence": ROBUSTNESS_SENTENCE if passed else SENSITIVITY_SENTENCE,
    }


def analyze_model(
    *,
    label: str,
    data_dir: str,
    layer: int,
    max_new_tokens: int,
    expected_prompts: int,
    expected_traces: int,
    pca_dim: int,
    n_splits: int,
    n_bootstrap: int,
    seed: int,
    load_workers: int,
) -> dict:
    rows, data_report = generate_oof_scores_layerwise(
        data_dir=data_dir,
        layers=[layer],
        expected_prompts=expected_prompts,
        n=expected_traces,
        allow_partial=False,
        pca_dim=pca_dim,
        n_splits=n_splits,
        seed=seed,
        load_workers=load_workers,
        show_progress=True,
        localized_rmd_regions=LOCALIZED_REGIONS,
        hidden_dtype=np.float16,
    )
    features = add_detector_features(
        rows,
        max_new_tokens=max_new_tokens,
        data_dir=data_dir,
        expected_traces=expected_traces,
    )
    prompt_ids = [
        prompt_id
        for prompt_id in _population_ids(features)["full_population"]
        if features[prompt_id]["fold"] is not None
    ]
    result = analyze_population(
        features, prompt_ids, n_bootstrap=n_bootstrap, seed=seed
    )
    result.update(
        {
            "label": label,
            "layer": int(layer),
            "max_new_tokens": int(max_new_tokens),
            "data_report": data_report,
            "prompt_features": [
                {
                    key: features[prompt_id][key]
                    for key in (
                        "prompt_id",
                        "fold",
                        "outcome",
                        *BASE_FEATURE_NAMES,
                        *DETECTORS,
                    )
                }
                for prompt_id in prompt_ids
            ],
        }
    )
    return result


def build_report(result: Mapping) -> str:
    lines = [
        "# RMD tail-window sensitivity",
        "",
        "Post-hoc sensitivity analysis. `rmd_tail_q20` remains frozen; this run does not select a detector.",
        "",
        "AURC deltas are `B0 + detector` minus `B0`; lower is better.",
        "",
        "| model | detector | delta AURC [95% CI] |",
        "|---|---|---:|",
    ]
    for model in result["models"]:
        for detector in DETECTORS:
            delta = model["paired_deltas_aurc"][f"{detector}_over_B0"]
            lines.append(
                f"| {model['label']} | `{detector}` | "
                f"{delta['point_estimate']:+.4f} [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}] |"
            )
    lines.extend(
        [
            "",
            f"**Cutoff-robustness rule:** {result['robustness']['rule']}",
            "",
            f"**Verdict:** {result['robustness']['paper_sentence']}",
            "",
            "ATRMD, high-entropy-q20, and random-q20 are contextual controls. They do not justify selecting 20% as optimal.",
            "",
        ]
    )
    return "\n".join(lines)


def _model_spec(value: str) -> dict:
    try:
        label, data_dir, layer, max_new_tokens = value.rsplit(":", 3)
        return {
            "label": label,
            "data_dir": data_dir,
            "layer": int(layer),
            "max_new_tokens": int(max_new_tokens),
        }
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "model must be LABEL:DATA_DIR:LAYER:MAX_NEW_TOKENS"
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=_model_spec, required=True)
    parser.add_argument("--output-dir", default="results/rmd_window_sensitivity")
    parser.add_argument("--expected-prompts", type=int, default=500)
    parser.add_argument("--expected-traces", type=int, default=8)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-workers", type=int, default=4)
    parser.add_argument("--max-reference-tokens", type=int, default=2_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_compute_dtype("float32")
    set_max_reference_tokens(args.max_reference_tokens)
    models = [
        analyze_model(
            **spec,
            expected_prompts=args.expected_prompts,
            expected_traces=args.expected_traces,
            pca_dim=args.pca_dim,
            n_splits=args.n_splits,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            load_workers=args.load_workers,
        )
        for spec in args.model
    ]
    result = {
        "status": "post_hoc_sensitivity",
        "frozen_detector": FROZEN_DETECTOR,
        "detectors": DETECTOR_SCORE_KEYS,
        "population": "full_population",
        "settings": {
            "pca_dim": args.pca_dim,
            "n_splits": args.n_splits,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "max_reference_tokens": args.max_reference_tokens,
        },
        "models": models,
    }
    result["robustness"] = robustness_verdict(models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rmd_window_sensitivity_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (output_dir / "rmd_window_sensitivity_report.md").write_text(
        build_report(result)
    )


if __name__ == "__main__":
    main()
