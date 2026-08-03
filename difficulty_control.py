"""Does the tail-RMD increment survive a sharper prompt-difficulty control?

The headline result is that a prompt-level tail-RMD feature adds AUACC over an
output-side readout B0 = (length, entropy, logprob, vote_agreement).  The
increment is *larger* on the cap-free population than the full one, which has
been read as evidence that truncation does not carry it.

That reading has a competitor.  Capping is prompt-structured, not
sample-structured: finished-sibling accuracy falls monotonically with the number
of capped siblings, and at affected prompts the longest *finishing* sibling
already burns a median 88% of the budget against 35% at unaffected prompts
(EXPERIMENT_LOG.md, 2026-08-03 budget-limited noncompletion).  So "hard prompt",
"prompt caps", and "prompt answers wrong" are one axis.  Removing capped prompts
removes the band where B0's cheap features are *most* informative, which weakens
B0 and widens geometry's margin over it for a reason that has nothing to do with
geometry.

B0 controls for the *mean* trace length, which is a blunt difficulty proxy.  This
module hands B0 the sharp one the sibling-structure work produced -- budget-edge
pressure, computed from trace lengths alone, with no geometry and no hidden
states -- and re-asks whether B1 still adds anything.

On `cap_free_valid_plurality`, the headline population, `capped_fraction` is zero
by construction.  The feature that has to do the work there is
`longest_finisher_frac`, which has variance everywhere.

Not a DVC stage and not a variant of the claim: it is a falsification of it.  It
imports the frozen analysis rather than copying it, and uses that module's seed
convention, so `B1_minus_B0` here must reproduce the locked artifact exactly --
that agreement is the harness check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from incremental_abstention import (
    BASE_FEATURE_NAMES,
    _finite,
    _group_rows,
    _population_ids,
    _read_oof,
    aggregate_prompt_features,
    crossfit_logistic_predictions,
    paired_bootstrap_delta,
    prompt_metrics,
)
from trace_caps import resolve_cap


# Endogenous difficulty: budget-edge pressure, measured from the traces.
DIFFICULTY_FEATURE_NAMES = (
    "longest_finisher_frac",
    "capped_fraction",
    "length_dispersion",
)

# Exogenous difficulty: MATH-500's own human-annotated level, 1-5. It is the
# stronger control precisely because it never saw the model -- the endogenous
# features are computed from the same traces the readouts are fitted on, so they
# can be collinear with what B0 already has, and are.
EXOGENOUS_FEATURE_NAMES = ("level",)

METRICS = ("auacc", "brier")


def sibling_difficulty(rows, *, cap: int) -> dict[int, dict[str, float]]:
    """Budget-edge pressure per prompt, from trace lengths only.

    ``longest_finisher_frac`` is the sibling-structure statistic: how far into the
    budget the longest sibling that actually terminated had to go.  It is NaN when
    every sibling capped, since there is then no finisher to measure; the
    cross-fit imputes it from the training prompts rather than inventing a value.

    ``length_dispersion`` is included because a prompt whose siblings disagree
    wildly about how long the problem takes is a different kind of hard from one
    whose siblings all run long together.
    """
    difficulty: dict[int, dict[str, float]] = {}
    for prompt_id, group in sorted(_group_rows(rows).items()):
        lengths = [
            value
            for row in group
            if (value := _finite(row.get("trace_length"))) is not None
        ]
        finished = [value for value in lengths if value < cap]
        difficulty[prompt_id] = {
            "longest_finisher_frac": max(finished) / cap if finished else float("nan"),
            "capped_fraction": (
                sum(value >= cap for value in lengths) / len(lengths)
                if lengths
                else float("nan")
            ),
            "length_dispersion": (
                float(np.std(lengths)) / cap if len(lengths) > 1 else float("nan")
            ),
        }
    return difficulty


def dataset_levels(dataset_label: str) -> dict[int, float]:
    """MATH-500's annotated difficulty level, keyed by ``prompt_id``.

    ``prompt_id`` is the row index of the test split: the collect took the first
    500 rows in order.  Verified against the stored gold answers -- 448/500 match
    as exact strings and the other 52 differ only by whitespace or case
    (``'p-q'`` against ``'p - q'``), so the alignment is the identity.
    """
    if dataset_label != "math500":
        return {}
    from datasets import load_dataset

    split = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return {index: float(row["level"]) for index, row in enumerate(split)}


def _delta_seed(seed: int, label: str, metric: str) -> int:
    """The seed convention of `incremental_abstention.run_incremental_analysis`.

    Matched exactly so `B1_minus_B0` reproduces the locked artifact rather than
    merely agreeing to three decimals.
    """
    return seed + 1000 + len(metric) + len(label)


def run_difficulty_control(
    *,
    oof_csv: str,
    output_dir: str,
    model_label: str,
    dataset_label: str = "math500",
    layer: int,
    data_dir: str | None = None,
    max_new_tokens: int | None = None,
    expected_traces: int = 8,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    rows = [row for row in _read_oof(oof_csv) if int(row["layer"]) == layer]
    if not rows:
        raise ValueError(f"no rows at layer {layer} in {oof_csv}")

    cap = resolve_cap(
        max_new_tokens,
        data_dir=data_dir,
        lengths=(_finite(row.get("trace_length")) for row in rows),
        context="difficulty_control",
    )
    features = aggregate_prompt_features(
        rows,
        max_new_tokens=max_new_tokens,
        data_dir=data_dir,
        expected_traces=expected_traces,
    )
    for prompt_id, values in sibling_difficulty(rows, cap=cap.value).items():
        features[prompt_id].update(values)
    levels = dataset_levels(dataset_label)
    for prompt_id, entry in features.items():
        entry["level"] = levels.get(prompt_id, float("nan"))

    controls = {"difficulty": DIFFICULTY_FEATURE_NAMES}
    if levels:
        controls["level"] = EXOGENOUS_FEATURE_NAMES

    specs = {"B0": BASE_FEATURE_NAMES, "B1": BASE_FEATURE_NAMES + ("rmd_tail_q20",)}
    comparisons = [
        # The frozen headline, recomputed here as the harness check.
        ("B1", "B0", "B1_minus_B0"),
    ]
    for control, columns in controls.items():
        specs[f"B0_plus_{control}"] = BASE_FEATURE_NAMES + columns
        specs[f"B1_plus_{control}"] = BASE_FEATURE_NAMES + columns + ("rmd_tail_q20",)
        comparisons += [
            # Does geometry still add over a control-aware output readout?
            (f"B1_plus_{control}", f"B0_plus_{control}", f"B1_minus_B0_given_{control}"),
            # And what is the control worth on its own?
            (f"B0_plus_{control}", "B0", f"{control}_minus_B0"),
        ]

    result = {
        "model": model_label,
        "dataset": dataset_label,
        "layer": layer,
        "max_new_tokens": cap.value,
        "cap_provenance": cap.provenance,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "controls": {name: list(columns) for name, columns in controls.items()},
        "question": (
            "Is the B1-B0 increment a geometry effect, or a prompt-difficulty "
            "effect that B0's mean-length feature is too blunt to absorb?"
        ),
        "populations": {},
    }

    for population, prompt_ids in _population_ids(features).items():
        prompt_ids = [pid for pid in prompt_ids if features[pid]["fold"] is not None]
        if len(prompt_ids) < 2:
            continue
        y = np.asarray([features[pid]["outcome"] for pid in prompt_ids], dtype=float)
        folds = np.asarray([features[pid]["fold"] for pid in prompt_ids])
        entry = {
            "n_prompts": len(prompt_ids),
            "base_accuracy": float(np.mean(y)),
            "models": {},
            "paired_deltas": {},
        }
        predictions: dict[str, np.ndarray] = {}
        for name, columns in specs.items():
            x = np.asarray(
                [[features[pid].get(column, np.nan) for column in columns]
                 for pid in prompt_ids],
                dtype=float,
            )
            predictions[name] = crossfit_logistic_predictions(x, y, folds, seed=seed)
            entry["models"][name] = {
                "features": list(columns),
                "metrics": prompt_metrics(predictions[name], y),
            }
        for left, right, label in comparisons:
            for metric in METRICS:
                entry["paired_deltas"][f"{label}_{metric}"] = paired_bootstrap_delta(
                    predictions[left],
                    predictions[right],
                    y,
                    metric=metric,
                    n_bootstrap=n_bootstrap,
                    seed=_delta_seed(seed, label, metric),
                )
        result["populations"][population] = entry

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{dataset_label}_difficulty_control_results.json"
    path.write_text(json.dumps(result, indent=2))
    return result


def _row(entry: dict, label: str) -> str:
    delta = entry["paired_deltas"].get(f"{label}_auacc") or {}
    point = delta.get("point_estimate")
    if point is None:
        return "n/a"
    return (
        f"{point:+.3f} [{delta['ci_low']:+.3f}, {delta['ci_high']:+.3f}] "
        f"p={delta['p_two_sided']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--dataset_label", default="math500")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--expected_traces", type=int, default=8)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_difficulty_control(
        oof_csv=args.oof_csv,
        output_dir=args.output_dir,
        model_label=args.model_label,
        dataset_label=args.dataset_label,
        layer=args.layer,
        data_dir=args.data_dir,
        max_new_tokens=args.max_new_tokens,
        expected_traces=args.expected_traces,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(f"\n{result['model']} @ cap {result['max_new_tokens']} ({result['cap_provenance']})")
    labels = ["B1_minus_B0"] + [
        f"B1_minus_B0_given_{name}" for name in result["controls"]
    ] + [f"{name}_minus_B0" for name in result["controls"]]
    header = "".join(f"{label:<32}" for label in labels)
    print(f"{'population':<28} {'n':>4}  {header}")
    for population, entry in result["populations"].items():
        cells = "".join(f"{_row(entry, label):<32}" for label in labels)
        print(f"{population:<28} {entry['n_prompts']:>4}  {cells}")


if __name__ == "__main__":
    main()
