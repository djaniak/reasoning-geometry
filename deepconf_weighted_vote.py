"""DeepConf as DeepConf uses it: weighted and filtered voting inside a prompt.

Every DeepConf comparison on record scores a prompt by the *mean* of a trace-level
confidence over its eight siblings, then asks whether that mean predicts plurality
correctness.  On 2026-08-06 all four statistics came out at chance under that
readout on both models.  The obvious objection is that the aggregation throws away
exactly what DeepConf is for: the statistic is not meant to be averaged across
siblings, it is meant to *reweight* them -- confidence-weighted majority voting,
``V(a) = sum_t C_t * I(answer_t = a)``, and confidence filtering, which drops the
least-confident traces before the vote.  A statistic can be useless as a prompt-level
mean and still improve the vote.

This module runs both.  It joins the exact per-trace confidences to the cached OOF
rows and asks three separate questions:

1. **Does the weighted vote beat plain agreement as an abstention feature?**  The
   target stays plurality-vote correctness, so the faithful drop-in for
   ``vote_agreement`` is the confidence-weighted share *of the plurality winner*.
2. **Does the geometry increment survive a DeepConf-weighted baseline?**  B0 with
   its vote replaced by the weighted share, and B0 with the weighted share added
   alongside, are both strictly stronger baselines than the frozen B0.
3. **Does DeepConf's own answer selection beat plurality voting at all?**  Weighted
   and top-k-filtered voting select a possibly different answer; whether that answer
   is more often right is DeepConf's own claim, and it is checkable here.

``bottom10_group_confidence`` is the primary weight because it is the trace-level
measure DeepConf's own headline results use -- and the one
``incremental_abstention._load_exact_prompt_scores`` never loaded.

Not a DVC stage: it re-reads two artifacts that already exist and imports the frozen
aggregation, folds, and bootstrap from ``incremental_abstention`` rather than copying
them.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from deepconf_asymmetry import auroc
from incremental_abstention import (
    BASE_FEATURE_NAMES,
    METRIC_NAMES,
    _auacc,
    _population_ids,
    _read_oof,
    _winning_answer,
    aggregate_prompt_features,
    crossfit_logistic_predictions,
    is_parseable_answer,
    paired_bootstrap_delta,
    prompt_metrics,
)

DEEPCONF_STATISTICS = (
    "bottom10_group_confidence",
    "deepconf_tail_q20",
    "deepconf_global",
    "lowest_group_confidence",
)
#: Statistics that get a full readout rather than only a vote-share AUROC. The
#: first is DeepConf's own headline measure, the second is the one on record.
READOUT_STATISTICS = ("bottom10_group_confidence", "deepconf_tail_q20")
#: Survivor counts for confidence filtering. DeepConf sweeps a retention *fraction*
#: over hundreds of traces; at eight siblings the reachable fractions are these.
FILTER_KEEP = (1, 2, 4, 6)


def load_trace_confidence(path: str | Path) -> dict[tuple[int, int], dict[str, float]]:
    """Per-trace DeepConf statistics keyed by ``(prompt_id, trace_id)``.

    Deliberately *not* aggregated: the sibling-mean is the thing under test.
    """
    with np.load(Path(path), allow_pickle=True) as data:
        rows = data["trace_summaries"].tolist()
    return {
        (int(row["prompt_id"]), int(row["trace_id"])): {
            key: float(row[key]) for key in DEEPCONF_STATISTICS
        }
        for row in rows
    }


def _answer_outcome(rows: list[dict], answer: str | None) -> float:
    """Whether ``answer`` is the gold answer, matching `prompt_selection`."""
    if not is_parseable_answer(answer):
        return 0.0
    gold = rows[0].get("gold_answer")
    if gold not in (None, ""):
        return float(str(answer) == str(gold))
    matching = [row for row in rows if str(row.get("predicted_answer")) == str(answer)]
    return float(any(int(row["is_correct"]) for row in matching))


def weighted_vote(
    rows: list[dict],
    weights: dict[int, float],
    *,
    keep: int | None = None,
) -> dict:
    """DeepConf's offline vote ``V(a) = sum_t C_t * I(answer_t = a)``.

    ``keep`` applies confidence filtering first, retaining the ``keep`` most
    confident traces.  Returns the selected answer, its weighted share, and the
    weighted share of the *plurality* winner -- the latter is the target-consistent
    drop-in for ``vote_agreement``, since the outcome being predicted is still
    plurality-vote correctness.
    """
    parsed = [
        row
        for row in rows
        if is_parseable_answer(row.get("predicted_answer"))
        and np.isfinite(weights.get(int(row["trace_id"]), np.nan))
    ]
    empty = {"winner": None, "own_share": float("nan"), "plurality_share": float("nan"),
             "n_traces": 0}
    if not parsed:
        return empty
    if any(weights[int(row["trace_id"])] <= 0 for row in parsed):
        # A weighted share is only a share if the weights are positive. DeepConf's
        # C = mean(-log p) over top-k candidates always is; a non-positive value
        # means the wrong column was passed, which must not degrade silently.
        raise ValueError("confidence weights must be strictly positive")
    # The outcome being predicted is the *prompt's* plurality correctness, so the
    # reference answer is fixed before filtering. Recomputing it on the survivors
    # makes the score constant at keep=1 and drifts off-target above it.
    plurality = _winning_answer(parsed)
    if keep is not None:
        # Sort by confidence, then trace_id, so equal confidences cannot make the
        # survivor set depend on row order.
        parsed = sorted(
            parsed,
            key=lambda row: (-weights[int(row["trace_id"])], int(row["trace_id"])),
        )[:keep]

    totals: dict[str, float] = defaultdict(float)
    for row in parsed:
        totals[str(row["predicted_answer"])] += weights[int(row["trace_id"])]
    total = sum(totals.values())
    winner = max(sorted(totals), key=lambda answer: totals[answer])
    return {
        "winner": winner,
        "own_share": totals[winner] / total,
        "plurality_share": totals.get(str(plurality), 0.0) / total,
        "n_traces": len(parsed),
    }


def prompt_vote_features(
    rows_by_prompt: dict[int, list[dict]],
    confidence: dict[tuple[int, int], dict[str, float]],
) -> dict[int, dict[str, float]]:
    """Weighted and filtered vote scores, one row per prompt.

    Keys are ``dcvote_<stat>`` (weighted share of the plurality winner),
    ``dcvote_own_<stat>`` (share of DeepConf's own winner), and
    ``dcfilter{k}_<stat>`` for each survivor count.
    """
    features: dict[int, dict[str, float]] = {}
    for prompt_id, rows in rows_by_prompt.items():
        entry: dict[str, float] = {}
        for statistic in DEEPCONF_STATISTICS:
            weights = {
                int(row["trace_id"]): confidence[(prompt_id, int(row["trace_id"]))][statistic]
                for row in rows
                if (prompt_id, int(row["trace_id"])) in confidence
            }
            full = weighted_vote(rows, weights)
            entry[f"dcvote_{statistic}"] = full["plurality_share"]
            entry[f"dcvote_own_{statistic}"] = full["own_share"]
            for keep in FILTER_KEEP:
                entry[f"dcfilter{keep}_{statistic}"] = weighted_vote(
                    rows, weights, keep=keep
                )["plurality_share"]
        features[prompt_id] = entry
    return features


def weight_dispersion(
    rows_by_prompt: dict[int, list[dict]],
    confidence: dict[tuple[int, int], dict[str, float]],
    prompt_ids: list[int],
) -> dict:
    """How much room the weights have to change a vote.

    Weighting can only move a vote to the extent siblings disagree about
    confidence.  ``C = mean(-log p)`` over top-k candidates sits in a narrow
    positive band, so this reports the within-prompt coefficient of variation and
    the max/min ratio -- the ceiling on any reweighting effect.
    """
    keep = set(prompt_ids)
    table = {}
    for statistic in DEEPCONF_STATISTICS:
        cvs, ratios = [], []
        for prompt_id, rows in rows_by_prompt.items():
            if prompt_id not in keep:
                continue
            values = np.asarray(
                [
                    confidence[(prompt_id, int(row["trace_id"]))][statistic]
                    for row in rows
                    if (prompt_id, int(row["trace_id"])) in confidence
                ],
                dtype=float,
            )
            if len(values) < 2 or values.min() <= 0:
                continue
            cvs.append(float(values.std() / values.mean()))
            ratios.append(float(values.max() / values.min()))
        table[statistic] = {
            "median_within_prompt_cv": float(np.median(cvs)) if cvs else float("nan"),
            "median_max_over_min": float(np.median(ratios)) if ratios else float("nan"),
            "n_prompts": len(cvs),
        }
    return table


def selection_accuracy(
    rows_by_prompt: dict[int, list[dict]],
    confidence: dict[tuple[int, int], dict[str, float]],
    prompt_ids: list[int],
) -> dict:
    """Accuracy of the answer each voting rule *selects*, DeepConf's own claim.

    Plurality voting is the incumbent; weighted and filtered voting may select a
    different answer.  All are scored against gold on the same prompts, alongside
    how often each rule departs from plurality at all -- an accuracy that matches
    plurality because the rule never disagrees is a different fact from one that
    disagrees often and breaks even.
    """
    keep_set = set(prompt_ids)
    rows_by_prompt = {k: v for k, v in rows_by_prompt.items() if k in keep_set}
    plurality_answers = {
        prompt_id: _winning_answer(rows) for prompt_id, rows in rows_by_prompt.items()
    }
    accuracy = {
        "plurality": float(
            np.mean([
                _answer_outcome(rows, plurality_answers[prompt_id])
                for prompt_id, rows in rows_by_prompt.items()
            ])
        )
    }
    disagreement = {"plurality": 0.0}
    for statistic in DEEPCONF_STATISTICS:
        for keep in (None, *FILTER_KEEP):
            outcomes, differs = [], []
            for prompt_id, rows in rows_by_prompt.items():
                weights = {
                    int(row["trace_id"]): confidence[(prompt_id, int(row["trace_id"]))][statistic]
                    for row in rows
                    if (prompt_id, int(row["trace_id"])) in confidence
                }
                selected = weighted_vote(rows, weights, keep=keep)["winner"]
                outcomes.append(_answer_outcome(rows, selected))
                differs.append(float(str(selected) != str(plurality_answers[prompt_id])))
            label = f"weighted_{statistic}" if keep is None else f"top{keep}_{statistic}"
            accuracy[label] = float(np.mean(outcomes))
            disagreement[label] = float(np.mean(differs))
    return {
        "accuracy": accuracy,
        "disagreement_with_plurality": disagreement,
        "n_prompts": len(rows_by_prompt),
    }


def _readout_specs(statistics: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """B0/B1 plus, for each statistic, a DeepConf-strengthened baseline family."""
    specs: dict[str, tuple[str, ...]] = {
        "B0": BASE_FEATURE_NAMES,
        "B1": BASE_FEATURE_NAMES + ("rmd_tail_q20",),
    }
    swapped = tuple(name for name in BASE_FEATURE_NAMES if name != "vote_agreement")
    for statistic in statistics:
        vote = f"dcvote_{statistic}"
        specs[f"B0_dcvote_{statistic}"] = swapped + (vote,)
        specs[f"B0_plus_dcvote_{statistic}"] = BASE_FEATURE_NAMES + (vote,)
        specs[f"B1_dcvote_{statistic}"] = swapped + (vote, "rmd_tail_q20")
        specs[f"B1_plus_dcvote_{statistic}"] = BASE_FEATURE_NAMES + (vote, "rmd_tail_q20")
    return specs


def analyze_model(
    *,
    label: str,
    oof_csv: str | Path,
    data_dir: str | Path,
    exact_scores_npz: str | Path,
    expected_traces: int = 8,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    rows = _read_oof(oof_csv)
    layer = max(int(row["layer"]) for row in rows)
    rows = [row for row in rows if int(row["layer"]) == layer]
    rows_by_prompt: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_prompt[int(row["prompt_id"])].append(row)

    confidence = load_trace_confidence(exact_scores_npz)
    vote_features = prompt_vote_features(dict(rows_by_prompt), confidence)
    features = aggregate_prompt_features(
        rows, data_dir=str(data_dir), expected_traces=expected_traces
    )
    for prompt_id, entry in features.items():
        entry.update(vote_features.get(prompt_id, {}))

    specs = _readout_specs(READOUT_STATISTICS)
    populations = {}
    for population, prompt_ids in _population_ids(features).items():
        prompt_ids = [i for i in prompt_ids if features[i]["fold"] is not None]
        if len(prompt_ids) < 2:
            continue
        y = np.asarray([features[i]["outcome"] for i in prompt_ids], dtype=float)
        folds = np.asarray([features[i]["fold"] for i in prompt_ids])
        entry = {
            "n_prompts": len(prompt_ids),
            "base_accuracy": float(np.mean(y)),
            "models": {},
            "paired_deltas": {},
            "raw_vote_features": {},
        }

        for name in ("vote_agreement",) + tuple(
            key for key in sorted(vote_features[prompt_ids[0]]) if key.startswith("dc")
        ):
            values = np.asarray([features[i].get(name, np.nan) for i in prompt_ids], dtype=float)
            entry["raw_vote_features"][name] = {
                "auroc": auroc(values, y),
                "auacc": _auacc(values, y),
                "excess_auacc": _auacc(values, y) - float(np.mean(y)),
            }

        predictions: dict[str, np.ndarray] = {}
        for name, columns in specs.items():
            x = np.asarray(
                [[features[i].get(column, np.nan) for column in columns] for i in prompt_ids],
                dtype=float,
            )
            predictions[name] = crossfit_logistic_predictions(x, y, folds, seed=seed)
            entry["models"][name] = {
                "features": list(columns),
                "metrics": prompt_metrics(predictions[name], y),
            }

        comparisons = [("B1", "B0", "B1_minus_B0")]
        for statistic in READOUT_STATISTICS:
            comparisons += [
                # Does the geometry increment survive when the baseline's vote is
                # DeepConf-weighted rather than plain agreement?
                (f"B1_dcvote_{statistic}", f"B0_dcvote_{statistic}",
                 f"B1_minus_B0_dcvote_{statistic}"),
                (f"B1_plus_dcvote_{statistic}", f"B0_plus_dcvote_{statistic}",
                 f"B1_minus_B0_plus_dcvote_{statistic}"),
                # Does the weighted vote strengthen the baseline at all?
                (f"B0_dcvote_{statistic}", "B0", f"B0_dcvote_{statistic}_minus_B0"),
                (f"B0_plus_dcvote_{statistic}", "B0", f"B0_plus_dcvote_{statistic}_minus_B0"),
                # And is frozen B1 still ahead of the strengthened baseline?
                ("B1", f"B0_plus_dcvote_{statistic}",
                 f"B1_minus_B0_plus_dcvote_{statistic}_baseline"),
            ]
        for left, right, name in comparisons:
            for metric in METRIC_NAMES:
                entry["paired_deltas"][f"{name}_{metric}"] = paired_bootstrap_delta(
                    predictions[left], predictions[right], y,
                    metric=metric, n_bootstrap=n_bootstrap,
                    seed=seed + 3000 + len(metric) + len(name),
                )
        entry["selection_accuracy"] = selection_accuracy(
            dict(rows_by_prompt), confidence, prompt_ids
        )
        entry["weight_dispersion"] = weight_dispersion(
            dict(rows_by_prompt), confidence, prompt_ids
        )
        populations[population] = entry

    return {
        "label": label,
        "layer": layer,
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "readout_statistics": list(READOUT_STATISTICS),
        "filter_keep": list(FILTER_KEEP),
        "populations": populations,
    }


def write_report(result: dict, path: str | Path, *, population: str) -> None:
    lines = [
        "# DeepConf used inside the prompt: weighted and filtered voting",
        "",
        f"Population: `{population}`. Seed {result['models'][0]['seed']},"
        f" {result['models'][0]['n_bootstrap']} bootstrap draws.",
        "",
        "## 1. Vote scores as raw abstention features (AUROC, 0.5 is chance)",
        "",
        "| feature | " + " | ".join(m["label"] for m in result["models"]) + " |",
        "|---" * (1 + len(result["models"])) + "|",
    ]
    names = list(result["models"][0]["populations"][population]["raw_vote_features"])
    for name in names:
        cells = [
            f"{m['populations'][population]['raw_vote_features'][name]['auroc']:.3f}"
            for m in result["models"]
        ]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    lines += ["", "## 2. Readouts (AUACC, and excess over the base rate)", "",
              "| readout | " + " | ".join(
                  f"{m['label']} AUACC | {m['label']} excess" for m in result["models"]
              ) + " |",
              "|---" * (1 + 2 * len(result["models"])) + "|"]
    for name in result["models"][0]["populations"][population]["models"]:
        cells = []
        for model in result["models"]:
            body = model["populations"][population]
            auacc = body["models"][name]["metrics"]["auacc"]
            cells += [f"{auacc:.4f}", f"{auacc - body['base_accuracy']:+.4f}"]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    lines += ["", "## 3. Paired deltas (AUACC)", "",
              "| comparison | " + " | ".join(m["label"] for m in result["models"]) + " |",
              "|---" * (1 + len(result["models"])) + "|"]
    deltas = [
        key for key in result["models"][0]["populations"][population]["paired_deltas"]
        if key.endswith("_auacc")
    ]
    for key in deltas:
        cells = []
        for model in result["models"]:
            body = model["populations"][population]["paired_deltas"][key]
            cells.append(
                f"{body['point_estimate']:+.4f} [{body['ci_low']:+.3f}, {body['ci_high']:+.3f}]"
                f" p={body['p_two_sided']:.3f}"
            )
        lines.append(f"| `{key[: -len('_auacc')]}` | " + " | ".join(cells) + " |")

    lines += ["", "## 4. Accuracy of the answer each rule selects"
              " (and how often it departs from plurality)", "",
              "| rule | " + " | ".join(
                  f"{m['label']} acc | {m['label']} differs" for m in result["models"]
              ) + " |",
              "|---" * (1 + 2 * len(result["models"])) + "|"]
    table = result["models"][0]["populations"][population]["selection_accuracy"]["accuracy"]
    for rule in table:
        cells = []
        for model in result["models"]:
            body = model["populations"][population]["selection_accuracy"]
            cells += [
                f"{body['accuracy'][rule]:.4f}",
                f"{body['disagreement_with_plurality'][rule]:.3f}",
            ]
        lines.append(f"| `{rule}` | " + " | ".join(cells) + " |")

    lines += ["", "## 5. How much room the weights have to change anything", "",
              "| statistic | " + " | ".join(
                  f"{m['label']} within-prompt CV | {m['label']} max/min"
                  for m in result["models"]
              ) + " |",
              "|---" * (1 + 2 * len(result["models"])) + "|"]
    for statistic in DEEPCONF_STATISTICS:
        cells = []
        for model in result["models"]:
            body = model["populations"][population]["weight_dispersion"][statistic]
            cells += [
                f"{body['median_within_prompt_cv']:.4f}",
                f"{body['median_max_over_min']:.3f}",
            ]
        lines.append(f"| `{statistic}` | " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL:OOF_CSV:DATA_DIR:EXACT_NPZ",
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
        label, oof_csv, data_dir, exact_npz = spec.split(":")
        models.append(
            analyze_model(
                label=label,
                oof_csv=oof_csv,
                data_dir=data_dir,
                exact_scores_npz=exact_npz,
                expected_traces=args.expected_traces,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
        )
    result = {"population": args.population, "models": models}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "deepconf_weighted_vote_results.json").write_text(json.dumps(result, indent=2))
    write_report(
        result, output_dir / "deepconf_weighted_vote_report.md", population=args.population
    )
    for model in models:
        body = model["populations"][args.population]
        primary = READOUT_STATISTICS[0]
        delta = body["paired_deltas"][f"B1_minus_B0_plus_dcvote_{primary}_auacc"]
        print(
            f"{model['label']}: n={body['n_prompts']} base={body['base_accuracy']:.4f} "
            f"B1-B0+dcvote={delta['point_estimate']:+.4f} "
            f"[{delta['ci_low']:+.3f}, {delta['ci_high']:+.3f}] p={delta['p_two_sided']:.3f} "
            f"| plurality={body['selection_accuracy']['accuracy']['plurality']:.4f} "
            f"weighted={body['selection_accuracy']['accuracy'][f'weighted_{primary}']:.4f}"
        )


if __name__ == "__main__":
    main()
