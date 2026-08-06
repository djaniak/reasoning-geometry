"""Is the geometry feature a re-reading of the vote it is scored against?

Orgad et al. (arXiv:2410.02707, ICLR 2025) resample K=30 responses per prompt,
build a taxonomy of error types out of the resulting answer distribution, and show
that probes on hidden states predict that taxonomy.  Read adversarially -- and a
reviewer will read it adversarially -- it says hidden states *encode the resampling
agreement structure*.  That is exactly the quantity ``vote_agreement`` measures, so
the objection writes itself: ``rmd_tail_q20`` is a worse-instrumented vote, and the
B1 - B0 increment is a fitting artifact rather than new information.

The increment is already measured over a baseline that contains ``vote_agreement``,
which answers the objection in the supervised sense.  This module answers it in the
three forms a reviewer is more likely to accept, none of which involve a fitted
model absorbing the question:

1. **Redundancy.**  Pearson and Spearman between the two features.  A proxy has to
   correlate; the magnitude bounds how much of one can be the other.
2. **Within-stratum signal.**  AUROC of ``rmd_tail_q20`` computed *inside* a fixed
   level of agreement.  Agreement is constant there by construction, so anything the
   geometry separates inside a stratum is information agreement does not carry.
   The unanimous stratum is the load-bearing one: it is where self-consistency has
   nothing left to say, and it is the majority of prompts.
3. **Orthogonal component.**  Out-of-fold linear residual of ``rmd_tail_q20`` on
   ``vote_agreement``, scored on its own.  This is the same construction the frozen
   2026-07-31 length control used, applied to the vote instead of to length.

It also runs the substitution both ways -- geometry in place of the vote, and the
vote added back on top of geometry -- because "not a proxy" and "not redundant" are
different claims and the write-up needs both.

Not a DVC stage: it re-reads cached OOF rows and imports the frozen aggregation,
folds, and bootstrap from ``incremental_abstention`` rather than restating them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deepconf_asymmetry import _average_ranks, _pearson, auroc, bootstrap_auroc
from incremental_abstention import (
    BASE_FEATURE_NAMES,
    _population_ids,
    _read_oof,
    aggregate_prompt_features,
    crossfit_logistic_predictions,
    paired_bootstrap_delta,
    prompt_metrics,
    select_layer_rows,
)

#: The baseline with its self-consistency term removed.  Everything else in B0 is an
#: output-side statistic that says nothing about agreement between siblings.
VOTELESS_FEATURE_NAMES = tuple(
    name for name in BASE_FEATURE_NAMES if name != "vote_agreement"
)

#: Minimum prompts of *each* class for a within-stratum AUROC to be reported.  Below
#: this the interval is too wide to distinguish "no signal" from "no data", and a
#: quoted point estimate would invite exactly that confusion.
MIN_STRATUM_CLASS = 10


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson on tie-averaged ranks, over the pairs where both are finite."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    usable = np.isfinite(left) & np.isfinite(right)
    if usable.sum() < 3:
        return float("nan")
    return _pearson(_average_ranks(left[usable]), _average_ranks(right[usable]))


def crossfit_residuals(
    target: np.ndarray,
    covariate: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    """Out-of-fold residual of ``target`` after a linear fit on ``covariate``.

    Fitting in-sample would let the residual keep whatever the covariate explains on
    these very prompts, which is the leak the frozen readouts avoid everywhere else.
    """
    target = np.asarray(target, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    folds = np.asarray(folds)
    residuals = np.full(len(target), np.nan, dtype=float)
    for fold in np.unique(folds):
        test = folds == fold
        train = ~test & np.isfinite(target) & np.isfinite(covariate)
        if train.sum() < 2 or np.std(covariate[train]) < 1e-12:
            continue
        design = np.column_stack([np.ones(int(train.sum())), covariate[train]])
        coefficients, *_ = np.linalg.lstsq(design, target[train], rcond=None)
        predicted = coefficients[0] + coefficients[1] * covariate[test]
        residuals[test] = target[test] - predicted
    return residuals


def agreement_strata(agreement: np.ndarray) -> dict[str, np.ndarray]:
    """Unanimous versus split, as boolean masks over the prompt list.

    Agreement is a share of *parseable* siblings, so the denominator varies and the
    levels are not a clean eighths grid.  Unanimity is the one cut that means the
    same thing at every denominator, and it is where the objection actually bites:
    if geometry were an agreement proxy it would be flat across a stratum where
    agreement does not vary at all.
    """
    agreement = np.asarray(agreement, dtype=float)
    return {
        "unanimous": agreement >= 1.0,
        "split": np.isfinite(agreement) & (agreement < 1.0),
    }


def stratum_readout(
    scores: np.ndarray,
    outcomes: np.ndarray,
    mask: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """AUROC inside one stratum, or an explicit refusal when it is too small."""
    scores, outcomes = np.asarray(scores, dtype=float)[mask], np.asarray(outcomes, dtype=float)[mask]
    n_correct = int((outcomes > 0.5).sum())
    n_wrong = int((outcomes <= 0.5).sum())
    body = {
        "n_prompts": int(mask.sum()),
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "base_accuracy": float(np.mean(outcomes)) if len(outcomes) else float("nan"),
    }
    if min(n_correct, n_wrong) < MIN_STRATUM_CLASS:
        return {**body, "auroc": None, "reported": False}
    return {
        **body,
        "auroc": bootstrap_auroc(scores, outcomes, n_bootstrap=n_bootstrap, seed=seed),
        "reported": True,
    }


def agreement_level_counts(agreement: np.ndarray, outcomes: np.ndarray) -> list[dict]:
    """Prompts and accuracy at each distinct agreement level, for the scope note."""
    agreement = np.asarray(agreement, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    table = []
    for level in sorted(set(agreement[np.isfinite(agreement)].tolist())):
        mask = agreement == level
        table.append(
            {
                "agreement": float(level),
                "n_prompts": int(mask.sum()),
                "accuracy": float(np.mean(outcomes[mask])),
            }
        )
    return table


def _readout_specs() -> dict[str, tuple[str, ...]]:
    """B0/B1 plus the two substitutions the proxy reading implies.

    If geometry were a proxy for the vote, swapping one for the other would cost
    nothing and adding the vote back on top of geometry would gain nothing.
    """
    return {
        "B0": BASE_FEATURE_NAMES,
        "B1": BASE_FEATURE_NAMES + ("rmd_tail_q20",),
        "B0_voteless": VOTELESS_FEATURE_NAMES,
        "B0_rmd_for_vote": VOTELESS_FEATURE_NAMES + ("rmd_tail_q20",),
    }


def analyze_model(
    label: str,
    oof_csv: str | Path,
    data_dir: str | Path,
    *,
    population: str = "cap_free_valid_plurality",
    expected_traces: int = 8,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    rows, layer = select_layer_rows(_read_oof(oof_csv), context=str(oof_csv))
    features = aggregate_prompt_features(
        rows, data_dir=str(data_dir), expected_traces=expected_traces
    )
    prompt_ids = [
        # An unfolded prompt has no held-out readout, and the frozen analysis drops it
        # before scoring; keeping it here would change the population under comparison.
        i for i in _population_ids(features)[population] if features[i]["fold"] is not None
    ]
    outcomes = np.asarray([features[i]["outcome"] for i in prompt_ids], dtype=float)
    folds = np.asarray([features[i]["fold"] for i in prompt_ids])
    columns = {
        name: np.asarray([features[i][name] for i in prompt_ids], dtype=float)
        for name in BASE_FEATURE_NAMES + ("rmd_tail_q20",)
    }
    geometry, agreement = columns["rmd_tail_q20"], columns["vote_agreement"]

    marginal = {
        name: bootstrap_auroc(values, outcomes, n_bootstrap=n_bootstrap, seed=seed)
        for name, values in (("rmd_tail_q20", geometry), ("vote_agreement", agreement))
    }
    residual = crossfit_residuals(geometry, agreement, folds)
    mirror = crossfit_residuals(agreement, geometry, folds)

    predictions = {
        name: crossfit_logistic_predictions(
            np.column_stack([columns[column] for column in spec]), outcomes, folds, seed=seed
        )
        for name, spec in _readout_specs().items()
    }
    deltas = {}
    for left, right in (
        ("B1", "B0"),
        ("B0_rmd_for_vote", "B0"),
        ("B1", "B0_rmd_for_vote"),
        ("B0_rmd_for_vote", "B0_voteless"),
    ):
        deltas[f"{left}_minus_{right}"] = paired_bootstrap_delta(
            predictions[left], predictions[right], outcomes,
            metric="aurc", n_bootstrap=n_bootstrap, seed=seed,
        )

    return {
        "label": label,
        "population": population,
        "layer": layer,
        "n_prompts": len(prompt_ids),
        "base_accuracy": float(np.mean(outcomes)),
        "redundancy": {
            "pearson": _pearson(geometry, agreement),
            "spearman": spearman(geometry, agreement),
        },
        "marginal_auroc": marginal,
        "strata": {
            name: stratum_readout(
                geometry, outcomes, mask, n_bootstrap=n_bootstrap, seed=seed
            )
            for name, mask in agreement_strata(agreement).items()
        },
        "agreement_levels": agreement_level_counts(agreement, outcomes),
        "residual_auroc": {
            "geometry_given_agreement": bootstrap_auroc(
                residual, outcomes, n_bootstrap=n_bootstrap, seed=seed
            ),
            "agreement_given_geometry": bootstrap_auroc(
                mirror, outcomes, n_bootstrap=n_bootstrap, seed=seed
            ),
        },
        "readouts": {
            name: prompt_metrics(values, outcomes) for name, values in predictions.items()
        },
        "paired_deltas_aurc": deltas,
    }


def _fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _signed(entry) -> str:
    if entry is None or entry.get("point_estimate") is None:
        return "n/a"
    return (
        f"{entry['point_estimate']:+.4f} "
        f"[{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}] p={entry['p_two_sided']:.3f}"
    )


def _band(interval) -> str:
    """One `bootstrap_auroc` result as `point [low, high]`."""
    if interval is None or interval.get("ci_low") is None:
        return "n/a"
    return (
        f"{interval['point_estimate']:.3f} "
        f"[{interval['ci_low']:.3f}, {interval['ci_high']:.3f}]"
    )


def write_report(results: list[dict], path: str | Path) -> None:
    lines = [
        "# Is tail RMD a proxy for the vote it is scored against?",
        "",
        "Direct answer to Orgad et al. (arXiv:2410.02707): hidden states encode the",
        "resampling agreement structure, so a reviewer will read `rmd_tail_q20` as a",
        "worse-instrumented `vote_agreement`. AURC is reported (lower is better);",
        "AUROC is base-rate invariant and needs no such caveat.",
        "",
        "## 1. Redundancy between the two features",
        "",
        "| model | n | Pearson | Spearman | AUROC rmd_tail_q20 | AUROC vote_agreement |",
        "|---|---:|---:|---:|---|---|",
    ]
    for body in results:
        marginal = body["marginal_auroc"]
        lines.append(
            f"| {body['label']} | {body['n_prompts']} | "
            f"{_fmt(body['redundancy']['pearson'])} | {_fmt(body['redundancy']['spearman'])} | "
            f"{_band(marginal['rmd_tail_q20'])} | {_band(marginal['vote_agreement'])} |"
        )

    lines += [
        "",
        "## 2. Geometry inside a fixed level of agreement",
        "",
        "Agreement does not vary within a stratum, so a proxy cannot separate anything",
        "there. The unanimous stratum is where self-consistency has nothing left to say.",
        "",
        "| model | stratum | n | base acc | AUROC rmd_tail_q20 |",
        "|---|---|---:|---:|---|",
    ]
    for body in results:
        for name, stratum in body["strata"].items():
            cell = (
                _band(stratum["auroc"])
                if stratum["reported"]
                else "not reported (too few of one class)"
            )
            lines.append(
                f"| {body['label']} | {name} | {stratum['n_prompts']} | "
                f"{_fmt(stratum['base_accuracy'])} | {cell} |"
            )

    lines += [
        "",
        "## 3. Orthogonal component (out-of-fold linear residual)",
        "",
        "| model | AUROC of rmd_tail_q20 given vote | AUROC of vote given rmd_tail_q20 |",
        "|---|---|---|",
    ]
    for body in results:
        residual = body["residual_auroc"]
        lines.append(
            f"| {body['label']} | {_band(residual['geometry_given_agreement'])} | "
            f"{_band(residual['agreement_given_geometry'])} |"
        )

    lines += [
        "",
        "## 4. Substitution, both directions (AURC, lower is better)",
        "",
        "| model | B1 - B0 | rmd for vote - B0 | B1 - (rmd for vote) | rmd added to voteless |",
        "|---|---|---|---|---|",
    ]
    for body in results:
        deltas = body["paired_deltas_aurc"]
        lines.append(
            f"| {body['label']} | {_signed(deltas['B1_minus_B0'])} | "
            f"{_signed(deltas['B0_rmd_for_vote_minus_B0'])} | "
            f"{_signed(deltas['B1_minus_B0_rmd_for_vote'])} | "
            f"{_signed(deltas['B0_rmd_for_vote_minus_B0_voteless'])} |"
        )

    lines += [
        "",
        "## 5. Agreement levels present",
        "",
        "| model | agreement | n | accuracy |",
        "|---|---:|---:|---:|",
    ]
    for body in results:
        for level in body["agreement_levels"]:
            lines.append(
                f"| {body['label']} | {level['agreement']:.3f} | "
                f"{level['n_prompts']} | {level['accuracy']:.3f} |"
            )
    Path(path).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL:OOF_CSV:DATA_DIR",
        help="repeatable; colon-separated triple",
    )
    parser.add_argument("--population", default="cap_free_valid_plurality")
    parser.add_argument("--expected_traces", type=int, default=8)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/orgad_agreement_control")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for spec in args.model:
        label, oof_csv, data_dir = spec.split(":", 2)
        results.append(
            analyze_model(
                label,
                oof_csv,
                data_dir,
                population=args.population,
                expected_traces=args.expected_traces,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "orgad_agreement_control_results.json").write_text(json.dumps(results, indent=2))
    write_report(results, output / "orgad_agreement_control_report.md")
    print(f"wrote {output}/orgad_agreement_control_report.md")


if __name__ == "__main__":
    main()
