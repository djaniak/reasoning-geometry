"""Does tail RMD survive a difficulty control the target model cannot see?

Experiment 2 of the 2026-08-09 direction review.  ``B0`` controls for difficulty
with the *mean trace length* of the target model's own siblings, which is blunt
and, worse, endogenous: it is computed from the same traces the readout is fitted
on, so it can be collinear with whatever ``rmd_tail_q20`` is picking up.  Two
existing rungs already probe this from inside the model -- ``difficulty_control``
hands ``B0`` the budget-edge statistics, and ``incremental_abstention`` hands it
the target's own prompt-state geometry -- and both share that weakness.

This rung uses a control that no part of the target model produced.  All three
collects ran the same 500 MATH-500 problems under the same prompt ids, so for
each prompt there are two *other* models' eight-sibling pass rates sitting in
cached CSVs.  A prompt that two other 7-8B models solve 8/8 is empirically easy;
one they solve 0/8 is empirically hard.  That is a sharper difficulty measure
than mean trace length and a sharper one than MATH-500's five annotated levels,
and it is exogenous to the target's hidden states by construction.

The question is narrow: with those two pass rates already in the readout, does
``rmd_tail_q20`` still add?

Two things this is **not**.  It is not a deployment-time method -- you cannot run
two other eight-sample models to decide whether to trust this one, so
``B0 + peer`` is a control, never a baseline the headline has to beat.  And the
peer rate is not label leakage: it is a property of the *problem*, computed from
generations the target never saw, exactly as MATH-500's annotated level is.  It
is an unusually informative such property, which is the entire point.

Pre-declared rules, written before the run:

* If ``B1 + peer`` minus ``B0 + peer`` has an AURC interval overlapping zero on
  two or more of the three models, the increment is reported as substantially a
  prompt-difficulty proxy and the "geometry adds beyond output-side confidence"
  framing narrows accordingly.
* One peer definition -- the mean of ``is_correct`` over all eight cached
  siblings -- and no sweep of alternatives follows, whichever way it lands.  No
  thresholding of the rate, no parseable-only variant, no third pooling.

Sign convention: **AURC, lower is better**, so a negative delta favours the
left-hand readout.

Not a DVC stage: it re-reads cached OOF rows and imports the frozen aggregation,
folds, populations, readout, bootstrap and seed convention rather than restating
any of them.  ``B1_minus_B0`` must therefore reproduce the locked
``incremental_abstention`` artifact exactly, intervals included; that agreement
is the harness check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from closest_baselines import _populations
from deepconf_asymmetry import _pearson, bootstrap_auroc
from difficulty_control import _delta_seed
from incremental_abstention import (
    BASE_FEATURE_NAMES,
    _finite,
    _group_rows,
    _read_oof,
    aggregate_prompt_features,
    crossfit_logistic_predictions,
    paired_bootstrap_delta,
    prompt_metrics,
    select_layer_rows,
)
from orgad_agreement_control import spearman

#: Column names are ``PEER_PREFIX + <peer label>``.  Each target model gets its
#: own design matrix, so naming the columns after the peer keeps the report
#: readable without making the readouts comparable across targets -- they are not
#: meant to be; only the contrasts are.
PEER_PREFIX = "peer_pass_rate__"

#: AURC carries the pre-declared rule, matching 1a and 1b.  AUACC is reported
#: alongside because the locked artifact and `difficulty_control` are both in it,
#: and a reader comparing rungs should not have to change metric mid-table.
METRICS = ("aurc", "auacc")

#: Above this |Spearman| against the target's own outcome, the control is close
#: enough to an oracle that "geometry survives it" and "geometry is redundant
#: with it" stop being the only two readings -- a near-oracle control can also
#: simply saturate the readout.  Reported, not enforced: it changes how the
#: number is discussed, not whether it is computed.
NEAR_ORACLE_SPEARMAN = 0.60


def oracle_aurc(outcomes: np.ndarray) -> float:
    """AURC of a ranker that orders every correct prompt ahead of every wrong one.

    AURC does not bottom out at zero: with a fixed base accuracy some risk is
    unrankable away, and the floor rises as accuracy falls.  This matters here
    because a strong control can push a readout close to that floor, and once it
    does, "the tail adds nothing on top" and "there was nothing left to add" are
    different statements that the delta alone cannot tell apart.

    Computed through ``prompt_metrics`` so it is the same trapezoidal AURC the
    readouts are scored with, not a second convention.
    """
    outcomes = np.asarray(outcomes, dtype=float)
    return float(prompt_metrics(outcomes, outcomes)["aurc"])


def _share(removed: float, headroom: float) -> float:
    """``removed`` as a fraction of the risk that was still removable.

    NaN rather than a large number when the headroom is non-positive: a readout
    already at or below the oracle floor has nothing left to give up, and any
    ratio there would be an artifact of the denominator.
    """
    return float(removed / headroom) if headroom > 0 else float("nan")


def peer_pass_rates(rows: Iterable[Mapping]) -> dict[int, float]:
    """Fraction of a model's cached siblings that were correct, per prompt.

    The denominator is every cached sibling, not the parseable ones: a trace that
    produced no extractable answer did not solve the problem, and empirical
    difficulty is about whether the problem got solved.  With eight siblings this
    is a nine-valued discrete variable, which is coarser than it looks and is why
    two peers are used rather than one.
    """
    rates: dict[int, float] = {}
    for prompt_id, group in sorted(_group_rows(rows).items()):
        values = [
            value for row in group if (value := _finite(row.get("is_correct"))) is not None
        ]
        rates[prompt_id] = float(np.mean(values)) if values else float("nan")
    return rates


def assert_shared_prompt_ids(golds: Mapping[str, Mapping[int, str]]) -> set[int]:
    """Fail loudly unless every model answered the same problem under each id.

    The whole experiment rests on ``prompt_id`` meaning the same problem in three
    separately collected runs.  Gold answers are stored per row, so that
    assumption is checkable rather than assumable, and a silent misalignment here
    would look exactly like a difficulty control that does not work.
    """
    labels = sorted(golds)
    shared = set.intersection(*(set(golds[label]) for label in labels))
    reference = golds[labels[0]]
    mismatched = sorted(
        prompt_id
        for prompt_id in shared
        for label in labels[1:]
        if _normalize_gold(golds[label][prompt_id]) != _normalize_gold(reference[prompt_id])
    )
    if mismatched:
        raise ValueError(
            f"prompt ids do not denote the same problem across models; "
            f"{len(mismatched)} mismatched gold answers, first at id {mismatched[0]}"
        )
    return shared


def _normalize_gold(value: object) -> str:
    return "".join(str(value).split()).lower()


def prompt_golds(rows: Iterable[Mapping]) -> dict[int, str]:
    return {
        prompt_id: str(group[0].get("gold_answer"))
        for prompt_id, group in _group_rows(rows).items()
    }


def readout_specs(peer_columns: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """``B0``/``B1`` with and without the peer control, and nothing else.

    Every readout is ``B0``-prefixed and the peer block is added or withheld as a
    unit, so the two contrasts differ in exactly one term each.
    """
    peer_columns = tuple(peer_columns)
    return {
        "B0": BASE_FEATURE_NAMES,
        "B1": BASE_FEATURE_NAMES + ("rmd_tail_q20",),
        "B0_plus_peer": BASE_FEATURE_NAMES + peer_columns,
        "B1_plus_peer": BASE_FEATURE_NAMES + peer_columns + ("rmd_tail_q20",),
    }


#: ``(left, right, label)``.  Negative AURC deltas favour ``left``.
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    # Reproduction of the frozen headline inside this harness.
    ("B1", "B0", "B1_minus_B0"),
    # Pre-declared: the tail increment once empirical difficulty is controlled.
    ("B1_plus_peer", "B0_plus_peer", "B1_minus_B0_given_peer"),
    # What the control itself is worth, so the rung above can be read.
    ("B0_plus_peer", "B0", "peer_minus_B0"),
    # Exploratory: is knowing two other models' pass rates worth more than the
    # target's own tail geometry?  Descriptive -- they are not substitutes, since
    # only one of the two is available at decision time.
    ("B0_plus_peer", "B1", "peer_minus_B1"),
)

PRE_DECLARED_CONTRAST = "B1_minus_B0_given_peer"


def analyze_population(
    features: Mapping[int, Mapping],
    prompt_ids: Sequence[int],
    specs: Mapping[str, tuple[str, ...]],
    peer_columns: Sequence[str],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    outcomes = np.asarray([features[i]["outcome"] for i in prompt_ids], dtype=float)
    folds = np.asarray([features[i]["fold"] for i in prompt_ids])
    used = sorted({column for spec in specs.values() for column in spec})
    columns = {
        name: np.asarray([features[i][name] for i in prompt_ids], dtype=float)
        for name in used
    }
    predictions = {
        name: crossfit_logistic_predictions(
            np.column_stack([columns[column] for column in spec]), outcomes, folds, seed=seed
        )
        for name, spec in specs.items()
    }
    readouts = {
        name: prompt_metrics(values, outcomes) for name, values in predictions.items()
    }
    floor = oracle_aurc(outcomes)
    headroom = {name: readouts[name]["aurc"] - floor for name in readouts}
    return {
        "n_prompts": len(prompt_ids),
        "base_accuracy": float(np.mean(outcomes)),
        "oracle_aurc": floor,
        # Removable risk left under each readout, and the share of it the tail
        # feature actually removes.  The share is the number to quote when a
        # delta is small: a small delta against a large headroom is weak
        # evidence, and a small delta against no headroom is no evidence.
        "aurc_headroom": headroom,
        "headroom_fraction_removed": {
            "B1_minus_B0": _share(readouts["B0"]["aurc"] - readouts["B1"]["aurc"], headroom["B0"]),
            PRE_DECLARED_CONTRAST: _share(
                readouts["B0_plus_peer"]["aurc"] - readouts["B1_plus_peer"]["aurc"],
                headroom["B0_plus_peer"],
            ),
        },
        "n_missing_peer": {
            column: int((~np.isfinite(columns[column])).sum()) for column in peer_columns
        },
        "marginal_auroc": {
            name: bootstrap_auroc(
                columns[name], outcomes, n_bootstrap=n_bootstrap, seed=seed
            )
            for name in (*peer_columns, "vote_agreement", "rmd_tail_q20")
        },
        # How close the control sits to the thing it is controlling for, and to
        # the features it is meant to be sharper than.
        "peer_association": {
            f"{column}_vs_{other}": {
                "pearson": _pearson(columns[column], other_values),
                "spearman": spearman(columns[column], other_values),
            }
            for column in peer_columns
            for other, other_values in (
                ("outcome", outcomes),
                ("length", columns["length"]),
                ("vote_agreement", columns["vote_agreement"]),
                ("rmd_tail_q20", columns["rmd_tail_q20"]),
            )
        },
        "readouts": readouts,
        "paired_deltas": {
            f"{label}_{metric}": paired_bootstrap_delta(
                predictions[left],
                predictions[right],
                outcomes,
                metric=metric,
                n_bootstrap=n_bootstrap,
                seed=_delta_seed(seed, label, metric),
            )
            for left, right, label in CONTRASTS
            for metric in METRICS
        },
    }


def analyze_model(
    label: str,
    rows: Sequence[Mapping],
    layer: int,
    data_dir: str | Path,
    peer_rates: Mapping[str, Mapping[int, float]],
    *,
    populations: Sequence[str] = ("cap_free_valid_plurality", "cap_free_all_eight_parseable"),
    expected_traces: int = 8,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    features = aggregate_prompt_features(
        rows, data_dir=str(data_dir), expected_traces=expected_traces
    )
    peer_columns = tuple(PEER_PREFIX + peer for peer in sorted(peer_rates))
    for prompt_id, entry in features.items():
        for peer in sorted(peer_rates):
            entry[PEER_PREFIX + peer] = float(
                peer_rates[peer].get(prompt_id, float("nan"))
            )
    specs = readout_specs(peer_columns)
    available = _populations(features)
    body = {
        "label": label,
        "layer": layer,
        "peers": sorted(peer_rates),
        "peer_columns": list(peer_columns),
        "readout_specs": {name: list(spec) for name, spec in specs.items()},
        "populations": {},
    }
    for population in populations:
        prompt_ids = [
            i for i in available[population] if features[i]["fold"] is not None
        ]
        if len(prompt_ids) < 2:
            continue
        body["populations"][population] = analyze_population(
            features, prompt_ids, specs, peer_columns,
            n_bootstrap=n_bootstrap, seed=seed,
        )
    return body


def _delta(body: Mapping, population: str, contrast: str, metric: str) -> Mapping | None:
    entry = (
        body["populations"]
        .get(population, {})
        .get("paired_deltas", {})
        .get(f"{contrast}_{metric}")
    )
    return entry if entry and entry.get("point_estimate") is not None else None


def stop_rule_verdict(
    results: Sequence[Mapping], population: str, *, metric: str = "aurc"
) -> dict:
    """The pre-declared rule, evaluated mechanically on one population."""
    overlapping = [
        body["label"]
        for body in results
        if (delta := _delta(body, population, PRE_DECLARED_CONTRAST, metric))
        and delta["ci_low"] <= 0.0 <= delta["ci_high"]
    ]
    return {
        "rule": "report the increment as substantially a difficulty proxy if two "
        "or more models have an interval overlapping zero",
        "contrast": PRE_DECLARED_CONTRAST,
        "metric": metric,
        "population": population,
        "models_with_interval_overlapping_zero": overlapping,
        "n_models": len(results),
        "triggered": len(overlapping) >= 2,
    }


def near_oracle_flags(
    results: Sequence[Mapping], population: str
) -> dict[str, dict[str, float]]:
    """Peer columns whose rank correlation with the outcome exceeds the threshold.

    Reported so a surviving increment is not oversold: an increment that survives
    a control this strong says more than one that survives a weak control, and an
    increment that dies against a near-oracle says less than one that dies against
    a realistic one.
    """
    flagged: dict[str, dict[str, float]] = {}
    for body in results:
        association = body["populations"].get(population, {}).get("peer_association", {})
        for column in body["peer_columns"]:
            value = association.get(f"{column}_vs_outcome", {}).get("spearman")
            if value is not None and abs(value) >= NEAR_ORACLE_SPEARMAN:
                flagged.setdefault(body["label"], {})[column] = float(value)
    return flagged


def holm_adjusted(
    results: Sequence[Mapping], population: str, *, metric: str = "aurc"
) -> dict:
    """Holm-Bonferroni over the pre-declared family: one contrast, three models.

    The other three contrasts are exploratory or harness checks and are not in the
    family; folding them in would let them inflate the threshold the confirmatory
    test has to clear.
    """
    tests = [
        (body["label"], delta["p_two_sided"])
        for body in results
        if (delta := _delta(body, population, PRE_DECLARED_CONTRAST, metric)) is not None
    ]
    order = sorted(range(len(tests)), key=lambda index: tests[index][1])
    adjusted: dict[str, dict] = {}
    running = 0.0
    for rank, index in enumerate(order):
        name, p_value = tests[index]
        running = max(running, min(1.0, p_value * (len(tests) - rank)))
        adjusted[name] = {
            "p_raw": p_value,
            "p_holm": running,
            "threshold_at_0.05": 0.05 / (len(tests) - rank),
            "significant_at_0.05": running <= 0.05,
        }
    return {"family_size": len(tests), "contrast": PRE_DECLARED_CONTRAST, "tests": adjusted}


def _fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def _band(interval) -> str:
    if interval is None or interval.get("ci_low") is None:
        return "n/a"
    return (
        f"{interval['point_estimate']:.3f} "
        f"[{interval['ci_low']:.3f}, {interval['ci_high']:.3f}]"
    )


def _signed(entry) -> str:
    if entry is None or entry.get("point_estimate") is None:
        return "n/a"
    return (
        f"{entry['point_estimate']:+.4f} "
        f"[{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}] p={entry['p_two_sided']:.3f}"
    )


def write_report(results: Sequence[Mapping], path: str | Path, *, headline: str) -> None:
    lines = [
        "# Experiment 2 -- the cross-model empirical difficulty control",
        "",
        "For each target model, the other two models' eight-sibling pass rates on the",
        "same prompt ids enter `B0` as two features. The question is whether",
        "`rmd_tail_q20` still adds once empirical problem difficulty is controlled by a",
        "signal the target model did not produce.",
        "",
        "AURC, **lower is better**; a negative delta favours the left-hand readout.",
        "Readouts, folds, populations and the paired prompt bootstrap are the frozen",
        "ones; the bootstrap runs over fixed OOF predictions and does not propagate",
        "reference refitting. Prompt-id alignment is asserted from stored gold answers,",
        "not assumed.",
        "",
        "`B0 + peer` is a **control, not a baseline**: two other models' pass rates are",
        "not available at decision time, so no method here competes with the headline.",
        "",
    ]
    for population in results[0]["populations"]:
        marker = " (headline)" if population == headline else " (sensitivity)"
        lines += [
            f"## Population: `{population}`{marker}",
            "",
            "| model | layer | peers | n | base acc |",
            "|---|---:|---|---:|---:|",
        ]
        for body in results:
            entry = body["populations"].get(population)
            if entry is None:
                continue
            lines.append(
                f"| {body['label']} | {body['layer']} | {', '.join(body['peers'])} | "
                f"{entry['n_prompts']} | {_fmt(entry['base_accuracy'])} |"
            )

        lines += [
            "",
            "### How strong is the control?",
            "",
            "Marginal AUROC of each peer pass rate against the target's own outcome,",
            "with the target's own `vote_agreement` and `rmd_tail_q20` for scale.",
            "",
            "| model | peer 1 | peer 2 | vote_agreement | rmd_tail_q20 |",
            "|---|---|---|---|---|",
        ]
        for body in results:
            entry = body["populations"].get(population)
            if entry is None:
                continue
            marginal = entry["marginal_auroc"]
            cells = [_band(marginal[column]) for column in body["peer_columns"]]
            lines.append(
                f"| {body['label']} | " + " | ".join(cells) + " | "
                f"{_band(marginal['vote_agreement'])} | {_band(marginal['rmd_tail_q20'])} |"
            )

        lines += [
            "",
            "### What the control is correlated with (Spearman)",
            "",
            "| model | peer column | vs outcome | vs length | vs vote | vs rmd_tail |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for body in results:
            entry = body["populations"].get(population)
            if entry is None:
                continue
            for column in body["peer_columns"]:
                association = entry["peer_association"]
                lines.append(
                    f"| {body['label']} | `{column}` | "
                    + " | ".join(
                        _fmt(association[f"{column}_vs_{other}"]["spearman"])
                        for other in ("outcome", "length", "vote_agreement", "rmd_tail_q20")
                    )
                    + " |"
                )

        lines += [
            "",
            "### Readout AURC, and how much risk was still removable",
            "",
            "`oracle` is the AURC of a ranker that puts every correct prompt ahead of",
            "every wrong one: AURC does not bottom out at zero, and the floor rises as",
            "base accuracy falls. `headroom` is `B0+peer` minus that floor -- the risk",
            "still available for the tail feature to remove. `share` is the fraction of",
            "that headroom the tail actually removes. When a delta is small, the share",
            "is the number to read: a small delta against a large headroom is weak",
            "evidence, a small delta against no headroom is no evidence at all.",
            "",
            "| model | oracle | B0 | B1 | B0+peer | B1+peer | headroom at B0+peer | share removed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for body in results:
            entry = body["populations"].get(population)
            if entry is None:
                continue
            lines.append(
                f"| {body['label']} | {_fmt(entry['oracle_aurc'], 4)} | "
                + " | ".join(
                    _fmt(entry["readouts"][name]["aurc"], 4)
                    for name in ("B0", "B1", "B0_plus_peer", "B1_plus_peer")
                )
                + f" | {_fmt(entry['aurc_headroom']['B0_plus_peer'], 4)} | "
                + f"{_fmt(100 * entry['headroom_fraction_removed'][PRE_DECLARED_CONTRAST], 0)}% |"
            )

        for metric in METRICS:
            lines += [
                "",
                f"### Paired deltas, {metric.upper()}"
                + (" (pre-declared metric)" if metric == "aurc" else " (secondary)"),
                "",
                "| model | B1 - B0 (reproduction) | **B1 - B0 given peer** | "
                "peer over B0 | peer over B1 |",
                "|---|---|---|---|---|",
            ]
            for body in results:
                entry = body["populations"].get(population)
                if entry is None:
                    continue
                deltas = entry["paired_deltas"]
                lines.append(
                    f"| {body['label']} | {_signed(deltas[f'B1_minus_B0_{metric}'])} | "
                    f"**{_signed(deltas[f'B1_minus_B0_given_peer_{metric}'])}** | "
                    f"{_signed(deltas[f'peer_minus_B0_{metric}'])} | "
                    f"{_signed(deltas[f'peer_minus_B1_{metric}'])} |"
                )
        lines.append("")

    verdict = stop_rule_verdict(results, headline)
    names = ", ".join(verdict["models_with_interval_overlapping_zero"]) or "none"
    lines += [
        "## Pre-declared rule",
        "",
        f"Evaluated on `{headline}`, metric `{verdict['metric']}`, contrast "
        f"`{verdict['contrast']}`.",
        "",
        f"{verdict['rule'].capitalize()}. Overlapping: {names} "
        f"({len(verdict['models_with_interval_overlapping_zero'])}/{verdict['n_models']}). "
        f"Triggered: **{'YES' if verdict['triggered'] else 'no'}**.",
        "",
    ]

    flagged = near_oracle_flags(results, headline)
    if flagged:
        detail = "; ".join(
            f"{label} ({', '.join(f'{c.split(PEER_PREFIX)[-1]} {v:+.2f}' for c, v in columns.items())})"
            for label, columns in flagged.items()
        )
        lines += [
            f"**Near-oracle note** (pre-declared). A peer rate reaches |Spearman| >= "
            f"{NEAR_ORACLE_SPEARMAN:.2f} against the target's own outcome on: {detail}.",
            "A control this strong makes a surviving increment mean more and a dying",
            "increment mean less -- a near-oracle can saturate the readout on its own,",
            "which is a third reading, distinct from 'geometry is redundant with",
            "difficulty'.",
            "",
            "The flag fires on every model, so on its own it does not separate them.",
            "The headroom column does, and it is the statistic that should have been",
            "pre-declared in its place:",
            "",
            "| model | headroom at `B0+peer` | delta | share of headroom removed |",
            "|---|---:|---:|---:|",
        ]
        for body in results:
            # `headline`, never the population loop variable above: this table sits
            # under the verdict and must report the population the verdict was
            # evaluated on.
            entry = body["populations"].get(headline)
            if entry is None:
                continue
            delta = _delta(body, headline, PRE_DECLARED_CONTRAST, "aurc")
            lines.append(
                f"| {body['label']} | {_fmt(entry['aurc_headroom']['B0_plus_peer'], 4)} | "
                f"{_fmt(delta['point_estimate'], 4) if delta else 'n/a'} | "
                f"{_fmt(100 * entry['headroom_fraction_removed'][PRE_DECLARED_CONTRAST], 0)}% |"
            )
        lines.append("")

    holm = holm_adjusted(results, headline)
    lines += [
        "### Multiplicity over the pre-declared family",
        "",
        f"Holm-Bonferroni over `{holm['contrast']}` across {holm['family_size']} models.",
        "The three other contrasts per model are harness checks or exploratory and are",
        "not in the family.",
        "",
        "| model | raw p | Holm p | significant at 0.05 |",
        "|---|---:|---:|---|",
    ]
    for name, entry in sorted(holm["tests"].items(), key=lambda item: item[1]["p_raw"]):
        lines.append(
            f"| {name} | {entry['p_raw']:.3f} | {entry['p_holm']:.3f} | "
            f"{'yes' if entry['significant_at_0.05'] else 'no'} |"
        )
    n_valid = results[0]["populations"][headline]["paired_deltas"]["B1_minus_B0_aurc"][
        "n_valid"
    ]
    lines += [
        "",
        f"The bootstrap resolves p to 1/{n_valid}, so a Holm p within a few thousandths",
        "of its threshold is borderline rather than a clean pass.",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL:OOF_CSV:DATA_DIR",
        help="repeatable; colon-separated triple. At least two are needed, since "
        "each model's control is built from the others.",
    )
    parser.add_argument("--population", action="append", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--expected_traces", type=int, default=8)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/peer_difficulty_control")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    populations = tuple(
        args.population or ("cap_free_valid_plurality", "cap_free_all_eight_parseable")
    )
    specs = [spec.split(":", 2) for spec in args.model]
    if len(specs) < 2:
        raise SystemExit("need at least two models: the control is built from the peers")

    loaded = {}
    for label, oof_csv, data_dir in specs:
        rows, layer = select_layer_rows(_read_oof(oof_csv), args.layer, context=str(oof_csv))
        loaded[label] = {"rows": rows, "layer": layer, "data_dir": data_dir}
    shared = assert_shared_prompt_ids(
        {label: prompt_golds(body["rows"]) for label, body in loaded.items()}
    )
    rates = {label: peer_pass_rates(body["rows"]) for label, body in loaded.items()}

    results = [
        analyze_model(
            label,
            loaded[label]["rows"],
            loaded[label]["layer"],
            loaded[label]["data_dir"],
            {peer: rates[peer] for peer in loaded if peer != label},
            populations=populations,
            expected_traces=args.expected_traces,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        for label, _, _ in specs
    ]

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "control": (
            "peer_pass_rate__<label>: fraction of that model's eight cached siblings "
            "with is_correct == 1 on the same prompt id; denominator is all cached "
            "siblings, so an unparseable trace counts as not solved"
        ),
        "metric": "aurc primary (lower is better; negative delta favours the left "
        "readout); auacc secondary",
        "n_shared_prompt_ids": len(shared),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "stop_rule": stop_rule_verdict(results, populations[0]),
        "near_oracle": near_oracle_flags(results, populations[0]),
        "holm": holm_adjusted(results, populations[0]),
        "models": results,
    }
    (output / "peer_difficulty_control_results.json").write_text(json.dumps(payload, indent=2))
    write_report(
        results, output / "peer_difficulty_control_report.md", headline=populations[0]
    )
    print(f"wrote {output}/peer_difficulty_control_report.md")


if __name__ == "__main__":
    main()
