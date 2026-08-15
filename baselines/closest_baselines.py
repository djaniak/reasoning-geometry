"""The two closest cheap baselines the B1 - B0 increment has never been run against.

Both contrasts were recommendation #1 and #2 of the 2026-08-09 direction review and
both were dropped without comment.  Neither needs a model call: every column they
require is already in the frozen prompt-decomposition OOF tables.

**1a -- answer-cluster entropy.**  ``B0`` carries one self-consistency scalar,
``vote_agreement``, the share of parseable siblings voting for the winner.  It cannot
tell ``5+3`` from ``5+1+1+1``: both are 0.625.  Discrete entropy of the answer
histogram is the direct generalization, and it is the honest comparator for a claim
phrased as "beyond self-consistency".  If it absorbs the geometry increment, the
result is not wrong but it means something else -- geometry recovers
answer-distribution shape the baseline omitted, rather than adding beyond
self-consistency broadly construed.

**1b -- whole-trace mean RMD versus the tail.**  ``rmd_tail_q20`` is the mean of
per-token RMD over the final 20% of tokens; ``rmd`` is the same mean over the whole
trace.  The whole-trace version is Vazhentsev et al.'s ATRMD (arXiv:2502.14427), the
nearest published construction, and it has never been scored against the headline in
a matched readout.  If it ties, the tail restriction contributes nothing and the
tail-aggregator novelty leg is gone on the data as well as in the documentation.

Pre-declared stop rules, written before the run:

* **1a** -- if ``B0 + H + rmd_tail`` minus ``B0 + H`` has an AURC interval
  overlapping zero on two or more of the three models, the "beyond self-consistency"
  framing stops and the project reframes before spending anything else.
* **1b** -- no region or percentile sweep follows this contrast, whichever way it
  lands.  One matched comparison, then the description gets fixed.

Sign convention throughout: **AURC, lower is better**, so a negative delta favours
the left-hand readout.

Not a DVC stage: it re-reads cached OOF rows and imports the frozen aggregation,
folds, populations, readout and bootstrap from ``incremental_abstention`` rather
than restating any of them.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from baselines.deepconf_asymmetry import _pearson, bootstrap_auroc
from applications.incremental_abstention import (
    BASE_FEATURE_NAMES,
    _group_rows,
    _mean_field,
    _population_ids,
    _read_oof,
    aggregate_prompt_features,
    crossfit_logistic_predictions,
    is_parseable_answer,
    paired_bootstrap_delta,
    prompt_metrics,
    select_layer_rows,
)
from controls.orgad_agreement_control import spearman

#: Both added columns are stored negated, the way every other score in the frozen
#: aggregation is, so "higher is better" holds across the whole design matrix.  The
#: logistic readout is exactly invariant to the flip; the convention is for readers.
EXTRA_FEATURE_NAMES = ("neg_answer_entropy", "rmd_full")

FEATURE_DEFINITIONS = {
    "neg_answer_entropy": (
        "-Shannon entropy (nats) of the normalized exact-answer histogram over the "
        "parseable siblings; NaN when no sibling parses"
    ),
    "rmd_full": (
        "sibling mean of -mean(relative Mahalanobis token distance) over the whole "
        "trace -- Vazhentsev et al.'s ATRMD, the untailed counterpart of rmd_tail_q20"
    ),
}


def answer_entropy(rows: Iterable[Mapping]) -> float:
    """Shannon entropy in nats of the parseable-answer histogram for one prompt.

    The denominator is the parseable siblings, matching ``vote_agreement``: this is
    meant to be the strictly more informative version of that same scalar, not a
    different population.  A prompt with nothing parseable has no histogram, and
    returns NaN rather than the zero that a degenerate one-cluster prompt earns --
    those two states are opposites and collapsing them would hand the readout a
    fake unanimity signal.
    """
    answers = [
        str(row["predicted_answer"])
        for row in rows
        if is_parseable_answer(row.get("predicted_answer"))
    ]
    if not answers:
        return float("nan")
    shares = np.asarray(list(Counter(answers).values()), dtype=float) / len(answers)
    return float(-np.sum(shares * np.log(shares)))


def extra_prompt_columns(rows: Iterable[Mapping]) -> dict[int, dict[str, float]]:
    """The two comparator features, aggregated per prompt exactly like ``B0`` is."""
    return {
        prompt_id: {
            "neg_answer_entropy": -answer_entropy(group),
            "rmd_full": _mean_field(group, "rmd_score"),
        }
        for prompt_id, group in sorted(_group_rows(rows).items())
    }


#: 1a and 1b share one design matrix so the two contrasts are scored on identical
#: prompts, folds and readouts.  ``B1`` is carried in both to reproduce the frozen
#: increment inside this harness before anything is read off the new rungs.
READOUT_SPECS: dict[str, tuple[str, ...]] = {
    "B0": BASE_FEATURE_NAMES,
    "B1": BASE_FEATURE_NAMES + ("rmd_tail_q20",),
    # 1a
    "B0_plus_H": BASE_FEATURE_NAMES + ("neg_answer_entropy",),
    "B0_plus_H_plus_rmd_tail": BASE_FEATURE_NAMES
    + ("neg_answer_entropy", "rmd_tail_q20"),
    # 1b
    "B0_plus_rmd_full": BASE_FEATURE_NAMES + ("rmd_full",),
    "B0_plus_both_rmd": BASE_FEATURE_NAMES + ("rmd_full", "rmd_tail_q20"),
}

#: ``(left, right, label)``.  Negative AURC deltas favour ``left``.
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    # Reproduction of the frozen headline inside this harness.
    ("B1", "B0", "B1_minus_B0"),
    # 1a: what the answer histogram adds, and whether geometry survives it.
    ("B0_plus_H", "B0", "H_over_B0"),
    ("B0_plus_H_plus_rmd_tail", "B0_plus_H", "rmd_tail_over_B0_plus_H"),
    ("B0_plus_H_plus_rmd_tail", "B1", "H_over_B1"),
    # 1b: the tail against the whole trace, both directions.
    ("B0_plus_rmd_full", "B0", "rmd_full_over_B0"),
    ("B0_plus_both_rmd", "B0_plus_rmd_full", "rmd_tail_over_rmd_full"),
    ("B0_plus_both_rmd", "B1", "rmd_full_over_rmd_tail"),
)

#: 1a's stop rule is a threshold over models; 1b's is a no-sweep rule with a two-way
#: branch and no trigger, so the two are reported as different objects rather than
#: forced into one "triggered" column.
STOP_RULE_1A_CONTRAST = "rmd_tail_over_B0_plus_H"
TAIL_VERSUS_FULL_CONTRAST = "rmd_tail_over_rmd_full"


#: Minimum prompts of *each* class before a stratified delta is reported.  The
#: readouts here carry six or seven features and are cross-fitted over five folds,
#: so a stratum with fewer than this trains each fold on ~20 minority prompts.  Below
#: it the interval cannot distinguish "the tail does nothing here" from "this stratum
#: has no incorrect prompts to rank", and those are the two answers the whole test is
#: trying to tell apart.
MIN_STRATUM_CLASS = 25


def tail_window_sizes(rows: Iterable[Mapping]) -> dict[int, float]:
    """Sibling-mean size of the ``tail_q20`` window, in tokens.

    ``rmd_tail_q20`` averages over ``ceil(0.20 * n_tokens)`` trailing tokens, so the
    window is a per-trace quantity and "the final 20%" is a different statistic at
    different trace lengths.  This is the covariate the model-family reading of 1b
    is confounded with: distillation, reasoning training and trace length all move
    together across the three models, and only length varies *within* one.
    """
    return {
        prompt_id: float(
            np.mean([np.ceil(0.20 * float(row["trace_length"])) for row in group])
        )
        for prompt_id, group in sorted(_group_rows(rows).items())
    }


def window_strata(
    windows: np.ndarray, *, absolute_threshold: float | None = None
) -> dict[str, np.ndarray]:
    """Within-model window terciles, plus an optional absolute-threshold stratum.

    The terciles are the load-bearing construction: they ask whether the tail's
    advantage decays with window size *inside a single model*, which no cross-model
    comparison can answer because distillation and length are collinear between
    models.  The absolute stratum exists to put a second model's short prompts on
    the same token scale as the reference model's whole population.
    """
    windows = np.asarray(windows, dtype=float)
    low, high = np.percentile(windows, [100 / 3, 200 / 3])
    middle = np.median(windows)
    strata = {
        "window_short": windows <= low,
        "window_mid": (windows > low) & (windows <= high),
        "window_long": windows > high,
        # The same contrast at half the resolution.  Added after the terciles ran,
        # because Qwen's short tercile landed one prompt under MIN_STRATUM_CLASS and
        # that is the single stratum the window hypothesis most wants to see.  Moving
        # the threshold to reach it would be the post-hoc move; a coarser cut that
        # both halves clear on their own is not, and it is reported alongside rather
        # than instead of the terciles.
        "window_below_median": windows <= middle,
        "window_above_median": windows > middle,
    }
    if absolute_threshold is not None:
        strata[f"window_le_{absolute_threshold:.0f}"] = windows <= absolute_threshold
    return strata


def analyze_window_strata(
    features: Mapping[int, Mapping],
    prompt_ids: list[int],
    windows: Mapping[int, float],
    *,
    absolute_threshold: float | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """The 1b contrast re-scored inside fixed trace-length strata.

    This is not a region sweep -- no new RMD region is opened and ``rmd_tail_q20``
    keeps its frozen definition.  What varies is the population, so that the token
    scale the tail window lands on can be read off against a fixed statistic.
    """
    prompt_ids = list(prompt_ids)
    window_values = np.asarray([windows[i] for i in prompt_ids], dtype=float)
    outcomes = np.asarray([features[i]["outcome"] for i in prompt_ids], dtype=float)
    folds = np.asarray([features[i]["fold"] for i in prompt_ids])
    columns = {
        name: np.asarray([features[i][name] for i in prompt_ids], dtype=float)
        for name in BASE_FEATURE_NAMES + ("rmd_tail_q20",) + EXTRA_FEATURE_NAMES
    }
    results = {}
    for name, mask in window_strata(
        window_values, absolute_threshold=absolute_threshold
    ).items():
        n_correct = int((outcomes[mask] > 0.5).sum())
        n_wrong = int((outcomes[mask] <= 0.5).sum())
        body = {
            "n_prompts": int(mask.sum()),
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "base_accuracy": float(np.mean(outcomes[mask])) if mask.any() else float("nan"),
            "window_median": float(np.median(window_values[mask])) if mask.any() else float("nan"),
            "window_min": float(window_values[mask].min()) if mask.any() else float("nan"),
            "window_max": float(window_values[mask].max()) if mask.any() else float("nan"),
        }
        if min(n_correct, n_wrong) < MIN_STRATUM_CLASS:
            results[name] = {**body, "reported": False, "paired_deltas_aurc": {}}
            continue
        predictions = {
            spec_name: crossfit_logistic_predictions(
                np.column_stack([columns[column][mask] for column in spec]),
                outcomes[mask],
                folds[mask],
                seed=seed,
            )
            for spec_name, spec in READOUT_SPECS.items()
        }
        results[name] = {
            **body,
            "reported": True,
            "paired_deltas_aurc": {
                label: paired_bootstrap_delta(
                    predictions[left], predictions[right], outcomes[mask],
                    metric="aurc", n_bootstrap=n_bootstrap, seed=seed,
                )
                for left, right, label in CONTRASTS
            },
        }
    return results


def _populations(features: Mapping[int, Mapping]) -> dict[str, list[int]]:
    """Frozen populations plus the cap-free intersection of the strict one.

    ``all_eight_parseable`` as the frozen module defines it is not cap-filtered, so
    on its own it is not a sensitivity analysis of the headline population -- it
    trades one selection for another.  The intersection is.
    """
    populations = _population_ids(features)
    cap_free = set(populations["cap_free_full_population"])
    populations["cap_free_all_eight_parseable"] = [
        prompt_id for prompt_id in populations["all_eight_parseable"] if prompt_id in cap_free
    ]
    return populations


def analyze_population(
    features: Mapping[int, Mapping],
    prompt_ids: list[int],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    outcomes = np.asarray([features[i]["outcome"] for i in prompt_ids], dtype=float)
    folds = np.asarray([features[i]["fold"] for i in prompt_ids])
    columns = {
        name: np.asarray([features[i][name] for i in prompt_ids], dtype=float)
        for name in BASE_FEATURE_NAMES + ("rmd_tail_q20",) + EXTRA_FEATURE_NAMES
    }
    predictions = {
        name: crossfit_logistic_predictions(
            np.column_stack([columns[column] for column in spec]),
            outcomes,
            folds,
            seed=seed,
        )
        for name, spec in READOUT_SPECS.items()
    }
    return {
        "n_prompts": len(prompt_ids),
        "base_accuracy": float(np.mean(outcomes)),
        "n_missing_answer_entropy": int(
            (~np.isfinite(columns["neg_answer_entropy"])).sum()
        ),
        "marginal_auroc": {
            name: bootstrap_auroc(
                columns[name], outcomes, n_bootstrap=n_bootstrap, seed=seed
            )
            for name in ("vote_agreement", "neg_answer_entropy", "rmd_tail_q20", "rmd_full")
        },
        "redundancy": {
            f"{left}_vs_{right}": {
                "pearson": _pearson(columns[left], columns[right]),
                "spearman": spearman(columns[left], columns[right]),
            }
            for left, right in (
                ("neg_answer_entropy", "vote_agreement"),
                ("neg_answer_entropy", "rmd_tail_q20"),
                ("rmd_full", "rmd_tail_q20"),
            )
        },
        "readouts": {
            name: prompt_metrics(values, outcomes) for name, values in predictions.items()
        },
        "paired_deltas_aurc": {
            label: paired_bootstrap_delta(
                predictions[left],
                predictions[right],
                outcomes,
                metric="aurc",
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            for left, right, label in CONTRASTS
        },
    }


def analyze_model(
    label: str,
    oof_csv: str | Path,
    data_dir: str | Path,
    *,
    populations: Iterable[str] = ("cap_free_valid_plurality", "cap_free_all_eight_parseable"),
    layer: int | None = None,
    expected_traces: int = 8,
    n_bootstrap: int = 1000,
    seed: int = 42,
    window_threshold: float | None = None,
) -> dict:
    rows, layer = select_layer_rows(_read_oof(oof_csv), layer, context=str(oof_csv))
    features = aggregate_prompt_features(
        rows, data_dir=str(data_dir), expected_traces=expected_traces
    )
    for prompt_id, extra in extra_prompt_columns(rows).items():
        features[prompt_id].update(extra)
    windows = tail_window_sizes(rows)
    available = _populations(features)
    body = {"label": label, "layer": layer, "populations": {}, "window_strata": {}}
    for population in populations:
        prompt_ids = [
            # An unfolded prompt has no held-out readout and the frozen analysis
            # drops it before scoring; keeping it would change the population.
            i for i in available[population] if features[i]["fold"] is not None
        ]
        if len(prompt_ids) < 2:
            continue
        body["populations"][population] = analyze_population(
            features, prompt_ids, n_bootstrap=n_bootstrap, seed=seed
        )
    headline = [
        i for i in available[tuple(populations)[0]] if features[i]["fold"] is not None
    ]
    body["window_summary"] = {
        "median": float(np.median([windows[i] for i in headline])),
        "min": float(min(windows[i] for i in headline)),
        "max": float(max(windows[i] for i in headline)),
    }
    body["window_strata"] = analyze_window_strata(
        features, headline, windows,
        absolute_threshold=window_threshold, n_bootstrap=n_bootstrap, seed=seed,
    )
    return body


def _delta(body: Mapping, population: str, contrast: str) -> Mapping | None:
    delta = (
        body["populations"]
        .get(population, {})
        .get("paired_deltas_aurc", {})
        .get(contrast)
    )
    return delta if delta and delta["point_estimate"] is not None else None


def stop_rule_verdicts(results: list[dict], population: str) -> dict[str, dict]:
    """Mechanical evaluation of both pre-declared rules on one population.

    1a is a genuine stop rule: two or more models whose interval covers zero and the
    "beyond self-consistency" framing stops.  1b has no trigger -- it forbids a
    follow-up sweep either way -- so what it needs is the per-model branch, "tail
    wins" versus "they tie or the whole trace wins", which is what decides whether
    the tail-aggregator novelty leg survives on the data as well as in the docs.
    """
    overlapping = [
        body["label"]
        for body in results
        if (delta := _delta(body, population, STOP_RULE_1A_CONTRAST))
        and delta["ci_low"] <= 0.0 <= delta["ci_high"]
    ]
    branches = {}
    for body in results:
        delta = _delta(body, population, TAIL_VERSUS_FULL_CONTRAST)
        if delta is None:
            continue
        branches[body["label"]] = (
            "tail_wins" if delta["ci_high"] < 0.0 else "tie_or_full_wins"
        )
    return {
        "1a": {
            "rule": "stop the 'beyond self-consistency' claim if two or more models "
            "have an interval overlapping zero",
            "contrast": STOP_RULE_1A_CONTRAST,
            "models_with_interval_overlapping_zero": overlapping,
            "n_models": len(results),
            "triggered": len(overlapping) >= 2,
        },
        "1b": {
            "rule": "no region or percentile sweep follows, whichever way this lands",
            "contrast": TAIL_VERSUS_FULL_CONTRAST,
            "branch_by_model": branches,
            "n_tail_wins": sum(value == "tail_wins" for value in branches.values()),
            "n_models": len(results),
        },
    }


def holm_adjusted(results: list[dict], population: str) -> dict:
    """Holm-Bonferroni over the pre-declared family only.

    The family is the two pre-declared contrasts across the three models -- six
    tests.  The other five contrasts per model were not pre-declared and are not
    corrected here; folding them in would let exploratory comparisons inflate the
    threshold the confirmatory ones have to clear, which is the wrong direction to
    be generous in.
    """
    tests = [
        (f"{body['label']}:{contrast}", delta["p_two_sided"])
        for contrast in (STOP_RULE_1A_CONTRAST, TAIL_VERSUS_FULL_CONTRAST)
        for body in results
        if (delta := _delta(body, population, contrast)) is not None
    ]
    order = sorted(range(len(tests)), key=lambda index: tests[index][1])
    adjusted: dict[str, dict] = {}
    running = 0.0
    for rank, index in enumerate(order):
        name, p_value = tests[index]
        # Holm's step-down threshold, and the monotone running maximum that keeps
        # adjusted p-values non-decreasing down the sorted list.
        running = max(running, min(1.0, p_value * (len(tests) - rank)))
        adjusted[name] = {
            "p_raw": p_value,
            "p_holm": running,
            "threshold_at_0.05": 0.05 / (len(tests) - rank),
            "significant_at_0.05": running <= 0.05,
        }
    return {"family_size": len(tests), "tests": adjusted}


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


def _fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_report(results: list[dict], path: str | Path, *, headline: str) -> None:
    lines = [
        "# The two closest cheap baselines (experiments 1a and 1b)",
        "",
        "AURC, **lower is better**; a negative delta favours the left-hand readout.",
        "Every readout is the frozen cross-fitted logistic on the frozen prompt folds,",
        "and every interval is the frozen paired prompt bootstrap over fixed OOF",
        "predictions -- it does not propagate reference refitting.",
        "",
        "`H` is `neg_answer_entropy`: minus the Shannon entropy (nats) of the",
        "normalized exact-answer histogram over parseable siblings. `rmd_full` is the",
        "whole-trace mean of per-token RMD (Vazhentsev ATRMD); `rmd_tail_q20` is the",
        "same mean restricted to the final 20% of tokens.",
        "",
    ]
    for population in results[0]["populations"]:
        marker = " (headline)" if population == headline else " (sensitivity)"
        lines += [f"## Population: `{population}`{marker}", ""]
        lines += [
            "| model | layer | n | base acc | prompts with no parseable answer |",
            "|---|---:|---:|---:|---:|",
        ]
        for body in results:
            entry = body["populations"].get(population)
            if entry is None:
                continue
            lines.append(
                f"| {body['label']} | {body['layer']} | {entry['n_prompts']} | "
                f"{_fmt(entry['base_accuracy'])} | {entry['n_missing_answer_entropy']} |"
            )

        lines += [
            "",
            "### Marginal AUROC of each single feature",
            "",
            "| model | vote_agreement | H | rmd_tail_q20 | rmd_full |",
            "|---|---|---|---|---|",
        ]
        for body in results:
            entry = body["populations"].get(population)
            if entry is None:
                continue
            marginal = entry["marginal_auroc"]
            lines.append(
                f"| {body['label']} | {_band(marginal['vote_agreement'])} | "
                f"{_band(marginal['neg_answer_entropy'])} | "
                f"{_band(marginal['rmd_tail_q20'])} | {_band(marginal['rmd_full'])} |"
            )

        lines += [
            "",
            "### Redundancy (Pearson / Spearman)",
            "",
            "| model | H vs vote | H vs rmd_tail | rmd_full vs rmd_tail |",
            "|---|---|---|---|",
        ]
        for body in results:
            entry = body["populations"].get(population)
            if entry is None:
                continue
            redundancy = entry["redundancy"]
            lines.append(
                f"| {body['label']} | "
                + " | ".join(
                    f"{_fmt(redundancy[key]['pearson'])} / {_fmt(redundancy[key]['spearman'])}"
                    for key in (
                        "neg_answer_entropy_vs_vote_agreement",
                        "neg_answer_entropy_vs_rmd_tail_q20",
                        "rmd_full_vs_rmd_tail_q20",
                    )
                )
                + " |"
            )

        lines += ["", "### Readout AURC", "", "| model | " + " | ".join(READOUT_SPECS) + " |",
                  "|---|" + "---:|" * len(READOUT_SPECS)]
        for body in results:
            entry = body["populations"].get(population)
            if entry is None:
                continue
            lines.append(
                f"| {body['label']} | "
                + " | ".join(
                    _fmt(entry["readouts"][name]["aurc"], 4) for name in READOUT_SPECS
                )
                + " |"
            )

        lines += [
            "",
            "### 1a -- does tail RMD survive the answer histogram?",
            "",
            "| model | B1 - B0 (reproduction) | H over B0 | **rmd_tail over B0+H** | H over B1 |",
            "|---|---|---|---|---|",
        ]
        for body in results:
            entry = body["populations"].get(population)
            if entry is None:
                continue
            deltas = entry["paired_deltas_aurc"]
            lines.append(
                f"| {body['label']} | {_signed(deltas['B1_minus_B0'])} | "
                f"{_signed(deltas['H_over_B0'])} | "
                f"**{_signed(deltas['rmd_tail_over_B0_plus_H'])}** | "
                f"{_signed(deltas['H_over_B1'])} |"
            )

        lines += [
            "",
            "### 1b -- the tail against the whole trace",
            "",
            "| model | rmd_full over B0 | rmd_tail over B0 | **rmd_tail over rmd_full** | rmd_full over rmd_tail |",
            "|---|---|---|---|---|",
        ]
        for body in results:
            entry = body["populations"].get(population)
            if entry is None:
                continue
            deltas = entry["paired_deltas_aurc"]
            lines.append(
                f"| {body['label']} | {_signed(deltas['rmd_full_over_B0'])} | "
                f"{_signed(deltas['B1_minus_B0'])} | "
                f"**{_signed(deltas['rmd_tail_over_rmd_full'])}** | "
                f"{_signed(deltas['rmd_full_over_rmd_tail'])} |"
            )
        lines.append("")

    if any(body.get("window_strata") for body in results):
        lines += [
            "## 1b follow-up: is the tail restriction a window-size effect?",
            "",
            "Distillation, reasoning training and trace length are collinear *between*",
            "these three models, so no cross-model comparison can separate them. Trace",
            "length varies *within* each model, so the terciles below ask whether the",
            "tail's advantage over the whole-trace mean decays with window size inside",
            "one model, holding everything else fixed. Not a region sweep:",
            "`rmd_tail_q20` keeps its frozen definition and no new region is opened.",
            "",
            "Strata are terciles of the sibling-mean tail-window size `ceil(0.20 * "
            "trace_length)`, in tokens. A stratum is reported only with at least",
            f"{MIN_STRATUM_CLASS} prompts of each class.",
            "",
            "| model | stratum | n | wrong | base acc | window med [min, max] | **rmd_tail over rmd_full** |",
            "|---|---|---:|---:|---:|---|---|",
        ]
        for body in results:
            for name, stratum in body.get("window_strata", {}).items():
                if stratum["reported"]:
                    cell = _signed(stratum["paired_deltas_aurc"]["rmd_tail_over_rmd_full"])
                else:
                    cell = f"not reported (min class {min(stratum['n_correct'], stratum['n_wrong'])})"
                lines.append(
                    f"| {body['label']} | {name} | {stratum['n_prompts']} | "
                    f"{stratum['n_wrong']} | {_fmt(stratum['base_accuracy'])} | "
                    f"{stratum['window_median']:.0f} [{stratum['window_min']:.0f}, "
                    f"{stratum['window_max']:.0f}] | {cell} |"
                )
        lines.append("")

    verdicts = stop_rule_verdicts(results, headline)
    rule_1a, rule_1b = verdicts["1a"], verdicts["1b"]
    names = ", ".join(rule_1a["models_with_interval_overlapping_zero"]) or "none"
    lines += [
        "## Pre-declared rules",
        "",
        f"Evaluated on `{headline}`.",
        "",
        f"**1a** (`{rule_1a['contrast']}`) -- {rule_1a['rule']}. Overlapping: {names} "
        f"({len(rule_1a['models_with_interval_overlapping_zero'])}/{rule_1a['n_models']}). "
        f"Triggered: **{'YES' if rule_1a['triggered'] else 'no'}**.",
        "",
        f"**1b** (`{rule_1b['contrast']}`) -- {rule_1b['rule']}. This one has no",
        "trigger; what it decides is whether a tail-specific contribution survives.",
        "",
        "| model | branch |",
        "|---|---|",
    ]
    for label, branch in rule_1b["branch_by_model"].items():
        lines.append(f"| {label} | `{branch}` |")
    lines += [
        "",
        f"Tail wins on {rule_1b['n_tail_wins']}/{rule_1b['n_models']} models.",
        "",
        "### Multiplicity over the pre-declared family",
        "",
        "Holm-Bonferroni over the two pre-declared contrasts across three models. The",
        "five other contrasts per model were exploratory and are not in the family.",
        "",
        "| test | raw p | Holm p | significant at 0.05 |",
        "|---|---:|---:|---|",
    ]
    holm = holm_adjusted(results, headline)
    for name, entry in sorted(holm["tests"].items(), key=lambda item: item[1]["p_raw"]):
        lines.append(
            f"| `{name}` | {entry['p_raw']:.3f} | {entry['p_holm']:.3f} | "
            f"{'yes' if entry['significant_at_0.05'] else 'no'} |"
        )
    lines += [
        "",
        f"Family size {holm['family_size']}. The bootstrap resolves p to "
        f"1/{results[0]['populations'][headline]['paired_deltas_aurc']['B1_minus_B0']['n_valid']}, "
        "so any Holm p within a few thousandths of its threshold should be read as "
        "borderline rather than as a clean pass.",
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
        help="repeatable; colon-separated triple",
    )
    parser.add_argument(
        "--population",
        action="append",
        default=None,
        help="repeatable; defaults to cap_free_valid_plurality then cap_free_all_eight_parseable",
    )
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--expected_traces", type=int, default=8)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--window_threshold",
        type=float,
        default=None,
        help=(
            "Extra stratum of prompts whose mean tail window is at most this many "
            "tokens. Set it to the reference model's maximum window to put another "
            "model's short prompts on the same token scale as that model's whole "
            "population."
        ),
    )
    parser.add_argument("--output_dir", default="results/closest_baselines")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    populations = tuple(
        args.population or ("cap_free_valid_plurality", "cap_free_all_eight_parseable")
    )
    results = []
    for spec in args.model:
        label, oof_csv, data_dir = spec.split(":", 2)
        results.append(
            analyze_model(
                label,
                oof_csv,
                data_dir,
                populations=populations,
                layer=args.layer,
                expected_traces=args.expected_traces,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                window_threshold=args.window_threshold,
            )
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_definitions": FEATURE_DEFINITIONS,
        "readout_specs": {name: list(spec) for name, spec in READOUT_SPECS.items()},
        "metric": "aurc (lower is better; negative delta favours the left readout)",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "stop_rules": stop_rule_verdicts(results, populations[0]),
        "models": results,
    }
    (output / "closest_baselines_results.json").write_text(json.dumps(payload, indent=2))
    write_report(results, output / "closest_baselines_report.md", headline=populations[0])
    print(f"wrote {output}/closest_baselines_report.md")


if __name__ == "__main__":
    main()
