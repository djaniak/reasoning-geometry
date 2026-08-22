"""Does single-trace geometry predict the *gain* from buying more samples?

Step 2 of the allocation direction, and a gate on whether step 3 gets written at
all.  Everything upstream of here asks whether geometry predicts **correctness**.
Allocation needs something else: the *marginal value of another sample*, which is
a different and non-monotone target.  A prompt the model solves 8/8 gains nothing
from more samples, and so does a prompt it solves 0/8; the gain lives in the
middle.  A feature can therefore be an excellent difficulty signal and a useless
allocation signal, and the 2026-08-10 peer control makes that the *expected*
outcome rather than a remote one -- it showed most of the ``rmd_tail_q20``
increment is prompt difficulty.

The target is built exhaustively rather than estimated.  For each prompt,
``a(p, k)`` is the expected plurality-vote correctness when ``k`` of the eight
cached siblings are drawn without replacement, computed over **all** ``C(8, k)``
subsets -- 8, 28, 70 and 1 subsets for k = 1, 2, 4, 8.  That is 107 majority
votes per prompt, which is cheap and has none of the sampling noise a plug-in
estimate from the pass rate would carry.  The gain target is

    g(p) = a(p, 8) - a(p, 1)

Stage-1 features are the four that exist at ``n = 1``: ``rmd_tail_q20``,
``length``, ``entropy``, ``logprob``.  **No ``vote_agreement``** -- a single
sample has no siblings to agree with, so carrying it would smuggle the eight-
sample world into a one-sample readout.

**Which trace is stage 1 is a random variable**, so it is not fixed at
``sample_id == 0``.  The whole precheck runs eight times, once per choice, and
the gate reads the median across those draws; the range is reported next to it so
a result that only holds for one lucky draw is visible as such.

Pre-declared gate, written before the run:

* **Pass** -- geometry alone beats the cross-fitted constant predictor (R^2 > 0),
  *and* adding geometry to the output features raises out-of-fold Spearman
  (median over the eight draws of the paired difference > 0), on at least two of
  the three models.  Step 3 (``allocation.py``) is then written.
* **Fail** -- output features alone match the combined set.  Geometry contributes
  nothing to allocation, step 3 is not run, and that is the finding.

The diagnostics below are not the gate.  They exist to explain a failure, and one
of them names the specific failure this precheck was built to catch: if
single-trace geometry tracks the pass rate strongly but the gain barely at all,
then **geometry reads difficulty but not marginal gain**.

Not a DVC stage: it re-reads cached OOF rows and imports the frozen aggregation,
folds, populations and majority-vote convention rather than restating any of
them.  ``a(p, 8)`` must equal the frozen prompt outcome exactly -- C(8,8) is the
one subset containing every sibling -- and that identity is asserted, not hoped
for.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from baselines.closest_baselines import _populations
from baselines.deepconf_asymmetry import bootstrap_auroc
from applications.incremental_abstention import (
    _finite,
    _group_rows,
    _impute_and_scale,
    _plurality_outcome,
    _read_oof,
    aggregate_prompt_features,
    select_layer_rows,
)
from controls.orgad_agreement_control import spearman

#: Subset sizes the accuracy curve is evaluated at.  Only 1 and 8 enter the gain
#: target; 2 and 4 are carried so the curve's shape is on record for step 3,
#: which needs ``a(p, k)`` at the budgets it allocates between.
SUBSET_SIZES: tuple[int, ...] = (1, 2, 4, 8)

#: ``g(p) = a(p, GAIN_TO) - a(p, GAIN_FROM)``.
GAIN_FROM, GAIN_TO = 1, 8

#: The three feature sets the gate compares.  ``vote_agreement`` is deliberately
#: absent from all of them: it is an eight-sibling statistic and does not exist
#: at the point an allocator has to decide.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "geometry": ("rmd_tail_q20",),
    "output": ("length", "entropy", "logprob"),
    "both": ("length", "entropy", "logprob", "rmd_tail_q20"),
}

#: Models that must pass for the gate to open.
GATE_MIN_MODELS = 2

#: Labelling thresholds for the named failure mode.  Descriptive only -- they
#: decide what the report *calls* the pattern, never whether the gate opens.
DIFFICULTY_NOT_GAIN_STRONG = 0.20
DIFFICULTY_NOT_GAIN_NEAR_ZERO = 0.10


def subset_accuracy(group: Sequence[Mapping], k: int) -> float:
    """Expected plurality-vote correctness over all ``C(len(group), k)`` subsets.

    Exhaustive, not sampled: with eight siblings the largest term is C(8,4) = 70,
    so the whole curve is 107 majority votes per prompt and there is no reason to
    accept estimator variance on top of the eight-Bernoulli-draw variance the
    cached siblings already carry.

    Each subset is scored by the frozen ``_plurality_outcome``, so the winner is
    ``_winning_answer``'s -- plurality over parseable answers, ties broken by the
    highest log-probability -- and a subset in which nothing parses scores zero,
    matching ``prompt_accounting``'s automatic-failure rule.
    """
    group = list(group)
    if not 1 <= k <= len(group):
        return float("nan")
    return float(
        np.mean([_plurality_outcome(list(subset)) for subset in combinations(group, k)])
    )


def accuracy_curves(
    rows: Iterable[Mapping], sizes: Sequence[int] = SUBSET_SIZES
) -> dict[int, dict[int, float]]:
    """``{prompt_id: {k: a(p, k)}}`` for every cached prompt."""
    return {
        prompt_id: {int(k): subset_accuracy(group, int(k)) for k in sizes}
        for prompt_id, group in sorted(_group_rows(rows).items())
    }


def gain_targets(curves: Mapping[int, Mapping[int, float]]) -> dict[int, float]:
    """``g(p) = a(p, 8) - a(p, 1)``, the quantity an allocator would want to rank by."""
    return {
        prompt_id: float(curve[GAIN_TO] - curve[GAIN_FROM])
        for prompt_id, curve in curves.items()
    }


def stage1_rows(rows: Iterable[Mapping], position: int) -> list[dict]:
    """The ``position``-th cached sibling of every prompt, ordered by ``sample_id``.

    Ordering is by the stored sample index rather than by file order so the eight
    draws are the same eight partitions on every model and every rerun.
    """
    selected: list[dict] = []
    for _, group in sorted(_group_rows(rows).items()):
        ordered = sorted(group, key=lambda row: (int(row["sample_id"]), int(row["trace_id"])))
        if position < len(ordered):
            selected.append(ordered[position])
    return selected


def crossfit_ridge_predictions(
    features: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Out-of-fold linear predictions of a continuous target on the frozen folds.

    The regression counterpart of ``crossfit_logistic_predictions``: same folds,
    same ``_impute_and_scale`` per fold, same "never predict a prompt from a model
    that saw it" discipline.  Ridge at ``alpha = 1`` on standardized columns
    mirrors the L2 default the frozen logistic readout carries; with at most four
    features and a few hundred prompts it is within rounding of ordinary least
    squares, so nothing here turns on the penalty.
    """
    from sklearn.linear_model import Ridge

    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(target, dtype=float)
    fold_values = np.asarray(folds)
    if len(x) != len(y) or len(y) != len(fold_values):
        raise ValueError("features, target, and folds must have the same length")
    predictions = np.full(len(y), np.nan, dtype=float)
    usable = np.isfinite(y) & np.isfinite(fold_values.astype(float))
    for fold in sorted(set(fold_values[usable].tolist())):
        test_mask = usable & (fold_values == fold)
        train_mask = usable & (fold_values != fold)
        if not test_mask.any() or not train_mask.any():
            continue
        train_x, test_x = _impute_and_scale(x[train_mask], x[test_mask])
        model = Ridge(alpha=alpha)
        model.fit(train_x, y[train_mask])
        predictions[test_mask] = model.predict(test_x)
    return predictions


def crossfit_constant_predictions(target: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """The training-fold mean, held out the same way the readouts are.

    This is the "constant predictor" the gate is scored against.  Using the
    *overall* mean instead would let the baseline see the held-out prompts while
    the readouts cannot, which biases every R^2 upward by exactly the amount the
    comparison is trying to measure.
    """
    y = np.asarray(target, dtype=float)
    fold_values = np.asarray(folds)
    predictions = np.full(len(y), np.nan, dtype=float)
    usable = np.isfinite(y) & np.isfinite(fold_values.astype(float))
    for fold in sorted(set(fold_values[usable].tolist())):
        test_mask = usable & (fold_values == fold)
        train_mask = usable & (fold_values != fold)
        if not test_mask.any() or not train_mask.any():
            continue
        predictions[test_mask] = float(np.mean(y[train_mask]))
    return predictions


def r2_against_constant(
    predictions: np.ndarray, constant: np.ndarray, target: np.ndarray
) -> float:
    """``1 - SSE(predictions) / SSE(constant)``, both out of fold.

    Negative means the readout is worse than predicting the training mean, which
    for a four-feature ridge on a few hundred prompts is an ordinary outcome and
    the one the gate is looking for.
    """
    predictions = np.asarray(predictions, dtype=float)
    constant = np.asarray(constant, dtype=float)
    target = np.asarray(target, dtype=float)
    usable = np.isfinite(predictions) & np.isfinite(constant) & np.isfinite(target)
    if usable.sum() < 3:
        return float("nan")
    residual = float(np.sum((target[usable] - predictions[usable]) ** 2))
    baseline = float(np.sum((target[usable] - constant[usable]) ** 2))
    return float("nan") if baseline <= 0 else 1.0 - residual / baseline


def analyze_draw(
    stage1: Mapping[int, Mapping],
    prompt_ids: Sequence[int],
    gains: Mapping[int, float],
    pass_rates: Mapping[int, float],
    outcomes: Mapping[int, float],
    own_correct: Mapping[int, float],
    folds: Mapping[int, int],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """One choice of stage-1 trace: the three readouts plus the diagnostics."""
    y = np.asarray([gains[i] for i in prompt_ids], dtype=float)
    fold_values = np.asarray([folds[i] for i in prompt_ids])
    columns = {
        name: np.asarray([stage1[i][name] for i in prompt_ids], dtype=float)
        for name in FEATURE_SETS["both"]
    }
    constant = crossfit_constant_predictions(y, fold_values)
    readouts = {}
    for name, spec in FEATURE_SETS.items():
        predictions = crossfit_ridge_predictions(
            np.column_stack([columns[column] for column in spec]), y, fold_values
        )
        readouts[name] = {
            "features": list(spec),
            "spearman": spearman(predictions, y),
            "r2_vs_constant": r2_against_constant(predictions, constant, y),
        }
    geometry = columns["rmd_tail_q20"]
    return {
        "readouts": readouts,
        # The paired quantity the gate reads.  Taking the median of this
        # difference is not the same as differencing the medians, and the draws
        # are paired by construction -- same prompts, same folds, same target --
        # so the paired form is the one with less noise in it.
        "spearman_gain_from_geometry": float(
            readouts["both"]["spearman"] - readouts["output"]["spearman"]
        ),
        "diagnostics": {
            # Feature degradation at n=1, holding the *target* fixed at the
            # eight-sibling plurality outcome so this is directly comparable to
            # the marginal AUROC of the sibling-mean feature in the 2026-08-10
            # table (0.806 / 0.686 / 0.709).
            "auroc_vs_prompt_outcome": bootstrap_auroc(
                geometry,
                np.asarray([outcomes[i] for i in prompt_ids], dtype=float),
                n_bootstrap=n_bootstrap,
                seed=seed,
            ),
            # The n=1 decision problem itself: does this trace's geometry predict
            # whether *this trace* was right?
            "auroc_vs_own_trace_correct": bootstrap_auroc(
                geometry,
                np.asarray([own_correct[i] for i in prompt_ids], dtype=float),
                n_bootstrap=n_bootstrap,
                seed=seed,
            ),
            # The pair that names the failure mode.
            "spearman_vs_pass_rate": spearman(
                geometry, np.asarray([pass_rates[i] for i in prompt_ids], dtype=float)
            ),
            "spearman_vs_gain": spearman(geometry, y),
        },
    }


def _summarize(values: Sequence[float]) -> dict:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(finite):
        return {"median": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "median": float(np.median(finite)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "n": int(len(finite)),
    }


def harness_checks(
    curves: Mapping[int, Mapping[int, float]],
    features: Mapping[int, Mapping],
    pass_rates: Mapping[int, float],
    prompt_ids: Sequence[int],
) -> dict:
    """``a(p, 8)`` is the frozen outcome, and ``a(p, 1)`` is the pass rate.

    The first is an identity -- C(8,8) is the single subset containing every
    sibling, so ``subset_accuracy(group, 8)`` *is* ``_plurality_outcome(group)``,
    which is what ``aggregate_prompt_features`` stores.  It is asserted rather
    than reported: if it ever fails, the subset machinery has drifted from the
    frozen majority-vote convention and every number below is meaningless.

    The second is not an identity.  ``a(p, 1)`` is the mean over siblings of
    "this trace's answer equals gold", while the cached ``is_correct`` column is
    the collector's own verdict; they can disagree wherever answer normalization
    differs.  The size of that disagreement is reported, not enforced.
    """
    outcome_gap = max(
        abs(curves[i][GAIN_TO] - float(features[i]["outcome"])) for i in prompt_ids
    )
    if outcome_gap > 1e-12:
        raise AssertionError(
            f"a(p,8) does not reproduce the frozen prompt outcome (max gap {outcome_gap}); "
            "the subset majority vote has drifted from incremental_abstention"
        )
    differences = [
        abs(curves[i][GAIN_FROM] - pass_rates[i])
        for i in prompt_ids
        if np.isfinite(pass_rates[i])
    ]
    return {
        "a8_equals_frozen_outcome": True,
        "a1_vs_cached_is_correct": {
            "max_abs_difference": float(max(differences)) if differences else float("nan"),
            "n_prompts_differing": int(sum(d > 1e-12 for d in differences)),
            "n_prompts": len(differences),
        },
    }


def cached_pass_rates(rows: Iterable[Mapping]) -> dict[int, float]:
    """Mean of the cached ``is_correct`` column over a prompt's siblings.

    The same definition ``peer_difficulty_control.peer_pass_rates`` uses, so the
    difficulty diagnostic below is on the same scale as the difficulty control
    that motivated this precheck.
    """
    rates: dict[int, float] = {}
    for prompt_id, group in sorted(_group_rows(rows).items()):
        values = [
            value for row in group if (value := _finite(row.get("is_correct"))) is not None
        ]
        rates[prompt_id] = float(np.mean(values)) if values else float("nan")
    return rates


def analyze_model(
    label: str,
    oof_csv: str | Path,
    data_dir: str | Path,
    *,
    population: str = "cap_free_valid_plurality",
    layer: int | None = None,
    expected_traces: int = 8,
    n_draws: int | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    rows, layer = select_layer_rows(_read_oof(oof_csv), layer, context=str(oof_csv))
    features = aggregate_prompt_features(
        rows, data_dir=str(data_dir), expected_traces=expected_traces
    )
    grouped = _group_rows(rows)
    available = _populations(features)
    eligible = [
        prompt_id
        for prompt_id in available[population]
        if features[prompt_id]["fold"] is not None
    ]
    # The exhaustive curve needs a full sibling set: C(8,k) is not comparable to
    # C(7,k), and a prompt short of the expected count has no a(p,8) at all.
    prompt_ids = [i for i in eligible if len(grouped[i]) == int(expected_traces)]

    curves = accuracy_curves(rows)
    gains = gain_targets(curves)
    pass_rates = cached_pass_rates(rows)
    checks = harness_checks(curves, features, pass_rates, prompt_ids)

    draws = int(expected_traces if n_draws is None else n_draws)
    per_draw = []
    for position in range(draws):
        selected = stage1_rows(rows, position)
        stage1 = aggregate_prompt_features(
            selected, data_dir=str(data_dir), expected_traces=1
        )
        own_correct = {
            # A missing verdict counts as not correct, matching the convention
            # everywhere else here that an unparseable trace did not solve it.
            int(row["prompt_id"]): (
                0.0 if (value := _finite(row.get("is_correct"))) is None else value
            )
            for row in selected
        }
        body = analyze_draw(
            stage1,
            prompt_ids,
            gains,
            pass_rates,
            {i: float(features[i]["outcome"]) for i in prompt_ids},
            own_correct,
            {i: int(features[i]["fold"]) for i in prompt_ids},
            n_bootstrap=n_bootstrap,
            # Vary the seed with the draw so the eight bootstraps are not the same
            # resampling pattern replayed eight times.
            seed=seed + position,
        )
        per_draw.append({"stage1_position": position, **body})

    gain_values = np.asarray([gains[i] for i in prompt_ids], dtype=float)
    rate_values = np.asarray([pass_rates[i] for i in prompt_ids], dtype=float)
    return {
        "label": label,
        "layer": layer,
        "population": population,
        "n_prompts": len(prompt_ids),
        "n_prompts_dropped_incomplete": len(eligible) - len(prompt_ids),
        "base_accuracy": float(np.mean([features[i]["outcome"] for i in prompt_ids])),
        "harness_checks": checks,
        "gain_summary": {
            "mean": float(np.mean(gain_values)),
            "sd": float(np.std(gain_values)),
            "min": float(gain_values.min()),
            "max": float(gain_values.max()),
            "share_exactly_zero": float(np.mean(gain_values == 0.0)),
            "share_negative": float(np.mean(gain_values < 0.0)),
            # The non-monotonicity, measured rather than asserted: if gain were a
            # monotone function of difficulty this would be near -1, and the whole
            # precheck would be redundant with the difficulty results.
            "spearman_pass_rate_vs_gain": spearman(rate_values, gain_values),
        },
        "accuracy_curve_mean": {
            str(k): float(np.mean([curves[i][k] for i in prompt_ids])) for k in SUBSET_SIZES
        },
        "draws": per_draw,
        "across_draws": {
            "spearman": {
                name: _summarize([draw["readouts"][name]["spearman"] for draw in per_draw])
                for name in FEATURE_SETS
            },
            "r2_vs_constant": {
                name: _summarize(
                    [draw["readouts"][name]["r2_vs_constant"] for draw in per_draw]
                )
                for name in FEATURE_SETS
            },
            "spearman_gain_from_geometry": _summarize(
                [draw["spearman_gain_from_geometry"] for draw in per_draw]
            ),
            "auroc_vs_prompt_outcome": _summarize(
                [draw["diagnostics"]["auroc_vs_prompt_outcome"]["point_estimate"]
                 for draw in per_draw]
            ),
            "auroc_vs_own_trace_correct": _summarize(
                [draw["diagnostics"]["auroc_vs_own_trace_correct"]["point_estimate"]
                 for draw in per_draw]
            ),
            "spearman_vs_pass_rate": _summarize(
                [draw["diagnostics"]["spearman_vs_pass_rate"] for draw in per_draw]
            ),
            "spearman_vs_gain": _summarize(
                [draw["diagnostics"]["spearman_vs_gain"] for draw in per_draw]
            ),
        },
    }


def gate_verdict(results: Sequence[Mapping]) -> dict:
    """The pre-declared gate, evaluated mechanically on the eight-draw medians."""
    per_model = {}
    for body in results:
        summary = body["across_draws"]
        beats_constant = summary["r2_vs_constant"]["geometry"]["median"] > 0.0
        adds_over_output = summary["spearman_gain_from_geometry"]["median"] > 0.0
        per_model[body["label"]] = {
            "geometry_r2_median": summary["r2_vs_constant"]["geometry"]["median"],
            "geometry_beats_constant": bool(beats_constant),
            "spearman_gain_from_geometry_median": summary["spearman_gain_from_geometry"][
                "median"
            ],
            "geometry_adds_over_output": bool(adds_over_output),
            "passes": bool(beats_constant and adds_over_output),
        }
    passing = [label for label, entry in per_model.items() if entry["passes"]]
    return {
        "rule": (
            "geometry alone beats the cross-fitted constant (R^2 > 0) and adds over "
            "output-alone in out-of-fold Spearman, on at least "
            f"{GATE_MIN_MODELS} of the models; medians over the stage-1 draws"
        ),
        "n_models": len(results),
        "models_passing": passing,
        "per_model": per_model,
        "passes": len(passing) >= GATE_MIN_MODELS,
        "consequence": (
            "write allocation.py (step 3)"
            if len(passing) >= GATE_MIN_MODELS
            else "step 3 is not run; geometry contributes nothing to allocation and "
            "that is the finding"
        ),
    }


def failure_mode_flags(results: Sequence[Mapping]) -> dict[str, bool]:
    """Models where geometry reads difficulty but not marginal gain.

    Descriptive labelling of the pattern the precheck exists to catch, using the
    two thresholds declared at the top of this module.  It does not feed the gate.
    """
    flags = {}
    for body in results:
        summary = body["across_draws"]
        flags[body["label"]] = bool(
            abs(summary["spearman_vs_pass_rate"]["median"]) >= DIFFICULTY_NOT_GAIN_STRONG
            and abs(summary["spearman_vs_gain"]["median"]) < DIFFICULTY_NOT_GAIN_NEAR_ZERO
        )
    return flags


def _fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def _spread(entry: Mapping, digits: int = 3) -> str:
    return (
        f"{_fmt(entry['median'], digits)} "
        f"[{_fmt(entry['min'], digits)}, {_fmt(entry['max'], digits)}]"
    )


def write_report(results: Sequence[Mapping], path: str | Path) -> None:
    verdict = gate_verdict(results)
    flags = failure_mode_flags(results)
    lines = [
        "# Step 2 -- allocation precheck",
        "",
        "Does single-trace hidden-state geometry predict the **gain from buying more",
        "samples**, `g(p) = a(p,8) - a(p,1)`? This is not the correctness question every",
        "other rung asks. Gain is non-monotone in difficulty: a prompt solved 0/8 and a",
        "prompt solved 8/8 both gain nothing.",
        "",
        "`a(p,k)` is the expected plurality-vote correctness over **all** `C(8,k)` sibling",
        "subsets, using the frozen `_winning_answer` and the frozen automatic-failure rule",
        "for an all-unparsed subset. Stage-1 features are the four that exist at one",
        "sample; `vote_agreement` is excluded because a single sample has no siblings.",
        "",
        "Every number below is the median over the **eight choices of stage-1 trace**, with",
        "the full range in brackets. Which trace you happen to draw first is a random",
        "variable, and fixing it at `sample_id == 0` would hide that variance.",
        "",
        "## Population and target",
        "",
        "| model | layer | n | base acc | mean g | share g = 0 | share g < 0 | rho(pass rate, g) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for body in results:
        gain = body["gain_summary"]
        lines.append(
            f"| {body['label']} | {body['layer']} | {body['n_prompts']} | "
            f"{_fmt(body['base_accuracy'])} | {_fmt(gain['mean'])} | "
            f"{_fmt(gain['share_exactly_zero'])} | {_fmt(gain['share_negative'])} | "
            f"{_fmt(gain['spearman_pass_rate_vs_gain'])} |"
        )
    lines += [
        "",
        "`rho(pass rate, g)` is the non-monotonicity, measured rather than asserted: were",
        "gain a monotone function of difficulty it would sit near -1 and this precheck",
        "would be redundant with the difficulty results.",
        "",
        "Mean expected accuracy along the curve:",
        "",
        "| model | " + " | ".join(f"a(p,{k})" for k in SUBSET_SIZES) + " |",
        "|---|" + "---:|" * len(SUBSET_SIZES),
    ]
    for body in results:
        lines.append(
            f"| {body['label']} | "
            + " | ".join(_fmt(body["accuracy_curve_mean"][str(k)]) for k in SUBSET_SIZES)
            + " |"
        )

    lines += [
        "",
        "## The gate",
        "",
        "Out-of-fold prediction of `g(p)` on the frozen prompt folds. `R^2` is against a",
        "**cross-fitted constant** -- the training-fold mean -- so the baseline is held out",
        "exactly as the readouts are.",
        "",
        "| model | Spearman geometry | Spearman output | Spearman both | R^2 geometry | R^2 output | R^2 both |",
        "|---|---|---|---|---|---|---|",
    ]
    for body in results:
        summary = body["across_draws"]
        lines.append(
            f"| {body['label']} | "
            + " | ".join(_spread(summary["spearman"][name]) for name in FEATURE_SETS)
            + " | "
            + " | ".join(_spread(summary["r2_vs_constant"][name]) for name in FEATURE_SETS)
            + " |"
        )

    lines += [
        "",
        "| model | geometry beats constant | geometry adds over output (paired Spearman) | passes |",
        "|---|---|---|---|",
    ]
    for body in results:
        entry = verdict["per_model"][body["label"]]
        summary = body["across_draws"]
        lines.append(
            f"| {body['label']} | "
            f"{'yes' if entry['geometry_beats_constant'] else 'no'} "
            f"(R^2 {_fmt(entry['geometry_r2_median'])}) | "
            f"{'yes' if entry['geometry_adds_over_output'] else 'no'} "
            f"({_spread(summary['spearman_gain_from_geometry'])}) | "
            f"**{'PASS' if entry['passes'] else 'fail'}** |"
        )
    lines += [
        "",
        f"Pre-declared rule: {verdict['rule']}.",
        "",
        f"Passing: {', '.join(verdict['models_passing']) or 'none'} "
        f"({len(verdict['models_passing'])}/{verdict['n_models']}). "
        f"Gate: **{'PASS' if verdict['passes'] else 'FAIL'}** -- {verdict['consequence']}.",
        "",
        "## Diagnostics (not the gate)",
        "",
        "These explain a failure; they do not decide one.",
        "",
        "`AUROC vs prompt outcome` holds the target fixed at the eight-sibling plurality",
        "outcome and varies only the feature, so it is directly comparable to the",
        "sibling-mean marginal AUROC in the 2026-08-10 table (0.806 / 0.686 / 0.709) and",
        "measures how much `rmd_tail_q20` degrades at n = 1. `AUROC vs own trace` is the",
        "n = 1 decision problem itself.",
        "",
        "| model | AUROC vs prompt outcome | AUROC vs own trace | rho(geometry, pass rate) | rho(geometry, g) |",
        "|---|---|---|---|---|",
    ]
    for body in results:
        summary = body["across_draws"]
        lines.append(
            f"| {body['label']} | {_spread(summary['auroc_vs_prompt_outcome'])} | "
            f"{_spread(summary['auroc_vs_own_trace_correct'])} | "
            f"{_spread(summary['spearman_vs_pass_rate'])} | "
            f"{_spread(summary['spearman_vs_gain'])} |"
        )

    named = [label for label, flagged in flags.items() if flagged]
    lines += [
        "",
        f"Flagged with |rho(geometry, pass rate)| >= {DIFFICULTY_NOT_GAIN_STRONG:.2f} and "
        f"|rho(geometry, g)| < {DIFFICULTY_NOT_GAIN_NEAR_ZERO:.2f}: "
        f"{', '.join(named) or 'none'}.",
        "",
    ]
    if named:
        lines += [
            "On these models **geometry reads difficulty but not marginal gain** -- the",
            "specific failure mode this precheck exists to catch. It is consistent with the",
            "2026-08-10 peer control, which found most of the `rmd_tail_q20` increment is",
            "prompt difficulty: difficulty is exactly the thing that does *not* order",
            "prompts by how much another sample would help.",
            "",
        ]

    lines += [
        "## Harness checks",
        "",
        "| model | a(p,8) == frozen outcome | max |a(p,1) - cached pass rate| | prompts differing |",
        "|---|---|---:|---:|",
    ]
    for body in results:
        checks = body["harness_checks"]
        a1 = checks["a1_vs_cached_is_correct"]
        lines.append(
            f"| {body['label']} | {'yes' if checks['a8_equals_frozen_outcome'] else 'NO'} | "
            f"{_fmt(a1['max_abs_difference'], 4)} | "
            f"{a1['n_prompts_differing']}/{a1['n_prompts']} |"
        )
    lines += [
        "",
        "`a(p,8)` reproducing the frozen prompt outcome is an identity -- C(8,8) is the one",
        "subset containing every sibling -- and it is asserted at run time, not merely",
        "reported. `a(p,1)` against the cached `is_correct` column is *not* an identity:",
        "the first recomputes the answer match through the frozen parser, the second is the",
        "collector's stored verdict, and any gap is answer-normalization drift.",
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
    parser.add_argument("--population", default="cap_free_valid_plurality")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--expected_traces", type=int, default=8)
    parser.add_argument(
        "--n_draws",
        type=int,
        default=None,
        help="stage-1 trace choices to run; defaults to --expected_traces, i.e. all of them",
    )
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/allocation_precheck")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [
        analyze_model(
            *spec.split(":", 2),
            population=args.population,
            layer=args.layer,
            expected_traces=args.expected_traces,
            n_draws=args.n_draws,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        for spec in args.model
    ]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": (
            "g(p) = a(p,8) - a(p,1), where a(p,k) is the expected plurality-vote "
            "correctness over all C(8,k) sibling subsets"
        ),
        "subset_sizes": list(SUBSET_SIZES),
        "feature_sets": {name: list(spec) for name, spec in FEATURE_SETS.items()},
        "readout": "cross-fitted ridge (alpha=1) on the frozen prompt folds",
        "constant_baseline": "cross-fitted training-fold mean",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "gate": gate_verdict(results),
        "difficulty_not_gain": failure_mode_flags(results),
        "models": results,
    }
    (output / "allocation_precheck_results.json").write_text(json.dumps(payload, indent=2))
    write_report(results, output / "allocation_precheck_report.md")
    print(f"wrote {output}/allocation_precheck_report.md")
    print(
        f"gate: {'PASS' if payload['gate']['passes'] else 'FAIL'} "
        f"({len(payload['gate']['models_passing'])}/{payload['gate']['n_models']}) "
        f"-- {payload['gate']['consequence']}"
    )


if __name__ == "__main__":
    main()
