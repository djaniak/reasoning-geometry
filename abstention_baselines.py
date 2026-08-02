"""Region-matched output-confidence baselines for the prompt-level abstention claim.

Wave 1's surviving result is between-prompt: `rmd_tail_q20` ranks prompts by
solvability better than length, logprob, or entropy. Those three baselines are all
*global* (whole-trace) statistics, so the comparison confounds "geometry beats
output signals" with "the tail region beats the whole trace". This module removes
that confound by scoring the output side on the *same token masks* the geometry
scorers use, and adds the DeepConf-style aggregations (sliding-window group
confidence, cross-trace vote agreement) that are the current SOTA for exactly this
task.

The decision this module exists to settle: if `rmd_tail_q20` does not beat
region-matched tail confidence, the wave-1 result is a reimplementation of an
output-side signal, not a geometry result.

Two things it also fixes, because ranking metrics alone cannot support a routing
claim: calibration (ECE/Brier, currently absent everywhere) and decision values
(selective accuracy and token savings at fixed abstention rates).

Approximation notice -- DeepConf (arXiv 2508.15260) defines token confidence as
the negative mean log-probability of the top-k next-token candidates. Collection
stored only the sampled token's log-probability and the full-distribution entropy,
not the top-k tail, so the exact statistic is unrecoverable from cached data.
Entropy is the untruncated limit of that quantity and the sampled-token
log-probability is its one-sample estimate, so both are reported side by side and
every DeepConf-derived scorer is named `*_ent` or `*_lp` accordingly. A claim that
survives under both is not sensitive to the substitution.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze import load_all_traces
from prompt_decomposition import region_indices
from wave1_experiments import (
    _ci_pvalue,
    _load_oof_csv,
    _majority_outcomes,
    _prompt_aurc,
    _prompt_coverage_accuracy,
    aggregate_prompt_scores,
    length_residualized_abstention,
    prompt_abstention_bootstrap,
)

# Trace-level scorers computed here, all oriented higher = more likely correct.
TRACE_METHODS = (
    "conf_tail_q20_ent",
    "conf_tail_q20_lp",
    "conf_he_q20_ent",
    "conf_he_q20_lp",
    "conf_lowest_group_ent",
    "conf_lowest_group_lp",
    "conf_bottom10_group_ent",
    "conf_bottom10_group_lp",
)

# Prompt-level scorers; they have no per-trace value, so they bypass aggregation.
PROMPT_METHODS = (
    "vote_agreement",
    "conf_weighted_vote_ent",
    "conf_weighted_vote_lp",
)

# Scorers carried over from the wave-1 OOF CSV for the head-to-head.
CARRIED_METHODS = (
    "rmd_tail_q20",
    "rmd_high_entropy_q20",
    "length",
    "logprob",
    "entropy",
)

# The Gate A family: does geometry beat the output side on a matched token mask?
# Holm-corrected together because they are the comparisons the decision rests on.
PRIMARY_COMPARISONS = (
    ("rmd_tail_q20", "conf_tail_q20_ent"),
    ("rmd_tail_q20", "conf_tail_q20_lp"),
    ("rmd_tail_q20", "conf_bottom10_group_ent"),
    ("rmd_tail_q20", "vote_agreement"),
)

METHOD_DESCRIPTIONS = {
    "conf_tail_q20_ent": "-mean(entropy) over the last 20% of tokens (mask-matched to rmd_tail_q20)",
    "conf_tail_q20_lp": "mean(token logprob) over the last 20% of tokens (mask-matched to rmd_tail_q20)",
    "conf_he_q20_ent": "-mean(entropy) over the highest-entropy 20% of tokens",
    "conf_he_q20_lp": "mean(token logprob) over the highest-entropy 20% of tokens",
    "conf_lowest_group_ent": "DeepConf lowest group confidence, entropy form, window = 20% of trace",
    "conf_lowest_group_lp": "DeepConf lowest group confidence, logprob form, window = 20% of trace",
    "conf_bottom10_group_ent": "DeepConf bottom-10% group confidence, entropy form",
    "conf_bottom10_group_lp": "DeepConf bottom-10% group confidence, logprob form",
    "vote_agreement": "share of parseable traces agreeing with the majority answer",
    "conf_weighted_vote_ent": "entropy-confidence-weighted vote share for the majority answer",
    "conf_weighted_vote_lp": "likelihood-weighted vote share for the majority answer",
    "rmd_tail_q20": "-mean(relative Mahalanobis) over the last 20% of tokens",
    "rmd_high_entropy_q20": "-mean(relative Mahalanobis) over the highest-entropy 20% of tokens",
    "length": "-log1p(token count)",
    "logprob": "mean token log-probability",
    "entropy": "-mean(token entropy)",
}

ABSTENTION_RATES = (0.10, 0.25, 0.50, 0.75, 0.90)


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def group_confidences(values: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window means of a per-token confidence series.

    DeepConf fixes the window at 2048 tokens, which exceeds most traces here, so
    the window is taken proportional to trace length instead. A fixed window would
    silently collapse every short trace's group statistics onto the whole-trace
    mean and make the "lowest group" scorers duplicates of the global ones.
    """
    values = np.asarray(values, dtype=float)
    window = max(1, min(int(window), values.size))
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    starts = np.arange(0, values.size - window + 1)
    return (cumulative[starts + window] - cumulative[starts]) / float(window)


def trace_confidence_scores(
    entropies: np.ndarray | None,
    token_logprobs: np.ndarray | None,
) -> dict[str, float]:
    """Region-matched and DeepConf-style confidence scores for one trace.

    Scores are ``-inf`` when the underlying token series is missing or non-finite,
    matching the sentinel the wave-1 aggregation already understands.
    """
    scores = {method: float("-inf") for method in TRACE_METHODS}
    if entropies is None:
        return scores
    entropies = np.asarray(entropies, dtype=float)
    if entropies.ndim != 1 or entropies.size == 0 or not np.all(np.isfinite(entropies)):
        return scores

    # Higher = more confident for both series, so the two forms stay comparable.
    series = {"ent": -entropies}
    if token_logprobs is not None:
        logprobs = np.asarray(token_logprobs, dtype=float)
        if logprobs.shape == entropies.shape and np.all(np.isfinite(logprobs)):
            series["lp"] = logprobs

    window = max(1, int(np.ceil(0.20 * entropies.size)))
    for suffix, values in series.items():
        # region_indices is imported rather than reimplemented so these masks are
        # bit-identical to the ones the RMD scorers were computed on.
        tail = region_indices(entropies, "tail_q20")
        high_entropy = region_indices(entropies, "high_entropy_q20")
        groups = group_confidences(values, window)
        keep = max(1, int(np.ceil(0.10 * groups.size)))
        scores[f"conf_tail_q20_{suffix}"] = float(values[tail].mean())
        scores[f"conf_he_q20_{suffix}"] = float(values[high_entropy].mean())
        scores[f"conf_lowest_group_{suffix}"] = float(groups.min())
        scores[f"conf_bottom10_group_{suffix}"] = float(np.sort(groups)[:keep].mean())
    return scores


def _winning_answer(rows: list[dict]) -> str | None:
    """The majority answer, broken by summed logprob -- as wave 1 breaks it."""
    parsed = [row for row in rows if row.get("predicted_answer") not in (None, "")]
    if not parsed:
        return None
    counts = Counter(str(row["predicted_answer"]) for row in parsed)
    best = max(counts.values())
    candidates = sorted(answer for answer, count in counts.items() if count == best)
    if len(candidates) == 1:
        return candidates[0]
    return max(
        candidates,
        key=lambda answer: max(
            float(row.get("logprob_score", -np.inf))
            for row in parsed
            if str(row["predicted_answer"]) == answer
        ),
    )


def prompt_vote_scores(rows_by_prompt: dict[int, list[dict]]) -> dict[int, dict[str, float]]:
    """Cross-trace agreement scores, DeepConf's confidence-weighted-vote family.

    These are prompt-level by construction: a single trace has no agreement with
    itself. Prompts whose traces are all unparseable score ``-inf``, the same
    sentinel the trace-level path uses.
    """
    scores: dict[int, dict[str, float]] = {}
    for prompt_id, rows in rows_by_prompt.items():
        parsed = [row for row in rows if row.get("predicted_answer") not in (None, "")]
        winner = _winning_answer(rows)
        if winner is None:
            scores[prompt_id] = {method: float("-inf") for method in PROMPT_METHODS}
            continue
        agree = np.asarray([str(row["predicted_answer"]) == winner for row in parsed])
        entry = {"vote_agreement": float(agree.mean())}
        for suffix in ("ent", "lp"):
            weights = np.asarray(
                [
                    float(row.get(f"conf_tail_q20_{suffix}_score", float("-inf")))
                    for row in parsed
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(weights)):
                entry[f"conf_weighted_vote_{suffix}"] = float("-inf")
                continue
            # DeepConf's weighted vote sums a strictly positive per-trace
            # confidence. Both of our forms are mean log-quantities, so exp maps
            # them into (0, 1] -- a per-token likelihood -- which restores the
            # positivity the weighting assumes without introducing a temperature.
            weights = np.exp(weights)
            total = float(weights.sum())
            entry[f"conf_weighted_vote_{suffix}"] = (
                float(weights[agree].sum() / total) if total > 0 else float("-inf")
            )
        scores[prompt_id] = entry
    return scores


def _oof_probabilities(
    scores: np.ndarray, outcomes: np.ndarray, *, n_splits: int = 5, seed: int = 42
) -> np.ndarray:
    """Out-of-fold logistic calibration of a single score into P(correct).

    Prompts are the unit of analysis here, so plain K-fold over prompts is already
    group-clean -- no prompt contributes to the fit that scores it.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    usable = np.isfinite(scores)
    probabilities = np.full(scores.shape, np.nan, dtype=float)
    if usable.sum() < n_splits * 2 or len(np.unique(outcomes[usable])) < 2:
        return probabilities
    index = np.flatnonzero(usable)
    features = scores[index].reshape(-1, 1)
    labels = outcomes[index].astype(int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in splitter.split(features, labels):
        if len(np.unique(labels[train])) < 2:
            continue
        model = LogisticRegression(max_iter=1000)
        model.fit(features[train], labels[train])
        probabilities[index[test]] = model.predict_proba(features[test])[:, 1]
    return probabilities


def calibration_metrics(
    probabilities: np.ndarray, outcomes: np.ndarray, *, n_bins: int = 15
) -> dict:
    """Expected calibration error and Brier score over the calibrated prompts."""
    usable = np.isfinite(probabilities)
    if usable.sum() == 0:
        return {"ece": None, "brier": None, "n_calibrated": 0, "n_bins": n_bins}
    probabilities = probabilities[usable]
    labels = outcomes[usable]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for bin_index in range(n_bins):
        members = assignments == bin_index
        if not members.any():
            continue
        weight = members.mean()
        ece += weight * abs(labels[members].mean() - probabilities[members].mean())
    return {
        "ece": float(ece),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "n_calibrated": int(usable.sum()),
        "n_bins": n_bins,
    }


def decision_values(
    scores: dict[int, float],
    outcomes: dict[int, float],
    prompt_tokens: dict[int, float],
    *,
    rates: tuple[float, ...] = ABSTENTION_RATES,
) -> dict:
    """Selective accuracy and token savings at fixed abstention rates.

    The optimal abstention policy under a fixed abstain reward is a threshold on
    P(correct) (Knowing When to Quit, arXiv 2604.18419), so what a routing user
    needs is the accuracy/spend pair at each operating point -- not an area
    summary that averages over operating points they will never run at.
    """
    prompt_ids = sorted(outcomes)
    order = sorted(prompt_ids, key=lambda pid: (-float(scores.get(pid, -np.inf)), pid))
    total_tokens = float(sum(prompt_tokens.get(pid, 0.0) for pid in prompt_ids))
    table = {}
    for rate in rates:
        kept = max(1, len(order) - int(np.floor(float(rate) * len(order))))
        selected = order[:kept]
        spent = float(sum(prompt_tokens.get(pid, 0.0) for pid in selected))
        table[str(rate)] = {
            "coverage": float(len(selected) / len(order)),
            "selective_accuracy": float(np.mean([outcomes[pid] for pid in selected])),
            "token_savings": float(1.0 - spent / total_tokens) if total_tokens > 0 else None,
        }
    return table


def holm_correction(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values within one comparison family."""
    usable = {key: value for key, value in pvalues.items() if value is not None}
    ordered = sorted(usable.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (len(ordered) - rank)))
        adjusted[key] = float(running)
    for key, value in pvalues.items():
        adjusted.setdefault(key, None)
    return adjusted


def run_abstention_baselines(
    *,
    data_dir: str,
    oof_csv: str,
    output_dir: str,
    dataset_label: str,
    model_label: str,
    layers: list[int],
    n_bootstrap: int = 1000,
    seed: int = 42,
    load_workers: int = 4,
    n_splits: int = 5,
) -> dict:
    rows = _load_oof_csv(oof_csv)
    layer = max(layers)
    layer_rows = [row for row in rows if int(row["layer"]) == layer]
    if not layer_rows:
        raise ValueError(f"no OOF rows at layer {layer} in {oof_csv}")

    _status(f"Loading token series from {data_dir} (no hidden states)")
    traces = load_all_traces(
        data_dir,
        [],
        max_workers=load_workers,
        show_progress=True,
        include_auxiliary=True,
        auxiliary_fields={"entropies", "token_logprobs"},
    )
    by_trace = {int(trace["trace_id"]): trace for trace in traces}
    missing = 0
    for row in layer_rows:
        trace = by_trace.get(int(row["trace_id"]))
        if trace is None:
            missing += 1
            row.update({f"{method}_score": float("-inf") for method in TRACE_METHODS})
            continue
        scores = trace_confidence_scores(trace.get("entropies"), trace.get("token_logprobs"))
        row.update({f"{method}_score": value for method, value in scores.items()})
    if missing:
        _status(f"  {missing}/{len(layer_rows)} OOF rows had no matching trace in {data_dir}")

    rows_by_prompt: dict[int, list[dict]] = defaultdict(list)
    for row in layer_rows:
        rows_by_prompt[int(row["prompt_id"])].append(row)

    carried = tuple(m for m in CARRIED_METHODS if f"{m}_score" in layer_rows[0])
    prompt_scores = aggregate_prompt_scores(layer_rows, methods=carried + TRACE_METHODS)
    vote_scores = prompt_vote_scores(rows_by_prompt)
    for prompt_id, entry in prompt_scores.items():
        entry.update(vote_scores.get(prompt_id, {m: float("-inf") for m in PROMPT_METHODS}))
    methods = carried + TRACE_METHODS + PROMPT_METHODS

    outcomes = _majority_outcomes(layer_rows)
    prompt_tokens = {
        prompt_id: float(sum(float(row.get("trace_length") or 0.0) for row in group))
        for prompt_id, group in rows_by_prompt.items()
    }

    _status(f"Bootstrapping abstention deltas over {len(outcomes)} prompts")
    abstention = prompt_abstention_bootstrap(
        prompt_scores,
        outcomes,
        n_bootstrap=n_bootstrap,
        seed=seed,
        # Every region-matched output scorer becomes a baseline, so the report
        # carries geometry-minus-output deltas directly rather than by subtraction
        # of two independently bootstrapped intervals.
        baseline_methods=(
            "length",
            "logprob",
            "entropy",
            "conf_tail_q20_ent",
            "conf_tail_q20_lp",
            "conf_bottom10_group_ent",
            "vote_agreement",
        ),
    )

    # The same head-to-head with the monotone length component removed. E1R showed
    # the raw comparison and the length-orthogonal one can disagree, so the gate is
    # only decided when both agree.
    residual = length_residualized_abstention(
        prompt_scores,
        outcomes,
        methods=methods,
        control="length",
        n_bootstrap=n_bootstrap,
        seed=seed + 90000,
        comparisons=PRIMARY_COMPARISONS,
    )

    prompt_ids = sorted(outcomes)
    outcome_array = np.asarray([outcomes[pid] for pid in prompt_ids], dtype=float)
    calibration = {}
    decisions = {}
    for method in methods:
        values = np.asarray(
            [float(prompt_scores[pid].get(method, -np.inf)) for pid in prompt_ids], dtype=float
        )
        values = np.where(np.isfinite(values), values, np.nan)
        probabilities = _oof_probabilities(values, outcome_array, n_splits=n_splits, seed=seed)
        calibration[method] = calibration_metrics(probabilities, outcome_array)
        decisions[method] = decision_values(
            {pid: prompt_scores[pid].get(method, -np.inf) for pid in prompt_ids},
            outcomes,
            prompt_tokens,
        )

    gate = _gate_a(abstention, residual)

    result = {
        "model": model_label,
        "dataset": dataset_label,
        "layer": layer,
        "n_prompts": len(prompt_ids),
        "base_accuracy": float(outcome_array.mean()),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "methods": list(methods),
        "method_descriptions": {m: METHOD_DESCRIPTIONS.get(m, "") for m in methods},
        "confidence_approximation": (
            "DeepConf top-k token confidence is unrecoverable from cached data; "
            "_ent uses full-distribution entropy, _lp uses the sampled token's "
            "log-probability."
        ),
        "abstention": abstention,
        "length_residualized": residual,
        "calibration": calibration,
        "decision_values": decisions,
        "gate_a": gate,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset_label}_abstention_baselines"
    write_json(result, out / f"{prefix}_results.json")
    write_report(result, out / f"{prefix}_report.md")
    return result


def _gate_a(abstention: dict, residual: dict) -> dict:
    """Does geometry beat every region-matched output baseline, raw and residual?

    Reported as a verdict rather than left to the reader because the handoff makes
    this comparison load-bearing: a loss here means the wave-1 result is an
    output-signal result and the cross-architecture GPU spend is not justified.
    """
    raw_p = {}
    entries = {}
    for method, baseline in PRIMARY_COMPARISONS:
        key = f"{method}_minus_{baseline}"
        entry = abstention["deltas"].get(key, {}).get("aurc")
        residual_entry = residual.get("deltas", {}).get(key, {}).get("aurc")
        if entry is None:
            continue
        raw_p[key] = entry.get("p_two_sided")
        entries[key] = {
            "raw": entry,
            "length_residualized": residual_entry,
        }
    adjusted = holm_correction(raw_p)
    for key, entry in entries.items():
        entry["raw_p_holm"] = adjusted.get(key)
    wins = [
        key
        for key, entry in entries.items()
        if (entry["raw"].get("point_estimate") or 0.0) > 0
        and (entry["raw"].get("ci_low") or -1.0) > 0
    ]
    residual_wins = [
        key
        for key, entry in entries.items()
        if entry["length_residualized"]
        and (entry["length_residualized"].get("point_estimate") or 0.0) > 0
        and (entry["length_residualized"].get("ci_low") or -1.0) > 0
    ]
    return {
        "comparisons": entries,
        "n_comparisons": len(entries),
        "raw_wins": wins,
        "length_residualized_wins": residual_wins,
        "passes": bool(entries) and len(wins) == len(entries) and len(residual_wins) == len(entries),
    }


def write_json(result: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as handle:
        json.dump(result, handle, indent=2)


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _interval(entry: dict | None) -> str:
    if not entry:
        return "n/a"
    return (
        f"{_fmt(entry.get('point_estimate'))} "
        f"[{_fmt(entry.get('ci_low'))}, {_fmt(entry.get('ci_high'))}] "
        f"p={_fmt(entry.get('p_two_sided'))}"
    )


def write_report(result: dict, path: str | Path) -> None:
    lines = [
        f"# Abstention baselines — {result['model']} / {result['dataset']} (L{result['layer']})",
        "",
        f"{result['n_prompts']} prompts, base accuracy {_fmt(result['base_accuracy'])}, "
        f"{result['n_bootstrap']} paired prompt bootstrap draws.",
        "",
        f"*{result['confidence_approximation']}*",
        "",
        "## Gate A — geometry vs region-matched output confidence",
        "",
        "Verdict: **" + ("PASS" if result["gate_a"]["passes"] else "FAIL") + "**"
        " (geometry beats every region-matched output baseline on AURC, raw and"
        " with length partialled out).",
        "",
        "| comparison (AURC) | raw delta | Holm p | length-residualized delta |",
        "| --- | --- | --- | --- |",
    ]
    for key, entry in result["gate_a"]["comparisons"].items():
        lines.append(
            f"| {key} | {_interval(entry['raw'])} | {_fmt(entry.get('raw_p_holm'))} | "
            f"{_interval(entry['length_residualized'])} |"
        )

    lines += ["", "## Point estimates", "", "| method | AURC | acc@50% | acc@80% | ECE | Brier |", "| --- | --- | --- | --- | --- | --- |"]
    for method in result["methods"]:
        point = result["abstention"]["point"].get(method, {})
        coverage = point.get("accuracy_at_coverage", {})
        calibration = result["calibration"].get(method, {})
        lines.append(
            f"| {method} | {_fmt(point.get('aurc'))} | {_fmt(coverage.get('0.5'))} | "
            f"{_fmt(coverage.get('0.8'))} | {_fmt(calibration.get('ece'))} | "
            f"{_fmt(calibration.get('brier'))} |"
        )

    lines += ["", "## Decision values — selective accuracy and token savings", ""]
    rates = [str(rate) for rate in ABSTENTION_RATES]
    lines.append("| method | " + " | ".join(f"acc@abstain {rate}" for rate in rates) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in rates) + " |")
    for method in result["methods"]:
        table = result["decision_values"].get(method, {})
        cells = [_fmt(table.get(rate, {}).get("selective_accuracy")) for rate in rates]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += ["", "Token savings at each abstention rate (identical across methods only if "
              "trace lengths are uncorrelated with the score):", ""]
    lines.append("| method | " + " | ".join(f"save@abstain {rate}" for rate in rates) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in rates) + " |")
    for method in result["methods"]:
        table = result["decision_values"].get(method, {})
        cells = [_fmt(table.get(rate, {}).get("token_savings")) for rate in rates]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")

    lines += ["", "## Method definitions", ""]
    for method in result["methods"]:
        lines.append(f"- `{method}` — {result['method_descriptions'].get(method, '')}")
    lines.append("")
    Path(path).write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--oof_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_label", default="math500")
    parser.add_argument("--model_label", default="model")
    parser.add_argument("--layers", default="7,14,21")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_workers", type=int, default=4)
    parser.add_argument("--n_splits", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_abstention_baselines(
        data_dir=args.data_dir,
        oof_csv=args.oof_csv,
        output_dir=args.output_dir,
        dataset_label=args.dataset_label,
        model_label=args.model_label,
        layers=[int(value) for value in args.layers.split(",") if value.strip()],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        load_workers=args.load_workers,
        n_splits=args.n_splits,
    )
    _status(
        f"Gate A: {'PASS' if result['gate_a']['passes'] else 'FAIL'} "
        f"({len(result['gate_a']['raw_wins'])}/{result['gate_a']['n_comparisons']} raw wins, "
        f"{len(result['gate_a']['length_residualized_wins'])} residual wins)"
    )


if __name__ == "__main__":
    main()
