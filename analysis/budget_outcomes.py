"""Budget-indexed outcome table: what the RMD conclusion is conditional on.

The paper's headline population is ``cap_free_valid_plurality`` -- prompts where
no sibling trace hit the generation budget.  Notebook 14 calls the excluded rows
censored and treats dropping them as missing-data handling.  That is not valid.
Capping is related to prompt difficulty, so complete-case filtering estimates
correctness *conditional on avoiding the cap*, not correctness under a larger
budget.  Base accuracy moves 0.62 -> 0.69 (Qwen) across that filter; the
population moves 500 -> 392.

Two quantities answer different questions, and neither is "eventual correctness":

``C_B``
    whether a correct, assessable answer is available by budget ``B`` under the
    fixed decoding and extraction configuration.  No answer by ``B`` is an
    observed failure, not a missing value.  This is the ``full_population``
    outcome that ``incremental_abstention`` already computes: the plurality is
    taken over whatever parsed, and an all-unparsed prompt scores zero.
``C_{B->B'}``
    whether the stored prefix is correct after continuing from ``B`` to a larger
    budget ``B'``.  Measured only for the 50 sampled DeepSeek traces in
    ``continue_capped``.  It is a one-model case study, not a label for the
    dataset.

This module reads the committed result JSONs and reports the selection ladder
side by side, so the sensitivity of the headline to the outcome definition is
visible rather than implied.  It fits nothing and runs no model.

A capped trace that still carries a parseable answer is scored at ``B``: its
stopping time is censored but its answer is observed.  ``parseable_at_cap``
counts those traces, which the "censored" framing silently discards.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from applications.incremental_abstention import is_parseable_answer

MODELS = ("qwen", "deepseek", "deepseek_llama")

# Ordered widest-to-narrowest.  Each step is a selection, and the report exists
# to show what each one buys and what it conditions on.
POPULATION_LADDER = (
    ("full_population", "C_B on all 500 prompts; unparsed scores 0"),
    ("valid_plurality", "drops prompts with no parseable sibling"),
    ("cap_free_valid_plurality", "headline; also drops any prompt with a capped sibling"),
    ("all_eight_parseable", "all 8 siblings parsed; not cap-filtered"),
)

CONTRAST = "B1_minus_B0_aurc"


def result_path(model: str, results_root: Path) -> Path:
    return results_root / f"{model}_bestofn_full" / "math500" / "math500_incremental_abstention_results.json"


def oof_path(model: str, results_root: Path) -> Path:
    return results_root / f"{model}_bestofn_full" / "math500" / "math500_prompt_decomposition_oof.csv"


def load_result(model: str, results_root: Path) -> dict:
    path = result_path(model, results_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is absent. This table is built from committed results, not "
            "from a refit; run applications/incremental_abstention.py for {model} first."
        )
    return json.loads(path.read_text())


def _delta(body: Mapping, contrast: str = CONTRAST) -> dict:
    """Point estimate and interval for one paired contrast, or NaNs if absent."""
    entry = (body.get("paired_deltas") or {}).get(contrast) or {}
    return {
        "estimate": entry.get("point_estimate", float("nan")),
        "ci_low": entry.get("ci_low", float("nan")),
        "ci_high": entry.get("ci_high", float("nan")),
        "p_two_sided": entry.get("p_two_sided", float("nan")),
    }


def population_table(results: Mapping[str, dict]) -> pd.DataFrame:
    """One row per (model, population): accounting plus the B1-B0 AURC delta."""
    rows = []
    for model, result in results.items():
        populations = result["populations"]
        reference = populations["full_population"]["n_prompts"]
        for name, definition in POPULATION_LADDER:
            body = populations.get(name)
            if body is None:
                continue
            delta = _delta(body)
            rows.append(
                {
                    "model": model,
                    "population": name,
                    "definition": definition,
                    "n_prompts": body["n_prompts"],
                    "retained": body["n_prompts"] / reference,
                    "base_accuracy": body["base_accuracy"],
                    "n_capped_prompts": body.get("n_capped_prompts"),
                    "n_automatic_failures": body.get("n_automatic_failures"),
                    "n_unparsed_traces": body.get("n_unparsed_traces"),
                    "aurc_b0": (body.get("models", {}).get("B0", {}).get("metrics", {}) or {}).get("aurc"),
                    "aurc_b1": (body.get("models", {}).get("B1", {}).get("metrics", {}) or {}).get("aurc"),
                    **{f"delta_{key}": value for key, value in delta.items()},
                    "excludes_zero": bool(delta["ci_high"] < 0 or delta["ci_low"] > 0),
                }
            )
    return pd.DataFrame(rows)


def cap_accounting(model: str, results_root: Path, cap: int) -> dict:
    """Trace-level cap accounting, including capped traces that still answered.

    OOF rows are per ``(trace, layer)``; the layer sweep would triple every count,
    so rows are deduplicated on ``trace_id`` before anything is counted.
    """
    frame = pd.read_csv(oof_path(model, results_root))
    traces = frame.drop_duplicates(subset="trace_id")
    at_cap = traces["trace_length"] >= cap
    # A missing answer reaches the repo's own accounting as ``None``, but through
    # pandas it is ``float('nan')`` -- for which ``is_parseable_answer`` is True,
    # since ``str(nan).strip()`` is the non-empty ``"nan"``.  Screen NaN out first
    # or every capped trace is counted as having answered.
    answers = traces["predicted_answer"]
    parseable = answers.notna() & answers.map(lambda value: is_parseable_answer(value))
    return {
        "n_traces": int(len(traces)),
        "n_capped": int(at_cap.sum()),
        "n_capped_parseable": int((at_cap & parseable).sum()),
        "n_capped_unparsed": int((at_cap & ~parseable).sum()),
        "n_uncapped_unparsed": int((~at_cap & ~parseable).sum()),
        "capped_parseable_accuracy": _accuracy(traces[at_cap & parseable]),
        "uncapped_parseable_accuracy": _accuracy(traces[~at_cap & parseable]),
    }


def _accuracy(frame: pd.DataFrame) -> float:
    return float(frame["is_correct"].mean()) if len(frame) else float("nan")


def continuation_case(results_root: Path) -> dict | None:
    """The DeepSeek C_{B->B'} case study, reported on its own terms."""
    path = results_root / "deepseek_bestofn_full" / "math500" / "math500_continue_capped_results.json"
    if not path.is_file():
        return None
    body = json.loads(path.read_text())
    outcomes = body["outcomes"]
    completed = outcomes["completed_correct"] + outcomes["completed_incorrect"]
    return {
        **body,
        "n_completed": completed,
        # Two different denominators, both defensible, easy to confuse.
        # ``accuracy_of_completions`` in continue_capped divides by traces that
        # *terminated*, which includes a degenerate loop that ran to a stop --
        # it is in the denominator and can never be correct, so it deflates the
        # rate.  The recomputed value divides by traces labelled
        # completed_correct/completed_incorrect.  Carry both and name which is
        # which; do not quote one as the other.
        "accuracy_of_completions_recomputed": (
            outcomes["completed_correct"] / completed if completed else float("nan")
        ),
        "accuracy_of_completions_as_stored": body.get("accuracy_of_completions"),
    }


def build(results_root: Path) -> dict:
    results = {model: load_result(model, results_root) for model in MODELS}
    table = population_table(results)
    caps = {
        model: {
            "max_new_tokens": result["max_new_tokens"],
            "cap_provenance": result["cap_provenance"],
            "layer": result.get("layer"),
            **cap_accounting(model, results_root, int(result["max_new_tokens"])),
        }
        for model, result in results.items()
    }
    return {
        "populations": table.to_dict(orient="records"),
        "cap_accounting": caps,
        "continuation_case_study": continuation_case(results_root),
    }


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(payload: Mapping, path: Path) -> None:
    table = pd.DataFrame(payload["populations"])
    lines = [
        "# Budget-indexed outcomes for the RMD prompt-level result",
        "",
        "Built by `analysis/budget_outcomes.py` from committed result JSONs. No refit.",
        "",
        "`C_B` is correctness available by the generation budget under the fixed",
        "decoding and extraction rule. It is the `full_population` row: an unparsed",
        "prompt is an observed failure, not a missing value. Every row below it is a",
        "conditional population, and `cap_free_valid_plurality` -- the current headline",
        "-- conditions on a difficulty-related event.",
        "",
        "## 1. Selection ladder",
        "",
        "| Model | Population | n | Retained | Base acc. | Capped prompts | Auto-failures |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples():
        lines.append(
            f"| {row.model} | `{row.population}` | {row.n_prompts} | {row.retained:.1%} | "
            f"{row.base_accuracy:.3f} | {row.n_capped_prompts} | {row.n_automatic_failures} |"
        )

    lines += [
        "",
        "## 2. Does the increment survive the outcome definition?",
        "",
        "Paired bootstrap `B1 - B0` on AURC. Negative favours B1 (RMD added to the",
        "output-side baseline). These intervals hold the fitted pipeline fixed and so",
        "understate uncertainty; see the outer-refit blocker.",
        "`vs headline` is the increment as a fraction of the `cap_free_valid_plurality`",
        "estimate. Below 100% means the headline population overstates the increment.",
        "",
        "| Model | Population | B1-B0 AURC | 95% CI | p | CI excludes 0 | vs headline |",
        "|---|---|---:|---|---:|:--:|---:|",
    ]
    headline = {
        row.model: row.delta_estimate
        for row in table.itertuples()
        if row.population == "cap_free_valid_plurality"
    }
    for row in table.itertuples():
        mark = "yes" if row.excludes_zero else "no"
        reference = headline.get(row.model)
        share = (
            f"{row.delta_estimate / reference:.0%}"
            if reference not in (None, 0) and not np.isnan(reference)
            else "—"
        )
        lines.append(
            f"| {row.model} | `{row.population}` | {_fmt(row.delta_estimate)} | "
            f"[{_fmt(row.delta_ci_low)}, {_fmt(row.delta_ci_high)}] | "
            f"{_fmt(row.delta_p_two_sided)} | {mark} | {share} |"
        )

    lines += [
        "",
        "## 3. Capped traces that still carry an answer",
        "",
        "Their stopping time is censored; their answer at `B` is observed and is",
        "scored. Dropping them treats an observed outcome as missing.",
        "",
        "| Model | Cap | Provenance | Traces | Capped | Capped & parseable | Capped & unparsed | Acc. capped-parseable | Acc. uncapped-parseable |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, body in payload["cap_accounting"].items():
        lines.append(
            f"| {model} | {body['max_new_tokens']} | {body['cap_provenance']} | "
            f"{body['n_traces']} | {body['n_capped']} | {body['n_capped_parseable']} | "
            f"{body['n_capped_unparsed']} | {_fmt(body['capped_parseable_accuracy'], 3)} | "
            f"{_fmt(body['uncapped_parseable_accuracy'], 3)} |"
        )

    case = payload.get("continuation_case_study")
    lines += ["", "## 4. Continuation case study, C_{B->B'}", ""]
    if case is None:
        lines.append("No continuation result on disk.")
    else:
        settings = case["settings"]
        lines += [
            f"DeepSeek only. {case['n_continued']} traces sampled from the capped",
            f"population and resumed from {settings['original_cap']} for a further",
            f"{settings['extra_tokens']} tokens at temperature {settings['temperature']}.",
            "This is a one-model sensitivity case. It is not a label for the other two",
            "models, for the unsampled capped traces, or for the dataset.",
            "",
            "| Outcome | n | Share |",
            "|---|---:|---:|",
        ]
        for name, count in case["outcomes"].items():
            lines.append(f"| {name} | {count} | {count / case['n_continued']:.1%} |")
        lines += [
            "",
            f"Of the {case['n_continued']} resumed traces, {case['n_completed']} are "
            "labelled completed (correct or incorrect), and "
            f"{_fmt(case['accuracy_of_completions_recomputed'], 3)} of those are correct.",
            "",
            f"The stored `accuracy_of_completions` is "
            f"{_fmt(case['accuracy_of_completions_as_stored'], 4)}, which is a different "
            "quantity: `continue_capped` divides by traces that *terminated*, and a "
            "degenerate loop that ran to a stop sits in that denominator while never "
            "counting as correct. Both are defensible; say which one is being quoted.",
        ]

    lines += [
        "",
        "## Reporting rule",
        "",
        "1. Report `full_population` (`C_B`) as the primary outcome.",
        "2. Report cap-free numbers as conditional, with the retained fraction beside them.",
        "3. Report the continuation study separately, as DeepSeek-only evidence about",
        "   what capped prefixes do next -- never as `C_B` for any population.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", type=Path)
    parser.add_argument("--output", default="results/budget_outcomes", type=Path)
    args = parser.parse_args()

    payload = build(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "budget_outcomes.json").write_text(json.dumps(payload, indent=2, default=float))
    write_report(payload, args.output / "budget_outcomes_report.md")
    print(f"wrote {args.output}/budget_outcomes.json and budget_outcomes_report.md")


if __name__ == "__main__":
    main()
