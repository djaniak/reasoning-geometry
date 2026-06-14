"""Evaluate prompt-level selectors from saved out-of-fold trace scores."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from prompt_decomposition import SCORE_METHODS, available_score_methods

INVALID_ANSWER = "<INVALID>"


def _group_rows(rows: list[dict], key: str) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row[key])].append(row)
    return dict(grouped)


def rank_weights(scores: list[float]) -> list[float]:
    """Return average ranks with the largest confidence receiving the largest rank."""
    return rankdata(np.asarray(scores, dtype=float), method="average").tolist()


def _logprob_value(row: dict) -> float:
    value = row.get("mean_logprob")
    return float(value) if value is not None else float("-inf")


def select_top_trace(rows: list[dict], score_key: str) -> dict:
    return max(
        rows,
        key=lambda row: (
            float(row[score_key]),
            _logprob_value(row),
            -int(row["sample_id"]),
            -int(row["trace_id"]),
        ),
    )


def _answer_value(row: dict) -> str:
    answer = row.get("predicted_answer")
    return str(answer) if answer not in (None, "") else INVALID_ANSWER


def _answer_tiebreak_key(
    answer: str,
    rows: list[dict],
) -> tuple[float, int, str]:
    best_logprob = max((_logprob_value(row) for row in rows), default=float("-inf"))
    return best_logprob, int(answer != INVALID_ANSWER), str(answer)


def majority_answer(rows: list[dict]) -> str | None:
    if not rows:
        return None
    counts = Counter(_answer_value(row) for row in rows)
    by_answer = {
        answer: [row for row in rows if _answer_value(row) == answer]
        for answer in counts
    }
    return max(
        counts,
        key=lambda answer: (
            counts[answer],
            *_answer_tiebreak_key(answer, by_answer[answer]),
        ),
    )


def rmd_weighted_answer(rows: list[dict]) -> str | None:
    if not rows:
        return None
    weights = rank_weights([float(row["rmd_score"]) for row in rows])
    totals: dict[str, float] = defaultdict(float)
    by_answer: dict[str, list[dict]] = defaultdict(list)
    for row, weight in zip(rows, weights):
        answer = _answer_value(row)
        totals[answer] += float(weight)
        by_answer[answer].append(row)
    return max(
        totals,
        key=lambda answer: (
            totals[answer],
            *_answer_tiebreak_key(answer, by_answer[answer]),
        ),
    )


def majority_rmd_tiebreak_answer(rows: list[dict]) -> str | None:
    if not rows:
        return None
    weights = rank_weights([float(row["rmd_score"]) for row in rows])
    counts: Counter[str] = Counter()
    totals: dict[str, float] = defaultdict(float)
    by_answer: dict[str, list[dict]] = defaultdict(list)
    for row, weight in zip(rows, weights):
        answer = _answer_value(row)
        counts[answer] += 1
        totals[answer] += float(weight)
        by_answer[answer].append(row)
    return max(
        counts,
        key=lambda answer: (
            counts[answer],
            totals[answer],
            *_answer_tiebreak_key(answer, by_answer[answer]),
        ),
    )


def _answer_outcome(rows: list[dict], answer: str | None) -> float:
    if answer in (None, INVALID_ANSWER):
        return 0.0
    gold = rows[0].get("gold_answer")
    if gold not in (None, ""):
        return float(str(answer) == str(gold))
    matching = [
        row for row in rows if str(row.get("predicted_answer")) == str(answer)
    ]
    return float(any(int(row["is_correct"]) for row in matching))


def _answer_parsing_diagnostics(groups: dict[int, list[dict]]) -> dict:
    rows = [row for group in groups.values() for row in group]
    parsed = [
        row for row in rows
        if row.get("predicted_answer") not in (None, "")
    ]
    correct = [row for row in rows if int(row["is_correct"])]
    incorrect = [row for row in rows if not int(row["is_correct"])]

    def parse_rate(subset: list[dict]) -> float | None:
        if not subset:
            return None
        return float(
            np.mean([
                row.get("predicted_answer") not in (None, "")
                for row in subset
            ])
        )

    return {
        "n_traces": len(rows),
        "n_parsed": len(parsed),
        "parse_rate": parse_rate(rows),
        "correct_parse_rate": parse_rate(correct),
        "incorrect_parse_rate": parse_rate(incorrect),
        "n_prompts_without_parsed_answer": sum(
            not any(
                row.get("predicted_answer") not in (None, "")
                for row in group
            )
            for group in groups.values()
        ),
    }


def _bootstrap_interval(
    outcomes: dict[int, float],
    n_bootstrap: int,
    seed: int,
) -> dict:
    values = np.asarray([outcomes[key] for key in sorted(outcomes)], dtype=float)
    if not len(values) or n_bootstrap <= 0:
        return {"ci_low": None, "ci_high": None, "n_valid": 0}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(values), size=len(values))
        draws[index] = float(values[sample].mean())
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "ci_low": float(low),
        "ci_high": float(high),
        "n_valid": int(n_bootstrap),
    }


def _pack_selector(
    outcomes: dict[int, float],
    n_bootstrap: int,
    seed: int,
) -> dict:
    return {
        "pass_at_1": float(np.mean(list(outcomes.values()))),
        "n_prompts": len(outcomes),
        "confidence_interval": _bootstrap_interval(
            outcomes, n_bootstrap=n_bootstrap, seed=seed
        ),
        "prompt_outcomes": {
            str(prompt_id): float(value)
            for prompt_id, value in sorted(outcomes.items())
        },
    }


def evaluate_prompt_selection(
    rows: list[dict],
    model: str,
    dataset: str,
    n_bootstrap: int,
    seed: int,
) -> dict:
    layers = sorted({int(row["layer"]) for row in rows})
    result = {
        "model": model,
        "dataset": dataset,
        "settings": {
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
            "score_orientation": "higher predicts correctness",
            "rmd_vote_weight": "average within-prompt rank",
            "invalid_answer_policy": "count as failure",
        },
        "layers": {},
    }

    for layer in layers:
        layer_rows = [row for row in rows if int(row["layer"]) == layer]
        groups = _group_rows(layer_rows, "prompt_id")
        methods = available_score_methods(layer_rows)
        outcomes: dict[str, dict[int, float]] = {
            "random": {},
            "oracle_pass_at_n": {},
            "majority_vote": {},
            "rmd_rank_weighted_vote": {},
            "majority_rmd_tiebreak": {},
        }
        for method in methods:
            outcomes[f"top1_{method}"] = {}

        for prompt_id, group in sorted(groups.items()):
            outcomes["random"][prompt_id] = float(
                np.mean([int(row["is_correct"]) for row in group])
            )
            outcomes["oracle_pass_at_n"][prompt_id] = float(
                any(int(row["is_correct"]) for row in group)
            )
            outcomes["majority_vote"][prompt_id] = _answer_outcome(
                group, majority_answer(group)
            )
            outcomes["rmd_rank_weighted_vote"][prompt_id] = _answer_outcome(
                group, rmd_weighted_answer(group)
            )
            outcomes["majority_rmd_tiebreak"][prompt_id] = _answer_outcome(
                group, majority_rmd_tiebreak_answer(group)
            )
            for method in methods:
                selected = select_top_trace(group, f"{method}_score")
                outcomes[f"top1_{method}"][prompt_id] = float(
                    selected["is_correct"]
                )

        result["layers"][str(layer)] = {
            "n_prompts": len(groups),
            "n": len(next(iter(groups.values()))) if groups else 0,
            "answer_parsing": _answer_parsing_diagnostics(groups),
            "selectors": {
                name: _pack_selector(
                    values,
                    n_bootstrap=n_bootstrap,
                    seed=seed + layer,
                )
                for name, values in outcomes.items()
            },
        }
    return result


def read_oof_csv(path: str | Path) -> list[dict]:
    int_fields = {
        "prompt_id",
        "trace_id",
        "sample_id",
        "is_correct",
        "fold",
        "layer",
        "trace_length",
    }
    float_fields = {
        "mean_logprob",
        *(f"{method}_score" for method in SCORE_METHODS),
    }
    rows = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "prompt_id",
            "trace_id",
            "sample_id",
            "is_correct",
            "fold",
            "layer",
            "predicted_answer",
            "gold_answer",
            "mean_logprob",
            "trace_length",
            *(f"{method}_score" for method in SCORE_METHODS),
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Prompt selection requires the enriched prompt decomposition "
                f"CSV; missing columns: {sorted(missing)}"
            )
        for raw in reader:
            row = dict(raw)
            for field in int_fields:
                if field in row and row[field] != "":
                    row[field] = int(row[field])
            for field in float_fields:
                if field in row:
                    row[field] = float(row[field]) if row[field] != "" else None
            for field in ("predicted_answer", "gold_answer"):
                if field in row and row[field] == "":
                    row[field] = None
            rows.append(row)
    return rows


def write_markdown(result: dict, path: str | Path) -> None:
    lines = [
        f"# {result['model']} {result['dataset']} prompt selection",
        "",
        "Unparsed answers are counted as an explicit invalid output. They are "
        "not silently removed from majority or weighted voting.",
        "",
        "## Answer parsing",
        "",
        "| Layer | Parsed traces | Parse rate | Correct parse rate "
        "| Incorrect parse rate | Prompts with no parsed answer |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, layer_result in result["layers"].items():
        parsing = layer_result["answer_parsing"]
        lines.append(
            f"| {layer} | {parsing['n_parsed']}/{parsing['n_traces']} "
            f"| {parsing['parse_rate']:.3f} "
            f"| {parsing['correct_parse_rate']:.3f} "
            f"| {parsing['incorrect_parse_rate']:.3f} "
            f"| {parsing['n_prompts_without_parsed_answer']} |"
        )
    lines.extend([
        "",
        "## Results",
        "",
        "| Layer | Selector | Pass@1 | 95% CI | Prompts |",
        "|---:|:---|---:|:---|---:|",
    ])
    for layer, layer_result in result["layers"].items():
        for selector, payload in layer_result["selectors"].items():
            interval = payload["confidence_interval"]
            ci = (
                "NA"
                if interval["ci_low"] is None
                else f"[{interval['ci_low']:.3f}, {interval['ci_high']:.3f}]"
            )
            lines.append(
                f"| {layer} | {selector} | {payload['pass_at_1']:.3f} "
                f"| {ci} | {payload['n_prompts']} |"
            )
    Path(path).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_label", default="math500")
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_oof_csv(args.input_csv)
    result = evaluate_prompt_selection(
        rows,
        model=args.model_label,
        dataset=args.dataset_label,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.dataset_label}_prompt_selection"
    (output_dir / f"{prefix}_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_markdown(result, output_dir / f"{prefix}_report.md")


if __name__ == "__main__":
    main()
