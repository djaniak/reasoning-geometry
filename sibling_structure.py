"""What kind of unfinished trace is a cap hit?

A capped trace is a censoring event, not a verdict: generation stopped because the
budget ran out, and nothing in the trace says whether the model was about to
finish.  Before asking what capping looks like geometrically, ask what it looks
like *combinatorially*, using only cached lengths and answers.

Each prompt was sampled eight times under one budget, so the siblings of a capped
trace say what the budget was up against:

prompt-limited
    most siblings cap.  The budget is short for this problem, and no single
    trajectory is at fault.
trajectory-limited
    one or two siblings cap while the rest finish, and the finishers are right.
    The budget was sufficient; that particular sample wandered.
budget-borderline
    siblings finish, but only just -- their lengths pile up against the cap.  A
    slightly larger budget would move traces across the line in both directions.

The three make different predictions about what a continuation study would find,
so separating them is what makes such a study worth running.  Nothing here is
fitted; every number is a count or a ratio over the cached out-of-fold table.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from trace_caps import resolve_cap

# A finisher this close to the budget had little room left: the prompt sits at the
# edge of what the budget affords rather than comfortably inside it.
BORDERLINE_FRACTION = 0.9
# "Most siblings cap" -- a strict majority of an eight-sample group.
PROMPT_LIMITED_CAPPED = 5


def load_rows(oof_csv: str | Path, layer: int | None = None) -> list[dict]:
    """Read one row per trace, from a single layer of the out-of-fold table.

    The table repeats every trace once per probed layer.  Lengths and answers are
    identical across layers, so any one layer is the trace population; taking all
    of them would multiply every count.
    """
    with open(oof_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {oof_csv}")
    layers = {int(row["layer"]) for row in rows}
    chosen = max(layers) if layer is None else int(layer)
    if chosen not in layers:
        raise ValueError(f"layer {chosen} not in {sorted(layers)} ({oof_csv})")
    return [
        {
            "prompt_id": int(row["prompt_id"]),
            "trace_length": int(float(row["trace_length"])),
            "is_correct": bool(int(float(row["is_correct"]))),
            "predicted_answer": row["predicted_answer"],
            "gold_answer": row["gold_answer"],
        }
        for row in rows
        if int(row["layer"]) == chosen
    ]


def _plurality_correct(group: list[dict]) -> bool | None:
    """Majority answer among parseable siblings; ``None`` if none parse."""
    votes = Counter(
        row["predicted_answer"] for row in group if row["predicted_answer"] not in (None, "")
    )
    if not votes:
        return None
    winner, _ = votes.most_common(1)[0]
    return winner == group[0]["gold_answer"]


def prompt_table(rows: list[dict], cap: int) -> list[dict]:
    """Per-prompt sibling accounting: who capped, who finished, who was right."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["prompt_id"]].append(row)

    table = []
    for prompt_id, group in sorted(grouped.items()):
        capped = [row for row in group if row["trace_length"] >= cap]
        finished = [row for row in group if row["trace_length"] < cap]
        finished_lengths = [row["trace_length"] for row in finished]
        table.append(
            {
                "prompt_id": prompt_id,
                "n_traces": len(group),
                "n_capped": len(capped),
                "n_finished": len(finished),
                "n_finished_correct": sum(row["is_correct"] for row in finished),
                "n_correct": sum(row["is_correct"] for row in group),
                "pass_rate": sum(row["is_correct"] for row in group) / len(group),
                "plurality_correct": _plurality_correct(group),
                # Plurality restricted to finishers, which is what a longer budget
                # would be voting over if every capped sibling were discarded.
                "finished_plurality_correct": (
                    _plurality_correct(finished) if finished else None
                ),
                "max_finished_length": max(finished_lengths) if finished else None,
                "longest_finisher_fraction": (
                    max(finished_lengths) / cap if finished else None
                ),
            }
        )
    return table


def _regime(prompt: dict) -> str:
    """Label a prompt that has at least one capped sibling."""
    if prompt["n_capped"] >= PROMPT_LIMITED_CAPPED:
        return "prompt_limited"
    fraction = prompt["longest_finisher_fraction"]
    if fraction is not None and fraction >= BORDERLINE_FRACTION:
        return "budget_borderline"
    if prompt["n_finished_correct"] > 0:
        return "trajectory_limited"
    return "unresolved"


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def sibling_report(rows: list[dict], cap: int) -> dict:
    """Answer the sibling-structure questions from counts alone."""
    table = prompt_table(rows, cap)
    affected = [prompt for prompt in table if prompt["n_capped"] > 0]
    all_capped = [prompt for prompt in table if prompt["n_capped"] == prompt["n_traces"]]

    # Correctness of the finishers, by how many of their siblings capped. This is
    # the sharpest read on whether capping marks a hard prompt or a lost sample.
    by_capped_count: dict[int, dict] = {}
    for count in range(0, 9):
        bucket = [prompt for prompt in table if prompt["n_capped"] == count]
        finished = sum(prompt["n_finished"] for prompt in bucket)
        by_capped_count[count] = {
            "n_prompts": len(bucket),
            "n_finished_traces": finished,
            "finished_accuracy": _rate(
                sum(prompt["n_finished_correct"] for prompt in bucket), finished
            ),
            "plurality_accuracy": _rate(
                sum(bool(prompt["plurality_correct"]) for prompt in bucket), len(bucket)
            ),
            "finished_plurality_accuracy": _rate(
                sum(bool(prompt["finished_plurality_correct"]) for prompt in bucket),
                sum(prompt["finished_plurality_correct"] is not None for prompt in bucket),
            ),
        }

    regimes = Counter(_regime(prompt) for prompt in affected)

    def _fractions(prompts: list[dict]) -> list[float]:
        return sorted(
            prompt["longest_finisher_fraction"]
            for prompt in prompts
            if prompt["longest_finisher_fraction"] is not None
        )

    # The control the borderline label needs: if prompts with no capped sibling
    # also press against the budget, "borderline" is describing the whole dataset
    # rather than anything about capping.
    unaffected = [prompt for prompt in table if prompt["n_capped"] == 0]
    return {
        "max_new_tokens": cap,
        "n_prompts": len(table),
        "n_traces": len(rows),
        "n_capped_traces": sum(prompt["n_capped"] for prompt in table),
        "n_prompts_with_a_capped_sibling": len(affected),
        "n_prompts_all_capped": len(all_capped),
        # The question a continuation study depends on: when one sibling runs out
        # of budget, does any other one get there?
        "p_a_sibling_finishes_given_a_cap": _rate(
            sum(prompt["n_finished"] > 0 for prompt in affected), len(affected)
        ),
        "p_a_finisher_is_correct_given_a_cap": _rate(
            sum(prompt["n_finished_correct"] > 0 for prompt in affected), len(affected)
        ),
        "capped_sibling_count_distribution": {
            str(count): sum(1 for prompt in table if prompt["n_capped"] == count)
            for count in range(0, 9)
        },
        "by_capped_sibling_count": by_capped_count,
        "regimes": {
            name: {
                "n_prompts": regimes.get(name, 0),
                "share_of_affected": _rate(regimes.get(name, 0), len(affected)),
            }
            for name in (
                "prompt_limited",
                "trajectory_limited",
                "budget_borderline",
                "unresolved",
            )
        },
        "regime_definitions": {
            "prompt_limited": f"n_capped >= {PROMPT_LIMITED_CAPPED} of eight",
            "budget_borderline": (
                f"not prompt-limited, and the longest finishing sibling used "
                f">= {BORDERLINE_FRACTION:.0%} of the budget"
            ),
            "trajectory_limited": (
                "not prompt-limited or borderline, and at least one sibling "
                "finished correctly"
            ),
            "unresolved": "capped siblings, finishers well inside budget, none correct",
        },
        "longest_finisher_fraction_percentiles": _percentiles(_fractions(affected)),
        "longest_finisher_fraction_percentiles_uncapped_prompts": _percentiles(
            _fractions(unaffected)
        ),
        "unconditional_accuracy": _rate(
            sum(prompt["n_correct"] for prompt in table), len(rows)
        ),
        "capped_trace_accuracy": _rate(
            sum(row["is_correct"] for row in rows if row["trace_length"] >= cap),
            sum(row["trace_length"] >= cap for row in rows),
        ),
    }


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        str(q): round(values[min(len(values) - 1, int(q / 100 * len(values)))], 4)
        for q in (10, 25, 50, 75, 90)
    }


def format_report(report: dict, label: str) -> str:
    lines = [
        f"# Sibling structure of budget-limited noncompletion -- {label}",
        "",
        f"Budget {report['max_new_tokens']} tokens; "
        f"{report['n_traces']} traces over {report['n_prompts']} prompts; "
        f"{report['n_capped_traces']} capped.",
        "",
        f"- prompts with >=1 capped sibling: {report['n_prompts_with_a_capped_sibling']}",
        f"- prompts where all eight cap: {report['n_prompts_all_capped']}",
        f"- P(some sibling finishes | a sibling capped): "
        f"{report['p_a_sibling_finishes_given_a_cap']}",
        f"- P(some finisher is correct | a sibling capped): "
        f"{report['p_a_finisher_is_correct_given_a_cap']}",
        "",
        "## Capped siblings per prompt",
        "",
        "| capped | prompts | finished traces | finished acc | plurality acc |",
        "| --- | --- | --- | --- | --- |",
    ]
    for count, block in report["by_capped_sibling_count"].items():
        lines.append(
            f"| {count} | {block['n_prompts']} | {block['n_finished_traces']} | "
            f"{block['finished_accuracy']} | {block['plurality_accuracy']} |"
        )
    lines += ["", "## Regimes among prompts with a capped sibling", "",
              "| regime | prompts | share |", "| --- | --- | --- |"]
    for name, block in report["regimes"].items():
        lines.append(f"| {name} | {block['n_prompts']} | {block['share_of_affected']} |")
    lines += [
        "",
        "## How close finishers run to the budget",
        "",
        f"- prompts with a capped sibling: "
        f"{report['longest_finisher_fraction_percentiles']}",
        f"- prompts with none:              "
        f"{report['longest_finisher_fraction_percentiles_uncapped_prompts']}",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof_csv", required=True)
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Trace directory, used to look the generation budget up in dvc.lock.",
    )
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--layer", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.oof_csv, args.layer)
    cap = resolve_cap(
        args.max_new_tokens,
        data_dir=args.data_dir,
        lengths=(row["trace_length"] for row in rows),
        context="sibling_structure",
    )
    report = sibling_report(rows, cap.value)
    report["model"] = args.model_label
    report["cap_provenance"] = cap.provenance
    text = format_report(report, args.model_label)
    print(text)
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "math500_sibling_structure_results.json").write_text(
            json.dumps(report, indent=2)
        )
        (out / "math500_sibling_structure_report.md").write_text(text)


if __name__ == "__main__":
    main()
