"""Resume capped traces past their budget and see what they were doing.

A cap hit tells us generation stopped, not that reasoning failed.  The sibling
accounting in ``sibling_structure.py`` shows the capped population splits into
prompts where the budget was short, prompts where one sample wandered, and
prompts sitting right at the edge -- but no arrangement of cached lengths can say
whether a given capped trace was two hundred tokens from an answer or would never
have produced one.

So continue them.  Each selected trace is re-fed to the model as prompt plus its
own 8192 generated tokens and allowed to keep going to a larger budget, under the
temperature it was sampled at.  What comes out separates the possibilities that
"non-convergence" would otherwise blur together:

completed_correct / completed_incorrect
    the trace terminates.  It was slow, not stuck; the budget was the binding
    constraint and a larger one recovers the answer (or a wrong one).
still_unfinished
    it terminates nowhere even at the larger budget.  Failure to converge.
degenerate_loop
    it collapses into repetition.  Failure to terminate.

``answered_then_continued`` is orthogonal: a trace that emitted a boxed answer and
kept generating was never budget-limited at all, only bad at stopping.

Continuation cannot reproduce the original sampling stream -- the RNG state is
gone -- so this measures what the model does next from that prefix, which is the
question, rather than what it did on the day.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from controls.loop_precursors import DEGENERATE_PERIODICITY, tail_periodicity
from data.trace_caps import resolve_cap

# A trace that boxes an answer and then generates more than this is not short of
# budget; it is failing to stop.  One short paragraph of wrap-up is normal.
WRAP_UP_TOKENS = 200


def capped_traces(data_dir: str | Path, cap: int) -> list[dict]:
    """Index every capped trace without loading hidden states or tokens.

    Token arrays are keyed by the globally-numbered ``trace_id``, not by position
    within the batch, and a trace whose tokens were never stored cannot be
    resumed, so it is skipped rather than silently continued from nothing.
    """
    found = []
    for path in sorted(Path(data_dir).glob("batch_*.npz")):
        with np.load(path, allow_pickle=True) as batch:
            stored = set(batch.files)
            for meta in batch["metadata"]:
                trace_id = int(meta["trace_id"])
                if int(meta["n_tokens"]) >= cap and f"tokens_{trace_id}" in stored:
                    found.append(
                        {
                            "batch": str(path),
                            "trace_id": trace_id,
                            "prompt_id": int(meta["idx"]),
                            "sample_id": int(meta["sample_id"]),
                            "n_tokens": int(meta["n_tokens"]),
                            "gold": str(meta["gold"]),
                            "was_correct": bool(meta["is_correct"]),
                        }
                    )
    return found


def load_tokens(record: dict) -> list[str]:
    with np.load(record["batch"], allow_pickle=True) as batch:
        return list(batch[f"tokens_{record['trace_id']}"])


def select_traces(
    records: list[dict],
    n: int,
    seed: int,
    *,
    degenerate_periodicity: float = DEGENERATE_PERIODICITY,
) -> tuple[list[dict], list[dict]]:
    """Sample ``n`` coherent capped traces; return them and the ones excluded.

    Traces already looping are dropped rather than continued: what they do next
    is known, and spending budget to confirm it would tell us nothing.  There are
    only a handful, so this barely perturbs the sample.
    """
    coherent, degenerate = [], []
    for record in records:
        _, periodicity = tail_periodicity(load_tokens(record))
        entry = dict(record, tail_periodicity=round(float(periodicity), 4))
        (degenerate if periodicity >= degenerate_periodicity else coherent).append(entry)
    rng = random.Random(seed)
    return rng.sample(coherent, min(n, len(coherent))), degenerate


def classify(
    continuation_tokens: list[str],
    continuation_text: str,
    terminated: bool,
    gold: str,
    *,
    extract_answer,
    normalize,
) -> dict:
    """Name what the trace did once it was allowed to keep going."""
    _, periodicity = tail_periodicity(continuation_tokens)
    answer = extract_answer(continuation_text)
    normalized = normalize(answer) if answer else None
    correct = bool(normalized is not None and normalized == gold)

    if periodicity >= DEGENERATE_PERIODICITY:
        outcome = "degenerate_loop"
    elif not terminated:
        outcome = "still_unfinished"
    elif correct:
        outcome = "completed_correct"
    else:
        outcome = "completed_incorrect"

    # Tokens generated after the answer was already in hand.
    tail_after_answer = None
    if answer is not None:
        closing = continuation_text.rfind(answer)
        if closing >= 0:
            after = continuation_text[closing + len(answer):]
            tail_after_answer = len(after) // 4  # ~4 chars/token, only a magnitude
    return {
        "outcome": outcome,
        "terminated": bool(terminated),
        "continuation_tokens": len(continuation_tokens),
        "tail_periodicity": round(float(periodicity), 4),
        "predicted": normalized,
        "correct": correct,
        "answered_then_continued": bool(
            answer is not None
            and tail_after_answer is not None
            and tail_after_answer > WRAP_UP_TOKENS
        ),
    }


def summarize(results: list[dict], excluded: list[dict]) -> dict:
    outcomes = Counter(entry["outcome"] for entry in results)
    completed = [entry for entry in results if entry["terminated"]]
    extra = sorted(entry["continuation_tokens"] for entry in completed)
    return {
        "n_continued": len(results),
        "n_excluded_as_already_degenerate": len(excluded),
        "outcomes": {name: outcomes.get(name, 0) for name in (
            "completed_correct",
            "completed_incorrect",
            "still_unfinished",
            "degenerate_loop",
        )},
        "outcome_shares": {
            name: round(outcomes.get(name, 0) / len(results), 4) if results else None
            for name in (
                "completed_correct",
                "completed_incorrect",
                "still_unfinished",
                "degenerate_loop",
            )
        },
        "n_answered_then_continued": sum(
            entry["answered_then_continued"] for entry in results
        ),
        "accuracy_of_completions": (
            round(sum(entry["correct"] for entry in completed) / len(completed), 4)
            if completed
            else None
        ),
        # The budget-engineering number: how much further the finishers needed.
        "extra_tokens_to_finish_percentiles": (
            {
                str(q): int(extra[min(len(extra) - 1, int(q / 100 * len(extra)))])
                for q in (10, 25, 50, 75, 90)
            }
            if extra
            else None
        ),
    }


def _generate(records: list[dict], *, model_name: str, extra_tokens: int,
              temperature: float, batch_size: int, seed: int) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from data.collect_data import DATASETS, extract_math_answer, normalize_math_answer
    from datasets import load_dataset

    questions = {
        index: example["problem"]
        for index, example in enumerate(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    }
    system_prompt = DATASETS["math500"]["system_prompt"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    torch.manual_seed(seed)

    results = []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        prefixes = []
        for record in chunk:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": questions[record["prompt_id"]]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            prefixes.append(
                tokenizer.encode(prompt)
                + tokenizer.convert_tokens_to_ids(load_tokens(record))
            )
        width = max(len(prefix) for prefix in prefixes)
        input_ids = torch.full((len(chunk), width), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, prefix in enumerate(prefixes):
            input_ids[row, width - len(prefix):] = torch.tensor(prefix)
            attention_mask[row, width - len(prefix):] = 1
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=extra_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
        for row, record in enumerate(chunk):
            new_ids = generated[row, width:].tolist()
            terminated = tokenizer.eos_token_id in new_ids
            if terminated:
                new_ids = new_ids[: new_ids.index(tokenizer.eos_token_id)]
            results.append(
                dict(
                    record,
                    **classify(
                        tokenizer.convert_ids_to_tokens(new_ids),
                        tokenizer.decode(new_ids),
                        terminated,
                        record["gold"],
                        extract_answer=extract_math_answer,
                        normalize=normalize_math_answer,
                    ),
                    continuation_text=tokenizer.decode(new_ids),
                )
            )
        done = start + len(chunk)
        print(f"[{done}/{len(records)}] " + ", ".join(
            f"{entry['outcome']}" for entry in results[-len(chunk):]
        ), flush=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--n_traces", type=int, default=50)
    parser.add_argument("--extra_tokens", type=int, default=8192,
                        help="Tokens of budget added on top of the original cap.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help=(
            "Split the selected traces across this many independent runs, one "
            "GPU each. Selection happens before the split, so the union over "
            "shards is exactly the unsharded sample."
        ),
    )
    parser.add_argument("--shard", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cap = resolve_cap(args.max_new_tokens, data_dir=args.data_dir, context="continue_capped")
    records = capped_traces(args.data_dir, cap.value)
    print(f"{len(records)} capped traces at cap {cap.value} ({cap.provenance})", flush=True)
    selected, degenerate = select_traces(records, args.n_traces, args.seed)
    print(f"continuing {len(selected)}; excluded {len(degenerate)} already looping", flush=True)
    if args.num_shards > 1:
        selected = selected[args.shard::args.num_shards]
        print(f"shard {args.shard}/{args.num_shards}: {len(selected)} traces", flush=True)

    results = _generate(
        selected,
        model_name=args.model_name,
        extra_tokens=args.extra_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    report = summarize(results, degenerate)
    report["settings"] = {
        "data_dir": args.data_dir,
        "model_name": args.model_name,
        "original_cap": cap.value,
        "cap_provenance": cap.provenance,
        "extra_tokens": args.extra_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # A shard's summary covers its own traces only; merge_shards rebuilds the
    # population summary from the per-trace files.
    suffix = f"_shard{args.shard}" if args.num_shards > 1 else ""
    (out / f"math500_continue_capped_results{suffix}.json").write_text(
        json.dumps(report, indent=2)
    )
    (out / f"math500_continue_capped_traces{suffix}.json").write_text(
        json.dumps(results, indent=2)
    )
    print(json.dumps(report, indent=2))


def merge_shards(paths: list[str | Path], excluded: list[dict]) -> tuple[dict, list[dict]]:
    """Rebuild the population summary from per-shard trace files."""
    results: list[dict] = []
    for path in paths:
        results.extend(json.loads(Path(path).read_text()))
    seen = {entry["trace_id"] for entry in results}
    if len(seen) != len(results):
        raise ValueError("shards overlap: the same trace was continued twice")
    return summarize(results, excluded), results


if __name__ == "__main__":
    main()
