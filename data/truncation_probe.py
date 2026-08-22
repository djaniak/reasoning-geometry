"""Budget-sizing probe for max_new_tokens (no hidden-state capture).

Answers one question cheaply: at a candidate generation budget, what fraction of
traces hit the cap without emitting a final answer, and how long are the traces
that DO finish? This is the censored information the existing OOF CSVs cannot give
(their truncated traces are all clipped at the old 2048 cap).

It is faithful to collect_data.py where it matters for the truncation rate -- same
model loader, same MATH-500 system prompt, same chat template, same answer parser --
but uses batched HuggingFace generate() and captures NO hidden states, so it runs in
minutes instead of hours and writes no large NPZ files.

Example:
    CUDA_VISIBLE_DEVICES=6 uv run python truncation_probe.py \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --max_new_tokens 8192 --limit 48 --num_samples 2 --batch_size 8 \
        --output results/truncation_probe/deepseek_8192.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from data.collect_data import (
    DATASETS,
    extract_math_answer,
    load_model,
    normalize_math_answer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=48, help="Number of problems")
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--output", default=None, help="JSON summary path")
    return parser.parse_args()


def eos_token_ids(tokenizer, model) -> set[int]:
    ids: set[int] = set()
    for source in (tokenizer.eos_token_id, model.generation_config.eos_token_id):
        if source is None:
            continue
        if isinstance(source, (list, tuple)):
            ids.update(int(value) for value in source)
        else:
            ids.add(int(source))
    return ids


def completion_length(generated: torch.Tensor, eos_ids: set[int], cap: int) -> int:
    """Number of tokens until (and including) the first EOS, else the cap."""
    for position, token in enumerate(generated.tolist()):
        if token in eos_ids:
            return position + 1
    return cap


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    started = time.perf_counter()

    print(f"Loading model {args.model_name} ...", flush=True)
    model, tokenizer = load_model(args.quantize, model_name=args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eos_ids = eos_token_ids(tokenizer, model)

    config = DATASETS["math500"]
    dataset = load_dataset(config["hf_path"], split=config["split"])
    dataset = dataset.select(range(min(args.limit, len(dataset))))
    system_prompt = config["system_prompt"]

    # One generation request per (problem, sample).
    requests = []
    for idx, example in enumerate(dataset):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["problem"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        gold = normalize_math_answer(example["answer"])
        for sample_id in range(args.num_samples):
            requests.append({"idx": idx, "sample_id": sample_id, "prompt": prompt, "gold": gold})

    traces = []
    for start in tqdm(range(0, len(requests), args.batch_size), desc="probe", unit="batch"):
        batch = requests[start : start + args.batch_size]
        encoded = tokenizer(
            [item["prompt"] for item in batch],
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        input_len = encoded["input_ids"].shape[1]
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output[:, input_len:]
        for item, row in zip(batch, generated):
            length = completion_length(row, eos_ids, args.max_new_tokens)
            text = tokenizer.decode(row, skip_special_tokens=True)
            predicted = normalize_math_answer(extract_math_answer(text))
            traces.append(
                {
                    "idx": item["idx"],
                    "sample_id": item["sample_id"],
                    "length": int(length),
                    "capped": bool(length >= args.max_new_tokens),
                    "parsed": bool(predicted),
                    "correct": bool(predicted and item["gold"] and predicted == item["gold"]),
                }
            )

    n = len(traces)
    lengths = np.array([t["length"] for t in traces])
    completed = lengths[~np.array([t["capped"] for t in traces])]
    n_capped = sum(t["capped"] for t in traces)
    n_unparsed = sum(not t["parsed"] for t in traces)
    n_correct = sum(t["correct"] for t in traces)

    def pct(values, q):
        return float(np.percentile(values, q)) if len(values) else None

    summary = {
        "model_name": args.model_name,
        "max_new_tokens": args.max_new_tokens,
        "n_traces": n,
        "n_problems": int(args.limit),
        "num_samples": int(args.num_samples),
        "temperature": args.temperature,
        "capped_rate": n_capped / n,
        "unparsed_rate": n_unparsed / n,
        "correct_rate": n_correct / n,
        "length": {
            "median": pct(lengths, 50),
            "p90": pct(lengths, 90),
            "p95": pct(lengths, 95),
            "max": int(lengths.max()),
        },
        "completed_length": {
            "n": int(len(completed)),
            "median": pct(completed, 50),
            "p90": pct(completed, 90),
            "p95": pct(completed, 95),
            "max": int(completed.max()) if len(completed) else None,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }

    print(json.dumps(summary, indent=2))
    print(
        f"\n>>> {args.model_name.split('/')[-1]} @ {args.max_new_tokens}: "
        f"capped={summary['capped_rate']:.1%}  unparsed={summary['unparsed_rate']:.1%}  "
        f"correct={summary['correct_rate']:.1%}  "
        f"completed p95 len={summary['completed_length']['p95']}",
        flush=True,
    )
    if summary["capped_rate"] > 0.10:
        print(
            ">>> Cap-hit still >10%: this budget is too small; raise --max_new_tokens.",
            flush=True,
        )

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summary": summary, "traces": traces}, indent=2) + "\n")
        print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
