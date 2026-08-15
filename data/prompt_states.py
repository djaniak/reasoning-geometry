"""Save prompt-position hidden states for the prompt-only geometry control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def gather_last_valid_hidden(hidden, attention_mask):
    """Select the hidden state at each row's last non-padding prompt token."""
    import torch

    if hidden.ndim != 3 or attention_mask.ndim != 2 or hidden.shape[:2] != attention_mask.shape:
        raise ValueError("hidden must be [batch, sequence, dim] and mask [batch, sequence]")
    reversed_mask = attention_mask.long().flip(dims=(1,))
    indices = attention_mask.shape[1] - 1 - reversed_mask.argmax(dim=1)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, indices]


def run_prompt_state_pass(
    *,
    output_path: str,
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    dataset_name: str = "HuggingFaceH4/MATH-500",
    layers: list[int] = (7, 14, 21),
    batch_size: int = 8,
) -> dict:
    import torch
    from datasets import load_dataset

    from data.collect_data import load_model

    if not torch.cuda.is_available():
        raise RuntimeError("prompt-position state pass requires CUDA")
    model, tokenizer = load_model(False, model_name=model_name)
    dataset = load_dataset(dataset_name, split="test")
    system_prompt = "Solve this math problem step by step. Put your final answer in \\boxed{}."
    prompts = []
    for example in dataset:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["problem"]},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    states = {layer: [] for layer in layers}
    prompt_ids = []
    with torch.no_grad():
        for start in range(0, len(prompts), int(batch_size)):
            batch_prompts = prompts[start : start + int(batch_size)]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)
            position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min(0)
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            for layer in layers:
                selected = gather_last_valid_hidden(
                    outputs.hidden_states[layer], attention_mask
                ).float().cpu().numpy()
                states[layer].append(selected)
            prompt_ids.extend(range(start, start + len(batch_prompts)))
            del outputs, input_ids, attention_mask, encoded
            if start and start % (int(batch_size) * 10) == 0:
                print(f"processed {start}/{len(prompts)} prompts", flush=True)
    tokenizer.padding_side = old_padding_side

    payload = {"prompt_ids": np.asarray(prompt_ids, dtype=np.int64)}
    for layer, values in states.items():
        payload[f"prompt_hidden_L{layer}"] = np.concatenate(values, axis=0).astype(np.float16)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    result = {
        "model": model_name,
        "dataset": dataset_name,
        "n_prompts": len(prompt_ids),
        "layers": list(layers),
        "batch_size": int(batch_size),
        "state_source": "last non-padding chat-template prompt token",
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--dataset_name", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--layers", default="7,14,21")
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run_prompt_state_pass(
        output_path=args.output_path,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        layers=[int(value) for value in args.layers.split(",") if value.strip()],
        batch_size=args.batch_size,
    ), indent=2))


if __name__ == "__main__":
    main()
