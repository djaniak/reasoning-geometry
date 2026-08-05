"""Exact top-k DeepConf pilot over cached DeepSeek traces.

The collection cache stores sampled token strings, entropy, log-probability, and
hidden states, but not the full next-token distribution.  This script reconstructs
the token IDs and teacher-forces the cached traces through the model once to
recover the actual top-k candidate statistic.  The pilot is diagnostic: its
correlations are not a 100-prompt significance gate.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def reconstruct_token_ids(tokens: Iterable[str], tokenizer) -> list[int]:
    """Map cached tokenizer token strings back to their integer IDs."""
    values = [str(token) for token in tokens]
    ids = tokenizer.convert_tokens_to_ids(values)
    if np.isscalar(ids):
        ids = [ids]
    result = [int(value) for value in ids]
    if len(result) != len(values):
        raise ValueError("tokenizer returned a different number of token IDs")
    return result


def topk_token_confidence(logits, k: int = 20):
    """Return ``-(1/k) sum(log p_j)`` over each row's top-k candidates."""
    if getattr(logits, "ndim", 0) < 2:
        raise ValueError("logits must have a vocabulary dimension")
    vocab = int(logits.shape[-1])
    if k <= 0 or k > vocab:
        raise ValueError(f"k must be in [1, {vocab}], got {k}")
    # Keep this helper usable in CPU unit tests while preserving a GPU path for
    # the pilot (calling .cpu() here would materialize the full vocabulary).
    if hasattr(logits, "__class__") and logits.__class__.__module__.startswith("torch"):
        import torch

        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return -torch.topk(log_probs, k=k, dim=-1).values.mean(dim=-1)
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    top = np.partition(log_probs, -k, axis=-1)[..., -k:]
    return -np.mean(top, axis=-1)


def _cached_groups(data_dir: str | Path, prompt_ids: set[int]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(Path(data_dir).glob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            available = set(data.files)
            for metadata in data["metadata"]:
                prompt_id = int(metadata["idx"])
                if prompt_id not in prompt_ids:
                    continue
                trace_id = int(metadata.get("trace_id", prompt_id))
                token_key = f"tokens_{trace_id}"
                if token_key not in available:
                    token_key = f"tokens_{int(metadata.get('idx', prompt_id))}"
                if token_key not in available:
                    raise ValueError(
                        f"{path} has no cached token strings for trace {trace_id}; "
                        "exact DeepConf cannot be recovered from this collection"
                    )
                entropy_key = f"entropies_{trace_id}"
                lp_key = f"token_logprobs_{trace_id}"
                if entropy_key not in available or lp_key not in available:
                    raise ValueError(f"{path} is missing cached entropy/logprob arrays for {trace_id}")
                groups[prompt_id].append(
                    {
                        "prompt_id": prompt_id,
                        "trace_id": trace_id,
                        "sample_id": int(metadata.get("sample_id", 0)),
                        "tokens": [str(value) for value in data[token_key].tolist()],
                        "entropies": np.asarray(data[entropy_key], dtype=np.float32),
                        "token_logprobs": np.asarray(data[lp_key], dtype=np.float32),
                    }
                )
    return {prompt_id: sorted(group, key=lambda row: row["sample_id"]) for prompt_id, group in groups.items()}


def complete_prompt_ids(groups: Mapping[int, list[dict]], expected_traces: int = 8) -> list[int]:
    return sorted(
        prompt_id
        for prompt_id, group in groups.items()
        if len(group) == int(expected_traces)
        and sorted(int(row["sample_id"]) for row in group) == list(range(int(expected_traces)))
    )


def _tail_indices(length: int) -> np.ndarray:
    count = max(1, int(math.ceil(0.20 * length)))
    return np.arange(length - count, length, dtype=int)


def _group_confidence(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    window = max(1, min(values.size, int(math.ceil(0.20 * values.size))))
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    starts = np.arange(0, values.size - window + 1)
    groups = (cumulative[starts + window] - cumulative[starts]) / window
    keep = max(1, int(math.ceil(0.10 * groups.size)))
    return {
        "lowest_group_confidence": float(groups.min()),
        "bottom10_group_confidence": float(np.sort(groups)[:keep].mean()),
    }


def summarize_exact_confidence(confidence: np.ndarray) -> dict[str, float]:
    """Summarize the raw DeepConf statistic.

    DeepConf defines confidence as ``-mean(log p)`` over the top-k candidates.
    Keep that raw quantity here; the proxy columns in the report are separately
    named score-orientations (``-entropy`` and sampled ``logprob``).
    """
    confidence = np.asarray(confidence, dtype=float)
    tail = _tail_indices(len(confidence))
    result = {
        "deepconf_global": float(confidence.mean()),
        "deepconf_tail_q20": float(confidence[tail].mean()),
    }
    result.update(_group_confidence(confidence))
    return result


def _correlation(x: list[float], y: list[float]) -> dict:
    from scipy.stats import pearsonr, spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    usable = np.isfinite(x) & np.isfinite(y)
    if usable.sum() < 3 or np.ptp(x[usable]) == 0 or np.ptp(y[usable]) == 0:
        return {"n": int(usable.sum()), "pearson": None, "spearman": None}
    return {
        "n": int(usable.sum()),
        "pearson": float(pearsonr(x[usable], y[usable]).statistic),
        "spearman": float(spearmanr(x[usable], y[usable]).statistic),
    }


def _load_model(model_name: str):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "exact DeepConf requires a CUDA device; torch.cuda.is_available() is false"
        )
    from collect_data import load_model

    return load_model(False, model_name=model_name)


def _teacher_force_prompt(
    model,
    tokenizer,
    question: str,
    traces: list[dict],
    *,
    system_prompt: str,
    layers: list[int],
    top_k: int,
    chunk_size: int,
) -> tuple[list[np.ndarray], dict[int, np.ndarray], dict[int, float]]:
    """Recover exact token confidence and prompt-position states for one prompt."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)[0]
    prompt_length = int(prompt_ids.shape[0])
    generated_ids = [reconstruct_token_ids(trace["tokens"], tokenizer) for trace in traces]
    lengths = [len(ids) for ids in generated_ids]
    if not lengths or min(lengths) <= 0:
        raise ValueError("exact DeepConf requires non-empty generated traces")
    batch = len(traces)
    max_new = max(lengths)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((batch, prompt_length + max_new), pad_id, dtype=torch.long, device=model.device)
    input_ids[:, :prompt_length] = prompt_ids
    attention_mask = torch.zeros_like(input_ids)
    for index, ids in enumerate(generated_ids):
        input_ids[index, prompt_length : prompt_length + len(ids)] = torch.as_tensor(ids, device=model.device)
        attention_mask[index, : prompt_length + len(ids)] = 1

    exact = torch.zeros(batch, max_new, dtype=torch.float32, device=model.device)
    prompt_states: dict[int, torch.Tensor] = {}
    max_entropy_error = 0.0
    max_logprob_error = 0.0
    sum_entropy_error = 0.0
    sum_logprob_error = 0.0
    n_error_values = 0
    with torch.no_grad():
        past_key_values = None
        total = int(input_ids.shape[1])
        for start in range(0, total, int(chunk_size)):
            end = min(start + int(chunk_size), total)
            outputs = model(
                input_ids[:, start:end],
                past_key_values=past_key_values,
                attention_mask=attention_mask[:, :end],
                output_hidden_states=bool(layers),
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            lo = max(start, prompt_length - 1)
            hi = min(end, total - 1)
            if lo < hi:
                cols = slice(lo - start, hi - start)
                ks = slice(lo - (prompt_length - 1), hi - (prompt_length - 1))
                logits = outputs.logits[:, cols, :].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                exact[:, ks] = topk_token_confidence(logits, k=top_k)
                next_ids = input_ids[:, lo + 1 : hi + 1]
                sampled_logprob = log_probs.gather(2, next_ids[:, :, None]).squeeze(2)
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1)
                for row_index, trace in enumerate(traces):
                    offset = lo - (prompt_length - 1)
                    count = min(lengths[row_index] - offset, hi - lo)
                    if count <= 0:
                        continue
                    cached_entropy = trace["entropies"][offset : offset + count]
                    cached_logprob = trace["token_logprobs"][offset : offset + count]
                    max_entropy_error = max(
                        max_entropy_error,
                        float(torch.max(torch.abs(entropy[row_index, :count] - torch.as_tensor(cached_entropy, device=model.device))).item()),
                    )
                    sum_entropy_error += float(
                        torch.sum(torch.abs(entropy[row_index, :count] - torch.as_tensor(cached_entropy, device=model.device))).item()
                    )
                    max_logprob_error = max(
                        max_logprob_error,
                        float(torch.max(torch.abs(sampled_logprob[row_index, :count] - torch.as_tensor(cached_logprob, device=model.device))).item()),
                    )
                    sum_logprob_error += float(
                        torch.sum(torch.abs(sampled_logprob[row_index, :count] - torch.as_tensor(cached_logprob, device=model.device))).item()
                    )
                    n_error_values += int(count)
            prompt_position = prompt_length - 1
            if layers and start <= prompt_position < end:
                local = prompt_position - start
                for layer in layers:
                    prompt_states[layer] = outputs.hidden_states[layer][:, local, :].detach().float().cpu()
            del outputs
    exact_np = [exact[index, :length].detach().cpu().numpy() for index, length in enumerate(lengths)]
    checks = {
        "max_entropy_abs_error": max_entropy_error,
        "max_sampled_logprob_abs_error": max_logprob_error,
        "mean_entropy_abs_error": sum_entropy_error / max(1, n_error_values),
        "mean_sampled_logprob_abs_error": sum_logprob_error / max(1, n_error_values),
        "n_error_values": n_error_values,
    }
    return exact_np, {layer: states.numpy() for layer, states in prompt_states.items()}, checks


def run_exact_pilot(
    *,
    data_dir: str,
    output_dir: str,
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    sample_size: int = 100,
    seed: int = 20260802,
    top_k: int = 20,
    expected_traces: int = 8,
    layers: list[int] = (7, 14, 21),
    chunk_size: int = 128,
    dataset_name: str = "HuggingFaceH4/MATH-500",
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict:
    from datasets import load_dataset

    # Read metadata/token arrays without touching the large hidden arrays.
    all_groups = _cached_groups(data_dir, set(range(500)))
    complete = complete_prompt_ids(all_groups, expected_traces=expected_traces)
    if len(complete) < sample_size:
        raise ValueError(f"only {len(complete)} complete prompts available; need {sample_size}")
    rng = np.random.default_rng(seed)
    prompt_ids = sorted(rng.choice(complete, size=sample_size, replace=False).tolist())
    if not 0 <= int(shard_index) < int(num_shards):
        raise ValueError("shard_index must be in [0, num_shards)")
    prompt_ids = prompt_ids[int(shard_index) :: int(num_shards)]
    groups = {prompt_id: all_groups[prompt_id] for prompt_id in prompt_ids}
    model, tokenizer = _load_model(model_name)
    dataset = load_dataset(dataset_name, split="test")
    system_prompt = "Solve this math problem step by step. Put your final answer in \\boxed{}."

    summaries: list[dict] = []
    exact_arrays: list[np.ndarray] = []
    prompt_state_arrays: dict[int, list[np.ndarray]] = defaultdict(list)
    max_entropy_error = 0.0
    max_logprob_error = 0.0
    sum_entropy_error = 0.0
    sum_logprob_error = 0.0
    n_error_values = 0
    roundtrip_mismatches = 0
    for position, prompt_id in enumerate(prompt_ids, start=1):
        traces = groups[prompt_id]
        for trace in traces:
            ids = reconstruct_token_ids(trace["tokens"], tokenizer)
            if tokenizer.convert_ids_to_tokens(ids) != trace["tokens"]:
                roundtrip_mismatches += 1
        exact, prompt_states, checks = _teacher_force_prompt(
            model,
            tokenizer,
            dataset[int(prompt_id)]["problem"],
            traces,
            system_prompt=system_prompt,
            layers=list(layers),
            top_k=top_k,
            chunk_size=chunk_size,
        )
        for layer, states in prompt_states.items():
            prompt_state_arrays[layer].append(states)
        max_entropy_error = max(max_entropy_error, checks["max_entropy_abs_error"])
        max_logprob_error = max(max_logprob_error, checks["max_sampled_logprob_abs_error"])
        sum_entropy_error += checks["mean_entropy_abs_error"] * checks["n_error_values"]
        sum_logprob_error += checks["mean_sampled_logprob_abs_error"] * checks["n_error_values"]
        n_error_values += checks["n_error_values"]
        for trace, confidence in zip(traces, exact):
            exact_arrays.append(confidence.astype(np.float32))
            summary = summarize_exact_confidence(confidence)
            tail = _tail_indices(len(confidence))
            summary.update(
                {
                    "prompt_id": int(prompt_id),
                    "trace_id": int(trace["trace_id"]),
                    "sample_id": int(trace["sample_id"]),
                    "entropy_global": float(-np.mean(trace["entropies"])),
                    "logprob_global": float(np.mean(trace["token_logprobs"])),
                    "entropy_tail_q20": float(-np.mean(trace["entropies"][tail])),
                    "logprob_tail_q20": float(np.mean(trace["token_logprobs"][tail])),
                }
            )
            summaries.append(summary)
        if position % 10 == 0:
            print(f"processed {position}/{len(prompt_ids)} prompts", flush=True)

    correlations = {}
    for exact_name, proxy_names in (
        ("deepconf_global", ("entropy_global", "logprob_global")),
        ("deepconf_tail_q20", ("entropy_tail_q20", "logprob_tail_q20")),
    ):
        for proxy in proxy_names:
            correlations[f"trace:{exact_name}:{proxy}"] = _correlation(
                [row[exact_name] for row in summaries], [row[proxy] for row in summaries]
            )
    prompt_summary: dict[int, dict[str, float]] = {}
    for prompt_id in prompt_ids:
        group = [row for row in summaries if row["prompt_id"] == prompt_id]
        prompt_summary[prompt_id] = {
            key: float(np.mean([row[key] for row in group]))
            for key in (
                "deepconf_global",
                "deepconf_tail_q20",
                "entropy_global",
                "logprob_global",
                "entropy_tail_q20",
                "logprob_tail_q20",
            )
        }
    for exact_name, proxy_names in (
        ("deepconf_global", ("entropy_global", "logprob_global")),
        ("deepconf_tail_q20", ("entropy_tail_q20", "logprob_tail_q20")),
    ):
        for proxy in proxy_names:
            correlations[f"prompt:{exact_name}:{proxy}"] = _correlation(
                [prompt_summary[prompt_id][exact_name] for prompt_id in prompt_ids],
                [prompt_summary[prompt_id][proxy] for prompt_id in prompt_ids],
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    npz_payload = {
        "prompt_ids": np.asarray(prompt_ids, dtype=np.int64),
        "trace_summaries": np.asarray(summaries, dtype=object),
        "exact_token_confidence": np.asarray(exact_arrays, dtype=object),
    }
    for layer, values in prompt_state_arrays.items():
        npz_payload[f"prompt_hidden_L{layer}"] = np.asarray(values, dtype=np.float16)
    np.savez_compressed(output / "deepconf_exact_pilot.npz", **npz_payload)
    result = {
        "model": model_name,
        "data_dir": str(data_dir),
        "sample_size": len(prompt_ids),
        "requested_sample_size": int(sample_size),
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "seed": seed,
        "top_k": top_k,
        "expected_traces": expected_traces,
        "prompt_ids": prompt_ids,
        "complete_prompt_count": len(complete),
        "roundtrip_token_mismatches": roundtrip_mismatches,
        "reconstruction_checks": {
            "max_entropy_abs_error": max_entropy_error,
            "max_sampled_logprob_abs_error": max_logprob_error,
            "mean_entropy_abs_error": sum_entropy_error / max(1, n_error_values),
            "mean_sampled_logprob_abs_error": sum_logprob_error / max(1, n_error_values),
            "n_error_values": n_error_values,
        },
        "correlations": correlations,
        "proxy_divergence_rule": "treat a materially low prompt-level Spearman correlation (<0.8) as a trigger for the full exact pass; this pilot is not a significance test",
        "prompt_position_state_source": "row zero of the teacher-forced hidden state, the last prompt position that predicts generated token zero",
    }
    (output / "deepconf_exact_pilot.json").write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--expected_traces", type=int, default=8)
    parser.add_argument("--layers", default="7,14,21")
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--dataset_name", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_exact_pilot(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        sample_size=args.sample_size,
        seed=args.seed,
        top_k=args.top_k,
        expected_traces=args.expected_traces,
        layers=[int(value) for value in args.layers.split(",") if value.strip()],
        chunk_size=args.chunk_size,
        dataset_name=args.dataset_name,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    print(json.dumps({"sample_size": result["sample_size"], "correlations": result["correlations"]}, indent=2))


if __name__ == "__main__":
    main()
