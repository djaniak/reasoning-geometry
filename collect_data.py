import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import re
import os
import argparse

# --- Config ---
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYERS_TO_CAPTURE = [7, 14, 21]  # early (~25%), mid (~50%), late (~75%) of 28 layers
MAX_NEW_TOKENS = 1024  # 512 is enough for GSM8K; MATH-500 needs more headroom
OUTPUT_DIR = "collected_data"
BATCH_SAVE_SIZE = 50

DATASETS = {
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "split": "test",
        "system_prompt": "Solve this math problem step by step. End with #### followed by the numerical answer.",
    },
    "math500": {
        "hf_path": "HuggingFaceH4/MATH-500",
        "hf_name": None,
        "split": "test",
        "system_prompt": "Solve this math problem step by step. Put your final answer in \\boxed{}.",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization (for <24GB VRAM)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems (default: all)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--resume_from", type=int, default=0, help="Skip first N problems (resume interrupted run)")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=list(DATASETS.keys()))
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help="HuggingFace model ID to load")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Max tokens to generate per problem")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to capture (e.g. 7,14,21)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (only used when temperature > 0)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of traces to generate per problem")
    return parser.parse_args()


def load_model(quantize: bool, model_name: str = DEFAULT_MODEL_NAME):
    hf_token = os.environ.get("HF_TOKEN", None)
    offline_mode = (
        os.environ.get("HF_HUB_OFFLINE", "").strip() == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE", "").strip() == "1"
    )
    if quantize:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            attn_implementation="sdpa",
            token=hf_token,
            local_files_only=offline_mode,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            token=hf_token,
            local_files_only=offline_mode,
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        local_files_only=offline_mode,
    )
    return model, tokenizer


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract number after #### in GSM8K format, with fallback to last number."""
    match = re.search(r'####\s*(-?[\d,]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else None


def extract_gsm8k_gold(answer_text: str) -> str | None:
    match = re.search(r'####\s*(-?[\d,]+)', answer_text)
    if match:
        return match.group(1).replace(',', '').strip()
    return None


def extract_math_answer(text: str) -> str | None:
    """Extract the content of the last balanced ``\\boxed`` or ``\\fbox``."""
    starts = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    for match in reversed(starts):
        opening = text.find("{", match.start())
        depth = 0
        for index in range(opening, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    answer = text[opening + 1:index].strip()
                    return answer or None
    return None


def normalize_math_answer(text: str) -> str:
    """Light normalization for MATH answers: strip spaces, lowercase."""
    return text.strip().lower().replace(' ', '') if text else text


def generate_trace(model, tokenizer, question: str, system_prompt: str = None,
                   layers_to_capture: list = None, max_new_tokens: int = None,
                   temperature: float = 0.0, generator: torch.Generator | None = None):
    """
    Run manual autoregressive generation with KV cache.
    Returns: (
        generated_ids,
        generated_tokens,
        token_entropies,
        token_hidden_states,
        chosen_token_logprobs,
    )

    Hidden states dict maps layer_idx -> list of [hidden_dim] float32 arrays (one per generated token).
    Layer indexing: outputs.hidden_states[i] = output of transformer layer i-1 (index 0 = embeddings).
    """
    if system_prompt is None:
        system_prompt = DATASETS["gsm8k"]["system_prompt"]
    if layers_to_capture is None:
        layers_to_capture = LAYERS_TO_CAPTURE
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    token_entropies = []
    token_logprobs = []
    token_hidden_states = {layer: [] for layer in layers_to_capture}
    generated_ids = []
    generated_tokens = []
    past_key_values = None

    with torch.no_grad():
        current_input = input_ids
        for step in range(max_new_tokens):
            outputs = model(
                current_input,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            # Logits for last token — cast to float32 for entropy precision
            logits = outputs.logits[:, -1, :].float()  # [1, vocab_size]
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).item()
            token_entropies.append(entropy)

            # Hidden states: with use_cache=True, each hidden_states[i] has shape [1, 1, hidden_dim]
            # (only the new token is processed). Index 0 = embedding output, index i = layer i output.
            for layer_idx in layers_to_capture:
                h = outputs.hidden_states[layer_idx][:, -1, :].detach().cpu().float().numpy()
                token_hidden_states[layer_idx].append(h.squeeze(0))  # [3584]

            if temperature > 0:
                scaled_logits = logits / temperature
                next_token = torch.multinomial(
                    torch.softmax(scaled_logits, dim=-1), num_samples=1, generator=generator
                ).squeeze(-1)
            else:
                next_token = logits.argmax(dim=-1)  # greedy
            token_logprobs.append(log_probs[0, next_token.item()].item())
            generated_ids.append(next_token.item())
            generated_tokens.append(tokenizer.convert_ids_to_tokens(next_token.item()))

            if next_token.item() == tokenizer.eos_token_id:
                break

            # Next iteration only processes the new token
            current_input = next_token.unsqueeze(0)

            del outputs, logits, probs, log_probs
            if step % 50 == 0:
                torch.cuda.empty_cache()

    del past_key_values
    torch.cuda.empty_cache()

    return generated_ids, generated_tokens, token_entropies, token_hidden_states, token_logprobs


def save_batch(batch_results: list, batch_num: int, output_dir: str, layers: list = None):
    if layers is None:
        layers = LAYERS_TO_CAPTURE
    save_path = os.path.join(output_dir, f"batch_{batch_num:04d}.npz")
    arrays = {}
    for r in batch_results:
        trace_id = r["trace_id"]
        arrays[f"entropies_{trace_id}"] = r["entropies"]
        arrays[f"token_logprobs_{trace_id}"] = r["token_logprobs"]
        arrays[f"tokens_{trace_id}"] = np.array(r["tokens"], dtype=object)
        for layer in layers:
            arrays[f"hidden_L{layer}_{trace_id}"] = r[f"hidden_layer_{layer}"]
    arrays["metadata"] = np.array(
        [{
            "trace_id": r["trace_id"],
            "idx": r["idx"],
            "sample_id": r["sample_id"],
            "is_correct": r["is_correct"],
            "n_tokens": r["n_tokens"],
            "gold": r["gold_answer"],
            "predicted": r["predicted_answer"],
            "generated_text": r.get("generated_text"),
            "mean_logprob": r["mean_logprob"],
            "seed": r["generation_seed"],
        } for r in batch_results],
        dtype=object,
    )
    np.savez_compressed(save_path, **arrays)
    n_correct = sum(r["is_correct"] for r in batch_results)
    print(f"Saved {save_path} ({len(batch_results)} examples, {n_correct} correct)")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    layers = [int(x) for x in args.layers.split(",")] if args.layers else LAYERS_TO_CAPTURE

    if args.temperature > 0:
        print(f"Temperature sampling: T={args.temperature}, base seed={args.seed}")
    elif args.num_samples > 1:
        print("WARNING: num_samples > 1 with greedy decoding will produce duplicate traces.")

    print(f"Loading model {args.model_name} {'(4-bit)' if args.quantize else '(bfloat16)'}...")
    model, tokenizer = load_model(args.quantize, model_name=args.model_name)
    print("Model loaded.")

    ds_config = DATASETS[args.dataset]
    load_kwargs = {"path": ds_config["hf_path"], "split": ds_config["split"]}
    if ds_config["hf_name"]:
        load_kwargs["name"] = ds_config["hf_name"]
    dataset = load_dataset(**load_kwargs)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # Dataset-specific accessors
    if args.dataset == "gsm8k":
        get_question = lambda ex: ex["question"]
        get_gold = lambda ex: extract_gsm8k_gold(ex["answer"])
        get_predicted = lambda text: extract_gsm8k_answer(text)
        answers_match = lambda pred, gold: pred == gold
    else:  # math500
        get_question = lambda ex: ex["problem"]
        get_gold = lambda ex: normalize_math_answer(ex["answer"])
        get_predicted = lambda text: normalize_math_answer(extract_math_answer(text))
        answers_match = lambda pred, gold: pred == gold

    system_prompt = ds_config["system_prompt"]
    batch_results = []
    total_correct = 0
    total_traces = 0
    next_trace_id = 0
    batch_num = 0

    for idx, example in enumerate(tqdm(dataset)):
        if idx < args.resume_from:
            continue

        question = get_question(example)
        gold = get_gold(example)

        for sample_id in range(args.num_samples):
            sample_seed = args.seed + idx * max(args.num_samples, 1) + sample_id
            generator = None
            if args.temperature > 0:
                generator = torch.Generator(device=model.device)
                generator.manual_seed(sample_seed)

            generated_ids, generated_tokens, token_entropies, token_hidden_states, token_logprobs = generate_trace(
                model, tokenizer, question, system_prompt,
                layers_to_capture=layers, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, generator=generator,
            )

            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            predicted_answer = get_predicted(generated_text)
            is_correct = answers_match(predicted_answer, gold) if (predicted_answer and gold) else False
            total_correct += int(is_correct)
            total_traces += 1

            result = {
                "trace_id": next_trace_id,
                "idx": idx,
                "sample_id": sample_id,
                "question": question,
                "gold_answer": gold,
                "predicted_answer": predicted_answer,
                "generated_text": generated_text,
                "is_correct": is_correct,
                "entropies": np.array(token_entropies, dtype=np.float32),
                "token_logprobs": np.array(token_logprobs, dtype=np.float32),
                "tokens": generated_tokens,
                "mean_logprob": float(np.mean(token_logprobs)) if token_logprobs else None,
                "n_tokens": len(generated_ids),
                "generation_seed": sample_seed,
            }
            for layer_idx in layers:
                result[f"hidden_layer_{layer_idx}"] = np.stack(token_hidden_states[layer_idx], axis=0)

            batch_results.append(result)
            next_trace_id += 1

            if len(batch_results) >= BATCH_SAVE_SIZE:
                save_batch(batch_results, batch_num, args.output_dir, layers=layers)
                batch_results = []
                batch_num += 1

    if batch_results:
        save_batch(batch_results, batch_num, args.output_dir, layers=layers)

    denom = max(total_traces, 1)
    print(f"\nDone. Total traces: {total_correct}/{denom} correct ({100*total_correct/denom:.1f}%)")


if __name__ == "__main__":
    main()
