import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import re
import os
import json
import argparse

# --- Config ---
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYERS_TO_CAPTURE = [7, 14, 21]  # early (~25%), mid (~50%), late (~75%) of 28 layers
MAX_NEW_TOKENS = 1024  # 512 is enough for GSM8K; MATH-500 needs more headroom
OUTPUT_DIR = "collected_data"
BATCH_SAVE_SIZE = 50

def olympiadbench_answerable(example) -> bool:
    """Keep the rows whose gold is a single unit-free numerical value.

    The other 173 rows of OE_TO_maths_en_COMP carry tuples, intervals, symbolic
    expressions, multiple accepted answers, or a unit stored outside
    ``final_answer``. None of those compare correctly against a ``\\boxed{}``
    extraction under string equality, so including them would score as model
    error what is really a matcher limitation. Order-preserving, so ``--limit``
    still selects a deterministic prefix.
    """
    return (
        example["answer_type"] == "Numerical"
        and not example["is_multiple_answer"]
        and not example["unit"]
        and example["final_answer"] is not None
        and len(example["final_answer"]) == 1
    )


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
    # Second prompt set, added 2026-08-10. Competition-level olympiad problems,
    # text-only and open-ended, sharing MATH-500's \boxed{} answer convention so
    # the frozen vote/parse machinery carries over unchanged.
    "olympiadbench": {
        "hf_path": "Hothan/OlympiadBench",
        "hf_name": "OE_TO_maths_en_COMP",
        "split": "train",
        "system_prompt": "Solve this math problem step by step. Put your final answer in \\boxed{}.",
        "row_filter": olympiadbench_answerable,
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
    parser.add_argument("--two_phase", action="store_true",
                        help="Token-only decode across --group_problems problems, then a chunked "
                             "teacher-forced forward reconstructs hidden states/entropy/logprobs")
    parser.add_argument("--group_problems", type=int, default=4,
                        help="Two-phase only: problems decoded together (rows = this x num_samples)")
    parser.add_argument("--capture_chunk_size", type=int, default=1024,
                        help="Two-phase only: positions per capture-forward chunk")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of traces to generate per problem")
    parser.add_argument("--summary_path", type=str, default=None,
                        help="Write pass/parse/truncation rates for the run to this JSON. "
                             "Separates the three reasons a new prompt set can look hard: "
                             "genuinely hard, unparseable, or budget-truncated.")
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


# A single (optionally subscripted) variable bound to the value, e.g. "k=1" or
# "M_{2}=3". Deliberately narrow: answer_type is Numerical for every row that
# survives olympiadbench_answerable, so an "=" there is presentation, never a
# relation the model is meant to output.
_ASSIGNMENT_PREFIX = re.compile(r"^[A-Za-z](?:_\{[^{}]*\}|_[0-9A-Za-z])?\s*=\s*")
_PRESENTATION_MACROS = (
    (r"\dfrac", r"\frac"),
    (r"\tfrac", r"\frac"),
    (r"\left", ""),
    (r"\right", ""),
    (r"\!", ""),
    (r"\,", ""),
)


def normalize_olympiadbench_answer(text: str) -> str:
    """Normalize an OlympiadBench answer to the form ``\\boxed{}`` yields.

    OlympiadBench stores golds as display strings rather than bare values: they
    arrive wrapped in ``$...$`` and sometimes as an assignment (``$k=1$``). This
    undoes presentation only -- delimiters, assignment prefix, and the LaTeX
    macros that render identically -- and then defers to the same normalization
    MATH-500 uses. It does no arithmetic or algebraic rewriting, so it cannot
    turn a wrong answer into a right one.

    Applied to both sides so the model is judged under the gold's conventions
    rather than penalized for not guessing them. MATH-500 keeps plain
    `normalize_math_answer`: its golds are already bare, and its results are
    frozen.
    """
    if not text:
        return text
    stripped = text.strip()
    while len(stripped) > 1 and stripped.startswith("$") and stripped.endswith("$"):
        stripped = stripped[1:-1].strip()
    stripped = _ASSIGNMENT_PREFIX.sub("", stripped, count=1)
    for macro, replacement in _PRESENTATION_MACROS:
        stripped = stripped.replace(macro, replacement)
    return normalize_math_answer(stripped)


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


def generate_traces_batched(model, tokenizer, question: str, system_prompt: str = None,
                            layers_to_capture: list = None, max_new_tokens: int = None,
                            temperature: float = 0.0, seeds: list = None,
                            num_samples: int = 1, device=None):
    """Generate ``num_samples`` traces for one question in a single batched decode.

    All samples share the same prompt (identical length), so no padding is
    needed: the prompt is simply repeated across the batch and the samples
    diverge as sampling proceeds. Everything is accumulated on-device and moved
    to the host once at the end, avoiding the per-token CPU<->GPU syncs that
    starve the GPU in the single-sequence path.

    Returns a list of ``num_samples`` dicts, each with keys:
      ``generated_ids`` (list[int]), ``generated_tokens`` (list[str]),
      ``token_entropies`` (np.float32 [seq]), ``token_logprobs`` (np.float32 [seq]),
      ``token_hidden_states`` (dict layer -> np.float32 [seq, hidden]).

    Alignment matches ``generate_trace``: hidden_states[k] is the representation
    of the input at step k (the last prompt token for k=0, generated token k-1
    otherwise), and entropies/logprobs[k] describe generated token k.
    """
    if system_prompt is None:
        system_prompt = DATASETS["gsm8k"]["system_prompt"]
    if layers_to_capture is None:
        layers_to_capture = LAYERS_TO_CAPTURE
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS
    if device is None:
        device = model.device
    B = num_samples

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)  # [1, L]
    current_input = prompt_ids.expand(B, -1).contiguous()  # [B, L]

    generators = None
    if temperature > 0:
        if seeds is None:
            seeds = [None] * B
        generators = []
        for s in seeds:
            g = torch.Generator(device=device)
            if s is not None:
                g.manual_seed(int(s))
            generators.append(g)

    eos_id = tokenizer.eos_token_id
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    seq_len = torch.full((B,), max_new_tokens, dtype=torch.long, device=device)

    hs_steps = {layer: [] for layer in layers_to_capture}
    ent_steps, lp_steps, tok_steps = [], [], []
    steps_run = 0

    with torch.no_grad():
        past_key_values = None
        for step in range(max_new_tokens):
            outputs = model(
                current_input,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            logits = outputs.logits[:, -1, :].float()  # [B, vocab]
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            ent_steps.append(-(probs * log_probs).sum(dim=-1))  # [B]

            for layer_idx in layers_to_capture:
                # clone the [B, hidden] slice so the full per-step hidden tensor
                # (incl. the whole prompt at step 0) can be freed.
                hs_steps[layer_idx].append(
                    outputs.hidden_states[layer_idx][:, -1, :].detach().clone()
                )

            if temperature > 0:
                scaled = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.empty(B, dtype=torch.long, device=device)
                for r in range(B):
                    next_token[r] = torch.multinomial(
                        scaled[r], num_samples=1, generator=generators[r]
                    )[0]
            else:
                next_token = logits.argmax(dim=-1)  # [B]

            lp_steps.append(log_probs.gather(1, next_token[:, None]).squeeze(1))  # [B]
            tok_steps.append(next_token)

            newly = (~finished) & (next_token == eos_id)
            seq_len = torch.where(newly, torch.full_like(seq_len, step + 1), seq_len)
            finished = finished | (next_token == eos_id)
            steps_run = step + 1

            # One small sync every 16 steps to allow early stop once all done.
            if (step + 1) % 16 == 0 and bool(finished.all()):
                break

            current_input = next_token[:, None]
            del outputs, logits, probs, log_probs

    # Single host transfer for the whole problem.
    toks = torch.stack(tok_steps, dim=1).cpu().numpy()          # [B, steps]
    ents = torch.stack(ent_steps, dim=1).cpu().numpy()          # [B, steps]
    lps = torch.stack(lp_steps, dim=1).cpu().numpy()            # [B, steps]
    hs = {layer: torch.stack(hs_steps[layer], dim=1).float().cpu().numpy()
          for layer in layers_to_capture}                       # [B, steps, hidden]
    lengths = torch.clamp(seq_len, max=steps_run).cpu().tolist()

    traces = []
    for r in range(B):
        n = int(lengths[r])
        ids = toks[r, :n].tolist()
        traces.append({
            "generated_ids": ids,
            "generated_tokens": tokenizer.convert_ids_to_tokens(ids),
            "token_entropies": ents[r, :n].astype(np.float32),
            "token_logprobs": lps[r, :n].astype(np.float32),
            "token_hidden_states": {layer: hs[layer][r, :n] for layer in layers_to_capture},
        })
    return traces


def generate_tokens_grouped(model, eos_token_id: int, prompt_ids_list: list,
                            max_new_tokens: int, temperature: float = 0.0,
                            seeds_list: list = None, num_samples: int = 1,
                            device=None):
    """Phase 1 of two-phase collection: token-only decode across problems.

    Batches ``len(prompt_ids_list) * num_samples`` rows in one decode loop
    (problem-major row order) with no hidden-state or entropy capture, so far
    more rows fit in memory than the capturing decode allows. Prompts of
    different lengths are left-padded; explicit position_ids and a full-history
    attention mask keep padded rows correct.

    ``seeds_list[p][s]`` seeds the generator for sample ``s`` of problem ``p``
    (same per-row convention as ``generate_traces_batched``).

    Returns ``[n_problems][num_samples]`` lists of generated token ids
    (EOS-inclusive, trimmed per row).
    """
    if device is None:
        device = model.device
    P = len(prompt_ids_list)
    R = P * num_samples

    prompt_lens = [len(p) for p in prompt_ids_list]
    max_L = max(prompt_lens)
    input_ids = torch.zeros(R, max_L, dtype=torch.long, device=device)
    attention_mask = torch.zeros(R, max_L, dtype=torch.long, device=device)
    row_prompt_len = torch.zeros(R, dtype=torch.long, device=device)
    for p in range(P):
        ids = torch.as_tensor(prompt_ids_list[p], dtype=torch.long, device=device)
        for s in range(num_samples):
            r = p * num_samples + s
            input_ids[r, max_L - prompt_lens[p]:] = ids
            attention_mask[r, max_L - prompt_lens[p]:] = 1
            row_prompt_len[r] = prompt_lens[p]

    generators = None
    if temperature > 0:
        generators = []
        for p in range(P):
            row_seeds = seeds_list[p] if seeds_list is not None else [None] * num_samples
            for s in range(num_samples):
                g = torch.Generator(device=device)
                if row_seeds[s] is not None:
                    g.manual_seed(int(row_seeds[s]))
                generators.append(g)

    finished = torch.zeros(R, dtype=torch.bool, device=device)
    seq_len = torch.full((R,), max_new_tokens, dtype=torch.long, device=device)
    tok_steps = []
    steps_run = 0

    # Left padding: position ids must count only real tokens.
    prefill_pos = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    next_pos = row_prompt_len.clone()  # position of the first generated token

    with torch.no_grad():
        past_key_values = None
        current_input = input_ids
        current_pos = prefill_pos
        full_mask = attention_mask
        for step in range(max_new_tokens):
            outputs = model(
                current_input,
                past_key_values=past_key_values,
                attention_mask=full_mask,
                position_ids=current_pos,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()  # [R, vocab]

            if temperature > 0:
                scaled = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.empty(R, dtype=torch.long, device=device)
                for r in range(R):
                    next_token[r] = torch.multinomial(
                        scaled[r], num_samples=1, generator=generators[r]
                    )[0]
            else:
                next_token = logits.argmax(dim=-1)

            tok_steps.append(next_token)
            newly = (~finished) & (next_token == eos_token_id)
            seq_len = torch.where(newly, torch.full_like(seq_len, step + 1), seq_len)
            finished = finished | (next_token == eos_token_id)
            steps_run = step + 1

            if (step + 1) % 16 == 0 and bool(finished.all()):
                break

            current_input = next_token[:, None]
            current_pos = (next_pos + step)[:, None]
            full_mask = torch.cat(
                [full_mask, torch.ones(R, 1, dtype=torch.long, device=device)], dim=1
            )
            del outputs, logits

    toks = torch.stack(tok_steps, dim=1).cpu().numpy()  # [R, steps]
    lengths = torch.clamp(seq_len, max=steps_run).cpu().tolist()

    out = []
    for p in range(P):
        samples = []
        for s in range(num_samples):
            r = p * num_samples + s
            samples.append(toks[r, :int(lengths[r])].tolist())
        out.append(samples)
    return out


def capture_features_batched(model, tokenizer, question: str, system_prompt: str = None,
                             generated_ids_list: list = None, layers_to_capture: list = None,
                             device=None, chunk_size: int = 1024):
    """Teacher-forced feature reconstruction for pre-generated token ids.

    Phase 2 of two-phase collection: given per-sample token ids produced by a
    fast generation pass over the same prompt, recompute -- for every generated
    token -- exactly the features ``generate_traces_batched`` captures during
    decode: hidden states at ``layers_to_capture``, full-vocab entropy, and the
    chosen-token logprob. Causal masking guarantees that teacher-forced
    position ``prompt_len - 1 + k`` reproduces the decode-time step-``k``
    computation.

    The forward runs in chunks of ``chunk_size`` positions with a KV cache so
    peak memory stays bounded: ``output_hidden_states`` materializes every
    layer for the current chunk only, and chunk logits are reduced to
    entropy/logprob immediately. Rows are right-padded, so default positional
    handling is correct and padded positions are simply sliced away.

    Returns trace dicts in the same schema as ``generate_traces_batched``.
    """
    if system_prompt is None:
        system_prompt = DATASETS["gsm8k"]["system_prompt"]
    if layers_to_capture is None:
        layers_to_capture = LAYERS_TO_CAPTURE
    if device is None:
        device = model.device
    B = len(generated_ids_list)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)[0]  # [L]
    L = int(prompt_ids.shape[0])
    lengths = [len(g) for g in generated_ids_list]
    max_n = max(lengths)
    total = L + max_n

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    input_ids = torch.full((B, total), pad_id, dtype=torch.long, device=device)
    input_ids[:, :L] = prompt_ids
    attention_mask = torch.zeros(B, total, dtype=torch.long, device=device)
    for r, gen in enumerate(generated_ids_list):
        if lengths[r]:
            input_ids[r, L:L + lengths[r]] = torch.as_tensor(gen, dtype=torch.long, device=device)
        attention_mask[r, :L + lengths[r]] = 1

    ents = torch.zeros(B, max_n, dtype=torch.float32, device=device)
    lps = torch.zeros(B, max_n, dtype=torch.float32, device=device)
    hs = {layer: None for layer in layers_to_capture}  # allocated on first chunk

    with torch.no_grad():
        past_key_values = None
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            outputs = model(
                input_ids[:, start:end],
                past_key_values=past_key_values,
                attention_mask=attention_mask[:, :end],
                output_hidden_states=True,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            # Position p predicts generated token k = p - (L - 1); the capture
            # region is k in [0, max_n), i.e. p in [L-1, total-2].
            lo = max(start, L - 1)
            hi = min(end, total - 1)
            if lo >= hi:
                del outputs
                continue
            cols = slice(lo - start, hi - start)   # within-chunk positions
            ks = slice(lo - (L - 1), hi - (L - 1))  # step indices

            logits = outputs.logits[:, cols, :].float()  # [B, C, vocab]
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            ents[:, ks] = -(probs * log_probs).sum(dim=-1)
            next_ids = input_ids[:, lo + 1:hi + 1]  # token k lives at position p+1
            lps[:, ks] = log_probs.gather(2, next_ids[:, :, None]).squeeze(2)

            for layer_idx in layers_to_capture:
                if hs[layer_idx] is None:
                    hidden_dim = outputs.hidden_states[layer_idx].shape[-1]
                    hs[layer_idx] = torch.zeros(
                        B, max_n, hidden_dim,
                        dtype=outputs.hidden_states[layer_idx].dtype, device=device,
                    )
                hs[layer_idx][:, ks] = outputs.hidden_states[layer_idx][:, cols, :]

            del outputs, logits, probs, log_probs

    ents_np = ents.cpu().numpy()
    lps_np = lps.cpu().numpy()
    hs_np = {layer: hs[layer].float().cpu().numpy() for layer in layers_to_capture}

    traces = []
    for r in range(B):
        n = lengths[r]
        ids = list(generated_ids_list[r])
        traces.append({
            "generated_ids": ids,
            "generated_tokens": tokenizer.convert_ids_to_tokens(ids),
            "token_entropies": ents_np[r, :n].astype(np.float32),
            "token_logprobs": lps_np[r, :n].astype(np.float32),
            "token_hidden_states": {layer: hs_np[layer][r, :n] for layer in layers_to_capture},
        })
    return traces


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
    # Filter before --limit so the prompt set is a deterministic prefix of the
    # answerable rows rather than a prefix of the raw split with holes in it.
    row_filter = ds_config.get("row_filter")
    if row_filter is not None:
        n_raw = len(dataset)
        dataset = dataset.filter(row_filter)
        print(f"Row filter kept {len(dataset)}/{n_raw} problems.")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # Dataset-specific accessors
    if args.dataset == "gsm8k":
        get_question = lambda ex: ex["question"]
        get_gold = lambda ex: extract_gsm8k_gold(ex["answer"])
        get_predicted = lambda text: extract_gsm8k_answer(text)
        answers_match = lambda pred, gold: pred == gold
    elif args.dataset == "olympiadbench":
        get_question = lambda ex: ex["question"]
        get_gold = lambda ex: normalize_olympiadbench_answer(ex["final_answer"][0])
        get_predicted = lambda text: normalize_olympiadbench_answer(extract_math_answer(text))
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
    total_parsed = 0
    total_truncated = 0
    next_trace_id = 0
    # Continue numbering after any batches already on disk so a resumed run
    # never overwrites existing data.
    existing_batches = sorted(
        int(m.group(1)) for f in os.listdir(args.output_dir)
        if (m := re.fullmatch(r"batch_(\d{4})\.npz", f))
    )
    batch_num = existing_batches[-1] + 1 if existing_batches else 0
    if args.resume_from:
        # Trace ids are globally sequential (num_samples per problem); keep the
        # resumed run consistent with the batches collected before the restart.
        next_trace_id = args.resume_from * max(args.num_samples, 1)
        print(f"Resuming from problem {args.resume_from}: "
              f"next batch {batch_num}, next trace_id {next_trace_id}")

    def problem_seeds(idx):
        return [
            args.seed + idx * max(args.num_samples, 1) + sample_id
            for sample_id in range(args.num_samples)
        ]

    def iter_problem_traces():
        """Yield (idx, example, traces) in global problem order."""
        pending = [(idx, ex) for idx, ex in enumerate(dataset) if idx >= args.resume_from]
        if not args.two_phase:
            for idx, example in tqdm(pending, initial=args.resume_from,
                                     total=len(dataset)):
                # All num_samples share this prompt; one batched capturing decode.
                seeds = problem_seeds(idx)
                yield idx, example, generate_traces_batched(
                    model, tokenizer, get_question(example), system_prompt,
                    layers_to_capture=layers, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    seeds=seeds if args.temperature > 0 else None,
                    num_samples=args.num_samples, device=model.device,
                )
            return

        # Two-phase: token-only decode across a group of problems (more rows in
        # flight than the capturing decode can hold), then a chunked
        # teacher-forced forward per problem reconstructs hidden states,
        # entropies, and logprobs for exactly the generated tokens.
        G = max(args.group_problems, 1)
        with tqdm(initial=args.resume_from, total=len(dataset)) as pbar:
            for gi in range(0, len(pending), G):
                group = pending[gi:gi + G]
                prompt_ids_list = []
                for idx, example in group:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": get_question(example)},
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                    prompt_ids_list.append(tokenizer.encode(prompt))
                gen_ids = generate_tokens_grouped(
                    model, tokenizer.eos_token_id, prompt_ids_list,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    seeds_list=[problem_seeds(idx) for idx, _ in group],
                    num_samples=args.num_samples, device=model.device,
                )
                for (idx, example), ids_per_sample in zip(group, gen_ids):
                    traces = capture_features_batched(
                        model, tokenizer, get_question(example), system_prompt,
                        generated_ids_list=ids_per_sample,
                        layers_to_capture=layers, device=model.device,
                        chunk_size=args.capture_chunk_size,
                    )
                    yield idx, example, traces
                    pbar.update(1)

    for idx, example, traces in iter_problem_traces():
        question = get_question(example)
        gold = get_gold(example)
        sample_seeds = problem_seeds(idx)

        for sample_id, trace in enumerate(traces):
            generated_ids = trace["generated_ids"]
            token_logprobs = trace["token_logprobs"]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            predicted_answer = get_predicted(generated_text)
            is_correct = answers_match(predicted_answer, gold) if (predicted_answer and gold) else False
            total_correct += int(is_correct)
            total_traces += 1
            total_parsed += int(bool(predicted_answer))
            total_truncated += int(len(generated_ids) >= args.max_new_tokens)

            result = {
                "trace_id": next_trace_id,
                "idx": idx,
                "sample_id": sample_id,
                "question": question,
                "gold_answer": gold,
                "predicted_answer": predicted_answer,
                "generated_text": generated_text,
                "is_correct": is_correct,
                "entropies": np.asarray(trace["token_entropies"], dtype=np.float32),
                "token_logprobs": np.asarray(token_logprobs, dtype=np.float32),
                "tokens": trace["generated_tokens"],
                "mean_logprob": float(np.mean(token_logprobs)) if len(token_logprobs) else None,
                "n_tokens": len(generated_ids),
                "generation_seed": sample_seeds[sample_id],
            }
            for layer_idx in layers:
                result[f"hidden_layer_{layer_idx}"] = trace["token_hidden_states"][layer_idx]

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
    print(f"Parsed an answer: {total_parsed}/{denom} ({100*total_parsed/denom:.1f}%); "
          f"hit the token budget: {total_truncated}/{denom} ({100*total_truncated/denom:.1f}%)")

    if args.summary_path:
        summary = {
            "dataset": args.dataset,
            "model_name": args.model_name,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "num_samples": args.num_samples,
            "limit": args.limit,
            "n_problems": len(dataset),
            "n_traces": total_traces,
            "n_correct": total_correct,
            "n_parsed": total_parsed,
            "n_truncated": total_truncated,
            "pass_rate": total_correct / denom,
            "parse_rate": total_parsed / denom,
            "truncation_rate": total_truncated / denom,
        }
        os.makedirs(os.path.dirname(args.summary_path) or ".", exist_ok=True)
        with open(args.summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Wrote run summary to {args.summary_path}")


if __name__ == "__main__":
    main()
