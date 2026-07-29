"""Logic tests for batched generation in collect_data.generate_traces_batched.

These use a CPU FakeModel/FakeTokenizer with controllable logits and hidden
states so we can verify the bug-prone bookkeeping without loading a real 7B
model: per-row EOS truncation, hidden-state/token alignment, and per-sample
RNG determinism. Numerical fidelity of a real forward pass is out of scope.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect_data import generate_traces_batched


class FakeOutput:
    def __init__(self, logits, hidden_states, past_key_values):
        self.logits = logits
        self.hidden_states = hidden_states
        self.past_key_values = past_key_values


class FakeModel:
    """Deterministic stand-in for a causal LM.

    hidden_states[L][b, s, :] encodes the input id at that position as
    id * L (broadcast over hidden dim), so tests can assert alignment.
    logits at the final position come from `logits_fn(step, input_ids)`.
    """

    def __init__(self, logits_fn, n_layers=3, hidden=4, vocab=100):
        self.logits_fn = logits_fn
        self.n_layers = n_layers
        self.hidden = hidden
        self.vocab = vocab
        self.device = torch.device("cpu")
        self._step = 0

    def eval(self):
        return self

    def __call__(self, input_ids, past_key_values=None,
                 output_hidden_states=False, use_cache=False, **kwargs):
        B, S = input_ids.shape
        hidden_states = tuple(
            input_ids.to(torch.float32)[:, :, None].expand(B, S, self.hidden) * L
            for L in range(self.n_layers + 1)
        )
        last_logits = self.logits_fn(self._step, input_ids)  # [B, vocab]
        logits = torch.zeros(B, S, self.vocab)
        logits[:, -1, :] = last_logits
        prev = 0 if past_key_values is None else int(past_key_values)
        self._step += 1
        return FakeOutput(logits, hidden_states, prev + S)


class FakeTokenizer:
    def __init__(self, eos_token_id, prompt_ids=(1, 2, 3)):
        self.eos_token_id = eos_token_id
        self._prompt_ids = list(prompt_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT"

    def encode(self, prompt, return_tensors="pt"):
        return torch.tensor([self._prompt_ids], dtype=torch.long)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return f"tok{ids}"
        return [f"tok{i}" for i in ids]


def _peaked_logits_from_plan(plan, vocab=100):
    def fn(step, input_ids):
        B = input_ids.shape[0]
        out = torch.full((B, vocab), -10.0)
        for b in range(B):
            out[b, plan[step][b]] = 10.0
        return out
    return fn


def test_per_row_eos_truncation_and_alignment():
    # Row0 emits eos(99) at step2 -> len 3; row1 at step4 -> len 5;
    # row2 never -> len 8 (== max_new_tokens). Loop must run all 8 steps.
    eos = 99
    plan = [
        [10, 11, 12],
        [20, 21, 22],
        [eos, 31, 32],
        [eos, 41, 42],
        [eos, eos, 52],
        [eos, eos, 62],
        [eos, eos, 72],
        [eos, eos, 82],
    ]
    model = FakeModel(_peaked_logits_from_plan(plan))
    tok = FakeTokenizer(eos_token_id=eos, prompt_ids=(1, 2, 3))
    traces = generate_traces_batched(
        model, tok, "q", system_prompt="s",
        layers_to_capture=[1, 2], max_new_tokens=8,
        temperature=0.0, num_samples=3, device=torch.device("cpu"),
    )
    assert len(traces) == 3
    expected_ids = [
        [10, 20, 99],
        [11, 21, 31, 41, 99],
        [12, 22, 32, 42, 52, 62, 72, 82],
    ]
    for r, exp in enumerate(expected_ids):
        assert traces[r]["generated_ids"] == exp, (r, traces[r]["generated_ids"])
        assert traces[r]["token_entropies"].shape == (len(exp),)
        assert traces[r]["token_logprobs"].shape == (len(exp),)
        for L in (1, 2):
            hs = traces[r]["token_hidden_states"][L]
            assert hs.shape == (len(exp), model.hidden)
            # hidden[k] encodes input at step k: k=0 -> last prompt token (3),
            # k>=1 -> generated token k-1, scaled by layer index L.
            expected_inputs = [3] + exp[:-1]
            for k, tid in enumerate(expected_inputs):
                assert np.allclose(hs[k], tid * L), (r, L, k, hs[k], tid * L)
        assert traces[r]["generated_tokens"][-1] == f"tok{exp[-1]}"


def test_capture_matches_batched_generation():
    """Two-phase crux: a teacher-forced capture forward over the generated ids
    must reproduce the hidden states / entropies / logprobs that
    generate_traces_batched captured during decode (exactly, on the fake model).
    """
    from collect_data import capture_features_batched

    eos = 99
    plan = [
        [10, 11, 12],
        [20, 21, 22],
        [eos, 31, 32],
        [eos, 41, 42],
        [eos, eos, 52],
        [eos, eos, 62],
        [eos, eos, 72],
        [eos, eos, 82],
    ]
    tok = FakeTokenizer(eos_token_id=eos, prompt_ids=(1, 2, 3))
    gen = generate_traces_batched(
        FakeModel(_peaked_logits_from_plan(plan)), tok, "q", system_prompt="s",
        layers_to_capture=[1, 2], max_new_tokens=8,
        temperature=0.0, num_samples=3, device=torch.device("cpu"),
    )

    cap = capture_features_batched(
        FakeCaptureModel(), tok, "q", system_prompt="s",
        generated_ids_list=[t["generated_ids"] for t in gen],
        layers_to_capture=[1, 2], device=torch.device("cpu"),
        chunk_size=4,  # force multi-chunk path (total len 3 + 8 = 11)
    )

    assert len(cap) == len(gen)
    for g, c in zip(gen, cap):
        assert c["generated_ids"] == g["generated_ids"]
        assert c["generated_tokens"] == g["generated_tokens"]
        np.testing.assert_allclose(c["token_logprobs"], g["token_logprobs"], atol=1e-6)
        for L in (1, 2):
            np.testing.assert_allclose(
                c["token_hidden_states"][L], g["token_hidden_states"][L], atol=1e-6
            )
        np.testing.assert_allclose(c["token_entropies"], g["token_entropies"], atol=1e-6)


class FakeCaptureModel(FakeModel):
    """FakeModel variant usable for teacher-forced capture: emits per-position
    logits that replay the same plan as _peaked_logits_from_plan, independent of
    decode stepping. Position p's logits must predict the id at position p+1.
    """

    def __init__(self, n_layers=3, hidden=4, vocab=100):
        super().__init__(logits_fn=None, n_layers=n_layers, hidden=hidden, vocab=vocab)

    def __call__(self, input_ids, past_key_values=None, attention_mask=None,
                 output_hidden_states=False, use_cache=False, **kwargs):
        B, S = input_ids.shape
        hidden_states = tuple(
            input_ids.to(torch.float32)[:, :, None].expand(B, S, self.hidden) * L
            for L in range(self.n_layers + 1)
        )
        # Replay the decode-time logits by absolute position: decode step k saw
        # logits peaked per plan[k], and step k corresponds to teacher-forced
        # position p = prompt_len - 1 + k. Positions past the plan (padding)
        # produce flat logits and must be sliced away by the implementation.
        logits = torch.full((B, S, self.vocab), -10.0)
        prompt_len = 3
        plan = [
            [10, 11, 12],
            [20, 21, 22],
            [99, 31, 32],
            [99, 41, 42],
            [99, 99, 52],
            [99, 99, 62],
            [99, 99, 72],
            [99, 99, 82],
        ]
        past = 0 if past_key_values is None else int(past_key_values)
        for b in range(B):
            for s in range(S):
                p = past + s  # absolute position
                k = p - (prompt_len - 1)  # decode step this position predicts
                if 0 <= k < len(plan):
                    logits[b, s, plan[k][b]] = 10.0
        return FakeOutput(logits, hidden_states, past + S)


def test_grouped_generation_bookkeeping():
    """Phase 1: cross-problem rows with different prompt lengths must keep
    per-row token streams, EOS trimming, and problem-major row order."""
    from collect_data import generate_tokens_grouped

    eos = 99
    # Rows: p0 has 2 samples, p1 has 2 samples -> 4 rows, problem-major.
    plan = [
        [10, 11, 12, 13],
        [20, 21, eos, 23],
        [eos, 31, eos, 33],
        [eos, eos, eos, eos],
    ]
    model = FakeModel(_peaked_logits_from_plan(plan))
    out = generate_tokens_grouped(
        model, eos_token_id=eos,
        prompt_ids_list=[[1, 2, 3], [4, 5]],  # different lengths -> left padding
        max_new_tokens=4, temperature=0.0, num_samples=2,
        device=torch.device("cpu"),
    )
    assert [[list(s) for s in p] for p in out] == [
        [[10, 20, 99], [11, 21, 31, 99]],
        [[12, 99], [13, 23, 33, 99]],
    ]


def test_grouped_generation_seed_streams():
    """Same seed -> same stream regardless of which group the row is in."""
    from collect_data import generate_tokens_grouped

    eos = 12345
    mk = lambda: FakeModel(lambda step, ids: torch.zeros(ids.shape[0], 100))
    a = generate_tokens_grouped(
        mk(), eos_token_id=eos, prompt_ids_list=[[1, 2, 3]],
        max_new_tokens=8, temperature=0.6, seeds_list=[[7, 8]], num_samples=2,
        device=torch.device("cpu"),
    )
    b = generate_tokens_grouped(
        mk(), eos_token_id=eos, prompt_ids_list=[[1, 2, 3]],
        max_new_tokens=8, temperature=0.6, seeds_list=[[7, 8]], num_samples=2,
        device=torch.device("cpu"),
    )
    assert a == b
    assert a[0][0] != a[0][1]


def test_two_phase_pipeline_matches_single_phase_ids():
    """Phase1 + phase2 on one problem must produce the same token streams as
    the capturing decode when fed the same seeds (identical prompt, no padding,
    identical per-row RNG consumption)."""
    from collect_data import generate_tokens_grouped, capture_features_batched

    eos = 12345
    tok = FakeTokenizer(eos_token_id=eos, prompt_ids=(1, 2, 3))
    mk = lambda: FakeModel(lambda step, ids: torch.zeros(ids.shape[0], 100))

    single = generate_traces_batched(
        mk(), tok, "q", system_prompt="s", layers_to_capture=[1],
        max_new_tokens=8, temperature=0.6, seeds=[7, 8, 9], num_samples=3,
        device=torch.device("cpu"),
    )
    gen = generate_tokens_grouped(
        mk(), eos_token_id=eos, prompt_ids_list=[[1, 2, 3]],
        max_new_tokens=8, temperature=0.6, seeds_list=[[7, 8, 9]], num_samples=3,
        device=torch.device("cpu"),
    )
    assert gen[0] == [t["generated_ids"] for t in single]

    cap = capture_features_batched(
        FakeCaptureModel(), tok, "q", system_prompt="s",
        generated_ids_list=gen[0], layers_to_capture=[1],
        device=torch.device("cpu"), chunk_size=3,
    )
    for c, g in zip(cap, single):
        assert c["generated_ids"] == g["generated_ids"]
        for k, tid in enumerate([3] + c["generated_ids"][:-1]):
            assert np.allclose(c["token_hidden_states"][1][k], tid * 1)


def test_per_sample_seed_determinism():
    eos = 12345  # unreachable -> every trace runs full length
    model = FakeModel(lambda step, ids: torch.zeros(ids.shape[0], 100))  # uniform
    tok = FakeTokenizer(eos_token_id=eos)

    def run():
        m = FakeModel(lambda step, ids: torch.zeros(ids.shape[0], 100))
        return generate_traces_batched(
            m, tok, "q", system_prompt="s", layers_to_capture=[1],
            max_new_tokens=8, temperature=0.6, seeds=[7, 7, 8], num_samples=3,
            device=torch.device("cpu"),
        )

    a = run()
    b = run()
    # Reproducible across calls with identical seeds.
    for r in range(3):
        assert a[r]["generated_ids"] == b[r]["generated_ids"]
    # Same seed -> same trace; different seed -> different (w.h.p. over 8 draws).
    assert a[0]["generated_ids"] == a[1]["generated_ids"]
    assert a[0]["generated_ids"] != a[2]["generated_ids"]
    for r in range(3):
        assert len(a[r]["generated_ids"]) == 8
