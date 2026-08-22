import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.deepconf_exact import reconstruct_token_ids, summarize_exact_confidence, topk_token_confidence
from data.prompt_states import gather_last_valid_hidden


def test_topk_token_confidence_is_negative_mean_log_probability_of_top_k():
    logits = np.asarray([[3.0, 2.0, 1.0], [0.0, 4.0, 1.0]], dtype=np.float32)
    got = topk_token_confidence(logits, k=2)
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    expected = -np.sort(log_probs, axis=1)[:, -2:].mean(axis=1)
    assert np.allclose(got, expected)


def test_topk_confidence_rejects_invalid_k():
    with pytest.raises(ValueError):
        topk_token_confidence(np.zeros((2, 3)), k=4)


def test_reconstruct_token_ids_round_trips_token_strings():
    class Tokenizer:
        def convert_tokens_to_ids(self, tokens):
            mapping = {"A": 1, " B": 2, "C": 3}
            return [mapping[token] for token in tokens]

    assert reconstruct_token_ids(["A", " B", "C"], Tokenizer()) == [1, 2, 3]


def test_summary_keeps_raw_deepconf_direction():
    summary = summarize_exact_confidence(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert summary["deepconf_global"] == pytest.approx(3.0)
    assert summary["deepconf_tail_q20"] == pytest.approx(5.0)


def test_prompt_state_gather_selects_last_nonpadding_position():
    import torch

    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
    assert torch.equal(gather_last_valid_hidden(hidden, mask), hidden[[0, 1], [3, 3]])
