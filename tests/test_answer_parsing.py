import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect_data import extract_math_answer, save_batch


def test_extract_math_answer_handles_nested_latex_and_uses_last_box():
    text = (
        r"An intermediate result is \boxed{3}. "
        r"Therefore the final answer is \boxed{\frac{1+\sqrt{5}}{2}}."
    )

    assert extract_math_answer(text) == r"\frac{1+\sqrt{5}}{2}"


def test_extract_math_answer_accepts_fbox_with_whitespace():
    assert extract_math_answer(r"Final: \fbox { \frac{3}{4} }") == r"\frac{3}{4}"


def test_save_batch_persists_generated_text_for_future_reparsing(tmp_path: Path):
    result = {
        "trace_id": 0,
        "idx": 0,
        "sample_id": 0,
        "is_correct": False,
        "n_tokens": 2,
        "gold_answer": "1/2",
        "predicted_answer": None,
        "generated_text": r"Final answer: \boxed{\frac{1}{2}}",
        "mean_logprob": -0.2,
        "generation_seed": 42,
        "entropies": np.asarray([0.1, 0.2], dtype=np.float32),
        "token_logprobs": np.asarray([-0.1, -0.3], dtype=np.float32),
        "tokens": ["Final", " answer"],
        "hidden_layer_7": np.zeros((2, 3), dtype=np.float32),
    }

    save_batch([result], 0, str(tmp_path), layers=[7])

    with np.load(tmp_path / "batch_0000.npz", allow_pickle=True) as data:
        metadata = data["metadata"][0]
        assert metadata["generated_text"] == result["generated_text"]
        assert data["tokens_0"].tolist() == result["tokens"]
