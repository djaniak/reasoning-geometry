import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect_data import (
    extract_math_answer,
    normalize_math_answer,
    normalize_olympiadbench_answer,
    olympiadbench_answerable,
    save_batch,
)


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


# --- OlympiadBench -----------------------------------------------------------
# Its golds are display strings, so a gold that matches must survive the trip
# through normalization intact; the risk is silently scoring correct answers as
# wrong (parser deflation) rather than the reverse.


def test_olympiadbench_gold_matches_the_boxed_answer_it_describes():
    cases = [
        (r"$\frac{25}{2}$", r"The answer is \boxed{\frac{25}{2}}."),
        ("$2^{1009}$", r"So \boxed{2^{1009}}."),
        ("$k=1$", r"Hence \boxed{1}."),
        (r"$M=\frac{9}{32} \sqrt{2}$", r"Thus \boxed{\dfrac{9}{32}\sqrt{2}}."),
        ("$1003$", r"\boxed{1003}"),
        ("2", r"\boxed{2}"),
    ]

    for gold, generated in cases:
        predicted = normalize_olympiadbench_answer(extract_math_answer(generated))
        assert predicted == normalize_olympiadbench_answer(gold), gold


def test_olympiadbench_normalization_still_separates_different_values():
    assert normalize_olympiadbench_answer("$k=1$") != normalize_olympiadbench_answer("$k=2$")
    assert normalize_olympiadbench_answer(r"$\frac{1}{2}$") != normalize_olympiadbench_answer(r"$\frac{1}{3}$")


def test_olympiadbench_normalization_leaves_bare_answers_where_math500_puts_them():
    # No $, no assignment, no presentation macro -> must agree with the frozen
    # MATH-500 normalizer, so pass rates across the two sets stay comparable.
    for text in ["1003", r"\frac{1}{2}", "2^{1009}", "-7"]:
        assert normalize_olympiadbench_answer(text) == normalize_math_answer(text)


def test_olympiadbench_answerable_keeps_only_single_unit_free_numerical_rows():
    def row(**overrides):
        base = {
            "answer_type": "Numerical",
            "is_multiple_answer": False,
            "unit": None,
            "final_answer": ["2"],
        }
        return {**base, **overrides}

    assert olympiadbench_answerable(row())
    assert not olympiadbench_answerable(row(answer_type="Expression"))
    assert not olympiadbench_answerable(row(answer_type="Tuple"))
    assert not olympiadbench_answerable(row(is_multiple_answer=True))
    assert not olympiadbench_answerable(row(unit="\\mathrm{m}"))
    assert not olympiadbench_answerable(row(final_answer=["2", "3"]))
    assert not olympiadbench_answerable(row(final_answer=None))
