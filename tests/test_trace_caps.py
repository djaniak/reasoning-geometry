import sys
import textwrap
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.trace_caps import collection_budget, resolve_cap


@contextmanager
def _no_warning():
    """Fail if the body warns -- the cap was validated, so nothing is in doubt."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _pipeline(root: Path, *, qwen_cap: int = 1024, llama_cap: int = 12288,
              locked: bool = True) -> Path:
    """Write a miniature copy of this repo's collect stages."""
    (root / "params.yaml").write_text(
        textwrap.dedent(
            f"""
            bestofn_collected:
              - {{model: qwen, scale: full, max_new_tokens: {qwen_cap}}}
            bestofn_pending:
              - {{model: deepseek_llama, scale: full, max_new_tokens: {llama_cap}}}
            """
        )
    )
    (root / "dvc.yaml").write_text(
        textwrap.dedent(
            """
            stages:
              collect_bestofn_full:
                foreach: ${bestofn_collected}
                do:
                  cmd: >-
                    uv run python collect_data.py --max_new_tokens ${item.max_new_tokens}
                    --output_dir data/${item.model}_bestofn_${item.scale}/math500
                  outs:
                    - data/${item.model}_bestofn_${item.scale}/math500
              collect_bestofn_pending:
                foreach: ${bestofn_pending}
                do:
                  cmd: >-
                    uv run python collect_data.py --max_new_tokens ${item.max_new_tokens}
                    --output_dir data/${item.model}_bestofn_${item.scale}/math500
                  outs:
                    - data/${item.model}_bestofn_${item.scale}/math500
            """
        )
    )
    if locked:
        # Only the finished collect appears in the lock; the pending one has not run.
        (root / "dvc.lock").write_text(
            textwrap.dedent(
                f"""
                stages:
                  collect_bestofn_full@0:
                    cmd: uv run python collect_data.py --max_new_tokens {qwen_cap}
                      --output_dir data/qwen_bestofn_full/math500
                    outs:
                    - path: data/qwen_bestofn_full/math500
                      md5: abc.dir
                """
            )
        )
    qwen = root / "data" / "qwen_bestofn_full" / "math500"
    qwen.mkdir(parents=True)
    (root / "data" / "deepseek_llama_bestofn_full" / "math500").mkdir(parents=True)
    return qwen


def test_missing_cap_and_missing_record_raises_instead_of_counting_nothing():
    with pytest.raises(ValueError, match="max_new_tokens is required"):
        resolve_cap(None, lengths=[100, 1024])


def test_cap_from_another_model_contradicts_the_pipeline_record(tmp_path):
    data_dir = _pipeline(tmp_path)

    # Qwen traces (budget 1024) handed DeepSeek's 8192.
    with pytest.raises(ValueError, match="contradicts the budget"):
        resolve_cap(8192, data_dir=data_dir, lengths=[79, 500, 1024])


def test_a_valid_cap_above_every_observed_length_is_accepted(tmp_path):
    """A clean collect that never hits its budget is not a mismatch."""
    _pipeline(tmp_path)
    data_dir = tmp_path / "data" / "deepseek_llama_bestofn_full" / "math500"

    with _no_warning():
        cap = resolve_cap(12288, data_dir=data_dir, lengths=[1200, 4000, 9000])

    assert cap.value == 12288
    assert cap.verified


def test_the_budget_is_recovered_when_the_caller_passes_none(tmp_path):
    data_dir = _pipeline(tmp_path)

    cap = resolve_cap(None, data_dir=data_dir)

    assert cap.value == 1024
    assert cap.sources == ("dvc.lock", "dvc.yaml/params.yaml")


def test_a_pending_collect_is_covered_by_params_alone(tmp_path):
    _pipeline(tmp_path)
    data_dir = tmp_path / "data" / "deepseek_llama_bestofn_full" / "math500"

    assert collection_budget(data_dir) == {"dvc.yaml/params.yaml": 12288}


def test_disagreeing_authoritative_records_raise(tmp_path):
    # params.yaml was edited to 2048 after the collect ran at 1024.
    data_dir = _pipeline(tmp_path, qwen_cap=1024)
    (tmp_path / "params.yaml").write_text(
        "bestofn_collected:\n  - {model: qwen, scale: full, max_new_tokens: 2048}\n"
        "bestofn_pending: []\n"
    )

    with pytest.raises(ValueError, match="records disagree"):
        resolve_cap(None, data_dir=data_dir)


def test_an_unrecorded_directory_leaves_the_cap_unvalidated(tmp_path):
    data_dir = tmp_path / "scratch"
    data_dir.mkdir()

    with pytest.warns(UserWarning, match="cap is unvalidated"):
        cap = resolve_cap(8192, data_dir=data_dir, lengths=[79, 500, 1024])

    assert cap.value == 8192
    assert not cap.verified
    assert "heuristic" in cap.provenance


def test_an_unrecorded_binding_cap_is_not_warned_about(tmp_path):
    data_dir = tmp_path / "scratch"
    data_dir.mkdir()

    with _no_warning():
        cap = resolve_cap(1024, data_dir=data_dir, lengths=[79, 500, 1024])

    assert cap.value == 1024
    assert not cap.verified


def test_none_lengths_are_ignored():
    assert resolve_cap(1024, lengths=[None, 1024, None]).value == 1024


def test_no_observed_lengths_cannot_be_checked():
    assert resolve_cap(1024, lengths=[]).value == 1024
