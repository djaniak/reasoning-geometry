import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from merge_deepconf_shards import merge


def _shard(root: Path, name: str, *, prompt_ids, checks, mismatches: int = 0) -> Path:
    """Write one shard in the layout `deepconf_exact.py` produces."""
    directory = root / name
    directory.mkdir(parents=True)
    summaries = [
        {"prompt_id": pid, "trace_id": t, "deepconf_global": float(pid), "deepconf_tail_q20": 0.5}
        for pid in prompt_ids
        for t in range(2)
    ]
    np.savez(
        directory / "deepconf_exact_pilot.npz",
        prompt_ids=np.asarray(prompt_ids, dtype=np.int64),
        exact_token_confidence=np.array(
            [np.full(3, float(row["prompt_id"])) for row in summaries], dtype=object
        ),
        trace_summaries=np.array(summaries, dtype=object),
    )
    (directory / "deepconf_exact_pilot.json").write_text(
        json.dumps(
            {
                "model": "m",
                "roundtrip_token_mismatches": mismatches,
                "reconstruction_checks": checks,
                "shard_index": 0,
                "num_shards": 2,
            }
        )
    )
    return directory


def _checks(*, max_e: float, mean_e: float, n: int) -> dict:
    return {
        "max_entropy_abs_error": max_e,
        "max_sampled_logprob_abs_error": max_e,
        "mean_entropy_abs_error": mean_e,
        "mean_sampled_logprob_abs_error": mean_e,
        "n_error_values": n,
    }


def test_mean_errors_are_weighted_by_token_count_not_averaged(tmp_path):
    """A shard covering 10x more tokens must dominate; a plain average would hide it."""
    a = _shard(tmp_path, "a", prompt_ids=[0], checks=_checks(max_e=0.1, mean_e=0.001, n=1000))
    b = _shard(tmp_path, "b", prompt_ids=[1], checks=_checks(max_e=0.2, mean_e=0.011, n=100))

    meta = merge([a, b], tmp_path / "out", stem="merged")

    checks = meta["reconstruction_checks"]
    assert checks["n_error_values"] == 1100
    # Weighted: (1000*0.001 + 100*0.011)/1100 = 0.0019...; the plain average is 0.006.
    assert checks["mean_entropy_abs_error"] == pytest.approx((1000 * 0.001 + 100 * 0.011) / 1100)
    assert checks["mean_entropy_abs_error"] != pytest.approx(0.006)


def test_max_error_is_the_worst_shard_not_a_mean(tmp_path):
    a = _shard(tmp_path, "a", prompt_ids=[0], checks=_checks(max_e=0.1, mean_e=0.001, n=10))
    b = _shard(tmp_path, "b", prompt_ids=[1], checks=_checks(max_e=0.9, mean_e=0.001, n=10))

    meta = merge([a, b], tmp_path / "out", stem="merged")

    assert meta["reconstruction_checks"]["max_entropy_abs_error"] == pytest.approx(0.9)


def test_token_mismatches_are_summed_so_none_are_lost(tmp_path):
    a = _shard(tmp_path, "a", prompt_ids=[0], checks=_checks(max_e=0.1, mean_e=0.0, n=10), mismatches=2)
    b = _shard(tmp_path, "b", prompt_ids=[1], checks=_checks(max_e=0.1, mean_e=0.0, n=10), mismatches=3)

    assert merge([a, b], tmp_path / "out", stem="merged")["roundtrip_token_mismatches"] == 5


def test_overlapping_shards_raise_instead_of_double_counting(tmp_path):
    a = _shard(tmp_path, "a", prompt_ids=[0, 1], checks=_checks(max_e=0.1, mean_e=0.0, n=10))
    b = _shard(tmp_path, "b", prompt_ids=[1, 2], checks=_checks(max_e=0.1, mean_e=0.0, n=10))

    with pytest.raises(ValueError, match="shards overlap"):
        merge([a, b], tmp_path / "out", stem="merged")


def test_prompts_come_out_sorted_regardless_of_shard_order(tmp_path):
    a = _shard(tmp_path, "a", prompt_ids=[0, 2], checks=_checks(max_e=0.1, mean_e=0.0, n=10))
    b = _shard(tmp_path, "b", prompt_ids=[1, 3], checks=_checks(max_e=0.1, mean_e=0.0, n=10))

    merge([b, a], tmp_path / "out", stem="merged")

    with np.load(tmp_path / "out" / "merged.npz", allow_pickle=True) as data:
        assert list(data["prompt_ids"]) == [0, 1, 2, 3]
        pairs = [(int(r["prompt_id"]), int(r["trace_id"])) for r in data["trace_summaries"]]
    assert pairs == sorted(pairs)


def test_each_trace_confidence_stays_with_its_own_trace(tmp_path):
    """Reordering must not decouple a confidence array from its summary row."""
    a = _shard(tmp_path, "a", prompt_ids=[3], checks=_checks(max_e=0.1, mean_e=0.0, n=10))
    b = _shard(tmp_path, "b", prompt_ids=[1], checks=_checks(max_e=0.1, mean_e=0.0, n=10))

    merge([a, b], tmp_path / "out", stem="merged")

    with np.load(tmp_path / "out" / "merged.npz", allow_pickle=True) as data:
        for row, values in zip(data["trace_summaries"], data["exact_token_confidence"]):
            # The fixture encodes prompt_id into the confidence values.
            assert values[0] == pytest.approx(float(row["prompt_id"]))
