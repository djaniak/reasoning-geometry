from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import layer_cache
from analysis.analyze import load_all_traces

LAYER = 21
DIM = 8


def _write_batch(data_dir: Path, batch_index: int, trace_ids: list[int], rng) -> None:
    """One NPZ shaped like the collector's output: metadata + per-trace members."""
    payload: dict[str, np.ndarray] = {}
    metadata = []
    for position, trace_id in enumerate(trace_ids):
        n_tokens = int(rng.integers(3, 12))
        payload[f"hidden_L{LAYER}_{trace_id}"] = rng.standard_normal(
            (n_tokens, DIM)
        ).astype(np.float32)
        payload[f"entropies_{trace_id}"] = rng.random(n_tokens).astype(np.float32)
        metadata.append(
            {
                "idx": trace_id,
                "trace_id": trace_id,
                "sample_id": position,
                "is_correct": bool(trace_id % 2),
                "gold": "1",
                "predicted": "1",
                "mean_logprob": -0.5,
                "seed": 42,
            }
        )
    payload["metadata"] = np.array(metadata, dtype=object)
    np.savez_compressed(data_dir / f"batch_{batch_index:04d}.npz", **payload)


@pytest.fixture(autouse=True)
def _isolate_cache_root(tmp_path: Path, monkeypatch) -> None:
    """Keep test caches out of the repository's own .layer_cache/."""
    monkeypatch.setenv("LAYER_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    root = tmp_path / "traces"
    root.mkdir()
    _write_batch(root, 0, [0, 1, 2], rng)
    _write_batch(root, 1, [3, 4], rng)
    return root


def _by_trace(traces: list[dict]) -> dict[int, dict]:
    return {int(trace["trace_id"]): trace for trace in traces}


def test_build_then_load_matches_npz_exactly(data_dir: Path) -> None:
    """The whole point: the cached path must not move a single bit."""
    reference = load_all_traces(
        str(data_dir), [LAYER], hidden_dtype=np.float16, use_layer_cache=False
    )
    layer_cache.build(str(data_dir), LAYER, workers=2, progress=False)
    cached = load_all_traces(str(data_dir), [LAYER], hidden_dtype=np.float16)

    assert len(cached) == len(reference) == 5
    ref_map, cache_map = _by_trace(reference), _by_trace(cached)
    assert ref_map.keys() == cache_map.keys()
    for trace_id, ref_trace in ref_map.items():
        got = cache_map[trace_id]["hiddens"][LAYER]
        want = ref_trace["hiddens"][LAYER]
        assert got.dtype == np.float16
        assert got.shape == want.shape
        np.testing.assert_array_equal(got, want)


def test_cached_hidden_states_are_memmap_views(data_dir: Path) -> None:
    """A copy would defeat the purpose, so assert the slice is really mapped."""
    layer_cache.build(str(data_dir), LAYER, workers=2, progress=False)
    traces = load_all_traces(str(data_dir), [LAYER], hidden_dtype=np.float16)
    hidden = traces[0]["hiddens"][LAYER]
    assert isinstance(hidden.base, np.memmap)


def test_float32_request_still_works(data_dir: Path) -> None:
    """Asking for a wider dtype forfeits the mapping but must stay correct."""
    reference = load_all_traces(
        str(data_dir), [LAYER], hidden_dtype=np.float32, use_layer_cache=False
    )
    layer_cache.build(str(data_dir), LAYER, workers=2, progress=False)
    cached = load_all_traces(str(data_dir), [LAYER], hidden_dtype=np.float32)
    for trace_id, ref_trace in _by_trace(reference).items():
        got = _by_trace(cached)[trace_id]["hiddens"][LAYER]
        assert got.dtype == np.float32
        # float32 -> float16 -> float32 is lossless for bf16-origin values, but
        # the fixture is random float32, so compare against the same round trip.
        np.testing.assert_array_equal(
            got, ref_trace["hiddens"][LAYER].astype(np.float16).astype(np.float32)
        )


def test_absent_cache_returns_none(data_dir: Path) -> None:
    assert layer_cache.LayerCache.open(str(data_dir), LAYER) is None


def test_stale_cache_is_refused(data_dir: Path, capsys) -> None:
    """A cache that no longer matches its sources must never be served."""
    layer_cache.build(str(data_dir), LAYER, workers=2, progress=False)
    assert layer_cache.LayerCache.open(str(data_dir), LAYER) is not None

    source = data_dir / "batch_0000.npz"
    os.utime(source, ns=(0, 0))
    assert layer_cache.LayerCache.open(str(data_dir), LAYER) is None
    assert "source NPZ files changed" in capsys.readouterr().out


def test_new_batch_invalidates_cache(data_dir: Path) -> None:
    layer_cache.build(str(data_dir), LAYER, workers=2, progress=False)
    _write_batch(data_dir, 2, [5, 6], np.random.default_rng(1))
    assert layer_cache.LayerCache.open(str(data_dir), LAYER) is None


def test_wrong_layer_is_refused(data_dir: Path) -> None:
    layer_cache.build(str(data_dir), LAYER, workers=2, progress=False)
    assert layer_cache.LayerCache.open(str(data_dir), LAYER + 1) is None


def test_load_falls_back_when_cache_is_stale(data_dir: Path) -> None:
    """Refusing the cache must degrade to the NPZ path, not to an error."""
    reference = load_all_traces(
        str(data_dir), [LAYER], hidden_dtype=np.float16, use_layer_cache=False
    )
    layer_cache.build(str(data_dir), LAYER, workers=2, progress=False)
    os.utime(data_dir / "batch_0001.npz", ns=(0, 0))
    traces = load_all_traces(str(data_dir), [LAYER], hidden_dtype=np.float16)
    for trace_id, ref_trace in _by_trace(reference).items():
        np.testing.assert_array_equal(
            _by_trace(traces)[trace_id]["hiddens"][LAYER],
            ref_trace["hiddens"][LAYER],
        )
