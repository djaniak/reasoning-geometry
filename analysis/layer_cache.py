"""Flat, memory-mappable caches for one layer's hidden states.

Why this exists
---------------
``load_all_traces`` materialises every trace's hidden states into anonymous
memory.  For the long-trace models that is 83 GiB (deepseek, layer 21) and
93 GiB (deepseek_llama, layer 24) held resident across all five CV folds, on
top of a ~29 GB per-fold reference fit.  Anonymous memory is not evictable, so
on a contended host the kernel can only make room by swapping -- and when swap
is exhausted the process spends its time in direct reclaim rather than in
BLAS.  Measured on ``argon``: 232 s of kernel time per 90 s of wall clock, and
a load that takes 3.3 min on an idle box projected to 9.5 hours.

A cache turns that memory into *page cache*, which the kernel can evict and
re-read on demand.  The worst case degrades to streaming off NVMe; it no
longer degrades to thrashing swap.  The hidden states are stored exactly as
the loader would have cast them -- float16, which round-trips the bf16 forward
pass without loss -- so slices handed back from the cache are bit-identical to
the arrays the NPZ path produces.

Layout, under ``<data_dir>/.layer_cache/``::

    L21.npy         (total_tokens, dim) float16, written via open_memmap
    L21.index.json  manifest: source fingerprints + per-member (offset, length)

The manifest fingerprints every source NPZ by name, size and mtime.  If any of
them has moved the cache is refused and the caller falls back to the NPZ path,
because silently scoring against stale hidden states is the one failure this
must not have.
"""
from __future__ import annotations

import json
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import numpy.lib.format as npformat

#: Deliberately *outside* the data tree.  The collected NPZ directories are DVC
#: outputs, so a cache written beside them would change the directory hash and
#: surface as a modified out against hundreds of GB that no remote can restore.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT_ENV = "LAYER_CACHE_DIR"
DEFAULT_CACHE_ROOT = os.path.join(REPO_ROOT, ".layer_cache")
CACHE_VERSION = 1
#: The cache stores what ``hidden_dtype=float16`` would have produced.  Callers
#: asking for anything else get a converted copy, which forfeits the point.
CACHE_DTYPE = np.dtype("float16")


def cache_root() -> str:
    return os.environ.get(CACHE_ROOT_ENV) or DEFAULT_CACHE_ROOT


def cache_dir_for(data_dir: str) -> str:
    """Mirror the data directory's layout under the cache root.

    ``data/deepseek_bestofn_full/math500`` caches to
    ``.layer_cache/data/deepseek_bestofn_full/math500``.  Data directories
    outside the repository fall back to a flattened slug of their absolute
    path, so two of them can never collide.
    """
    absolute = os.path.abspath(data_dir)
    relative = os.path.relpath(absolute, REPO_ROOT)
    if relative.startswith(os.pardir):
        relative = absolute.strip(os.sep).replace(os.sep, "__")
    return os.path.join(cache_root(), relative)


def cache_paths(data_dir: str, layer: int) -> tuple[str, str]:
    root = cache_dir_for(data_dir)
    return (
        os.path.join(root, f"L{layer}.npy"),
        os.path.join(root, f"L{layer}.index.json"),
    )


def source_npz_paths(data_dir: str) -> list[str]:
    """The NPZ files, in the exact order ``load_all_traces`` walks them."""
    return [
        os.path.join(data_dir, fname)
        for fname in sorted(os.listdir(data_dir))
        if fname.endswith(".npz")
    ]


def fingerprint(path: str) -> dict:
    stat = os.stat(path)
    return {
        "name": os.path.basename(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _member_pattern(layer: int) -> re.Pattern:
    return re.compile(rf"^hidden_L{layer}_(\d+)\.npy$")


def _read_member_header(zf: zipfile.ZipFile, name: str) -> tuple[tuple[int, ...], np.dtype]:
    """Shape and dtype of one NPZ member, decompressing only its header."""
    with zf.open(name) as handle:
        version = npformat.read_magic(handle)
        if version == (1, 0):
            shape, _fortran, dtype = npformat.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _fortran, dtype = npformat.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported .npy version {version} in {name}")
    return shape, dtype


def scan_batch(path: str, layer: int) -> list[tuple[str, int, int]]:
    """Return ``(member_stem, n_tokens, dim)`` for each layer member in one NPZ."""
    pattern = _member_pattern(layer)
    found = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not pattern.match(name):
                continue
            shape, dtype = _read_member_header(zf, name)
            if len(shape) != 2:
                raise ValueError(f"{path}:{name} has shape {shape}, expected 2-D")
            if dtype.kind != "f":
                raise ValueError(f"{path}:{name} has non-float dtype {dtype}")
            found.append((name[: -len(".npy")], int(shape[0]), int(shape[1])))
    # Sort for a deterministic layout; lookup is by name, so order is free.
    found.sort()
    return found


def build(
    data_dir: str,
    layer: int,
    *,
    workers: int = 8,
    progress: bool = True,
) -> tuple[str, str]:
    """Decompress one layer into a flat float16 memmap plus its manifest.

    Two passes.  The first reads only .npy headers -- a few hundred bytes per
    member -- to size the output and assign offsets; the second decompresses
    the data.  Peak memory is one NPZ batch per worker, not the whole layer.
    """
    paths = source_npz_paths(data_dir)
    if not paths:
        raise ValueError(f"no NPZ files in {data_dir}")

    def _say(message: str) -> None:
        if progress:
            print(f"[layer-cache] {message}", flush=True)

    _say(f"[1/3] Scanning {len(paths)} batches for layer {layer} headers")
    layout: dict[str, dict] = {}
    offset = 0
    dim: int | None = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scan_batch, path, layer): path for path in paths}
        scanned = {futures[f]: f.result() for f in as_completed(futures)}
    for path in paths:  # deterministic offsets: assign in sorted path order
        batch = os.path.basename(path)
        for stem, n_tokens, member_dim in scanned[path]:
            if dim is None:
                dim = member_dim
            elif member_dim != dim:
                raise ValueError(
                    f"{batch}:{stem} has dim {member_dim}, expected {dim}"
                )
            layout[f"{batch}|{stem}"] = {"offset": offset, "length": n_tokens}
            offset += n_tokens
    if dim is None:
        raise ValueError(f"no layer-{layer} members found under {data_dir}")

    total_tokens = offset
    nbytes = total_tokens * dim * CACHE_DTYPE.itemsize
    _say(
        f"[2/3] Writing {len(layout)} traces, {total_tokens:,} tokens x {dim} "
        f"= {nbytes / 2**30:.1f} GiB float16"
    )

    array_path, index_path = cache_paths(data_dir, layer)
    os.makedirs(os.path.dirname(array_path), exist_ok=True)
    tmp_array = array_path + ".partial"
    out = npformat.open_memmap(
        tmp_array, mode="w+", dtype=CACHE_DTYPE, shape=(total_tokens, dim)
    )

    pattern = _member_pattern(layer)

    def _fill(path: str) -> int:
        batch = os.path.basename(path)
        written = 0
        with np.load(path, allow_pickle=False) as data:
            for name in data.files:
                if not pattern.match(f"{name}.npy"):
                    continue
                slot = layout[f"{batch}|{name}"]
                block = data[name].astype(CACHE_DTYPE, copy=False)
                out[slot["offset"] : slot["offset"] + slot["length"]] = block
                written += 1
        return written

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fill, path): path for path in paths}
        for future in as_completed(futures):
            future.result()
            done += 1
            if progress:
                print(
                    f"[layer-cache]   batch {done}/{len(paths)}",
                    end="\r",
                    flush=True,
                )
    out.flush()
    del out
    if progress:
        print(flush=True)

    _say("[3/3] Writing manifest and committing")
    manifest = {
        "version": CACHE_VERSION,
        "layer": int(layer),
        "dtype": CACHE_DTYPE.name,
        "dim": int(dim),
        "total_tokens": int(total_tokens),
        "sources": [fingerprint(path) for path in paths],
        "members": layout,
    }
    tmp_index = index_path + ".partial"
    with open(tmp_index, "w") as handle:
        json.dump(manifest, handle)
    # Rename the array first: a manifest is only ever visible with its data.
    os.replace(tmp_array, array_path)
    os.replace(tmp_index, index_path)
    _say(f"Done: {array_path}")
    return array_path, index_path


class LayerCache:
    """Read-side view of a built cache: name -> memmap slice, no copy."""

    def __init__(self, array: np.memmap, members: dict[str, dict], layer: int, dim: int):
        self._array = array
        self._members = members
        self.layer = int(layer)
        self.dim = int(dim)

    def __len__(self) -> int:
        return len(self._members)

    @property
    def nbytes(self) -> int:
        return int(self._array.nbytes)

    def get(self, batch_name: str, stem: str) -> np.ndarray | None:
        """A float16 view of one trace, or None if this cache lacks it.

        The returned array is a slice of the memmap, so it costs no anonymous
        memory and keeps the mapping alive for as long as any caller holds it.
        """
        slot = self._members.get(f"{batch_name}|{stem}")
        if slot is None:
            return None
        start = slot["offset"]
        return self._array[start : start + slot["length"]]

    @classmethod
    def open(cls, data_dir: str, layer: int, *, verbose: bool = True) -> "LayerCache | None":
        """Open a validated cache, or return None to fall back to the NPZ path."""
        array_path, index_path = cache_paths(data_dir, layer)
        if not (os.path.exists(array_path) and os.path.exists(index_path)):
            return None

        def _refuse(reason: str) -> None:
            if verbose:
                print(
                    f"  Ignoring layer cache {array_path} ({reason}); "
                    f"falling back to NPZ",
                    flush=True,
                )

        try:
            with open(index_path) as handle:
                manifest = json.load(handle)
        except (OSError, ValueError) as error:
            _refuse(f"unreadable manifest: {error}")
            return None

        if manifest.get("version") != CACHE_VERSION:
            _refuse(f"version {manifest.get('version')} != {CACHE_VERSION}")
            return None
        if int(manifest.get("layer", -1)) != int(layer):
            _refuse(f"built for layer {manifest.get('layer')}")
            return None
        if manifest.get("dtype") != CACHE_DTYPE.name:
            _refuse(f"dtype {manifest.get('dtype')} != {CACHE_DTYPE.name}")
            return None

        # A cache that no longer matches its sources is worse than no cache.
        try:
            current = [fingerprint(path) for path in source_npz_paths(data_dir)]
        except OSError as error:
            _refuse(f"cannot stat sources: {error}")
            return None
        if current != manifest.get("sources"):
            _refuse("source NPZ files changed since it was built")
            return None

        try:
            array = np.load(array_path, mmap_mode="r")
        except (OSError, ValueError) as error:
            _refuse(f"cannot map array: {error}")
            return None
        if array.dtype != CACHE_DTYPE or array.shape[1] != int(manifest["dim"]):
            _refuse(f"array is {array.shape} {array.dtype}")
            return None

        return cls(array, manifest["members"], layer, int(manifest["dim"]))


def open_caches(
    data_dir: str, layers: list[int], *, verbose: bool = True
) -> dict[int, "LayerCache"]:
    """Open whichever requested layers have a valid cache. Missing ones are fine."""
    caches = {}
    for layer in layers:
        cache = LayerCache.open(data_dir, layer, verbose=verbose)
        if cache is not None:
            caches[layer] = cache
    return caches


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data_dir", required=True, help="directory of collected NPZ batches")
    parser.add_argument(
        "--layer",
        required=True,
        type=int,
        action="append",
        dest="layers",
        help="repeatable; the layer to cache (the one the analysis reads)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even if a valid cache is already present",
    )
    args = parser.parse_args()

    for layer in args.layers:
        if not args.force and LayerCache.open(args.data_dir, layer, verbose=False):
            print(f"[layer-cache] layer {layer}: valid cache present, skipping")
            continue
        build(args.data_dir, layer, workers=args.workers)


if __name__ == "__main__":
    main()
