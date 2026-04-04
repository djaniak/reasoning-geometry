from __future__ import annotations

import numpy as np


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def build_relative_positions(length: int) -> np.ndarray:
    length = _validate_positive_int("length", length)
    if length == 1:
        return np.array([0.0], dtype=float)
    return np.linspace(0.0, 1.0, length, dtype=float)


def resample_1d_sequence(values: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be 1D")
    if values.size == 0:
        raise ValueError("values must be non-empty")
    target_len = _validate_positive_int("target_len", target_len)
    if values.size == 1:
        return np.repeat(values, target_len)

    src_x = build_relative_positions(values.size)
    dst_x = build_relative_positions(target_len)
    return np.interp(dst_x, src_x, values)


def stack_trace_channels(
    entropies: np.ndarray,
    mahal_distances: np.ndarray,
    target_len: int,
    include_relpos: bool = True,
) -> np.ndarray:
    ent = resample_1d_sequence(entropies, target_len)
    mah = resample_1d_sequence(mahal_distances, target_len)
    channels = [ent, mah]
    if include_relpos:
        channels.append(build_relative_positions(target_len))
    return np.column_stack(channels)
