from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA

__all__ = ["flatten_resampled_trajectories", "FunctionalPCAEncoder", "LinearGaussianSequenceModel"]


def flatten_resampled_trajectories(stacked: np.ndarray) -> np.ndarray:
    stacked = np.asarray(stacked, dtype=float)
    if stacked.ndim != 3:
        raise ValueError("stacked must be 3D with shape [n_traces, target_len, n_channels]")
    n_traces, target_len, n_channels = stacked.shape
    return stacked.reshape(n_traces, target_len * n_channels)


class FunctionalPCAEncoder:
    def __init__(self, n_components: int | None = None, **pca_kwargs: Any) -> None:
        self.n_components = n_components
        self.pca_kwargs = dict(pca_kwargs)
        self.pca_: PCA | None = None
        self.trace_shape_: tuple[int, int] | None = None

    def fit(self, stacked: np.ndarray) -> FunctionalPCAEncoder:
        stacked = np.asarray(stacked, dtype=float)
        if stacked.ndim != 3:
            raise ValueError("stacked must be 3D with shape [n_traces, target_len, n_channels]")
        if stacked.shape[0] == 0:
            raise ValueError("stacked must contain at least one trace")

        self.trace_shape_ = tuple(stacked.shape[1:])
        flattened = flatten_resampled_trajectories(stacked)
        self.pca_ = PCA(n_components=self.n_components, **self.pca_kwargs)
        self.pca_.fit(flattened)
        return self

    def transform(self, stacked: np.ndarray) -> np.ndarray:
        if self.pca_ is None or self.trace_shape_ is None:
            raise RuntimeError("FunctionalPCAEncoder must be fit before calling transform")

        stacked = np.asarray(stacked, dtype=float)
        if stacked.ndim != 3:
            raise ValueError("stacked must be 3D with shape [n_traces, target_len, n_channels]")
        if tuple(stacked.shape[1:]) != self.trace_shape_:
            raise ValueError(
                "stacked must have the same [target_len, n_channels] shape as the fitted data"
            )

        flattened = flatten_resampled_trajectories(stacked)
        return self.pca_.transform(flattened)

    def fit_transform(self, stacked: np.ndarray) -> np.ndarray:
        return self.fit(stacked).transform(stacked)


class LinearGaussianSequenceModel:
    def __init__(self, ridge: float = 1e-6) -> None:
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        self.ridge = float(ridge)
        self.transition_coef_: np.ndarray | None = None
        self.transition_cov_inv_: np.ndarray | None = None
        self.transition_logdet_: float | None = None
        self.initial_mean_: np.ndarray | None = None
        self.initial_cov_inv_: np.ndarray | None = None
        self.initial_logdet_: float | None = None

    def _require_fitted(self) -> None:
        if (
            self.transition_coef_ is None
            or self.transition_cov_inv_ is None
            or self.transition_logdet_ is None
            or self.initial_mean_ is None
            or self.initial_cov_inv_ is None
            or self.initial_logdet_ is None
        ):
            raise RuntimeError("LinearGaussianSequenceModel must be fit before scoring")

    def fit(self, sequences: list[np.ndarray]) -> LinearGaussianSequenceModel:
        if not sequences:
            raise ValueError("sequences must contain at least one sequence")

        starts: list[np.ndarray] = []
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []

        for seq in sequences:
            seq_arr = np.asarray(seq, dtype=float)
            if seq_arr.ndim != 2:
                raise ValueError("each sequence must be 2D with shape [length, 2]")
            if seq_arr.shape[1] != 2:
                raise ValueError("each sequence must have exactly 2 channels")
            if seq_arr.shape[0] < 2:
                raise ValueError("each sequence must contain at least two steps")
            if not np.isfinite(seq_arr).all():
                raise ValueError("each sequence must contain only finite values")

            starts.append(seq_arr[0])
            xs.append(seq_arr[:-1])
            ys.append(seq_arr[1:])

        x_mat = np.concatenate(xs, axis=0)
        y_mat = np.concatenate(ys, axis=0)

        x_design = np.concatenate([x_mat, np.ones((x_mat.shape[0], 1))], axis=1)
        reg = self.ridge * np.eye(x_design.shape[1], dtype=float)
        coef = np.linalg.solve(x_design.T @ x_design + reg, x_design.T @ y_mat)

        resid = y_mat - x_design @ coef
        trans_cov = (resid.T @ resid) / max(resid.shape[0], 1)
        trans_cov = trans_cov + self.ridge * np.eye(2, dtype=float)
        trans_cov_inv = np.linalg.inv(trans_cov)
        _, trans_logdet = np.linalg.slogdet(trans_cov)

        start_mat = np.asarray(starts, dtype=float)
        initial_mean = start_mat.mean(axis=0)
        centered = start_mat - initial_mean
        initial_cov = (centered.T @ centered) / max(start_mat.shape[0] - 1, 1)
        initial_cov = initial_cov + self.ridge * np.eye(2, dtype=float)
        initial_cov_inv = np.linalg.inv(initial_cov)
        _, initial_logdet = np.linalg.slogdet(initial_cov)

        self.transition_coef_ = coef
        self.transition_cov_inv_ = trans_cov_inv
        self.transition_logdet_ = float(trans_logdet)
        self.initial_mean_ = initial_mean
        self.initial_cov_inv_ = initial_cov_inv
        self.initial_logdet_ = float(initial_logdet)
        return self

    def score_sequence(self, seq: np.ndarray) -> float:
        self._require_fitted()

        seq_arr = np.asarray(seq, dtype=float)
        if seq_arr.ndim != 2:
            raise ValueError("seq must be 2D with shape [length, 2]")
        if seq_arr.shape[1] != 2:
            raise ValueError("seq must have exactly 2 channels")
        if seq_arr.shape[0] < 2:
            raise ValueError("seq must contain at least two steps")
        if not np.isfinite(seq_arr).all():
            raise ValueError("seq must contain only finite values")

        start = seq_arr[0]
        centered_start = start - self.initial_mean_
        initial_quad = float(centered_start @ self.initial_cov_inv_ @ centered_start)
        initial_ll = -0.5 * (
            2.0 * np.log(2.0 * np.pi) + self.initial_logdet_ + initial_quad
        )

        x_mat = seq_arr[:-1]
        y_mat = seq_arr[1:]
        x_design = np.concatenate([x_mat, np.ones((x_mat.shape[0], 1))], axis=1)
        resid = y_mat - x_design @ self.transition_coef_
        trans_quad = np.einsum("ni,ij,nj->n", resid, self.transition_cov_inv_, resid)
        trans_ll = -0.5 * (
            2.0 * np.log(2.0 * np.pi) + self.transition_logdet_ + trans_quad
        )

        return float(initial_ll + trans_ll.sum())

    def score_sequences(self, sequences: list[np.ndarray]) -> np.ndarray:
        self._require_fitted()
        return np.asarray([self.score_sequence(seq) for seq in sequences], dtype=float)
