import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry.rmd_qmd_geometry import choose_plane, mahal


def _rotation(dim: int, seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(dim, dim)))
    return q


def _covariance(scales: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return rotation @ np.diag(scales) @ rotation.T


def _log_ratio(direction: np.ndarray, sigma_c: np.ndarray, sigma_i: np.ndarray) -> float:
    return float(abs(np.log((direction @ sigma_c @ direction)
                            / (direction @ sigma_i @ direction))))


def test_mahal_matches_the_square_rooted_quadratic_form():
    """The features are differences of distances, not of squared distances."""
    rng = np.random.default_rng(0)
    cov = _covariance(np.array([4.0, 1.0, 0.25]), _rotation(3, 1))
    precision = np.linalg.inv(cov)
    mu = rng.normal(size=3)
    points = rng.normal(size=(7, 3))

    got = mahal(points, mu, precision)

    diff = points - mu
    expected = np.sqrt(np.einsum("ij,jk,ik->i", diff, precision, diff))
    np.testing.assert_allclose(got, expected, rtol=1e-10)
    assert np.all(got >= 0)


def test_choose_plane_returns_an_orthonormal_basis_spanning_the_mean_shift():
    dim = 12
    rotation = _rotation(dim, 2)
    sigma_c = _covariance(np.linspace(1.0, 3.0, dim), rotation)
    sigma_i = _covariance(np.linspace(3.0, 1.0, dim), rotation)
    mu_c = np.zeros(dim)
    mu_i = _rotation(dim, 3)[:, 0] * 2.0

    basis = choose_plane(mu_c, mu_i, sigma_c, sigma_i)

    assert basis.shape == (2, dim)
    np.testing.assert_allclose(basis @ basis.T, np.eye(2), atol=1e-9)
    # e1 is the class-contrast direction, up to sign.
    np.testing.assert_allclose(
        np.abs(basis[0] @ (mu_i - mu_c)) / np.linalg.norm(mu_i - mu_c), 1.0, atol=1e-9
    )


def test_choose_plane_is_extremal_when_the_covariances_do_not_commute():
    """The regression this guards: a symmetric solver on a non-symmetric product.

    ``inv(Sigma_inc) @ Sigma_cor`` is symmetric only when the two covariances
    commute, so a fixture with shared eigenvectors cannot detect the defect --
    ``np.linalg.eigh`` happens to be right there.  With genuinely different
    eigenbases the unfixed code reaches |log ratio| ~0.50 against the attainable
    ~1.95, i.e. it returns an essentially uninformative axis.
    """
    dim = 10
    sigma_c = _covariance(np.linspace(1.0, 20.0, dim), _rotation(dim, 4))
    sigma_i = _covariance(np.linspace(20.0, 1.0, dim), _rotation(dim, 11))
    assert np.linalg.norm(sigma_c @ sigma_i - sigma_i @ sigma_c) > 1.0

    mu_c = np.zeros(dim)
    mu_i = _rotation(dim, 3)[:, 0] * 3.0

    e1, e2 = choose_plane(mu_c, mu_i, sigma_c, sigma_i)
    attained = _log_ratio(e2, sigma_c, sigma_i)

    assert abs(e1 @ e2) < 1e-8
    assert attained > 1.9

    # No direction in the complement of e1 does better.
    complement = np.linalg.svd(e1[None], full_matrices=True)[2][1:]
    rng = np.random.default_rng(7)
    candidates = rng.normal(size=(4000, dim - 1)) @ complement
    best = max(_log_ratio(v / np.linalg.norm(v), sigma_c, sigma_i) for v in candidates)
    assert best <= attained + 1e-9


def test_choose_plane_recovers_a_planted_shape_axis():
    """Sanity case: the mean shift carries no shape information at all."""
    dim = 10
    rotation = _rotation(dim, 4)
    scales_c = np.ones(dim)
    scales_c[7] = 25.0  # the correct class is 5x wider along the planted axis
    sigma_c = _covariance(scales_c, rotation)
    sigma_i = _covariance(np.ones(dim), rotation)

    # Deliberately orthogonal to the planted axis: invisible to any hyperplane.
    e1, e2 = choose_plane(np.zeros(dim), rotation[:, 0] * 3.0, sigma_c, sigma_i)

    assert abs(e2 @ rotation[:, 7]) > 0.99
    assert (e2 @ sigma_c @ e2) / (e2 @ sigma_i @ e2) == pytest.approx(25.0, rel=1e-6)


def test_choose_plane_keeps_e2_usable_when_the_extremal_axis_is_the_mean_shift():
    """Searching the whole space and projecting afterwards collapses here.

    The global extremum lies exactly along e1, so the projection leaves a zero
    vector; searching inside the complement returns the best axis available.
    """
    dim = 8
    rotation = _rotation(dim, 5)
    scales_c = np.ones(dim)
    scales_c[0] = 100.0  # extremal ratio lies exactly along the mean shift
    scales_c[3] = 9.0    # the best available direction inside the complement
    sigma_c = _covariance(scales_c, rotation)
    sigma_i = _covariance(np.ones(dim), rotation)

    e1, e2 = choose_plane(np.zeros(dim), rotation[:, 0] * 2.0, sigma_c, sigma_i)

    assert abs(e1 @ e2) < 1e-8
    assert np.all(np.isfinite(e2))
    assert abs(e2 @ rotation[:, 3]) > 0.99
    assert (e2 @ sigma_c @ e2) / (e2 @ sigma_i @ e2) == pytest.approx(9.0, rel=1e-6)


def test_choose_plane_rejects_coincident_means():
    dim = 5
    identity = np.eye(dim)
    with pytest.raises(ValueError, match="coincide"):
        choose_plane(np.zeros(dim), np.zeros(dim), identity, identity)
