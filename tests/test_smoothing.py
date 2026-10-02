"""Tests for diffusion-flow smoothing (issues #7, #8)."""

import numpy as np
import pytest

from fairing.laplacian import LAPLACIANS, uniform_laplacian
from fairing.mesh import add_noise, boundary_vertices, irregular_grid, one_rings, uv_sphere
from fairing.smoothing import explicit_smoothing, explicit_step_limit, roughness


def noisy_sphere(n_lat=10, sigma=0.01, seed=0):
    V, F = uv_sphere(n_lat, 2 * n_lat)
    return add_noise(V, sigma, seed), F


# ---------------------------------------------------------------------------
# Explicit smoothing, Sec. 4.2
# ---------------------------------------------------------------------------

def test_add_noise_is_reproducible_and_has_the_right_size():
    V, _ = uv_sphere(20, 40)
    N1, N2 = add_noise(V, 0.01, seed=3), add_noise(V, 0.01, seed=3)
    assert np.array_equal(N1, N2)
    assert np.isclose((N1 - V).std(), 0.01, rtol=0.1)


@pytest.mark.parametrize("laplacian", ["uniform", "cotan"])
def test_step_limit_matches_dense_eigenvalues(laplacian):
    V, F = noisy_sphere(6)
    L = LAPLACIANS[laplacian](V, F)[0].toarray()
    mu = np.linalg.eigvals(L)
    assert np.abs(mu.imag).max() < 1e-8                      # real: L is similar to a symmetric matrix
    assert np.isclose(explicit_step_limit(V, F, laplacian=laplacian), 2 / abs(mu.real.min()))


def test_uniform_step_limit_is_at_least_one():
    # mu_min >= -2 for the uniform Laplacian, so h * lam <= 1 is always stable.
    V, F = noisy_sphere(10)
    assert explicit_step_limit(V, F, lam=1.0, laplacian="uniform") >= 1.0


def test_uniform_step_with_h_lam_one_moves_to_neighbor_centroid():
    V, F = irregular_grid(6, 6, seed=1)
    V1 = explicit_smoothing(V, F, h=1.0, lam=1.0, n_iter=1, laplacian="uniform")
    rings = one_rings(F, len(V))
    for i in np.setdiff1d(np.arange(len(V)), boundary_vertices(F)):
        assert np.allclose(V1[i], V[rings[i]].mean(axis=0))


@pytest.mark.parametrize("laplacian, n_iter", [("uniform", 10), ("cotan", 50)])
def test_smoothing_removes_most_of_the_noise(laplacian, n_iter):
    # A coarse mesh is "rough" even without noise (faceting), so measure the
    # roughness *added* by the noise, relative to the clean sphere.
    V0, F = uv_sphere(16, 32)
    V = add_noise(V0, 0.02, seed=0)
    h = 0.5 * explicit_step_limit(V, F, laplacian=laplacian)
    Vs = explicit_smoothing(V, F, h, n_iter=n_iter, laplacian=laplacian)
    excess = lambda X: roughness(X, F) - roughness(V0, F)
    assert excess(Vs) < 0.5 * excess(V)


def test_step_above_the_limit_blows_up():
    V, F = noisy_sphere(10, sigma=0.01)
    limit = explicit_step_limit(V, F, laplacian="uniform")
    stable = explicit_smoothing(V, F, 0.95 * limit, n_iter=300, laplacian="uniform")
    unstable = explicit_smoothing(V, F, 1.05 * limit, n_iter=300, laplacian="uniform")
    assert np.abs(stable).max() <= np.abs(V).max()          # bounded (it even shrinks)
    assert np.abs(unstable).max() > 1e3                      # the worst mode grows like 1.1^n


def test_roughness_of_a_flat_mesh_is_zero():
    V, F = irregular_grid(5, 5, seed=0)
    assert roughness(V, F) < 1e-12
