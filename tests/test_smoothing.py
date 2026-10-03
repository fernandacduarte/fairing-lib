"""Tests for diffusion-flow smoothing (issues #7, #8)."""

import numpy as np
import pytest
from scipy.sparse import diags as sp_diags

from fairing.laplacian import LAPLACIANS, uniform_laplacian
from fairing.mesh import (add_noise, boundary_vertices, irregular_grid, irregular_sphere, one_rings,
                          uv_sphere)
from fairing.smoothing import explicit_smoothing, explicit_step_limit, implicit_smoothing, roughness


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


# ---------------------------------------------------------------------------
# Implicit smoothing, Sec. 4.2 and App. A.1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("laplacian", ["uniform", "cotan"])
def test_implicit_system_is_symmetric_positive_definite(laplacian):
    V, F = noisy_sphere(6)
    _, D, M = LAPLACIANS[laplacian](V, F)
    A = (sp_diags(1 / D.diagonal()) - 0.1 * M).toarray()       # D^-1 - h lam M, h lam = 0.1
    assert np.allclose(A, A.T)
    np.linalg.cholesky(A)                                       # raises if not positive definite


@pytest.mark.parametrize("laplacian", ["uniform", "cotan"])
def test_implicit_step_solves_the_unsymmetric_system(laplacian):
    # The symmetric system (D^-1 - h lam M) x' = D^-1 x is the same as (I - h lam L) x' = x.
    V, F = noisy_sphere(8)
    h = 0.05
    V1 = implicit_smoothing(V, F, h, laplacian=laplacian)
    L = LAPLACIANS[laplacian](V, F)[0]
    assert np.allclose(V1 - h * (L @ V1), V, atol=1e-10)


@pytest.mark.parametrize("laplacian", ["uniform", "cotan"])
def test_implicit_matches_explicit_for_tiny_steps(laplacian):
    # Both are first-order accurate: for small h they differ by O(h^2).
    V, F = noisy_sphere(8)
    h = 1e-3 * explicit_step_limit(V, F, laplacian=laplacian)
    step = lambda f: f(V, F, h, laplacian=laplacian) - V
    explicit, implicit = step(explicit_smoothing), step(implicit_smoothing)
    assert np.abs(implicit - explicit).max() < 1e-2 * np.abs(explicit).max()


@pytest.mark.parametrize("laplacian, factor", [("uniform", 10), ("cotan", 300)])
def test_implicit_is_stable_far_above_h_max(laplacian, factor):
    V0, F = uv_sphere(16, 32)
    V = add_noise(V0, 0.02, seed=0)
    h = factor * explicit_step_limit(V, F, laplacian=laplacian)
    Vs = implicit_smoothing(V, F, h, laplacian=laplacian)
    assert np.all(np.isfinite(Vs)) and np.abs(Vs).max() <= np.abs(V).max()
    excess = lambda X: roughness(X, F) - roughness(V0, F)
    assert excess(Vs) < 0.5 * excess(V)                        # one step removes most of the noise


def test_irregular_sphere_is_an_exact_sphere_with_irregular_triangles():
    V, F = irregular_sphere(12, 24, radius=2.0, jitter=0.15, seed=1)
    assert np.allclose(np.linalg.norm(V, axis=1), 2.0)
    assert not np.allclose(V, uv_sphere(12, 24, radius=2.0)[0])
    with pytest.raises(ValueError):
        irregular_sphere(12, 24, jitter=0.2)


def triangle_angles(V, F):
    angles = []
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u, v = V[F[:, b]] - V[F[:, a]], V[F[:, c]] - V[F[:, a]]
        cos = np.einsum("ij,ij->i", u, v) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
        angles.append(np.degrees(np.arccos(cos)))
    return np.stack(angles, axis=1)


def test_cotan_preserves_triangle_shapes_uniform_does_not():
    # Fig. 4.6: on an exact sphere with irregular triangles, the cotangent flow only
    # shrinks the sphere (Lx is normal), while the uniform flow also slides vertices
    # tangentially toward a more regular triangulation.
    V, F = irregular_sphere(24, 48, jitter=0.15, seed=0)
    A0 = triangle_angles(V, F)
    change = lambda lap, h: np.abs(triangle_angles(implicit_smoothing(V, F, h, laplacian=lap), F) - A0).mean()
    assert change("cotan", 0.05) < 0.5          # degrees
    assert change("uniform", 5.0) > 5.0
