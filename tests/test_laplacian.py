"""Tests for the discrete Laplacians (issues #5, #6)."""

import numpy as np
import pytest

from fairing.laplacian import uniform_laplacian
from fairing.mesh import boundary_vertices, grid, irregular_grid, one_rings, tube, uv_sphere


def interior(V, F):
    return np.setdiff1d(np.arange(len(V)), boundary_vertices(F))


# ---------------------------------------------------------------------------
# Uniform Laplacian, Eq. (3.10)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mesh", [irregular_grid(8, 7, seed=2), uv_sphere(6, 9), tube(8, 4)])
def test_uniform_matrix_structure(mesh):
    V, F = mesh
    L, D, M = uniform_laplacian(V, F)
    assert (M - M.T).nnz == 0                               # M is symmetric
    assert np.allclose(M.sum(axis=1), 0)                    # rows sum to 0: L(const) = 0
    assert np.allclose((L - D @ M).toarray(), 0)            # L = D M
    deg = np.array([len(r) for r in one_rings(F, len(V))])
    assert np.allclose(D.diagonal(), 1 / deg)               # wi = 1 / deg(vi)


def test_uniform_Lx_points_to_neighbor_centroid():
    V, F = irregular_grid(6, 6, seed=4)
    L, _, _ = uniform_laplacian(V, F)
    LV = L @ V
    for i, ring in enumerate(one_rings(F, len(V))):
        assert np.allclose(LV[i], V[ring].mean(axis=0) - V[i])


def test_uniform_is_zero_on_regular_grid_interior():
    # Every neighbor has a mirror partner through the vertex (hexagonal lattice).
    V, F = grid(9, 9)
    L, _, _ = uniform_laplacian(V, F)
    assert np.abs(L @ V)[interior(V, F)].max() < 1e-12


def test_uniform_is_not_zero_on_irregular_planar_grid():
    # The flaw (Sec. 3.3.4): the mesh is planar, so the true Laplace-Beltrami of the
    # positions is 0 (Eq. 3.7), but the uniform Lx is not.
    V, F = irregular_grid(9, 9, jitter=0.15, seed=0)
    L, _, _ = uniform_laplacian(V, F)
    norms = np.linalg.norm(L @ V, axis=1)[interior(V, F)]
    assert norms.min() > 1e-6
    assert np.allclose((L @ V)[:, 2], 0)                    # the error is purely tangential
