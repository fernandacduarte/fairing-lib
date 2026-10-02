"""Tests for the discrete Laplacians (issues #5, #6)."""

import numpy as np
import pytest

from fairing.laplacian import (cotan_laplacian, face_areas, mean_curvature, uniform_laplacian,
                               vertex_areas)
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


# ---------------------------------------------------------------------------
# Cotangent Laplacian, Eq. (3.11)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mesh", [irregular_grid(8, 7, seed=2), uv_sphere(6, 9), tube(8, 4)])
def test_cotan_matrix_structure(mesh):
    V, F = mesh
    L, D, M = cotan_laplacian(V, F)
    assert np.allclose((M - M.T).toarray(), 0)              # M is symmetric
    assert np.allclose(M.sum(axis=1), 0)                    # rows sum to 0
    assert np.allclose((L - D @ M).toarray(), 0)            # L = D M
    assert np.allclose(D.diagonal(), 1 / (2 * vertex_areas(V, F)))   # wi = 1 / (2 Ai)


@pytest.mark.parametrize("mesh", [irregular_grid(8, 7, seed=2), uv_sphere(6, 9), tube(8, 4)])
def test_barycentric_areas_tile_the_mesh(mesh):
    V, F = mesh
    assert np.isclose(vertex_areas(V, F).sum(), face_areas(V, F).sum())


def test_cotan_weights_on_unit_right_triangles():
    # grid(3, 3, size=2): spacing 1, every triangle is a right isosceles triangle.
    # Axis edges are opposite 45° angles (cot = 1); diagonals are opposite 90° (cot = 0).
    V, F = grid(3, 3, size=2.0)
    _, _, M = cotan_laplacian(V, F)
    center = 4
    assert np.isclose(M[center, 5], 2) and np.isclose(M[center, 7], 2)   # two 45° angles each
    assert np.isclose(M[center, 8], 0)                                   # diagonal: 90° + 90°


def test_cotan_linear_precision_on_irregular_planar_grid():
    # Linear precision: L f = 0 at interior vertices for every linear function f.
    # Applied to the positions of a planar mesh, Lx = 0 (unlike the uniform Laplacian).
    V, F = irregular_grid(9, 9, jitter=0.15, seed=0)
    L, _, _ = cotan_laplacian(V, F)
    inner = interior(V, F)
    assert np.abs(L @ V)[inner].max() < 1e-10
    f = 3 * V[:, 0] - 2 * V[:, 1] + 5
    assert np.abs(L @ f)[inner].max() < 1e-10


def test_mean_curvature_of_sphere_converges_away_from_poles():
    # Eq. (3.13): H = ||Lx|| / 2 should approach 1/R. The poles are excluded: there the
    # barycentric area is 4/3 of the Voronoi area and H stays 25% low (see phase-2 notes).
    R, errors = 2.0, []
    for n_lat in (10, 20, 40):
        V, F = uv_sphere(n_lat, 2 * n_lat, radius=R)
        H = mean_curvature(V, F)
        band = np.abs(V[:, 2]) < 0.7 * R
        errors.append(np.abs(H * R - 1)[band].max())
    assert errors[-1] < 1e-3
    assert errors[0] > errors[1] > errors[2]                # shrinks with resolution


def test_cotan_Lx_points_inward_on_sphere():
    # Delta x = -2H n: on a sphere, Lx points toward the center (opposite the outward normal).
    V, F = uv_sphere(12, 24)
    L, _, _ = cotan_laplacian(V, F)
    assert np.all(np.einsum("ij,ij->i", L @ V, V) < 0)


def test_barycentric_pole_area_gives_three_quarters_of_H():
    # Known limitation of barycentric cells: at a pole (apex of a fan of thin triangles)
    # the barycentric cell is about 4/3 of the Voronoi cell, so H -> (3/4)(1/R).
    V, F = uv_sphere(80, 160, radius=1.0)
    H = mean_curvature(V, F)
    assert np.isclose(H[0], 0.75, atol=0.01)
