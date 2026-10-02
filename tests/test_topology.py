"""Tests for the topology helpers (issue #4)."""

import numpy as np
import pytest

from fairing.mesh import (adjacency, boundary_vertices, edges, grid, k_ring, one_rings,
                          ring_distance, tube, uv_sphere)


def test_edges_are_unique_and_sorted():
    E = edges(grid(4, 3)[1])
    assert np.all(E[:, 0] < E[:, 1])
    assert len(np.unique(E, axis=0)) == len(E)


@pytest.mark.parametrize("mesh, chi", [(grid(6, 5), 1), (uv_sphere(8, 12), 2), (tube(10, 5), 0)])
def test_euler_characteristic(mesh, chi):
    # V - E + F: 1 for a disk (grid), 2 for a sphere, 0 for an open cylinder (tube)
    V, F = mesh
    assert len(V) - len(edges(F)) + len(F) == chi


def test_adjacency_is_symmetric_and_matches_one_rings():
    _, F = uv_sphere(6, 8)
    A = adjacency(F)
    assert (A != A.T).nnz == 0
    assert A.diagonal().sum() == 0
    rings = one_rings(F)
    assert [len(r) for r in rings] == list(np.diff(A.indptr))
    assert len(rings[0]) == 8                      # the pole is connected to its whole ring


def test_boundary_counts():
    nx, ny, n_theta = 7, 5, 10
    assert len(boundary_vertices(grid(nx, ny)[1])) == 2 * (nx + ny) - 4
    assert len(boundary_vertices(uv_sphere(8, 12)[1])) == 0
    assert len(boundary_vertices(tube(n_theta, 6)[1])) == 2 * n_theta


def test_regular_grid_valence_and_ring_sizes():
    # On the hexagonal lattice of grid(), an interior vertex has valence 6 and
    # its k-ring holds 1 + 3k(k+1) vertices: 7, 19, 37 for k = 1, 2, 3.
    nx = 11
    _, F = grid(nx, nx)
    center = nx // 2 + nx * (nx // 2)
    assert len(one_rings(F)[center]) == 6
    for k in (1, 2, 3):
        assert len(k_ring(F, [center], k)) == 1 + 3 * k * (k + 1)


def test_k_ring_zero_is_the_seeds():
    _, F = grid(5, 5)
    assert k_ring(F, [3, 12], 0).tolist() == [3, 12]


def test_ring_distance_from_grid_boundary():
    # Each edge changes i and j by at most 1, so the distance from (i, j) to
    # the boundary is min(i, j, nx-1-i, ny-1-j): rings are nested rectangles.
    nx, ny = 9, 7
    _, F = grid(nx, ny)
    dist = ring_distance(F, boundary_vertices(F))
    i, j = np.arange(nx * ny) % nx, np.arange(nx * ny) // nx
    assert np.array_equal(dist, np.minimum.reduce([i, j, nx - 1 - i, ny - 1 - j]))


def test_ring_distance_respects_max_k():
    _, F = grid(9, 9)
    full = ring_distance(F, boundary_vertices(F))
    capped = ring_distance(F, boundary_vertices(F), max_k=2)
    assert np.array_equal(capped[full <= 2], full[full <= 2])    # same distances up to 2
    assert np.all(capped[full > 2] == -1)                         # farther ones are cut off
