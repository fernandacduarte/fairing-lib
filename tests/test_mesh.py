"""Tests for the mesh generators and OBJ I/O (issue #2)."""

import numpy as np
import pytest

from fairing.mesh import grid, irregular_grid, load_obj, save_obj, tube, uv_sphere


def face_normals(V, F):
    """Unnormalized normals; their length is twice the triangle area."""
    return np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])


GENERATORS = {
    "grid": lambda: grid(6, 5),
    "irregular_grid": lambda: irregular_grid(6, 5, jitter=0.16, seed=1),
    "uv_sphere": lambda: uv_sphere(8, 12, radius=2.0),
    "tube": lambda: tube(10, 5),
}


@pytest.mark.parametrize("name", GENERATORS)
def test_shapes_and_dtypes(name):
    V, F = GENERATORS[name]()
    assert V.ndim == 2 and V.shape[1] == 3 and V.dtype == float
    assert F.ndim == 2 and F.shape[1] == 3 and np.issubdtype(F.dtype, np.integer)
    assert F.min() == 0 and F.max() == len(V) - 1        # every vertex is used


def test_triangle_counts():
    assert len(grid(6, 5)[1]) == 2 * 5 * 4
    assert len(tube(10, 5)[1]) == 2 * 10 * 4
    V, F = uv_sphere(8, 12)
    assert len(V) == 2 + 12 * 7
    assert len(F) == 2 * 12 * 7


def test_orientation_and_positive_area():
    # planar grids: every normal points to +z (no flipped or degenerate triangle)
    for V, F in (grid(6, 5), irregular_grid(6, 5, jitter=0.16, seed=1)):
        assert np.all(face_normals(V, F)[:, 2] > 0)

    # sphere: every normal points away from the center
    V, F = uv_sphere(8, 12)
    assert np.all(np.einsum("ij,ij->i", face_normals(V, F), V[F].mean(axis=1)) > 0)

    # tube: every normal points away from the z axis
    V, F = tube(10, 5)
    radial = V[F].mean(axis=1) * [1, 1, 0]
    assert np.all(np.einsum("ij,ij->i", face_normals(V, F), radial) > 0)


def test_sphere_vertices_lie_on_sphere():
    V, _ = uv_sphere(8, 12, radius=2.0)
    assert np.allclose(np.linalg.norm(V, axis=1), 2.0)


def test_irregular_grid_moves_only_interior_in_plane():
    V0, F0 = grid(6, 5)
    V, F = irregular_grid(6, 5, jitter=0.1, seed=3)
    assert np.array_equal(F, F0)
    assert np.all(V[:, 2] == 0)
    moved = np.any(V != V0, axis=1)
    i, j = np.arange(30) % 6, np.arange(30) // 6
    interior = (i > 0) & (i < 5) & (j > 0) & (j < 4)
    assert np.array_equal(moved, interior)


def test_irregular_grid_rejects_large_jitter():
    with pytest.raises(ValueError):
        irregular_grid(6, 5, jitter=0.2)


@pytest.mark.parametrize("name", GENERATORS)
def test_obj_round_trip(name, tmp_path):
    V, F = GENERATORS[name]()
    path = tmp_path / "mesh.obj"
    save_obj(path, V, F)
    V2, F2 = load_obj(path)
    assert np.array_equal(V2, V)          # 17 significant digits: bit-exact
    assert np.array_equal(F2, F)


def test_load_obj_ignores_texture_and_normal_indices(tmp_path):
    path = tmp_path / "tri.obj"
    path.write_text("# comment\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1/1/1 2/1/1 3/1/1\n")
    V, F = load_obj(path)
    assert V.shape == (3, 3)
    assert F.tolist() == [[0, 1, 2]]


def test_load_obj_rejects_polygons(tmp_path):
    path = tmp_path / "quad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    with pytest.raises(ValueError):
        load_obj(path)


def test_pipe_elbow():
    from fairing.mesh import pipe_elbow
    n_theta = 24
    V, F, bend = pipe_elbow(n_theta=n_theta, bend_radius=1.5, pipe_length=1.0, spacing=0.25)
    n_rings = len(V) // n_theta
    assert len(F) == 2 * n_theta * (n_rings - 1)
    rings = V.reshape(n_rings, n_theta, 3)
    centers = rings.mean(axis=1)
    assert np.allclose(np.linalg.norm(rings - centers[:, None], axis=2), 1.0)     # every ring has radius 1
    # every triangle faces away from the pipe's axis (the center of its first vertex's ring)
    away = V[F].mean(axis=1) - centers[F[:, 0] // n_theta]
    assert np.all(np.einsum("ij,ij->i", face_normals(V, F), away) > 0)
    # the bend is a contiguous block of rings strictly between the two pipes
    ring_in_bend = bend.reshape(n_rings, n_theta)[:, 0]
    assert ring_in_bend.any() and not ring_in_bend[0] and not ring_in_bend[-1]
    assert np.all(np.diff(np.flatnonzero(ring_in_bend)) == 1)
