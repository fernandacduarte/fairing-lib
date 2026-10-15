"""Fairing with a separate reference metric and prescribed output positions."""

from pathlib import Path
import runpy

import numpy as np
import pytest

from fairing.fairing import solve_fair, solve_fair_with_reference
from fairing.laplacian import LAPLACIANS
from fairing.mesh import irregular_grid, pipe_elbow, ring_distance, boundary_vertices


def reference_problem():
    G, F = irregular_grid(11, 11)
    G[:, 2] = 0.1 * np.sin(6 * G[:, 0]) * np.cos(4 * G[:, 1])
    free = np.linalg.norm(G[:, :2], axis=1) < 0.3
    V = G.copy()
    V[:, 2] += 0.3 * V[:, 0] + 0.1
    return V, F, free, G


@pytest.mark.parametrize("laplacian", ["uniform", "cotan"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_reference_solution_uses_reference_operator_and_target_constraints(k, laplacian):
    V, F, free, G = reference_problem()
    originals = [a.copy() for a in (V, F, free, G)]
    W = solve_fair_with_reference(V, F, free, k, laplacian, reference_V=G)
    L = LAPLACIANS[laplacian](G, F)[0]
    residual, initial = W.copy(), V.copy()
    for _ in range(k):
        residual, initial = L @ residual, L @ initial
    assert np.abs(residual[free]).max() < 1e-9 * np.abs(initial[free]).max()
    assert np.array_equal(W[~free], V[~free])
    assert not np.allclose(W[~free], G[~free])
    assert W is not V
    for actual, original in zip((V, F, free, G), originals):
        assert np.array_equal(actual, original)


@pytest.mark.parametrize("laplacian", ["uniform", "cotan"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_same_reference_reproduces_original_solver(k, laplacian):
    V, F, free, _ = reference_problem()
    assert np.allclose(solve_fair_with_reference(V, F, free, k, laplacian, reference_V=V),
                       solve_fair(V, F, free, k, laplacian), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_free_target_placeholders_do_not_affect_cotangent_solution(k):
    V, F, free, G = reference_problem()
    damaged = V.copy()
    damaged[free] = np.random.default_rng(0).normal(0, 100, (free.sum(), 3))
    W = solve_fair_with_reference(V, F, free, k, reference_V=G)
    other = solve_fair_with_reference(damaged, F, free, k, reference_V=G)
    assert np.array_equal(W, other)


@pytest.mark.parametrize("laplacian", ["uniform", "cotan"])
def test_changing_reference_affects_only_geometry_dependent_weights(laplacian):
    V, F, free, G = reference_problem()
    other_G = G.copy()
    other_G[:, 2] += 0.4 * np.exp(-20 * np.sum(G[:, :2] ** 2, axis=1))
    W = solve_fair_with_reference(V, F, free, 2, laplacian, reference_V=G)
    other = solve_fair_with_reference(V, F, free, 2, laplacian, reference_V=other_G)
    if laplacian == "uniform":
        assert np.array_equal(W, other)
    else:
        assert np.abs(W[free] - other[free]).max() > 1e-3


def test_empty_and_unconstrained_regions():
    V, F, free, G = reference_problem()
    W = solve_fair_with_reference(V, F, np.zeros_like(free), 2, reference_V=G)
    assert np.array_equal(W, V) and W is not V
    with pytest.raises(ValueError, match="at least one vertex"):
        solve_fair_with_reference(V, F, np.ones_like(free), 2, reference_V=G)


def test_reference_correspondence_and_mask_shapes_are_checked():
    V, F, free, G = reference_problem()
    for invalid in (G[:-1], G[:, :2], G.ravel()):
        with pytest.raises(ValueError, match="same shape"):
            solve_fair_with_reference(V, F, free, 2, reference_V=invalid)
    with pytest.raises(ValueError, match="free_mask"):
        solve_fair_with_reference(V, F, free[:, None], 2, reference_V=G)
    G[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solve_fair_with_reference(V, F, free, 2, reference_V=G)


def test_straight_reference_elbow_keeps_original_connectivity_and_fixed_pipes():
    script = Path(__file__).resolve().parents[1] / "examples" / "04_reference_elbow.py"
    example = runpy.run_path(str(script))
    options = dict(n_theta=24, radius=1.0, bend_radius=1.25, pipe_length=1.2, spacing=0.1)
    V, F, free, G = example["elbow_problem"](**options)
    original, original_F, original_free = pipe_elbow(**options)
    assert np.array_equal(F, original_F)
    assert np.array_equal(free, original_free)
    assert np.allclose(V[~free], original[~free], atol=1e-14)
    assert np.allclose(np.linalg.norm(G[:, :2], axis=1), 1.0)
    assert np.array_equal(V[free], G[free])
    distance = ring_distance(F, np.flatnonzero(free))
    assert distance[boundary_vertices(F)].min() > 3
    for k in (1, 2, 3):
        W = solve_fair_with_reference(V, F, free, k, reference_V=G)
        assert np.isfinite(W).all()
        assert np.array_equal(W[~free], V[~free])
