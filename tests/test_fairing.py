"""Tests for constrained fairing (issues #9-#13)."""

import numpy as np
import pytest

from fairing.fairing import fairing_matrix, solve_fair
from fairing.laplacian import LAPLACIANS
from fairing.mesh import add_noise, irregular_grid, k_ring


def noisy_patch(n=12, seed=0):
    """Irregular grid with noisy heights; a free blob of 3 rings around the center."""
    V, F = irregular_grid(n, n, seed=seed)
    V = add_noise(V, 0.03, seed=seed) * [0, 0, 1] + V * [1, 1, 0]     # noise in z only
    center = n // 2 + n * (n // 2)
    free = np.zeros(len(V), dtype=bool)
    free[k_ring(F, [center], 3)] = True
    return V, F, free


# ---------------------------------------------------------------------------
# Solver, Eq. (A.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("laplacian", ["uniform", "cotan"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_A_ff_is_symmetric_positive_definite(k, laplacian):
    V, F, free = noisy_patch()
    A = fairing_matrix(V, F, k, laplacian)
    A_ff = A[free][:, free].toarray()
    assert np.allclose(A_ff, A_ff.T, rtol=1e-12, atol=1e-12 * np.abs(A_ff).max())
    assert np.linalg.eigvalsh((A_ff + A_ff.T) / 2).min() > 0       # positive definite
    np.linalg.cholesky(A_ff)                                         # and Cholesky exists


@pytest.mark.parametrize("laplacian", ["uniform", "cotan"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_solution_satisfies_Lk_x_zero_at_free_vertices(k, laplacian):
    V, F, free = noisy_patch()
    W = solve_fair(V, F, free, k, laplacian)
    L = LAPLACIANS[laplacian](V, F)[0]                                # built from the input, like the solver
    Lk = np.linalg.matrix_power(L.toarray(), k)
    power = lambda X: Lk @ X
    residual = np.abs(power(W)[free]).max()
    assert residual < 1e-9 * np.abs(power(V)).max()                  # relative: L^k entries scale like 1/h^2k


def test_constrained_vertices_do_not_move():
    V, F, free = noisy_patch()
    W = solve_fair(V, F, free, 2)
    assert np.array_equal(W[~free], V[~free])
    assert not np.allclose(W[free], V[free])


def test_nothing_free_returns_V_unchanged():
    V, F, _ = noisy_patch()
    W = solve_fair(V, F, np.zeros(len(V), dtype=bool), 2)
    assert np.array_equal(W, V) and W is not V


def test_everything_free_is_rejected():
    # With no constraint, constants (and more) solve L^k x = 0: no unique answer.
    V, F, _ = noisy_patch()
    with pytest.raises(ValueError):
        solve_fair(V, F, np.ones(len(V), dtype=bool), 1)


def test_fairing_matrix_k1_is_minus_M():
    V, F, _ = noisy_patch()
    _, _, M = LAPLACIANS["cotan"](V, F)
    assert np.allclose(fairing_matrix(V, F, 1).toarray(), -M.toarray())


# ---------------------------------------------------------------------------
# Membrane surface, k = 1 (Eq. 4.7-4.8)
# ---------------------------------------------------------------------------

def disk_problem(n, heights, interior_noise=0.02, seed=0):
    """Irregular planar grid on [-0.5, 0.5]^2 whose disk of radius 0.3 is free.

    The constrained vertices get z = heights(x, y); the free ones start at
    heights(x, y) plus noise. Returns the input V, the flat reference V_ref
    (same x, y, z = 0), F, and the free mask.
    """
    V_ref, F = irregular_grid(n, n, jitter=0.15, seed=seed)
    free = np.linalg.norm(V_ref[:, :2], axis=1) < 0.3
    V = V_ref.copy()
    V[:, 2] = heights(V[:, 0], V[:, 1])
    V[free, 2] += np.random.default_rng(seed).normal(0, interior_noise, free.sum())
    return V, V_ref, F, free


flat = lambda x, y: 0 * x
saddle = lambda x, y: x**2 - y**2                    # harmonic: 2 - 2 = 0


@pytest.mark.parametrize("laplacian", ["uniform", "cotan"])
@pytest.mark.parametrize("use_ref", [False, True])
def test_membrane_with_planar_boundary_is_exactly_planar(laplacian, use_ref):
    # z = 0 on the boundary => z = 0 is the unique discrete harmonic function.
    V, V_ref, F, free = disk_problem(21, flat, interior_noise=0.05)
    W = solve_fair(V, F, free, 1, laplacian, V_ref=V_ref if use_ref else None)
    assert np.abs(W[:, 2]).max() < 1e-12


def test_cotan_membrane_keeps_planar_vertices_in_place_only_with_a_clean_reference():
    # Linear precision (#6) holds for weights from the flat grid, not from the noisy input.
    V, V_ref, F, free = disk_problem(21, flat, interior_noise=0.05)
    moved = lambda W: np.abs(W[free, :2] - V[free, :2]).max()
    assert moved(solve_fair(V, F, free, 1, "cotan", V_ref=V_ref)) < 1e-12
    assert moved(solve_fair(V, F, free, 1, "cotan")) > 1e-3


def saddle_error(n, laplacian, use_ref):
    V, V_ref, F, free = disk_problem(n, saddle)
    W = solve_fair(V, F, free, 1, laplacian, V_ref=V_ref if use_ref else None)
    x, y, z = W[free].T
    return np.abs(z - saddle(x, y)).max()             # compare at the final (x, y)


@pytest.mark.parametrize("laplacian", ["uniform", "cotan"])
def test_harmonic_boundary_error_shrinks_with_resolution(laplacian):
    errors = [saddle_error(n, laplacian, use_ref=True) for n in (11, 21, 41)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 5e-4


def test_cotan_weights_from_the_noisy_input_stop_the_convergence():
    # Documented pitfall: with L built from the noisy input, the error stalls.
    assert saddle_error(41, "cotan", use_ref=False) > 10 * saddle_error(41, "cotan", use_ref=True)
