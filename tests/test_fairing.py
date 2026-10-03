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
