"""Tests for constrained fairing (issues #9-#13)."""

import numpy as np
import pytest

from fairing.fairing import fairing_matrix, solve_fair
from fairing.laplacian import LAPLACIANS, mean_curvature
from fairing.mesh import add_noise, boundary_vertices, irregular_grid, k_ring, ring_distance, tube


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

def disk_problem(n, heights, interior_noise, seed=0):
    """Irregular planar grid on [-0.5, 0.5]^2 whose disk of radius 0.3 is free.

    Every vertex gets z = heights(x, y); the free ones also get noise.
    """
    V, F = irregular_grid(n, n, jitter=0.15, seed=seed)
    r = np.linalg.norm(V[:, :2], axis=1)
    free = r < 0.3
    V[:, 2] = heights(V[:, 0], V[:, 1])
    V[free, 2] += np.random.default_rng(seed).normal(0, interior_noise, free.sum())
    return V, F, free, r


flat = lambda x, y: 0 * x
saddle = lambda x, y: x**2 - y**2                    # harmonic in the plane: 2 - 2 = 0


@pytest.mark.parametrize("laplacian", ["uniform", "cotan"])
def test_case_A_planar_boundary_gives_an_exactly_planar_membrane(laplacian):
    # z = 0 on the boundary => z = 0 is the unique solution, whatever the weights.
    V, F, free, _ = disk_problem(21, flat, interior_noise=0.05)
    W = solve_fair(V, F, free, 1, laplacian)
    assert np.abs(W[:, 2]).max() < 1e-12


def test_case_A_cotan_weights_from_a_noisy_start_let_vertices_slide():
    # The weights are frozen from the noisy input, which is not planar, so the cotangent
    # Laplacian loses its linear precision there (#6) and x, y move. Uniform: no such effect.
    V, F, free, _ = disk_problem(21, flat, interior_noise=0.05)
    slide = lambda W: np.abs(W[free, :2] - V[free, :2]).max()
    assert slide(solve_fair(V, F, free, 1, "cotan")) > 1e-3


def saddle_case(n, laplacian, interior_noise):
    V, F, free, r = disk_problem(n, saddle, interior_noise)
    W = solve_fair(V, F, free, 1, laplacian)
    x, y, z = W[free].T
    gap = np.abs(z - saddle(x, y)).max()                       # compare at the final (x, y)
    inner = r < 0.25                                           # away from the rim of the disk
    return gap, mean_curvature(W, F)[inner].mean()


def test_case_B_uniform_converges_to_the_harmonic_graph():
    # Uniform weights ignore geometry; on this nearly regular grid they act like the flat
    # (x, y) parameter domain of Eq. 4.8, whose exact answer is z = x^2 - y^2.
    gaps = [saddle_case(n, "uniform", interior_noise=0.02)[0] for n in (11, 21, 41)]
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < 5e-4


def test_case_B_cotan_from_a_smooth_start_is_nearly_a_minimal_surface():
    # Cotangent weights from the surface itself approximate the true membrane of Eq. 4.7,
    # a minimal surface (H = 0). x^2 - y^2 is not one (|H| ~ 0.07), so a fixed gap remains.
    V, F, free, r = disk_problem(41, saddle, interior_noise=0.0)
    H_saddle = mean_curvature(V, F)[r < 0.25].mean()
    gaps, H = zip(*[saddle_case(n, "cotan", interior_noise=0.0) for n in (21, 41, 81)])
    assert all(h < 0.2 * H_saddle for h in H)                  # much closer to H = 0
    assert all(5e-4 < g < 3e-3 for g in gaps)                  # a gap that does not vanish
    assert max(gaps) / min(gaps) < 1.5                         # ... and does not depend on h


def test_case_B_cotan_from_a_badly_damaged_start_is_corrupted():
    # Documented pitfall: noise larger than the edges (sigma = 0.02 > h = 0.0125) gives
    # cotangent weights from a crumpled surface, and the frozen weights crumple the result.
    _, H_noisy = saddle_case(81, "cotan", interior_noise=0.02)
    _, H_clean = saddle_case(81, "cotan", interior_noise=0.0)
    assert H_noisy > 100 * H_clean


# ---------------------------------------------------------------------------
# Thin plate and minimum variation, k = 2, 3 (Sec. 4.3, Fig. 4.8)
# ---------------------------------------------------------------------------

def tube_blend(n_theta, n_z, shift=1.0):
    """Tube of radius 1 and height 4. Bottom band (z <= 1) fixed, top band (z >= 3) fixed and
    shifted sideways by `shift`, middle free. The middle starts as a straight sheared tube
    (a clean start for the frozen cotangent weights, see #10). Both bands are >= 3 rings thick."""
    V, F = tube(n_theta, n_z, radius=1.0, height=4.0)
    z = V[:, 2]
    free = (z > 1 + 1e-9) & (z < 3 - 1e-9)
    V[:, 0] += shift * np.clip((z - 1) / 2, 0, 1)
    return V, F, free


def joint_turning_angle(W, n_theta, n_z, free):
    """Turning angle (degrees) of the theta = 0 meridian at the last fixed ring of the bottom band."""
    meridian = np.arange(n_z) * n_theta
    j = np.flatnonzero(~free[meridian] & (np.arange(n_z) < n_z // 2)).max()
    a, b = W[meridian[j]] - W[meridian[j - 1]], W[meridian[j + 1]] - W[meridian[j]]
    return np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))


def test_tube_blend_has_enough_fixed_rings_for_k3():
    # The 3 rings around the free region must be fixed vertices that end before the
    # tube's open ends; otherwise L^3 would reach the one-sided boundary Laplacian (#9).
    V, F, free = tube_blend(32, 41)
    dist = ring_distance(F, np.flatnonzero(free))
    assert dist[boundary_vertices(F)].min() > 3


def test_joint_is_sharper_for_k1_than_for_k2_and_k3():
    V, F, free = tube_blend(32, 41)
    angle = {k: joint_turning_angle(solve_fair(V, F, free, k), 32, 41, free) for k in (1, 2, 3)}
    assert angle[1] > angle[2] > angle[3]


@pytest.mark.parametrize("k, low, high", [(1, 0.9, 2.0), (2, 0.4, 0.7), (3, 0.0, 0.4)])
def test_refinement_reveals_C_k_minus_1_continuity(k, low, high):
    # Halving the spacing: a kink (C0) keeps its angle; a C1 joint's angle halves (~ h);
    # a C2 joint's angle drops faster (~ h^2), since the curvature is continuous too.
    coarse, fine = (32, 41), (64, 81)
    angles = []
    for n_theta, n_z in (coarse, fine):
        V, F, free = tube_blend(n_theta, n_z)
        angles.append(joint_turning_angle(solve_fair(V, F, free, k), n_theta, n_z, free))
    assert low < angles[1] / angles[0] < high
