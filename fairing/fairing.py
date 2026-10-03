"""Constrained fairing: solve L^k x = 0 as an SPD system (Sec. 4.3, Eq. A.2).

The free vertices move so that the k-th power of the Laplacian vanishes there,
while the constrained vertices stay where they are:

* k = 1: membrane surface (minimal area, Eq. 4.7-4.8),
* k = 2: thin-plate surface (minimal curvature),
* k = 3: minimum variation surface (Eq. 4.9).

The linear system is solved with a sparse Cholesky factorization (CHOLMOD),
as App. A recommends for symmetric positive definite systems.
"""

import numpy as np
from sksparse.cholmod import cho_factor

from .laplacian import LAPLACIANS


def fairing_matrix(V, F, k, laplacian="cotan"):
    """Symmetric fairing matrix ``A = (-1)^k M (D M)^(k-1)`` (App. A.1, Eq. A.2).

    ``L^k = (D M)^k = D M (D M)^(k-1)``. The left-most ``D`` is diagonal with
    positive entries, so ``(L^k x)_i = 0`` exactly when ``(M (D M)^(k-1) x)_i = 0``:
    dropping it leaves a symmetric matrix (``M``, ``M D M``, ``M D M D M``, ...).
    The sign ``(-1)^k`` makes it positive (semi-)definite, since ``-M`` is.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    _, D, M = LAPLACIANS[laplacian](V, F)
    A = M
    for _ in range(k - 1):
        A = A @ D @ M                                      # M (D M)^(k-1)
    return ((-1) ** k * A).tocsr()


def solve_fair(V, F, free_mask, k, laplacian="cotan"):
    """Move the free vertices so that ``L^k x = 0`` there; keep the others fixed.

    Splitting the unknowns into free (f) and constrained (c) vertices, the rows
    of the free vertices read ``A_ff x_f + A_fc x_c = 0``. The constrained
    positions are known, so they move to the right-hand side (App. A.1):

        A_ff x_f = -A_fc x_c.

    ``A_ff`` is symmetric positive definite as long as some vertex is
    constrained; it is factorized once with Cholesky and solved for x, y and z
    together.

    ``L`` is built once from the input geometry ``V`` and then frozen, which
    is what makes the system linear. The cotangent weights therefore depend on
    where the free vertices start: a smooth start gives a sensible result, a
    badly damaged one can corrupt it (see docs/phases/phase-4.md, #10). The
    uniform weights do not depend on positions at all.

    For ``C^(k-1)`` continuity at the border of the free region, the ``k``
    rings of vertices around it should be constrained (Sec. 4.3).

    Returns a new ``(n, 3)`` array; constrained vertices keep their positions.
    """
    V = np.asarray(V, dtype=float)
    free = np.asarray(free_mask, dtype=bool)
    if not free.any():
        return V.copy()
    if free.all():
        raise ValueError("at least one vertex must be constrained, or L^k x = 0 has no unique solution")

    A = fairing_matrix(V, F, k, laplacian)
    A_ff = A[free][:, free]
    A_fc = A[free][:, ~free]
    rhs = -(A_fc @ V[~free])                               # constrained values move to the right
    V_new = V.copy()
    V_new[free] = cho_factor(A_ff.tocsc()).solve(rhs)      # Cholesky: A_ff = R^T R (permuted)
    return V_new
