"""Discrete Laplace-Beltrami operators in the form L = D M (Sec. 3.3.4, App. A.1).

Every discrete Laplacian here has the form

    Δf(vi) = wi * Σ_{vj in N1(vi)} wij (fj - fi)

and is stored as the matrix product ``L = D @ M`` (App. A.1):

* ``D = diag(wi)`` holds the per-vertex normalization;
* ``M`` is symmetric, with ``mij = wij`` for every edge (i, j) and
  ``mii = -Σj wij``, so each row of ``M`` sums to zero.

Keeping ``D`` and ``M`` separate matters later: ``M`` is symmetric while ``L``
is not, and the solvers of Ch. 4 and App. A work with the symmetric part.
"""

import numpy as np
import scipy.sparse as sp

from .mesh import adjacency


def uniform_laplacian(V, F):
    """Uniform (graph) Laplacian, Eq. (3.10): ``Δf(vi) = 1/|N1(vi)| Σj (fj - fi)``.

    In the ``L = D M`` form of App. A.1 the weights are ``wij = 1`` (Sec. 4.1.2)
    and ``wi = 1 / deg(vi)``. Applied to the positions, ``(L @ V)[i]`` is the
    vector from ``vi`` to the centroid of its one-ring neighbors.

    The operator depends only on the connectivity ``F``; ``V`` is used just for
    the number of vertices. That is exactly its flaw: on a planar but irregular
    mesh, ``L @ V`` is not zero even though the mean curvature is (Eq. 3.7).

    Returns ``L, D, M`` as ``scipy.sparse`` CSR matrices.
    """
    n = len(V)
    W = adjacency(F, n)                                  # wij = 1 on every edge, Eq. (3.10)
    deg = np.asarray(W.sum(axis=1)).ravel()              # |N1(vi)|
    M = (W - sp.diags(deg)).tocsr()                      # mii = -Σj wij, App. A.1
    D = sp.diags(1.0 / deg).tocsr()                      # wi = 1 / |N1(vi)|
    return (D @ M).tocsr(), D, M
