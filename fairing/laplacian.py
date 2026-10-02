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


def face_areas(V, F):
    """Area of every triangle: half the length of the cross product of two edges."""
    V, F = np.asarray(V, dtype=float), np.asarray(F)
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def vertex_areas(V, F):
    """Barycentric vertex areas (Fig. 3.7): ``Ai = 1/3 Σ`` areas of the triangles at ``vi``.

    The barycentric cells tile the mesh, so ``Σ Ai`` equals the total area.
    """
    third = np.repeat(face_areas(V, F) / 3, 3)              # one third to each corner
    return np.bincount(np.asarray(F).ravel(), weights=third, minlength=len(V))


def cotan_laplacian(V, F):
    """Cotangent Laplacian, Eq. (3.11): ``Δf(vi) = 1/(2Ai) Σj (cot αij + cot βij)(fj - fi)``.

    ``αij`` and ``βij`` are the two angles opposite the edge (i, j) (Fig. 3.10),
    and ``Ai`` is the barycentric area. In the ``L = D M`` form of App. A.1:
    ``wij = cot αij + cot βij`` and ``wi = 1 / (2 Ai)``.

    Each triangle (i, j, k) contributes ``cot`` of its angle at k to the edge
    (i, j), and likewise for its other two edges. An interior edge collects one
    cotangent from each of its two triangles; a boundary edge gets only one.

    Returns ``L, D, M`` as ``scipy.sparse`` CSR matrices.
    """
    V, F = np.asarray(V, dtype=float), np.asarray(F)
    n = len(V)
    rows, cols, vals = [], [], []
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        i, j, k = F[:, a], F[:, b], F[:, c]
        u, v = V[i] - V[k], V[j] - V[k]                     # the two edges at corner k
        cot = np.einsum("ij,ij->i", u, v) / np.linalg.norm(np.cross(u, v), axis=1)
        rows += [i, j]
        cols += [j, i]
        vals += [cot, cot]
    # duplicate (i, j) entries are summed: cot αij + cot βij, Eq. (3.11)
    W = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()
    M = (W - sp.diags(np.asarray(W.sum(axis=1)).ravel())).tocsr()   # mii = -Σj wij, App. A.1
    D = sp.diags(1.0 / (2.0 * vertex_areas(V, F))).tocsr()          # wi = 1 / (2 Ai)
    return (D @ M).tocsr(), D, M


LAPLACIANS = {"uniform": uniform_laplacian, "cotan": cotan_laplacian}


def mean_curvature(V, F, laplacian="cotan"):
    """Absolute mean curvature per vertex, Eq. (3.13): ``H = 1/2 ‖Δx‖``.

    It follows from ``Δx = -2H n`` (Eq. 3.7). Only the cotangent Laplacian
    approximates ``Δ`` (units 1/length); with ``laplacian="uniform"`` the
    result scales with the edge length and is not a curvature.
    """
    L = LAPLACIANS[laplacian](V, F)[0]
    return 0.5 * np.linalg.norm(L @ np.asarray(V, dtype=float), axis=1)
