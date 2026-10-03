"""Explicit and implicit diffusion flow (Sec. 4.2).

The diffusion equation ``∂f/∂t = λ Δf`` (Eq. 4.5), discretized in space with a
Laplace matrix ``L`` (Eq. 4.6), becomes ``∂x/∂t = λ L x`` for the vertex
positions. Discretizing time with steps of size ``h`` gives the update rules.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .laplacian import LAPLACIANS


def explicit_smoothing(V, F, h, lam=1.0, n_iter=1, laplacian="uniform"):
    """Explicit Euler Laplacian smoothing, Sec. 4.2: ``x <- x + h λ L x``.

    ``L`` is rebuilt from the current positions at every iteration, so each
    step uses the Laplace-Beltrami operator of the current surface (for the
    cotangent Laplacian this is the discrete mean curvature flow). With the
    uniform Laplacian ``L`` depends only on ``F``, so rebuilding changes nothing.

    The step is stable only for ``h <= h_max = explicit_step_limit(...)``; see there.
    """
    V = np.asarray(V, dtype=float).copy()
    build = LAPLACIANS[laplacian]
    for _ in range(n_iter):
        L = build(V, F)[0]
        V += h * lam * (L @ V)
    return V


def explicit_step_limit(V, F, lam=1.0, laplacian="uniform"):
    """Largest stable step ``h_max = 2 / (λ |μ_min|)`` for explicit smoothing.

    One explicit step multiplies each eigencomponent of ``x`` (eigenvalue ``μ``
    of ``L``, all ``μ <= 0``) by ``1 + h λ μ``. It stays bounded only if
    ``|1 + h λ μ| <= 1`` for every ``μ``, i.e. ``h λ <= 2 / |μ_min|``.

    ``L = D M`` has real eigenvalues because it is similar to the symmetric
    matrix ``D^½ M D^½``, which is what we pass to the eigensolver. For the
    uniform Laplacian ``μ_min >= -2``, so ``h λ <= 1`` is always stable; for
    the cotangent Laplacian ``|μ_min|`` grows like ``1 / (edge length)²`` and
    is dominated by the worst triangles.
    """
    _, D, M = LAPLACIANS[laplacian](V, F)
    s = sp.diags(np.sqrt(D.diagonal()))
    v0 = np.ones(M.shape[0])                     # fixed start vector: ARPACK is random otherwise
    mu_min = spla.eigsh(s @ M @ s, k=1, which="LM", v0=v0, return_eigenvectors=False)[0]
    return 2.0 / (lam * abs(mu_min))


def implicit_smoothing(V, F, h, lam=1.0, n_iter=1, laplacian="uniform"):
    """Implicit Euler Laplacian smoothing, Sec. 4.2: ``(I - h λ L) x' = x``.

    The Laplacian is evaluated at the new positions ``x'``, so each step solves
    a sparse linear system. In the eigenvector picture every component is
    multiplied by ``1 / (1 - h λ μ)``, which lies in (0, 1] for all ``μ <= 0``:
    the scheme is stable for any ``h`` (no ``h_max``).

    ``I - h λ L = I - h λ D M`` is not symmetric. Multiplying by ``D^-1``
    (App. A.1, as in Eq. A.2) gives the symmetric positive definite system

        (D^-1 - h λ M) x' = D^-1 x,

    which is factorized once per step and reused for x, y and z. As in
    :func:`explicit_smoothing`, ``L`` is rebuilt from the current positions at
    every step.
    """
    V = np.asarray(V, dtype=float).copy()
    build = LAPLACIANS[laplacian]
    for _ in range(n_iter):
        _, D, M = build(V, F)
        D_inv = sp.diags(1.0 / D.diagonal())
        solve = spla.factorized((D_inv - h * lam * M).tocsc())   # (D^-1 - h λ M), App. A.1
        rhs = D_inv @ V                                           # D^-1 x
        V = np.column_stack([solve(rhs[:, c]) for c in range(3)])
    return V


def roughness(V, F):
    """Mean angle (radians) between the normals of triangles sharing an edge.

    A local measure of noise that ignores the overall shape: it is small on a
    finely sampled smooth surface and grows with high-frequency noise.
    """
    V, F = np.asarray(V, dtype=float), np.asarray(F)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    # pair up the two triangles of every interior edge
    key = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    face = np.tile(np.arange(len(F)), 3)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, face = key[order], face[order]
    pair = np.all(key[1:] == key[:-1], axis=1)
    a, b = face[:-1][pair], face[1:][pair]
    return np.arccos(np.clip(np.einsum("ij,ij->i", n[a], n[b]), -1.0, 1.0)).mean()
