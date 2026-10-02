"""Mesh generators, OBJ I/O, and topology helpers.

A mesh is a pair of plain NumPy arrays: ``V`` with shape ``(n, 3)`` (float
vertex positions) and ``F`` with shape ``(m, 3)`` (int vertex indices of the
triangles). All generators orient triangles counter-clockwise when seen from
the "outside" (``+z`` for planar grids, away from the axis or center for the
tube and the sphere).
"""

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _quad_faces(nu, nv, wrap_u=False):
    """Triangulate a lattice of ``nu x nv`` vertices, indexed ``i + nu*j``.

    Every quad ``(a, b, d, c)`` is split along the same diagonal ``a-d``:

        c ---- d        a = (i,   j)      b = (i+1, j)
        |    / |        c = (i,   j+1)    d = (i+1, j+1)
        |  /   |
        a ---- b        triangles (a, b, d) and (a, d, c)

    Using one diagonal everywhere gives every interior vertex valence 6. If
    ``wrap_u`` is true, column ``nu-1`` connects back to column 0 (a tube).
    The triangle normals point along ``u x v``.
    """
    i, j = np.meshgrid(np.arange(nu if wrap_u else nu - 1), np.arange(nv - 1))
    i, j = i.ravel(), j.ravel()
    i1 = (i + 1) % nu
    a, b = i + nu * j, i1 + nu * j
    c, d = i + nu * (j + 1), i1 + nu * (j + 1)
    return np.concatenate([np.stack([a, b, d], axis=1), np.stack([a, d, c], axis=1)])


def grid(nx, ny, size=1.0):
    """Regular planar grid of ``nx x ny`` vertices in ``z = 0``.

    The grid covers the square ``[-size/2, size/2]^2`` and has
    ``2 (nx-1)(ny-1)`` triangles. Vertex ``(i, j)`` has index ``i + nx*j``.
    """
    x, y = np.meshgrid(np.linspace(-size / 2, size / 2, nx), np.linspace(-size / 2, size / 2, ny))
    V = np.stack([x.ravel(), y.ravel(), np.zeros(nx * ny)], axis=1)
    return V, _quad_faces(nx, ny)


def irregular_grid(nx, ny, size=1.0, jitter=0.15, seed=0):
    """Planar grid like :func:`grid`, with the interior vertices jittered in the plane.

    Each interior vertex moves by a uniform random offset of at most
    ``jitter`` times the grid spacing along x and along y; boundary vertices
    stay put. The mesh stays planar, so its true mean curvature is zero, but
    the vertices are no longer evenly spaced.

    ``jitter < 1/6`` guarantees that no triangle flips. For a triangle with
    legs ``h``, the doubled area is ``h^2`` and each edge vector changes by at
    most ``2*jitter*h`` per coordinate, so the doubled area stays at least
    ``h^2 (1 - 6*jitter) > 0``.
    """
    if not 0 <= jitter < 1 / 6:
        raise ValueError("jitter must be in [0, 1/6) to guarantee that no triangle flips")
    V, F = grid(nx, ny, size)
    h = np.array([size / (nx - 1), size / (ny - 1)])
    i, j = np.arange(nx * ny) % nx, np.arange(nx * ny) // nx
    interior = (i > 0) & (i < nx - 1) & (j > 0) & (j < ny - 1)
    rng = np.random.default_rng(seed)
    V[interior, :2] += rng.uniform(-jitter, jitter, (interior.sum(), 2)) * h
    return V, F


def uv_sphere(n_lat, n_lon, radius=1.0):
    """UV sphere with ``n_lat`` latitude bands and ``n_lon`` longitude segments.

    There is one vertex at each pole and ``n_lat - 1`` rings of ``n_lon``
    vertices in between, so ``n = 2 + n_lon (n_lat - 1)`` vertices and
    ``2 n_lon (n_lat - 1)`` triangles. Vertex 0 is the south pole, the last
    vertex is the north pole, and the rings go from south to north.
    """
    theta = np.pi * np.arange(1, n_lat) / n_lat           # polar angle measured from the south pole
    phi = 2 * np.pi * np.arange(n_lon) / n_lon
    t, p = np.meshgrid(theta, phi, indexing="ij")         # rows = rings
    ring = radius * np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), -np.cos(t)], axis=-1)
    V = np.vstack([[0, 0, -radius], ring.reshape(-1, 3), [0, 0, radius]])

    # u = east, v = north, so u x v points outward
    bands = 1 + _quad_faces(n_lon, n_lat - 1, wrap_u=True)
    i = np.arange(n_lon)
    south, north = 0, len(V) - 1
    last = 1 + n_lon * (n_lat - 2)                         # first vertex of the northernmost ring
    south_cap = np.stack([np.full(n_lon, south), 1 + (i + 1) % n_lon, 1 + i], axis=1)
    north_cap = np.stack([np.full(n_lon, north), last + i, last + (i + 1) % n_lon], axis=1)
    return V, np.vstack([south_cap, bands, north_cap])


def tube(n_theta, n_z, radius=1.0, height=2.0):
    """Open cylinder along ``z`` from ``z = 0`` to ``z = height``.

    It has ``n_z`` rings of ``n_theta`` vertices (index ``i + n_theta*j`` for
    angle ``i`` and ring ``j``) and ``2 n_theta (n_z - 1)`` triangles.
    """
    phi = 2 * np.pi * np.arange(n_theta) / n_theta
    p, z = np.meshgrid(phi, np.linspace(0, height, n_z))
    V = np.stack([radius * np.cos(p).ravel(), radius * np.sin(p).ravel(), z.ravel()], axis=1)
    # u = around the axis, v = up, so u x v points away from the axis
    return V, _quad_faces(n_theta, n_z, wrap_u=True)


# ---------------------------------------------------------------------------
# OBJ I/O
# ---------------------------------------------------------------------------

def save_obj(path, V, F):
    """Write a triangle mesh as a Wavefront OBJ file (1-based indices)."""
    with open(path, "w") as f:
        for x, y, z in V:
            f.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")   # 17 digits: exact float64 round trip
        for a, b, c in np.asarray(F) + 1:
            f.write(f"f {a} {b} {c}\n")


def load_obj(path):
    """Read the vertices and triangles of a Wavefront OBJ file.

    Face entries such as ``7/2/5`` keep only the vertex index. Faces that are
    not triangles raise a ``ValueError``.
    """
    V, F = [], []
    with open(path) as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "v":
                V.append([float(t) for t in tokens[1:4]])
            elif tokens[0] == "f":
                if len(tokens) != 4:
                    raise ValueError(f"only triangles are supported, got: {line.strip()}")
                F.append([int(t.split("/")[0]) - 1 for t in tokens[1:]])
    return np.array(V, dtype=float).reshape(-1, 3), np.array(F, dtype=int).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def _half_edges(F):
    """The three directed edges (a, b), (b, c), (c, a) of every triangle."""
    F = np.asarray(F)
    return np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])


def edges(F):
    """Unique undirected edges as an ``(E, 2)`` array with ``i < j`` in each row."""
    return np.unique(np.sort(_half_edges(F), axis=1), axis=0)


def adjacency(F, n=None):
    """Symmetric 0/1 vertex adjacency matrix (``scipy.sparse``, CSR).

    Entry ``(i, j)`` is 1 when ``vi`` and ``vj`` share an edge. Row ``i``
    therefore lists the one-ring ``N1(vi)`` of Sec. 3.3.1, the same sparsity
    pattern as the Laplacian matrix (App. A.1, "Sparsity").
    """
    E = edges(F)
    n = int(np.max(F)) + 1 if n is None else n
    ones = np.ones(2 * len(E))
    A = sp.coo_matrix((ones, (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n))
    return A.tocsr()


def one_rings(F, n=None):
    """List of one-ring neighbor arrays: ``one_rings(F)[i]`` is ``N1(vi)``, sorted.

    ``len(one_rings(F)[i])`` is the valence ``deg(vi)`` used by the uniform
    Laplacian (Eq. 3.10).
    """
    A = adjacency(F, n)
    return [A.indices[A.indptr[i]:A.indptr[i + 1]] for i in range(A.shape[0])]


def boundary_vertices(F):
    """Sorted indices of the vertices on the mesh boundary.

    An interior edge is shared by two triangles; a boundary edge belongs to
    exactly one. The boundary vertices are the endpoints of boundary edges.
    """
    E, counts = np.unique(np.sort(_half_edges(F), axis=1), axis=0, return_counts=True)
    return np.unique(E[counts == 1])


def ring_distance(F, seed_vertices, max_k=None, n=None):
    """Graph distance (number of edges) from the seed vertices to every vertex.

    Breadth-first search: ring 0 is the seeds, ring ``d`` is every vertex
    whose nearest seed is ``d`` edges away. Vertices farther than ``max_k``
    (or unreachable) get ``-1``.
    """
    A = adjacency(F, n)
    dist = np.full(A.shape[0], -1)
    dist[np.asarray(seed_vertices)] = 0
    frontier = dist == 0
    d = 0
    while frontier.any() and (max_k is None or d < max_k):
        d += 1
        frontier = (A @ frontier > 0) & (dist == -1)   # neighbors of the last ring, not yet seen
        dist[frontier] = d
    return dist


def k_ring(F, seed_vertices, k, n=None):
    """Sorted indices of all vertices within ``k`` edges of the seeds (seeds included).

    This is the n-ring neighborhood ``N_k`` of Sec. 3.3.1, grown from a set of
    vertices. ``k_ring(F, [i], 1)`` is ``vi`` plus its one-ring.
    """
    dist = ring_distance(F, seed_vertices, max_k=k, n=n)
    return np.flatnonzero(dist >= 0)
