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


def graded_grid(nx, ny, size=1.0, strength=0.7):
    """Planar grid like :func:`grid` whose vertex density varies smoothly (bands).

    Each coordinate is warped by ``x = u + s sin(4 pi u / size) size / (4 pi)``
    for u in [-size/2, size/2]. The spacing is then scaled by
    ``1 + s cos(4 pi u / size)``, so dense bands (around u = +-size/4) alternate
    with sparse ones (center and edges), with a density ratio of (1 + s)/(1 - s).
    The square outline and its corners stay put (boundary vertices only slide
    along it), the connectivity is the regular grid's,
    and ``strength < 1`` keeps every triangle correctly oriented (the map is
    monotone). This is the "varying vertex density" setting of Fig. 4.9.
    """
    if not 0 <= strength < 1:
        raise ValueError("strength must be in [0, 1) to keep the warp monotone")
    V, F = grid(nx, ny, size)
    warp = lambda u: u + strength * np.sin(4 * np.pi * u / size) * size / (4 * np.pi)
    V[:, 0], V[:, 1] = warp(V[:, 0]), warp(V[:, 1])
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


def irregular_sphere(n_lat, n_lon, radius=1.0, jitter=0.15, seed=0):
    """UV sphere like :func:`uv_sphere`, with the ring vertices moved *along* the sphere.

    Each ring vertex gets a uniform random offset of at most ``jitter`` times
    the grid spacing in latitude and in longitude, and stays on the sphere.
    The geometry is still an exact sphere (H = 1/R), but the triangles are
    irregular: the setting of Fig. 4.6, where the uniform Laplacian also moves
    vertices tangentially and the cotangent Laplacian does not.

    As for :func:`irregular_grid`, ``jitter < 1/6`` keeps every triangle
    correctly oriented (the bound holds in the (latitude, longitude) grid).
    """
    if not 0 <= jitter < 1 / 6:
        raise ValueError("jitter must be in [0, 1/6) to guarantee that no triangle flips")
    V, F = uv_sphere(n_lat, n_lon, radius)
    rng = np.random.default_rng(seed)
    ring = V[1:-1]
    theta = np.arccos(np.clip(-ring[:, 2] / radius, -1, 1))     # polar angle from the south pole
    phi = np.arctan2(ring[:, 1], ring[:, 0])
    theta += rng.uniform(-jitter, jitter, len(ring)) * np.pi / n_lat
    phi += rng.uniform(-jitter, jitter, len(ring)) * 2 * np.pi / n_lon
    V = V.copy()
    V[1:-1] = radius * np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.cos(theta)], axis=1)
    return V, F


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


def pipe_elbow(n_theta=48, radius=1.0, bend_radius=1.5, pipe_length=1.2, spacing=0.1):
    """Two pipes at 90 degrees joined by a quarter-torus bend (the setting of Fig. 4.8).

    A vertical pipe (axis z, from z = -pipe_length to 0) and a horizontal pipe
    (axis x, at height ``bend_radius``) are joined by a bend around the corner
    point (bend_radius, 0, 0). The mesh reuses the tube's connectivity (angle i
    on ring j is vertex ``i + n_theta*j``); only the ring positions change.
    Ring j is a circle of the given radius around a center c_j, spanned by
    e2 = y and a direction e1_j that turns with the bend, so it stays
    perpendicular to the pipe's axis and the triangles keep their outward
    orientation. The bend rings lie on an exact quarter torus.

    Returns ``V, F, bend`` where ``bend`` is a boolean mask of the bend's
    vertices (the free region of Fig. 4.8); the two straight pipes are the rest.
    The default proportions (pipes 1.2 long for radius 1) are close to the
    book's figure; the pipe length does not change the fairing result as long
    as each pipe has at least k rings (only those enter the equations).
    """
    n_bend = int(round(bend_radius * np.pi / 2 / spacing))       # bend segments, each ~ spacing long
    n_pipe = int(round(pipe_length / spacing))                    # segments per straight pipe
    centers, e1 = [], []
    for j in range(n_pipe + 1):                                   # vertical pipe, ends at z = 0
        centers.append((0, 0, -pipe_length + j * spacing))
        e1.append((1, 0, 0))
    for m in range(1, n_bend):                                    # bend: angle phi from 0 to 90 degrees
        phi = m * (np.pi / 2) / n_bend
        centers.append((bend_radius * (1 - np.cos(phi)), 0, bend_radius * np.sin(phi)))
        e1.append((np.cos(phi), 0, -np.sin(phi)))
    for j in range(n_pipe + 1):                                   # horizontal pipe, along +x
        centers.append((bend_radius + j * spacing, 0, bend_radius))
        e1.append((0, 0, -1))
    centers, e1 = np.array(centers, float), np.array(e1, float)
    _, F = tube(n_theta, len(centers))
    theta = 2 * np.pi * np.arange(n_theta) / n_theta
    circle = (np.cos(theta)[None, :, None] * e1[:, None, :]
              + np.sin(theta)[None, :, None] * np.array([0.0, 1.0, 0.0]))
    V = (centers[:, None, :] + radius * circle).reshape(-1, 3)
    ring = np.repeat(np.arange(len(centers)), n_theta)
    bend = (ring > n_pipe) & (ring < n_pipe + n_bend)
    return V, F, bend


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


def load_mesh(path):
    """Read a triangle mesh in any format trimesh understands (e.g. PLY), as plain ``V, F``.

    The file is read as is (``process=False``: no merging or repairing), and
    vertices that no triangle uses are dropped, with ``F`` re-indexed. Needs the
    optional dependency: ``pip install -e ".[mesh]"``.
    """
    try:
        import trimesh
    except ImportError as err:
        raise ImportError('trimesh is not installed; run: pip install -e ".[mesh]"') from err
    m = trimesh.load(path, process=False, force="mesh")
    V, F = np.asarray(m.vertices, dtype=float), np.asarray(m.faces, dtype=int)
    used, F = np.unique(F, return_inverse=True)
    return V[used], F.reshape(-1, 3)


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


def edge_face_counts(F):
    """Unique edges ``(E, 2)`` and, for each, the number of triangles that contain it.

    On a manifold mesh every interior edge is in exactly 2 triangles and every
    boundary edge in 1. A count above 2 marks a *non-manifold* edge, where the
    discrete operators of Ch. 3 (one-rings, cotangent weights) are not defined
    as in the book.
    """
    return np.unique(np.sort(_half_edges(F), axis=1), axis=0, return_counts=True)


def boundary_vertices(F):
    """Sorted indices of the vertices on the mesh boundary.

    An interior edge is shared by two triangles; a boundary edge belongs to
    exactly one. The boundary vertices are the endpoints of boundary edges.
    """
    E, counts = edge_face_counts(F)
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


def add_noise(V, sigma, seed=0):
    """Add isotropic Gaussian noise (standard deviation ``sigma`` per coordinate) to every vertex."""
    rng = np.random.default_rng(seed)
    return np.asarray(V, dtype=float) + rng.normal(0.0, sigma, np.shape(V))


def flatten_onto_border_plane(V, F, free_mask):
    """Project the free vertices onto the plane that best fits the region's border.

    The border is the first ring of fixed vertices around the free region; the
    plane is its least-squares fit (through their centroid, normal = direction
    of least spread). Used as a crude "patch" to be refilled by fairing (#14).
    """
    V = np.asarray(V, dtype=float).copy()
    free = np.asarray(free_mask, dtype=bool)
    border = ring_distance(F, np.flatnonzero(free), max_k=1, n=len(V)) == 1
    center = V[border].mean(axis=0)
    normal = np.linalg.svd(V[border] - center)[2][-1]
    V[free] -= np.outer((V[free] - center) @ normal, normal)
    return V
