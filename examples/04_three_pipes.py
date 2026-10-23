"""Fair the junction of three pipes in a plane, for k = 1, 2, 3.

Run from the repository root:

    python examples/04_three_pipes.py

Three fixed pipes of radius 1 lie in the xy-plane at 120 degrees, pointing up-right,
up-left and down. The free junction between them starts as a smooth blend: its top
and bottom halves are mirror images, each made of three Coons patches (one per pipe)
bounded by the pipe's joint half-ring, two halves of the fillet curves between
neighboring pipes in the plane z = 0, and two quarter ellipses rising to the hub
(0, 0, 1). The fairing operator is frozen on this start mesh and solved for k = 1,
2, 3 with cotangent weights (top row) and uniform weights (bottom row).
Saves docs/img/11-three-pipes-k123.png.
"""

from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

from fairing import viz
from fairing.fairing import solve_fair


IMG = Path(__file__).resolve().parents[1] / "docs" / "img"
EZ = np.array([0.0, 0.0, 1.0])


def _split(a, b, c, d, flip):
    """Two triangles per quad ``a b c d``, cut along ``a c``, or ``b d`` where ``flip``."""
    one = np.where(flip[:, None], np.stack([a, b, d], 1), np.stack([a, b, c], 1))
    two = np.where(flip[:, None], np.stack([b, c, d], 1), np.stack([a, c, d], 1))
    return np.concatenate([one, two])


def _grid_faces(nu, nv, offset):
    """Triangles of an ``(nu+1) x (nv+1)`` grid stored as ``i + (nu+1)*j``.

    The diagonals flip at ``i = nu/2``, so the triangulation is mirror symmetric
    in ``i``; one diagonal everywhere gives it a handedness that uniform weights,
    which only see connectivity, turn into a twist.
    """
    i, j = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    i, j = i.ravel(), j.ravel()
    a = i + (nu + 1) * j + offset
    return _split(a, a + 1, a + 1 + (nu + 1), a + (nu + 1), i >= nu // 2)


def _orient(F):
    """Flip triangles so that neighbors agree (each directed edge used once)."""
    F = F.copy()
    owner = {}
    for f, tri in enumerate(F):
        for e in range(3):
            owner.setdefault(frozenset((tri[e], tri[(e + 1) % 3])), []).append(f)
    seen = np.zeros(len(F), bool)
    for start in range(len(F)):
        if seen[start]:
            continue
        seen[start] = True
        queue = deque([start])
        while queue:
            f = queue.popleft()
            for e in range(3):
                u, v = F[f, e], F[f, (e + 1) % 3]
                for g in owner[frozenset((u, v))]:
                    if g == f or seen[g]:
                        continue
                    # a consistent neighbor walks the shared edge as v -> u
                    if any(F[g, k] == u and F[g, (k + 1) % 3] == v for k in range(3)):
                        F[g] = F[g, ::-1]
                    seen[g] = True
                    queue.append(g)
    return F


def three_pipes(n_half=48, n_len=32, radius=1.0, joint_distance=2.0, pipe_length=1.5,
                fillet=1.0):
    """Return ``V, F, free`` for three pipes at 120 degrees joined by a free junction.

    Pipe i has axis direction ``d_i`` (angles 30, 150, 270 degrees), joint ring
    centered at ``joint_distance * d_i`` and ``2 * n_half`` vertices per ring. The
    fillet between neighboring pipes is a cubic Bezier curve in z = 0, tangent to
    both pipes' sides, with control arms of length ``fillet``.
    """
    angles = np.radians([30.0, 150.0, 270.0])
    d = np.stack([np.cos(angles), np.sin(angles), 0 * angles], 1)
    s = np.cross(EZ, d)                                      # in-plane side, counterclockwise
    c = joint_distance * d
    n_theta = 2 * n_half

    def ring(i, theta, t=0.0):                              # theta = +pi/2 is the side +s_i
        return (c[i] + t * d[i])[None] + radius * (np.cos(theta)[:, None] * EZ
                                                   + np.sin(theta)[:, None] * s[i])

    def fillet_curve(i, t):                                 # from pipe i to pipe i+1, t in [0, 1]
        j = (i + 1) % 3
        P = [c[i] + radius * s[i], c[i] + radius * s[i] - fillet * d[i],
             c[j] - radius * s[j] - fillet * d[j], c[j] - radius * s[j]]
        t = np.asarray(t, float)[:, None]
        return ((1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1]
                + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3])

    def to_hub(M, phi):                                     # quarter ellipse from M (z = 0) to the hub
        return np.cos(phi)[:, None] * M + radius * np.sin(phi)[:, None] * EZ

    u = np.linspace(0, 1, n_half + 1)
    v = np.linspace(0, 1, n_len + 1)
    mid = [fillet_curve(i, [0.5])[0] for i in range(3)]
    points, faces, pipe_vertices = [], [], []
    count = 0
    for i in range(3):
        B = ring(i, np.pi / 2 - np.pi * u)                  # joint half-ring, side +s_i to side -s_i
        Lside = fillet_curve(i, v / 2)                      # toward pipe i+1
        Rside = fillet_curve((i - 1) % 3, 1 - v / 2)        # toward pipe i-1
        first = u <= 0.5
        T = np.empty_like(B)
        T[first] = to_hub(mid[i], np.pi * u[first])
        T[~first] = to_hub(mid[(i - 1) % 3], np.pi * (1 - u[~first]))
        U, W = u[:, None, None], v[None, :, None]
        S = ((1 - W) * B[:, None] + W * T[:, None] + (1 - U) * Lside[None] + U * Rside[None]
             - ((1 - U) * (1 - W) * B[0] + U * (1 - W) * B[-1]
                + (1 - U) * W * T[0] + U * W * T[-1]))
        top = S.transpose(1, 0, 2).reshape(-1, 3)           # index i + (n_half+1)*j
        for patch in (top, top * [1, 1, -1]):               # the bottom half mirrors the top
            points.append(patch)
            faces.append(_grid_faces(n_half, n_len, count))
            count += len(patch)
    for i in range(3):
        theta = np.pi / 2 - 2 * np.pi * np.arange(n_theta) / n_theta
        rings = int(round(pipe_length / (np.pi * radius / n_half)))
        tube = np.concatenate([ring(i, theta, t) for t in np.linspace(0, pipe_length, rings + 1)])
        k = np.arange(n_theta)
        flip = (k // (n_half // 2)) % 2 == 1   # per quadrant, as above
        for r in range(rings):
            a, b = count + r * n_theta + k, count + r * n_theta + (k + 1) % n_theta
            faces.append(_split(a, b, b + n_theta, a + n_theta, flip))
        points.append(tube)
        pipe_vertices.append(np.arange(count, count + len(tube)))
        count += len(tube)

    P = np.concatenate(points)
    # Weld the copies of every shared boundary point.
    tree = cKDTree(P)
    pairs = tree.query_pairs(1e-9, output_type="ndarray")
    rep = np.arange(len(P))
    for a, b in sorted(map(tuple, np.sort(pairs, axis=1)), reverse=True):
        rep[b] = rep[a]
    while (rep[rep] != rep).any():
        rep = rep[rep]
    keep, index = np.unique(rep, return_inverse=True)
    V = P[keep]
    F = index[np.concatenate(faces)]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    F = _orient(F)
    fixed = np.zeros(len(V), bool)
    fixed[index[np.concatenate(pipe_vertices)]] = True
    # Outward normals: a pipe triangle should point away from its axis.
    a, b, cc = V[F[0]]
    normal = np.cross(b - a, cc - a)
    if F[0][0] in set(index[pipe_vertices[0]]):
        axis_point = c[0] + np.dot(a - c[0], d[0]) * d[0]
    else:
        axis_point = np.zeros(3)
    if np.dot(normal, a - axis_point) < 0:
        F = F[:, ::-1]
    return V, F, ~fixed


def figure_three_pipes():
    V, F, free = three_pipes()
    free_face = free[F].any(axis=1)
    fig = plt.figure(figsize=(18.4, 9.6))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.90, wspace=0.02, hspace=0.12)
    for r, (lap, name) in enumerate((("cotan", "cotangent"), ("uniform", "uniform"))):
        ax = fig.add_subplot(2, 4, 4 * r + 1, projection="3d")
        _draw(ax, V, F, free_face, V[~free])
        ax.set_title(f"start mesh, {name} row", fontsize=12, pad=2)
        for k in (1, 2, 3):
            W = solve_fair(V, F, free, k, laplacian=lap)
            assert np.array_equal(W[~free], V[~free]) and np.isfinite(W).all()
            ax = fig.add_subplot(2, 4, 4 * r + k + 1, projection="3d")
            _draw(ax, W, F, free_face, V[~free])
            ax.set_title(f"{name}, k = {k}", fontsize=12, pad=2)
    fig.suptitle("Three fixed pipes joined by a free junction, faired with k = 1, 2, 3",
                 fontsize=16, y=0.98)
    fig.text(0.5, 0.015,
             f"{len(V):,} vertices · {len(F):,} triangles · gray: fixed pipes · red: faired region",
             ha="center", fontsize=10, color="#555555")
    IMG.mkdir(parents=True, exist_ok=True)
    target = IMG / "11-three-pipes-k123.png"
    fig.savefig(target, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{len(V):,} vertices, {len(F):,} triangles, {free.sum():,} free")
    print("saved", target)


def _draw(ax, W, F, free_face, frame):
    """Draw fixed and free triangles as one collection, shaded like ``plot_trisurf``.

    Matplotlib depth-sorts triangles only within a collection and paints whole
    collections one after the other. With the gray and red parts in two
    ``plot_trisurf`` calls, the back wall of each gray pipe was painted over the
    red junction, which looked like a dark gap at every joint.
    """
    T = W[F]
    normals = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    base = np.where(free_face[:, None], matplotlib.colors.to_rgb("#c8372d"),
                    matplotlib.colors.to_rgb("#d9d8d4"))
    shade = LightSource(azdeg=225, altdeg=19.4712).shade_normals(normals, fraction=1.0)
    colors = base * (0.3 + 0.7 * shade[:, None])            # matplotlib's own shading mix
    ax.add_collection3d(Poly3DCollection(T, facecolors=colors, linewidths=0, antialiased=False))
    viz._set_equal_aspect(ax, frame)
    ax.view_init(70, -90)
    ax.set_axis_off()


if __name__ == "__main__":
    figure_three_pipes()
