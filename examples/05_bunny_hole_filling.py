"""Phase 5 example, complementary analysis: filling a real hole on the bunny's upper back.

The other bunny examples keep the region's triangles and only move its vertices. Here the
region is really removed, and the hole is filled in two separate steps:

1. Patch construction (this script): cut out the faces and vertices of the upper-back region
   of examples/05_bunny_back_region.py, fit a plane to the hole's boundary loop, project the
   loop into the plane, triangulate the polygon with a constrained Delaunay triangulation
   (Shewchuk's Triangle, through the `triangle` package) that adds interior vertices but no
   vertex on the boundary, place the new interior vertices on the plane, and attach the
   patch to the surviving boundary vertices, which keep their exact 3D positions.
2. Fairing (the library, unchanged): solve_fair(V_init, F_new, free_new, 2, weights) with
   only the patch's new interior vertices free, once with uniform and once with cotangent
   weights, both frozen from the same initialized mesh.

Patch construction supplies the triangulation; fairing computes a smooth shape for its
interior. The removed vertices' positions and triangles are never used to build the patch:
the original bunny serves only to select the region, to draw it, and to measure the result.

Run from the repository root (needs the full-resolution bunny, see README, and the `mesh` and
`hole` extras: pip install -e ".[mesh,hole]"):

    python examples/05_bunny_hole_filling.py

Saves docs/img/14-bunny-hole-filling-stages.png and docs/img/14-bunny-hole-filling-weights.png
and prints the counts, checks and distances quoted in the phase-5 note.
Data: Stanford Computer Graphics Laboratory (Stanford 3D Scanning Repository).
"""

import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

from fairing import mesh, viz
from fairing.fairing import fairing_matrix, solve_fair
from fairing.laplacian import face_areas, mean_curvature

HERE = Path(__file__).resolve().parent
back = runpy.run_path(str(HERE / "05_bunny_back_region.py"), run_name="back_helpers")
book, helpers = back["book"], back["helpers"]
IMG = helpers["IMG"]
K = 2
MIN_ANGLE = 30                              # Triangle's quality bound, in degrees ("q30")
VIEW = back["VIEW"]                         # the picture's camera of 05_bunny_back_region.py
CLOSE_VIEW = (32, -28)                      # close-ups: the region at an angle, its profile visible
GRAY, BLUE, LILAC, GREEN = "#d9d8d4", "#5b62c9", "#a5a9e6", "#3aa655"
ORANGE, TEAL, INK = "#d4581f", "#13897d", "#3d3c39"   # uniform labels, cotangent labels, text
LIGHT = back["LIGHT"]


# ---------------------------------------------------------------------------
# Cutting the hole
# ---------------------------------------------------------------------------

def boundary_loops(F):
    """The mesh's boundary loops, each a list of vertex indices in the order of its half-edges.

    A boundary half-edge (a, b) is one whose twin (b, a) is in no triangle. Each boundary
    vertex must start exactly one of them, otherwise the boundary pinches (not a simple loop).
    """
    half = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    present = set(map(tuple, half.tolist()))
    after = {}
    for a, b in half.tolist():
        if (b, a) not in present:
            if a in after:
                raise ValueError(f"vertex {a} starts two boundary half-edges: the boundary pinches there")
            after[a] = b
    loops = []
    while after:
        start = min(after)
        loop, v = [start], after.pop(start)
        while v != start:
            loop.append(v)
            v = after.pop(v)
        loops.append(loop)
    return loops


def directed(loop):
    """The half-edges (a, b) of a loop, as a set of pairs."""
    return set(zip(loop, loop[1:] + loop[:1]))


def cut_hole(V, F, free):
    """Remove the region's faces (every face with a free vertex) and its vertices.

    Returns the surviving mesh ``V_hole, F_hole`` (re-indexed, without the removed vertices),
    ``kept`` (the original index of each surviving vertex) and ``loop``, the new boundary loop
    in surviving-mesh indices. The bunny's other boundary loops must survive unchanged.
    """
    F_keep = F[~free[F].any(axis=1)]
    kept = np.unique(F_keep)
    if not np.array_equal(kept, np.flatnonzero(~free)):
        raise ValueError("removing the faces would also isolate fixed vertices")
    new_index = np.full(len(V), -1)
    new_index[kept] = np.arange(len(kept))
    V_hole, F_hole = V[kept], new_index[F_keep]

    before = [directed([int(new_index[v]) for v in loop]) for loop in boundary_loops(F)]
    after = boundary_loops(F_hole)
    new = [loop for loop in after if directed(loop) not in before]
    if len(new) != 1 or len(after) != len(before) + 1:
        raise ValueError(f"expected one new boundary loop, found {len(new)}")
    border = mesh.ring_distance(F, np.flatnonzero(free), max_k=1, n=len(V)) == 1
    if set(kept[new[0]].tolist()) != set(np.flatnonzero(border).tolist()):
        raise ValueError("the new loop is not the region's first fixed ring")
    return V_hole, F_hole, kept, np.array(new[0])


# ---------------------------------------------------------------------------
# Constructing the replacement patch (uses only the surviving mesh)
# ---------------------------------------------------------------------------

def check_simple_polygon(uv):
    """Raise ValueError unless the closed polygon ``uv`` is simple: no repeated vertex, no two
    non-adjacent edges that touch or cross, and a nonzero area."""
    n = len(uv)
    if cKDTree(uv).query(uv, k=2)[0][:, 1].min() == 0:
        raise ValueError("the projected boundary has repeated vertices")
    a, b = uv, np.roll(uv, -1, axis=0)
    i, j = np.triu_indices(n, k=2)
    keep = (j - i) % n != n - 1                                 # the last edge touches the first
    i, j = i[keep], j[keep]
    orient = lambda p, q, r: np.sign((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0]))
    cross = ((orient(a[i], b[i], a[j]) * orient(a[i], b[i], b[j]) <= 0)
             & (orient(a[j], b[j], a[i]) * orient(a[j], b[j], b[i]) <= 0))
    if cross.any():
        raise ValueError(f"the projected boundary intersects itself ({cross.sum()} edge pairs): "
                         "the hole is too curved for a planar parametrization")
    if (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() == 0:
        raise ValueError("the projected boundary encloses no area")


def build_patch(V, F, loop, min_angle=MIN_ANGLE):
    """Triangulate the hole bounded by ``loop`` and attach the patch to the mesh ``V, F``.

    1. Fit a plane to the loop (least squares, through its centroid), with the normal on the
       side of the surviving faces around the loop.
    2. Project the loop into the plane's 2D coordinates and check that it is a simple polygon.
    3. Constrained Delaunay triangulation of the polygon (Triangle): the loop's edges are
       segments, no vertex may be added on them ("Y"), angles are at least ``min_angle``
       ("q"), and triangle areas at most that of an equilateral triangle whose side is the
       loop's mean 3D edge length ("a"), so the patch is sampled like the surviving mesh.
    4. Put the new interior vertices on the plane; the loop vertices are the mesh's own
       vertices, so they keep their exact 3D positions and are not duplicated.

    Only ``V`` and ``F`` (the surviving mesh) are read. Returns ``V_init, F_new, free_new``
    (the patch's interior vertices are appended after ``V``, its faces after ``F``) and a
    dict with the plane, the 2D triangulation and the edge length used.
    """
    import triangle

    loop = np.asarray(loop)
    P = V[loop]
    h = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1).mean()
    center = P.mean(axis=0)
    e1, _, normal = np.linalg.svd(P - center)[2]
    around = F[np.isin(F, loop).any(axis=1)]
    outward = np.cross(V[around[:, 1]] - V[around[:, 0]], V[around[:, 2]] - V[around[:, 0]]).sum(axis=0)
    if normal @ outward < 0:
        normal = -normal
    e2 = np.cross(normal, e1)                                   # e1 x e2 = normal
    uv = (P - center) @ np.stack([e1, e2]).T / h                # in units of the edge length
    check_simple_polygon(uv)

    n_loop = len(loop)
    segments = np.stack([np.arange(n_loop), (np.arange(n_loop) + 1) % n_loop], axis=1)
    out = triangle.triangulate({"vertices": uv, "segments": segments},
                               f"pq{min_angle}Ya{np.sqrt(3) / 4:.6f}Q")
    if not np.array_equal(out["vertices"][:n_loop], uv):
        raise RuntimeError("Triangle moved or reordered the boundary vertices")
    inner_uv = out["vertices"][n_loop:]
    index = np.concatenate([loop, len(V) + np.arange(len(inner_uv))])
    faces = index[out["triangles"]]                              # counter-clockwise in (e1, e2)
    if directed(loop.tolist()) & set(map(tuple, np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).tolist())):
        faces = faces[:, ::-1]                                  # the patch must run against the hole
    inner = center + h * inner_uv @ np.stack([e1, e2])          # on the fitted plane

    V_init = np.vstack([V, inner])
    F_new = np.vstack([F, faces])
    free_new = np.zeros(len(V_init), dtype=bool)
    free_new[len(V):] = True
    info = dict(h=h, center=center, normal=normal, e1=e1, e2=e2)
    return V_init, F_new, free_new, info


# ---------------------------------------------------------------------------
# Checks and measurements
# ---------------------------------------------------------------------------

def euler(F):
    """V - E + F of the mesh (vertices counted as those used by a face)."""
    return len(np.unique(F)) - len(mesh.edges(F)) + len(F)


def check_attachment(V_hole, F_hole, loop, V_init, F_new, free_new, h):
    """Check that the patch closes the hole cleanly; returns a few numbers for the report."""
    n = len(V_hole)
    patch = F_new[len(F_hole):]
    assert np.isfinite(V_init).all(), "non-finite coordinates"
    assert np.array_equal(V_init[:n], V_hole) and np.array_equal(F_new[:len(F_hole)], F_hole)
    old = np.unique(patch[patch < n])
    assert np.array_equal(old, np.sort(loop)), "the patch must reuse exactly the loop's vertices"
    assert np.array_equal(np.unique(patch[patch >= n]), np.arange(n, len(V_init))), "unused new vertices"
    gap = cKDTree(V_init[loop]).query(V_init[free_new])[0].min()
    assert gap > 0.1 * h, "a new vertex (nearly) duplicates a boundary vertex"

    areas = face_areas(V_init, patch)
    assert areas.min() > 0, "degenerate initial triangle"
    E, counts = mesh.edge_face_counts(F_new)
    assert counts.max() <= 2, "non-manifold edge"
    open_edges = {tuple(e) for e in E[counts == 1].tolist()}
    others = {tuple(sorted(e)) for l in boundary_loops(F_hole) if set(l) != set(loop.tolist()) for e in directed(l)}
    assert open_edges == others, "gap or seam: boundary edges other than the bunny's own"
    half = np.concatenate([F_new[:, [0, 1]], F_new[:, [1, 2]], F_new[:, [2, 0]]])
    assert len(np.unique(half, axis=0)) == len(half), "inconsistent orientation (a half-edge appears twice)"
    return dict(min_area=areas.min() / (np.sqrt(3) / 4 * h ** 2), gap=gap / h, open_edges=len(open_edges))


def min_angles(V, F):
    """Smallest interior angle of each triangle, in degrees."""
    angles = []
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u, v = V[F[:, b]] - V[F[:, a]], V[F[:, c]] - V[F[:, a]]
        angles.append(np.degrees(np.arccos(np.clip(
            (u * v).sum(axis=1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1), -1, 1))))
    return np.min(angles, axis=0)


def distance_to_surface(points, V, F):
    """Exact distance from each point to the triangle mesh ``V, F``, and the closest triangle.

    Candidates: the triangles whose centroid lies within d0 + R of the point, where d0 is the
    distance to the nearest centroid (an upper bound of the true distance) and R the largest
    centroid-to-corner distance. For each, the point is projected onto the triangle's plane;
    if the projection falls outside, the closest point lies on one of its three edges.
    """
    T = V[F]
    centroid = T.mean(axis=1)
    R = np.linalg.norm(T - centroid[:, None], axis=2).max()
    tree = cKDTree(centroid)
    d0 = tree.query(points)[0]
    lists = tree.query_ball_point(points, d0 + R)
    who = np.repeat(np.arange(len(points)), [len(c) for c in lists])
    tri = np.concatenate(lists).astype(int)
    p, (a, b, c) = points[who], T[tri].transpose(1, 0, 2)

    n = np.cross(b - a, c - a)
    q = p - ((p - a) * n).sum(axis=1, keepdims=True) / (n * n).sum(axis=1, keepdims=True) * n
    inside = np.ones(len(p), dtype=bool)
    for s, t in ((a, b), (b, c), (c, a)):                       # q on the inner side of every edge
        inside &= (np.cross(t - s, q - s) * n).sum(axis=1) >= 0
    best = np.where(inside, np.linalg.norm(p - q, axis=1), np.inf)
    for s, t in ((a, b), (b, c), (c, a)):
        w = np.clip(((p - s) * (t - s)).sum(axis=1) / ((t - s) ** 2).sum(axis=1), 0, 1)
        best = np.minimum(best, np.linalg.norm(p - s - w[:, None] * (t - s), axis=1))

    dist = np.full(len(points), np.inf)
    np.minimum.at(dist, who, best)
    closest = np.full(len(points), -1)
    hit = best == dist[who]
    closest[who[hit]] = tri[hit]
    return dist, closest


def surface_distances(W, patch, V, removed):
    """Compare the filled patch with the original surface it replaces (evaluation only).

    Both surfaces are sampled at their vertices and face centroids. Returns the mean and max
    distance from the patch to the original, the max in the other direction, and the mean
    signed offset of the patch along the original's face normals (< 0: below the original).
    """
    samples = lambda X, T: np.vstack([X[np.unique(T)], X[T].mean(axis=1)])
    there = samples(W, patch)
    d, tri = distance_to_surface(there, V, removed)
    normal = np.cross(V[removed[:, 1]] - V[removed[:, 0]], V[removed[:, 2]] - V[removed[:, 0]])[tri]
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    side = np.sign(((there - V[removed[tri, 0]]) * normal).sum(axis=1))
    back_d, _ = distance_to_surface(samples(V, removed), W, patch)
    return dict(mean=d.mean(), max=d.max(), back_max=back_d.max(), offset=(side * d).mean())


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def render(ax, V, F, color, view, frame, title, title_color=INK, edges=None, loop=None):
    """Lit faces in one depth-sorted collection (see 05_bunny_back_region.py), per-face colors.

    ``edges`` marks the faces whose triangle edges are drawn; ``loop`` is drawn as a green
    line on top. ``frame`` (points) sets the axes' limits, so panels share one scale."""
    normal = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    brightness = 0.3 + 0.35 * (1 + normal @ LIGHT)              # the shading of plot_trisurf
    edge_colors = np.zeros((len(F), 4))
    if edges is not None:
        edge_colors[edges] = to_rgba("#eef0ff", 0.7)
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=color * brightness[:, None],
                                         edgecolors=edge_colors, linewidths=0.25, antialiased=edges is not None))
    if loop is not None:
        ax.computed_zorder = False
        ring = V[np.append(loop, loop[0])]
        ax.plot(*ring.T, color=GREEN, linewidth=1.6, zorder=10)
    viz._set_equal_aspect(ax, frame)
    ax.view_init(*view)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, color=title_color)


def colors(n_faces, patch_from):
    """Gray for faces before ``patch_from``, blue for the patch's faces."""
    c = np.tile(to_rgb(GRAY), (n_faces, 1))
    c[patch_from:] = to_rgb(BLUE)
    return c


def crop(V, F, center, radius):
    """The faces with a corner within ``radius`` of ``center`` (for close-ups)."""
    return F[(np.linalg.norm(V[F] - center, axis=2) < radius).any(axis=1)]


def section(V, F, origin, along, up, across):
    """Segments where the plane through ``origin`` with normal ``across`` cuts the triangles,
    in the plane's 2D coordinates (``along``, ``up``)."""
    s = (V - origin) @ across
    below = s < 0
    cut = F[below[F].any(axis=1) & ~below[F].all(axis=1)]       # exactly two edges cross the plane
    points, crossing = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        i, j = cut[:, a], cut[:, b]
        crossing.append(below[i] != below[j])
        t = s[i] / np.where(crossing[-1], s[i] - s[j], 1)
        points.append(V[i] + t[:, None] * (V[j] - V[i]))
    points, crossing = np.stack(points, axis=1), np.stack(crossing, axis=1)
    first_two = np.argsort(~crossing, axis=1, kind="stable")[:, :2]      # the two crossing edges
    ends = np.take_along_axis(points, first_two[:, :, None], axis=1)
    return np.stack([(ends - origin) @ along, (ends - origin) @ up], axis=-1)


# ---------------------------------------------------------------------------

def main():
    V, F = helpers["load_bunny"](book["FULL"])
    free, region_euler, clearance = book["book_region"](V, F, back["RADIUS"], back["TARGET"])
    assert region_euler == 1 and clearance >= K + 1
    removed = F[free[F].any(axis=1)]

    # 1. the hole. From here on, only the surviving mesh is used to build and fair the patch.
    V_hole, F_hole, kept, loop = cut_hole(V, F, free)
    # 2. the replacement patch, then 3. the two thin plates, from one initialized mesh
    V_init, F_new, free_new, info = build_patch(V_hole, F_hole, loop)
    results = {lap: solve_fair(V_init, F_new, free_new, K, laplacian=lap) for lap in ("uniform", "cotan")}

    # the removed region's positions never enter: poison them and rebuild everything
    poisoned = V.copy()
    poisoned[free] = np.nan
    V2, F2, _, loop2 = cut_hole(poisoned, F, free)
    V2_init, F2_new, free2, _ = build_patch(V2, F2, loop2)
    rebuilt = all(np.array_equal(a, b) for a, b in ((V2_init, V_init), (F2_new, F_new), (free2, free_new)))
    rebuilt &= all(np.array_equal(solve_fair(V2_init, F2_new, free2, K, laplacian=lap), W) for lap, W in results.items())
    assert rebuilt, "the patch or its fairing depends on the removed positions"

    h = info["h"]
    checks = check_attachment(V_hole, F_hole, loop, V_init, F_new, free_new, h)
    assert euler(F_new) == euler(F), "the filled mesh does not have the original's topology"
    patch = F_new[len(F_hole):]
    angles = min_angles(V_init, patch)
    loop_points = V_hole[loop]
    flatness = np.abs((loop_points - info["center"]) @ info["normal"])

    print(f"full-resolution bunny: {len(V)} vertices, {len(F)} triangles, Euler characteristic {euler(F)}")
    print(f"removed region: {free.sum()} vertices, {len(removed)} triangles "
          f"(a disk, at least {clearance} rings from the bunny's other boundary edges)")
    print(f"surviving mesh: {len(V_hole)} vertices, {len(F_hole)} triangles; hole loop: {len(loop)} vertices; "
          f"{len(boundary_loops(F_hole)) - 1} other boundary loops unchanged")
    print(f"hole loop: mean edge h = {h:.5f}; distance from the fitted plane up to {flatness.max():.4f} "
          f"({flatness.max() / h:.1f} h), mean {flatness.mean() / h:.1f} h")
    print(f"patch: {free_new.sum()} new interior vertices, {len(patch)} triangles "
          f"(Triangle 'pq{MIN_ANGLE}Y', max area of an equilateral triangle of side h)")
    print(f"filled mesh: {len(V_init)} vertices, {len(F_new)} triangles, Euler characteristic {euler(F_new)}")
    print("checks on the initialized patch:")
    print("  built from the surviving mesh only: rebuilt identically with the removed positions set to NaN")
    print(f"  attached to the {len(loop)} loop vertices themselves (no copies; nearest new vertex {checks['gap']:.2f} h away); "
          f"no gaps, seams or non-manifold edges ({checks['open_edges']} open edges = the bunny's own)")
    print(f"  consistent orientation; all coordinates finite; smallest triangle area {checks['min_area']:.2f} "
          f"of an equilateral one of side h; smallest angle {angles.min():.1f} deg (median of the per-triangle "
          f"minima {np.median(angles):.1f} deg)")

    print(f"thin plate (k = {K}), {free_new.sum()} free vertices, weights frozen from the initialized mesh:")
    H = lambda W, Fm, mask: mean_curvature(W, Fm)[mask].mean()
    print(f"  original removed region: mean |H| {H(V, F, free):.1f}")
    dist = surface_distances(V_init, patch, V, removed)
    print(f"  initial planar patch: distance to the original surface mean {dist['mean']:.4f} ({dist['mean'] / h:.1f} h), "
          f"max {dist['max']:.4f}, other way max {dist['back_max']:.4f}; mean offset {dist['offset']:+.4f}")
    for lap, W in results.items():
        A = fairing_matrix(V_init, F_new, K, lap)                 # the matrix solve_fair froze
        residual = np.linalg.norm((A @ W)[free_new]) / np.linalg.norm(A[free_new][:, ~free_new] @ W[~free_new])
        fixed_same = np.array_equal(W[~free_new], V_init[~free_new])
        assert fixed_same and residual < 1e-10
        dist = surface_distances(W, patch, V, removed)
        print(f"  {lap:7s}: fixed vertices unchanged {fixed_same}; relative residual {residual:.1e}; "
              f"mean |H| {H(W, F_new, free_new):.1f}; distance to the original surface mean {dist['mean']:.4f} "
              f"({dist['mean'] / h:.1f} h), max {dist['max']:.4f}, other way max {dist['back_max']:.4f}; "
              f"mean offset {dist['offset']:+.4f}")
    gap = np.linalg.norm(results["uniform"] - results["cotan"], axis=1).max()
    print(f"  uniform vs cotangent patches: vertices differ by up to {gap:.4f} ({gap / h:.1f} h)")

    draw_stages(V, F, free, V_hole, F_hole, loop, V_init, F_new, results["cotan"])
    draw_stages(V, F, free, V_hole, F_hole, loop, V_init, F_new, results["cotan"], gray_final=True)
    draw_weights(V, F, free, V_hole, F_hole, V_init, F_new, results, info)


def draw_stages(V, F, free, V_hole, F_hole, loop, V_init, F_new, W_cotan, gray_final=False):
    """Figure A: original with the region, the hole, the initial patch, the cotangent thin plate.
    Top row: the whole bunny from the picture's angle. Bottom row: close-ups, one frame.
    With ``gray_final``, the faired bunny is drawn all gray, as in the book's Fig. 4.7 (right),
    the close-ups have no titles, and the figure is saved to a separate file."""
    region = free[F].any(axis=1)
    original_colors = np.where(region[:, None], to_rgb(LILAC), to_rgb(GRAY))
    center = V[free].mean(axis=0)
    radius = 0.052
    frame = V[np.unique(crop(V, F, center, radius))]
    n_hole = len(F_hole)
    patch_edges = np.arange(len(F_new)) >= n_hole

    stages = [
        (V, F, original_colors, None, None, "1. Original bunny,\nselected region highlighted"),
        (V_hole, F_hole, np.tile(to_rgb(GRAY), (n_hole, 1)), None, loop, "2. The actual hole\n(faces and vertices removed)"),
        (V_init, F_new, colors(len(F_new), n_hole), patch_edges, None, "3. Initial patch: new triangulation,\ninterior on the fitted plane"),
        (W_cotan, F_new, colors(len(F_new), len(F_new) if gray_final else n_hole), None, None,
         "4. Faired patch, cotangent weights\n(thin plate, k = 2)"),
    ]
    fig = plt.figure(figsize=(16, 8.4), layout="constrained")
    for col, (X, T, c, e, l, title) in enumerate(stages):
        render(fig.add_subplot(2, 4, col + 1, projection="3d"), X, T, c, VIEW, X, title,
               TEAL if col == 3 else INK, loop=l)
        close = (np.linalg.norm(X[T] - center, axis=2) < radius).any(axis=1)
        render(fig.add_subplot(2, 4, col + 5, projection="3d"), X, T[close], c[close], CLOSE_VIEW, frame,
               "" if gray_final else "close-up",
               TEAL if col == 3 else INK, edges=None if e is None else e[close], loop=l)
    fig.suptitle("Hole filling: construct a patch, then fair it", fontsize=17, fontweight="bold")
    fig.text(0.5, -0.01, "Patch construction supplies the triangulation; fairing computes a smooth shape for its interior. "
             + ("\nGray: surviving bunny (fixed). Blue: replacement patch. Green: hole boundary. "
                "Panel 4 is drawn all gray, as in the book's Fig. 4.7." if gray_final else
                "Gray: surviving bunny (fixed). Blue: replacement patch. Green: hole boundary."),
             ha="center", fontsize=11, color=INK)
    path = IMG / ("14-bunny-hole-filling-stages-gray.png" if gray_final else "14-bunny-hole-filling-stages.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def draw_weights(V, F, free, V_hole, F_hole, V_init, F_new, results, info):
    """Figure B: the shared initial patch, the uniform and the cotangent thin plates (k = 2).
    Row 1: close-ups at an angle (shape). Row 2: face on, with the triangles (sampling).
    Row 3: a cross-section through the patch, with the original surface for reference."""
    n_hole = len(F_hole)
    patch = F_new[n_hole:]
    new = np.unique(patch[patch >= len(V_hole)])
    center = V_init[new].mean(axis=0)
    radius = 0.05
    face_on = back["above"](results["cotan"], patch, np.isin(np.arange(len(V_init)), new))
    frame = V_init[np.unique(crop(V_init, F_new, center, radius))]
    panels = [(V_init, "Shared initial patch", INK),
              (results["uniform"], "Uniform weights, k = 2", ORANGE),
              (results["cotan"], "Cotangent weights, k = 2", TEAL)]
    fig = plt.figure(figsize=(13.5, 13.4), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.72])
    for col, (X, title, ink) in enumerate(panels):
        close = (np.linalg.norm(X[F_new] - center, axis=2) < radius).any(axis=1)
        T = F_new[close]
        c = colors(len(F_new), n_hole)[close]
        edges = (np.arange(len(F_new)) >= n_hole)[close]
        render(fig.add_subplot(grid[0, col], projection="3d"), X, T, c, CLOSE_VIEW, frame, title, ink)
        render(fig.add_subplot(grid[1, col], projection="3d"), X, T, c, face_on, frame, "face on, triangles shown", ink,
               edges=edges)

    # the cross-section: the plane through the loop's centroid, spanned by the fitted normal
    # and the loop's longest direction; only faces near the hole, so the far side is not cut
    ax = fig.add_subplot(grid[2, :])
    cut = lambda X, T, **kw: LineCollection(section(X, crop(X, T, info["center"], 0.06), info["center"],
                                                    info["e1"], info["normal"], info["e2"]), **kw)
    removed = F[free[F].any(axis=1)]
    ax.add_collection(cut(V, removed, color=LILAC, linewidth=5, label="original surface (removed)"))
    ax.add_collection(cut(V_hole, F_hole, color="#8a8985", linewidth=1.6, label="surviving surface (fixed)"))
    for X, label, ink in ((V_init, "initial patch (fitted plane)", INK), (results["uniform"], "uniform, k = 2", ORANGE),
                          (results["cotan"], "cotangent, k = 2", TEAL)):
        ax.add_collection(cut(X, patch, color=ink, linewidth=1.8, label=label))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("along the hole (fitted plane)")
    ax.set_ylabel("height above\nthe fitted plane")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", ncols=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.45))
    ax.set_title("Cross-section through the patch (true scale)", fontsize=11, color=INK)

    fig.suptitle("Same initial patch, different weights", fontsize=17, fontweight="bold")
    fig.text(0.5, -0.01, "Both solves use the same initialized mesh, connectivity and free vertices; "
             "only the frozen Laplacian weights differ.", ha="center", fontsize=11, color=INK)
    path = IMG / "14-bunny-hole-filling-weights.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()
