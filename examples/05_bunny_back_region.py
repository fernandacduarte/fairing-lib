"""Phase 5 example, complementary analysis: a hole on the upper back of the bunny.

The region was outlined by hand on a picture of the bunny (head left, as in the book's
Fig. 4.7): the top of the haunch, from the middle of the back to the rear silhouette. It is
higher and further back than the hole of examples/05_bunny_book_region.py, and it wraps over
the back, so part of it is hidden from the picture's angle.

How the outline became vertices: the picture's silhouette was matched to the mesh's (camera
VIEW below), the vertices visible inside the outline were collected, and the ball (TARGET,
RADIUS) that covers them best was kept. The region is then built by `book_region` of
examples/05_bunny_book_region.py, on the same full-resolution bunny (bun_zipper.ply; see README).

Run from the repository root:

    python examples/05_bunny_back_region.py

Saves docs/img/14-bunny-back-region.png and prints the numbers quoted in the phase-5 note.
Data: Stanford Computer Graphics Laboratory (Stanford 3D Scanning Repository).
"""

import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, to_rgb
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from fairing import mesh, viz
from fairing.fairing import solve_fair
from fairing.laplacian import mean_curvature

HERE = Path(__file__).resolve().parent
book = runpy.run_path(str(HERE / "05_bunny_book_region.py"), run_name="book_helpers")
helpers = book["helpers"]
IMG = helpers["IMG"]
K = 2
TARGET = np.array([0.027, -0.011, 0.123])    # top of the haunch (z-up coordinates)
RADIUS = 0.04                                # covers 89% of the outlined vertices seen from VIEW
VIEW = (16, -62)                             # the picture's camera: head left, back on the right
GRAY, BLUE, GREEN = "#d9d8d4", "#5b62c9", "#3aa655"
LIGHT = LightSource(azdeg=225, altdeg=19.4712).direction    # matplotlib's default light


def draw(ax, V, F, free, title, view, hole=False):
    """Lit surface with the free region in blue; with `hole`, the region removed, border in green.

    All faces go into one collection, so matplotlib sorts them by depth together. The other
    bunny figures draw fixed and free faces as two collections, painted one over the other:
    fine for a region facing the camera, but here the fixed faces would hide the region where
    it wraps over the back."""
    touches_free = free[F].any(axis=1)
    color = np.where(touches_free[:, None], to_rgb(BLUE), to_rgb(GRAY))
    if hole:
        border = mesh.ring_distance(F, np.flatnonzero(free), max_k=1, n=len(V)) == 1
        color = np.where(border[F].any(axis=1)[:, None], to_rgb(GREEN), to_rgb(GRAY))[~touches_free]
        F = F[~touches_free]
    normal = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    brightness = 0.3 + 0.35 * (1 + normal @ LIGHT)          # the shading of plot_trisurf
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=color * brightness[:, None],
                                         linewidths=0, antialiased=False))
    viz._set_equal_aspect(ax, V)
    ax.view_init(*view)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def above(V, F, free):
    """Camera (elev, azim) along the region's mean normal: the region seen face on."""
    faces = F[free[F].all(axis=1)]
    n = np.cross(V[faces[:, 1]] - V[faces[:, 0]], V[faces[:, 2]] - V[faces[:, 0]]).sum(axis=0)
    n /= np.linalg.norm(n)
    return np.degrees(np.arcsin(n[2])), np.degrees(np.arctan2(n[1], n[0]))


def vertex_normals(V, F):
    """Unit vertex normals: the sum of the area vectors of the triangles around each vertex."""
    n = np.zeros_like(V)
    area_vectors = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for j in range(3):
        np.add.at(n, F[:, j], area_vectors)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def main():
    V, F = helpers["load_bunny"](book["FULL"])
    free, euler, clearance = book["book_region"](V, F, RADIUS, TARGET)
    assert euler == 1 and clearance >= K + 1
    damaged, edge = helpers["damage"](V, F, free)
    results = {(d, lap): solve_fair(D, F, free, K, lap) for d, D in damaged.items() for lap in ("uniform", "cotan")}

    H = lambda W: mean_curvature(W, F)[free].mean()
    normal = vertex_normals(V, F)
    offset = lambda W: ((W - V) * normal).sum(axis=1)[free].mean()      # < 0: below the original
    face_on = above(V, F, free)
    print(f"full-resolution bunny: {len(V)} vertices, {len(F)} triangles")
    print(f"free region: {free.sum()} vertices (a disk), at least {clearance} rings from boundary edges; "
          f"mean edge {edge:.4f}; seen face on from (elev, azim) = ({face_on[0]:.0f}, {face_on[1]:.0f})")
    print(f"original: mean |H| on the region {H(V):.1f}")
    for (d, lap), W in results.items():
        print(f"  {d:7s} {lap:7s}: max distance to the original {np.abs(W - V)[free].max():.4f}, "
              f"mean offset along the normal {offset(W):+.4f}, mean |H| {H(W):.1f}")
    for lap in ("uniform", "cotan"):
        gap = np.abs(results["noise", lap] - results["flatten", lap]).max()
        print(f"  {lap}: noisy-start vs flattened-start refills differ by {gap:.2e}")

    # one group of three panels per row: first the original bunny, apart from the refills,
    # then one group per start, each laid out as damaged input | uniform | cotangent
    fig = plt.figure(figsize=(11, 11.25), layout="constrained")
    original, *starts = fig.subfigures(3, 1)
    panels = lambda group: [group.add_subplot(1, 3, i, projection="3d") for i in (1, 2, 3)]
    heading = lambda group, text: group.suptitle(text, x=0.02, ha="left", fontsize=11, fontweight="bold")

    heading(original, "The region on the original bunny")
    hole_ax, region_ax, face_on_ax = panels(original)
    draw(hole_ax, V, F, free, "the hole, from the picture's angle\n(region removed, border in green)",
         VIEW, hole=True)
    draw(region_ax, V, F, free, "free region in blue,\nfrom the picture's angle", VIEW)
    draw(face_on_ax, V, F, free, "free region, face on\n(along the region's mean normal)", face_on)
    original.add_artist(Line2D([0.02, 0.98], [0.0, 0.0], color="#8a8985", linewidth=1))

    for group, (d, text) in zip(starts, (("noise", "From the noisy start"), ("flatten", "From the flattened start"))):
        heading(group, f"{text}: damaged input, then thin-plate refills (k = 2)")
        input_ax, *refill_axes = panels(group)
        draw(input_ax, damaged[d], F, free, f"damaged input ({'noise' if d == 'noise' else 'flattened'})", VIEW)
        for ax, lap in zip(refill_axes, ("uniform", "cotan")):
            draw(ax, results[d, lap], F, free, f"{lap} weights", VIEW)
    fig.suptitle("A hole on the bunny's upper back, refilled with a thin plate (k = 2)\n"
                 "Full-resolution Stanford bunny (Stanford Computer Graphics Laboratory)", fontsize=12)
    fig.savefig(IMG / "14-bunny-back-region.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "14-bunny-back-region.png")


if __name__ == "__main__":
    main()
