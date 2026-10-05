"""Phase 5 example, complementary analysis: a hole like the one in the book's Fig. 4.7.

The book's figure removes a large region on the bunny's haunch and refills it with a smooth
patch. This script does the same on the full-resolution Stanford bunny (bun_zipper.ply,
34834 used vertices, from the same download as examples/05_bunny.py; see README).
Why full resolution: on the decimated bunny the same region contains 7 non-manifold edges
(edges in more than 2 triangles) and is not a disk; the full mesh has no non-manifold edges,
only boundary edges around the holes in its base, far from the region.

Run from the repository root:

    python examples/05_bunny_book_region.py

Saves docs/img/14-bunny-book-region.png and prints the numbers quoted in the phase-5 note.
Data: Stanford Computer Graphics Laboratory (Stanford 3D Scanning Repository).
"""

import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components

from fairing import mesh, viz
from fairing.fairing import solve_fair
from fairing.laplacian import mean_curvature

HERE = Path(__file__).resolve().parent
helpers = runpy.run_path(str(HERE / "05_bunny.py"), run_name="bunny_helpers")
FULL = helpers["ROOT"] / "data" / "bunny" / "reconstruction" / "bun_zipper.ply"
IMG = helpers["IMG"]
K = 2
TARGET = np.array([0.03, -0.07, 0.10])        # a point just outside the haunch (z-up coordinates)
RADIUS = 0.035                                # about the size of the hole in Fig. 4.7
VIEW = (15, -60)                              # matched by eye to Fig. 4.7: head left, haunch right


def book_region(V, F, radius=RADIUS):
    """The free region: vertices within `radius` of the haunch vertex closest to TARGET,
    keeping only the connected piece that contains it.

    Also returns the region's Euler characteristic (1 for a disk) and its distance, in rings,
    to the mesh's boundary edges (the region's equations reach k + 1 rings beyond it)."""
    seed = int(np.argmin(np.linalg.norm(V - TARGET, axis=1)))
    near = np.flatnonzero(np.linalg.norm(V - V[seed], axis=1) < radius)
    _, label = connected_components(mesh.adjacency(F, len(V))[near][:, near], directed=False)
    free = np.zeros(len(V), dtype=bool)
    free[near[label == label[np.searchsorted(near, seed)]]] = True
    patch = F[free[F].all(axis=1)]
    euler = free.sum() - len(mesh.edges(patch)) + len(patch)        # 1 for a disk
    E, counts = mesh.edge_face_counts(F)
    clearance = int(mesh.ring_distance(F, np.unique(E[counts != 2]), n=len(V))[free].min())
    return free, int(euler), clearance


def draw_hole(ax, V, F, free):
    """As in the book's Fig. 4.7 (left): the region's triangles removed, its border in green."""
    border = mesh.ring_distance(F, np.flatnonzero(free), max_k=1, n=len(V)) == 1
    ax.computed_zorder = False
    ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F[~free[F].any(axis=1)], color="#d9d8d4",
                    shade=True, linewidth=0, antialiased=False)
    ax.scatter(*V[border].T, s=1.5, color="#3aa655", depthshade=False, zorder=10)
    viz._set_equal_aspect(ax, V)
    ax.view_init(*VIEW)
    ax.set_axis_off()
    ax.set_title("the hole, as in Fig. 4.7\n(region removed, border in green)", fontsize=9)


def main():
    V, F = helpers["load_bunny"](FULL)
    free, euler, clearance = book_region(V, F)
    assert euler == 1 and clearance >= K + 1
    damaged, edge = helpers["damage"](V, F, free)
    results = {(d, lap): solve_fair(D, F, free, K, lap) for d, D in damaged.items() for lap in ("uniform", "cotan")}

    H = lambda W: mean_curvature(W, F)[free].mean()
    print(f"full-resolution bunny: {len(V)} vertices, {len(F)} triangles")
    print(f"free region: {free.sum()} vertices (a disk), at least {clearance} rings from boundary edges; "
          f"mean edge {edge:.4f}")
    print(f"original: mean |H| on the region {H(V):.1f}")
    for (d, lap), W in results.items():
        print(f"  {d:7s} {lap:7s}: max distance to the original {np.abs(W - V)[free].max():.4f}, mean |H| {H(W):.1f}")
    for lap in ("uniform", "cotan"):
        gap = np.abs(results["noise", lap] - results["flatten", lap]).max()
        print(f"  {lap}: noisy-start vs flattened-start refills differ by {gap:.2e}")

    draw = helpers["draw"]
    fig = plt.figure(figsize=(14, 7.4))
    draw_hole(fig.add_subplot(2, 4, 1, projection="3d"), V, F, free)
    draw(fig.add_subplot(2, 4, 5, projection="3d"), V, F, free, "original; free region in blue", VIEW)
    for row, d in enumerate(("noise", "flatten")):
        label = "noisy start" if d == "noise" else "flattened start"
        draw(fig.add_subplot(2, 4, 4 * row + 2, projection="3d"), damaged[d], F, free, f"damaged: {label}", VIEW)
        for col, lap in ((3, "uniform"), (4, "cotan")):
            W = results[d, lap]
            draw(fig.add_subplot(2, 4, 4 * row + col, projection="3d"), W, F, free,
                 f"thin plate (k = 2), {lap} weights\nfrom the {label}; mean |H| {H(W):.0f}", VIEW)
    fig.suptitle("A hole like the book's Fig. 4.7, refilled with a thin plate (k = 2). "
                 "Full-resolution Stanford bunny (Stanford Computer Graphics Laboratory).", y=0.99)
    fig.tight_layout()
    fig.savefig(IMG / "14-bunny-book-region.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "14-bunny-book-region.png")


if __name__ == "__main__":
    main()
