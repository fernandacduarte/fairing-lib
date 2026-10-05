"""Phase 5 example: refilling a damaged region of the Stanford bunny with a thin plate (k = 2).

Data: the Stanford Bunny, Stanford 3D Scanning Repository (Stanford Computer Graphics
Laboratory), https://graphics.stanford.edu/data/3Dscanrep/ . It is not part of this repo;
download it first (see README, "The Stanford bunny"). This script uses the decimated
reconstruction bun_zipper_res2.ply (8146 used vertices).

Run from the repository root:

    python examples/05_bunny.py

Saves docs/img/14-bunny-before-after.png and prints the numbers quoted in the phase-5 note.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fairing import mesh, viz
from fairing.fairing import solve_fair
from fairing.laplacian import mean_curvature

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
BUNNY = ROOT / "data" / "bunny" / "reconstruction" / "bun_zipper_res2.ply"
K, RINGS = 2, 6                                  # thin plate; free region = 6-ring around a seed


def load_bunny():
    """The bunny as plain V, F, rotated from the file's y-up to z-up (for the 3D plots)."""
    if not BUNNY.exists():
        sys.exit(f"{BUNNY} not found: download the Stanford bunny first (see README, 'The Stanford bunny').")
    V, F = mesh.load_mesh(BUNNY)
    return V[:, [0, 2, 1]] * [1, -1, 1], F


def choose_region(V, F):
    """A 6-ring free region as far as possible from the mesh's problem edges.

    The decimated bunny has non-manifold edges (in more than 2 triangles) and boundary edges.
    The free region's equations reach k rings beyond it, plus the triangles around them, so
    the region must stay at least k + 1 rings away from those edges.
    """
    E, counts = mesh.edge_face_counts(F)
    problem = np.unique(E[counts != 2])
    dist = mesh.ring_distance(F, problem, n=len(V))
    seed = int(np.argmax(dist))
    free = np.zeros(len(V), dtype=bool)
    free[mesh.k_ring(F, [seed], RINGS, n=len(V))] = True
    assert dist[free].min() >= K + 1
    return free, len(problem), int(dist[free].min())


def damage(V, F, free):
    """The two damaged inputs: noise (sigma = half an edge) and flattening onto the border's plane."""
    E, _ = mesh.edge_face_counts(F)
    edge = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).mean()
    noisy = np.where(free[:, None], mesh.add_noise(V, 0.5 * edge, seed=0), V)
    return {"noise": noisy, "flatten": mesh.flatten_onto_border_plane(V, F, free)}, edge


def crop_around(V, F, free, radius):
    """Faces within `radius` of the free region's center, re-indexed (for close-ups)."""
    center = V[free].mean(axis=0)
    keep = (np.linalg.norm(V[F].mean(axis=1) - center, axis=1) < radius)
    used, Fc = np.unique(F[keep], return_inverse=True)
    return used, Fc.reshape(-1, 3)


def draw(ax, V, F, free, title, view):
    """Lit surface: faces with any free vertex in blue, the rest gray (as in #11)."""
    free_face = free[F].any(axis=1)
    for faces, color in ((F[~free_face], "#d9d8d4"), (F[free_face], "#5b62c9")):
        ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=faces, color=color, shade=True,
                        linewidth=0, antialiased=False)
    viz._set_equal_aspect(ax, V)
    ax.view_init(*view)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def main():
    V, F = load_bunny()
    free, n_problem, clearance = choose_region(V, F)
    damaged, edge = damage(V, F, free)
    results = {(d, lap): solve_fair(D, F, free, K, lap) for d, D in damaged.items() for lap in ("uniform", "cotan")}

    H = lambda W: mean_curvature(W, F)[free].mean()
    print(f"bunny: {len(V)} vertices, {len(F)} triangles; {n_problem} vertices on problem edges")
    print(f"free region: {free.sum()} vertices ({RINGS} rings), at least {clearance} rings from problem edges; "
          f"mean edge {edge:.4f}")
    print(f"original: mean |H| on the region {H(V):.1f}")
    for (d, lap), W in results.items():
        print(f"  {d:7s} {lap:7s}: max distance to the original {np.abs(W - V)[free].max():.4f}, mean |H| {H(W):.1f}")
    for lap in ("uniform", "cotan"):
        gap = np.abs(results["noise", lap] - results["flatten", lap]).max()
        print(f"  {lap}: noisy-start vs flattened-start refills differ by {gap:.2e}")

    view = (15, 120)                                              # the region faces the camera
    used, Fc = crop_around(V, F, free, radius=0.035)
    close = lambda W: W[used]
    border = mesh.ring_distance(F, np.flatnonzero(free), max_k=1, n=len(V))[used] == 1

    def curvature_panel(position, W, title):
        """Close-up colored by mean curvature (shared scale), the region's border as dots."""
        ax = fig.add_subplot(2, 4, position, projection="3d")
        Hc = mean_curvature(W, F)[used]
        _, surf = viz.plot_mesh(close(W), Fc, Hc, ax, vmin=0, vmax=120, edges=False, colorbar=False,
                                elev=view[0], azim=view[1], title=title)
        Wc = close(W)[border]
        ax.computed_zorder = False                                # draw the dots on top of the surface
        ax.scatter(*Wc.T, s=5, color="#eb6834", depthshade=False, zorder=10)
        return surf

    fig = plt.figure(figsize=(14, 7.8))
    draw(fig.add_subplot(2, 4, 1, projection="3d"), V, F, free, "the bunny; free region highlighted", view)
    curvature_panel(5, V, f"original (close-up)\nmean |H| on the region {H(V):.0f}")
    for row, d in enumerate(("noise", "flatten")):
        label = "noisy start" if d == "noise" else "flattened start"
        curvature_panel(4 * row + 2, damaged[d], f"damaged: {label}")
        for col, lap in ((3, "uniform"), (4, "cotan")):
            W = results[d, lap]
            surf = curvature_panel(4 * row + col, W, f"thin plate (k = 2), {lap} weights\nfrom the {label}; mean |H| {H(W):.0f}")
    cax = fig.add_axes([0.93, 0.15, 0.01, 0.3])
    fig.colorbar(surf, cax=cax, label="mean curvature H\n(≥ 120 saturates)")
    fig.text(0.5, 0.01, "Close-ups colored by mean curvature on one scale; orange dots: the first fixed ring (the region's border).",
             ha="center", fontsize=9, color="#3d3c39")
    fig.suptitle("Refilling a damaged region of the Stanford bunny with a thin plate (k = 2). "
                 "Data: Stanford Computer Graphics Laboratory.", y=0.99)
    fig.tight_layout(rect=(0, 0.03, 0.92, 1))
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / "14-bunny-before-after.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "14-bunny-before-after.png")


if __name__ == "__main__":
    main()
