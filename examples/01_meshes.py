"""Phase 1 example: the synthetic meshes used throughout the project.

Run from the repository root:

    python examples/01_meshes.py

Saves docs/img/03-meshes.png and docs/img/04-rings.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fairing import mesh, viz

IMG = Path(__file__).resolve().parents[1] / "docs" / "img"


def figure_meshes():
    """Issue #3: the four generators side by side."""
    meshes = [
        mesh.grid(10, 10),
        mesh.irregular_grid(10, 10, jitter=0.15, seed=0),
        mesh.uv_sphere(12, 20),
        mesh.tube(20, 10, radius=0.5, height=2.0),
    ]
    titles = ["grid", "irregular_grid", "uv_sphere", "tube"]
    views = [(90, -90), (90, -90), None, None]   # look straight down on the planar grids
    fig = viz.compare(meshes, titles, panel_size=(3.6, 3.6), views=views)
    fig.savefig(IMG / "03-meshes.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "03-meshes.png")


# Ordered rings: steps of one blue ramp, darkest (ring 0) to lightest (ring 3).
RING_COLORS = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef"]
FREE_COLOR = "#eb6834"   # a different hue: free vertices are a different kind, not "ring 0"


def _scatter(ax, P, color, label, size=40, marker="o"):
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=size, marker=marker, color=color, label=label,
               depthshade=False, edgecolors="white", linewidths=0.8)


def figure_rings():
    """Issue #4: the fixed rings around a free region, and the k-rings of one vertex."""
    n = 15
    V, F = mesh.grid(n, n)
    fig = plt.figure(figsize=(10, 5.2))

    # Left: the fairing setup of #9-#11. A disk of free vertices; the fixed
    # rings 1, 2, 3 are grown outward from it (k rings are needed for L^k x = 0).
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    viz.plot_mesh(V, F, ax=ax, title="free region + fixed rings grown outward",
                  elev=90, azim=-90, color="#f0efec", shade=False)
    free = np.flatnonzero(np.linalg.norm(V[:, :2], axis=1) < 0.2)
    dist = mesh.ring_distance(F, free, max_k=3)
    _scatter(ax, V[dist == 0], FREE_COLOR, "free (will move)")
    for d, label in zip((1, 2, 3), ("fixed ring 1 (k ≥ 1)", "fixed ring 2 (k ≥ 2)", "fixed ring 3 (k = 3)")):
        _scatter(ax, V[dist == d], RING_COLORS[d], label)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)

    # Right: the k-rings of one vertex. Ring k adds 6k vertices on this lattice.
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    viz.plot_mesh(V, F, ax=ax, title="rings grown from one vertex: 1 + 3k(k+1)",
                  elev=90, azim=-90, color="#f0efec", shade=False)
    center = n // 2 + n * (n // 2)
    dist = mesh.ring_distance(F, [center], max_k=3)
    _scatter(ax, V[dist == 0], RING_COLORS[0], "seed (ring 0)", size=160, marker="*")
    for d in (1, 2, 3):
        _scatter(ax, V[dist == d], RING_COLORS[d], f"ring {d}: {6 * d}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)

    fig.savefig(IMG / "04-rings.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "04-rings.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    figure_meshes()
    figure_rings()
