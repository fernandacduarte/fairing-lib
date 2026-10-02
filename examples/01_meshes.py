"""Phase 1 example: the synthetic meshes used throughout the project.

Run from the repository root:

    python examples/01_meshes.py

Saves docs/img/03-meshes.png and docs/img/04-rings.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


# Rings 0..3, darkest (ring 0) to lightest: steps of one blue ramp, since the rings are ordered.
RING_COLORS = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef"]


def figure_rings():
    """Issue #4: rings grown from the boundary, and from one interior vertex."""
    n = 15
    V, F = mesh.grid(n, n)
    center = n // 2 + n * (n // 2)
    panels = [
        (mesh.boundary_vertices(F), ["boundary (ring 0)", "ring 1", "ring 2", "ring 3"],
         "rings grown from the boundary"),
        ([center], ["seed (ring 0)", "ring 1: 6", "ring 2: 12", "ring 3: 18"],
         "rings grown from one vertex: 1 + 3k(k+1)"),
    ]
    fig = plt.figure(figsize=(10, 5.2))
    for k, (seeds, labels, title) in enumerate(panels):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        viz.plot_mesh(V, F, ax=ax, title=title, elev=90, azim=-90, color="#f0efec", shade=False)
        dist = mesh.ring_distance(F, seeds, max_k=3)
        for d, (color, label) in enumerate(zip(RING_COLORS, labels)):
            P = V[dist == d]
            single_seed = d == 0 and len(P) == 1        # a lone seed gets a bigger star
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=160 if single_seed else 40,
                       marker="*" if single_seed else "o", color=color, label=label,
                       depthshade=False, edgecolors="white", linewidths=0.8)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
    fig.savefig(IMG / "04-rings.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "04-rings.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    figure_meshes()
    figure_rings()
