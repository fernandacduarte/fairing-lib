"""Render the refined blue fairing comparison for the README.

Run from the repository root, independently of examples/04_fairing.py:

    python examples/04_readme_elbow.py

Uses pipe radius 1, bend radius 1.25, 192 vertices per ring and longitudinal
spacing 0.025 (33,792 vertices and 67,200 triangles), solved for k = 1, 2, 3.
Saves docs/img/11-elbow-k123-readme.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fairing import mesh, viz
from fairing.fairing import solve_fair


IMG = Path(__file__).resolve().parents[1] / "docs" / "img"


def figure_readme_elbow():
    """Compare the three fairing orders on the same refined elbow mesh."""
    V, F, free = mesh.pipe_elbow(n_theta=192, radius=1.0, bend_radius=1.25,
                                pipe_length=1.2, spacing=0.025)
    free_face = free[F].any(axis=1)
    titles = ("k = 1: membrane (C⁰)", "k = 2: thin plate (C¹)",
              "k = 3: minimum variation (C²)")
    fig = plt.figure(figsize=(13.8, 4.8))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.06, top=0.84, wspace=0.02)
    for k, title in enumerate(titles, start=1):
        W = solve_fair(V, F, free, k)
        ax = fig.add_subplot(1, 3, k, projection="3d")
        for faces, color in ((F[~free_face], "#d9d8d4"), (F[free_face], "#5b62c9")):
            ax.plot_trisurf(W[:, 0], W[:, 1], W[:, 2], triangles=faces, color=color,
                            shade=True, linewidth=0, antialiased=False)
        viz._set_equal_aspect(ax, V)
        ax.view_init(10, -100)
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, pad=4)
    fig.suptitle("Bend radius 1.25 · pipe radius 1 · refined mesh", fontsize=16, y=0.98)
    fig.text(0.5, 0.015,
             f"{len(V):,} vertices · {len(F):,} triangles · gray: fixed pipes · blue: faired region",
             ha="center", fontsize=10, color="#555555")
    target = IMG / "11-elbow-k123-readme.png"
    fig.savefig(target, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", target)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    figure_readme_elbow()
