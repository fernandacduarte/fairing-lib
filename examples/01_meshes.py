"""Phase 1 example: the synthetic meshes used throughout the project.

Run from the repository root:

    python examples/01_meshes.py

Saves docs/img/03-meshes.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

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


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    figure_meshes()
