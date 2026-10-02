"""Phase 2 example: discrete Laplace-Beltrami operators (Ch. 3).

Run from the repository root:

    python examples/02_laplacian.py

Saves docs/img/05-uniform-Lx.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from fairing import mesh, viz
from fairing.laplacian import uniform_laplacian

IMG = Path(__file__).resolve().parents[1] / "docs" / "img"


def interior_norms(V, F, L):
    """||Lx|| per vertex, with NaN on the boundary (where Lx is one-sided and large)."""
    norms = np.linalg.norm(L @ V, axis=1)
    norms[mesh.boundary_vertices(F)] = np.nan
    return norms


def figure_uniform():
    """Issue #5: the uniform Laplacian is zero on the regular grid, not on the irregular one."""
    regular = mesh.grid(13, 13)
    irregular = mesh.irregular_grid(13, 13, jitter=0.15, seed=0)
    L_reg = uniform_laplacian(*regular)[0]
    L_irr = uniform_laplacian(*irregular)[0]

    top = (90, -90)
    fig = viz.compare(
        [regular, irregular, irregular],
        ["regular grid: ‖Lx‖", "irregular grid: ‖Lx‖", "irregular grid: Lx vectors (×3)"],
        [interior_norms(*regular, L_reg), interior_norms(*irregular, L_irr), None],
        views=[top, top, top], panel_size=(4.2, 4.6),
        color="#f0efec", shade=False)                       # flat light gray under the arrows

    # Third panel: arrows from each interior vertex toward the centroid of its neighbors.
    ax = [a for a in fig.axes if a.name == "3d"][2]
    V, F = irregular
    inner = np.setdiff1d(np.arange(len(V)), mesh.boundary_vertices(F))
    LV = 3 * (L_irr @ V)[inner]
    ax.quiver(V[inner, 0], V[inner, 1], V[inner, 2], LV[:, 0], LV[:, 1], LV[:, 2],
              color="#0d366b", linewidth=1.6, arrow_length_ratio=0.4)

    fig.savefig(IMG / "05-uniform-Lx.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "05-uniform-Lx.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    figure_uniform()
