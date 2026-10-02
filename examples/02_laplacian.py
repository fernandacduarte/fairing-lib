"""Phase 2 example: discrete Laplace-Beltrami operators (Ch. 3).

Run from the repository root:

    python examples/02_laplacian.py

Saves docs/img/05-uniform-Lx.png, docs/img/06-uniform-vs-cotan.png and
docs/img/06-sphere-H.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from fairing import mesh, viz
from fairing.laplacian import cotan_laplacian, mean_curvature, uniform_laplacian

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


def figure_uniform_vs_cotan():
    """Issue #6: on the same flat irregular grid, the cotangent Lx vanishes (linear precision)."""
    V, F = mesh.irregular_grid(13, 13, jitter=0.15, seed=0)
    top = (90, -90)
    fig = viz.compare(
        [(V, F), (V, F)],
        ["uniform: ‖Lx‖", "cotangent: ‖Lx‖"],
        [interior_norms(V, F, uniform_laplacian(V, F)[0]), interior_norms(V, F, cotan_laplacian(V, F)[0])],
        views=[top, top], panel_size=(4.4, 4.6))
    fig.savefig(IMG / "06-uniform-vs-cotan.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "06-uniform-vs-cotan.png")


def figure_sphere_H():
    """Issue #6: H = ||Lx||/2 on spheres of radius R. Converges away from the poles, not at them."""
    R = 2.0
    fig = plt.figure(figsize=(10, 4.4))

    # Left: one sphere colored by H*R (exact value 1). The pole cap is visibly low.
    V, F = mesh.uv_sphere(12, 24, radius=R)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    viz.plot_mesh(V, F, mean_curvature(V, F) * R, ax, title="H·R on a UV sphere (exact: 1)", elev=35)

    # Right: relative error |H*R - 1| as the sphere is refined.
    n_lats = [6, 10, 16, 24, 40, 64, 100]
    band_err, pole_err = [], []
    for n in n_lats:
        V, F = mesh.uv_sphere(n, 2 * n, radius=R)
        err = np.abs(mean_curvature(V, F) * R - 1)
        band_err.append(err[np.abs(V[:, 2]) < 0.7 * R].max())
        pole_err.append(err[[0, -1]].max())
    h = np.pi / np.array(n_lats)                                # latitude spacing (radians)
    ax = fig.add_subplot(1, 2, 2)
    ax.loglog(h, pole_err, "o-", color="#eb6834", lw=2, ms=7, label="at the poles")
    ax.loglog(h, band_err, "o-", color="#2a78d6", lw=2, ms=7, label="away from the poles (max)")
    ax.loglog(h, band_err[0] * (h / h[0]) ** 2, "--", color="#8a8984", lw=1.2, label="slope 2 (∝ h²)")
    ax.text(h[0], pole_err[0] * 1.25, "stuck near 0.25", color="#3d3c39", ha="right", fontsize=9)
    ax.set_xlabel("latitude spacing h (radians)")
    ax.set_ylabel("relative error |H·R − 1|")
    ax.set_title("refining the sphere")
    ax.xaxis.set_minor_formatter(NullFormatter())             # avoid overlapping minor labels
    ax.grid(True, which="major", color="#e6e5e1", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(IMG / "06-sphere-H.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "06-sphere-H.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    figure_uniform()
    figure_uniform_vs_cotan()
    figure_sphere_H()
