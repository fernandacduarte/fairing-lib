"""Phase 4 example: fairing, L^k x = 0 (Sec. 4.3, App. A).

Run from the repository root:

    python examples/04_fairing.py

Saves docs/img/09-sparsity.png, docs/img/10-membrane.png and
docs/img/10-two-membranes.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import NullFormatter

from fairing import mesh, viz
from fairing.fairing import fairing_matrix, solve_fair
from fairing.laplacian import mean_curvature

IMG = Path(__file__).resolve().parents[1] / "docs" / "img"


def figure_sparsity():
    """Issue #9: the sparsity pattern of A = (-1)^k M (D M)^(k-1) grows with k."""
    n = 12
    V, F = mesh.irregular_grid(n, n, seed=0)
    center = n // 2 + n * (n // 2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, k in zip(axes, (1, 2, 3)):
        A = fairing_matrix(V, F, k)
        A.eliminate_zeros()
        per_row = np.diff(A.indptr)[center]
        ax.spy(A, markersize=1.2, color="#1c5cab")
        # the row of the center vertex: its non-zeros are the vertices within k rings
        ax.axhline(center, color="#eb6834", lw=0.8, alpha=0.8)
        ax.set_title(f"k = {k}: {per_row} non-zeros in an interior row\n"
                     f"{A.nnz} non-zeros in total ({A.nnz / A.shape[0]:.1f} per row)", fontsize=10)
        ax.set_xlabel("column (vertex index)")
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("row (vertex index)")
    fig.suptitle(f"Fairing matrix A = (−1)ᵏ M (DM)ᵏ⁻¹ on a {n} × {n} grid ({n * n} vertices); "
                 "orange line = the center vertex's row", y=1.02)
    fig.tight_layout()
    fig.savefig(IMG / "09-sparsity.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "09-sparsity.png")


def disk_problem(n, heights, interior_noise, seed=0):
    """Irregular grid on [-0.5, 0.5]^2 whose disk of radius 0.3 is free.

    Every vertex gets z = heights(x, y); the free ones also get noise.
    """
    V, F = mesh.irregular_grid(n, n, jitter=0.15, seed=seed)
    r = np.linalg.norm(V[:, :2], axis=1)
    free = r < 0.3
    V[:, 2] = heights(V[:, 0], V[:, 1])
    V[free, 2] += np.random.default_rng(seed).normal(0, interior_noise, free.sum())
    return V, F, free, r


flat = lambda x, y: 0 * x
saddle = lambda x, y: x**2 - y**2                    # harmonic in the plane


def figure_membrane():
    """Issue #10: k = 1 with a planar boundary (case A) and a harmonic boundary (case B)."""
    meshes, scalars, titles = [], [], []
    for name, heights in (("A: planar boundary", flat), ("B: boundary z = x² − y²", saddle)):
        V, F, free, _ = disk_problem(31, heights, interior_noise=0.01)
        W = solve_fair(V, F, free, 1, "cotan")
        meshes += [(V, F), (W, F)]
        scalars += [V[:, 2], W[:, 2]]
        titles += [f"{name}\ninput: free disk is noisy", f"{name}\nmembrane (k = 1, cotangent)"]
    fig = viz.compare(meshes, titles, scalars, ncols=2, diverging=True, panel_size=(4.6, 4.2), elev=30)
    fig.suptitle("Membrane surfaces, colored by height z (gray = 0)", y=1.0)
    fig.savefig(IMG / "10-membrane.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "10-membrane.png")


def figure_two_membranes():
    """Issue #10: case B with uniform vs cotangent weights, as the grid is refined.

    Left: distance to the harmonic graph z = x^2 - y^2 (the answer of Eq. 4.8 over the flat
    (x, y) domain). Right: mean |H| (0 for the minimal surface of Eq. 4.7).
    """
    ns = [11, 21, 41, 81, 161]
    h = 1 / (np.array(ns) - 1)
    runs = [("uniform (noisy start)", "uniform", 0.02, "#86b6ef"),
            ("cotangent (smooth start)", "cotan", 0.0, "#1c5cab"),
            ("cotangent (noisy start, σ = 0.02)", "cotan", 0.02, "#eb6834")]
    fig, (ax_gap, ax_H) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for label, laplacian, noise, color in runs:
        gaps, Hs = [], []
        for n in ns:
            V, F, free, r = disk_problem(n, saddle, noise)
            W = solve_fair(V, F, free, 1, laplacian)
            x, y, z = W[free].T
            gaps.append(np.abs(z - saddle(x, y)).max())
            Hs.append(mean_curvature(W, F)[r < 0.25].mean())
        ax_gap.loglog(h, gaps, "o-", color=color, lw=2, ms=6, label=label)
        ax_H.loglog(h, Hs, "o-", color=color, lw=2, ms=6, label=label)
    H_saddle = [mean_curvature(*disk_problem(n, saddle, 0.0)[:2])[disk_problem(n, saddle, 0.0)[3] < 0.25].mean()
                for n in ns]
    ax_H.loglog(h, H_saddle, "--", color="#8a8984", lw=1.5, label="the saddle x² − y² itself")

    ax_gap.set_title("distance to the harmonic graph x² − y²")
    ax_gap.set_ylabel("max |z − (x² − y²)| on the free disk")
    ax_H.set_title("mean |H| on the free disk (minimal surface: 0)")
    ax_H.set_ylabel("mean |H|")
    for ax in (ax_gap, ax_H):
        ax.set_xlabel("grid spacing h")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, color="#e6e5e1", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax_gap.legend(frameon=False, fontsize=9, loc="lower right")
    ax_H.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Case B: two different membranes. Uniform → harmonic graph (Eq. 4.8); "
                 "cotangent → minimal surface (Eq. 4.7)", y=1.02)
    fig.tight_layout()
    fig.savefig(IMG / "10-two-membranes.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "10-two-membranes.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    figure_sparsity()
    figure_membrane()
    figure_two_membranes()
