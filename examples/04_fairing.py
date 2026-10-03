"""Phase 4 example: fairing, L^k x = 0 (Sec. 4.3, App. A).

Run from the repository root:

    python examples/04_fairing.py

Saves docs/img/09-sparsity.png, docs/img/10-membrane.png and
docs/img/10-convergence.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import NullFormatter

from fairing import mesh, viz
from fairing.fairing import fairing_matrix, solve_fair

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
    """Irregular grid on [-0.5, 0.5]^2; the disk of radius 0.3 is free.

    Constrained vertices get z = heights(x, y); free ones start there plus noise.
    Also returns the flat reference (z = 0) used to build the cotangent weights.
    """
    V_ref, F = mesh.irregular_grid(n, n, jitter=0.15, seed=seed)
    free = np.linalg.norm(V_ref[:, :2], axis=1) < 0.3
    V = V_ref.copy()
    V[:, 2] = heights(V[:, 0], V[:, 1])
    V[free, 2] += np.random.default_rng(seed).normal(0, interior_noise, free.sum())
    return V, V_ref, F, free


flat = lambda x, y: 0 * x
saddle = lambda x, y: x**2 - y**2                    # harmonic in the plane


def figure_membrane():
    """Issue #10: k = 1 with a planar boundary (case A) and a harmonic boundary (case B)."""
    meshes, scalars, titles = [], [], []
    for name, heights, noise in (("A: planar boundary", flat, 0.05), ("B: boundary z = x² − y²", saddle, 0.05)):
        V, V_ref, F, free = disk_problem(31, heights, noise)
        W = solve_fair(V, F, free, 1, "cotan", V_ref=V_ref)
        meshes += [(V, F), (W, F)]
        scalars += [V[:, 2], W[:, 2]]
        titles += [f"{name}\ninput: free disk is noisy", f"{name}\nmembrane (k = 1, cotangent)"]
    fig = viz.compare(meshes, titles, scalars, ncols=2, diverging=True, panel_size=(4.6, 4.2), elev=30)
    fig.suptitle("Membrane surfaces, colored by height z (gray = 0)", y=1.0)
    fig.savefig(IMG / "10-membrane.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "10-membrane.png")


def figure_convergence():
    """Issue #10: error of the harmonic case B as the grid is refined."""
    ns = [11, 21, 41, 81, 161]
    runs = [("cotangent, L from the flat reference", "cotan", True, "#1c5cab"),
            ("uniform", "uniform", True, "#86b6ef"),
            ("cotangent, L from the noisy input", "cotan", False, "#eb6834")]
    h = 1 / (np.array(ns) - 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for label, laplacian, use_ref, color in runs:
        errors = []
        for n in ns:
            V, V_ref, F, free = disk_problem(n, saddle, 0.02)
            W = solve_fair(V, F, free, 1, laplacian, V_ref=V_ref if use_ref else None)
            x, y, z = W[free].T
            errors.append(np.abs(z - saddle(x, y)).max())
        ax.loglog(h, errors, "o-", color=color, lw=2, ms=6, label=label)
        if use_ref and laplacian == "cotan":
            ax.loglog(h, errors[0] * (h / h[0]) ** 2, "--", color="#8a8984", lw=1.2, label="slope 2 (∝ h²)")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("grid spacing h")
    ax.set_ylabel("max error |z − (x² − y²)| on the free disk")
    ax.set_title("Case B: does the membrane converge to the exact answer?")
    ax.grid(True, color="#e6e5e1", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(IMG / "10-convergence.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "10-convergence.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    figure_sparsity()
    figure_membrane()
    figure_convergence()
