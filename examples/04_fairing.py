"""Phase 4 example: fairing, L^k x = 0 (Sec. 4.3, App. A).

Run from the repository root:

    python examples/04_fairing.py

Saves docs/img/09-sparsity.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fairing import mesh
from fairing.fairing import fairing_matrix

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


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    figure_sparsity()
