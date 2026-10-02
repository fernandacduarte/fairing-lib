"""Phase 3 example: diffusion flow (Sec. 4.2).

Run from the repository root:

    python examples/03_smoothing.py

Saves docs/img/07-explicit-iters.png and docs/img/07-explicit-unstable.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fairing import mesh, viz
from fairing.laplacian import mean_curvature
from fairing.smoothing import explicit_smoothing, explicit_step_limit, roughness

IMG = Path(__file__).resolve().parents[1] / "docs" / "img"


def noisy_sphere():
    V, F = mesh.uv_sphere(24, 48)
    return mesh.add_noise(V, 0.01, seed=0), F


def figure_explicit_iterations():
    """Issue #7: iterations 0, 10, 100 of explicit smoothing, at 0.9x each operator's step limit."""
    V0, F = noisy_sphere()
    meshes, scalars, titles = [], [], []
    for laplacian in ("uniform", "cotan"):
        h = 0.9 * explicit_step_limit(V0, F, laplacian=laplacian)
        V, done = V0, 0
        for it in (0, 10, 100):
            V = explicit_smoothing(V, F, h, n_iter=it - done, laplacian=laplacian)
            done = it
            meshes.append((V, F))
            scalars.append(mean_curvature(V, F))
            titles.append(f"{laplacian}, h = {h:.2g}\niteration {it}, roughness {roughness(V, F):.3f}")
    # The true H of the unit sphere is 1; noise gives a long tail (up to ~40), so the
    # scale stops at 4 and larger values saturate in the darkest blue.
    fig = viz.compare(meshes, titles, scalars, ncols=3, vmin=0, vmax=4, panel_size=(4.0, 4.0))
    fig.suptitle("Explicit smoothing of a noisy unit sphere, colored by mean curvature H (≥ 4 saturates)\n"
                 "each panel is zoomed to fit its mesh, so sizes are not comparable across panels", y=1.02)
    fig.savefig(IMG / "07-explicit-iters.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "07-explicit-iters.png")


def figure_explicit_unstable():
    """Issue #7: just above the step limit, explicit smoothing blows up."""
    V0, F = noisy_sphere()
    limit = explicit_step_limit(V0, F, laplacian="uniform")
    runs = [(0.5, "#86b6ef"), (0.9, "#1c5cab"), (1.1, "#eb6834")]
    n_iter = 60
    fig = plt.figure(figsize=(10, 4.4))
    ax_plot = fig.add_subplot(1, 2, 2)
    spiky = None
    for frac, color in runs:
        V, sizes = V0, [np.abs(V0).max()]
        for it in range(1, n_iter + 1):
            V = explicit_smoothing(V, F, frac * limit, laplacian="uniform")
            sizes.append(np.abs(V).max())
            if frac > 1 and spiky is None and sizes[-1] > 1.6:
                spiky = (V, it)
        ax_plot.semilogy(sizes, color=color, lw=2, label=f"h = {frac} × limit")
    ax_plot.axhline(1.0, color="#8a8984", lw=1, ls="--")
    ax_plot.set_xlabel("iteration")
    ax_plot.set_ylabel("largest coordinate |x|")
    ax_plot.set_title(f"uniform Laplacian, step limit = {limit:.2f}")
    ax_plot.grid(True, color="#e6e5e1", lw=0.8)
    for side in ("top", "right"):
        ax_plot.spines[side].set_visible(False)
    ax_plot.legend(frameon=False, loc="upper left")

    V, it = spiky
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    viz.plot_mesh(V, F, ax=ax, title=f"h = 1.1 × limit, iteration {it}")
    fig.tight_layout()
    fig.savefig(IMG / "07-explicit-unstable.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "07-explicit-unstable.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    figure_explicit_iterations()
    figure_explicit_unstable()
