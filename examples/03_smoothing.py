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
from fairing.laplacian import mean_curvature, uniform_laplacian
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
    """Issue #7: just above the step limit, explicit smoothing blows up.

    The plotted quantity is the mean ||Lx|| of the uniform Laplacian: the average
    distance from a vertex to the centroid of its neighbors, i.e. the size of the
    zig-zag (high-frequency) part of the mesh.
    """
    V0, F = noisy_sphere()
    limit = explicit_step_limit(V0, F, laplacian="uniform")
    L = uniform_laplacian(V0, F)[0]                        # depends only on F
    zigzag = lambda V: np.linalg.norm(L @ V, axis=1).mean()
    runs = [(0.5, "#86b6ef"), (0.9, "#1c5cab"), (1.1, "#eb6834")]
    n_iter, snap_it = 40, 10
    fig = plt.figure(figsize=(10, 4.6))
    ax_plot = fig.add_subplot(1, 2, 2)
    for frac, color in runs:
        V, amp = V0, [zigzag(V0)]
        for it in range(1, n_iter + 1):
            V = explicit_smoothing(V, F, frac * limit, laplacian="uniform")
            amp.append(zigzag(V))
            if frac > 1 and it == snap_it:
                snapshot = V
        ax_plot.semilogy(amp, color=color, lw=2, label=f"h = {frac} × limit")
        if frac > 1:
            unstable_amp = amp
    ax_plot.plot(snap_it, unstable_amp[snap_it], "o", ms=9, color="#eb6834", mec="white", mew=1.5, zorder=3)
    ax_plot.annotate(f"step {snap_it} (left)", (snap_it, unstable_amp[snap_it]),
                     xytext=(snap_it + 3, unstable_amp[snap_it] * 4), fontsize=9, color="#3d3c39",
                     arrowprops=dict(arrowstyle="-", color="#8a8984", lw=0.8))
    ax_plot.annotate("above the limit the zig-zag grows\n×1.2 per step: a straight line\non this log axis",
                     (27, unstable_amp[22]), fontsize=9, color="#3d3c39", ha="left")
    ax_plot.set_ylim(bottom=2.5e-3)
    ax_plot.annotate("below the limit the noise is damped; what remains\nis the smooth sphere's own (curvature) part of Lx",
                     (12, 3.0e-3), fontsize=9, color="#3d3c39")
    ax_plot.set_xlabel("iteration")
    ax_plot.set_ylabel("zig-zag size: mean ‖Lx‖ (uniform)")
    ax_plot.set_title(f"uniform Laplacian, step limit = {limit:.2f}")
    ax_plot.grid(True, color="#e6e5e1", lw=0.8)
    for side in ("top", "right"):
        ax_plot.spines[side].set_visible(False)
    ax_plot.legend(frameon=False, loc="upper left")

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    viz.plot_mesh(snapshot, F, ax=ax,
                  title=f"h = 1.1 × limit, step {snap_it}:\nneighbors pushed in opposite directions")
    fig.tight_layout()
    fig.savefig(IMG / "07-explicit-unstable.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "07-explicit-unstable.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    figure_explicit_iterations()
    figure_explicit_unstable()
