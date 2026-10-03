"""Phase 3 example: diffusion flow (Sec. 4.2).

Run from the repository root:

    python examples/03_smoothing.py

Saves docs/img/07-explicit-iters.png, docs/img/07-explicit-unstable.png,
docs/img/08-implicit-large-h.png and docs/img/08-uniform-vs-cotan-shapes.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fairing import mesh, viz
from fairing.laplacian import mean_curvature, uniform_laplacian
from fairing.smoothing import explicit_smoothing, explicit_step_limit, implicit_smoothing, roughness

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
    n_iter, snap_its = 40, (10, 32)

    fig = plt.figure(figsize=(11, 7.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.45])
    ax_plot = fig.add_subplot(grid[:, 1])
    snapshots = {}
    for frac, color in runs:
        V, amp = V0, [zigzag(V0)]
        for it in range(1, n_iter + 1):
            V = explicit_smoothing(V, F, frac * limit, laplacian="uniform")
            amp.append(zigzag(V))
            if frac > 1 and it in snap_its:
                snapshots[it] = V
        ax_plot.semilogy(amp, color=color, lw=2, label=f"h = {frac} × h_max")
        if frac > 1:
            unstable_amp = amp
    for it, (dx, dy) in zip(snap_its, ((-9, 4), (-12, 2))):     # label offsets, clear of the curve
        ax_plot.plot(it, unstable_amp[it], "o", ms=9, color="#eb6834", mec="white", mew=1.5, zorder=3)
        ax_plot.annotate(f"step {it}", (it, unstable_amp[it]), xytext=(it + dx, unstable_amp[it] * dy),
                         fontsize=9, color="#3d3c39", arrowprops=dict(arrowstyle="-", color="#8a8984", lw=0.8))
    ax_plot.annotate("above h_max the zig-zag grows\n×1.2 per step: a straight line\non this log axis",
                     (27, unstable_amp[20]), fontsize=9, color="#3d3c39", ha="left")
    ax_plot.set_ylim(bottom=2.5e-3)
    ax_plot.annotate("below h_max the noise is damped; what remains\nis the smooth sphere's own (curvature) part of Lx",
                     (12, 3.0e-3), fontsize=9, color="#3d3c39")
    ax_plot.set_xlabel("iteration")
    ax_plot.set_ylabel("zig-zag size: mean ‖Lx‖ (uniform)")
    ax_plot.set_title(f"uniform Laplacian, h_max = {limit:.2f}")
    ax_plot.grid(True, color="#e6e5e1", lw=0.8)
    for side in ("top", "right"):
        ax_plot.spines[side].set_visible(False)
    ax_plot.legend(frameon=False, loc="upper left")

    titles = {10: "h = 1.1 × h_max, step 10:\nneighbors pushed in opposite directions",
              32: "h = 1.1 × h_max, step 32:\nthe zig-zag has torn the mesh apart"}
    for row, it in enumerate(snap_its):
        ax = fig.add_subplot(grid[row, 0], projection="3d")
        viz.plot_mesh(snapshots[it], F, ax=ax, title=titles[it])
    fig.tight_layout()
    fig.savefig(IMG / "07-explicit-unstable.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "07-explicit-unstable.png")


def figure_implicit_large_h():
    """Issue #8: one step with h ~ 340 x h_max. Explicit explodes; implicit smooths."""
    V0, F = noisy_sphere()
    h_max = explicit_step_limit(V0, F, laplacian="cotan")
    h = 0.01
    explicit = explicit_smoothing(V0, F, h, laplacian="cotan")
    implicit = implicit_smoothing(V0, F, h, laplacian="cotan")
    meshes = [(V0, F), (explicit, F), (implicit, F)]
    titles = ["noisy input",
              f"explicit, 1 step\nh = {h} ≈ {h / h_max:.0f} × h_max",
              f"implicit, 1 step, same h\nroughness {roughness(V0, F):.3f} → {roughness(implicit, F):.3f}"]
    fig = viz.compare(meshes, titles, [mean_curvature(V, F) for V, _ in meshes],
                      vmin=0, vmax=4, panel_size=(4.0, 4.2))
    fig.suptitle("Cotangent Laplacian, colored by mean curvature H (true value 1, ≥ 4 saturates)", y=1.0)
    fig.savefig(IMG / "08-implicit-large-h.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "08-implicit-large-h.png")


def crop(V, F, keep_face):
    """Sub-mesh made of the selected faces, with vertices re-indexed."""
    F = F[keep_face]
    used, F = np.unique(F, return_inverse=True)
    return V[used], F.reshape(-1, 3), used


def figure_shapes():
    """Issue #8, in the spirit of Fig. 4.6: uniform drifts vertices tangentially, cotangent does not."""
    V0, F = mesh.irregular_sphere(24, 48, jitter=0.15, seed=0)
    runs = [("input: exact sphere,\nirregular triangles", V0),
            ("uniform, implicit, h = 5", implicit_smoothing(V0, F, 5.0, laplacian="uniform")),
            ("cotangent, implicit, h = 0.05", implicit_smoothing(V0, F, 0.05, laplacian="cotan"))]
    # front patch around the equator, chosen on the input so every panel shows the same triangles
    c = V0[F].mean(axis=1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    patch = (c[:, 0] > 0.8) & (np.abs(c[:, 2]) < 0.45)
    unit0 = V0 / np.linalg.norm(V0, axis=1, keepdims=True)

    meshes, scalars, titles = [], [], []
    for title, V in runs:
        unit = V / np.linalg.norm(V, axis=1, keepdims=True)
        drift = np.degrees(np.arccos(np.clip((unit * unit0).sum(axis=1), -1, 1)))   # sliding along the sphere
        Vp, Fp, used = crop(V, F, patch)
        meshes.append((Vp, Fp))
        scalars.append(None if V is V0 else drift[used])
        titles.append(title if V is V0 else f"{title}\nmean tangential drift {drift.mean():.2f}°")
    fig = viz.compare(meshes, titles, scalars, views=[(0, 0)] * 3, panel_size=(4.0, 4.4),
                      color="#f0efec", shade=False)               # flat light input panel
    fig.suptitle("Front patch of a sphere after one implicit step; color = how far each vertex slid along "
                 "the sphere (degrees; grid spacing 7.5°)", y=1.0)
    fig.savefig(IMG / "08-uniform-vs-cotan-shapes.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "08-uniform-vs-cotan-shapes.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    figure_explicit_iterations()
    figure_explicit_unstable()
    figure_implicit_large_h()
    figure_shapes()
