"""Phase 4 example: fairing, L^k x = 0 (Sec. 4.3, App. A).

Run from the repository root:

    python examples/04_fairing.py

Saves docs/img/09-sparsity.png, docs/img/10-membrane.png,
docs/img/10-two-membranes.png, docs/img/11-tube-k123.png, docs/img/11-profile.png and docs/img/11-elbow-k123.png.
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


def tube_blend(n_theta, n_z, shift=1.0):
    """Tube (radius 1, height 4): bottom band z <= 1 fixed, top band z >= 3 fixed and shifted
    sideways by `shift`, middle free, starting as a straight sheared tube (clean start, #10)."""
    V, F = mesh.tube(n_theta, n_z, radius=1.0, height=4.0)
    z = V[:, 2]
    free = (z > 1 + 1e-9) & (z < 3 - 1e-9)
    V[:, 0] += shift * np.clip((z - 1) / 2, 0, 1)
    return V, F, free


K_COLORS = {1: "#86b6ef", 2: "#3987e5", 3: "#0d366b"}        # k is ordered: one blue ramp, light to dark


def figure_tube_blend():
    """Issue #11: the same blend solved with k = 1, 2, 3 (Fig. 4.8), colored by mean curvature."""
    n_theta, n_z = 48, 61
    V, F, free = tube_blend(n_theta, n_z)
    meshes = [(solve_fair(V, F, free, k), F) for k in (1, 2, 3)]
    titles = ["k = 1: membrane (C⁰)", "k = 2: thin plate (C¹)", "k = 3: minimum variation (C²)"]
    scalars = [mean_curvature(W, F) for W, _ in meshes]
    for H in scalars:
        H[mesh.boundary_vertices(F)] = np.nan                  # open ends: one-sided Laplacian
    fig = viz.compare(meshes, titles, scalars, vmin=0, vmax=1.5, views=[(8, -90)] * 3,
                      panel_size=(3.8, 5.2), edges=False)
    fig.suptitle("Tube blend: bottom band (z ≤ 1) and shifted top band (z ≥ 3) fixed, middle free.\n"
                 "Colored by mean curvature H (straight tube: 0.5; ≥ 1.5 saturates)", y=0.95)
    fig.savefig(IMG / "11-tube-k123.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "11-tube-k123.png")


def figure_profile():
    """Issue #11: the meridian profile in the x-z plane; zoom on the bottom joint."""
    n_theta, n_z = 48, 61
    V, F, free = tube_blend(n_theta, n_z)
    right = np.arange(n_z) * n_theta                           # theta = 0 meridian (x > 0 side)
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(9.5, 5.6), gridspec_kw={"width_ratios": [0.7, 1.3]})
    for k in (1, 2, 3):
        W = solve_fair(V, F, free, k)
        for a in (ax, zoom):
            a.plot(W[right, 0], W[right, 2], "-o", color=K_COLORS[k], lw=2, ms=3 if a is zoom else 0,
                   label=f"k = {k}")
    fixed = ~free[right]
    for a in (ax, zoom):
        a.plot(V[right[fixed], 0], V[right[fixed], 2], "o", color="#8a8984", ms=3.5, zorder=3,
               label="fixed vertices")
        a.axhline(1.0, color="#c3c2b7", lw=0.8, ls="--")
        a.set_xlabel("x")
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    ax.axhline(3.0, color="#c3c2b7", lw=0.8, ls="--")
    ax.set_ylabel("z")
    ax.set_aspect("equal")
    ax.set_title("meridian profile (θ = 0 side)")
    zoom.set_xlim(0.55, 1.3)
    zoom.set_ylim(0.55, 1.65)
    zoom.set_aspect("equal")
    zoom.set_title("zoom on the bottom joint (z = 1, dashed)")
    arrow = dict(arrowstyle="-", color="#8a8984", lw=0.8)
    zoom.annotate("k = 1 leaves the straight\ntube with a kink (C⁰)", xy=(0.985, 1.06), xytext=(0.6, 0.8),
                  fontsize=9, color="#3d3c39", arrowprops=arrow)
    zoom.annotate("k = 2, 3 leave it\ntangentially (C¹, C²)", xy=(1.012, 1.1), xytext=(1.06, 0.75),
                  fontsize=9, color="#3d3c39", arrowprops=arrow)
    # one legend for both panels, below them: the left panel is too narrow to hold it
    handles, labels = zoom.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    fig.savefig(IMG / "11-profile.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "11-profile.png")


def pipe_elbow(n_theta=48, radius=1.0, bend_radius=2.5, pipe_length=2.0, spacing=0.1):
    """Two pipes at 90 degrees, joined by a quarter-torus bend (the setting of Fig. 4.8).

    A vertical pipe (axis z, z from -pipe_length to 0) and a horizontal pipe (axis x, at
    height bend_radius) are fixed; the bend between them is free. The mesh reuses the tube's
    connectivity (ring j, angle i -> vertex i + n_theta*j); only the ring positions change:
    ring j has a center c_j and an in-plane direction e1_j that turns with the bend (e2 = y).
    The free bend starts as an exact quarter torus, a clean surface for the frozen weights.
    """
    n_bend = int(round(bend_radius * np.pi / 2 / spacing))       # bend segments (≈ spacing long)
    n_pipe = int(round(pipe_length / spacing))                    # segments per straight pipe
    centers, e1 = [], []
    for j in range(n_pipe + 1):                                   # vertical pipe, ends at z = 0
        centers.append((0, 0, -pipe_length + j * spacing))
        e1.append((1, 0, 0))
    for m in range(1, n_bend):                                    # bend: angle phi from 0 to 90 deg
        phi = m * (np.pi / 2) / n_bend
        centers.append((bend_radius * (1 - np.cos(phi)), 0, bend_radius * np.sin(phi)))
        e1.append((np.cos(phi), 0, -np.sin(phi)))                 # stays perpendicular to the axis
    for j in range(n_pipe + 1):                                   # horizontal pipe, along +x
        centers.append((bend_radius + j * spacing, 0, bend_radius))
        e1.append((0, 0, -1))
    centers, e1 = np.array(centers, float), np.array(e1, float)
    _, F = mesh.tube(n_theta, len(centers))
    theta = 2 * np.pi * np.arange(n_theta) / n_theta
    circle = np.cos(theta)[None, :, None] * e1[:, None, :] + np.sin(theta)[None, :, None] * np.array([0, 1.0, 0])
    V = (centers[:, None, :] + radius * circle).reshape(-1, 3)
    ring = np.repeat(np.arange(len(centers)), n_theta)
    free = (ring > n_pipe) & (ring < n_pipe + n_bend)
    return V, F, free


def figure_elbow():
    """Issue #11 (extra): Fig. 4.8's setting, two pipes at 90 degrees, solved with k = 1, 2, 3."""
    V, F, free = pipe_elbow()
    meshes, scalars = [], []
    for k in (1, 2, 3):
        W = solve_fair(V, F, free, k)
        H = mean_curvature(W, F)
        H[~free] = np.nan                                         # fixed pipes in gray, as in the book
        meshes.append((W, F))
        scalars.append(H)
    titles = ["k = 1: membrane (C⁰)", "k = 2: thin plate (C¹)", "k = 3: minimum variation (C²)"]
    fig = viz.compare(meshes, titles, scalars, vmin=0, vmax=1.5, views=[(15, -90)] * 3,
                      panel_size=(4.2, 4.4), edges=False)
    fig.suptitle("Two pipes at 90° (fixed, gray) joined by a free bend, colored by mean curvature H\n"
                 "(straight pipe: 0.5; ≥ 1.5 saturates). Compare with Fig. 4.8.", y=0.98)
    fig.savefig(IMG / "11-elbow-k123.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "11-elbow-k123.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    figure_sparsity()
    figure_membrane()
    figure_two_membranes()
    figure_tube_blend()
    figure_profile()
    figure_elbow()
