"""Phase 4 example: fairing, L^k x = 0 (Sec. 4.3, App. A).

Run from the repository root:

    python examples/04_fairing.py

Saves docs/img/09-sparsity.png, docs/img/10-membrane.png,
docs/img/10-two-membranes.png, docs/img/11-tube-k123.png, docs/img/11-profile.png, docs/img/11-elbow-k123.png,
docs/img/11-elbow-profiles.png and docs/img/11-elbow-weights.png.
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
from sksparse.cholmod import cho_factor

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


def figure_elbow():
    """Issue #11 (extra): Fig. 4.8's setting, two pipes at 90 degrees, solved with k = 1, 2, 3.

    Top row: rendered like the book (lit surfaces, fixed pipes gray, free bend blue).
    Bottom row: the same surfaces colored by mean curvature (fixed pipes gray).
    """
    V, F, free = mesh.pipe_elbow()
    # a face belongs to the free surface if it has at least one free vertex: the color boundary
    # is then the last fixed ring (the true border, with the pipe's radius)
    free_face = free[F].any(axis=1)
    joint = mesh.ring_distance(F, np.flatnonzero(free), max_k=1) == 1      # the last fixed rings
    view = (10, -100)                         # matched by eye to Fig. 4.8: nearly frontal, slightly above
    fig = plt.figure(figsize=(12.5, 8.2))
    fig.subplots_adjust(left=0.02, right=0.9, wspace=0.05, hspace=0.12)
    for col, k in enumerate((1, 2, 3)):
        W = solve_fair(V, F, free, k)
        name = ["membrane (C⁰)", "thin plate (C¹)", "minimum variation (C²)"][col]

        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        for faces, color in ((F[~free_face], "#d9d8d4"), (F[free_face], "#5b62c9")):
            ax.plot_trisurf(W[:, 0], W[:, 1], W[:, 2], triangles=faces, color=color,
                            shade=True, linewidth=0, antialiased=False)
        viz._set_equal_aspect(ax, W)
        ax.view_init(*view)
        ax.set_axis_off()
        ax.set_title(f"k = {k}: {name}")

        H = mean_curvature(W, F)
        H[~(free | joint)] = np.nan                               # pipes in gray, except the joint rings
        ax = fig.add_subplot(2, 3, col + 4, projection="3d")
        _, surf = viz.plot_mesh(W, F, H, ax, vmin=0, vmax=1.5, edges=False, colorbar=False,
                                elev=view[0], azim=view[1], title=f"k = {k}: mean curvature H")
    cax = fig.add_axes([0.93, 0.12, 0.012, 0.3])                 # own slot: do not shrink the panels
    fig.colorbar(surf, cax=cax, label="H (straight pipe: 0.5;\n≥ 1.5 saturates)")
    fig.suptitle("Two pipes at 90° (fixed) joined by a free bend: compare with Fig. 4.8", y=0.98)
    fig.savefig(IMG / "11-elbow-k123.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "11-elbow-k123.png")


def chamfered_start(V, bend, n_theta):
    """Alternative start for the elbow's free bend: each bend ring placed on the straight
    line between the two joint rings (a "chamfered" connector instead of a quarter torus)."""
    rings = V.reshape(-1, n_theta, 3).copy()
    b = np.flatnonzero(bend.reshape(-1, n_theta)[:, 0])
    first, last = rings[b[0] - 1], rings[b[-1] + 1]               # the two joint rings (fixed)
    for i, j in enumerate(b, start=1):
        t = i / (len(b) + 1)
        rings[j] = (1 - t) * first + t * last
    return rings.reshape(-1, 3)


def figure_elbow_profiles():
    """Issue #11 (extra): the elbow's outer and inner meridians in the bend plane (y = 0).

    Left: k = 2, 3 from the quarter-torus start. Right: k = 3 from two different starts
    (cotangent weights depend on the start; uniform weights do not).
    """
    n_theta = 48
    V, F, bend = mesh.pipe_elbow(n_theta=n_theta)
    rings = lambda W: W.reshape(-1, n_theta, 3)
    joints = np.flatnonzero(bend.reshape(-1, n_theta)[:, 0])[[0, -1]] + [-1, 1]
    V_chamfer = chamfered_start(V, bend, n_theta)

    def meridians(ax, W, color, label, dashed=False, lw=2.0):
        for i in (n_theta // 2, 0):                                # outer (theta = pi), inner (theta = 0)
            ax.plot(rings(W)[:, i, 0], rings(W)[:, i, 2], color=color, lw=lw,
                    dashes=(4, 3) if dashed else (), label=label if i == n_theta // 2 else None)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6.4))
    meridians(left, V, "#8a8984", "quarter torus (start)", dashed=True, lw=1.5)
    meridians(left, solve_fair(V, F, bend, 2), "#3987e5", "k = 2")
    meridians(left, solve_fair(V, F, bend, 3), "#0d366b", "k = 3")
    left.set_title("from the quarter-torus start:\nthe outer side takes a shortcut")

    meridians(right, V, "#8a8984", "quarter torus (start A)", dashed=True, lw=1.5)
    meridians(right, V_chamfer, "#d9a38a", "chamfer (start B)", dashed=True, lw=1.5)
    meridians(right, solve_fair(V, F, bend, 3), "#0d366b", "k = 3, cotangent, from A")
    meridians(right, solve_fair(V_chamfer, F, bend, 3), "#eb6834", "k = 3, cotangent, from B")
    meridians(right, solve_fair(V, F, bend, 3, "uniform"), "#86b6ef", "k = 3, uniform (same from A or B)")
    right.set_title("k = 3 from two different starts:\ncotangent depends on the start, uniform does not")

    for ax in (left, right):
        for j in joints:                                            # the two joint rings
            ax.plot(rings(V)[j, [0, n_theta // 2], 0], rings(V)[j, [0, n_theta // 2], 2],
                    color="#c3c2b7", lw=1, zorder=0)
        ax.set_aspect("equal")
        ax.set_xlim(-1.3, 2.9)
        ax.set_ylim(-1.3, 2.9)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Elbow in the bend plane (y = 0): outer and inner sides of the pipe; "
                 "gray lines = the two joints", y=1.0)
    fig.tight_layout()
    fig.savefig(IMG / "11-elbow-profiles.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "11-elbow-profiles.png")


# --- Investigation: where did Fig. 4.8's weights come from? -------------------------------
# Botsch & Kobbelt 2004 (the source of Fig. 4.8) compute L once on the *original* surface and
# move a handle. For a bent pipe, the original was plausibly a straight tube. These two helpers
# compare that with our choice (weights from the quarter torus). Not part of the library.

def straight_tube_like(V, n_theta, pipe_length=1.2, spacing=0.1):
    """A straight vertical tube with the same rings as the elbow: ring j at z = (j - n_pipe) * spacing,
    so the bottom pipe coincides with the elbow's and the bend rings continue straight up."""
    n_rings = len(V) // n_theta
    n_pipe = int(round(pipe_length / spacing))
    theta = 2 * np.pi * np.arange(n_theta) / n_theta
    z = (np.arange(n_rings) - n_pipe) * spacing
    return np.stack([np.tile(np.cos(theta), n_rings), np.tile(np.sin(theta), n_rings), np.repeat(z, n_theta)], 1)


def solve_with_weights_from(G, V, F, free, k):
    """Like solve_fair, but the matrix is built from geometry G instead of V (the paper's setting)."""
    A = fairing_matrix(G, F, k)
    W = V.copy()
    W[free] = cho_factor(A[free][:, free].tocsc()).solve(-(A[free][:, ~free] @ V[~free]))
    return W


def figure_elbow_weights():
    """Issue #11 (investigation): k = 3 with weights from the quarter torus vs from a straight tube,
    for several bend radii."""
    radii = (1.25, 1.5, 2.0, 3.0)
    columns = ("quarter torus (start)", "k = 3, weights from the torus", "k = 3, weights from a straight tube")
    fig = plt.figure(figsize=(11, 13))
    for row, Rb in enumerate(radii):
        V, F, free = mesh.pipe_elbow(bend_radius=Rb)
        face_free = free[F].any(axis=1)
        results = (V, solve_fair(V, F, free, 3),
                   solve_with_weights_from(straight_tube_like(V, 48), V, F, free, 3))
        for col, W in enumerate(results):
            ax = fig.add_subplot(len(radii), 3, 3 * row + col + 1, projection="3d")
            for faces, color in ((F[~face_free], "#d9d8d4"), (F[face_free], "#5b62c9")):
                ax.plot_trisurf(W[:, 0], W[:, 1], W[:, 2], triangles=faces, color=color,
                                shade=True, linewidth=0, antialiased=False)
            viz._set_equal_aspect(ax, W)
            ax.view_init(10, -100)
            ax.set_axis_off()
            ax.set_title(f"bend radius {Rb}: {columns[col]}", fontsize=8.5)
    fig.suptitle("Where do the weights come from? k = 3 elbows (pipe radius 1)", y=1.0)
    fig.tight_layout()
    fig.savefig(IMG / "11-elbow-weights.png", dpi=120, bbox_inches="tight")
    print("saved", IMG / "11-elbow-weights.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    figure_sparsity()
    figure_membrane()
    figure_two_membranes()
    figure_tube_blend()
    figure_profile()
    figure_elbow()
    figure_elbow_profiles()
    figure_elbow_weights()
