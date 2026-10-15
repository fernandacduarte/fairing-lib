"""Render the README's elbow comparison using a straight reference tube.

Run from the repository root:

    python examples/04_reference_elbow.py

Uses the dimensions, connectivity, resolution, and camera of 04_readme_elbow.py.
The operator is built on an aligned straight tube. Only the target fixed pipes
are placed at 90 degrees; no quarter-torus geometry is constructed.
Saves docs/img/11-elbow-k123-reference.png, preserving the original comparison.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fairing import mesh, viz
from fairing.fairing import solve_fair_with_reference
from fairing.laplacian import cotan_laplacian


IMG = Path(__file__).resolve().parents[1] / "docs" / "img"


def elbow_problem(n_theta=192, radius=1.0, bend_radius=1.25,
                  pipe_length=1.2, spacing=0.025):
    """Return target positions, faces, free mask, and straight reference positions.

    The free span of the reference is approximately ``pi*bend_radius/2`` long,
    matching the longitudinal sampling of the original elbow comparison. This
    length is a reference-model parameter, not an imposed curved surface.
    Free target vertices retain their straight reference positions as unused
    placeholders. The upper fixed pipe is rotated 90 degrees about y and moved
    so that its first ring is centered at ``(bend_radius, 0, bend_radius)``.
    """
    n_pipe = int(round(pipe_length / spacing))
    n_span = int(round(bend_radius * np.pi / 2 / spacing))
    n_rings = 2 * n_pipe + n_span + 1
    reference_V, F = mesh.tube(n_theta, n_rings, radius=radius,
                              height=(n_rings - 1) * spacing)
    reference_V[:, 2] -= pipe_length
    ring = np.repeat(np.arange(n_rings), n_theta)
    free = (ring > n_pipe) & (ring < n_pipe + n_span)
    upper = ring >= n_pipe + n_span
    V = reference_V.copy()
    # Rotate the reference pipe's local (x, y, z) to (z, y, -x).
    upper_origin = (n_pipe + n_span) * spacing - pipe_length
    V[upper, 0] = bend_radius + reference_V[upper, 2] - upper_origin
    V[upper, 2] = bend_radius - reference_V[upper, 0]
    return V, F, free, reference_V


def figure_reference_elbow():
    """Compare k = 1, 2, 3 with the same straight reference and target constraints."""
    V, F, free, reference_V = elbow_problem()
    free_face = free[F].any(axis=1)
    titles = ("k = 1: membrane (C⁰)", "k = 2: thin plate (C¹)",
              "k = 3: minimum variation (C²)")
    L = cotan_laplacian(reference_V, F)[0]
    fig = plt.figure(figsize=(13.8, 4.8))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.10, top=0.82, wspace=0.02)
    print(f"straight reference: {len(V):,} vertices, {len(F):,} triangles; "
          f"{free.sum():,} free vertices")
    for k, title in enumerate(titles, start=1):
        W = solve_fair_with_reference(V, F, free, k, reference_V=reference_V)
        residual, initial = W.copy(), V.copy()
        for _ in range(k):
            residual, initial = L @ residual, L @ initial
        relative = np.abs(residual[free]).max() / np.abs(initial[free]).max()
        assert np.array_equal(W[~free], V[~free])
        assert np.isfinite(W).all() and relative < 1e-8
        print(f"k = {k}: fixed vertices unchanged; relative L^k residual {relative:.2e}")
        ax = fig.add_subplot(1, 3, k, projection="3d")
        for faces, color in ((F[~free_face], "#d9d8d4"), (F[free_face], "#5b62c9")):
            ax.plot_trisurf(W[:, 0], W[:, 1], W[:, 2], triangles=faces, color=color,
                            shade=True, linewidth=0, antialiased=False)
        # The fixed pipes define identical framing for all three solutions.
        viz._set_equal_aspect(ax, V[~free])
        ax.view_init(10, -100)
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, pad=4)
    fig.suptitle("Two pipes at 90°: cotangent weights from a straight reference tube",
                 fontsize=15, y=0.98)
    fig.text(0.5, 0.055,
             f"{len(V):,} vertices · {len(F):,} triangles · gray: fixed pipes · blue: faired region",
             ha="center", fontsize=10, color="#555555")
    fig.text(0.5, 0.015, "Fixed reference operator; the linear model does not prevent folds or intersections.",
             ha="center", fontsize=9, color="#555555")
    IMG.mkdir(parents=True, exist_ok=True)
    target = IMG / "11-elbow-k123-reference.png"
    fig.savefig(target, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", target)
    return target


if __name__ == "__main__":
    figure_reference_elbow()
