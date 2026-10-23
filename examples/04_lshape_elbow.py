"""Fair the elbow starting from a sharp L-shaped (mitered) bend.

Run from the repository root:

    python examples/04_lshape_elbow.py                  # L-shaped start
    python examples/04_lshape_elbow.py --start chamfer  # chamfered start

Uses the dimensions and connectivity of 04_readme_elbow.py, but the free
region does not start on the quarter torus: each free vertex starts on the
mitered joint of the two cylinders, i.e. the vertical pipe is extended up to
the plane x + z = bend_radius and the horizontal pipe is extended back to it,
meeting at a 90-degree crease. The fairing operator is frozen on this start
mesh, then solved for k = 1, 2, 3 with cotangent and uniform weights.
With ``--start chamfer`` the free rings instead start on the straight\nlines between the two joint rings. Saves docs/img/11-elbow-<start>-cotan-uniform.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fairing import mesh, viz
from fairing.fairing import solve_fair


IMG = Path(__file__).resolve().parents[1] / "docs" / "img"


def lshape_elbow(n_theta=192, radius=1.0, bend_radius=1.25, pipe_length=1.2, spacing=0.025):
    """Return ``V, F, free`` of mesh.pipe_elbow with the free rings on a mitered L.

    Vertex at angle theta walks from the top ring of the vertical pipe,
    ``(r cos, r sin, 0)``, straight up to the miter plane at
    ``(r cos, r sin, R - r cos)``, then straight along x to the first ring of
    the horizontal pipe, ``(R, r sin, R - r cos)``. Both legs have length
    ``R - r cos(theta)``, so the free rings split evenly between them and the
    middle ring lies on the crease (it needs ``bend_radius > radius``).
    """
    V, F, free = mesh.pipe_elbow(n_theta=n_theta, radius=radius, bend_radius=bend_radius,
                                 pipe_length=pipe_length, spacing=spacing)
    n_pipe = int(round(pipe_length / spacing))
    n_bend = int(round(bend_radius * np.pi / 2 / spacing))
    theta = 2 * np.pi * np.arange(n_theta) / n_theta
    c, s = radius * np.cos(theta), radius * np.sin(theta)
    leg = bend_radius - c
    for m in range(1, n_bend):
        t = 2 * m / n_bend                                     # 0..2 along the two legs
        ring = slice((n_pipe + m) * n_theta, (n_pipe + m + 1) * n_theta)
        up, along = np.minimum(t, 1) * leg, np.maximum(t - 1, 0) * leg
        V[ring] = np.stack([c + along, s, up], axis=1)
    return V, F, free


def chamfer_elbow(n_theta=192, radius=1.0, bend_radius=1.25, pipe_length=1.2, spacing=0.025):
    """Return ``V, F, free`` with each free ring on the straight line between the
    two joint rings (the chamfer of examples/04_fairing.py, ``chamfered_start``)."""
    V, F, free = mesh.pipe_elbow(n_theta=n_theta, radius=radius, bend_radius=bend_radius,
                                 pipe_length=pipe_length, spacing=spacing)
    rings = V.reshape(-1, n_theta, 3)
    b = np.flatnonzero(free.reshape(-1, n_theta)[:, 0])
    first, last = rings[b[0] - 1].copy(), rings[b[-1] + 1].copy()
    for i, j in enumerate(b, start=1):
        t = i / (len(b) + 1)
        rings[j] = (1 - t) * first + t * last
    return V, F, free


STARTS = {"lshape": (lshape_elbow, "L-shaped (mitered)", "L"),
          "chamfer": (chamfer_elbow, "chamfered", "chamfer")}


def figure_lshape_elbow(start="lshape"):
    build, long_name, short = STARTS[start]
    V, F, free = build()
    free_face = free[F].any(axis=1)
    rows = (("cotan", "cotangent"), ("uniform", "uniform"))
    fig = plt.figure(figsize=(18.4, 9.6))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.90, wspace=0.02, hspace=0.12)
    for r, (lap, name) in enumerate(rows):
        for k in (1, 2, 3):
            W = solve_fair(V, F, free, k, laplacian=lap)
            assert np.array_equal(W[~free], V[~free]) and np.isfinite(W).all()
            ax = fig.add_subplot(2, 4, 4 * r + k + 1, projection="3d")
            _draw(ax, W, F, free_face, V[~free])
            ax.set_title(f"{name}, k = {k}", fontsize=12, pad=2)
        ax = fig.add_subplot(2, 4, 4 * r + 1, projection="3d")
        _draw(ax, V, F, free_face, V[~free])
        ax.set_title(f"start mesh ({short}), {name} row", fontsize=12, pad=2)
    fig.suptitle(f"Fairing two pipes at 90° from a {long_name} start: cotangent vs uniform",
                 fontsize=16, y=0.98)
    fig.text(0.5, 0.015,
             f"{len(V):,} vertices · {len(F):,} triangles · gray: fixed pipes · blue: faired region",
             ha="center", fontsize=10, color="#555555")
    IMG.mkdir(parents=True, exist_ok=True)
    target = IMG / f"11-elbow-{start}-cotan-uniform.png"
    fig.savefig(target, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", target)


def _draw(ax, W, F, free_face, frame):
    for faces, color in ((F[~free_face], "#d9d8d4"), (F[free_face], "#5b62c9")):
        ax.plot_trisurf(W[:, 0], W[:, 1], W[:, 2], triangles=faces, color=color,
                        shade=True, linewidth=0, antialiased=False)
    viz._set_equal_aspect(ax, frame)
    ax.view_init(10, -100)
    ax.set_axis_off()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", choices=sorted(STARTS), default="lshape")
    figure_lshape_elbow(parser.parse_args().start)
