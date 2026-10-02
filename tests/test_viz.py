"""Smoke tests for the plotting helpers (issue #3). They never import polyscope."""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fairing import viz
from fairing.mesh import grid, uv_sphere


def test_plot_mesh_plain_and_with_scalars(tmp_path):
    V, F = uv_sphere(6, 8)
    ax, _ = viz.plot_mesh(V, F, title="plain")
    ax.figure.savefig(tmp_path / "plain.png")

    ax, surf = viz.plot_mesh(V, F, scalars=V[:, 2])
    face_values = V[F, 2].mean(axis=1)
    assert np.allclose(surf.get_array(), face_values)       # one color value per face
    ax.figure.savefig(tmp_path / "scalars.png")
    plt.close("all")


def test_plot_mesh_diverging_is_centered_at_zero():
    V, F = grid(5, 5)
    _, surf = viz.plot_mesh(V, F, scalars=V[:, 0] + 0.1, diverging=True)
    assert surf.norm.vcenter == 0.0
    assert surf.norm.vmin == -surf.norm.vmax
    plt.close("all")


def test_compare_shares_the_color_scale(tmp_path):
    meshes = [grid(5, 5), grid(6, 6)]
    scalars = [np.linspace(0, 1, 25), np.linspace(2, 3, 36)]
    fig = viz.compare(meshes, ["a", "b"], scalars, views=[(90, -90), None])
    norms = [c.norm for ax in fig.axes if ax.name == "3d" for c in ax.collections]
    assert all((n.vmin, n.vmax) == (0, 3) for n in norms)
    fig.savefig(tmp_path / "compare.png")
    plt.close("all")


def test_polyscope_is_never_imported():
    assert "polyscope" not in sys.modules
