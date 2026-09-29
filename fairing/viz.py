"""Plotting with matplotlib (PNG) and an optional polyscope viewer.

Per-vertex scalars (for example ``‖Lx‖`` or the mean curvature ``H``) are shown
as face colors, like the color-coded figures of the book (Fig. 3.6, Fig. 4.5).
Magnitudes use a single-hue sequential map; signed quantities use a diverging
map with a neutral gray at zero.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

# Sequential: one hue (blue), light -> dark. For magnitudes such as ||Lx|| or H.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "fairing_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
# Diverging: blue <-> red with a neutral gray midpoint. For signed quantities.
DIVERGING = LinearSegmentedColormap.from_list(
    "fairing_div", ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#8f1d1c"])

SURFACE_COLOR = "#b7d3f6"   # plain meshes (no scalars)
EDGE_COLOR = "#52514e"


def _set_equal_aspect(ax, V):
    """Scale the three axes equally, so that shapes are not distorted."""
    lo, hi = V.min(axis=0), V.max(axis=0)
    extent = np.maximum(hi - lo, 0.02 * (hi - lo).max())   # flat meshes keep a thin box
    center = (lo + hi) / 2
    ax.set_xlim(center[0] - extent[0] / 2, center[0] + extent[0] / 2)
    ax.set_ylim(center[1] - extent[1] / 2, center[1] + extent[1] / 2)
    ax.set_zlim(center[2] - extent[2] / 2, center[2] + extent[2] / 2)
    ax.set_box_aspect(extent)


def plot_mesh(V, F, scalars=None, ax=None, *, title=None, edges=True, diverging=False,
              vmin=None, vmax=None, colorbar=True, elev=25, azim=-60, axis_off=True):
    """Draw a triangle mesh with ``plot_trisurf``, optionally colored by per-vertex scalars.

    Each face gets the mean of its three vertex scalars. With ``diverging=True``
    the color scale is centered at 0. Returns the axes and the surface artist
    (the artist is the ``mappable`` for a shared colorbar).
    """
    V, F = np.asarray(V, dtype=float), np.asarray(F)
    if ax is None:
        ax = plt.figure(figsize=(5, 4.5)).add_subplot(projection="3d")
    edge_kw = dict(edgecolor=EDGE_COLOR, linewidth=0.2) if edges else dict(linewidth=0)

    if scalars is None:
        surf = ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F,
                               color=SURFACE_COLOR, shade=True, **edge_kw)
    else:
        face_values = np.asarray(scalars, dtype=float)[F].mean(axis=1)
        lo = face_values.min() if vmin is None else vmin
        hi = face_values.max() if vmax is None else vmax
        if diverging:
            bound = max(abs(lo), abs(hi)) or 1.0
            norm, cmap = TwoSlopeNorm(0.0, -bound, bound), DIVERGING
        else:
            norm, cmap = Normalize(lo, hi), SEQUENTIAL
        surf = ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, **edge_kw)
        surf.set_array(face_values)
        surf.set_cmap(cmap)
        surf.set_norm(norm)
        if colorbar:
            ax.figure.colorbar(surf, ax=ax, shrink=0.6, pad=0.02)

    _set_equal_aspect(ax, V)
    ax.view_init(elev=elev, azim=azim)
    if axis_off:
        ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax, surf


def compare(meshes, titles, scalars=None, *, shared_scale=True, diverging=False,
            ncols=None, panel_size=(4.0, 4.0), views=None, **kwargs):
    """Plot several meshes side by side and return the figure.

    ``meshes`` is a list of ``(V, F)`` pairs and ``scalars`` an optional list of
    per-vertex arrays (``None`` entries are drawn plain). With ``shared_scale``
    all panels use the same color range and one shared colorbar, so colors can
    be compared across panels. ``views`` is an optional list of ``(elev, azim)``
    camera angles, one per panel (e.g. ``(90, -90)`` looks straight down on a
    planar mesh). Extra keyword arguments go to :func:`plot_mesh`.
    """
    n = len(meshes)
    scalars = [None] * n if scalars is None else list(scalars)
    views = [None] * n if views is None else list(views)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(panel_size[0] * ncols, panel_size[1] * nrows))

    colored = [s for s in scalars if s is not None]
    if shared_scale and colored:
        kwargs.setdefault("vmin", min(np.min(s) for s in colored))
        kwargs.setdefault("vmax", max(np.max(s) for s in colored))

    axes, mappable = [], None
    for k, ((V, F), s, title, view) in enumerate(zip(meshes, scalars, titles, views)):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d")
        camera = dict(zip(("elev", "azim"), view)) if view else {}
        _, surf = plot_mesh(V, F, s, ax, title=title, diverging=diverging,
                            colorbar=not shared_scale, **camera, **kwargs)
        axes.append(ax)
        if s is not None:
            mappable = surf
    if shared_scale and mappable is not None:
        fig.colorbar(mappable, ax=axes, shrink=0.6, pad=0.02)
    return fig


def show_polyscope(V, F, scalars=None, name="mesh"):
    """Open an interactive polyscope window (optional dependency).

    Install it with ``pip install -e ".[viewer]"``. If polyscope is missing,
    a message is printed and nothing else happens.
    """
    try:
        import polyscope as ps
    except ImportError:
        print('polyscope is not installed; run: pip install -e ".[viewer]"')
        return
    ps.init()
    mesh = ps.register_surface_mesh(name, np.asarray(V), np.asarray(F))
    if scalars is not None:
        mesh.add_scalar_quantity("scalars", np.asarray(scalars), enabled=True)
    ps.show()
