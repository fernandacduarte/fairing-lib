"""Phase 4 example: the Fig. 4.9 setting of issue #12 in polyscope.

Needs the optional viewer: pip install -e ".[viewer]"
Run from the repository root:

    python examples/04_polyscope_fig49.py

A graded grid (vertex density varies in bands) with heights z = x^2 - y^2; the disk of
radius 0.3 is free and is refilled by a thin plate (k = 2). Side by side along x:

    input mesh | exact surface | thin plate, uniform weights | thin plate, cotangent weights

Quantities (left panel, per mesh; tick one at a time):
  - "mean curvature H": the same color scale (0 to 0.4) as docs/img/12-uniform-vs-cotan-fairing.png
  - "error |z - (x^2 - y^2)|": distance to the exact surface, scale 0 to 0.01 (the cotangent
    error, about 1e-4, looks almost white; the uniform one reaches 0.01)
  - "free region": the free disk (1) and the fixed vertices (0)
  - "sideways slide" (uniform only): arrows from each vertex's start to where it ended, in x and y
Edges are drawn so the density bands are visible; turn them off under a mesh's options.
Close the window (or press Esc) to return to the terminal.
"""

import numpy as np
import polyscope as ps

from fairing import mesh
from fairing.fairing import solve_fair
from fairing.laplacian import mean_curvature

saddle = lambda x, y: x**2 - y**2

V, F = mesh.graded_grid(41, 41, strength=0.7)
V[:, 2] = saddle(V[:, 0], V[:, 1])
free = np.linalg.norm(V[:, :2], axis=1) < 0.3

meshes = {
    "1 input mesh": V,
    "2 exact surface x^2 - y^2": V,
    "3 thin plate, uniform weights": solve_fair(V, F, free, 2, "uniform"),
    "4 thin plate, cotangent weights": solve_fair(V, F, free, 2, "cotan"),
}

ps.init()
ps.set_up_dir("z_up")
for i, (name, W) in enumerate(meshes.items()):
    m = ps.register_surface_mesh(name, W + [1.3 * i, 0.0, 0.0], F, smooth_shade=True,
                                 edge_width=1.0 if i == 0 else 0.0, color=[0.94, 0.94, 0.92])
    m.add_scalar_quantity("free region", free.astype(float), cmap="blues", vminmax=(0, 1.5), enabled=(i == 0))
    if i == 0:
        continue
    H = mean_curvature(W, F)
    m.add_scalar_quantity("mean curvature H", H, cmap="blues", vminmax=(0, 0.4), enabled=True)
    error = np.abs(W[:, 2] - saddle(W[:, 0], W[:, 1]))
    m.add_scalar_quantity("error |z - (x^2 - y^2)|", error, cmap="reds", vminmax=(0, 0.01))
    if "uniform" in name:
        slide = (W - V) * [1, 1, 0]                                # sideways part of the motion
        m.add_vector_quantity("sideways slide", slide, vectortype="ambient", radius=0.002, color=[0.92, 0.41, 0.2])
ps.show()
