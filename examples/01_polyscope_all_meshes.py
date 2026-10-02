"""Phase 1 example: view all four synthetic meshes together in polyscope.

Needs the optional viewer: pip install -e ".[viewer]"
Run from the repository root:

    python examples/01_polyscope_all_meshes.py

Each mesh is shifted along x so that they sit side by side instead of
overlapping at the origin. Use the checkboxes in the left panel to show or
hide each mesh. Close the window (or press Esc) to return to the terminal.
"""

import polyscope as ps

from fairing import mesh

meshes = {
    "grid": mesh.grid(10, 10),
    "irregular_grid": mesh.irregular_grid(10, 10, jitter=0.15, seed=0),
    "uv_sphere": mesh.uv_sphere(12, 20, radius=0.5),
    "tube": mesh.tube(20, 10, radius=0.5, height=1.0),
}

ps.init()
for k, (name, (V, F)) in enumerate(meshes.items()):
    V = V + [1.5 * k, 0.0, 0.0]          # side by side, 1.5 units apart
    ps_mesh = ps.register_surface_mesh(name, V, F)
    ps_mesh.set_edge_width(1.0)           # show the triangles
ps.show()
