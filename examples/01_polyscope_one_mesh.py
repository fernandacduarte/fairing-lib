"""Phase 1 example: view one mesh in polyscope, colored by a per-vertex scalar.

Needs the optional viewer: pip install -e ".[viewer]"
Run from the repository root:

    python examples/01_polyscope_one_mesh.py

A window opens; close it (or press Esc) to return to the terminal.
"""

from fairing import mesh, viz

V, F = mesh.uv_sphere(16, 24)

# Any array with one value per vertex works as "scalars"; here, the height z.
viz.show_polyscope(V, F, scalars=V[:, 2], name="uv_sphere")
