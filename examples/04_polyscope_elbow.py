"""Phase 4 example: view the two-pipe elbow of Fig. 4.8 in polyscope.

Needs the optional viewer: pip install -e ".[viewer]"
Run from the repository root:

    python examples/04_polyscope_elbow.py

Shows, side by side along x: the quarter-torus start and the results for
k = 1, 2, 3 (cotangent weights), with the fixed pipes in gray and the free bend
in blue. polyscope renders with smooth shading and highlights, so the shapes
can be compared with the book's figure more fairly than in the matplotlib PNGs.
Use the checkboxes in the left panel to show or hide each mesh. Close the
window (or press Esc) to return to the terminal.
"""

import numpy as np
import polyscope as ps

from fairing import mesh
from fairing.fairing import solve_fair

V, F, bend = mesh.pipe_elbow()
results = {"start: quarter torus": V}
for k in (1, 2, 3):
    results[f"k = {k}"] = solve_fair(V, F, bend, k)

# one color per face: blue if all three vertices are in the free bend, gray otherwise
face_colors = np.where(bend[F].all(axis=1)[:, None], [0.36, 0.38, 0.79], [0.85, 0.85, 0.83])

ps.init()
ps.set_up_dir("z_up")
for i, (name, W) in enumerate(results.items()):
    m = ps.register_surface_mesh(name, W + [6.0 * i, 0.0, 0.0], F, smooth_shade=True)
    m.add_color_quantity("fixed / free", face_colors, defined_on="faces", enabled=True)
ps.show()
