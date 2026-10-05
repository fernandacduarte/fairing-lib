"""Phase 5 example: the bunny refills of issue #14 in polyscope.

Needs the optional viewer and the downloaded bunny (see README, "The Stanford bunny"):

    pip install -e ".[viewer,mesh]"
    python examples/05_polyscope_bunny.py

Shows, side by side: the original bunny, the two damaged inputs (noise, flattened) and the
thin-plate refills (k = 2) from each, with uniform and with cotangent weights. Each mesh has
two quantities in the left panel: "free region" (the refilled vertices) and "mean curvature H"
(same 0-120 scale as docs/img/14-bunny-before-after.png). Zoom into the region on the bunny's
side to compare the refills. Close the window (or press Esc) to return to the terminal.
"""

import runpy
from pathlib import Path

import polyscope as ps

from fairing.fairing import solve_fair
from fairing.laplacian import mean_curvature

# reuse the loading, region and damage helpers of the PNG script
bunny = runpy.run_path(str(Path(__file__).with_name("05_bunny.py")), run_name="bunny_helpers")
V, F = bunny["load_bunny"]()
free, _, _ = bunny["choose_region"](V, F)
damaged, _ = bunny["damage"](V, F, free)

meshes = {"original": V}
for d, D in damaged.items():
    meshes[f"damaged: {d}"] = D
    for lap in ("uniform", "cotan"):
        meshes[f"{lap} refill from {d}"] = solve_fair(D, F, free, 2, lap)

ps.init()
ps.set_up_dir("z_up")
for i, (name, W) in enumerate(meshes.items()):
    m = ps.register_surface_mesh(name, W + [0.2 * i, 0.0, 0.0], F, smooth_shade=True)
    m.add_scalar_quantity("free region", free.astype(float), cmap="blues", vminmax=(0, 1.5), enabled=True)
    m.add_scalar_quantity("mean curvature H", mean_curvature(W, F), cmap="blues", vminmax=(0, 120))
ps.show()
