"""Phase 5 example: the book-like hole refilled from the flattened start, in polyscope.

Needs the optional viewer and the full-resolution bunny (see README, "The Stanford bunny"):

    pip install -e ".[viewer,mesh]"
    python examples/05_polyscope_bunny_flatten.py

Shows three bunnies side by side, seen from the camera angle of the book's Fig. 4.7, with the
free region in blue and the fixed vertices in gray:
  1. the flattened input: the free vertices projected onto the plane of the region's border;
  2. the thin-plate refill (k = 2) with uniform weights;
  3. the thin-plate refill (k = 2) with cotangent weights, frozen from the flattened input.
The region is built like the one of examples/05_bunny_book_region.py, but on the upper back
instead of the side of the haunch, so it is seen at a grazing angle. The camera is orthographic,
so all three bunnies are seen from the same direction. The panel (top right) repeats the labels,
left to right, with mean |H| on the region and the largest distance to the original. Close the
window (or press Esc) to return to the terminal.
"""

import runpy
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from fairing.fairing import solve_fair
from fairing.laplacian import mean_curvature

# --- the full-resolution bunny and the helpers of the PNG script ---------------------------
book = runpy.run_path(str(Path(__file__).with_name("05_bunny_book_region.py")), run_name="book_helpers")
V0, F = book["helpers"]["load_bunny"](book["FULL"])          # z-up, unused vertices dropped
K = book["K"]                                                # thin plate, k = 2
GRAY, BLUE = [0.85, 0.85, 0.83], [0.36, 0.38, 0.79]
TARGET = np.array([0.016, -0.034, 0.121])                    # a point on the upper back
RADIUS = 0.03

free, euler, clearance = book["book_region"](V0, F, RADIUS, TARGET)
assert euler == 1 and clearance >= K + 1, "the region must be a disk, k + 1 rings from boundary edges"
flat = book["helpers"]["damage"](V0, F, free)[0]["flatten"]
panels = {
    "Flattened input": flat,
    "Uniform result": solve_fair(flat, F, free, K, "uniform"),
    "Cotangent result": solve_fair(flat, F, free, K, "cotan"),
}

# numbers for the panel: mean |H| on the region and the largest distance to the original
H = lambda W: mean_curvature(W, F)[free].mean()
numbers = {name: (H(W), np.abs(W - V0)[free].max()) for name, W in panels.items()}

# the book's camera (matplotlib's elev, azim), and the screen's "right" direction for that
# camera, so the three bunnies line up from left to right in the picture
elev, azim = np.radians(book["VIEW"])
toward_camera = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
right = np.cross(-toward_camera, [0.0, 0.0, 1.0])
right /= np.linalg.norm(right)
spacing = 1.25 * np.ptp(V0, axis=0).max()                    # a little more than one bunny

# one color per face: blue if it has at least one free vertex (it moved in the solve), gray if
# all three are fixed, so the color boundary is the last fixed ring (as in the elbow viewer)
face_colors = np.where(free[F].any(axis=1)[:, None], BLUE, GRAY)


def panel():
    """The labels, left to right, with their numbers."""
    psim.TextUnformatted(f"Thin plate (k = {K}), {free.sum()} free vertices, left to right:")
    psim.Separator()
    for name, (h, d) in numbers.items():
        psim.TextUnformatted(f"{name:17s} mean |H| {h:5.1f}   max distance to original {d:.4f}")
    psim.Separator()
    psim.TextUnformatted(f"original bunny:   mean |H| {H(V0):5.1f}")


ps.init()
ps.set_up_dir("z_up")
ps.set_ground_plane_mode("none")
ps.set_view_projection_mode("orthographic")                  # all three bunnies seen from the same angle
for i, (name, W) in enumerate(panels.items()):
    m = ps.register_surface_mesh(f"{i + 1}. {name}", W + (i - 1) * spacing * right, F, smooth_shade=True)
    m.add_color_quantity("fixed / free", face_colors, defined_on="faces", enabled=True)
center = (V0.min(axis=0) + V0.max(axis=0)) / 2               # the middle bunny sits at the original place
ps.look_at(center + 0.5 * toward_camera, center)
ps.set_user_callback(panel)
ps.show()
