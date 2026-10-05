"""Phase 5 example: the book-like hole of examples/05_bunny_book_region.py, in polyscope.

Needs the optional viewer and the full-resolution bunny (see README, "The Stanford bunny"):

    pip install -e ".[viewer,mesh]"
    python examples/05_polyscope_bunny_book.py

Opens on the full-resolution bunny from the camera angle of the book's Fig. 4.7 (head left,
haunch right), and a "Fig. 4.7" panel (top right) to change:
  - Show: the hole (the region's triangles removed, as on the figure's left), the damaged
    input, the refill (as on the figure's right) or the original.
  - Region: the hole's radius, around the same haunch vertex as the PNG (0.035 there).
    The region is re-built when you release the slider.
  - Damage, order k and weights, as in examples/05_polyscope_bunny_explorer.py.
  - Color: plain gray (as in the book), the free region in blue, or mean curvature.
The green points are the region's border (the first fixed ring), as in the book.
Close the window (or press Esc) to return to the terminal.
"""

import runpy
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from fairing import mesh
from fairing.fairing import solve_fair
from fairing.laplacian import mean_curvature

# --- the full-resolution bunny and the helpers of the PNG script ---------------------------
book = runpy.run_path(str(Path(__file__).with_name("05_bunny_book_region.py")), run_name="book_helpers")
V0, F = book["helpers"]["load_bunny"](book["FULL"])          # z-up, unused vertices dropped
H_MAX = 120                                                  # curvature color scale, as in #14's PNGs
GRAY, BLUE, GREEN = [0.85, 0.85, 0.83], [0.36, 0.38, 0.79], [0.23, 0.65, 0.33]

state = {"show": "refill", "radius": book["RADIUS"], "damage": "flatten", "k": 2, "weights": "cotan",
         "color": "plain", "border": True}
current = {}                                                 # region, start and refill being shown
info = {}                                                    # numbers shown in the panel


def solve():
    """Free region, damaged start and refill for the current settings."""
    free, euler, clearance = book["book_region"](V0, F, state["radius"])
    damaged, _ = book["helpers"]["damage"](V0, F, free)
    start = damaged.get(state["damage"], V0)                 # "none": start from the original
    result = solve_fair(start, F, free, state["k"], state["weights"])
    border = mesh.ring_distance(F, np.flatnonzero(free), max_k=1, n=len(V0)) == 1
    current.update(free=free, start=start, result=result, border=border)
    H = lambda W: mean_curvature(W, F)[free].mean()
    info.update(free=int(free.sum()), disk=(euler == 1), clearance=clearance, needed=state["k"] + 1,
                H_original=H(V0), H_start=H(start), H_result=H(result),
                distance=np.abs(result - V0)[free].max())


def paint():
    """Push the current geometry, colors and visibility to polyscope."""
    free = current["free"]
    shown = {"hole": ps.register_surface_mesh("the hole (Fig. 4.7, left)", V0, F[~free[F].any(axis=1)],
                                              color=GRAY, smooth_shade=True, back_face_policy="different")}
    for key, W in (("damaged", current["start"]), ("refill", current["result"]), ("original", V0)):
        m = surfaces[key]
        m.update_vertex_positions(W)
        m.add_color_quantity("free region", np.where(free[:, None], BLUE, GRAY),
                             enabled=(state["color"] == "region"))
        m.add_scalar_quantity("mean curvature H", mean_curvature(W, F), cmap="blues", vminmax=(0, H_MAX),
                              enabled=(state["color"] == "curvature"))
        shown[key] = m
    for key, m in shown.items():
        m.set_enabled(key == state["show"])
    ps.register_point_cloud("region border", V0[current["border"]], radius=0.0012, color=GREEN,
                            enabled=state["border"])


def book_view():
    """The camera of examples/05_bunny_book_region.py (matplotlib's elev, azim), whole bunny."""
    elev, azim = np.radians(book["VIEW"])
    direction = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    center = (V0.min(axis=0) + V0.max(axis=0)) / 2
    ps.look_at(center + 0.35 * direction, center)


def close_up():
    """The camera on the hole, from outside the bunny."""
    free = current["free"]
    faces = F[free[F].all(axis=1)]
    normal = np.cross(V0[faces[:, 1]] - V0[faces[:, 0]], V0[faces[:, 2]] - V0[faces[:, 0]]).sum(axis=0)
    center = V0[free].mean(axis=0)
    ps.look_at(center + 0.15 * normal / np.linalg.norm(normal), center)


def radio_row(label, key, options):
    """A row of radio buttons; returns True if the selection changed."""
    changed = False
    psim.TextUnformatted(label)
    for i, option in enumerate(options):
        if i:
            psim.SameLine()
        if psim.RadioButton(f"{option}##{key}", state[key] == option):
            changed = changed or state[key] != option
            state[key] = option
    return changed


def panel():
    """The 'Fig. 4.7' panel, drawn every frame."""
    psim.PushItemWidth(160)
    repaint = radio_row("SHOW", "show", ("hole", "damaged", "refill", "original"))
    psim.Separator()

    psim.TextUnformatted("REGION")
    _, state["radius"] = psim.SliderFloat("radius around the haunch", state["radius"], 0.015, 0.045, "%.3f")
    resolve = psim.IsItemDeactivatedAfterEdit()              # re-build only when the slider is released
    resolve |= radio_row("DAMAGE (start of the free vertices)", "damage", ("none", "noise", "flatten"))
    resolve |= radio_row("ORDER k", "k", (1, 2, 3))
    resolve |= radio_row("WEIGHTS", "weights", ("uniform", "cotan"))
    psim.Separator()

    repaint |= radio_row("COLOR", "color", ("plain", "region", "curvature"))
    c, state["border"] = psim.Checkbox("green border points", state["border"])
    repaint |= c
    if psim.Button("book's view"):
        book_view()
    psim.SameLine()
    if psim.Button("close-up on the hole"):
        close_up()
    psim.Separator()

    if resolve:
        solve()
    if resolve or repaint:
        paint()
    psim.TextUnformatted(f"free vertices: {info['free']} (radius {state['radius']:.3f})")
    psim.TextUnformatted(f"distance to boundary edges: {info['clearance']} rings (k + 1 = {info['needed']} needed)")
    if not info["disk"] or info["clearance"] < info["needed"]:
        psim.TextUnformatted("WARNING: the region is not a disk away from the boundary")
    psim.TextUnformatted(f"mean |H| on the region: original {info['H_original']:.0f}, "
                         f"start {info['H_start']:.0f}, refill {info['H_result']:.0f}")
    psim.TextUnformatted(f"max distance to the original: {info['distance']:.4f}")
    psim.PopItemWidth()


ps.init()
ps.set_up_dir("z_up")
ps.set_ground_plane_mode("none")
surfaces = {key: ps.register_surface_mesh(name, V0, F, smooth_shade=True, color=GRAY)
            for key, name in (("damaged", "damaged input"), ("refill", "refill (Fig. 4.7, right)"),
                              ("original", "original"))}
solve()
paint()
book_view()
ps.set_user_callback(panel)
ps.show()
