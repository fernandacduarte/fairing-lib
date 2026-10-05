"""Phase 5 example: an interactive fairing explorer on the Stanford bunny (polyscope).

Needs the optional viewer and the downloaded bunny (see README, "The Stanford bunny"):

    pip install -e ".[viewer,mesh]"
    python examples/05_polyscope_bunny_explorer.py

One bunny, with the camera on the free region, and a "Fairing" panel (top right) to change:
  - Region: how many rings around the seed vertex are free.
  - Damage: how the free vertices start (none, noise, flattened onto the border's plane).
  - Fairing: the order k (1 membrane, 2 thin plate, 3 minimum variation) and the weights
    (uniform or cotangent, frozen from the damaged input as in the book's linear method).
  - Display: color by mean curvature or by region, the original as a transparent ghost,
    the damaged input, edges.
Every change re-solves at once (a few milliseconds) and updates the numbers at the bottom
of the panel. The orange points are the region's border (the first fixed ring).
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

# --- the bunny and its fixed facts ---------------------------------------------------------
helpers = runpy.run_path(str(Path(__file__).with_name("05_bunny.py")), run_name="bunny_helpers")
V0, F = helpers["load_bunny"]()                              # z-up, unused vertices dropped
E, counts = mesh.edge_face_counts(F)
DIST_TO_PROBLEM = mesh.ring_distance(F, np.unique(E[counts != 2]), n=len(V0))
SEED = int(np.argmax(DIST_TO_PROBLEM))                       # the vertex farthest from problem edges
EDGE = np.linalg.norm(V0[E[:, 0]] - V0[E[:, 1]], axis=1).mean()
H_MAX = 120                                                  # curvature color scale, as in the PNG

state = {"rings": 6, "damage": "flatten", "noise": 0.5, "k": 2, "weights": "cotan",
         "color": "curvature", "ghost": False, "damaged": False, "edges": False}
info = {}                                                    # numbers shown in the panel


def compute(s):
    """Free region, damaged start and fairing result for the current settings."""
    free = np.zeros(len(V0), dtype=bool)
    free[mesh.k_ring(F, [SEED], s["rings"], n=len(V0))] = True
    if s["damage"] == "noise":
        start = np.where(free[:, None], mesh.add_noise(V0, s["noise"] * EDGE, seed=0), V0)
    elif s["damage"] == "flatten":
        start = mesh.flatten_onto_border_plane(V0, F, free)
    else:
        start = V0.copy()
    result = solve_fair(start, F, free, s["k"], s["weights"])
    border = mesh.ring_distance(F, np.flatnonzero(free), max_k=1, n=len(V0)) == 1
    return free, start, result, border


def refresh():
    """Re-solve and push the new geometry and quantities to polyscope."""
    free, start, result, border = compute(state)
    H = lambda W: mean_curvature(W, F)
    for m, W in ((result_mesh, result), (damaged_mesh, start)):
        m.update_vertex_positions(W)
        m.add_scalar_quantity("free region", free.astype(float), cmap="blues", vminmax=(0, 1.5),
                              enabled=(state["color"] == "region"))
        m.add_scalar_quantity("mean curvature H", H(W), cmap="blues", vminmax=(0, H_MAX),
                              enabled=(state["color"] == "curvature"))
    ps.register_point_cloud("region border", result[border], radius=0.0012, color=[0.92, 0.41, 0.2])
    clearance = int(DIST_TO_PROBLEM[free].min())
    info.update(
        free=int(free.sum()), clearance=clearance, needed=state["k"] + 1,
        H_original=H(V0)[free].mean(), H_start=H(start)[free].mean(), H_result=H(result)[free].mean(),
        distance=np.abs(result - V0)[free].max())


def focus_on_region():
    """Point the camera at the free region, from outside the bunny."""
    free = np.zeros(len(V0), dtype=bool)
    free[mesh.k_ring(F, [SEED], state["rings"], n=len(V0))] = True
    faces = F[free[F].all(axis=1)]
    normal = np.cross(V0[faces[:, 1]] - V0[faces[:, 0]], V0[faces[:, 2]] - V0[faces[:, 0]]).sum(axis=0)
    normal /= np.linalg.norm(normal)
    center = V0[free].mean(axis=0)
    ps.look_at(center + 0.09 * normal, center)


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
    """The 'Fairing' panel, drawn every frame."""
    changed = False
    psim.PushItemWidth(160)

    psim.TextUnformatted("REGION")
    c, state["rings"] = psim.SliderInt("rings around the seed", state["rings"], 2, 10)
    changed |= c
    psim.Separator()

    changed |= radio_row("DAMAGE (start of the free vertices)", "damage", ("none", "noise", "flatten"))
    if state["damage"] == "noise":
        c, state["noise"] = psim.SliderFloat("noise (in edge lengths)", state["noise"], 0.0, 1.0)
        changed |= c
    psim.Separator()

    changed |= radio_row("ORDER k", "k", (1, 2, 3))
    changed |= radio_row("WEIGHTS", "weights", ("uniform", "cotan"))
    psim.Separator()

    if radio_row("COLOR", "color", ("curvature", "region")):
        changed = True
    for key, label in (("ghost", "original as a transparent ghost"), ("damaged", "show the damaged input"),
                       ("edges", "show edges")):
        c, state[key] = psim.Checkbox(label, state[key])
        if c:
            original_mesh.set_enabled(state["ghost"])
            damaged_mesh.set_enabled(state["damaged"])
            result_mesh.set_enabled(not state["damaged"])
            for m in (result_mesh, damaged_mesh):
                m.set_edge_width(1.0 if state["edges"] else 0.0)
    if psim.Button("focus on the region"):
        focus_on_region()
    psim.SameLine()
    if psim.Button("whole bunny"):
        ps.reset_camera_to_home_view()
    psim.Separator()

    if changed:
        refresh()
    psim.TextUnformatted(f"free vertices: {info['free']}  ({state['rings']} rings)")
    psim.TextUnformatted(f"distance to problem edges: {info['clearance']} rings (k + 1 = {info['needed']} needed)")
    if info["clearance"] < info["needed"]:
        psim.TextUnformatted("WARNING: the region reaches non-manifold or boundary edges")
    psim.TextUnformatted(f"mean |H| on the region: original {info['H_original']:.0f}, "
                         f"start {info['H_start']:.0f}, result {info['H_result']:.0f}")
    psim.TextUnformatted(f"max distance to the original: {info['distance']:.4f} "
                         f"({info['distance'] / EDGE:.1f} edges)")
    psim.PopItemWidth()


ps.init()
ps.set_up_dir("z_up")
ps.set_ground_plane_mode("none")
original_mesh = ps.register_surface_mesh("original (ghost)", V0, F, smooth_shade=True, color=[0.8, 0.8, 0.78],
                                         transparency=0.35, enabled=False)
damaged_mesh = ps.register_surface_mesh("damaged input", V0, F, smooth_shade=True, enabled=False)
result_mesh = ps.register_surface_mesh("fairing result", V0, F, smooth_shade=True)
refresh()
focus_on_region()
ps.set_user_callback(panel)
ps.show()
