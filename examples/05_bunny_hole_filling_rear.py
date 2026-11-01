"""Phase 5 example, complementary analysis: a second real hole, on the upper rear of the back.

Same method as examples/05_bunny_hole_filling.py, on a different, hand-outlined region: cut out
the region's faces and vertices, build a replacement patch from the hole's boundary loop alone
(plane fit, constrained Delaunay triangulation with Triangle, interior on the plane), then fair
the patch's interior with a thin plate (k = 2). That script is reused as is; only the region,
the cameras and the figure are new.

The region was outlined by hand (a pink lens) on a picture of the bunny, head left: the upper
rear of the back, up to the silhouette. How the outline became vertices, as for
examples/05_bunny_back_region.py: the picture's silhouette was matched to the mesh's, seen
orthographically from (elev, azim) = (10, -61) (IoU 0.90); the faces visible inside the outline
gave 766 vertices; and the ball (TARGET, RADIUS) whose region matches them best was kept (IoU 0.93
on the visible vertices, 97% of them covered). This one-off matching was done outside the
repository; the script stores only the ball. The region lies inside the first hole's region:
it is its upper rear part, about a third of its size.

With --larger, the same ball grows from radius 0.022 to 0.028 (1715 vertices instead of 1063,
still inside the first hole's region), to see how the result changes with a larger hole.

Run from the repository root (needs the full-resolution bunny, see README, and the `mesh` and
`hole` extras: pip install -e ".[mesh,hole]"):

    python examples/05_bunny_hole_filling_rear.py
    python examples/05_bunny_hole_filling_rear.py --larger

Saves docs/img/14-bunny-hole-filling-rear-stages-gray.png and
docs/img/14-bunny-hole-filling-rear-weights.png (with --larger:
docs/img/14-bunny-hole-filling-rear-large-stages-gray.png and
docs/img/14-bunny-hole-filling-rear-large-weights.png, plus a slide version of the latter,
docs/img/14-bunny-hole-filling-rear-large-weights-slide.png), and prints the counts, checks and
distances quoted in the phase-5 note.
Data: Stanford Computer Graphics Laboratory (Stanford 3D Scanning Repository).
"""

import argparse
import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

from fairing.fairing import fairing_matrix, solve_fair
from fairing.laplacian import mean_curvature

HERE = Path(__file__).resolve().parent
holes = runpy.run_path(str(HERE / "05_bunny_hole_filling.py"), run_name="hole_helpers")
book, helpers = holes["book"], holes["helpers"]
IMG, K = holes["IMG"], holes["K"]
GRAY, LILAC, ORANGE, TEAL, INK = holes["GRAY"], holes["LILAC"], holes["ORANGE"], holes["TEAL"], holes["INK"]
TARGET = np.array([0.0275, -0.0041, 0.1212])   # upper rear of the back (z-up coordinates)
RADIUS = 0.022                                 # covers 97% of the outlined vertices seen from (10, -61)
VIEW = holes["VIEW"]                           # the first hole's camera, so the two figures compare
CLOSE_VIEW = (40, -35)                         # close-ups: the region at an angle, its profile visible
CUT_STYLE = dict(color="#ffffff", linewidth=2.4, linestyle=(0, (0.1, 2.2)),   # the section's cut on the
                 dash_capstyle="round")                                         # slide's renderings: white dots
# per hole: the ball's radius, the close-ups' crop radius, the radius of the faces that the
# cross-section cuts (so it does not cut the far side of the bunny), and the two figures
VARIANTS = {
    "outlined": dict(radius=RADIUS, close=0.036, section=0.04, stages="14-bunny-hole-filling-rear-stages-gray.png",
                     weights="14-bunny-hole-filling-rear-weights.png", slide=None),
    "larger": dict(radius=0.028, close=0.042, section=0.046, stages="14-bunny-hole-filling-rear-large-stages-gray.png",
                   weights="14-bunny-hole-filling-rear-large-weights.png",
                   slide="14-bunny-hole-filling-rear-large-weights-slide.png"),
}


def main(variant):
    V, F = helpers["load_bunny"](book["FULL"])
    free, region_euler, clearance = book["book_region"](V, F, variant["radius"], TARGET)
    assert region_euler == 1 and clearance >= K + 1
    removed = F[free[F].any(axis=1)]

    # 1. the hole; from here on, only the surviving mesh builds and fairs the patch
    V_hole, F_hole, _, loop = holes["cut_hole"](V, F, free)
    # 2. the replacement patch, then 3. the two thin plates, from one initialized mesh
    V_init, F_new, free_new, info = holes["build_patch"](V_hole, F_hole, loop)
    results = {lap: solve_fair(V_init, F_new, free_new, K, laplacian=lap) for lap in ("uniform", "cotan")}

    # the removed region's positions never enter: poison them and rebuild everything
    poisoned = V.copy()
    poisoned[free] = np.nan
    V2, F2, _, loop2 = holes["cut_hole"](poisoned, F, free)
    V2_init, F2_new, free2, _ = holes["build_patch"](V2, F2, loop2)
    rebuilt = all(np.array_equal(a, b) for a, b in ((V2_init, V_init), (F2_new, F_new), (free2, free_new)))
    rebuilt &= all(np.array_equal(solve_fair(V2_init, F2_new, free2, K, laplacian=lap), W) for lap, W in results.items())
    assert rebuilt, "the patch or its fairing depends on the removed positions"

    h = info["h"]
    checks = holes["check_attachment"](V_hole, F_hole, loop, V_init, F_new, free_new, h)
    euler = holes["euler"]
    assert euler(F_new) == euler(F), "the filled mesh does not have the original's topology"
    patch = F_new[len(F_hole):]
    angles = holes["min_angles"](V_init, patch)
    flatness = np.abs((V_hole[loop] - info["center"]) @ info["normal"])
    first, *_ = book["book_region"](V, F, holes["back"]["RADIUS"], holes["back"]["TARGET"])

    print(f"ball: center {TARGET}, radius {variant['radius']}")
    print(f"full-resolution bunny: {len(V)} vertices, {len(F)} triangles, Euler characteristic {euler(F)}")
    print(f"removed region: {free.sum()} vertices, {len(removed)} triangles (a disk, at least {clearance} rings "
          f"from the bunny's other boundary edges); {(free & first).sum() / free.sum():.0%} of it inside the first "
          f"hole's region ({first.sum()} vertices)")
    print(f"surviving mesh: {len(V_hole)} vertices, {len(F_hole)} triangles; hole loop: {len(loop)} vertices; "
          f"{len(holes['boundary_loops'](F_hole)) - 1} other boundary loops unchanged")
    print(f"hole loop: mean edge h = {h:.5f}; distance from the fitted plane up to {flatness.max():.4f} "
          f"({flatness.max() / h:.1f} h), mean {flatness.mean() / h:.1f} h")
    print(f"patch: {free_new.sum()} new interior vertices, {len(patch)} triangles "
          f"(Triangle 'pq{holes['MIN_ANGLE']}Y', max area of an equilateral triangle of side h)")
    print(f"filled mesh: {len(V_init)} vertices, {len(F_new)} triangles, Euler characteristic {euler(F_new)}")
    print("checks on the initialized patch:")
    print("  built from the surviving mesh only: rebuilt identically with the removed positions set to NaN")
    print(f"  attached to the {len(loop)} loop vertices themselves (no copies; nearest new vertex {checks['gap']:.2f} h away); "
          f"no gaps, seams or non-manifold edges ({checks['open_edges']} open edges = the bunny's own)")
    print(f"  consistent orientation; all coordinates finite; smallest triangle area {checks['min_area']:.2f} "
          f"of an equilateral one of side h; smallest angle {angles.min():.1f} deg (median of the per-triangle "
          f"minima {np.median(angles):.1f} deg)")

    print(f"thin plate (k = {K}), {free_new.sum()} free vertices, weights frozen from the initialized mesh:")
    H = lambda W, Fm, mask: mean_curvature(W, Fm)[mask].mean()
    print(f"  original removed region: mean |H| {H(V, F, free):.1f}")
    dist = holes["surface_distances"](V_init, patch, V, removed)
    print(f"  initial planar patch: distance to the original surface mean {dist['mean']:.4f} ({dist['mean'] / h:.1f} h), "
          f"max {dist['max']:.4f}, other way max {dist['back_max']:.4f}; mean offset {dist['offset']:+.4f}")
    for lap, W in results.items():
        A = fairing_matrix(V_init, F_new, K, lap)                 # the matrix solve_fair froze
        residual = np.linalg.norm((A @ W)[free_new]) / np.linalg.norm(A[free_new][:, ~free_new] @ W[~free_new])
        fixed_same = np.array_equal(W[~free_new], V_init[~free_new])
        assert fixed_same and residual < 1e-10
        dist = holes["surface_distances"](W, patch, V, removed)
        print(f"  {lap:7s}: fixed vertices unchanged {fixed_same}; relative residual {residual:.1e}; "
              f"mean |H| {H(W, F_new, free_new):.1f}; distance to the original surface mean {dist['mean']:.4f} "
              f"({dist['mean'] / h:.1f} h), max {dist['max']:.4f}, other way max {dist['back_max']:.4f}; "
              f"mean offset {dist['offset']:+.4f}")
    gap = np.linalg.norm(results["uniform"] - results["cotan"], axis=1).max()
    print(f"  uniform vs cotangent patches: vertices differ by up to {gap:.4f} ({gap / h:.1f} h)")

    draw_stages(V, F, free, V_hole, F_hole, loop, V_init, F_new, results["cotan"], variant)
    draw_weights(V, F, free, V_hole, F_hole, V_init, F_new, results, info, variant)
    if variant["slide"]:
        draw_weights_slide(V, F, free, V_hole, F_hole, V_init, F_new, results, info, variant)


def draw_stages(V, F, free, V_hole, F_hole, loop, V_init, F_new, W_cotan, variant):
    """The four stages, laid out as docs/img/14-bunny-hole-filling-stages-gray.png: original with
    the region, the hole, the initial patch, the cotangent thin plate drawn all gray (as in the
    book's Fig. 4.7). Top row: the whole bunny. Bottom row: untitled close-ups, one frame."""
    render, colors = holes["render"], holes["colors"]
    region = free[F].any(axis=1)
    center = V[free].mean(axis=0)
    frame = V[np.unique(holes["crop"](V, F, center, variant["close"]))]
    n_hole = len(F_hole)
    stages = [
        (V, F, np.where(region[:, None], to_rgb(LILAC), to_rgb(GRAY)), None, None,
         "1. Original bunny,\nselected region highlighted"),
        (V_hole, F_hole, np.tile(to_rgb(GRAY), (n_hole, 1)), None, loop, "2. The actual hole\n(faces and vertices removed)"),
        (V_init, F_new, colors(len(F_new), n_hole), np.arange(len(F_new)) >= n_hole, None,
         "3. Initial patch: new triangulation,\ninterior on the fitted plane"),
        (W_cotan, F_new, colors(len(F_new), len(F_new)), None, None, "4. Faired patch, cotangent weights\n(thin plate, k = 2)"),
    ]
    fig = plt.figure(figsize=(16, 8.4), layout="constrained")
    for col, (X, T, c, e, l, title) in enumerate(stages):
        ink = TEAL if col == 3 else INK
        render(fig.add_subplot(2, 4, col + 1, projection="3d"), X, T, c, VIEW, X, title, ink, loop=l)
        close = (np.linalg.norm(X[T] - center, axis=2) < variant["close"]).any(axis=1)
        render(fig.add_subplot(2, 4, col + 5, projection="3d"), X, T[close], c[close], CLOSE_VIEW, frame, "", ink,
               edges=None if e is None else e[close], loop=l)
    fig.suptitle("Hole filling: construct a patch, then fair it", fontsize=17, fontweight="bold")
    fig.text(0.5, -0.01, "Patch construction supplies the triangulation; fairing computes a smooth shape for its interior."
             "\nGray: surviving bunny (fixed). Blue: replacement patch. Green: hole boundary. "
             "Panel 4 is drawn all gray, as in the book's Fig. 4.7.", ha="center", fontsize=11, color=INK)
    path = IMG / variant["stages"]
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def draw_weights(V, F, free, V_hole, F_hole, V_init, F_new, results, info, variant):
    """The shared initial patch, the uniform and the cotangent thin plates (k = 2), laid out as
    docs/img/14-bunny-hole-filling-weights.png. Row 1: close-ups at an angle (shape). Row 2: face
    on, with the triangles (sampling). Row 3: a cross-section, with the original for reference."""
    render, colors, crop, section = holes["render"], holes["colors"], holes["crop"], holes["section"]
    n_hole = len(F_hole)
    patch = F_new[n_hole:]
    new = np.unique(patch[patch >= len(V_hole)])
    center = V_init[new].mean(axis=0)
    face_on = holes["back"]["above"](results["cotan"], patch, np.isin(np.arange(len(V_init)), new))
    frame = V_init[np.unique(crop(V_init, F_new, center, variant["close"]))]
    panels = [(V_init, "Shared initial patch", INK),
              (results["uniform"], "Uniform weights, k = 2", ORANGE),
              (results["cotan"], "Cotangent weights, k = 2", TEAL)]
    fig = plt.figure(figsize=(13.5, 13.4), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.72])
    for col, (X, title, ink) in enumerate(panels):
        close = (np.linalg.norm(X[F_new] - center, axis=2) < variant["close"]).any(axis=1)
        T = F_new[close]
        c = colors(len(F_new), n_hole)[close]
        edges = (np.arange(len(F_new)) >= n_hole)[close]
        render(fig.add_subplot(grid[0, col], projection="3d"), X, T, c, CLOSE_VIEW, frame, title, ink)
        render(fig.add_subplot(grid[1, col], projection="3d"), X, T, c, face_on, frame, "face on, triangles shown", ink,
               edges=edges)

    # the cross-section: the plane through the loop's centroid, spanned by the fitted normal
    # and the loop's longest direction; only faces near the hole, so the far side is not cut
    ax = fig.add_subplot(grid[2, :])
    cut = lambda X, T, **kw: LineCollection(section(X, crop(X, T, info["center"], variant["section"]), info["center"],
                                                    info["e1"], info["normal"], info["e2"]), **kw)
    removed = F[free[F].any(axis=1)]
    ax.add_collection(cut(V, removed, color=LILAC, linewidth=5, label="original surface (removed)"))
    ax.add_collection(cut(V_hole, F_hole, color="#8a8985", linewidth=1.6, label="surviving surface (fixed)"))
    for X, label, ink in ((V_init, "initial patch (fitted plane)", INK), (results["uniform"], "uniform, k = 2", ORANGE),
                          (results["cotan"], "cotangent, k = 2", TEAL)):
        ax.add_collection(cut(X, patch, color=ink, linewidth=1.8, label=label))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("along the hole (fitted plane)")
    ax.set_ylabel("height above\nthe fitted plane")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", ncols=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.45))
    ax.set_title("Cross-section through the patch (true scale)", fontsize=11, color=INK)

    fig.suptitle("Same initial patch, different weights", fontsize=17, fontweight="bold")
    fig.text(0.5, -0.01, "Both solves use the same initialized mesh, connectivity and free vertices; "
             "only the frozen Laplacian weights differ.", ha="center", fontsize=11, color=INK)
    path = IMG / variant["weights"]
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def draw_weights_slide(V, F, free, V_hole, F_hole, V_init, F_new, results, info, variant, cut_style=None):
    """The weights comparison laid out for a landscape slide: the three oblique close-ups on top,
    each with the cross-section's cut drawn as a thin line, and the cross-section enlarged below.
    The face-on wireframes of draw_weights are left out."""
    render, colors, crop, section = holes["render"], holes["colors"], holes["crop"], holes["section"]
    n_hole = len(F_hole)
    patch = F_new[n_hole:]
    new = np.unique(patch[patch >= len(V_hole)])
    center = V_init[new].mean(axis=0)
    frame = V_init[np.unique(crop(V_init, F_new, center, variant["close"]))]
    c0, e1, up, across = info["center"], info["e1"], info["normal"], info["e2"]
    cut2d = lambda X, T, radius=variant["section"]: section(X, crop(X, T, c0, radius), c0, e1, up, across)
    lift = lambda segments: c0 + segments[..., :1] * e1 + segments[..., 1:] * up      # back to 3D

    fig = plt.figure(figsize=(16, 9), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 0.92], hspace=0.02)
    panels = [(V_init, "Initial patch", INK), (results["uniform"], "Uniform · k = 2", ORANGE),
              (results["cotan"], "Cotangent · k = 2", TEAL)]
    for col, (X, title, ink) in enumerate(panels):
        ax = top.add_subplot(1, 3, col + 1, projection="3d")
        close = (np.linalg.norm(X[F_new] - center, axis=2) < variant["close"]).any(axis=1)
        render(ax, X, F_new[close], colors(len(F_new), n_hole)[close], CLOSE_VIEW, frame, title, ink)
        ax.title.set_fontsize(15)
        ax.title.set_fontweight("bold")
        ax.computed_zorder = False                            # the cut on top of the surface
        ends = cut2d(X, F_new, variant["close"]).reshape(-1, 2)
        curve = lift(np.unique(ends, axis=0))                 # one polyline, ordered along the section,
        ax.plot(*curve.T, zorder=20, **(cut_style or CUT_STYLE))     # so that dash patterns show
    top.suptitle("Same initial patch, different weights", fontsize=22, fontweight="bold")
    top.supxlabel("Same initial positions, connectivity, free vertices, and fixed surrounding surface.  "
                  "White dotted line: where the cross-section below is taken.", fontsize=12, color=INK)

    ax = bottom.add_subplot()
    ax.add_collection(LineCollection(cut2d(V, F[free[F].any(axis=1)]), color=LILAC, linewidth=6,
                                     label="Original surface — reference only"))
    ax.add_collection(LineCollection(cut2d(V_hole, F_hole), color="#8a8985", linewidth=2, label="Fixed surrounding surface"))
    for X, label, ink in ((V_init, "Initial patch (fitted plane)", INK), (results["uniform"], "Uniform · k = 2", ORANGE),
                          (results["cotan"], "Cotangent · k = 2", TEAL)):
        ax.add_collection(LineCollection(cut2d(X, patch), color=ink, linewidth=2.4, label=label))
    ax.autoscale()
    ax.set_aspect("equal")                                    # equal scales: the profile at true scale
    ax.set_xlabel("Position along the section (model units)", fontsize=13)
    ax.set_ylabel("Height above the fitted plane\n(model units)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Cross-section through the patch", fontsize=15, fontweight="bold", color=INK)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=12.5)
    bottom.supxlabel("Equal horizontal and vertical scales.  The original surface is drawn for reference only: "
                     "it was removed and is never used to build or solve the patch.", fontsize=12.5, color=INK)
    fig.supxlabel("Both weightings produce a curved fill, but different profiles — "
                  "even with the same initial patch and constraints.", fontsize=14, fontweight="bold", color=INK)
    path = IMG / variant["slide"]
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--larger", action="store_true", help="a larger ball around the same center (radius 0.028)")
    main(VARIANTS["larger" if parser.parse_args().larger else "outlined"])
