# Phase 1 — Foundations

This phase builds the tools that every later experiment relies on: a package, synthetic meshes, plotting, and connectivity queries. No book equation is implemented here, but the meshes are chosen for the experiments in Chapters 3–4.

## #1 Project setup

**Theory → code.** No equations. The package layout mirrors the book's progression:

| Concept | Module |
|---|---|
| meshes and their connectivity | `fairing/mesh.py` |
| discrete Laplace–Beltrami, `L = D·M` (§3.3.4, App. A.1) | `fairing/laplacian.py` |
| diffusion flow (§4.2) | `fairing/smoothing.py` |
| fairing, `Lᵏx = 0` (§4.3, Eq. A.2) | `fairing/fairing.py` |
| figures | `fairing/viz.py` |

**What the figures show.** None yet.

**Pitfalls.** None so far.

## #2 Synthetic meshes and OBJ I/O

**Theory → code.** No equation is implemented; each mesh exists to test one claim of the book later on.

| Mesh | Function | Later test | Book |
|---|---|---|---|
| regular grid | `grid` in `fairing/mesh.py:39` | uniform Laplacian is exactly 0 on it (baseline) | Eq. 3.10 |
| irregular planar grid | `irregular_grid`, `mesh.py:50` | planar ⇒ true Δx = 0; uniform ≠ 0, cotangent = 0 | Eq. 3.7, 3.10, 3.11 |
| UV sphere, radius R | `uv_sphere`, `mesh.py:74` | H = ½‖Δx‖ ≈ 1/R | Eq. 3.13 |
| open tube | `tube`, `mesh.py:98` | blend surfaces for k = 1, 2, 3 | Fig. 4.8 |
| shared triangulation | `_quad_faces`, `mesh.py:17` | one diagonal ⇒ interior valence 6 | — |

`save_obj` and `load_obj` (`mesh.py:115`, `mesh.py:124`) read and write triangle meshes in OBJ format (indices are 1-based in the file, 0-based in `F`).

**What the figures show.** No figure yet; the first one (all four meshes) comes in #3.

**Pitfalls.**
- *The diagonal direction matters.* If quads were split with alternating diagonals, interior valences would alternate between 4 and 8 instead of all being 6, so the ring sizes would vary from vertex to vertex. With a single diagonal the grid is a hexagonal lattice: every interior neighborhood looks the same, and every neighbor has a mirror partner through the center vertex.
- *Jitter can flip triangles.* If each vertex moves by at most δ times the spacing per coordinate, the doubled area of a triangle with legs h is at least h²(1 − 6δ). So δ < 1/6 is safe. This bound is conservative: over 200 random seeds at δ = 0.1666, the smallest doubled area was still 0.39·h².
- *Orientation is easy to get backwards.* On the sphere, the direction "east × south" points *inward*. That is why the rings are stored from south to north, so that "east × north" points outward. A test checks every triangle normal on every generator.

## #3 Visualization

**Theory → code.** No equations. The helpers in `fairing/viz.py` produce the color-coded pictures the book uses to judge its operators (e.g. mean curvature in Fig. 3.6 and Fig. 4.5):

| Function | What it does |
|---|---|
| `plot_mesh`, `viz.py:35` | one mesh with `plot_trisurf`; per-vertex scalars become face colors (mean of the 3 vertices) |
| `compare`, `viz.py:76` | panels side by side, one shared color scale and colorbar, optional camera per panel |
| `show_polyscope`, `viz.py:113` | interactive viewer; polyscope is imported only inside this function |
| `SEQUENTIAL`, `DIVERGING`, `viz.py:14` | blue ramp for magnitudes; blue–gray–red for signed values, with gray at 0 |

To inspect a mesh interactively, run `examples/01_polyscope_one_mesh.py` (one mesh colored by a scalar) or `examples/01_polyscope_all_meshes.py` (all four meshes side by side). The README explains how to install and use the viewer.

**What the figures show.**

![The four synthetic meshes](../img/03-meshes.png)

`docs/img/03-meshes.png` shows the four generators. Compare the two grids, which are seen from above: they have the same connectivity, but the interior vertices of the irregular one are shifted. Every quad is split along the same diagonal, which is what gives interior vertices valence 6. The sphere shows its pole fans of thin triangles; the tube is open at both ends.

**Pitfalls.**
- *Planar meshes seen from the side hide everything.* At the default camera angle the jitter was invisible, which is why `compare` takes a per-panel `views` argument.
- *Colored surfaces are unlit.* Once `plot_trisurf` colors faces by a scalar, it no longer shades them by their orientation, so shape cues vanish. For judging shape, look at a plain mesh or use polyscope.
- *Unequal axis scaling distorts shapes.* matplotlib 3D axes are not equal by default, so a sphere would look like an ellipsoid; `_set_equal_aspect` fixes the limits and the box aspect.
