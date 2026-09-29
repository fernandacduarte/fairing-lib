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
