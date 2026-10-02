# Progress

## Status
- [x] #1 Project setup: PR #17
- [x] #2 Synthetic meshes and OBJ I/O: PR #18
- [x] #3 Visualization helpers: PR #19
- [x] #4 Topology helpers: PR #20 (Phase 1 complete)
- [x] #5 Uniform Laplacian: PR #21

## Next
- #6 Cotangent Laplacian with barycentric areas, Eq. 3.11 (Phase 2)

## Decisions
- **Issue numbers = plan step numbers.** All issues were created before any PR, so issue #N is step N of PLAN.md §3 (M1 is #16).
- **Topology helpers live in `fairing/mesh.py`**, as in the PLAN.md §1 layout; there is no separate module.
- **Figures are named `docs/img/<NN>-<name>.png`** with a zero-padded issue number, so they sort in order.
- **Local environment:** `.venv` with Python 3.11 (`pip install -e ".[dev]"`); CI uses Python 3.11 too.
- **One diagonal for every quad** (`_quad_faces`), because it gives every interior vertex valence 6. The regular grid is then a hexagonal lattice: all interior neighborhoods are identical and point-symmetric, so the uniform Laplacian is 0 there (baseline for #5), and k-rings have 1 + 3k(k+1) vertices (#4).
- **Grid is centered** on `[-size/2, size/2]²`, which suits the `z = x² − y²` tests in #10 and #12. The **tube** runs from `z = 0` to `height`, and the **sphere** stores its rings from south to north.
- **`irregular_grid` rejects `jitter ≥ 1/6`:** below that value, no triangle can flip (proof in the docstring).
- **OBJ writes 17 significant digits,** so a save/load round trip is bit-exact.
- **Colormaps by the job they do:** magnitudes (‖Lx‖, H) use a single-hue blue ramp (`viz.SEQUENTIAL`); signed quantities use blue ↔ gray ↔ red, centered at 0 (`viz.DIVERGING`). No rainbow maps, which invent false boundaries.
- **Panels that are compared share one color scale** (`compare(shared_scale=True)`), so equal colors mean equal values.
- **Interactive viewing lives in separate `examples/01_polyscope_*.py` scripts** (documented in the README), so the PNG scripts and the tests never need polyscope.
- **Rings are computed by breadth-first search on the sparse adjacency matrix** (`ring_distance`). It returns a distance per vertex (−1 beyond `max_k`), and `k_ring` builds on it. One function covers both fixing k rings next to a free region (#9–#11) and coloring rings in figures.
- **Index arrays, not masks:** `boundary_vertices` and `k_ring` return sorted vertex indices. Build a boolean mask with `np.isin(np.arange(n), idx)` when needed (e.g. `free_mask` in #9).
- **Laplacians return `(L, D, M)`** with `L = D @ M` (App. A.1). M is kept separately because it is symmetric and the solvers in #8–#9 need it.
- **NaN means "not shown" in plots:** `plot_mesh` draws faces with a NaN vertex scalar in neutral gray. Figures use this to exclude boundary vertices, whose one-sided Lx would swamp the color scale.
- **Planar meshes are drawn from above** (`views=[(90, -90)]`); from the default oblique angle the jitter is invisible.

## Gotchas
- The book PDFs must never enter the repo (it is public). `.gitignore` excludes `*.pdf` as a safety net.
- The UV sphere has very thin triangles near the poles. Expect worse curvature estimates there (#6).
- *Graph distance is not Euclidean distance.* On the grid, the k-ring around a vertex is a hexagon sheared along the diagonal direction. Fixing "k rings" is a topological notion, so on irregular meshes the fixed band can be uneven in width.
- The uniform ‖Lx‖ scales with the edge length h (≈ 0.12·h on the irregular grids), so it is not a curvature estimate: 0.004 instead of 2H = 2 on the unit sphere. Only area-normalized Laplacians (#6) approximate Δx = −2H·n.
- Colored surfaces have no lighting (matplotlib limitation), so 3D shape is harder to read than on plain meshes. Use polyscope when shape matters.
