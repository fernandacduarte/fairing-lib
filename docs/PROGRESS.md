# Progress

## Status
- [x] Phase 1: #1 setup (PR #17), #2 meshes + OBJ (PR #18), #3 viz (PR #19), #4 topology (PR #20)
- [x] Phase 2: #5 uniform Laplacian (PR #21), #6 cotangent Laplacian (PR #22)
- [x] Phase 3: #7 explicit smoothing (PR #23), #8 implicit smoothing (PR #24)

## Next
- #9 Constrained SPD fairing solver, Eq. A.2 (Phase 4)

## Decisions
- **Issue #N = plan step N** (all issues were created before any PR; M1 is #16). Figures are `docs/img/<NN>-<name>.png`.
- **Environment:** `.venv`, Python 3.11, `pip install -e ".[dev]"`; CI uses 3.11. polyscope only in `examples/01_polyscope_*.py`, never in tests.
- **Topology helpers live in `fairing/mesh.py`.** They return sorted index arrays; build masks with `np.isin(np.arange(n), idx)`.
- **One diagonal per quad** (`_quad_faces`): interior valence 6, a hexagonal lattice where the uniform Laplacian is 0 and k-rings have 1 + 3k(k+1) vertices.
- **Grid centered** on `[-size/2, size/2]²` (for the `z = x² − y²` tests in #10, #12); tube from `z = 0` to `height`; sphere rings stored south to north.
- **`irregular_grid` rejects `jitter ≥ 1/6`** (no flipped triangles); OBJ writes 17 digits (bit-exact round trip).
- **Rings by BFS** on the sparse adjacency (`ring_distance`, −1 beyond `max_k`); `k_ring` builds on it.
- **Laplacians return `(L, D, M)`** with `L = D @ M`; M is symmetric, which the solvers in #8–#9 need. `laplacian="uniform" | "cotan"` selects one through `LAPLACIANS`.
- **Cotangent weights accumulate per triangle** (cot of the angle at k goes to edge (i, j)); duplicates sum to cot α + cot β.
- **Explicit smoothing rebuilds L every step** (mean curvature flow for cotan). `explicit_step_limit` = 2/(λ|μ_min|), via `eigsh` on the symmetric D^½MD^½, with a fixed start vector so figures are reproducible.
- **Implicit smoothing solves the symmetric (D⁻¹ − hλM)x' = D⁻¹x** with `factorized` (SuperLU), one factorization per step reused for x, y, z; L rebuilt each step, as in the explicit version.
- **Noise is measured by `roughness`** (mean angle between adjacent face normals), not by the radius spread, which mixes noise with shape change.
- **Plots:** sequential blue for magnitudes, blue–gray–red for signed values; compared panels share one scale; NaN = "not shown" (gray), used for boundary vertices; planar meshes drawn from above.

## Gotchas
- The book PDFs must never enter the repo (public). `.gitignore` excludes `*.pdf`.
- **Barycentric area is wrong at UV-sphere poles** (4/3 of the Voronoi cell): H = ¾·(1/R) there at any resolution; elsewhere H converges like h². Fixed by M1 (#16).
- The uniform ‖Lx‖ is a length (≈ 0.12·h), not a curvature; only zero vs non-zero is comparable with the cotangent ‖Lx‖.
- **Explicit cotangent smoothing is stiff:** stable h = 2.7·10⁻⁵ on the 24×48 UV sphere (uniform: 1.34), set by the pole triangles. Laplacian flow shrinks (r² = r₀² − 4λt on a sphere); the uniform flow also distorts the shape.
- Implicit smoothing is stable for any h but still shrinks (radius × 1/(1 + 2hλ) on the unit sphere); cotan leaves tangential irregularity untouched (roughness floor ≈ 0.088).
- Graph distance is not Euclidean: a band of k fixed rings can vary in physical width on irregular meshes.
- Colored matplotlib surfaces are unlit, and face colors average out single-vertex outliers: use numbers for quantitative claims.
