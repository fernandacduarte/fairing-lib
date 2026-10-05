# Progress

## Status
- [x] Phase 1: #1 setup (PR #17), #2 meshes + OBJ (PR #18), #3 viz (PR #19), #4 topology (PR #20)
- [x] Phase 2: #5 uniform Laplacian (PR #21), #6 cotangent Laplacian (PR #22)
- [x] Phase 3: #7 explicit smoothing (PR #23), #8 implicit smoothing (PR #24)
- [x] Setup: sparse Cholesky (CHOLMOD via scikit-sparse) for the fairing solver (PR #25)
- [x] Phase 4: #9 constrained solver (PR #26), #10 membrane (PR #27), #11 thin plate / min. variation (PR #28), #12 uniform vs cotan (PR #29), #13 flow → fair (PR #30) (Phase 4 complete)

## Next
- #14 Bunny hole filling with thin-plate fairing (Phase 5). Ask the user before downloading the bunny; decide how the damaged region's weights are obtained (deferred from #10).

## Decisions
- **Issue #N = plan step N** (all issues were created before any PR; M1 is #16). Figures are `docs/img/<NN>-<name>.png`.
- **Environment:** `.venv`, Python 3.11, `pip install -e ".[dev]"`; CI uses 3.11. polyscope only in `examples/01_polyscope_*.py`, never in tests.
- **Topology helpers live in `fairing/mesh.py`.** They return sorted index arrays (masks via `np.isin`); rings by BFS on the sparse adjacency (`ring_distance`, −1 beyond `max_k`; `k_ring` builds on it).
- **One diagonal per quad** (`_quad_faces`): interior valence 6, a hexagonal lattice where the uniform Laplacian is 0 and k-rings have 1 + 3k(k+1) vertices.
- **Grid centered** on `[-size/2, size/2]²` (for the `z = x² − y²` tests in #10, #12); tube from `z = 0` to `height`; sphere rings stored south to north.
- **`irregular_grid` rejects `jitter ≥ 1/6`** (no flipped triangles); OBJ writes 17 digits (bit-exact round trip).
- **Laplacians return `(L, D, M)`** with `L = D @ M`; M is symmetric, which the solvers in #8–#9 need. `laplacian="uniform" | "cotan"` selects one through `LAPLACIANS`.
- **Cotangent weights accumulate per triangle** (cot of the angle at k goes to edge (i, j)); duplicates sum to cot α + cot β.
- **Explicit smoothing rebuilds L every step** (mean curvature flow for cotan). `explicit_step_limit` = 2/(λ|μ_min|), via `eigsh` on the symmetric D^½MD^½, with a fixed start vector so figures are reproducible.
- **Implicit smoothing solves the symmetric (D⁻¹ − hλM)x' = D⁻¹x** (optional `fixed_mask`; one step with h → ∞ gives `solve_fair(k=1)`, gap ∝ 1/h; many steps with L rebuilt → the minimal surface instead, #13) with `factorized` (SuperLU), one factorization per step reused for x, y, z; L rebuilt each step, as in the explicit version.
- **Solvers: Cholesky for fairing only.** From #9 on, the fairing solver uses a sparse Cholesky factorization (CHOLMOD, `sksparse.cholmod.cho_factor`), as App. A recommends for SPD systems. Implicit smoothing (#8) deliberately keeps SciPy's LU (`factorized`) and is not migrated. The contrast is part of the study.
- **Fairing:** `solve_fair` builds A = (−1)ᵏM(DM)ᵏ⁻¹ once from the input V (book's linear method), solves A_ff x_f = −A_fc x_c with `cho_factor`, takes a boolean `free_mask`, and refuses an all-free mask. Only the k fixed rings next to the free region matter (pipe length is irrelevant). Fig. 4.8 investigation (phase-4 note): our k = 3 elbow differs from the book's mostly by proportions/shading; straight-tube weights would fold the inner side.
- **Noise is measured by `roughness`** (mean angle between adjacent face normals), not by the radius spread, which mixes noise with shape change.
- **Plots:** sequential blue for magnitudes, blue–gray–red for signed values; compared panels share one scale; NaN = "not shown" (gray), used for boundary vertices; planar meshes drawn from above.

## Gotchas
- The book PDFs must never enter the repo (public). `.gitignore` excludes `*.pdf`.
- **Barycentric area is wrong at UV-sphere poles** (4/3 of the Voronoi cell): H = ¾·(1/R) there at any resolution; elsewhere H converges like h². Fixed by M1 (#16).
- The uniform ‖Lx‖ is a length (≈ 0.12·h), not a curvature; only zero vs non-zero is comparable with the cotangent ‖Lx‖.
- **Explicit cotangent smoothing is stiff:** stable h = 2.7·10⁻⁵ on the 24×48 UV sphere (uniform: 1.34), set by the pole triangles; the uniform flow also distorts the shape. Linear methods shrink: Laplacian flow (r² = r₀² − 4λt on a sphere) and fairing (bending energy ∝ r² around a tube in a frozen parametrization).
- Implicit smoothing is stable for any h but still shrinks (radius × 1/(1 + 2hλ) on the unit sphere); cotan leaves tangential irregularity untouched (roughness floor ≈ 0.088).
- **Uniform weights parametrize by connectivity:** exact on regular meshes (thin plate 10⁻¹⁴), artifacts on graded ones (error 10⁻² that refinement does not reduce, |H| ×6, vertices slide); cotan is density-independent (`graded_grid`, #12).
- **Two membranes:** uniform ≈ Eq. 4.8 (harmonic in the connectivity/parameter domain; → x² − y² in case B); cotan from a smooth start ≈ Eq. 4.7 (minimal surface, H → 0, 1.3·10⁻³ from x² − y²). Cotan weights frozen from an input damaged at edge scale corrupt the result (|H| ≈ 47).
- **Book methods only (user decision):** no weight refresh or reference geometry for now; how to get good cotan weights for a real hole is deferred to #14. The free vertices' start only enters through the weights: uniform results ignore it, cotan results inherit it (k = 3 elbow: up to 0.36 apart from two starts).
- scikit-sparse has no wheels: install SuiteSparse first (`brew install suite-sparse` / `apt install libsuitesparse-dev`). On macOS also set `SUITESPARSE_*_DIR` and `CPLUS_INCLUDE_PATH` (README), or clang++ cannot find `<complex>`.
- **C^(k−1) is tested by refinement:** the meridian's turning angle at the joint stays finite (k = 1), ∝ h (k = 2), ∝ h² (k = 3). One resolution alone cannot tell a small kink from a smooth bend.
- Colored matplotlib surfaces are unlit, and face colors average out single-vertex outliers: use numbers for quantitative claims. Notes cite `function`, `file.py:N`; `tests/test_docs.py` checks N is inside that function.
