# Progress

## Status
- [x] Phase 1: #1 setup (PR #17), #2 meshes + OBJ (PR #18), #3 viz (PR #19), #4 topology (PR #20)
- [x] Phase 2: #5 uniform Laplacian (PR #21), #6 cotangent Laplacian (PR #22)
- [x] Phase 3: #7 explicit smoothing (PR #23), #8 implicit smoothing (PR #24)
- [x] Setup: sparse Cholesky (CHOLMOD via scikit-sparse) for the fairing solver (PR #25)
- [x] Phase 4: #9 constrained solver (PR #26), #10 membrane (PR #27), #11 thin plate / min. variation (PR #28), #12 uniform vs cotan (PR #29), #13 flow → fair (PR #30)
- [x] Phase 5: #14 bunny hole filling, with a Fig. 4.7-like hole (PR #31), #15 README, gallery, notes review (PR #32). **The plan is complete.**
- [x] Follow-up: #33 stricter citation test, which checks the cited statement (PR #34)
- [x] Follow-up: three-panel elbow-profile comparison with explicit cotangent/uniform labels and a dedicated regeneration command.
- [x] Follow-up: a hand-outlined hole on the bunny's upper back (`examples/05_bunny_back_region.py`), refilled with a thin plate; the outline was matched to the mesh once, and the script stores the fitted ball.
- [x] Follow-up: a real hole on the same region (`examples/05_bunny_hole_filling.py`): faces and vertices removed, a new patch built from the boundary loop alone (plane fit, constrained Delaunay with Triangle, interior on the plane), then faired with uniform and cotangent weights (k = 2).
- [x] Follow-up: a second real hole (`examples/05_bunny_hole_filling_rear.py`), outlined by hand on the upper rear of the back and matched to the mesh once (the script stores the fitted ball); filled with the same patch construction and thin plate, drawn all gray as in Fig. 4.7.
- [x] Follow-up: the second hole made a little larger (`05_bunny_hole_filling_rear.py --larger`: same center, radius 0.028), with both figures.
- [x] Follow-up: Fig. 4.9 on a real head (`examples/06_ogre_face.py`): the face of Keenan Crane's ogre (public domain, read from `../Example-Meshes/ogre/`, not committed) refilled by a thin plate with uniform and cotangent weights.

## Next
- Nothing planned. Moonshot M1 (#16, mixed Voronoi area): only if the user asks.

## Decisions
- **Issue #N = plan step N**; figures are `docs/img/<NN>-<name>.png`; the README's table says which script writes which figure.
- **Environment:** `.venv`, Python 3.11, `pip install -e ".[dev,mesh,viewer]"`; CI uses 3.11 with `.[dev,mesh,hole]` (no bunny: its 4 tests skip). `hole` = `triangle`, only for the hole-filling example and its test. polyscope only in `examples/*polyscope*.py`, never in tests.
- **Data:** plain arrays `V` (n, 3), `F` (m, 3); masks are boolean; topology helpers in `fairing/mesh.py` (rings by BFS, `ring_distance`/`k_ring`). One diagonal per quad: interior valence 6, k-rings of 1 + 3k(k+1) vertices.
- **Laplacians return `(L, D, M)`** with `L = D @ M`, M symmetric; `laplacian="uniform" | "cotan"`; barycentric areas (M1 would add mixed Voronoi).
- **Smoothing:** explicit rebuilds L every step by default (`rebuild=False` keeps it fixed, #13); `explicit_step_limit` = 2/(λ|μ_min|) via `eigsh`. Implicit solves the symmetric (D⁻¹ − hλM)x' = D⁻¹x with SciPy's LU (`factorized`), on purpose.
- **Fairing:** `solve_fair` builds (−1)ᵏM(DM)ᵏ⁻¹ once from the input (the book's linear method), solves A_ff x_f = −A_fc x_c with CHOLMOD (§A.4, §A.6), refuses an all-free mask. Book methods only: no weight refresh, no reference geometry.
- **Plots:** sequential blue for magnitudes, blue–gray–red for signed values; compared panels share one scale; NaN = gray ("not shown").
- **Docs:** claims are checked against the book's wording; what we derived (e.g. implicit stability for any h) is labeled as ours. Code citations read `` `function`, `file.py:N`<!-- statement --> ``; the comment is not needed for a `def` line (#33).

## Gotchas
- The book PDFs must never enter the repo (public); `.gitignore` excludes `*.pdf`. The bunny lives in git-ignored `data/` (credit Stanford, no commercial use). The ogre lives outside the repo, in `../Example-Meshes/ogre/`.
- scikit-sparse has no wheels: install SuiteSparse first; on macOS set `SUITESPARSE_*_DIR` and `CPLUS_INCLUDE_PATH` (README).
- Barycentric area is wrong at UV-sphere poles: H = ¾·(1/R) there at any resolution.
- The uniform ‖Lx‖ is a length, the cotangent one a curvature: a shared scale only compares zero vs non-zero.
- Explicit cotangent smoothing is stiff (h_max set by the thinnest triangles); every linear method shrinks.
- Uniform weights parametrize by connectivity (artifacts on graded meshes that refinement does not fix); cotangent weights frozen from a damaged start inherit the damage. The free vertices' start only enters through the weights.
- `bun_zipper_res2` is non-manifold: keep free regions ≥ k + 1 rings from problem edges (`edge_face_counts`).
- C^(k−1) is tested by refinement (the joint angle stays finite, ∝ h, ∝ h²), never at one resolution.
- matplotlib surfaces are unlit, and face colors average away single-vertex outliers: use numbers for quantitative claims.
- matplotlib sorts faces by depth only within one collection: a region that wraps out of view needs fixed and free faces in one `Poly3DCollection` (#14 upper-back hole).
- A hole's patch must be built from the surviving mesh only: `05_bunny_hole_filling.py` proves it by rebuilding with the removed positions set to NaN. Compare patches with new connectivity by surface distance, never vertex by vertex.
- After editing code, run `pytest tests/test_docs.py`: it names the new line of every cited statement that moved; update N in the note.
- §A.4 writes Cholesky as LLᵀ: that L is the triangular factor, not the Laplacian.
