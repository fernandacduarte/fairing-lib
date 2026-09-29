# Progress

## Status
- [x] #1 Project setup: PR #17
- [x] #2 Synthetic meshes and OBJ I/O: PR #18

## Next
- #3 Visualization helpers (Phase 1)

## Decisions
- **Issue numbers = plan step numbers.** All issues were created before any PR, so issue #N is step N of PLAN.md §3 (M1 is #16).
- **Topology helpers live in `fairing/mesh.py`**, as in the PLAN.md §1 layout; there is no separate module.
- **Figures are named `docs/img/<NN>-<name>.png`** with a zero-padded issue number, so they sort in order.
- **Local environment:** `.venv` with Python 3.11 (`pip install -e ".[dev]"`); CI uses Python 3.11 too.
- **One diagonal for every quad** (`_quad_faces`), because it gives every interior vertex valence 6. The regular grid is then a hexagonal lattice: all interior neighborhoods are identical and point-symmetric, so the uniform Laplacian is 0 there (baseline for #5), and k-rings have 1 + 3k(k+1) vertices (#4).
- **Grid is centered** on `[-size/2, size/2]²`, which suits the `z = x² − y²` tests in #10 and #12. The **tube** runs from `z = 0` to `height`, and the **sphere** stores its rings from south to north.
- **`irregular_grid` rejects `jitter ≥ 1/6`:** below that value, no triangle can flip (proof in the docstring).
- **OBJ writes 17 significant digits,** so a save/load round trip is bit-exact.

## Gotchas
- The book PDFs must never enter the repo (it is public). `.gitignore` excludes `*.pdf` as a safety net.
- The UV sphere has very thin triangles near the poles. Expect worse curvature estimates there (#6).
