# fairing-lib — Implementation Plan

A small Python library for studying **mesh fairing** as presented in *Polygon Mesh Processing* (Botsch et al.), Chapters 3–4 and Appendix A. The goal is learning: every step is small, grounded in a specific equation from the book, and ends with a test and an image that make it easy to check.

---

## 0. Instructions for the LLM (read first)

**Your role.** You implement this plan one issue at a time. The user follows along and approves each step. Keep the code small and readable, and prefer clarity over cleverness. Add abstractions only when a later issue actually needs them.

**Resuming in a new conversation.** If `docs/PROGRESS.md` exists, the project is already underway. Read this plan and then `docs/PROGRESS.md`. Run `gh issue list` and `gh pr list` to check the state on GitHub. Then tell the user where things stand and which issue comes next before you touch any code.

**Before any work** (first session only)
1. Check that `gh auth status` succeeds. If it does not, stop and ask the user to authenticate.
2. In this folder, run `git init`, then create the repo with `gh repo create fairing-lib --public --source . --push`. Commit this `PLAN.md` so that future sessions can find it.
3. Create one **milestone per phase** (see §3) and one **GitHub issue per step**, using the template in §4. Create all issues up front so the user can see the whole roadmap. This includes the moonshots, which get their own label and milestone.

**Loop for each issue, in order**
1. Create a branch named `issue-<N>-<short-name>` from an up-to-date `main`.
2. Post a short comment on the issue (3–6 lines) that explains the theory behind the step, quotes the equation number from the book, and says what you are about to do.
3. Implement **only** what that issue asks for.
4. Add or extend tests in `tests/` and make sure `pytest` passes.
5. Add or extend the example script in `examples/`, which saves its figures to `docs/img/<issue>-*.png`.
6. **Update the docs in the same PR** (see "Documentation" below): always update `docs/PROGRESS.md`, and add to the current phase note.
7. Open a PR whose body contains `Closes #N`, a summary, the command to reproduce the result, the embedded image(s), and the "Done when" checklist from the issue with each item ticked.
8. **Stop and wait for the user's approval.** Apply any requested changes.
9. After approval, squash-merge, delete the branch, and pull `main`.

**Documentation** (keep it short; it is for the user and for future sessions)
- `docs/PROGRESS.md` is the handoff file, **rewritten** (not appended to) on every PR. It stays under about 40 lines and has four sections:
  - **Status:** the issues that are done, with their PR links.
  - **Next:** the next issue to work on.
  - **Decisions:** design choices and the reason for each, e.g. "barycentric area, because …".
  - **Gotchas:** pitfalls and open questions found so far.
- `docs/phases/phase-<N>.md` is a study note for each phase, built up as the phase's issues are merged. For each issue, it gives:
  - **Theory → code:** a table mapping each equation to the function and line that implements it.
  - **What the figures show:** an image with 1–2 sentences on how to read it.
  - **Pitfalls:** what went wrong or surprised us.

  Write for a student who has read the chapter but not the code.

**Rules**
- Never skip ahead or bundle issues together. If something is unclear, ask.
- Keep mesh data as plain NumPy arrays: `V` is `(n, 3)` float and `F` is `(m, 3)` int. Use `scipy.sparse` for all matrices.
- Cite the book's equation numbers in docstrings, for example `# Eq. (3.11)`.
- Before downloading any external file (such as the bunny), ask the user.

---

## 1. Stack and layout

- **Python 3.11+**, with `numpy`, `scipy`, `matplotlib`, and `pytest`.
- `polyscope` is optional and only used for interactive viewing. It is never imported by the tests.
- `trimesh` is used only to load the real mesh in the last phase.

```
fairing-lib/
├── fairing/
│   ├── mesh.py        # generators, OBJ I/O, topology helpers
│   ├── laplacian.py   # uniform + cotangent Laplacian (L = D·M)
│   ├── smoothing.py   # explicit / implicit diffusion flow
│   ├── fairing.py     # constrained solve of L^k x = 0
│   └── viz.py         # matplotlib (PNG) + optional polyscope
├── examples/          # one script per phase, writes to docs/img/
├── tests/
├── docs/
│   ├── PROGRESS.md    # handoff: status, next step, decisions
│   ├── phases/        # one study note per phase
│   └── img/
├── pyproject.toml
└── README.md
```

## 2. Key formulas (the reference for the whole project)

| Concept | Formula | Book |
|---|---|---|
| Uniform Laplacian | `Δf(vᵢ) = 1/deg(vᵢ) · Σⱼ (fⱼ − fᵢ)` | Eq. 3.10 |
| Cotangent Laplacian | `Δf(vᵢ) = 1/(2Aᵢ) · Σⱼ (cot αᵢⱼ + cot βᵢⱼ)(fⱼ − fᵢ)` | Eq. 3.11 |
| Matrix form | `L = D·M`, with `D = diag(wᵢ)` and `M` symmetric (`mᵢᵢ = −Σ wᵢⱼ`) | App. A.1 |
| Mean curvature | `Δx = −2H·n`, so `H = ½‖Δx‖` | Eq. 3.7, 3.13 |
| Explicit flow | `x ← x + hλ·L x` | §4.2 |
| Implicit flow | `(I − hλL) x' = x`, solved as `(D⁻¹ − hλM) x' = D⁻¹x` (symmetric) | §4.2, A.1 |
| Fairing | `Lᵏ x = 0`, with k = 1 (membrane), 2 (thin plate), 3 (minimum variation) | §4.3 |
| SPD system | `(−1)ᵏ M (DM)ᵏ⁻¹ x = 0`; constrained vertices move to the right-hand side | Eq. A.2, A.1 |

For the vertex area `Aᵢ`, use the **barycentric cell** (1/3 of the incident triangle areas, Fig. 3.7). The mixed Voronoi cell is a moonshot (M1). To reach C^(k−1) continuity at the boundary of the free region, fix **k rings** of boundary vertices (§4.3).

---

## 3. Phases and issues

Each bullet below is one GitHub issue. The text after **✔** is the "Done when" criterion.

### Phase 1 — Foundations
1. **Project setup.** Add `pyproject.toml`, the package skeleton, `.gitignore`, a README stub, and a GitHub Actions workflow that runs `pytest`.
   ✔ CI is green on an empty test.
2. **Synthetic meshes + OBJ I/O.** Write generators for a regular grid, an *irregular* planar grid (with jittered interior vertices), a UV sphere, and a tube/cylinder. Add `load_obj` and `save_obj`.
   ✔ A round trip (save, then load) gives back the same `V` and `F`, and the triangle counts match what is expected.
3. **Visualization.** Write `viz.plot_mesh(V, F, scalars=None, ax=None)` with matplotlib `plot_trisurf` and a colorbar, plus `viz.compare([...], titles)` for side-by-side figures, and `viz.show_polyscope(...)` as an optional viewer.
   ✔ `examples/01_meshes.py` saves a PNG that shows all the generators.
4. **Topology helpers.** Compute the unique edges, one-ring neighbors, boundary vertices, and `k_ring(seed_vertices, k)`.
   ✔ The tests check the boundary count on a grid and the ring sizes on a regular grid. A PNG colors the boundary vertices and rings 1, 2, and 3.

### Phase 2 — Discrete Laplace–Beltrami (Ch. 3)
5. **Uniform Laplacian (Eq. 3.10).** Return `L`, and also `D` and `M` separately.
   ✔ Every row of `M` sums to 0. On the *irregular* planar grid, `‖Lx‖ ≠ 0` at interior vertices, which is the flaw the book points out. A PNG shows `‖Lx‖` as colors.
6. **Cotangent Laplacian (Eq. 3.11).** Use cotangent weights with barycentric areas.
   ✔ `M` is symmetric and its rows sum to 0. On the irregular planar grid, `‖Lx‖ ≈ 0` at interior vertices (linear precision). On a sphere of radius R, `H = ½‖Lx‖ ≈ 1/R`. A PNG compares the uniform and cotangent `‖Lx‖` side by side.

### Phase 3 — Diffusion flow (§4.2)
7. **Explicit Laplacian smoothing.** Run `x ← x + hλLx` for n iterations, with either Laplacian.
   ✔ On a noisy sphere, the PNG shows iterations 0, 10, and 100. A second PNG shows that a large `h` blows up, which illustrates the instability.
8. **Implicit Laplacian smoothing.** Solve the symmetric system `(D⁻¹ − hλM)x' = D⁻¹x` with `scipy.sparse.linalg.factorized`.
   ✔ A large `h` stays stable. A PNG reproduces the idea of Fig. 4.6: with the uniform Laplacian the triangles get regularized (they drift tangentially), while with the cotangent Laplacian the triangle shapes are preserved.

### Phase 4 — Fairing (§4.3, App. A)
9. **Constrained solver.** Write `solve_fair(V, F, free_mask, k, laplacian="cotan")`. It builds `(−1)ᵏ M(DM)ᵏ⁻¹`, splits the vertices into free and fixed, solves `A_ff x_f = −A_fc x_c` per coordinate, and returns the new `V`.
   ✔ `A_ff` is symmetric positive definite, which the test checks with a Cholesky or eigenvalue test on a small mesh. If nothing is free, `V` comes back unchanged.
10. **Membrane surface (k = 1).** Use a grid whose interior disk is free.
    ✔ If the boundary is planar and the interior is noisy, the result is exactly planar. If the boundary heights are `z = x² − y²` (a harmonic function), the error is small and shrinks as the resolution grows. A PNG shows the result.
11. **Thin plate and minimum variation (k = 2, 3).** Build a tube blend like Fig. 4.8: the bottom band is fixed, the top band is fixed and shifted sideways, and the middle is free. Fix k rings on each side.
    ✔ A PNG shows k = 1, 2, 3 side by side, plus a 2D **profile plot** of one meridian. The profile shows a C⁰ kink for k = 1 and a smooth joint for k = 2 and 3.
12. **Uniform vs. cotangent in fairing (Fig. 4.9).** Solve k = 2 on an irregular mesh with each Laplacian.
    ✔ A PNG shows the two results colored by mean curvature. The uniform result shows artifacts that the cotangent result does not.
13. **Fairing as the limit of the flow.** Run implicit smoothing with the same fixed vertices and increasing `h`, and compare the result to `solve_fair(k=1)`.
    ✔ A plot shows `‖x_flow − x_fair‖` falling toward 0 as `h` grows. This confirms the claim on p. 61 of the book.

### Phase 5 — Real mesh and docs
14. **Bunny hole filling.** Load a low-resolution bunny (ask the user before downloading it). Mark a k-ring region as free, add noise to it or flatten it, and fair it with k = 2.
    ✔ A PNG shows the mesh before and after, and a polyscope snippet is documented in the README.
15. **Final documentation.** Write the README: what fairing is, how to install, how to run each example, and an image gallery. It should link to the phase notes in `docs/phases/`. Review the phase notes for consistency and fill any gaps.
    ✔ Someone new can reproduce every figure by following the README.

### Moonshots (optional)
Create these issues with the `moonshot` label, in a "Moonshots" milestone. Work on them only after Phase 5 is merged, and only if the user asks.

- **M1. Mixed Voronoi cell (Fig. 3.7, §3.3.1).** Add `area="barycentric" | "mixed"` to the cotangent Laplacian. The parameter only changes `D`; `M` stays the same. For each triangle corner at vertex P:
  - if the triangle is non-obtuse, add `⅛(|PR|²·cot Q + |PQ|²·cot R)`;
  - if it is obtuse at P, add `A_T/2`;
  - if it is obtuse at another corner, add `A_T/4`.

  ✔ For both options, `Σ Aᵢ` equals the total mesh area. On an irregularly sampled sphere, `|H − 1/R|` is smaller with `mixed`. The k = 1 fairing results are **identical** with either area, because `D` drops out of `Lx = 0`. A PNG shows the k = 2 results colored by their per-vertex difference.

---

## 4. Issue template

```markdown
## Goal
<one sentence>

## Book reference
Section X.Y, Eq. (N)

## Tasks
- [ ] ...

## Done when
- [ ] <test that passes>
- [ ] <image in docs/img/ that shows ...>
```
