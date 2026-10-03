# Phase 4 — Fairing (§4.3, App. A)

Fairing computes the smoothest surface that fits fixed boundary vertices by solving Lᵏx = 0 on the free vertices: k = 1 gives a membrane, k = 2 a thin plate, k = 3 a minimum variation surface. After constraints are moved to the right-hand side, App. A.1 turns this into a sparse **symmetric positive definite** (SPD) system (Eq. A.2). This note is built up issue by issue (#9–#13).

## Solver choice: sparse Cholesky (CHOLMOD)

**Decision.** The fairing solver factorizes its SPD matrix with a **sparse Cholesky** factorization, A = LLᵀ (up to a fill-reducing reordering), using CHOLMOD from SuiteSparse through `scikit-sparse` (`sksparse.cholmod.cho_factor`). This applies to fairing only. Implicit smoothing (#8) keeps SciPy's sparse LU (`scipy.sparse.linalg.factorized`), and is deliberately left as it is.

**Why Cholesky.** App. A recommends sparse direct solvers, and Cholesky in particular, for SPD systems:
- *Half the work and memory.* LU stores two triangular factors; Cholesky stores one, Lᵀ being the transpose of L. On the implicit-smoothing matrix of the 24 × 48 sphere (n = 1106), CHOLMOD's factor has **23 796** non-zeros, against **80 258** for SciPy's LU factors, about 3.4× less.
- *No pivoting.* An SPD matrix can be factorized in any order without numerical trouble. The reordering can therefore be chosen purely to limit fill-in (AMD/METIS), not for stability.
- *A free check.* Cholesky only exists for positive definite matrices. CHOLMOD raises `CholmodNotPositiveDefiniteError` otherwise, so a successful factorization *proves* that the constrained fairing matrix is SPD, which is the "Done when" of #9.

**How it is used.** `f = cho_factor(A)` computes the factorization once; `f.solve(b)` then solves for all right-hand sides at once, with b an n × 3 array for x, y and z. `tests/test_cholmod.py` pins down these two behaviors.

**Installation.** `scikit-sparse` has no prebuilt wheels: pip compiles it against the SuiteSparse C library, which must be installed first (README, "Install"). CI installs `libsuitesparse-dev` before `pip install`.

## #9 Constrained solver

**Theory → code.** We want **Lᵏx = 0 at every free vertex**, with the constrained vertices fixed (§4.3, p. 59–60). Three steps turn this into an SPD system (App. A.1).

1. **Drop the left-most D (symmetry, Eq. A.2).** Lᵏ = (DM)ᵏ = D · M(DM)ᵏ⁻¹. D is diagonal with positive entries, so it only rescales each row: row i of Lᵏx is zero exactly when row i of M(DM)ᵏ⁻¹x is zero. What is left, M, MDM, MDMDM, …, is symmetric, because M and D are.
2. **Move the known values to the right-hand side (constraints).** Order the unknowns as free (f) and constrained (c). The equations of the free vertices are A_ff x_f + A_fc x_c = 0, and x_c is known, so **A_ff x_f = −A_fc x_c**. The rows and columns of the constrained vertices are removed together, so A_ff stays symmetric.
3. **Fix the sign (definiteness).** App. A.1 states (citing Pinkall and Polthier) that once constraints are incorporated, the Laplacian system becomes *negative* definite; multiplying by (−1)ᵏ makes it positive definite. Two cases you can check by hand:
   - k = 1: A_ff = −M_ff. −M is the "graph Laplacian with weights wᵢⱼ"; its quadratic form is xᵀ(−M)x = ½ Σ wᵢⱼ (xᵢ − xⱼ)², which is ≥ 0 for positive weights and zero only for constants. Pinning at least one vertex rules out constants, so A_ff is positive definite.
   - k = 2: A_ff = (MDM)_ff = Bᵀ D B, where B = M restricted to the free columns. A "weighted square" like this is always ≥ 0, and it is zero only if B x_f = 0. With constraints, that only happens for x_f = 0.

   The tests confirm positive definiteness numerically for k = 1, 2, 3 (and CHOLMOD would refuse to factorize otherwise).

**From the book's formula to the code: where did D⁻¹b go?** App. A.1 writes the general system as

  (−1)ᵏ M(DM)ᵏ⁻¹ x = (−1)ᵏ D⁻¹ b.

Fairing asks for Lᵏx = **0** (§4.3), so **b = 0**, and the term (−1)ᵏD⁻¹b vanishes: the right-hand side starts as zero. Then the constraints are applied as the book describes. The column aᵢ of each constrained vertex moves to the right (b ← b − xᵢaᵢ), and its row is removed. Starting from zero, the right-hand side becomes

  0 − Σᵢ∈C xᵢ aᵢ = **−A_fc x_c**,

where A_fc collects those columns, restricted to the free rows. That is `fairing.py:71`. The factor (−1)ᵏ is not lost: `fairing_matrix` already multiplies the whole matrix by it, so A_fc carries it, and (−1)ᵏ·0 is still 0.

Compare with implicit smoothing (#8), where b ≠ 0. There the system (I − hλL)x' = x has b = x, and the code does compute D⁻¹b (`rhs = D_inv @ V`, `smoothing.py:76`). The two steps show the two halves of the formula. `solve_fair` handles only b = 0. A problem with b ≠ 0, such as the deformations mentioned on p. 61, would add (−1)ᵏ D_ff⁻¹ b_f to the right-hand side, using the D entries of the free rows.

| Book | Formula | Code |
|---|---|---|
| App. A.1, Eq. A.2 | M(DM)ᵏ⁻¹, built by repeated `A @ D @ M` | `fairing_matrix`, `fairing/fairing.py:33` |
| App. A.1 ("Definiteness") | sign (−1)ᵏ | `fairing.py:34` |
| App. A.1 ("Definiteness") | A_ff and A_fc: free rows, split columns | `solve_fair`, `fairing.py:69–70` |
| App. A.1 | right-hand side −A_fc x_c | `fairing.py:71` |
| App. A (sparse Cholesky) | factorize A_ff once, solve x, y, z together | `fairing.py:73` |

L is built once from the input geometry and then frozen; that is what makes the system linear. The starting positions of the free vertices enter only through the cotangent weights, and #10 shows that this matters. The uniform weights do not depend on positions at all. If every vertex is free, the problem has no unique answer (constants solve it), and `solve_fair` raises a `ValueError`.

Tests (`tests/test_fairing.py`), for k = 1, 2, 3 and both Laplacians: A_ff is symmetric, its smallest eigenvalue is positive, and a Cholesky factorization exists. The solution satisfies Lᵏx = 0 at the free vertices, to 10⁻⁹ relative to the size of Lᵏx on the input. Constrained vertices do not move; nothing free → V unchanged; everything free → error. For k = 1, A = −M.

**What the figures show.**

![Sparsity of the fairing matrix for k = 1, 2, 3](../img/09-sparsity.png)

`docs/img/09-sparsity.png`: the non-zero pattern of A on a 12 × 12 irregular grid, with vertices numbered row by row. Row i has a non-zero in column j exactly when vertex j is within k rings of vertex i. So an interior row has **7, 19 and 37** non-zeros for k = 1, 2, 3: the k-ring sizes 1 + 3k(k+1) from #4. App. A.1 quotes about 7 for L and about 19 for L². Because the vertices are numbered row by row, each ring of neighbors shows up as a diagonal band offset by multiples of 12 (one grid row): 3, 5 and 7 bands. The matrix stays sparse (6.3, 15.8 and 28.2 non-zeros per row on average, out of 144), but every extra order makes it denser and the solve more expensive.

**Pitfalls.**
- *Residuals must be relative.* The entries of Lᵏ grow like 1/h²ᵏ (h = edge length), so the absolute residual of Lᵏx at the solution grows with k: about 10⁻¹⁴, 10⁻¹¹ and 10⁻⁸ for k = 1, 2, 3 on the test patch. Compared with the size of Lᵏx on the input, all of them are at round-off level.
- *A_ff is only symmetric up to round-off.* The products MDM and MDMDM are computed in floating point, so A_ff and its transpose differ by about 10⁻¹³ (k = 2) or 10⁻¹⁰ (k = 3), relative to entries of order 10² to 10⁵. CHOLMOD reads only one triangle, so this does not matter for the solve, but tests must compare with a tolerance.
- *Fill-in is modest.* The Cholesky factor of A_ff (37 free vertices) has 219, 404 and 552 non-zeros for k = 1, 2, 3. It stores one triangle, so compare it with half of A_ff: about 30% extra entries for k = 3.
- *Free vertices near the mesh boundary.* Row i of Lᵏ reaches k rings around vertex i. If those rings include the mesh boundary, the one-sided boundary Laplacian (#5) enters the equations. Keeping k constrained rings between the free region and the boundary avoids it (#10–#11).

## #10 Membrane surface (k = 1)

**Theory → code.** `solve_fair(..., k=1)` solves **Lx = 0** on the free vertices: every free vertex ends up at the weighted average of its neighbors. The book reaches this equation in two steps (§4.3, p. 57–59), and the step in between is what this issue is really about.

**Two membranes.**
- **Eq. 4.7, the true membrane:** the surface of *minimal area* spanning the boundary. Its condition is Δₛx = 0, with Δₛ the Laplace–Beltrami operator *of the result surface itself*. By Eq. 3.7 this means H = 0 everywhere: a **minimal surface**, like a soap film. The operator depends on the unknown surface, so the problem is non-linear.
- **Eq. 4.8, the linearized membrane:** the book replaces area with the Dirichlet energy ∫‖x_u‖² + ‖x_v‖², whose derivatives are taken with respect to a **fixed parametrization** (u, v). Its solution is harmonic with respect to (u, v). In general it is a different surface, and not a minimal one.

**What the code does.** The book's discrete method (p. 59) is one linear solve, **Lx = 0, with L built once from the input mesh and frozen**. Which of the two membranes that approximates depends on where the weights come from:
- **Uniform weights** ignore the geometry. Their implicit "parametrization" is the mesh connectivity. On our nearly regular grid that behaves like the flat (x, y) domain, so uniform fairing approximates **Eq. 4.8 over (x, y)**.
- **Cotangent weights** are the Laplace–Beltrami operator of the *input surface*. When the input is already close to the answer, freezing them is a good approximation of Δₛ, and one solve lands close to **Eq. 4.7, the minimal surface**. When the input is badly damaged, the frozen weights describe a crumpled surface, and the result inherits the damage.

**Is this really what the book does, and why is it not the minimal surface?**

1. *Frozen weights are the book's method.* On p. 59 the book discretizes Δx = 0 with the discrete Laplace–Beltrami operator and calls the result a **linear** system, Lx = 0. Linear means L is a constant matrix: computed once from the mesh at hand (here, the input) and never updated during the solve.
2. *The book deliberately solves a substitute problem.* Area (Eq. 4.7) is strongly non-linear, so the book replaces it with the quadratic Dirichlet energy (Eq. 4.8), whose minimum is one linear solve (p. 58). In "Nonlinear smoothing" (§4.4, p. 62) it acknowledges the price: methods that solve the true non-linear problem are harder, but their results are better and depend less on the initial triangulation or parametrization.
3. *When the substitute equals the real thing.* For any parametrization, ‖x_u‖² + ‖x_v‖² ≥ 2‖x_u‖‖x_v‖ ≥ 2‖x_u × x_v‖, so **Dirichlet energy ≥ 2 × area**. Equality holds only when ‖x_u‖ = ‖x_v‖ and x_u ⊥ x_v, i.e. for a *conformal* parametrization. Then minimizing one minimizes the other.
4. *Where the parametrization comes from.* Cotangent weights computed from a mesh M measure the Dirichlet energy of the result **relative to M**: M plays the role of the parametrization. So the answer depends on what the weights were frozen from:

| weights frozen from | the substitute energy is | one solve gives |
|---|---|---|
| a surface already close to the answer (smooth saddle) | ≈ area (the result is nearly a conformal copy of M) | ≈ minimal surface, mean \|H\| ≈ 0.002 |
| the flat grid | Dirichlet energy over the plane | the harmonic graph x² − y² (Eq. 4.8) |
| a mildly noisy start (σ ≈ 0.3 h) | Dirichlet energy relative to a noisy shape | in between: smooth, mean \|H\| ≈ 0.064 |
| a heavily noisy start (σ > h) | Dirichlet energy relative to a crumpled shape | corrupted, mean \|H\| ≈ 2 to 47 |

The uniform weights are the same story with a "parametrization" that ignores geometry altogether: the mesh connectivity. **Practical lesson:** the book's linear method needs a reasonable starting shape for the free region. Repeating the solve with weights recomputed from each result would converge to the true minimal surface; that non-linear route is left for #14, by decision.

**The test cases.** An irregular grid on [−0.5, 0.5]² whose disk of radius 0.3 is free; the free vertices start at the target height, with or without noise.
- **A, planar boundary.** z = 0 on all constrained vertices. Then z = 0 is the unique solution of Lz = 0, *whatever the weights*, so the result is exactly flat for both Laplacians.
- **B, boundary heights z = x² − y².** This function is harmonic in the plane (∂²/∂x² + ∂²/∂y² = 2 − 2 = 0), so it is the exact answer of **Eq. 4.8 over the flat (x, y) domain**. But it is **not** a minimal surface: its mean curvature on the disk is |H| ≈ 0.07. So it is the right reference for the uniform weights, and the wrong one for the cotangent weights.

| Book | Formula | Code |
|---|---|---|
| Eq. 4.8, p. 59 | Lx = 0 on the free vertices, L frozen from the input | `solve_fair(V, F, free, k=1)`, `fairing/fairing.py:68` |
| Eq. 3.7 | H = ½‖Lx‖, to tell a minimal surface (H = 0) apart | `mean_curvature` |

Tests (`tests/test_fairing.py`):
- **Case A:** flat to 10⁻¹² for both Laplacians. With cotangent weights from a noisy start, the free vertices also slide in the plane (by more than 10⁻³), because the noisy input is not planar and the weights lose linear precision (#6).
- **Case B, uniform:** the distance to x² − y² shrinks over n = 11, 21, 41.
- **Case B, cotangent from a smooth start:** mean |H| is below 20% of the saddle's, and the distance to x² − y² stays at a fixed gap between 5·10⁻⁴ and 3·10⁻³, independent of h.
- **Case B, cotangent from a badly damaged start** (noise σ = 0.02, larger than the edge length at n = 81): mean |H| more than 100× larger than from a smooth start.

**What the figures show.**

![Membrane surfaces, cases A and B](../img/10-membrane.png)

`docs/img/10-membrane.png`, colored by height (gray = 0, shared scale ±0.25). Setup: a 31 × 31 grid, h ≈ 0.033; the 249 free vertices start at the target height plus noise σ = 0.01 ≈ 0.3 h. One solve with cotangent weights frozen from that noisy input.
- **Top (case A):** the result is exactly flat (max |z| = 0), as it must be for any weights. What the picture cannot show: the frozen weights come from a non-planar mesh, so the free vertices also slide *sideways*, by up to 0.007 (about 0.2 h).
- **Bottom (case B):** the result is smooth in height and continues the saddle; its heights are within 1.6·10⁻³ of x² − y², less than 1% of the color scale. But it is **not** the minimal surface: its mean |H| is 0.064, almost the saddle's 0.068, whereas a perfectly smooth start reaches ≈ 0.002 (see the next figure). The weights were frozen from the noisy start, so the shape depends partly on the noise, and the free vertices slide by up to 0.011 (a third of an edge).
- With stronger noise (σ = 0.05 ≈ 1.5 h) the same picture would show sliding of more than one edge and a mean |H| of about 2: the "corrupted" case of the pitfall. That is why this figure uses σ = 0.01.

Heights alone cannot tell these surfaces apart; `10-two-membranes.png` measures H for that. The joint at the rim is C⁰: a membrane matches the boundary *positions*, not its slope (#11 shows the kink).

![Two different membranes](../img/10-two-membranes.png)

`docs/img/10-two-membranes.png`: case B as the grid is refined from 11² to 161² vertices.
- *Left, distance to the harmonic graph x² − y².* Uniform (light blue) converges to it. Cotangent from a smooth start (dark blue) stays at a fixed gap of about 1.3·10⁻³: it is converging to a *different* surface.
- *Right, mean |H| on the disk.* Uniform has exactly the saddle's curvature (dashed): it *is* the harmonic graph, and not a minimal surface. Cotangent from a smooth start goes to H = 0 like h: it is the minimal surface, and the 1.3·10⁻³ gap on the left is the distance between the two membranes.
- *Orange, cotangent from a noisy start.* Noise of fixed size σ = 0.02 becomes larger than the edges as h shrinks. The frozen weights then describe a crumpled surface, and the result's curvature explodes (mean |H| ≈ 47 at 161²), even though its heights stay within 2·10⁻³ of the saddle.

**Pitfalls.**
- *An "exact answer" belongs to a specific problem.* x² − y² is exact for Eq. 4.8 over the flat (x, y) domain, not for Eq. 4.7. A first version of this step compared the cotangent result against it, concluded "it does not converge", and "fixed" this by building the weights from the flat grid. That only forced the cotangent solver to answer Eq. 4.8, and such a flat reference does not exist for a real mesh with a hole. It was removed. The correct check for the cotangent membrane is H → 0.
- *The weights are frozen from the input.* Linearity comes from computing L once. With cotangent weights, a smooth start gives a sensible result in one solve; a start damaged at the scale of the edges does not. Height errors alone can hide this (orange: heights within 2·10⁻³, curvature 47); check H, or look at the mesh. The book's method for hole filling, and ways to obtain good weights without a reference, are discussed in #14.
- *Measure at the final (x, y).* The free vertices can move in x and y too, so the target height must be evaluated at their new position.
