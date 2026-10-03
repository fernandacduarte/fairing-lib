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

where A_fc collects those columns, restricted to the free rows. That is `fairing.py:73`. The factor (−1)ᵏ is not lost: `fairing_matrix` already multiplies the whole matrix by it, so A_fc carries it, and (−1)ᵏ·0 is still 0.

Compare with implicit smoothing (#8), where b ≠ 0. There the system (I − hλL)x' = x has b = x, and the code does compute D⁻¹b (`rhs = D_inv @ V`, `smoothing.py:76`). The two steps show the two halves of the formula. `solve_fair` handles only b = 0. A problem with b ≠ 0, such as the deformations mentioned on p. 61, would add (−1)ᵏ D_ff⁻¹ b_f to the right-hand side, using the D entries of the free rows.

| Book | Formula | Code |
|---|---|---|
| App. A.1, Eq. A.2 | M(DM)ᵏ⁻¹, built by repeated `A @ D @ M` | `fairing_matrix`, `fairing/fairing.py:33` |
| App. A.1 ("Definiteness") | sign (−1)ᵏ | `fairing.py:34` |
| App. A.1 ("Definiteness") | A_ff and A_fc: free rows, split columns | `solve_fair`, `fairing.py:71–72` |
| App. A.1 | right-hand side −A_fc x_c | `fairing.py:73` |
| App. A (sparse Cholesky) | factorize A_ff once, solve x, y, z together | `fairing.py:75` |

L is built from the input geometry, or from a reference shape `V_ref` if one is given (added in #10, where it turns out to matter). The starting positions of the free vertices enter only through the cotangent weights; the uniform weights do not depend on positions at all. If every vertex is free, the problem has no unique answer (constants solve it), and `solve_fair` raises a `ValueError`.

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

**Theory → code.** The membrane is the surface of minimal area spanning the fixed boundary (Eq. 4.7). Area is non-linear in x, so the book replaces it with the **Dirichlet energy** ∫‖x_u‖² + ‖x_v‖² du dv (Eq. 4.8), which is quadratic. Its minimizer satisfies **Δx = 0** (Euler–Lagrange, p. 59), discretized as **Lx = 0**: `solve_fair(..., k=1)`. Each coordinate of the result is a *discrete harmonic function*: every free vertex sits at the weighted average of its neighbors, with weights wᵢⱼ.

Two cases have exact answers:
- **A, planar boundary.** z = 0 on all constrained vertices. Then z = 0 everywhere solves Lz = 0, and the solution is unique (A_ff is SPD), so the result is exactly flat, however noisy the free disk starts.
- **B, boundary heights z = x² − y².** This function is harmonic in the plane (∂²/∂x² + ∂²/∂y² = 2 − 2 = 0), so the exact continuous membrane is z = x² − y². The discrete result should approach it as the grid is refined.

The setup (`tests/test_fairing.py`, `examples/04_fairing.py`): an irregular grid on [−0.5, 0.5]² whose disk of radius 0.3 is free. The free vertices start at the target height plus noise.

**Which geometry is L built from?** This is the main lesson of this step. Eq. 4.8 measures derivatives with respect to a *fixed parametrization* (u, v). On a mesh, "the parametrization" is whatever geometry the cotangent weights are computed from. If they are computed from the noisy input, the noise leaks into the weights:
- in case A, z is still exactly 0 (the weights do not matter when every boundary value is 0), but the free vertices slide in the plane by up to two grid spacings, because weights from a noisy mesh lose linear precision (#6);
- in case B, the error stops shrinking (orange curve below).

`solve_fair` therefore takes an optional **`V_ref`**: the geometry L is built from (default: `V` itself). Here `V_ref` is the clean flat grid, the natural parameter domain. With it, x and y stay exactly in place in case A (to 10⁻¹⁵), and case B converges.

| Book | Formula | Code |
|---|---|---|
| Eq. 4.8, p. 59 | Lx = 0 on the free vertices | `solve_fair(V, F, free, k=1, ...)` |
| Eq. 4.8 ("parametrization") | weights from a reference shape | `V_ref`, `fairing/fairing.py:70` |

Tests: case A is flat to 10⁻¹² for both Laplacians, with and without `V_ref`. With `V_ref`, the cotangent result keeps the free vertices in place; without it, they move by more than 10⁻³. Case B errors shrink over n = 11, 21, 41 for both Laplacians. Without `V_ref`, the cotangent error at n = 41 is more than 10× larger than with it.

**What the figures show.**

![Membrane surfaces, cases A and B](../img/10-membrane.png)

`docs/img/10-membrane.png`, colored by height (gray = 0). Top: the noisy disk (left) becomes exactly flat (right). Bottom: with saddle-shaped boundary heights, the noisy disk becomes a smooth saddle that continues the surrounding surface. Note the C⁰ joint: the membrane matches the *positions* of the boundary, not its slope. In case B the slope happens to match too, because x² − y² is itself harmonic, but #11 will show the kink in general.

![Convergence of the harmonic case](../img/10-convergence.png)

`docs/img/10-convergence.png`: the maximum error |z − (x² − y²)| on the free disk, measured at the final (x, y), versus grid spacing h, from 11 × 11 to 161 × 161 vertices.
- **Cotangent, L from the flat reference (dark blue):** follows the h² guide; the error drops by about 4× per halving of h (from 2.7·10⁻⁴ to 2·10⁻⁶).
- **Uniform (light blue):** also converges here, but more slowly (about h^1.5) and with 5–12× larger errors; it also slides the free vertices in the plane. The jitter is mild, so the uniform weights are not far off.
- **Cotangent, L from the noisy input (orange):** stuck around 2·10⁻³ at every resolution. The noise is the same relative size at every resolution, so the weight errors never go away.

**Pitfalls.**
- *The weights come from somewhere.* The fairing equations are linear only because the weights are frozen. Freezing them on a damaged surface (noise, a crude hole patch) freezes the damage into the result. Use a clean reference when one exists (a parameter domain, the original surface). When none exists, use the uniform weights, which do not depend on geometry, or iterate (re-solve with weights from the previous result).
- *A smooth-looking result can still be wrong.* The orange case looks perfectly fine in a picture. Only the convergence test against an exact answer reveals that it never converges.
- *Measure the error at the final (x, y).* The free vertices may move in x and y too (uniform, or cotan from a noisy input), so the target height must be evaluated at their new position, not the old one.
