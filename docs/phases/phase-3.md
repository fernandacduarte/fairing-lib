# Phase 3 — Diffusion flow (§4.2)

Smoothing treats the vertex positions like heat: the diffusion equation ∂f/∂t = λΔf (Eq. 4.5) evens out high-frequency bumps. Discretizing Δ with a Laplace matrix gives one ODE per vertex, ∂x/∂t = λLx (Eq. 4.6). Discretizing time with steps h gives an update rule. With the cotangent Laplacian, Lx ≈ −2H·n (Eq. 3.7), so each vertex moves along its normal by an amount proportional to its mean curvature: this is *mean curvature flow*.

## #7 Explicit Laplacian smoothing

**Theory → code.** Explicit Euler evaluates the right-hand side at the current time: x ← x + hλ·Lx (§4.2, p. 55–56). To see when this is stable, write x in the eigenvectors of L. L = D·M is similar to the symmetric matrix D^½ M D^½, so its eigenvalues μ are real and ≤ 0. One step multiplies each component by 1 + hλμ. It stays bounded only if |1 + hλμ| ≤ 1 for every μ, i.e. **hλ ≤ 2/|μ_min|**.

| Book | Formula | Code |
|---|---|---|
| §4.2, Eq. 4.6 | x ← x + hλ·Lx, L rebuilt from the current x | `explicit_smoothing`, `fairing/smoothing.py:29` |
| §4.2 ("sufficiently small h") | h ≤ 2/(λ\|μ_min\|), μ_min from the symmetric D^½MD^½ | `explicit_step_limit`, `smoothing.py:48–49` |
| — | noise measure: mean angle between adjacent face normals | `roughness`, `smoothing.py:52` |

For the uniform Laplacian μ_min ≥ −2, so hλ ≤ 1 is always safe. hλ = 1 moves each vertex exactly to its neighbor centroid, which a test checks. For the cotangent Laplacian, |μ_min| grows like 1/(edge length)² and is set by the worst triangles.

Tests (`tests/test_smoothing.py`): the limit matches a dense eigenvalue computation; at hλ = 1 the uniform step lands on the centroid; smoothing removes most of the noise's excess roughness with either Laplacian; 0.95 × limit stays bounded for 300 steps, while 1.05 × limit grows past 10³.

**What the figures show.**

![Explicit smoothing, iterations 0, 10, 100](../img/07-explicit-iters.png)

`docs/img/07-explicit-iters.png`: a noisy unit sphere (σ = 0.01) smoothed at 0.9 × each operator's step limit, colored by H (true value 1).
- **Uniform (top),** h = 1.2: the noise is gone after 10 steps (roughness 0.274 → 0.074, the clean sphere's own value). By step 100 the sphere has shrunk to 0.61 × 0.61 × 0.93, from 2 × 2 × 2, and is no longer round.
- **Cotangent (bottom),** h = 2.7·10⁻⁵: the stable step is about 45 000× smaller. After 100 steps the noise is only partly removed (roughness 0.100), and the radius has barely changed (0.994).

Each panel is zoomed to fit its mesh, so shrinking is visible only in the numbers.

![Explicit smoothing blows up above the step limit](../img/07-explicit-unstable.png)

`docs/img/07-explicit-unstable.png`: the largest coordinate over 60 uniform steps. At 0.5× and 0.9× the limit the mesh slowly shrinks. At 1.1× the limit the worst mode is multiplied by |1 − 1.1·2| = 1.2 per step. It is invisible at first (it starts as a tiny part of the noise), then grows exponentially (a straight line on the log axis) and tears the mesh apart (left, step 32).

**Pitfalls.**
- *The cotangent flow is stiff.* Its stable step is dictated by the thinnest triangles (here the UV-sphere poles), not by the noise you want to remove. Explicit cotangent smoothing therefore needs very many tiny steps. This is the book's reason for implicit integration (#8).
- *Laplacian smoothing shrinks.* For mean curvature flow on a sphere, dr/dt = −2λH = −2λ/r, so r² = r₀² − 4λt. Our cotangent run gives exactly that (r = 0.994 after t = 0.0027). The uniform flow is not mean curvature flow: its speed depends on local edge lengths, which vary over the UV sphere. So it shrinks much faster and unevenly, and the sphere turns into an elongated shape.
- *The limit depends on the mesh.* It must be computed for each mesh (`explicit_step_limit`); a value from another mesh, even a slightly noisier one, may be unsafe. It can also drift while the mesh evolves, since L is rebuilt every step.
- *The obvious noise measure failed.* The spread of the vertex radii *increased* under uniform smoothing, because the shape stops being a sphere even as the noise vanishes. A local measure (angles between neighboring normals) separates "noise" from "shape". On coarse meshes, compare it with the clean mesh's value, since faceting alone makes it non-zero.
- *A capped color scale.* The noisy H has a long tail (up to ~40), so the figure caps the scale at 4. Without the cap, every panel looks equally pale.
