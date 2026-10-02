# Phase 2 — Discrete Laplace–Beltrami (Ch. 3)

Everything in Chapters 4 and App. A rests on one matrix: a discrete version of the Laplace–Beltrami operator Δ. Applied to the vertex positions it should give the mean curvature normal, Δx = −2H·n (Eq. 3.7). This phase builds two discretizations and tests them against that promise.

Both share the form of App. A.1:

  Δf(vᵢ) = wᵢ · Σⱼ wᵢⱼ (fⱼ − fᵢ),  stored as **L = D·M**,

with D = diag(wᵢ) (per-vertex normalization) and M symmetric (mᵢⱼ = wᵢⱼ on edges, mᵢᵢ = −Σⱼ wᵢⱼ). Each row of M sums to 0, so a constant function has zero Laplacian. M stays symmetric even though L is not, which the solvers of Phases 3–4 rely on.

## #5 Uniform Laplacian

**Theory → code.** With wᵢⱼ = 1 and wᵢ = 1/deg(vᵢ), Eq. 3.10 becomes Δxᵢ = (centroid of the neighbors) − xᵢ. The operator only looks at connectivity.

| Book | Formula | Code |
|---|---|---|
| Eq. 3.10, §4.1.2 | wᵢⱼ = 1 on every edge | `fairing/laplacian.py:37` |
| Eq. 3.10 | deg(vᵢ) = \|N₁(vᵢ)\| | `laplacian.py:38` |
| App. A.1 | M = W − diag(Σⱼ wᵢⱼ) | `laplacian.py:39` |
| App. A.1 | D = diag(1/deg) | `laplacian.py:40` |
| App. A.1 | L = D·M | `laplacian.py:41` |

Tests (`tests/test_laplacian.py`): M is symmetric with zero row sums, L = D @ M, and (Lx)ᵢ equals the neighbor centroid minus xᵢ. ‖Lx‖ = 0 at the interior of the regular grid, while on the irregular grid it is non-zero at every interior vertex and purely in-plane.

**What the figures show.**

![Uniform Laplacian on regular and irregular planar grids](../img/05-uniform-Lx.png)

`docs/img/05-uniform-Lx.png` shows three views of flat grids, from above. Left: on the regular grid, ‖Lx‖ = 0 at every interior vertex, because each neighbor has a mirror partner. Middle: on the irregular grid, same scale, ‖Lx‖ reaches about 25% of the grid spacing, although the surface is perfectly flat and its mean curvature is 0. Right: the Lx vectors (×3) themselves. They lie *in* the plane and point to each vertex's neighbor centroid, so this "curvature" is pure tangential error. Gray faces touch the boundary, where Lx is excluded (see Pitfalls).

**Pitfalls.**
- *The boundary is always "wrong".* A boundary vertex has all its neighbors on one side, so even on the regular grid its Lx points inward with length about h/2. Including it would swamp the color scale, so figures mark boundary values as NaN (gray).
- *The uniform Lx is not a curvature estimate.* It has units of length, not 1/length. On irregular grids with fixed relative jitter, mean ‖Lx‖ ≈ 0.12·h at every resolution (0.0106, 0.0050, 0.0026 for h = 1/12, 1/24, 1/48). On the unit sphere it gives ‖Lx‖ ≈ 0.004 where 2H = 2. The missing piece is the division by an area, which the cotangent version adds in #6.
- *"Zero on the regular grid" depends on symmetry, not on flatness.* It holds because every neighbor has a mirror partner. The irregular grid is just as flat but loses that symmetry.
