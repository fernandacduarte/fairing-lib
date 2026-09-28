# Phase 1 — Foundations

This phase builds the tools that every later experiment relies on: a package, synthetic meshes, plotting, and connectivity queries. No book equation is implemented here, but the meshes are chosen for the experiments in Chapters 3–4.

## #1 Project setup

**Theory → code.** No equations. The package layout mirrors the book's progression:

| Concept | Module |
|---|---|
| meshes and their connectivity | `fairing/mesh.py` |
| discrete Laplace–Beltrami, `L = D·M` (§3.3.4, App. A.1) | `fairing/laplacian.py` |
| diffusion flow (§4.2) | `fairing/smoothing.py` |
| fairing, `Lᵏx = 0` (§4.3, Eq. A.2) | `fairing/fairing.py` |
| figures | `fairing/viz.py` |

**What the figures show.** None yet.

**Pitfalls.** None so far.
