# fairing-lib

A small Python library for studying **mesh fairing** as presented in *Polygon Mesh Processing* (Botsch et al.), Chapters 3–4 and Appendix A. Each step is small, cites the equation it implements, and comes with a test and a figure. See [PLAN.md](PLAN.md) for the roadmap and [docs/PROGRESS.md](docs/PROGRESS.md) for the current status.

## Install

**1. SuiteSparse (once per machine).** The fairing solver uses a sparse Cholesky factorization from [CHOLMOD](https://github.com/DrTimothyAldenDavis/SuiteSparse), through the Python package `scikit-sparse`. It has no prebuilt wheels, so pip compiles it against the SuiteSparse C library, which must be installed first.

- macOS (Homebrew):

  ```bash
  brew install suite-sparse
  ```

- Ubuntu / Debian:

  ```bash
  sudo apt-get install libsuitesparse-dev
  ```

**2. The package, in a virtual environment.**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On macOS, tell the build where Homebrew put SuiteSparse. The last line works around a Command Line Tools glitch where `clang++` cannot find the standard C++ headers (`fatal error: 'complex' file not found`):

```bash
export SUITESPARSE_INCLUDE_DIR="$(brew --prefix suite-sparse)/include/suitesparse"
export SUITESPARSE_LIBRARY_DIR="$(brew --prefix suite-sparse)/lib"
export CPLUS_INCLUDE_PATH="$(xcrun --show-sdk-path)/usr/include/c++/v1"
```

On Linux these are not needed. Then:

```bash
pip install -e ".[dev]"
pytest
```

## The Stanford bunny

Phase 5 uses the Stanford Bunny from the [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/) (Stanford Computer Graphics Laboratory). It is **not** included in this repository. The repository allows using and redistributing it for free for research, with credit to the Stanford Computer Graphics Laboratory, but not for commercial use. Download it into the git-ignored `data/` folder:

```bash
mkdir -p data
curl -L -o data/bunny.tar.gz https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz
tar -xzf data/bunny.tar.gz -C data bunny/reconstruction
```

The examples read `data/bunny/reconstruction/bun_zipper_res2.ply` (a decimated version with about 8k vertices) with `trimesh`:

```bash
pip install -e ".[mesh]"
python examples/05_bunny.py
```

The tests do not need the download; the one test that uses the bunny is skipped when the file is missing.

## Viewing meshes interactively with polyscope

The figures in `docs/img/` are static matplotlib PNGs. To rotate, zoom, and inspect a mesh, use [polyscope](https://polyscope.run/py/), an optional viewer. It also lights colored surfaces, so their shape stays readable, which matplotlib cannot do.

**1. Install it** (once, inside `.venv`):

```bash
pip install -e ".[viewer]"
```

**2. Run an example.** Each one opens a window; close it, or press `Esc`, to return to the terminal.

| Script | What it shows |
|---|---|
| `python examples/01_polyscope_one_mesh.py` | the UV sphere colored by its height `z` |
| `python examples/01_polyscope_all_meshes.py` | the four synthetic meshes side by side, with edges |
| `python examples/04_polyscope_elbow.py` | the two-pipe elbow of Fig. 4.8: start and results for k = 1, 2, 3 (fixed gray, free blue), smoothly shaded |
| `python examples/04_polyscope_fig49.py` | Fig. 4.9 setting (#12): graded mesh, exact surface, uniform and cotangent thin plates, with curvature, error and sideways-slide quantities |
| `python examples/05_polyscope_bunny.py` | the Stanford bunny (#14): original, damaged inputs and thin-plate refills, with free region and curvature quantities (needs the download below) |

**3. In the window:** drag with the left mouse button to rotate, drag with the right button to pan, and scroll to zoom. The left panel lists the meshes: use the checkboxes to show or hide them, and open a mesh's entry to change its color, edges, or scalar colormap.

**4. In your own code**, `viz.show_polyscope` opens any mesh, optionally colored by one value per vertex:

```python
from fairing import mesh, viz

V, F = mesh.uv_sphere(16, 24)
viz.show_polyscope(V, F, scalars=V[:, 2])   # scalars: one value per vertex, or None
```

To show several meshes in one window, call polyscope directly, as `examples/01_polyscope_all_meshes.py` does: `ps.init()`, then `ps.register_surface_mesh(name, V, F)` for each mesh, then `ps.show()`.

**Example: refill a region of the bunny and look at it.** After downloading the bunny (above):

```python
import numpy as np
from fairing import mesh, viz
from fairing.fairing import solve_fair

V, F = mesh.load_mesh("data/bunny/reconstruction/bun_zipper_res2.ply")
free = np.zeros(len(V), dtype=bool)
free[mesh.k_ring(F, [3649], 6, n=len(V))] = True      # a 6-ring region on the bunny's side
damaged = mesh.flatten_onto_border_plane(V, F, free)   # crude patch, to be refilled
W = solve_fair(damaged, F, free, 2)                    # thin plate, cotangent weights
viz.show_polyscope(W, F, scalars=free.astype(float))   # the refilled region in color
```

polyscope is never needed by the tests or by CI.
