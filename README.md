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

**3. In the window:** drag with the left mouse button to rotate, drag with the right button to pan, and scroll to zoom. The left panel lists the meshes: use the checkboxes to show or hide them, and open a mesh's entry to change its color, edges, or scalar colormap.

**4. In your own code**, `viz.show_polyscope` opens any mesh, optionally colored by one value per vertex:

```python
from fairing import mesh, viz

V, F = mesh.uv_sphere(16, 24)
viz.show_polyscope(V, F, scalars=V[:, 2])   # scalars: one value per vertex, or None
```

To show several meshes in one window, call polyscope directly, as `examples/01_polyscope_all_meshes.py` does: `ps.init()`, then `ps.register_surface_mesh(name, V, F)` for each mesh, then `ps.show()`.

polyscope is never needed by the tests or by CI.
