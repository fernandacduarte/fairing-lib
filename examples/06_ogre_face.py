"""Fig. 4.9 on a real head: the ogre's face refilled by a thin plate (k = 2), uniform vs. cotangent.

Data: "Jerry the Ogre" by Keenan Crane (public domain), from Keenan Crane's 3D Model
Repository, https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/ . It is not part of
this repo. The script reads the rest-pose blend shape bs_rest.obj (19985 vertices, a disk:
the back of the head is cut away), by default from ../Example-Meshes/ogre/ next to the repo.

Run from the repository root:

    python examples/06_ogre_face.py [path/to/bs_rest.obj]

As in the book's figure, the whole face is free and solves Delta^2 x = 0 with the rest of the
head fixed. The edge lengths vary by about 10x across the mesh, so the uniform weights (which
ignore edge lengths) pull vertices sideways and wrinkle the patch, while the cotangent patch
stays smooth. Saves docs/img/15-ogre-face.png and prints the numbers behind it.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components

from fairing import mesh, viz
from fairing.fairing import solve_fair
from fairing.laplacian import mean_curvature

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
OGRE = ROOT.parent / "Example-Meshes" / "ogre" / "bs_rest.obj"
K = 2                                             # thin plate, Delta^2 x = 0
FACE = dict(center_z=-0.1, half_width=0.32, half_height=0.45)   # ellipse seen from the front
VIEW = (5, -90)                                   # camera in front of the face (looking along +y)


def load_ogre(path):
    """The ogre as plain V, F, rotated from the file's y-up (face along +z) to z-up (face along -y)."""
    if not path.exists():
        sys.exit(f"{path} not found: download 'Jerry the Ogre' from Keenan Crane's 3D Model Repository "
                 "or pass the path to bs_rest.obj.")
    V, F = mesh.load_obj(path)
    return V[:, [0, 2, 1]] * [1, -1, 1], F


def face_region(V, F):
    """The free region: the face, from forehead to chin, without the ears.

    An ellipse in the front view (the x-z plane) also catches vertices behind the face, so
    keep only its connected piece that contains the tip of the nose (the front-most vertex
    near the middle of the face).
    """
    x, z = V[:, 0], V[:, 2] - FACE["center_z"]
    inside = np.flatnonzero((x / FACE["half_width"]) ** 2 + (z / FACE["half_height"]) ** 2 < 1)
    middle = inside[(np.abs(x[inside]) < 0.05) & (np.abs(V[inside, 2]) < 0.1)]
    nose = middle[np.argmin(V[middle, 1])]
    A = mesh.adjacency(F, len(V)).tocsr()
    _, label = connected_components(A[inside][:, inside], directed=False)
    free = np.zeros(len(V), dtype=bool)
    free[inside[label == label[np.searchsorted(inside, nose)]]] = True

    clearance = mesh.ring_distance(F, mesh.boundary_vertices(F), n=len(V))[free].min()
    assert clearance >= K + 1, "the free region's equations would reach the mesh boundary"
    return free, int(clearance)


def draw_head(ax, V, F, free, title, edges=False):
    """Lit surface like the book's figure: fixed faces gray, faces with a free vertex blue.

    With `edges`, the triangle edges are drawn too, so the irregular triangulation shows.
    """
    free_face = free[F].any(axis=1)
    for faces, color in ((F[~free_face], "#d9d8d4"), (F[free_face], "#7aa0d8")):
        if len(faces) == 0:
            continue
        ax.plot_trisurf(*V.T, triangles=faces, color=color, shade=True, antialiased=False,
                        **(dict(edgecolor=(0.24, 0.24, 0.22, 0.3), linewidth=0.05) if edges else dict(linewidth=0)))
    viz._set_equal_aspect(ax, V)
    ax.view_init(*VIEW)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def main():
    V, F = load_ogre(Path(sys.argv[1]) if len(sys.argv) > 1 else OGRE)
    free, clearance = face_region(V, F)
    results = {lap: solve_fair(V, F, free, K, lap) for lap in ("uniform", "cotan")}

    E, _ = mesh.edge_face_counts(F)
    length = lambda W: np.linalg.norm(W[E[:, 0]] - W[E[:, 1]], axis=1)
    lo, hi = np.percentile(length(V), [1, 99])
    absH = lambda W: np.abs(mean_curvature(W, F))[free]
    print(f"ogre: {len(V)} vertices, {len(F)} triangles; edge lengths {lo:.4f} to {hi:.4f} "
          f"(1st to 99th percentile, ratio {hi / lo:.0f})")
    print(f"free region: {free.sum()} vertices, at least {clearance} rings from the mesh boundary")
    for name, W in (("original", V), *results.items()):
        h = absH(W)
        print(f"  {name:8s}: |H| on the region median {np.median(h):6.1f}, 95th percentile {np.percentile(h, 95):6.1f}"
              + ("" if W is V else f"; max distance to the original {np.abs(W - V)[free].max():.3f}"))

    fig = plt.figure(figsize=(13, 9))
    nothing = np.zeros(len(V), dtype=bool)
    draw_head(fig.add_subplot(2, 3, 2, projection="3d"), V, F, nothing, "the original ogre, with its triangle edges", edges=True)
    panels = (("input mesh; free region in blue", V),
              ("thin plate, uniform weights", results["uniform"]),
              ("thin plate, cotangent weights", results["cotan"]))
    for col, (title, W) in enumerate(panels):
        draw_head(fig.add_subplot(2, 3, col + 4, projection="3d"), W, F, free, title)
    fig.suptitle("The ogre face refilled by a thin plate (k = 2). "
                 "Data: 'Jerry the Ogre', Keenan Crane (public domain).", y=0.99)
    fig.tight_layout()
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / "15-ogre-face.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "15-ogre-face.png")


if __name__ == "__main__":
    main()
