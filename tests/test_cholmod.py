"""The sparse Cholesky solver (CHOLMOD, via scikit-sparse) that the fairing solver uses.

These tests pin down the two behaviors #9 relies on: solving an SPD system for
several right-hand sides at once, and refusing a matrix that is not positive
definite.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from sksparse.cholmod import CholmodNotPositiveDefiniteError, cho_factor

from fairing.laplacian import cotan_laplacian
from fairing.mesh import uv_sphere


def spd_system():
    # The SPD matrix of an implicit smoothing step: D^-1 - h M (App. A.1)
    V, F = uv_sphere(8, 16)
    _, D, M = cotan_laplacian(V, F)
    D_inv = sp.diags(1 / D.diagonal())
    return (D_inv - 0.01 * M).tocsc(), D_inv @ V


def test_cholesky_solves_three_right_hand_sides_at_once():
    A, b = spd_system()
    x = cho_factor(A).solve(b)                    # b is (n, 3): x, y, z in one call
    assert np.allclose(A @ x, b)


def test_cholesky_rejects_a_matrix_that_is_not_positive_definite():
    A, _ = spd_system()
    with pytest.raises(CholmodNotPositiveDefiniteError):    # CHOLMOD also prints a warning to stderr
        cho_factor((-A).tocsc())
