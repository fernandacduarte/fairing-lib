"""Smoke test: the package and all its modules import."""

import importlib

import pytest

MODULES = ["fairing", "fairing.mesh", "fairing.laplacian", "fairing.smoothing", "fairing.fairing", "fairing.viz"]


@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    importlib.import_module(name)
