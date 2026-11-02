"""Every module of ``dmipy_analytic.jax`` imports where the ``jax`` extra is installed."""
import importlib
import pkgutil

import pytest

pytest.importorskip("jax")


def test_every_jax_module_imports():
    import dmipy_analytic.jax as twins
    names = [m.name for m in pkgutil.walk_packages(twins.__path__, "dmipy_analytic.jax.")]
    assert "dmipy_analytic.jax.restricted.matrix_method" in names, names
    for name in names:
        importlib.import_module(name)
