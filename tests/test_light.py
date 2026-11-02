"""The package and every numpy module import with numpy alone: no scipy, jax or torch is loaded by importing them."""
import subprocess
import sys


def test_the_import_is_numpy_only():
    code = ("import sys, dmipy_analytic; heavy = [m for m in ('scipy', 'jax', 'torch') if m in sys.modules]; "
            "assert not heavy, heavy")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_every_numpy_module_imports_with_numpy_alone():
    code = ("import importlib, pkgutil, sys, dmipy_analytic\n"
            "names = [m.name for m in pkgutil.walk_packages(dmipy_analytic.__path__, 'dmipy_analytic.')\n"
            "         if not m.name.startswith('dmipy_analytic.jax')]\n"
            "assert 'dmipy_analytic.restricted.matrix_method' in names, names\n"
            "for name in names:\n"
            "    importlib.import_module(name)\n"
            "heavy = [m for m in ('scipy', 'jax', 'torch') if m in sys.modules]\n"
            "assert not heavy, heavy\n")
    subprocess.run([sys.executable, "-c", code], check=True)
