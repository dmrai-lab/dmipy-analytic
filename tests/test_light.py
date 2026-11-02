"""The package imports with numpy alone: no scipy, jax or torch is loaded by `import dmipy_analytic`."""
import subprocess
import sys


def test_the_import_is_numpy_only():
    code = ("import sys, dmipy_analytic; heavy = [m for m in ('scipy', 'jax', 'torch') if m in sys.modules]; "
            "assert not heavy, heavy")
    subprocess.run([sys.executable, "-c", code], check=True)
