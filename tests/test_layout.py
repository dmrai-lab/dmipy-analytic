"""``dmipy_analytic.jax`` mirrors the numpy modules: every module of forms has its twin module at the same path (the
constants and the private helpers have none)."""
from pathlib import Path

import dmipy_analytic

ROOT = Path(dmipy_analytic.__file__).parent
WITHOUT_TWIN = {"__init__.py", "constants.py", "_extras.py"}


def _modules(root):
    return {p.relative_to(root).as_posix() for p in root.rglob("*.py")}


def test_every_module_of_forms_has_its_jax_twin_module():
    numpy_side = {m for m in _modules(ROOT) if not m.startswith("jax/") and m not in WITHOUT_TWIN}
    jax_side = {m.removeprefix("jax/") for m in _modules(ROOT) if m.startswith("jax/")}
    assert numpy_side - jax_side == set()
    assert jax_side - numpy_side - {"__init__.py"} == set()
