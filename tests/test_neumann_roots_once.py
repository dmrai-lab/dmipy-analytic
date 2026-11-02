"""The Neumann roots of the pores (the zeros of ``J_m'`` and of ``j_l'``) are found once, by
:func:`dmipy_analytic.restricted.matrix_method.cylinder_neumann_roots` and ``sphere_neumann_roots``; every form that
sums over them calls those, and no other module of the package solves for a root or tabulates one."""
import pathlib
import re

import dmipy_analytic

ROOT_FINDERS = re.compile(r"\b(brentq|bisect|newton|fsolve|root_scalar|jn_zeros|jnp_zeros|jnyn_zeros)\b")


def test_only_the_matrix_method_solves_for_roots():
    package = pathlib.Path(dmipy_analytic.__file__).parent
    offenders = []
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package).as_posix()
        if relative == "restricted/matrix_method.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if ROOT_FINDERS.search(line):
                offenders.append(f"{relative}:{number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_the_sphere_forms_sum_over_the_shared_roots():
    """The Callaghan series of the sphere and its Gaussian phase read the roots of ``sphere_neumann_roots``."""
    from dmipy_analytic.restricted import sphere
    from dmipy_analytic.restricted.matrix_method import sphere_neumann_roots
    import numpy as np
    alpha = sphere._callaghan_roots(6, 4)
    for l in range(4):
        assert np.array_equal(alpha[l], sphere_neumann_roots(l, 6))
    assert np.array_equal(sphere._gaussian_phase_roots(30), sphere_neumann_roots(1, 30))
