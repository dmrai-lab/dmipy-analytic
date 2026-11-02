"""``cylinder_neumann_roots`` and ``sphere_neumann_roots`` return exactly ``n_roots`` roots of ``J_m'`` and ``j_l'``, in
ascending order, from a single root up; for the order 0 the first is the constant mode at 0."""
import numpy as np
import pytest

pytest.importorskip("scipy.special")
from scipy import special

from dmipy_analytic.restricted.matrix_method import cylinder_neumann_roots, sphere_neumann_roots


@pytest.mark.parametrize("n_roots", [1, 2, 3, 8])
@pytest.mark.parametrize("m", [0, 1, 2, 5])
def test_the_roots_of_a_cylinder_mode(m, n_roots):
    roots = cylinder_neumann_roots(m, n_roots)
    assert roots.shape == (n_roots,)
    assert np.all(np.diff(roots) > 0)
    nonzero = roots[roots > 0]
    assert len(nonzero) == n_roots - (1 if m == 0 else 0)
    assert special.jvp(m, nonzero) == pytest.approx(0.0, abs=1e-9)
    if m == 0:
        assert roots[0] == 0.0


@pytest.mark.parametrize("n_roots", [1, 2, 3, 8])
@pytest.mark.parametrize("l", [0, 1, 2, 5])
def test_the_roots_of_a_sphere_mode(l, n_roots):
    roots = sphere_neumann_roots(l, n_roots)
    assert roots.shape == (n_roots,)
    assert np.all(np.diff(roots) > 0)
    nonzero = roots[roots > 0]
    assert len(nonzero) == n_roots - (1 if l == 0 else 0)
    assert special.spherical_jn(l, nonzero, derivative=True) == pytest.approx(0.0, abs=1e-9)
    if l == 0:
        assert roots[0] == 0.0


@pytest.mark.parametrize("function", [cylinder_neumann_roots, sphere_neumann_roots])
@pytest.mark.parametrize("order", [0, 2])
def test_a_count_below_one_is_refused(function, order):
    with pytest.raises(ValueError, match="at least 1"):
        function(order, 0)
