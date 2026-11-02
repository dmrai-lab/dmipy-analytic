"""The powder average of an oblate zeppelin in JAX, by Gauss-Legendre quadrature, held to its numpy ``erfi`` definition.

The tolerances are derived, not fitted:

* the quadrature error is at most ``OBLATE_TOLERANCE`` relative to the average for ``kappa <= OBLATE_KAPPA_MAX``: the
  proof is in ``_quadrature_bound``, whose bound the tests check in exact arithmetic (50-digit nodes and weights)
  against the 60-digit series of the integral, and the node count is the smallest that holds it;
* the rounding of the float evaluation, relative, with ``u`` the unit roundoff of the dtype and ``x = b D_max`` the
  largest exponent: ``n/2`` terms are summed (``n/2 u``); each carries the rounding of its weight and of ``exp``
  (``2 u`` plus ``2 EXP_ULPS u`` in float32), and its exponent ``b (D_par s_i + D_perp c_i)`` is formed from the
  rounded inputs (``r`` of them, float32 only), the rounded ``s_i`` and ``c_i`` and four operations, so it carries
  ``(r + 6) u`` of itself, which ``exp`` turns into ``(r + 6) u x`` relative;
* the numpy form adds the rounding of ``exp(-b D_perp)``, ``u b D_perp``, and of ``erfi``, 4 ulps.
"""
from decimal import Decimal, localcontext

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy.special")

import jax
import jax.numpy as jnp

from dmipy_analytic import gaussian
from dmipy_analytic.jax import gaussian as jgaussian
from .rounding import EXP_ULPS, U32, U64, x64


def _series(kappa):
    """``J(kappa) = int_0^1 exp(kappa t^2) dt = sum_k kappa^k / ((2 k + 1) k!)`` in 60 digits (all terms positive)."""
    with localcontext() as context:
        context.prec = 60
        k = Decimal(float(kappa))
        term, total, n = Decimal(1), Decimal(0), 0
        while True:
            add = term / (2 * n + 1)
            total += add
            if n > 5 and add < total * Decimal(10) ** -55:
                return total
            n += 1
            term = term * k / n


def _grid():
    """(b, D_par, D_perp) of oblate tensors with kappa from 1e-8 to just below the domain's end."""
    points = []
    for kappa in (1e-8, 1e-3, 0.05, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 63.0):
        for b in (1e9, 5e9, 1.5e10):
            delta = kappa / b
            if delta > 3.2e-9:
                continue
            perp = 0.1e-9 + delta + 0.4e-9
            points.append((b, perp - delta, perp))
    return points


def _tolerance(b, par, perp, unit, rounded_inputs, exp_ulps=0):
    x = b * max(par, perp)
    n_terms = jgaussian._oblate_nodes() // 2
    return (jgaussian.OBLATE_TOLERANCE + unit * (n_terms + 2 + 2 * exp_ulps + (rounded_inputs + 6) * x)
            + U64 * (b * perp + 4))


def test_the_node_count_is_the_smallest_that_holds_the_bound():
    n = jgaussian._oblate_nodes()
    grid = jgaussian.OBLATE_KAPPA_MAX * np.arange(1, 65) / 64.0
    assert n % 2 == 0
    assert max(jgaussian._quadrature_bound(k, n) for k in grid) <= jgaussian.OBLATE_TOLERANCE
    assert max(jgaussian._quadrature_bound(k, n - 2) for k in grid) > jgaussian.OBLATE_TOLERANCE


@pytest.mark.parametrize("kappa", [0.5, 5.0, 20.0, 64.0])
@pytest.mark.parametrize("n_nodes", [10, 20, 40, 58])
def test_the_quadrature_bound_holds_in_exact_arithmetic(kappa, n_nodes):
    with localcontext() as context:
        context.prec = 60
        rule = jgaussian._half_rule_exact(n_nodes)
        approximation = sum(w * (Decimal(float(kappa)) * x * x).exp() for x, w in rule)
        exact = _series(kappa)
        relative = float(abs(approximation - exact) / exact)
    # the nodes and weights are 50-digit numbers: the comparison resolves relative errors down to 1e-44
    assert relative <= jgaussian._quadrature_bound(kappa, n_nodes) + 1e-44


def test_the_rule_is_exact_for_even_polynomials_and_its_weights_sum_to_one():
    squares, complements, weights = jgaussian._half_rule(jgaussian._oblate_nodes())
    n = jgaussian._oblate_nodes()
    assert float(np.sum(weights)) == pytest.approx(1.0, abs=n * U64)
    assert np.allclose(squares + complements, 1.0, rtol=0, atol=U64)
    for k in range(n):
        assert float(np.sum(weights * squares ** k)) * (2 * k + 1) == pytest.approx(1.0, abs=n * U64)


def test_the_oblate_average_equals_the_series_in_float64():
    with x64():
        for b, par, perp in _grid():
            exact = float(Decimal(float(-b * perp)).exp() * _series(b * (perp - par)))
            twin = float(jgaussian.zeppelin_powder_average(jnp.asarray([b]), par, perp)[0])
            assert twin == pytest.approx(exact, rel=_tolerance(b, par, perp, U64, 0), abs=0)


def test_the_oblate_average_equals_the_numpy_erfi_form_in_float64():
    with x64():
        for b, par, perp in _grid():
            twin = float(jgaussian.zeppelin_powder_average(jnp.asarray([b]), par, perp)[0])
            numpy = float(gaussian.zeppelin_powder_average(np.array([b]), par, perp)[0])
            assert twin == pytest.approx(numpy, rel=_tolerance(b, par, perp, U64, 0), abs=0)


def test_the_oblate_average_equals_the_numpy_erfi_form_in_float32():
    for b, par, perp in _grid():
        twin = float(jgaussian.zeppelin_powder_average(jnp.asarray([b], dtype=jnp.float32), par, perp)[0])
        numpy = float(gaussian.zeppelin_powder_average(np.array([b]), par, perp)[0])
        if numpy < np.finfo(np.float32).tiny * 1e3:        # below the float32 range of a result: nothing to compare
            continue
        assert twin == pytest.approx(numpy, rel=_tolerance(b, par, perp, U32, 3, EXP_ULPS), abs=0)


def test_the_average_is_continuous_across_the_ball():
    with x64():
        b = jnp.asarray([2e9, 6e9])
        ball = np.asarray(jgaussian.zeppelin_powder_average(b, 1.5e-9, 1.5e-9))
        for rel in (1e-9, 1e-6, 1e-3):
            oblate = np.asarray(jgaussian.zeppelin_powder_average(b, 1.5e-9 * (1 - rel), 1.5e-9))
            prolate = np.asarray(jgaussian.zeppelin_powder_average(b, 1.5e-9, 1.5e-9 * (1 - rel)))
            assert np.allclose(oblate, ball, rtol=3 * rel * 6e9 * 1.5e-9 + 1e-14, atol=0)
            assert np.allclose(prolate, ball, rtol=3 * rel * 6e9 * 1.5e-9 + 1e-14, atol=0)


def test_beyond_the_domain_is_nan_and_never_a_wrong_number():
    with x64():
        b = jnp.asarray([1e10])
        edge = jgaussian.OBLATE_KAPPA_MAX
        inside = jgaussian.zeppelin_powder_average(b, 7e-9 - (edge - 1e-3) / 1e10, 7e-9)
        outside = jgaussian.zeppelin_powder_average(b, 7e-9 - (edge + 1e-3) / 1e10, 7e-9)
        assert np.isfinite(np.asarray(inside)).all() and np.isnan(np.asarray(outside)).all()


def test_the_gradient_in_the_oblate_region_is_that_of_the_numpy_form():
    with x64():
        b = np.array([2e9, 6e9, 1.2e10])
        par, perp = 0.6e-9, 2.2e-9

        def total(d_par, d_perp):
            return jnp.sum(jgaussian.zeppelin_powder_average(jnp.asarray(b), d_par, d_perp))

        g_par, g_perp = jax.grad(total, argnums=(0, 1))(par, perp)
        h = 1e-4
        n_par = (gaussian.zeppelin_powder_average(b, par * (1 + h), perp).sum()
                 - gaussian.zeppelin_powder_average(b, par * (1 - h), perp).sum()) / (2 * h * par)
        n_perp = (gaussian.zeppelin_powder_average(b, par, perp * (1 + h)).sum()
                  - gaussian.zeppelin_powder_average(b, par, perp * (1 - h)).sum()) / (2 * h * perp)
        assert float(g_par) == pytest.approx(float(n_par), rel=1e-6)
        assert float(g_perp) == pytest.approx(float(n_perp), rel=1e-6)
