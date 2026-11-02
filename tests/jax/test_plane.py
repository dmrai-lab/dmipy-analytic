"""The JAX twins of ``dmipy_analytic.restricted.plane`` against their float64 numpy definitions.

Tolerances, derived from the rounding of each form (``U32 = 2**-24``, ``U64 = 2**-53``):

* ``sgp_long_time`` in float32, ``a = 2 pi q d`` of relative error ``5 U32`` (two rounded inputs, the rounded ``pi``,
  two products), ``cos a`` of absolute error ``4 U32`` (two ulps), so ``1 - cos a`` has the relative error
  ``(5 a |sin a| + 4) U32 / (1 - cos a)``; dividing by ``a^2`` (error ``11 U32``) and the product add ``13 U32``. The
  test keeps ``a`` in ``[0.5, 3]``, away from the zero of the pattern at ``a = 2 pi`` where the difference cancels, and
  evaluates the bound at each point.
* ``sgp`` in float64: a term of the series is a chain of about 40 operations (an exponential, a quotient, two trigonometric
  products, a square and a quotient) and the 80 terms are positive, so the sum has the relative error of its worst term
  plus ``80 U64`` for the summation. The denominator ``(a^2 - xi^2)^2`` amplifies the relative error ``5 U64`` of ``a``
  by ``4 a^2 / |a^2 - xi^2|``; the test evaluates that factor at each term and takes the largest, ``K``, so
  ``rtol = (40 K + 80) U64``.
* ``gaussian_phase`` in float32: the bracket of a mode is a sum of terms of at most two times its size
  (``delta / tau > 1`` for the first mode, so ``2 lambda delta - 2`` does not cancel by more than a factor
  ``1 / (1 - 1/(lambda delta))``), the 100 modes add terms of one sign, and the exponent of the result is the sum:
  the relative error of the exponent is ``(12 + 100) U32`` with a conditioning of at most 1.5, and the exponential adds
  ``|x|`` times that and 4 ulps (``8 U32``).
"""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy.special")

from dmipy_analytic.restricted import plane
from dmipy_analytic.jax.restricted import plane as jplane
from .rounding import U32, U64, x64


def test_sgp_long_time():
    d = 6e-6
    a = np.linspace(0.5, 3.0, 40)
    q = a / (2 * np.pi * d)
    twin = np.asarray(jplane.sgp_long_time(q, d), dtype=np.float64)
    form = plane.sgp_long_time(q, d)
    rtol = ((5 * a * np.abs(np.sin(a)) + 4) / (1 - np.cos(a)) + 13) * U32
    assert np.all(np.abs(twin - form) <= rtol * form)
    assert float(jplane.sgp_long_time(np.zeros(1), d)[0]) == 1.0


def test_sgp():
    d, D = 5e-6, 2e-9
    radius = d / 2
    a = (np.arange(10) + 0.5) * np.pi / 2                     # midway between the resonances (multiples of pi / 2)
    q = a / (2 * np.pi * radius)
    tau = 0.003
    xi = np.concatenate([np.arange(40) * np.pi, np.arange(40) * np.pi + np.pi / 2])
    K = np.max(4 * a[:, None] ** 2 / np.abs(a[:, None] ** 2 - xi[None, :] ** 2))
    with x64():
        twin = np.asarray(jplane.sgp(q, tau, d, D), dtype=np.float64)
        form = plane.sgp(q, tau, d, D)
    assert np.all(np.abs(twin - form) <= (40 * max(K, 1.0) + 80) * U64 * form)


def test_gaussian_phase():
    d, D = 4e-6, 2e-9
    G = np.array([0.02, 0.05, 0.1, 0.2])
    delta, Delta = np.array([0.01, 0.01, 0.014, 0.014]), np.array([0.03, 0.03, 0.04, 0.04])
    twin = np.asarray(jplane.gaussian_phase(G, delta, Delta, d, D), dtype=np.float64)
    form = plane.gaussian_phase(G, delta, Delta, d, D)
    x = -np.log(form)
    rtol = (x * (12 + 100) * 1.5 + 8) * U32
    assert np.all(np.abs(twin - form) <= rtol * form)
