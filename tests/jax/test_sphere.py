"""The JAX twins of ``dmipy_analytic.restricted.sphere`` against their float64 numpy definitions.

Tolerances, derived from the rounding of each form (``U32 = 2**-24``, ``U64 = 2**-53``):

* ``sgp_long_time`` in float32: ``x = 2 pi q R`` has the relative error ``5 U32``; the bracket ``sin x / x - cos x``
  cancels by the factor ``x cos... `` of its terms, at most ``|sin x / x| + |cos x|`` over ``|sin x / x - cos x|`` which
  the test evaluates at each point (``x`` in ``[0.5, 4]``, away from the zero at ``x = 4.49``); the bracket's error is
  that factor times ``(5 + 4) U32`` and the square doubles it; the quotient by ``x^2`` adds ``13 U32``.
* ``gaussian_phase`` in float32: the sum over 100 roots of positive terms ``~ alpha^-6`` (the bracket
  ``2 delta - (...)/(a D)`` does not cancel for ``a D delta >= 5``): the exponent has the relative error
  ``(14 + 100) U32``; the exponential adds ``|x|`` times that and 4 ulps (``8 U32``).
* ``gaussian_phase_waveform`` in float64: the recursion accumulates a relative error of ``(3 n_t + 100) U64`` and
  the exponential ``|x|`` times it.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy.special")

from dmipy_analytic.restricted import sphere
from dmipy_analytic.jax.restricted import sphere as jsphere
from .rounding import U32, U64, x64

D = 2e-9


def test_sgp_long_time():
    d = 8e-6
    x = np.linspace(0.5, 4.0, 40)
    q = x / (np.pi * d)
    twin = np.asarray(jsphere.sgp_long_time(q, d), dtype=np.float64)
    form = sphere.sgp_long_time(q, d)
    cancellation = (np.abs(np.sin(x) / x) + np.abs(np.cos(x))) / np.abs(np.sin(x) / x - np.cos(x))
    rtol = 2 * cancellation * 9 * U32 + 13 * U32
    assert np.all(np.abs(twin - form) <= rtol * form)
    assert float(jsphere.sgp_long_time(np.zeros(1), d)[0]) == 1.0


def test_gaussian_phase():
    d = 6e-6
    G = np.array([0.02, 0.05, 0.1, 0.2])
    delta, Delta = np.array([0.01, 0.01, 0.014, 0.014]), np.array([0.03, 0.03, 0.04, 0.04])
    twin = np.asarray(jsphere.gaussian_phase(G, delta, Delta, d, D), dtype=np.float64)
    form = sphere.gaussian_phase(G, delta, Delta, d, D)
    x = -np.log(form)
    assert np.all(np.abs(twin - form) <= (x * (14 + 100) + 8) * U32 * form)


def test_gaussian_phase_waveform():
    d, dt, n_t = 6e-6, 1e-4, 150
    t = np.arange(n_t) * dt
    G = np.stack([0.06 * np.cos(2 * np.pi * 80 * t), 0.04 * np.sin(2 * np.pi * 80 * t),
                  0.05 * np.ones_like(t)], axis=1)
    with x64():
        twin = float(jsphere.gaussian_phase_waveform(G, dt, d, D, n_roots=20))
        single = float(jsphere.gaussian_phase_waveform(G[:, 0], dt, d, D, n_roots=20))
    form = sphere.gaussian_phase_waveform(G, dt, d, D, n_roots=20)
    x = -np.log(form)
    assert abs(twin - form) <= (x * (3 * n_t + 100) + 2) * U64 * form
    form1 = sphere.gaussian_phase_waveform(G[:, 0], dt, d, D, n_roots=20)
    assert abs(single - form1) <= (-np.log(form1) * (3 * n_t + 100) + 2) * U64 * form1
