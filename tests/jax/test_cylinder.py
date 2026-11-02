"""The JAX twins of ``dmipy_analytic.restricted.cylinder`` against their float64 numpy definitions.

Tolerances, derived from the rounding of each form (``U64 = 2**-53``; ``U32 = 2**-24``). The twins that contain Bessel
functions or sums of hundreds of terms of ``alpha^-6`` are evaluated in float64.

* ``sgp_long_time``: ``x = 2 pi q R`` has the relative error ``5 U64``; ``jax.scipy.special.bessel_jn`` has the absolute
  error ``BESSEL_OPERATIONS U64`` on the unit scale of the Bessel functions; for ``x`` in ``[0.5, 3]`` ``J_1 >= 0.24``,
  so ``E = (2 J_1 / x)^2`` has the relative error ``2 (BESSEL_OPERATIONS U64 / 0.24 + 5 U64) + 6 U64``.
* ``sgp``: the series has 1000 positive terms, each a chain of about 30 operations with ``J_m'`` from two Bessel
  values; every Bessel value contributes the absolute error ``BESSEL_OPERATIONS U64``, so the series has the absolute
  error ``1000 * 16 * 2 * BESSEL_OPERATIONS U64`` on the unit scale (the coefficient ``8 alpha^2 / (alpha^2 - m^2)``
  is at most 16 beyond the first mode) plus ``(30 + 1000) U64`` relative.
* ``gaussian_phase``: the 100 roots give positive terms ``~ alpha^-6`` with a cancellation in the numerator
  ``2 a D delta - 2 + ...`` of at most the factor ``2 a D delta / (2 a D delta - 2) <= 1.6`` for ``a D delta >= 5``,
  a chain of 25 operations and a sum of 100 terms: the exponent has the relative error ``1.6 (25 + 100) U64``, the
  exponential ``|x|`` times that and ``2 U64``.
* ``gaussian_phase_waveform``: the recursion is a sum of ``n_t`` terms for each of 100 roots; its relative error is
  ``(3 n_t + 100) U64`` at the largest (a recursion of two operations per step, accumulating).
* ``gaussian_phase_cosine``, ``geometry_factor``, ``gaussian_phase_gamma_lm`` and ``gaussian_phase_exponent_coefficients``
  are closed forms of at most 40 operations over 100 positive terms: ``(40 + 100) U64`` times the exponent.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy.special")

from dmipy_analytic.restricted import cylinder
from dmipy_analytic.jax.restricted import cylinder as jcylinder
from .rounding import BESSEL_OPERATIONS, U64, x64

D = 2e-9


def test_sgp_long_time():
    d = 6e-6
    x = np.linspace(0.5, 3.0, 30)
    q = x / (np.pi * d)
    with x64():
        twin = np.asarray(jcylinder.sgp_long_time(q, d), dtype=np.float64)
    form = cylinder.sgp_long_time(q, d)
    rtol = 2 * (BESSEL_OPERATIONS * U64 / 0.24 + 5 * U64) + 6 * U64
    assert np.all(np.abs(twin - form) <= rtol * form)


def test_sgp():
    d = 5e-6
    q = np.array([0.2e5, 0.5e5, 0.9e5, 1.3e5, 1.8e5])
    tau = np.array([0.001, 0.003, 0.006, 0.01, 0.02])
    with x64():
        twin = np.asarray(jcylinder.sgp(q, tau, d, D), dtype=np.float64)
    form = cylinder.sgp(q, tau, d, D)
    atol = 1000 * 16 * 2 * BESSEL_OPERATIONS * U64
    assert np.all(np.abs(twin - form) <= atol + (30 + 1000) * U64 * form)


def test_gaussian_phase():
    d = 3e-6
    G = np.array([0.02, 0.06, 0.12, 0.2])
    delta = np.array([0.008, 0.008, 0.012, 0.012])
    Delta = np.array([0.02, 0.02, 0.03, 0.03])
    with x64():
        twin = np.asarray(jcylinder.gaussian_phase(G, delta, Delta, d, D), dtype=np.float64)
    form = cylinder.gaussian_phase(G, delta, Delta, d, D)
    x = -np.log(form)
    assert np.all(np.abs(twin - form) <= (x * 1.6 * 125 + 2) * U64 * form)
    # a scalar pulse timing broadcasts the same
    with x64():
        scalar = np.asarray(jcylinder.gaussian_phase(G, 0.01, 0.03, d, D), dtype=np.float64)
    assert np.all(np.abs(scalar - cylinder.gaussian_phase(G, 0.01, 0.03, d, D)) <= 1e-9)


def test_gaussian_phase_waveform_and_cosine():
    d, dt, n_t = 4e-6, 1e-4, 200
    t = (np.arange(n_t) + 0.5) * dt
    g = 0.08 * np.cos(2 * np.pi * 100.0 * t)
    with x64():
        twin = float(jcylinder.gaussian_phase_waveform(g, dt, d, D, n_roots=30))
        cosine = float(jcylinder.gaussian_phase_cosine(0.08, 2 * np.pi * 100.0, n_t * dt, d, D, n_roots=30))
    form = cylinder.gaussian_phase_waveform(g, dt, d, D, n_roots=30)
    x = -np.log(form)
    assert abs(twin - form) <= (x * (3 * n_t + 100) + 2) * U64 * form
    closed = cylinder.gaussian_phase_cosine(0.08, 2 * np.pi * 100.0, n_t * dt, d, D, n_roots=30)
    assert abs(cosine - closed) <= (x * 140 + 2) * U64 * closed


def test_fast_eigenmode_forms():
    d = 2e-6
    gamma_lm = np.array([[0.7, 0.1, -0.2, 0.3, 0.05, -0.1], [1.4, 0.0, 0.0, 0.0, 0.0, 0.0]]) * 1e-4
    b_par = np.array([4e8, 1e9])
    axis = np.array([0.36, -0.48, 0.8])
    with x64():
        twin = np.asarray(jcylinder.gaussian_phase_gamma_lm(gamma_lm, b_par, axis, d, D, 1.7e-9), dtype=np.float64)
        c0, c2m = (np.asarray(x, dtype=np.float64)
                   for x in jcylinder.gaussian_phase_exponent_coefficients(gamma_lm, d, D))
        factor = float(jcylinder.geometry_factor(d, D))
    form = cylinder.gaussian_phase_gamma_lm(gamma_lm, b_par, axis, d, D, 1.7e-9)
    x = -np.log(form)
    assert np.all(np.abs(twin - form) <= (x * 140 + 2) * U64 * form)
    f0, f2 = cylinder.gaussian_phase_exponent_coefficients(gamma_lm, d, D)
    assert np.all(np.abs(c0 - f0) <= 140 * U64 * np.abs(f0)) and np.all(np.abs(c2m - f2) <= 140 * U64 * np.abs(f2) + 1e-300)
    assert abs(factor - cylinder.geometry_factor(d, D)) <= 140 * U64 * factor


def test_capped_forms():
    q_perp, q_par = np.array([0.4e5, 0.9e5]), np.array([0.7e5, 0.2e5])
    d, L, tau = 4e-6, 12e-6, 0.01
    with x64():
        a = np.asarray(jcylinder.capped_sgp_long_time(q_perp, q_par, d, L), dtype=np.float64)
        b = np.asarray(jcylinder.capped_sgp(q_perp, q_par, tau, d, L, D), dtype=np.float64)
    fa = cylinder.capped_sgp_long_time(q_perp, q_par, d, L)
    fb = cylinder.capped_sgp(q_perp, q_par, tau, d, L, D)
    assert np.all(np.abs(a - fa) <= 1e-12 * fa)
    assert np.all(np.abs(b - fb) <= (1000 * 16 * 2 * BESSEL_OPERATIONS + 4000) * U64 * np.maximum(fb, 1e-3))
