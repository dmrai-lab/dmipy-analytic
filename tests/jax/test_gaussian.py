"""The JAX twins of ``dmipy_analytic.gaussian`` against their float64 numpy definitions, in float32.

The tolerance of every test is derived from the rounding of the form (``rounding.exp_rtol``), counted per form:

* a twin ``exp(-x)`` has a relative error of ``|x| e + 2 EXP_ULPS U32`` for ``x`` of relative error ``e``;
* ``ball``: ``x = b D`` from two rounded inputs and one product, ``e = 3 U32``;
* ``stick``: ``x = b D c^2`` with ``c = n . u`` (a sum of three products of rounded inputs, absolute error at most
  ``5 U32`` for unit vectors), so the absolute error of ``x`` is at most ``b D (2 |c| 5 U32 + U32) + 4 U32 b D c^2``,
  at most ``15 U32 b D``;
* ``zeppelin``: the same for both terms of ``x = b (D_par c^2 + D_perp (|n|^2 - c^2))``, a sum of two nonnegative terms:
  at most ``20 U32 x_max`` (the difference ``|n|^2 - c^2`` is of unit-vector size and carries ``8 U32`` absolutely);
* the b-tensor forms: ``x`` is a sum of nine products of rounded inputs: ``(2 + 9 + 8) U32 x_max`` for a b-tensor
  with entries of the size of ``b``;
* ``time_dependent_diffusivity``: ``ln(Delta/delta) + 3/2`` and ``Delta - delta/3`` carry ``(4 + 3) U32`` relative
  (one quotient of rounded inputs, one logarithm of 2 ulps, one sum, one product and one quotient: ``12 U32`` is
  taken), so ``D`` has a relative error of at most ``12 U32`` of its disorder term and ``U32`` of the rest;
* the powder averages: ``erf`` of float32 is accurate to 4 ulps of float32 (``8 U32``) and ``sqrt`` to 1 ulp, with the
  argument ``s = sqrt(b D)`` of relative error ``(2 + 1 + 1) U32 / 2 + U32``; the relative error of ``erf(s)/s`` is
  at most ``2 U32 s^2``-scaled by the log-derivative of ``erf(s)/s``, which is bounded by 1 on ``s >= 0``, so
  ``(8 + 2 + 4 + 8) U32`` suffices.
"""
import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from dmipy_analytic import gaussian
from dmipy_analytic.jax import gaussian as jgaussian
from .rounding import U32, exp_rtol

RNG = np.random.default_rng(11)


def _directions(n):
    d = RNG.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1)[:, None]


def _close(twin, form, rtol, atol=0.0):
    twin = np.asarray(twin, dtype=np.float64)
    assert np.all(np.abs(twin - form) <= rtol * np.abs(form) + atol), (np.abs(twin - form) / np.abs(form)).max()


B = RNG.uniform(0.0, 5e9, 60)
N = _directions(60)
AXIS = np.array([0.36, -0.48, 0.8])


def test_ball():
    D = 3.0e-9
    x_max = (B * D).max()
    _close(jgaussian.ball(B, D), gaussian.ball(B, D), exp_rtol(x_max, 2, 1))


def test_stick():
    D = 2.2e-9
    x_max = (B * D).max()
    _close(jgaussian.stick(B, N, AXIS, D), gaussian.stick(B, N, AXIS, D), (15 * x_max + 4) * U32)


def test_zeppelin():
    d_par, d_perp = 2.4e-9, 0.7e-9
    x_max = (B * d_par).max()
    _close(jgaussian.zeppelin(B, N, AXIS, d_par, d_perp), gaussian.zeppelin(B, N, AXIS, d_par, d_perp),
           (20 * x_max + 4) * U32)


def _btensors(b, n):
    return b[:, None, None] * n[:, :, None] * n[:, None, :]


def test_btensor_forms():
    Bt = _btensors(B, N)
    d_par, d_perp = 2.4e-9, 0.7e-9
    x_max = (B * d_par).max()
    rtol = exp_rtol(x_max, 2 + 9, 8)
    _close(jgaussian.stick_btensor(Bt, AXIS, d_par), gaussian.stick_btensor(Bt, AXIS, d_par), rtol)
    _close(jgaussian.zeppelin_btensor(Bt, AXIS, d_par, d_perp), gaussian.zeppelin_btensor(Bt, AXIS, d_par, d_perp),
           exp_rtol(x_max, 3 + 9, 12))
    D = np.array([[1.8, 0.2, -0.1], [0.2, 0.9, 0.15], [-0.1, 0.15, 0.6]]) * 1e-9
    _close(jgaussian.tensor(B, N, D), gaussian.tensor(B, N, D), exp_rtol(x_max, 2 + 9, 8))
    _close(jgaussian.tensor_btensor(Bt, D), gaussian.tensor_btensor(Bt, D), exp_rtol(x_max, 9 + 9, 8))


def test_time_dependent_diffusivity_and_temporal_zeppelin():
    delta = RNG.uniform(0.008, 0.015, 60)
    Delta = RNG.uniform(0.025, 0.06, 60)
    d_inf, A, d_par = 0.6e-9, 2e-12, 1.9e-9
    twin = np.asarray(jgaussian.time_dependent_diffusivity(d_inf, A, delta, Delta), dtype=np.float64)
    form = gaussian.time_dependent_diffusivity(d_inf, A, delta, Delta)
    disorder = form - d_inf
    assert np.all(np.abs(twin - form) <= 12 * U32 * disorder + U32 * form)
    x_max = (B * d_par).max()
    _close(jgaussian.temporal_zeppelin(B, N, AXIS, d_par, d_inf, A, delta, Delta),
           gaussian.temporal_zeppelin(B, N, AXIS, d_par, d_inf, A, delta, Delta), (30 * x_max + 4) * U32)


def test_powder_averages():
    b = np.linspace(1e7, 8e9, 40)
    d_par, d_perp = 2.1e-9, 0.6e-9
    _close(jgaussian.stick_powder_average(b, d_par), gaussian.stick_powder_average(b, d_par), 22 * U32)
    _close(jgaussian.zeppelin_powder_average(b, d_par, d_perp),
           gaussian.zeppelin_powder_average(b, d_par, d_perp), exp_rtol((b * d_perp).max(), 2, 1) + 22 * U32)
    assert float(jgaussian.zeppelin_powder_average(jnp.asarray(b[:3]), d_par, d_par)[0]) == pytest.approx(
        float(gaussian.ball(b[:3], d_par)[0]), rel=3 * U32 * 100)
    # an oblate tensor is outside the twin's domain: NaN, not a finite value of another tensor
    assert np.all(np.isnan(np.asarray(jgaussian.zeppelin_powder_average(b, d_perp, d_par))))
