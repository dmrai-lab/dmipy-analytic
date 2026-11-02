"""The JAX twins of ``dmipy_analytic.exchange`` against their float64 numpy definitions, in float32.

Tolerances derived from the rounding of each form:

* ``diffusion_time``: ``Delta - delta/3`` is a quotient, a difference of terms of the same sign in the tested range
  (``Delta > delta``): ``4 U32``;
* ``karger``: every one of ``Tr``, ``Det``, ``disc^2`` is a sum or product of at most five rounded terms; the
  difference ``Tr^2 - 4 Det`` loses at most the factor ``Tr^2 / disc^2``, kept at most 16 by testing
  ``disc >= Tr / 4``; the square root halves a relative error, the eigenvalues ``lambda_+/-`` are sums of terms of
  one size, and the weights are a product of five factors and a quotient, so the longest chain is 30 operations of
  float32 with the conditioning 16: ``16 * 30 U32``, the 2 ulps of each of two exponentials adding to it;
* the propagators: four (spin echo) or six (stimulated echo) matrix exponentials of ``(N, N)`` matrices of norm at
  most ``a``; the Pade approximant with scaling and squaring of float32 is accurate to ``c U32`` per exponential with
  ``c = 8 (1 + log2 a)`` for the 13th-order approximant and its squarings, and the product of the factors is stable
  (each factor is a contraction), so the bounds add: ``6 * 8 (1 + log2 a) U32`` plus ``N`` per matrix-vector product.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy.linalg")

from dmipy_analytic import exchange
from dmipy_analytic.jax import exchange as jexchange
from .rounding import U32, EXP_ULPS


def _close(twin, form, rtol, atol=0.0):
    twin = np.asarray(twin, dtype=np.float64)
    assert np.all(np.abs(twin - form) <= rtol * np.abs(form) + atol), (np.abs(twin - form) / np.abs(form)).max()


def test_diffusion_time():
    delta = np.array([0.008, 0.012, 0.02])
    _close(jexchange.diffusion_time(delta, 0.04), exchange.diffusion_time(delta, 0.04), 4 * U32)


def test_exchange_matrix():
    _close(jexchange.exchange_matrix(30.0, 0.4), exchange.exchange_matrix(30.0, 0.4), 4 * U32)


@pytest.mark.parametrize("kappa", [2.0, 15.0, 60.0])
def test_karger(kappa):
    rng = np.random.default_rng(4)
    R1 = rng.uniform(0.2, 2.5, 80)
    R2 = rng.uniform(0.2, 2.5, 80)
    f, t_d = 0.45, 0.04
    Tr = R1 + R2 + kappa * t_d + kappa * f / (1 - f) * t_d
    Det = R1 * R2 + R1 * kappa * f / (1 - f) * t_d + R2 * kappa * t_d
    well_separated = Tr ** 2 - 4 * Det >= (Tr / 4.0) ** 2
    R1, R2 = R1[well_separated], R2[well_separated]
    assert len(R1) > 20
    form = exchange.karger(R1, R2, kappa, f, t_d)
    rtol = 16 * 30 * U32 + 2 * 2 * EXP_ULPS * U32
    _close(jexchange.karger(R1, R2, kappa, f, t_d), form, rtol)


def test_karger_limits_agree_with_numpy():
    R1 = np.array([2.0, 0.5, 1.0])
    R2 = np.array([0.5, 2.0, 1.0])
    _close(jexchange.karger(R1, R2, 0.0, 0.3, 0.04), exchange.karger(R1, R2, 0.0, 0.3, 0.04), 30 * U32)


def _pools():
    return dict(M0=np.array([0.4, 0.6]), D=np.array([1.7e-9, 2.6e-9]), T2=np.array([0.07, 0.11]),
                T1=np.array([1.2, 1.8]))


def test_spin_echo_signal():
    p = _pools()
    K = exchange.exchange_matrix(25.0, 0.4)
    norm_a = np.linalg.norm(K * 0.03 - 2.0e9 / 2 * np.diag(p["D"]), 2) + 0.03 / 0.07
    c = 4 * 8 * (1 + np.log2(max(norm_a, 1.0))) + 2 * 4
    for tau_exc, tau_180 in ((0.0, 0.0), (0.002, 0.006)):
        form = exchange.spin_echo_signal(K, p["M0"], p["D"], p["T2"], p["T1"], 2.0e9, 0.03, 0.03, tau_exc, tau_180)
        twin = jexchange.spin_echo_signal(K, p["M0"], p["D"], p["T2"], p["T1"], 2.0e9, 0.03, 0.03, tau_exc, tau_180)
        _close(twin, form, c * U32)


def test_stimulated_echo_signal():
    p = _pools()
    K = exchange.exchange_matrix(25.0, 0.4)
    norm_a = np.linalg.norm(K * 0.2 - np.diag(1.0 / p["T1"]) * 0.2, 2) + 1.0
    c = 6 * 8 * (1 + np.log2(max(norm_a, 1.0))) + 2 * 6
    for tau_90 in (0.0, 0.002):
        form = exchange.stimulated_echo_signal(K, p["M0"], p["D"], p["T2"], p["T1"], 2.0e9, 0.015, 0.2, 0.015,
                                               tau_90)
        twin = jexchange.stimulated_echo_signal(K, p["M0"], p["D"], p["T2"], p["T1"], 2.0e9, 0.015, 0.2, 0.015,
                                                tau_90)
        _close(twin, form, c * U32)
