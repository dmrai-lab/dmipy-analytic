"""The Karger forms against their independent routes: the closed form against the matrix exponential of the same
two-pool rate equation, the mixture of the pools without exchange, the single averaged exponent at fast exchange, and
the multi-interval propagators against the products they are built from."""
import numpy as np
import pytest

pytest.importorskip("scipy.linalg")                             # the matrix exponential's extra

from scipy.linalg import expm

from dmipy_analytic import exchange


def test_diffusion_time_of_a_pulsed_gradient():
    assert exchange.diffusion_time(0.01, 0.03) == pytest.approx(0.03 - 0.01 / 3.0, rel=1e-15)


def test_the_exchange_matrix_conserves_magnetisation_and_has_the_volume_fractions_as_equilibrium():
    kappa, f = 37.0, 0.62
    K = exchange.exchange_matrix(kappa, f)
    assert K.sum(axis=0) == pytest.approx([0.0, 0.0], abs=1e-12)
    assert K @ np.array([f, 1.0 - f]) == pytest.approx([0.0, 0.0], abs=1e-12)


@pytest.mark.parametrize("kappa", [0.5, 8.0, 40.0, 300.0])
def test_karger_is_the_matrix_exponential_of_the_two_pool_rate_equation(kappa):
    """``E = sum(expm(K t_d - diag(R)) (f, 1-f))``: the closed form is that product's spectral sum."""
    f, t_d = 0.45, 0.04
    R1 = np.array([0.05, 0.4, 1.2, 2.5, 0.9])
    R2 = np.array([0.3, 0.7, 0.6, 1.0, 2.2])
    K = exchange.exchange_matrix(kappa, f)
    expected = np.array([np.sum(expm(K * t_d - np.diag([r1, r2])) @ np.array([f, 1.0 - f]))
                         for r1, r2 in zip(R1, R2)])
    assert exchange.karger(R1, R2, kappa, f, t_d) == pytest.approx(expected, rel=1e-11)


@pytest.mark.parametrize("R1, R2", [(2.0, 0.5), (0.5, 2.0), (1.0, 1.0)])
def test_without_exchange_the_signal_is_the_mixture_of_the_pools(R1, R2):
    f = 0.35
    E = exchange.karger(np.array([R1]), np.array([R2]), 0.0, f, 0.04)
    assert E[0] == pytest.approx(f * np.exp(-R1) + (1.0 - f) * np.exp(-R2), rel=1e-13)


def test_fast_exchange_averages_the_exponents():
    f, R1, R2 = 0.4, 1.8, 0.6
    E = exchange.karger(np.array([R1]), np.array([R2]), 1e6, f, 0.04)
    assert E[0] == pytest.approx(np.exp(-(f * R1 + (1.0 - f) * R2)), rel=1e-5)


def test_equal_pools_do_not_see_the_exchange():
    E = exchange.karger(np.array([0.7]), np.array([0.7]), 25.0, 0.5, 0.04)
    assert E[0] == pytest.approx(np.exp(-0.7), rel=1e-13)


def _pools(tau_free=0.0):
    return dict(M0=np.array([0.4, 0.6]), D=np.array([1.7e-9, 2.6e-9]), T2=np.array([0.07, 0.11]),
                T1=np.array([1.2, 1.8]))


def test_spin_echo_without_exchange_or_relaxation_is_the_mixture_of_the_gaussian_pools():
    p = _pools()
    p["T2"] = np.array([1e10, 1e10])
    p["T1"] = np.array([1e10, 1e10])
    b = 2.0e9
    E = exchange.spin_echo_signal(np.zeros((2, 2)), p["M0"], p["D"], p["T2"], p["T1"], b, 0.03, 0.03)
    assert E == pytest.approx(np.sum(p["M0"] * np.exp(-b * p["D"])), rel=1e-9)


def test_spin_echo_with_relaxation_and_no_exchange_decays_with_the_echo_time_of_each_pool():
    p = _pools()
    b, dt1, dt2 = 1.5e9, 0.02, 0.025
    E = exchange.spin_echo_signal(np.zeros((2, 2)), p["M0"], p["D"], p["T2"], p["T1"], b, dt1, dt2)
    assert E == pytest.approx(np.sum(p["M0"] * np.exp(-b * p["D"] - (dt1 + dt2) / p["T2"])), rel=1e-12)


def test_spin_echo_with_equal_pools_is_unaffected_by_exchange():
    kappa, f = 30.0, 0.5
    K = exchange.exchange_matrix(kappa, f)
    one = np.array([1e10, 1e10])
    E = exchange.spin_echo_signal(K, np.array([f, 1.0 - f]), np.array([2e-9, 2e-9]), one, one, 1e9, 0.02, 0.02)
    assert E == pytest.approx(np.exp(-1e9 * 2e-9), rel=1e-9)                      # 1e10 s is no relaxation to 1e-10


def test_finite_pulses_relax_with_both_times():
    """A refocusing pulse of duration ``tau`` relaxes each pool by ``exp(-tau (2/pi)(1/T2 + 1/T1))``."""
    p = _pools()
    tau = 0.004
    K = np.zeros((2, 2))
    base = exchange.spin_echo_signal(K, p["M0"], p["D"], p["T2"], p["T1"], 0.0, 0.02, 0.02)
    pulsed = exchange.spin_echo_signal(K, p["M0"], p["D"], p["T2"], p["T1"], 0.0, 0.02, 0.02, tau_180=tau)
    rate = (2.0 / np.pi) * (1.0 / p["T2"] + 1.0 / p["T1"])
    ref = np.sum(p["M0"] * np.exp(-0.04 / p["T2"] - rate * tau))
    assert base == pytest.approx(np.sum(p["M0"] * np.exp(-0.04 / p["T2"])), rel=1e-12)
    assert pulsed == pytest.approx(ref, rel=1e-12)


def test_stimulated_echo_without_exchange_is_half_the_mixture_with_t1_in_the_mixing_time():
    p = _pools()
    b, delta, TM = 2.0e9, 0.015, 0.2
    E = exchange.stimulated_echo_signal(np.zeros((2, 2)), p["M0"], p["D"], p["T2"], p["T1"], b, delta, TM, delta)
    expected = 0.5 * np.sum(p["M0"] * np.exp(-b * p["D"] - 2.0 * delta / p["T2"] - TM / p["T1"]))
    assert E == pytest.approx(expected, rel=1e-12)


def test_the_stimulated_echo_carries_the_exchange_during_the_mixing_time():
    """With a long mixing time the pools equilibrate: the signal tends to the volume-weighted sum of the pools'
    encoding attenuations only through the equilibrium magnetisation (the pools' exchange conserves it)."""
    f, kappa = 0.5, 40.0
    K = exchange.exchange_matrix(kappa, f)
    one = np.array([1e10, 1e10])
    E = exchange.stimulated_echo_signal(K, np.array([f, 1.0 - f]), np.array([2e-9, 2e-9]), one, one, 1e9, 0.015,
                                        0.5, 0.015)
    assert E == pytest.approx(0.5 * np.exp(-1e9 * 2e-9), rel=1e-9)                # 1e10 s is no relaxation to 1e-10
