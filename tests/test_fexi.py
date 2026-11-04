"""The two-pool forms of a sampled waveform and of filter-exchange imaging against their independent routes.

* :func:`~dmipy_analytic.exchange.karger_waveform`: the pools without exchange (the mixture of two Gaussian decays with
  the exact b of the piecewise-linear wave number), equal pools (exchange conserves the magnetisation), the relaxation
  of the stored and the transverse periods, a stiff ODE solution of the generalised Karger equation (the steps converge
  at fourth order onto it), and the narrow-pulse closed form as the pulses shorten at a fixed wave number (first order).
* :func:`~dmipy_analytic.exchange.fexi_adc`: on the two-pool signal at a small detection b, ``ADC_eq - ADC'(t_m)``
  decays as ``exp(-(k_12 + k_21) t_m)``, to the bound the detection b sets on the cumulant (Hoeffding's lemma).
* :func:`~dmipy_analytic.exchange.fexi_two_site`: the filter efficiency the exact slope reads converges onto it at
  first order as the blocks shorten.
"""
import numpy as np
import pytest

from dmipy_analytic import exchange
from dmipy_analytic.constants import GAMMA

F, KAPPA = 0.4, 30.0
D2 = np.array([0.3e-9, 2.0e-9])
M0 = np.array([F, 1.0 - F])


def _fexi_wave(g_f, g_d, n_lobe=10, n_gap=5, n_m=50):
    """A filter pair, a stored mixing period, a detection pair (rectangular lobes on whole samples), and the readout
    sample; the detection pair's sign is the recall's."""
    blk = lambda g: np.r_[np.full(n_lobe, g), np.zeros(n_gap), np.full(n_lobe, -g)]
    G = np.r_[blk(g_f), np.zeros(n_m), -blk(g_d), 0.0]
    stored = np.zeros(G.size, bool)
    i0 = 2 * n_lobe + n_gap
    stored[i0:i0 + n_m] = True
    return G, stored


def _b_pair(g, n_lobe, n_gap, dt):
    """The exact b of a rectangular bipolar pair, ``gamma^2 g^2 delta^2 (Delta - delta/3)``."""
    delta, Delta = n_lobe * dt, (n_lobe + n_gap) * dt
    return (GAMMA * g * delta) ** 2 * (Delta - delta / 3.0)


def test_without_exchange_the_waveform_signal_is_the_mixture_of_the_relaxed_gaussian_pools():
    dt, n_lobe, n_gap, n_m = 4e-4, 10, 5, 50
    G, st = _fexi_wave(0.06, 0.04, n_lobe, n_gap, n_m)
    T2, T1 = np.array([0.06, 0.09]), np.array([0.9, 1.3])
    b = _b_pair(0.06, n_lobe, n_gap, dt) + _b_pair(0.04, n_lobe, n_gap, dt)
    t_stored = n_m * dt
    t_perp = (G.size - 1) * dt - t_stored
    want = np.sum(M0 * np.exp(-D2 * b - t_perp / T2 - t_stored / T1))
    got = exchange.karger_waveform(G, dt, D2, np.zeros((2, 2)), M0, stored=st, T2=T2, T1=T1)
    assert got == pytest.approx(want, rel=1e-13)


def test_equal_pools_do_not_see_the_exchange_of_any_waveform():
    dt = 4e-4
    G, st = _fexi_wave(0.06, 0.04)
    D = np.array([1.1e-9, 1.1e-9])
    b = _b_pair(0.06, 10, 5, dt) + _b_pair(0.04, 10, 5, dt)
    got = exchange.karger_waveform(G, dt, D, exchange.exchange_matrix(KAPPA, F), M0, stored=st)
    assert got == pytest.approx(np.exp(-1.1e-9 * b), rel=1e-13)


def _ode_reference(G, st, dt, K, D, T2, T1):
    """The generalised Karger equation integrated sample by sample by an embedded Runge-Kutta method of order 8."""
    solve_ivp = pytest.importorskip("scipy.integrate").solve_ivp
    q = np.r_[0.0, np.cumsum(GAMMA * G[:-1] * dt)]
    M = M0.copy()
    for k in range(G.size - 1):
        R = np.diag(1.0 / (T1 if st[k] else T2))
        rhs = lambda t, M, k=k, R=R: (K - R - (q[k] + GAMMA * G[k] * (t - k * dt)) ** 2 * np.diag(D)) @ M
        M = solve_ivp(rhs, (k * dt, (k + 1) * dt), M, method="DOP853", rtol=1e-13, atol=1e-16).y[:, -1]
    return M.sum()


def test_the_magnus_steps_converge_at_fourth_order_onto_the_karger_equation():
    """Halving the sub-step divides the error by 16: the observed order exceeds 3.5 at both halvings, which a step
    without the commutator term (second order) or with it of the wrong sign does not reach."""
    dt = 4e-4
    G, st = _fexi_wave(0.06, 0.04)
    K = exchange.exchange_matrix(KAPPA, F)
    T2, T1 = np.array([0.06, 0.09]), np.array([0.9, 1.3])
    ref = _ode_reference(G, st, dt, K, D2, T2, T1)
    err = [exchange.karger_waveform(G, dt, D2, K, M0, stored=st, T2=T2, T1=T1, sub_steps=s) - ref for s in (1, 2, 4)]
    assert abs(err[0]) / abs(err[1]) > 2 ** 3.5 and abs(err[1]) / abs(err[2]) > 2 ** 3.5, err


def test_the_narrow_pulse_form_is_the_limit_of_short_pulses_at_a_fixed_wave_number():
    """Each pair's lobes of ``delta`` at a fixed ``q = gamma g delta``: the waveform signal approaches the narrow-pulse
    product at first order in ``delta`` (the error halves with ``delta``), the pair's wave number held over ``Delta``
    and the rest of the pair's span counted as mixing."""
    K = exchange.exchange_matrix(KAPPA, F)
    q, Delta, t_m = 2.0e5, 6e-3, 0.03
    err = []
    for delta in (1e-3, 5e-4, 2.5e-4):
        dt = delta / 20
        n_gap, n_m = int(round((Delta - delta) / dt)), int(round(t_m / dt))
        G, st = _fexi_wave(q / (GAMMA * delta), 0.7 * q / (GAMMA * delta), 20, n_gap, n_m)
        got = exchange.karger_waveform(G, dt, D2, K, M0, stored=st)
        err.append(got - exchange.fexi_narrow_pulse(q, Delta, t_m + delta, 0.7 * q, Delta, D2, K, M0))
    r1, r2 = err[0] / err[1], err[1] / err[2]
    assert 1.8 < r1 < 2.2 and 1.8 < r2 < 2.2 and abs(r2 - 2.0) < abs(r1 - 2.0), err


def _slope_adc(q_f, t_m, eps, K, Delta):
    """``-ln(S(b_d = eps) / S(0)) / eps`` of the narrow-pulse two-pool signal: the detection pair of ``Delta`` reads a
    b-value ``q_d^2 Delta``."""
    q_d = np.sqrt(eps / Delta)
    S1 = exchange.fexi_narrow_pulse(q_f, Delta, t_m, q_d, Delta, D2, K, M0)
    S0 = exchange.fexi_narrow_pulse(q_f, Delta, t_m, 0.0, Delta, D2, K, M0)
    return -np.log(S1 / S0) / eps


def _worst(amp, rate, t_m, bound):
    """The largest disagreement between two mixing times of ``amp`` in units of what their bound allows."""
    return max(abs(amp[i] - amp[j]) / ((np.exp(rate * t_m[i]) + np.exp(rate * t_m[j])) * bound)
               for i in range(len(t_m)) for j in range(i + 1, len(t_m)))


@pytest.mark.parametrize("kappa", [0.5, 6.0, 9.0])
def test_the_filtered_adc_recovers_at_the_two_site_rate(kappa):
    """``(ADC_eq - ADC'(t_m)) exp(AXR t_m)`` is the same at every mixing time with ``AXR = k_12 + k_21``. At a detection
    b of ``eps`` each ADC sits within ``[mu_1 - eps dD^2 / 8, mu_1]`` of the initial slope ``mu_1`` (the log of a
    Laplace transform over diffusivities in ``[D_1, D_2]``, Hoeffding's lemma), so two mixing times agree to
    ``(e^(AXR t_1) + e^(AXR t_2)) eps dD^2 / 8``. A rate of ``kappa`` alone, or of ``2 (k_12 + k_21)``, breaks it."""
    K = exchange.exchange_matrix(kappa, F)
    axr = exchange.fexi_two_site(F, kappa, D2, 0.0)[2]
    assert axr == pytest.approx(-np.trace(K), rel=1e-15)            # k_12 + k_21: the decaying mode of K
    Delta, q_f, eps = 8e-3, np.sqrt(1.2e9 / 8e-3), 1e6
    t_m = np.array([0.01, 0.05, 0.15, 0.3])
    adc_eq = _slope_adc(0.0, 0.1, eps, K, Delta)
    dadc = np.array([adc_eq - _slope_adc(q_f, t, eps, K, Delta) for t in t_m])
    bound = eps * (D2[1] - D2[0]) ** 2 / 8.0
    assert _worst(dadc * np.exp(axr * t_m), axr, t_m, bound) <= 1.0
    for wrong in (kappa, 2.0 * axr):
        assert _worst(dadc * np.exp(wrong * t_m), wrong, t_m, bound) > 1.0, wrong


def test_the_short_block_filter_efficiency_is_the_limit_of_the_exact_slope():
    """At a fixed filter b-value and pairs of shrinking ``Delta``, the filter efficiency the exact slope reads
    approaches :func:`fexi_two_site`'s at first order in ``Delta`` (the gap halves with it)."""
    kappa, b_f, eps = 6.0, 1.2e9, 1e5
    K = exchange.exchange_matrix(kappa, F)
    adc_eq, sigma, axr = exchange.fexi_two_site(F, kappa, D2, b_f)
    gap = []
    for Delta in (4e-3, 2e-3, 1e-3):
        t_m = 0.05
        a_eq = _slope_adc(0.0, t_m, eps, K, Delta)
        a_f = _slope_adc(np.sqrt(b_f / Delta), t_m, eps, K, Delta)
        sigma_exact = (1.0 - a_f / a_eq) * np.exp(axr * t_m)
        gap.append(sigma_exact - sigma)
    assert 1.8 < gap[0] / gap[1] < 2.2 and 1.8 < gap[1] / gap[2] < 2.2, gap
    assert adc_eq == pytest.approx(F * D2[0] + (1 - F) * D2[1], rel=1e-15)


def test_the_axr_signal_reads_the_equilibrium_unfiltered_and_the_recovery_filtered():
    adc_eq, sigma, axr = 0.9e-9, 0.45, 4.0
    b = np.array([0.0, 5e8, 1e9, 1e9])
    t_m = np.array([0.1, 0.1, 0.1, 0.4])
    off = exchange.fexi_signal(b, t_m, False, adc_eq, sigma, axr)
    assert off == pytest.approx(np.exp(-b * adc_eq), rel=1e-15)
    on = exchange.fexi_signal(b, t_m, True, adc_eq, sigma, axr)
    assert on == pytest.approx(np.exp(-b * adc_eq * (1 - sigma * np.exp(-axr * t_m))), rel=1e-15)
    assert exchange.fexi_adc(0.0, adc_eq, sigma, axr) == pytest.approx(adc_eq * (1 - sigma), rel=1e-15)
    assert exchange.fexi_adc(1e3, adc_eq, sigma, axr) == pytest.approx(adc_eq, rel=1e-15)


def test_two_pools_are_required():
    with pytest.raises(ValueError, match="two pools"):
        exchange.karger_waveform(np.zeros(4), 1e-3, [1e-9, 1e-9, 1e-9], np.zeros((2, 2)), M0)
    with pytest.raises(ValueError, match="one flag per sample"):
        exchange.karger_waveform(np.zeros(4), 1e-3, D2, np.zeros((2, 2)), M0, stored=np.zeros(3, bool))
