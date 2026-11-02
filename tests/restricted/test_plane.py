"""The closed forms of the plane against independent routes: the slit's diffraction pattern by numerical Fourier
transform, Callaghan's series by the spectral sum of the cosine modes with numerically integrated overlaps, the
limits of the finite-time form, and the Gaussian-phase form against the matrix method at small gradients (the
difference of the exponents falls as the fourth power of the gradient)."""
import numpy as np
import pytest

pytest.importorskip("scipy.special")

from dmipy_analytic.constants import GAMMA
from dmipy_analytic.restricted import plane
from dmipy_analytic.restricted.matrix_method import MatrixPore

D = 2e-9


def _gauss_legendre(n, a, b):
    """The nodes and weights of the ``n``-point Gauss-Legendre rule on ``[a, b]`` (exact for smooth integrands)."""
    t, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * (t + 1.0) + a, 0.5 * (b - a) * w


def test_sgp_long_time_is_the_diffraction_pattern_of_the_slit():
    d = 8e-6
    q = np.array([0.0, 0.3e5, 0.7e5, 1.2e5, 2.5e5, 4.0e5])
    x, w = _gauss_legendre(200, 0.0, d)
    fourier = np.array([np.abs(np.sum(w * np.exp(2j * np.pi * qi * x)) / d) ** 2 for qi in q])
    assert plane.sgp_long_time(q, d) == pytest.approx(fourier, rel=1e-10, abs=1e-14)
    zeros = plane.sgp_long_time(np.array([1.0, 2.0, 3.0]) / d, d)
    assert zeros == pytest.approx(0.0, abs=1e-30)


def _spectral_sum(q, tau, d, n_modes=60):
    """``E = sum_n exp(-D (n pi / d)^2 tau) |<phi_n | e^{i 2 pi q x}>|^2 / d`` for the normalised cosine modes
    ``phi_n`` of the gap (uniform initial density ``1/d``), with the overlaps by Gauss-Legendre quadrature."""
    x, w = _gauss_legendre(400, 0.0, d)
    E = np.zeros(len(q))
    for n in range(n_modes):
        phi = np.cos(n * np.pi * x / d) * (1.0 if n == 0 else np.sqrt(2.0)) / np.sqrt(d)
        overlap = np.array([np.abs(np.sum(w * phi * np.exp(2j * np.pi * qi * x))) ** 2 for qi in q])
        E = E + overlap * np.exp(-D * (n * np.pi / d) ** 2 * tau) / d
    return E


@pytest.mark.parametrize("tau", [0.0005, 0.004, 0.03])
def test_sgp_is_the_spectral_sum_of_the_cosine_modes(tau):
    d = 6e-6
    q = np.array([0.2e5, 0.6e5, 1.1e5, 1.7e5, 2.3e5])
    assert plane.sgp(q, tau, d, D) == pytest.approx(_spectral_sum(q, tau, d), rel=1e-6)


def test_sgp_limits_in_time_and_at_zero_q():
    d = 6e-6
    q = np.array([0.0, 0.4e5, 1.0e5, 2.0e5])
    # at a time far below d^2 / D the spins have not met a wall: free diffusion (the series needs the modes whose
    # decay is not yet complete, hence 400 of them)
    free = np.exp(-(2 * np.pi * q) ** 2 * D * 1e-8)
    assert plane.sgp(q, 1e-8, d, D, n_roots=400) == pytest.approx(free, rel=1e-6)
    long_time = plane.sgp(q, 50.0, d, D)
    assert long_time == pytest.approx(plane.sgp_long_time(q, d), rel=1e-9)         # the pattern of the slit
    assert plane.sgp(q, 0.01, d, D)[0] == 1.0


def _pgse(G, delta, Delta, steps_per_pulse):
    """The rectangular PGSE on a grid of ``dt = delta / steps_per_pulse``, which holds ``delta`` and ``Delta`` exactly
    when ``Delta / dt`` is a whole number."""
    dt = delta / steps_per_pulse
    nd, nD = steps_per_pulse, int(round(Delta / dt))
    assert abs(nD * dt - Delta) < 1e-12 * Delta
    n_t = nD + nd
    g = np.zeros(n_t)
    g[:nd] = G
    g[nD:nD + nd] = -G
    return g, dt


def test_gaussian_phase_is_the_low_gradient_limit_of_the_matrix_method():
    """``ln E_matrix - ln E_gaussian_phase`` is of fourth order in the gradient (the fourth cumulant): the ratio to
    the exponent falls by 16 when the gradient halves."""
    d, delta, Delta = 3e-6, 0.008, 0.02
    pore = MatrixPore("plane", d, D, n_modes=60)
    ratios = []
    for G in (0.2, 0.1, 0.05):
        g, dt = _pgse(G, delta, Delta, 2000)
        exact = np.log(abs(pore.signal(g, dt)))
        gpa = np.log(plane.gaussian_phase(G, delta, Delta, d, D))
        ratios.append((exact - gpa) / gpa)
    assert abs(ratios[1]) < abs(ratios[0]) / 3.0 and abs(ratios[2]) < abs(ratios[1]) / 3.0
    assert abs(ratios[2]) < 1e-3


def test_gaussian_phase_broadcasts_and_is_one_without_gradient():
    G = np.array([0.0, 0.05, 0.1])
    out = plane.gaussian_phase(G, np.array([0.01, 0.01, 0.015]), 0.03, 4e-6, D)
    assert out.shape == (3,) and out[0] == 1.0
    single = [plane.gaussian_phase(g, dl, 0.03, 4e-6, D) for g, dl in zip(G, (0.01, 0.01, 0.015))]
    assert np.array_equal(out, np.array(single))


def test_gaussian_phase_at_long_pulses_is_the_free_attenuation_of_the_gap_scale():
    """When the gap is much larger than the diffusion length the spins are free and the Gaussian phase is
    ``exp(-b D)`` with ``b = gamma^2 G^2 delta^2 (Delta - delta/3)``."""
    d, delta, Delta, G = 2e-3, 0.002, 0.004, 0.003
    b = GAMMA ** 2 * G ** 2 * delta ** 2 * (Delta - delta / 3.0)
    assert plane.gaussian_phase(G, delta, Delta, d, D) == pytest.approx(np.exp(-b * D), rel=1e-3)
