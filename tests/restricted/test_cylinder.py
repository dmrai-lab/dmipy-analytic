"""The closed forms of the cylinder against independent routes: the circular aperture's diffraction pattern by
numerical Fourier transform over the disk, Callaghan's series by the spectral sum of the disk's modes with
numerically integrated overlaps, the Gaussian-phase forms against the matrix method at small gradients and against
each other, the closed form of the cosine gradient against its sampled-waveform integral, and the fast-eigenmode form
against the rectangular-pulse form in its limit."""
import numpy as np
import pytest

pytest.importorskip("scipy.special")

from scipy import special

from dmipy_analytic.constants import GAMMA
from dmipy_analytic.restricted import cylinder, plane
from dmipy_analytic.restricted.matrix_method import MatrixPore, cylinder_neumann_roots

D = 2e-9


def _gauss_legendre(n, a, b):
    t, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * (t + 1.0) + a, 0.5 * (b - a) * w


def test_sgp_long_time_is_the_diffraction_pattern_of_the_disk():
    d = 6e-6
    R = d / 2
    q = np.array([0.0, 0.5e5, 1.0e5, 2.0e5, 3.5e5])
    r, w = _gauss_legendre(200, 0.0, R)
    # the 2D Fourier transform of the uniform disk, integrated over the angle analytically (2 pi J_0) and over r
    fourier = np.array([(np.sum(w * r * 2 * np.pi * special.j0(2 * np.pi * qi * r)) / (np.pi * R ** 2)) ** 2
                        for qi in q])
    assert cylinder.sgp_long_time(q, d) == pytest.approx(fourier, rel=1e-10, abs=1e-14)
    first_zero = special.jn_zeros(1, 1)[0] / (np.pi * d)                           # 2 pi q R = j_{1,1}
    assert cylinder.sgp_long_time(np.array([first_zero]), d)[0] == pytest.approx(0.0, abs=1e-25)


def _spectral_sum(q, tau, d, m_max=40, k_max=40):
    """``E = sum_{m,k} exp(-alpha^2 D tau / R^2) |<psi_mk | e^{i 2 pi q x}>|^2 / (pi R^2)`` for the normalised modes
    ``psi_mk = N J_m(alpha r / R) cos(m theta)`` of the disk; the radial overlaps by Gauss-Legendre quadrature, the
    angular ones by the Jacobi-Anger expansion ``e^{i k r cos theta} = sum_m i^m eps_m J_m(k r) cos(m theta)``."""
    R = d / 2
    r, w = _gauss_legendre(600, 0.0, R)
    k = 2 * np.pi * q
    E = np.zeros(len(q))
    for m in range(m_max):
        roots = cylinder_neumann_roots(m, k_max)
        a_m = 2 * np.pi if m == 0 else np.pi
        eps_m = 1.0 if m == 0 else 2.0
        for alpha in roots:
            radial = special.jv(m, alpha * r / R)
            norm = np.sum(w * radial ** 2 * r)
            for i, ki in enumerate(k):
                overlap = np.sum(w * radial * special.jv(m, ki * r) * r)
                E[i] += (np.exp(-alpha ** 2 * D * tau / R ** 2) * eps_m ** 2 * a_m * overlap ** 2 / norm
                         / (np.pi * R ** 2))
    return E


@pytest.mark.parametrize("tau", [0.0006, 0.004, 0.03])
def test_sgp_is_the_spectral_sum_of_the_disks_modes(tau):
    d = 6e-6
    q = np.array([0.3e5, 0.8e5, 1.4e5, 2.0e5])
    assert cylinder.sgp(q, tau, d, D) == pytest.approx(_spectral_sum(q, tau, d), rel=1e-6)


def test_sgp_limits_and_zero_q():
    d = 6e-6
    q = np.array([0.0, 0.4e5, 1.0e5, 2.0e5])
    assert cylinder.sgp(q, 80.0, d, D) == pytest.approx(cylinder.sgp_long_time(q, d), rel=1e-9)
    assert cylinder.sgp(q, 0.01, d, D)[0] == 1.0
    assert cylinder.sgp(np.float64(1.0e5), 0.01, d, D).shape == ()


def test_neumann_roots_table_is_the_roots_of_the_derivative_of_each_order():
    table = cylinder.neumann_roots_table(6, 5)
    assert table.shape == (6, 5) and not table.flags.writeable
    assert table[0, 0] == 0.0
    for m in range(5):
        nonzero = table[1:, m] if m == 0 else table[:, m]
        assert special.jvp(m, nonzero, 1) == pytest.approx(0.0, abs=1e-12)


def _pgse(G, delta, Delta, steps_per_pulse):
    dt = delta / steps_per_pulse
    nd, nD = steps_per_pulse, int(round(Delta / dt))
    assert abs(nD * dt - Delta) < 1e-12 * Delta
    g = np.zeros(nD + nd)
    g[:nd] = G
    g[nD:nD + nd] = -G
    return g, dt


def test_gaussian_phase_is_the_low_gradient_limit_of_the_matrix_method():
    d, delta, Delta = 3e-6, 0.008, 0.02
    pore = MatrixPore("cylinder", d, D, n_modes=(6, 14))
    ratios = []
    for G in (0.2, 0.1, 0.05):
        g, dt = _pgse(G, delta, Delta, 1000)
        exact = np.log(abs(pore.signal(g, dt)))
        gpa = np.log(cylinder.gaussian_phase(G, delta, Delta, d, D))
        ratios.append((exact - gpa) / gpa)
    assert abs(ratios[1]) < abs(ratios[0]) / 3.0 and abs(ratios[2]) < abs(ratios[1]) / 3.0
    assert abs(ratios[2]) < 2e-3


def test_gaussian_phase_broadcasts_like_its_scalar_calls():
    rng = np.random.default_rng(1)
    G = rng.uniform(0.01, 0.2, 9)
    delta = rng.uniform(0.006, 0.012, 9)
    Delta = rng.uniform(0.02, 0.04, 9)
    batch = cylinder.gaussian_phase(G, delta, Delta, 4e-6, D)
    scalar = np.array([cylinder.gaussian_phase(g, dl, DL, 4e-6, D) for g, dl, DL in zip(G, delta, Delta)])
    assert np.array_equal(batch, scalar)
    grouped = cylinder.gaussian_phase(G, 0.01, 0.03, 4e-6, D)
    assert np.array_equal(grouped, np.array([cylinder.gaussian_phase(g, 0.01, 0.03, 4e-6, D) for g in G]))


def _cosine(G, frequency, duration, n_t):
    t = np.arange(n_t) * duration / n_t + 0.5 * duration / n_t           # midpoint samples
    return G * np.cos(2 * np.pi * frequency * t), duration / n_t


def test_the_cosine_closed_form_is_the_limit_of_the_sampled_waveform():
    """The error of the sampled waveform halves when the step halves (the recursion is first order in ``dt``)."""
    d, G, f, duration = 4e-6, 0.1, 100.0, 0.02
    exact = cylinder.gaussian_phase_cosine(G, 2 * np.pi * f, duration, d, D, n_roots=30)
    errors = []
    for n_t in (200, 400, 800):
        g, dt = _cosine(G, f, duration, n_t)
        errors.append(abs(cylinder.gaussian_phase_waveform(g, dt, d, D, n_roots=30) - exact))
    assert errors[1] < 0.6 * errors[0] and errors[2] < 0.6 * errors[1] and errors[2] < 1e-4


def test_the_waveform_form_of_a_rectangular_pgse_converges_to_the_closed_form():
    d, delta, Delta, G = 4e-6, 0.01, 0.03, 0.08
    exact = cylinder.gaussian_phase(G, delta, Delta, d, D, n_roots=30)
    errors = []
    for steps in (100, 200, 400):
        g, dt = _pgse(G, delta, Delta, steps)
        errors.append(abs(cylinder.gaussian_phase_waveform(g, dt, d, D, n_roots=30) - exact))
    assert errors[1] < 0.6 * errors[0] and errors[2] < 0.6 * errors[1] and errors[2] < 2e-4


def test_the_fast_eigenmode_form_is_the_long_pulse_limit_of_the_rectangular_form():
    """For a gradient along ``n`` the angular spectrum gives ``phi_perp = C int G^2 dt sin^2(theta)``; it equals the
    Gaussian phase of the perpendicular gradient when the pulses are long against ``R^2 / D`` and the error falls as
    the pulses lengthen."""
    d = 2e-6
    axis = np.array([0.0, 0.6, 0.8])
    n = np.array([0.8, 0.6, 0.0])
    cos2 = np.dot(n, axis) ** 2
    errors = []
    for delta in (0.002, 0.004, 0.008):
        Delta, G = 0.03, 0.05
        int_G2 = 2 * G ** 2 * delta
        Y00 = 1.0 / np.sqrt(4 * np.pi)
        x, y, z = n
        Y2 = np.array([np.sqrt(15 / (4 * np.pi)) * x * y, np.sqrt(15 / (4 * np.pi)) * y * z,
                       np.sqrt(5 / (16 * np.pi)) * (2 * z ** 2 - x ** 2 - y ** 2),
                       np.sqrt(15 / (4 * np.pi)) * x * z, np.sqrt(15 / (16 * np.pi)) * (x ** 2 - y ** 2)])
        gamma_lm = np.concatenate([[int_G2 * Y00], int_G2 * Y2])[None, :]
        b = GAMMA ** 2 * G ** 2 * delta ** 2 * (Delta - delta / 3.0)
        fast = cylinder.gaussian_phase_gamma_lm(gamma_lm, np.array([b * cos2]), axis, d, D, 1.7e-9)
        rectangular = (cylinder.gaussian_phase(G * np.sqrt(1 - cos2), delta, Delta, d, D)
                       * np.exp(-b * cos2 * 1.7e-9))
        errors.append(abs(np.log(fast[0]) - np.log(rectangular)) / abs(np.log(rectangular)))
    assert errors[0] > errors[1] > errors[2] and errors[2] < 0.02


def test_the_exponent_coefficients_are_the_perpendicular_exponent_in_the_axis_harmonics():
    d = 3e-6
    gamma_lm = np.array([[0.7, 0.1, -0.2, 0.3, 0.05, -0.1], [1.4, 0.0, 0.0, 0.0, 0.0, 0.0]])
    c0, c2m = cylinder.gaussian_phase_exponent_coefficients(gamma_lm, d, D)
    axis = np.array([0.36, -0.48, 0.8])
    Y2 = cylinder._real_harmonics_l2(axis)
    phi_perp = c0 + c2m @ Y2
    full = cylinder.gaussian_phase_gamma_lm(gamma_lm, np.zeros(2), axis, d, D, 1.7e-9)
    assert np.exp(-phi_perp) == pytest.approx(full, rel=1e-13)
    # an isotropic encoding (no order-two part) does not see the axis
    assert c2m[1] == pytest.approx(0.0, abs=0)


def test_the_capped_cylinder_is_the_cylinder_times_the_plates():
    q_perp = np.array([0.3e5, 1.1e5, 0.0])
    q_par = np.array([0.9e5, 0.0, 0.4e5])
    d, L, tau = 4e-6, 15e-6, 0.02
    assert cylinder.capped_sgp_long_time(q_perp, q_par, d, L) == pytest.approx(
        cylinder.sgp_long_time(q_perp, d) * plane.sgp_long_time(q_par, L), rel=1e-15)
    assert cylinder.capped_sgp(q_perp, q_par, tau, d, L, D) == pytest.approx(
        cylinder.sgp(q_perp, tau, d, D) * plane.sgp(q_par, tau, L, D), rel=1e-15)
    assert cylinder.capped_sgp_long_time(np.zeros(1), np.zeros(1), d, L)[0] == 1.0
