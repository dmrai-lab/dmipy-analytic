"""The closed forms of the sphere against independent routes: the ball's diffraction pattern by numerical Fourier
transform, Callaghan's series by the spectral sum of the ball's modes with numerically integrated overlaps (the roots
of the derivative of the spherical Bessel functions from the matrix-method module), the Gaussian-phase forms against
the matrix method at small gradients, the sampled-waveform form against the closed form of a rectangular pulse pair,
and the phases of the Cartesian components of a gradient adding."""
import numpy as np
import pytest

pytest.importorskip("scipy.special")

from scipy import special

from dmipy_analytic.restricted import sphere
from dmipy_analytic.restricted.matrix_method import MatrixPore, sphere_neumann_roots

D = 2e-9


def _gauss_legendre(n, a, b):
    t, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * (t + 1.0) + a, 0.5 * (b - a) * w


def test_sgp_long_time_is_the_diffraction_pattern_of_the_ball():
    d = 8e-6
    R = d / 2
    q = np.array([0.0, 0.4e5, 0.9e5, 1.6e5, 3.0e5])
    r, w = _gauss_legendre(300, 0.0, R)
    # the Fourier transform of the uniform ball, integrated over the angles analytically: 4 pi r^2 sin(kr)/(kr)
    volume = 4 / 3 * np.pi * R ** 3
    fourier = np.array([(np.sum(w * 4 * np.pi * r ** 2 * np.sinc(2 * qi * r)) / volume) ** 2 for qi in q])
    assert sphere.sgp_long_time(q, d) == pytest.approx(fourier, rel=1e-10, abs=1e-14)


def _spectral_sum(q, tau, d, l_max=14, k_max=24):
    """``E = sum_{l,k} exp(-alpha^2 D tau / R^2) |<psi_lk | e^{i 2 pi q z}>|^2 / V`` for the normalised modes
    ``psi_lk = N j_l(alpha r / R) P_l(cos theta)`` of the ball; with ``e^{ikz} = sum_l (2l + 1) i^l j_l(kr) P_l`` the
    squared overlap is ``4 pi (2l + 1) I_lk^2 / int j_l(alpha r / R)^2 r^2 dr`` with the radial overlap
    ``I_lk = int j_l(alpha r / R) j_l(2 pi q r) r^2 dr`` by Gauss-Legendre quadrature."""
    R = d / 2
    r, w = _gauss_legendre(800, 0.0, R)
    k = 2 * np.pi * q
    V = 4 / 3 * np.pi * R ** 3
    E = np.zeros(len(q))
    for l in range(l_max):
        for alpha in sphere_neumann_roots(l, k_max):
            radial = special.spherical_jn(l, alpha * r / R)
            norm = np.sum(w * radial ** 2 * r ** 2)
            for i, ki in enumerate(k):
                overlap = np.sum(w * radial * special.spherical_jn(l, ki * r) * r ** 2)
                E[i] += (np.exp(-alpha ** 2 * D * tau / R ** 2) * 4 * np.pi * (2 * l + 1) * overlap ** 2
                         / norm / V)
    return E


@pytest.mark.parametrize("tau", [0.001, 0.006, 0.04])
def test_sgp_is_the_spectral_sum_of_the_balls_modes(tau):
    d = 8e-6
    q = np.array([0.2e5, 0.6e5, 1.0e5, 1.5e5])
    assert sphere.sgp(q, tau, d, D) == pytest.approx(_spectral_sum(q, tau, d), rel=1e-6)


def test_sgp_of_the_ground_state_alone_is_the_long_time_form():
    """One radial and one angular mode is the constant mode, which the series skips: only the uniform-sphere structure
    factor remains, at any diffusion time."""
    d = 8e-6
    q = np.array([0.0, 0.4e5, 1.0e5, 2.0e5])
    assert sphere.sgp(q, 0.006, d, D, n_roots=1, n_functions=1) == pytest.approx(sphere.sgp_long_time(q, d), rel=1e-12)


def test_sgp_limits_and_zero_q():
    d = 8e-6
    q = np.array([0.0, 0.4e5, 1.0e5, 2.0e5])
    assert sphere.sgp(q, 100.0, d, D) == pytest.approx(sphere.sgp_long_time(q, d), rel=1e-9)
    assert sphere.sgp(q, 0.01, d, D)[0] == 1.0
    assert sphere.sgp(np.float64(1.0e5), 0.01, d, D).shape == ()


def _pgse(G, delta, Delta, steps_per_pulse):
    dt = delta / steps_per_pulse
    nd, nD = steps_per_pulse, int(round(Delta / dt))
    assert abs(nD * dt - Delta) < 1e-12 * Delta
    g = np.zeros(nD + nd)
    g[:nd] = G
    g[nD:nD + nd] = -G
    return g, dt


def test_gaussian_phase_is_the_low_gradient_limit_of_the_matrix_method():
    d, delta, Delta = 5e-6, 0.008, 0.02
    pore = MatrixPore("sphere", d, D, n_modes=(5, 14))
    ratios = []
    for G in (0.2, 0.1, 0.05):
        g, dt = _pgse(G, delta, Delta, 1000)
        exact = np.log(abs(pore.signal(g, dt)))
        gpa = np.log(sphere.gaussian_phase(G, delta, Delta, d, D))
        ratios.append((exact - gpa) / gpa)
    assert abs(ratios[1]) < abs(ratios[0]) / 3.0 and abs(ratios[2]) < abs(ratios[1]) / 3.0
    assert abs(ratios[2]) < 2e-3


def test_gaussian_phase_broadcasts_like_its_scalar_calls():
    rng = np.random.default_rng(2)
    G = rng.uniform(0.01, 0.2, 9)
    delta = rng.uniform(0.006, 0.012, 9)
    Delta = rng.uniform(0.02, 0.04, 9)
    batch = sphere.gaussian_phase(G, delta, Delta, 6e-6, D)
    assert np.array_equal(batch, np.array([sphere.gaussian_phase(g, dl, DL, 6e-6, D)
                                           for g, dl, DL in zip(G, delta, Delta)]))


def test_the_waveform_form_of_a_rectangular_pgse_converges_to_the_closed_form():
    d, delta, Delta, G = 6e-6, 0.01, 0.03, 0.08
    exact = sphere.gaussian_phase(G, delta, Delta, d, D, n_roots=40)
    errors = []
    for steps in (100, 200, 400):
        g, dt = _pgse(G, delta, Delta, steps)
        errors.append(abs(sphere.gaussian_phase_waveform(g, dt, d, D, n_roots=40) - exact))
    assert errors[1] < 0.6 * errors[0] and errors[2] < 0.6 * errors[1] and errors[2] < 2e-4


def test_the_phases_of_the_components_of_a_gradient_add():
    """The sphere is isotropic and the Gaussian phase quadratic in the gradient, so the signal of a rotating
    gradient is the product of the signals of its Cartesian components."""
    d, dt = 6e-6, 1e-4
    t = np.arange(120) * dt
    G = np.stack([0.06 * np.cos(2 * np.pi * 80 * t), 0.04 * np.sin(2 * np.pi * 80 * t), 0.05 * np.ones_like(t)], axis=1)
    together = sphere.gaussian_phase_waveform(G, dt, d, D, n_roots=20)
    apart = np.prod([sphere.gaussian_phase_waveform(G[:, c], dt, d, D, n_roots=20) for c in range(3)])
    assert together == pytest.approx(apart, rel=1e-12)
