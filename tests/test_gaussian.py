"""The Gaussian forms against independent routes: the pulsed-gradient attenuation from the gradient waveform's own
parameters, the full tensor as the general case of the axially symmetric ones, the b-tensor forms as the b-value forms
for a pulsed gradient and as isotropic for a spherical encoding, and the powder averages against Gauss-Legendre
quadrature over the axis direction."""
import numpy as np
import pytest

pytest.importorskip("scipy.special")                            # the powder averages' extra

from dmipy_analytic import gaussian
from dmipy_analytic.constants import GAMMA

RNG = np.random.default_rng(3)


def _directions(n):
    d = RNG.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1)[:, None]


def _pulsed_btensor(b, direction):
    return b[:, None, None] * direction[:, :, None] * direction[:, None, :]


def test_ball_is_the_stejskal_tanner_attenuation():
    G, delta, Delta, D = 0.04, 0.01, 0.03, 2.0e-9
    b = GAMMA ** 2 * G ** 2 * delta ** 2 * (Delta - delta / 3.0)
    expected = np.exp(-GAMMA ** 2 * G ** 2 * delta ** 2 * D * (Delta - delta / 3.0))
    assert gaussian.ball(b, D) == pytest.approx(expected, rel=1e-14)


def test_stick_and_zeppelin_are_the_tensor_of_their_symmetry():
    axis = np.array([0.3, -0.5, 0.8124038404635961])
    axis = axis / np.linalg.norm(axis)
    n = _directions(40)
    b = RNG.uniform(0.0, 4e9, 40)
    d_par, d_perp = 1.9e-9, 0.55e-9
    D = d_perp * np.eye(3) + (d_par - d_perp) * np.outer(axis, axis)
    assert gaussian.zeppelin(b, n, axis, d_par, d_perp) == pytest.approx(gaussian.tensor(b, n, D), rel=1e-13)
    stick_tensor = d_par * np.outer(axis, axis)
    assert gaussian.stick(b, n, axis, d_par) == pytest.approx(gaussian.tensor(b, n, stick_tensor), rel=1e-13)


def test_btensor_forms_are_the_b_value_forms_for_a_pulsed_gradient():
    axis = np.array([0.0, 0.6, 0.8])
    n = _directions(30)
    b = RNG.uniform(0.0, 3e9, 30)
    B = _pulsed_btensor(b, n)
    d_par, d_perp = 1.7e-9, 0.7e-9
    assert gaussian.stick_btensor(B, axis, d_par) == pytest.approx(gaussian.stick(b, n, axis, d_par), rel=1e-13)
    assert gaussian.zeppelin_btensor(B, axis, d_par, d_perp) == pytest.approx(
        gaussian.zeppelin(b, n, axis, d_par, d_perp), rel=1e-13)
    D = d_perp * np.eye(3) + (d_par - d_perp) * np.outer(axis, axis)
    assert gaussian.tensor_btensor(B, D) == pytest.approx(gaussian.tensor(b, n, D), rel=1e-13)


def test_a_spherical_encoding_does_not_see_the_axis():
    """For an isotropic b-tensor ``B = (b/3) I`` the attenuation is ``exp(-b (2 D_perp + D_par)/3)`` whatever the
    axis."""
    b = np.array([0.5e9, 1e9, 2.5e9])
    B = (b / 3.0)[:, None, None] * np.eye(3)
    d_par, d_perp = 2.1e-9, 0.4e-9
    expected = np.exp(-b * (d_par + 2.0 * d_perp) / 3.0)
    for axis in (np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)):
        assert gaussian.zeppelin_btensor(B, axis, d_par, d_perp) == pytest.approx(expected, rel=1e-13)


def test_a_planar_encoding_in_the_plane_normal_to_the_stick_is_not_attenuated():
    b = np.array([1e9, 2e9])
    B = (b / 2.0)[:, None, None] * np.diag([1.0, 1.0, 0.0])
    assert gaussian.stick_btensor(B, np.array([0.0, 0.0, 1.0]), 2e-9) == pytest.approx(1.0, abs=0)


def test_time_dependent_diffusivity_tends_to_its_limit_and_adds_the_disorder_term():
    D_inf, A = 0.8e-9, 3e-12
    delta = 0.01
    long_time = gaussian.time_dependent_diffusivity(D_inf, A, delta, np.array([0.1, 1.0, 100.0]))
    assert np.all(np.diff(long_time) < 0) and long_time[-1] == pytest.approx(D_inf, rel=1e-4)
    Delta = 0.05
    expected = D_inf + A * (np.log(Delta / delta) + 1.5) / (Delta - delta / 3.0)
    assert gaussian.time_dependent_diffusivity(D_inf, A, delta, Delta) == pytest.approx(expected, rel=1e-14)
    assert gaussian.time_dependent_diffusivity(D_inf, 0.0, delta, Delta) == D_inf


def test_temporal_zeppelin_is_the_zeppelin_at_the_time_dependent_diffusivity():
    axis = np.array([0.6, 0.0, 0.8])
    n = _directions(25)
    b = RNG.uniform(0.0, 3e9, 25)
    delta = RNG.uniform(0.008, 0.015, 25)
    Delta = RNG.uniform(0.025, 0.05, 25)
    d_par, d_inf, A = 1.8e-9, 0.6e-9, 2e-12
    d_perp = gaussian.time_dependent_diffusivity(d_inf, A, delta, Delta)
    B = _pulsed_btensor(b, n)
    assert gaussian.temporal_zeppelin(b, n, axis, d_par, d_inf, A, delta, Delta) == pytest.approx(
        gaussian.zeppelin_btensor(B, axis, d_par, d_perp), rel=1e-13)


def _axis_average(signal, n_nodes=96):
    """The average over all axis directions at fixed gradient direction, by Gauss-Legendre in ``t = cos(theta)``."""
    t, w = np.polynomial.legendre.leggauss(n_nodes)
    return np.sum(w * signal(t)) / 2.0


@pytest.mark.parametrize("b", [0.0, 3e8, 1e9, 3e9, 8e9])
def test_stick_powder_average_is_the_mean_over_the_sphere(b):
    d = 1.7e-9
    expected = _axis_average(lambda t: np.exp(-b * d * t ** 2))
    assert gaussian.stick_powder_average(np.array([b]), d)[0] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("b", [0.0, 3e8, 1e9, 3e9, 8e9])
@pytest.mark.parametrize("d_par, d_perp", [(1.9e-9, 0.5e-9), (0.5e-9, 1.9e-9), (1.2e-9, 1.2e-9)])
def test_zeppelin_powder_average_is_the_mean_over_the_sphere(b, d_par, d_perp):
    """Prolate, oblate and spherical tensors."""
    expected = _axis_average(lambda t: np.exp(-b * (d_perp + (d_par - d_perp) * t ** 2)))
    assert gaussian.zeppelin_powder_average(np.array([b]), d_par, d_perp)[0] == pytest.approx(expected, rel=1e-12)


def test_the_powder_average_of_a_spherical_zeppelin_is_the_ball_and_of_the_stick_is_the_zeppelin_without_perp():
    b = np.array([0.0, 1e9, 4e9])
    assert gaussian.zeppelin_powder_average(b, 1.1e-9, 1.1e-9) == pytest.approx(gaussian.ball(b, 1.1e-9), rel=1e-13)
    assert gaussian.zeppelin_powder_average(b, 1.7e-9, 0.0) == pytest.approx(
        gaussian.stick_powder_average(b, 1.7e-9), rel=1e-12)
