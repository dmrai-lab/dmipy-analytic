"""The relaxation factors of a diffusion measurement against independent routes: the pore's slowest decay rate from
the eigenvalue equation of the Robin boundary problem (Brownstein-Tarr's fast-diffusion limit), the surface-to-volume
ratios from the pores' areas and volumes, and the Gamma-averaged closed forms against numerical integration of the
Gamma density."""
import numpy as np
import pytest

pytest.importorskip("scipy.special")                            # the Bessel average's and the oracles' extra

from scipy import integrate, optimize, special, stats

from dmipy_analytic import relaxation


def _gamma_expectation(function, shape, scale, times):
    """``E[function(d, t)]`` for ``d ~ Gamma(shape, scale)``, by adaptive quadrature in units of the scale."""
    law = stats.gamma(shape)
    return [integrate.quad(lambda u: function(scale * u, t) * law.pdf(u), 0.0, np.inf, epsabs=0, epsrel=1e-12,
                           limit=500)[0] for t in times]


def test_transverse_and_longitudinal_decay():
    tau = np.array([0.0, 0.03, 0.08, 0.2])
    assert relaxation.transverse_decay(tau, 0.08) == pytest.approx(np.exp(-tau / 0.08), rel=1e-15)
    assert relaxation.longitudinal_decay(tau, 1.2) == pytest.approx(np.exp(-tau / 1.2), rel=1e-15)
    assert relaxation.transverse_decay(0.05, np.inf) == 1.0


def test_surface_to_volume_of_the_pores():
    d = 3.2e-6
    R = d / 2.0
    assert relaxation.surface_to_volume(d, "sphere") == pytest.approx(4 * np.pi * R ** 2 / (4 / 3 * np.pi * R ** 3))
    assert relaxation.surface_to_volume(d, "cylinder") == pytest.approx(2 * np.pi * R / (np.pi * R ** 2))
    assert relaxation.surface_to_volume(d, "plane") == pytest.approx(2.0 / d)
    with pytest.raises(ValueError, match="geometry"):
        relaxation.surface_to_volume(d, "cube")


def _slowest_rate(geometry, rho, D, d):
    """The slowest decay rate of ``dm/dt = D lap m`` with ``D dm/dn + rho m = 0`` on the wall: ``D x^2 / a^2`` with
    ``x`` the smallest root of the geometry's eigenvalue equation, ``a`` the radius (half the plates' separation)."""
    a = d / 2.0
    h = rho * a / D
    equation = {
        "plane": lambda x: x * np.tan(x) - h,
        "cylinder": lambda x: x * special.jv(1, x) - h * special.jv(0, x),
        "sphere": lambda x: 1.0 - x / np.tan(x) - h,
    }[geometry]
    x = optimize.brentq(equation, 1e-9, {"plane": 1.5, "cylinder": 2.4, "sphere": 3.0}[geometry])
    return D * x ** 2 / a ** 2


@pytest.mark.parametrize("geometry", ["plane", "cylinder", "sphere"])
def test_surface_relaxation_is_the_slowest_rate_of_the_boundary_problem_in_the_fast_diffusion_regime(geometry):
    """For ``rho a / D`` well below one the decay rate is ``rho S/V`` (Brownstein-Tarr)."""
    rho, D, d, tau = 2e-6, 2e-9, 4e-6, 0.05
    assert rho * (d / 2) / D < 1e-2
    rate = _slowest_rate(geometry, rho, D, d)
    form = relaxation.surface_relaxation(rho, relaxation.surface_to_volume(d, geometry), tau)
    assert form == pytest.approx(np.exp(-rate * tau), rel=2e-3)


@pytest.mark.parametrize("volume_weighted, power", [(True, 2), (False, 0)])
def test_gamma_average_over_cylinders_is_the_integral_over_the_density(volume_weighted, power):
    shape, scale, rho = 2.0, 0.3e-6, 8e-6
    tau = np.array([0.0, 0.02, 0.08, 0.3])
    form = relaxation.surface_relaxation_gamma_cylinders(shape, scale, rho, tau, volume_weighted)
    expected = _gamma_expectation(lambda d, t: np.exp(-4.0 * rho * t / d), shape + power, scale, tau)
    assert form == pytest.approx(expected, rel=1e-8)


def test_gamma_average_over_spheres_is_the_integral_over_the_density():
    shape, scale, rho = 2.5, 1.0e-6, 15e-6
    tau = np.array([0.0, 0.01, 0.06, 0.2])
    expected = _gamma_expectation(lambda d, t: np.exp(-6.0 * rho * t / d), shape + 3.0, scale, tau)
    assert relaxation.surface_relaxation_gamma_spheres(shape, scale, rho, tau) == pytest.approx(expected, rel=1e-8)


def test_gamma_averages_are_one_without_relaxation_and_decrease_with_time():
    tau = np.linspace(0.0, 0.2, 9)
    for form in (relaxation.surface_relaxation_gamma_cylinders, relaxation.surface_relaxation_gamma_spheres):
        no_relaxivity = form(2.0, 0.4e-6, 0.0, tau)
        assert no_relaxivity == pytest.approx(1.0, abs=0)
        decay = form(2.0, 0.4e-6, 10e-6, tau)
        assert decay[0] == 1.0 and np.all(np.diff(decay) < 0)


def test_exterior_surface_to_volume_of_a_packed_population():
    f, shape, scale = 0.6, 2.0, 0.3e-6
    for geometry, (k, m) in {"sphere": (6.0, 3), "cylinder": (4.0, 2), "plane": (2.0, 1)}.items():
        expected = k * f / ((1.0 - f) * (shape + m - 1.0) * scale)
        assert relaxation.exterior_surface_to_volume(f, shape, scale, geometry=geometry) == pytest.approx(expected)
    # the mean over the volume-weighted diameters of k/d
    mean_sv = _gamma_expectation(lambda d, _: 4.0 / d, shape + 2.0, scale, [0.0])[0]
    assert relaxation.exterior_surface_to_volume(f, shape, scale, geometry="cylinder") == pytest.approx(
        mean_sv * f / (1.0 - f), rel=1e-8)
    with pytest.raises(ValueError, match="geometry"):
        relaxation.exterior_surface_to_volume(f, shape, scale, geometry="blob")
    with pytest.raises(TypeError):
        relaxation.exterior_surface_to_volume(f, shape, scale)          # the geometry is required
