"""The Watson and Bingham forms: the zonal ratios against the first step of their recurrence, the exact ``c_0^0``,
the isotropic and zonal limits, the power per order under rotation, the Watson as the Bingham of equal concentrations,
and the Bingham partition function against direct quadrature over the sphere."""
import numpy as np
import pytest

special = pytest.importorskip("scipy.special")                  # the forms' own extra

from dmipy_analytic.harmonics import sh_block
from dmipy_analytic.orientation import (bingham_canonical_sh, bingham_normalization, bingham_sh, watson_sh,
                                        watson_zonal_ratios)

L_MAX = 8
L0_EXACT = 1.0 / (2.0 * np.sqrt(np.pi))              # Y_0^0
BLOCKS = [sh_block(l) for l in range(0, L_MAX + 1, 2)]


def _J0(kappa):
    """``J_0(kappa) = int_{-1}^{1} exp(kappa t^2) dt``."""
    return 2.0 if kappa < 1e-12 else np.sqrt(np.pi) * special.erfi(np.sqrt(kappa)) / np.sqrt(kappa)


def _J2(kappa):
    """``J_2`` from the first step of the recurrence."""
    J0 = _J0(kappa)
    K1 = np.exp(kappa) / kappa - J0 / (2 * kappa)
    return (3 * K1 - J0) / 2.0


def _rot(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def test_the_zonal_ratios_start_at_one_and_vanish_for_the_isotropic_watson():
    for kappa in [0.0, 0.1, 1.0, 5.0, 20.0, 100.0]:
        assert abs(watson_zonal_ratios(kappa, l_max=L_MAX)[0] - 1.0) < 1e-14
    np.testing.assert_array_less(np.abs(watson_zonal_ratios(0.0, l_max=L_MAX)[1:]), 1e-12)


def test_the_second_ratio_is_J2_over_J0():
    for kappa in [0.5, 1.0, 2.0, 5.0, 10.0]:
        assert abs(watson_zonal_ratios(kappa, l_max=L_MAX)[1] - _J2(kappa) / _J0(kappa)) < 1e-10


def test_the_ratios_lie_in_the_unit_interval_and_grow_with_the_concentration():
    for kappa in [0.1, 1.0, 5.0, 20.0]:
        r = watson_zonal_ratios(kappa, l_max=L_MAX)
        assert np.all(r >= -1e-12) and np.all(r <= 1.0 + 1e-10)
    assert watson_zonal_ratios(10.0, l_max=4)[1] > watson_zonal_ratios(1.0, l_max=4)[1]


def test_the_ratios_approach_one_as_the_watson_sharpens():
    """``r_l ~ 1 - l(l+1) / (4 kappa)``: at ``kappa = 5e4`` the ``l = 8`` deviation is 3.6e-4."""
    np.testing.assert_allclose(watson_zonal_ratios(50000.0, l_max=L_MAX), 1.0, atol=1e-3)


def test_the_l0_coefficient_is_exact_for_every_watson():
    for kappa in [0.0, 0.5, 2.0, 10.0, 50.0]:
        for mu in [np.r_[0., 0., 1.], np.r_[1., 0., 0.], np.r_[1., 1., 1.] / np.sqrt(3.)]:
            assert abs(watson_sh(mu, kappa, l_max=L_MAX)[0] - L0_EXACT) < 1e-13


def test_a_watson_about_z_is_zonal():
    sh = watson_sh(np.r_[0., 0., 1.], kappa=3.0, l_max=L_MAX)
    zonal = [b.start + l for b, l in zip(BLOCKS, range(0, L_MAX + 1, 2))]
    off = np.setdiff1d(np.arange(sh.size), zonal)
    np.testing.assert_array_less(np.abs(sh[off]), 1e-12)


def test_rotating_a_watson_keeps_its_power_per_order():
    sh_z = watson_sh(np.r_[0., 0., 1.], 3.0, l_max=L_MAX)
    sh_x = watson_sh(np.r_[1., 0., 0.], 3.0, l_max=L_MAX)
    for b in BLOCKS:
        assert abs(np.sum(sh_z[b] ** 2) - np.sum(sh_x[b] ** 2)) < 1e-12


def test_the_watson_is_finite_over_six_decades_of_concentration():
    mu = np.r_[1., 1., 1.] / np.sqrt(3.)
    for kappa in np.logspace(-3, 3, 50):
        assert np.all(np.isfinite(watson_sh(mu, kappa, l_max=L_MAX)))


def test_the_canonical_bingham_has_the_exact_l0_and_only_even_nonnegative_m():
    for k1, k2 in [(0.5, 0.0), (2.0, 1.0), (10.0, 5.0), (50.0, 20.0)]:
        assert abs(bingham_canonical_sh(k1, k2, l_max=L_MAX)[0] - L0_EXACT) < 1e-13
    sh = bingham_canonical_sh(3.0, 1.5, l_max=L_MAX)
    for b, l in zip(BLOCKS, range(0, L_MAX + 1, 2)):
        for m in range(-l, l + 1):
            if m < 0 or m % 2:
                assert abs(sh[b.start + m + l]) < 1e-15


def test_the_isotropic_bingham_is_uniform():
    sh = bingham_canonical_sh(0.0, 0.0, l_max=L_MAX)
    assert abs(sh[0] - L0_EXACT) < 1e-13
    np.testing.assert_array_less(np.abs(sh[1:]), 1e-12)


def test_equal_concentrations_are_the_watson():
    np.testing.assert_allclose(bingham_canonical_sh(2.0, 2.0, l_max=4), watson_sh(np.r_[0., 0., 1.], 2.0, l_max=4),
                               atol=1e-13)
    R = _rot([0.3, -0.8, 0.5], 1.3)
    np.testing.assert_allclose(bingham_sh(R, (3.0, 3.0)), watson_sh(R[:, 2], 3.0), atol=1e-12)


def test_rotating_a_bingham_keeps_its_l0_and_its_power_per_order():
    c = bingham_canonical_sh(4.0, 1.0, l_max=L_MAX)
    for R in (_rot([1.0, 2.0, 0.5], 0.7), _rot([0.0, 1.0, 1.0], -1.1)):
        sh = bingham_sh(R, (4.0, 1.0), l_max=L_MAX)
        assert abs(sh[0] - L0_EXACT) < 1e-12
        for b in BLOCKS:
            assert abs(np.sum(sh[b] ** 2) - np.sum(c[b] ** 2)) < 1e-11


def test_the_partition_function_is_the_integral_over_the_sphere():
    quad = pytest.importorskip("scipy.integrate")
    k1, k2 = 1.5, 0.7

    def integrand(phi, theta):
        s2 = np.sin(theta) ** 2
        return np.exp(-k1 * s2 * np.cos(phi) ** 2 - k2 * s2 * np.sin(phi) ** 2) * np.sin(theta)
    Z_ref, _ = quad.dblquad(integrand, 0, np.pi, 0, 2 * np.pi, epsabs=1e-8, epsrel=1e-8)
    assert abs(bingham_normalization(k1, k2) - Z_ref) / Z_ref < 1e-7
