"""``dmipy_analytic.restricted.sheath`` and the annulus of ``matrix_method``, against independent routes.

1. The roots: zeros of the cross product to rounding, none missed (a grid twenty times finer finds no other sign
   change), and the thin annulus's limits ``x_{0,1} -> pi / (1 - ratio)`` and ``x_{m,1} -> m / r_mean``.
2. The modes: orthonormal radial functions, and the ground row of the position matrix exhausting the exact second
   moment of the annulus, ``<x^2> = (r_in^2 + r_out^2) / 4``.
3. The long-time narrow-pulse limit: the annulus's diffraction pattern ``[2 (J_1(q) - rho J_1(q rho)) / (q (1 -
   rho^2))]^2``, approached at first order in ``delta``.
4. The two limits of the shape: a thin annulus tends to the circle of its mean radius (``PeriodicPore``) at second
   order in its thickness, and an annulus with a vanishing hole to the disk (``MatrixPore("cylinder")``).
5. The truncation and the quadrature: the default modes against more, the Gauss-Legendre nodes against twice as many.
6. The sheath forms: the annulus's echo along an oblique direction is the product across and along the axis, the same
   for every direction across it; the 180 as played conjugates the effective waveform's echo; the lamellae's average
   collapses to one circle on a vanishing sheath, converges in its radial nodes, and agrees with the annulus at
   second order in the thickness; the refusals.
"""
import numpy as np
import pytest

pytest.importorskip("scipy.special")

from scipy import special

from dmipy_analytic.constants import GAMMA
from dmipy_analytic.restricted import sheath, surface
from dmipy_analytic.restricted import matrix_method as mm
from dmipy_analytic.restricted.matrix_method import AnnulusPore, MatrixPore, PeriodicPore, annulus_neumann_roots

D = 2.0e-9
R_OUT = 4e-6
DT = 1e-5


def _pgse(G, delta, Delta, direction=(1.0, 0.0, 0.0), ramp=1e-3):
    return surface.pgse_waveform(G, delta, Delta, DT, list(direction), ramp=ramp)


def _cross(m, rho, x):
    return special.jvp(m, x * rho) * special.yvp(m, x) - special.jvp(m, x) * special.yvp(m, x * rho)


@pytest.mark.parametrize("m", [0, 1, 3, 10])
@pytest.mark.parametrize("rho", [0.3, 0.6, 0.9])
def test_the_roots_are_the_zeros_of_the_cross_product_and_none_is_missed(m, rho):
    roots = annulus_neumann_roots(m, rho, 6)
    pos = roots[roots > 0]
    scale = np.abs(_cross(m, rho, pos * 1.01)) + np.abs(_cross(m, rho, pos * 0.99))
    assert np.all(np.abs(_cross(m, rho, pos)) <= 1e-9 * scale)
    assert (roots[0] == 0.0) == (m == 0) and np.all(np.diff(roots) > 0) and np.all(pos >= m)
    xs = np.linspace(max(m, 1e-6), pos[-1] * (1 + 1e-9), 200_000)      # twenty times the bracketing grid
    f = _cross(m, rho, xs)
    assert int(np.sum(np.sign(f[1:]) * np.sign(f[:-1]) < 0)) == len(pos)


def test_the_thin_annulus_has_the_slab_s_radial_and_the_circle_s_angular_roots():
    rho = 0.995
    assert annulus_neumann_roots(0, rho, 2)[1] == pytest.approx(np.pi / (1 - rho), rel=1e-4)
    r_mean = 0.5 * (1 + rho)
    for m in (1, 4):
        assert annulus_neumann_roots(m, rho, 1)[0] == pytest.approx(m / r_mean, rel=1e-4)


@pytest.mark.parametrize("rho", [0.4, 0.7])
def test_the_modes_are_orthonormal_and_exhaust_the_second_moment(rho):
    t, wq = np.polynomial.legendre.leggauss(mm.ANNULUS_NODES)
    r = rho + 0.5 * (1 - rho) * (t + 1)
    wr = 0.5 * (1 - rho) * wq
    for m in (0, 2):
        F = []
        for x in annulus_neumann_roots(m, rho, 6):
            R = np.ones_like(r) if x == 0 else special.jv(m, x * r) * special.yvp(m, x) - special.yv(m, x * r) * special.jvp(m, x)
            F.append(R / np.sqrt(np.sum(wr * R * R * r)))
        gram = np.array([[np.sum(wr * a * b * r) for b in F] for a in F])
        np.testing.assert_allclose(gram, np.eye(len(F)), atol=1e-12)
    # x = r cos(theta) from the ground mode reaches the m = 1 modes only; their squares sum to <x^2> from below
    # (Bessel's inequality), the deficit falling as n^-3: r's slope at the walls makes its Neumann coefficients ~ n^-2
    exact = (1 + rho ** 2) / 4
    deficit = []
    for n in (12, 24):
        lam, B, _, _ = mm._annulus_modes_unit(rho, 4, n, mm.ANNULUS_NODES)
        np.testing.assert_allclose(B, B.T, atol=1e-15)
        deficit.append(exact - np.sum(B[0] ** 2))
    assert 0.0 < deficit[1] < deficit[0] < 1e-4 * exact and 6.0 < deficit[0] / deficit[1] < 10.0


def test_the_long_time_narrow_pulse_limit_is_the_annulus_s_diffraction_pattern():
    rho, G, Delta = 0.6, 0.25, 0.12                    # D Delta / R^2 = 15: the m = 1 mode has decayed by e^-25
    errs = []
    for delta in (4e-4, 2e-4, 1e-4):
        g, ref = _pgse(G * 4e-4 / delta, delta, Delta, ramp=0.0)
        E = AnnulusPore(2 * R_OUT, 2 * rho * R_OUT, D).signal(g[:, 0], DT, refocus=ref).real
        q = GAMMA * G * 4e-4 * R_OUT                     # the dimensionless wavenumber, q R
        limit = (2 * (special.j1(q) - rho * special.j1(q * rho)) / (q * (1 - rho ** 2))) ** 2
        errs.append(abs(E - limit))
    assert errs[2] < 2e-3 and 0.35 < errs[1] / errs[0] < 0.65 and 0.35 < errs[2] / errs[1] < 0.65


def test_a_thin_annulus_tends_to_the_circle_of_its_mean_radius_at_second_order():
    g, ref = _pgse(0.25, 5e-3, 12e-3)
    errs = []
    for rho in (0.98, 0.99):
        Ea = AnnulusPore(2 * R_OUT, 2 * rho * R_OUT, D, n_modes=(24, 3)).signal(g[:, 0], DT, refocus=ref)
        Ec = PeriodicPore.circle((1 + rho) * R_OUT, D, n_modes=24).signal(g[:, :2], DT, refocus=ref)
        errs.append(abs(Ea - Ec))
    assert errs[1] < 1e-4 and 3.0 < errs[0] / errs[1] < 5.0


def test_an_annulus_with_a_vanishing_hole_tends_to_the_disk():
    g, ref = _pgse(0.25, 5e-3, 12e-3)
    disk = MatrixPore("cylinder", 2 * R_OUT, D).signal(g[:, 0], DT, refocus=ref)
    errs = [abs(AnnulusPore(2 * R_OUT, 2 * rho * R_OUT, D, n_modes=(24, 14)).signal(g[:, 0], DT, refocus=ref) - disk)
            for rho in (0.2, 0.1, 0.05)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 2e-3


def test_the_default_truncation_and_the_quadrature_are_converged():
    g, ref = _pgse(0.3, 7e-3, 13e-3)
    for rho in (0.6, 0.8):
        E0 = AnnulusPore(2 * R_OUT, 2 * rho * R_OUT, D).signal(g[:, 0], DT, refocus=ref)
        E1 = AnnulusPore(2 * R_OUT, 2 * rho * R_OUT, D, n_modes=(32, 12)).signal(g[:, 0], DT, refocus=ref)
        assert abs(E0 - E1) < 1e-7
        lam0, B0, _, _ = mm._annulus_modes_unit(rho, 8, 6, mm.ANNULUS_NODES)
        lam1, B1, _, _ = mm._annulus_modes_unit(rho, 8, 6, 2 * mm.ANNULUS_NODES)
        np.testing.assert_allclose(lam0, lam1, rtol=1e-13)
        np.testing.assert_allclose(B0, B1, atol=1e-13)


def test_the_annulus_factorises_and_is_the_same_along_every_direction_across_the_axis():
    rho = 0.7
    g, ref = _pgse(0.25, 5e-3, 12e-3, direction=(0.6, -0.3, 0.5))
    E = sheath.annulus_signal(g, DT, 2 * R_OUT, 2 * rho * R_OUT, D, refocus=ref)
    perp = np.hypot(g[:, 0], g[:, 1])
    across = AnnulusPore(2 * R_OUT, 2 * rho * R_OUT, D).signal(perp, DT, refocus=ref)
    along = surface.axial_signal(g[:, 2], DT, D, refocus=ref)
    assert E == pytest.approx(across * along, abs=1e-13)
    gx = np.column_stack([perp, np.zeros(len(g)), g[:, 2]])
    assert sheath.annulus_signal(gx, DT, 2 * R_OUT, 2 * rho * R_OUT, D, refocus=ref) == pytest.approx(E, abs=1e-12)


def test_the_180_as_played_conjugates_the_effective_waveform_s_echo():
    rho = 0.7
    g, ref = _pgse(0.25, 5e-3, 12e-3)
    eff = g.copy()
    eff[ref[0]:] *= -1.0
    a = sheath.annulus_signal(g, DT, 2 * R_OUT, 2 * rho * R_OUT, D, refocus=ref)
    b = sheath.annulus_signal(eff, DT, 2 * R_OUT, 2 * rho * R_OUT, D)
    assert a == pytest.approx(np.conj(b), abs=1e-12)
    assert 0.0 < a.real < 1.0 and abs(a.imag) < 1e-12


def test_the_lamellae_collapse_to_one_circle_converge_and_meet_the_annulus_on_a_thin_sheath():
    g, ref = _pgse(0.25, 5e-3, 12e-3, direction=(0.8, 0.0, 0.6))
    one = surface.circle_signal(g, DT, 2 * R_OUT, D, refocus=ref)
    thin = sheath.lamellae_signal(g, DT, 2 * R_OUT, 2 * (1 - 1e-6) * R_OUT, D, refocus=ref)
    assert thin == pytest.approx(one, abs=1e-6)
    a = sheath.lamellae_signal(g, DT, 2 * R_OUT, 1.2 * R_OUT, D, refocus=ref)
    b = sheath.lamellae_signal(g, DT, 2 * R_OUT, 1.2 * R_OUT, D, refocus=ref, n_radii=32)
    assert a == pytest.approx(b, abs=1e-12)
    gaps = [abs(sheath.lamellae_signal(g, DT, 2 * R_OUT, 2 * rho * R_OUT, D, refocus=ref)
                - sheath.annulus_signal(g, DT, 2 * R_OUT, 2 * rho * R_OUT, D, refocus=ref, n_modes=(24, 4)))
            for rho in (0.96, 0.98)]
    assert gaps[1] < 1e-4 and 3.0 < gaps[0] / gaps[1] < 5.0
    # the radial diffusion the lamellae forbid is felt on a thick sheath
    assert abs(a - sheath.annulus_signal(g, DT, 2 * R_OUT, 1.2 * R_OUT, D, refocus=ref)) > 1e-3


def test_a_turning_gradient_is_refused_by_the_annulus_and_taken_by_the_lamellae():
    t = 2.0 * np.pi * 2.0 * (np.arange(400) + 0.5) / 400
    turning = np.stack([0.1 * np.cos(t), 0.1 * np.sin(t), np.zeros(400)], axis=1)
    with pytest.raises(ValueError, match="one fixed direction across the axis"):
        sheath.annulus_signal(turning, DT, 2 * R_OUT, 1.2 * R_OUT, D)
    E = sheath.lamellae_signal(turning, DT, 2 * R_OUT, 1.2 * R_OUT, D)
    assert 0.0 < abs(E) < 1.0


@pytest.mark.parametrize("outer,inner", [(2e-6, 2e-6), (2e-6, 3e-6), (2e-6, 0.0)])
def test_a_sheath_needs_an_inner_wall_inside_the_outer_one(outer, inner):
    g, _ = _pgse(0.1, 2e-3, 5e-3)
    with pytest.raises(ValueError, match="inner diameter < outer diameter"):
        sheath.annulus_signal(g, DT, outer, inner, D)
    with pytest.raises(ValueError, match="inner diameter < outer diameter"):
        sheath.lamellae_signal(g, DT, outer, inner, D)
    with pytest.raises(ValueError, match="inner diameter < outer diameter"):
        AnnulusPore(outer, inner, D)


def test_a_ratio_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        annulus_neumann_roots(1, 1.0, 3)
