"""The field forms against independent routes: the hollow cylinder's lumen field is Wharton and Bowtell's, its outside
field their two-dimensional dipole, a long segment is the infinite line, segments add, a short segment far away is a
point dipole; the annulus mean of ``ln(b/r)`` is its quadrature; the current segment is the Biot-Savart integral by
quadrature and, on its bisector, the finite wire's closed form."""
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_analytic.constants import MU0
from dmipy_analytic.fields import annulus_mean_log, hollow_cylinder_basis, segment_basis, segment_field

SYM6 = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
SYM5 = ((0, 0), (1, 1), (0, 1), (0, 2), (1, 2))


def _frame(u):
    u = np.asarray(u, float) / np.linalg.norm(u)
    e1 = np.cross(u, [0.3, 0.5, 0.8]); e1 /= np.linalg.norm(e1)
    return u, e1, np.cross(u, e1)


def _HMH(channels, H):
    """``(iso, H.M_P.H, H.M_A.H)`` from the 12 channels (``M_P``'s ``zz`` from its trace, ``3 iso``)."""
    iso, P5, A6 = channels[..., 0], channels[..., 1:6], channels[..., 6:12]
    MP = np.zeros(channels.shape[:-1] + (3, 3)); MA = np.zeros_like(MP)
    for k, (i, j) in enumerate(SYM5):
        MP[..., i, j] = MP[..., j, i] = P5[..., k]
    MP[..., 2, 2] = 3.0 * iso - P5[..., 0] - P5[..., 1]
    for k, (i, j) in enumerate(SYM6):
        MA[..., i, j] = MA[..., j, i] = A6[..., k]
    return iso, np.einsum("i,...ij,j->...", H, MP, H), np.einsum("i,...ij,j->...", H, MA, H)


@pytest.mark.parametrize("u", [(0.0, 0.0, 1.0), (0.3, -0.5, 0.81)])
def test_the_lumen_holds_the_wharton_bowtell_field_and_the_outside_their_dipole(u):
    a, b = 0.6e-6, 0.9e-6
    u, e1, e2 = _frame(u)
    H = np.array([0.4, 0.2, 0.89]); H /= np.linalg.norm(H)
    sin2 = 1.0 - (H @ u) ** 2
    phis = np.linspace(0.0, 2 * np.pi, 9, endpoint=False) + 0.2
    lumen = np.array([r * (np.cos(p) * e1 + np.sin(p) * e2) for r in (0.0, 0.2 * a, 0.7 * a) for p in phis])
    iso, hp, ha = _HMH(hollow_cylinder_basis(lumen, np.broadcast_to(u, lumen.shape), a, b), H)
    npt.assert_array_equal(iso, 0.0)
    npt.assert_allclose(hp, 0.0, atol=1e-15)
    npt.assert_allclose(ha, 0.5 * np.log(b / a) * sin2, rtol=1e-12)            # uniform: (1/2) ln(1/g) sin^2 theta
    radii = (1.0001 * b, 1.7 * b, 6.0 * b)
    outside = np.array([r * (np.cos(p) * e1 + np.sin(p) * e2) for r in radii for p in phis])
    r = np.linalg.norm(outside, axis=1)
    Hp = H - (H @ u) * u                                                        # B0's projection on the cross-section
    cos2alpha = 2.0 * ((outside / r[:, None]) @ Hp) ** 2 / (Hp @ Hp) - 1.0
    iso, hp, ha = _HMH(hollow_cylinder_basis(outside, np.broadcast_to(u, outside.shape), a, b), H)
    dipole = (b * b - a * a) * sin2 * cos2alpha / (2.0 * r * r)
    npt.assert_array_equal(iso, 0.0)
    npt.assert_allclose(-hp, dipole, rtol=1e-12, atol=1e-15)                   # chi_I: -H.M_P.H
    npt.assert_allclose(ha, dipole / 6.0, rtol=1e-12, atol=1e-15)              # chi_A: (1/6) of it


def test_the_sheath_holds_a_third_of_the_local_term_and_m_p_has_its_trace():
    a, b = 0.6e-6, 0.9e-6
    u, e1, e2 = _frame((0.0, 0.0, 1.0))
    pts = np.array([r * e1 for r in (a, 0.7e-6, 0.85e-6)])
    c = hollow_cylinder_basis(pts, np.broadcast_to(u, pts.shape), a, b)
    npt.assert_array_equal(c[:, 0], 1.0 / 3.0)
    iso, hp, _ = _HMH(c, u)                                                     # H along the axis: H.P.H = 0, H.S.H = 0
    npt.assert_allclose(hp, 0.0, atol=1e-15)


@pytest.mark.parametrize("u", [(0.0, 0.0, 1.0), (0.3, -0.5, 0.81)])
def test_a_long_segment_is_the_infinite_line_and_segments_add(u):
    a, b = 0.6e-6, 0.9e-6
    u, e1, e2 = _frame(u)
    pts = np.array([r * (np.cos(p) * e1 + np.sin(p) * e2) for r in (1.2 * b, 3.0 * b, 9.0 * b)
                    for p in (0.1, 1.3, 2.9)])
    U = np.broadcast_to(u, pts.shape)
    n = len(pts)
    L = 1e5 * b                                                                 # the ends' share is (rho / L)^2 ~ 1e-8
    line = hollow_cylinder_basis(pts, U, a, b)
    seg = segment_basis(pts, U, np.full(n, -L), np.full(n, L), np.full(n, a), np.full(n, b))
    npt.assert_allclose(seg[:, 1:], line[:, 1:], rtol=0, atol=1e-7 * np.abs(line[:, 1:]).max())
    z0 = np.linspace(-3e-6, 5e-6, n)
    parts = (segment_basis(pts, U, np.full(n, -4e-6), z0, np.full(n, a), np.full(n, b))
             + segment_basis(pts, U, z0, np.full(n, 7e-6), np.full(n, a), np.full(n, b)))
    whole = segment_basis(pts, U, np.full(n, -4e-6), np.full(n, 7e-6), np.full(n, a), np.full(n, b))
    npt.assert_allclose(parts, whole, rtol=1e-12, atol=1e-15 * np.abs(whole).max())


def test_a_short_segment_far_away_is_the_point_dipole_of_its_moment():
    a, b, half = 0.6e-6, 0.9e-6, 0.05e-6
    u = np.array([0.0, 0.0, 1.0])
    A = np.pi * (b * b - a * a)
    for d in (np.array([40e-6, 0.0, 30e-6]), np.array([-20e-6, 35e-6, -50e-6])):
        rho_vec = d - (d @ u) * u                                              # the point's foot is at -z' = d.u
        z = d @ u
        c = segment_basis(rho_vec[None], u[None], np.array([-half - z]), np.array([half - z]), np.array([a]),
                          np.array([b]))[0]
        r = np.linalg.norm(d); n = d / r
        Lt = 2 * half * (np.eye(3) - 3.0 * np.outer(n, n)) / (4 * np.pi * r ** 3)
        want = A * Lt
        got = np.array([c[1 + k] for k in range(5)])
        npt.assert_allclose(got, [want[i, j] for i, j in SYM5], rtol=0, atol=1e-4 * np.abs(want).max())


def test_the_annulus_mean_of_the_log_is_its_quadrature():
    for a, b in ((0.4e-6, 0.5e-6), (0.7e-6, 1.0e-6), (1.0e-6, 3.5e-6)):
        x, w = np.polynomial.legendre.leggauss(64)
        r = 0.5 * (b - a) * x + 0.5 * (b + a)
        mean = np.sum(0.5 * (b - a) * w * np.log(b / r) * 2 * np.pi * r) / (np.pi * (b * b - a * a))
        npt.assert_allclose(annulus_mean_log(a, b), mean, rtol=1e-13)


def test_the_current_segment_is_the_biot_savart_integral_and_the_wire_on_its_bisector():
    a, b = np.array([0.0, 0.0, -0.1]), np.array([0.02, 0.01, 0.15])
    P = np.array([[0.05, -0.02, 0.0], [0.3, 0.1, -0.2], [-0.04, 0.2, 0.3]])
    x, w = np.polynomial.legendre.leggauss(200)
    s = 0.5 * (x + 1.0)
    pts = a + s[:, None] * (b - a)
    dl = b - a
    for p, B in zip(P, segment_field(a, b, P, 2.5)):
        R = p - pts
        integrand = np.cross(dl, R) / np.linalg.norm(R, axis=1)[:, None] ** 3
        want = MU0 * 2.5 / (4 * np.pi) * 0.5 * np.sum(w[:, None] * integrand, axis=0)
        npt.assert_allclose(B, want, rtol=1e-10)
    L, d = 10.0, 0.02                                     # on the bisector: mu0 I / (2 pi d) * L / sqrt(L^2 + d^2)
    mid = segment_field(np.array([0.0, 0.0, -L]), np.array([0.0, 0.0, L]), np.array([[d, 0.0, 0.0]]))[0]
    npt.assert_allclose(mid, [0.0, MU0 / (2 * np.pi * d) * L / np.hypot(L, d), 0.0], rtol=1e-9, atol=1e-20)
    with pytest.raises(ValueError, match="lies on"):
        segment_field(a, b, 0.5 * (a + b))
