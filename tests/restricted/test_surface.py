"""``dmipy_analytic.restricted.surface``: water on the surface of a straight tube, against independent routes.

1. The narrow-pulse limit: the ring's propagator series ``sum_m eps_m J_m(kR)^2 exp(-m^2 D Delta/R^2)``, reached as
   ``delta -> 0`` within a derived first-order bound.
2. The long-time limit: ``J_0(kR)^2``, the diffraction pattern of the ring, zero at the roots of ``J_0``.
3. The low-q limit: ``1 - <phi^2>/2`` with the exact second moment of the phase from the circle's correlation
   ``<(x(t) - x(t'))^2> = R^2 (1 - exp(-D |t - t'|/R^2))``, within the fourth-order remainder.
4. The truncation rule: the change of the echo with the order against ``sqrt(2) I_{M+2}(Y)``.
5. The outline family: the aspect-1 ellipse (a regular polygon) against the circle of its perimeter, within the
   distance between the two curves; and a Monte Carlo of the exact Brownian walk in ``(s, z)`` mapped by the outline,
   within the Bonferroni-corrected ``z`` standard errors plus its derived time-quadrature bias.
6. Factorisation: an angled PGSE against the paper's closed form ``E_perp(q sin beta) exp(-q^2 (Delta - delta/3) D
   cos^2 beta)``, and a capped tube against the joint propagation in the product of the outline's and the plates'
   modes.

7. Beyond the PGSE: the circle and the lamellae of a sheath under a trapezoidal OGSE below and past the circle's
   diffusion rate and under a gradient turning in the plane, each against the Monte Carlo of the exact walk, within the
   Bonferroni-corrected ``z`` standard errors plus the derived trapezoid-rule bias.

Plus the position matrices against the circle's identities and a quadrature of the multiplication by a polygon's
coordinates, the 180 as a conjugation against the effective waveform, the circle's invariance under a rotation of a
turning waveform, and the refusals. The rounding allowed a matrix-exponential product is derived once
(:func:`_rounding`).
"""
import numpy as np
import pytest

pytest.importorskip("scipy.special")

from scipy import special, stats

import dmipy_analytic.restricted.matrix_method as mm
from dmipy_analytic.constants import GAMMA
from dmipy_analytic.restricted import sheath, surface
from dmipy_analytic.restricted.matrix_method import MatrixPore, PeriodicPore, polygon_fourier

D = 0.8e-9          # m^2/s, the paper's surface diffusivity
U64 = 2.0 ** -53
DT = 1e-5           # s, the step that holds the paper's timings
B1 = (2.88e-3, 7.72e-3)     # (delta, Delta) of the paper's Table I at b = 1 ms/um^2
B3 = (4.62e-3, 9.45e-3)     # ... at b = 3
B6 = (6.14e-3, 10.97e-3)    # ... at b = 6
RAMP = 1e-3         # s, 500 mT/m at 500 T/m/s


def _rounding(lam, B, g, dt, refocus=()):
    """The rounding allowed one echo of :func:`mm._echo`: the matrix exponential is designed to a backward error of
    ``U64 ||tau H||_1`` (Al-Mohy and Higham 2009, SIAM J. Matrix Anal. Appl. 31:970), which a product of contractions
    passes on unamplified, plus ``N U64`` for each product of ``N`` modes."""
    runs = mm._runs(np.asarray(g, float), mm._refocus_steps(refocus, len(g)))
    total = 0.0
    for amp, ns in runs:
        if any(amp):
            H = mm._generator(D, lam, B, amp)
            total += U64 * ns * dt * float(np.abs(H).sum(axis=0).max())
        total += len(lam) * U64
    return total


def _circle_rounding(g, dt, diameter, n_modes, refocus=()):
    pore = PeriodicPore.circle(diameter, D, n_modes=n_modes)
    return _rounding(pore.lam, pore.B, np.asarray(g)[:, :2], dt, refocus)


def _tube_rounding(pore, g, dt, refocus=(), length=None):
    """The rounding of a tube's echo: the factor across the axis (:func:`_rounding`), the factor along it -- the
    plates' product for a capped tube, or for an infinite one the exponent's sum of ``n_t`` terms of five operations,
    ``5 n_t U64`` relative, which moves ``e^{-x}`` by at most that since ``x e^{-x} <= 1`` -- and their product."""
    across = _rounding(pore.lam, pore.B, g[:, :2], dt, refocus)
    if length is None:
        return across + 5 * len(g) * U64 + U64
    plates = MatrixPore("plane", length, D)
    return across + _rounding(plates.lam, plates.B[None], g[:, 2:], dt, refocus) + U64


def _truncation(g, dt, R, M):
    """The circle's truncation bound ``sqrt(2) I_{M+2}(Y)``, ``Y = gamma R int |G_perp| dt``."""
    Y = GAMMA * R * float(np.hypot(g[:, 0], g[:, 1]).sum()) * dt
    return np.sqrt(2.0) * special.iv(M + 2, Y)


def _effective(g, refocus):
    """The effective waveform: the sign of the gradient flipped after each 180."""
    marks = np.zeros(len(g))
    marks[[k for k in refocus if k < len(g)]] = 1
    return np.asarray(g) * ((-1.0) ** np.cumsum(marks))[:, None]


# ======================================================================== 1-2: the narrow-pulse and long-time limits
def _ring_series(kR, a_Delta, m_max=80):
    """The ring's narrow-pulse echo: impulses ``k`` apart in time ``Delta`` see the angle's propagator
    ``(1/2 pi) sum_m exp(i m (theta - theta0) - m^2 a Delta)``, and ``exp(-i k R cos theta) = sum_m (-i)^m J_m(kR)
    exp(i m theta)``, so ``E = sum_m J_m(kR)^2 exp(-m^2 a Delta)``, ``a = D/R^2``."""
    m = np.arange(1, m_max + 1)
    return special.j0(kR) ** 2 + 2.0 * np.sum(special.jv(m, kR) ** 2 * np.exp(-m ** 2 * a_Delta))


def _narrow_pgse(k, delta, n):
    """The effective PGSE of two rectangular lobes of one step ``delta`` each, ``2^n`` steps apart, of the area
    ``k / gamma`` (``k`` in rad/m)."""
    G = k / (GAMMA * delta)
    g = np.zeros((2 ** n + 1, 3))
    g[0, 0], g[-1, 0] = G, -G
    return g


def _finite_pulse_gap(kR, a_delta):
    """The bound on ``|E(delta) - E_NP|`` for the circle at a fixed ``k = gamma G delta``.

    With ``xi`` a lobe's mean position less its position at the lobe's centre, ``eta = k (xi_1 - xi_2)`` and
    ``|e^{-i eta} - 1 + i eta| <= eta^2/2``: ``E[eta^2]/2 <= (4/9) k^2 D delta`` from ``<(x(t) - x(t'))^2> <= D |t -
    t'|``; and ``|E[e^{-i phi_NP} eta]| <= k a R delta / 2 + (sqrt2/3) k^2 D delta``: given the angle at a lobe's
    centre, the mean of ``x`` a time ``u`` away is ``e^{-a u} x`` (``x`` is the first mode, ``a = D/R^2``), which
    leaves ``a R delta/8`` per half lobe, and the half lobe that faces the other pulse adds, by Cauchy-Schwarz, at
    most ``sqrt(D delta/18)`` times the change of the far pulse's conditional phase, whose angle derivative is at
    most ``kR``, over an angle step of variance ``a delta``. Together ``a delta (kR/2 + (sqrt2/3 + 4/9) (kR)^2)``.
    """
    return a_delta * (kR / 2.0 + (np.sqrt(2.0) / 3.0 + 4.0 / 9.0) * kR ** 2)


@pytest.mark.parametrize("kR", [1.0, 2.5, 4.0])
def test_the_narrow_pulse_limit_is_the_ring_propagator_series(kR):
    """At a fixed ``k = gamma G delta`` and ``Delta = R^2/D``, the finite-pulse echo approaches the ring's series as
    ``delta -> 0``, within the first-order bound :func:`_finite_pulse_gap` (plus the truncation and the rounding),
    and its gap shrinks with ``delta``."""
    R = 1e-6
    a = D / R ** 2
    Delta = 1.0 / a
    series = _ring_series(kR, a * Delta)
    gaps = []
    for n in (6, 10, 14):
        delta = Delta / 2 ** n
        g = _narrow_pgse(kR / R, delta, n)
        M = surface.circle_order(g, delta, 2 * R)
        E = surface.circle_signal(g, delta, 2 * R, D)
        allowance = (_finite_pulse_gap(kR, a * delta) + _truncation(g, delta, R, M)
                     + _circle_rounding(g, delta, 2 * R, M))
        assert abs(E - series) <= allowance, (n, abs(E - series), allowance)
        gaps.append(abs(E - series))
    assert gaps[0] > gaps[1] > gaps[2]


@pytest.mark.parametrize("kR", [1.0, special.jn_zeros(0, 1)[0], special.jn_zeros(1, 1)[0], special.jn_zeros(0, 2)[1]])
def test_the_long_time_limit_is_the_diffraction_pattern_of_the_ring(kR):
    """At ``Delta = 40 R^2/D`` every mode but the uniform one has decayed (``e^{-40}`` of the series) and narrow
    pulses see ``J_0(kR)^2``: zero at the roots of ``J_0`` (the second argument is the first root of ``J_1``)."""
    R = 1e-6
    a = D / R ** 2
    Delta = 40.0 / a
    n = 18
    delta = Delta / 2 ** n
    g = _narrow_pgse(kR / R, delta, n)
    M = surface.circle_order(g, delta, 2 * R)
    E = surface.circle_signal(g, delta, 2 * R, D)
    allowance = (_finite_pulse_gap(kR, a * delta) + np.exp(-40.0) + _truncation(g, delta, R, M)
                 + _circle_rounding(g, delta, 2 * R, M))
    assert abs(E - special.j0(kR) ** 2) <= allowance, (abs(E - special.j0(kR) ** 2), allowance)


# ========================================================================================= 3: the low-q limit
def _phase_second_moment(g_eff, dt, R):
    """``<phi^2>`` of the circle for the effective transverse waveform ``g_eff`` (n_t, 2), exact for a
    piecewise-constant waveform: ``gamma^2 (R^2/2) sum_jk G_j.G_k K_jk``, with ``<x_a(t) x_b(t')> = (R^2/2) delta_ab
    exp(-a |t - t'|)`` (``x`` and ``y`` are the first modes of the ring, ``a = D/R^2``). At an echo this is
    ``-(gamma^2/2) int int G(t).G(t') <(x(t) - x(t'))^2> dt dt'`` with ``<(x(t) - x(t'))^2> = R^2 (1 - exp(-a |t -
    t'|))``. ``K`` over one step is ``2 (a h - 1 + e^{-a h})/a^2``, over two steps ``k - j`` apart
    ``e^{-a (k-j) h} (e^{a h} - 1)(1 - e^{-a h})/a^2``, summed by the recursion ``H_k = e^{-a h} (H_{k-1} +
    G_{k-1})``. Returns the moment and the sum of the magnitudes of its terms (the scale of its rounding)."""
    a, h = D / R ** 2, dt
    own = 2.0 * (a * h + np.expm1(-a * h)) / a ** 2
    cross = np.expm1(a * h) * (-np.expm1(-a * h)) / a ** 2
    decay = np.exp(-a * h)
    H, H_abs = np.zeros(2), np.zeros(2)
    total = own * float(np.sum(g_eff * g_eff))
    size = total
    for k in range(1, len(g_eff)):
        H = decay * (H + g_eff[k - 1])
        H_abs = decay * (H_abs + np.abs(g_eff[k - 1]))
        total += 2.0 * cross * float(g_eff[k] @ H)
        size += 2.0 * cross * float(np.abs(g_eff[k]) @ H_abs)
    scale = GAMMA ** 2 * 0.5 * R ** 2
    return scale * total, scale * size


def test_the_low_q_limit_is_the_exact_second_moment_of_the_phase():
    """``E = 1 - <phi^2>/2 + R_4`` with ``|R_4| <= <phi^4>/24 <= Y^4/24``: the circle is symmetric under ``theta ->
    theta + pi``, so the odd moments vanish and ``E = <cos phi>``, and ``|phi| <= Y = gamma R int |G_perp| dt``.
    The trapezoidal PGSE (the paper's b = 3 timing, 1 ms ramps, the 180 as a conjugation) is scaled to ``Y = 0.02``;
    the rectangular PGSE's moment is also its closed form ``(gamma G R/a)^2 (2 a delta - 2 + 2 e^{-a delta} + 2
    e^{-a Delta} - e^{-a (Delta - delta)} - e^{-a (Delta + delta)})``."""
    R = 2e-6
    a = D / R ** 2
    delta, Delta = B3
    for ramp in (RAMP, 0.0):
        g, ref = surface.pgse_waveform(1.0, delta, Delta, DT, [np.cos(0.35), np.sin(0.35), 0.0], ramp=ramp)
        Y_unit = GAMMA * R * float(np.hypot(g[:, 0], g[:, 1]).sum()) * DT
        G = 0.02 / Y_unit
        g = G * g
        Y = G * Y_unit
        M = surface.circle_order(g, DT, 2 * R)
        E = surface.circle_signal(g, DT, 2 * R, D, refocus=ref)
        phi2, size = _phase_second_moment(_effective(g, ref)[:, :2], DT, R)
        allowance = Y ** 4 / 24.0 + _truncation(g, DT, R, M) + _circle_rounding(g, DT, 2 * R, M, ref)
        assert abs(E - (1.0 - phi2 / 2.0)) <= allowance, (ramp, abs(E - (1 - phi2 / 2)), allowance)
        if ramp == 0.0:
            closed = (GAMMA * G * R / a) ** 2 * (2 * a * delta - 2 + 2 * np.exp(-a * delta) + 2 * np.exp(-a * Delta)
                                                 - np.exp(-a * (Delta - delta)) - np.exp(-a * (Delta + delta)))
            # the recursion's rounding: ten operations a step, on terms whose magnitudes sum to ``size``
            assert abs(phi2 - closed) <= 10 * len(g) * U64 * size + 16 * U64 * abs(closed)


# =================================================================================== 4: the truncation rule
@pytest.mark.parametrize("R, timing", [(0.5e-6, B6), (2e-6, B6), (5e-6, B1), (5e-6, B6)])
def test_the_truncation_rule_bounds_the_change_of_the_echo_with_the_order(R, timing):
    """For the paper's radii and timings at 500 mT/m, the echo at every order ``M`` up to the rule's differs from the
    echo eight orders above the rule's by at most the sum of the two bounds (plus rounding), and the rule's order is
    the smallest whose bound is below ``TRUNCATION_TOLERANCE``."""
    delta, Delta = timing
    g, ref = surface.pgse_waveform(0.5, delta, Delta, DT, [1.0, 0.0, 0.0])
    M_rule = surface.circle_order(g, DT, 2 * R)
    assert _truncation(g, DT, R, M_rule) <= surface.TRUNCATION_TOLERANCE < _truncation(g, DT, R, M_rule - 1)
    M_ref = M_rule + 8
    E_ref = surface.circle_signal(g, DT, 2 * R, D, refocus=ref, n_modes=M_ref)
    slack = _truncation(g, DT, R, M_ref) + _circle_rounding(g, DT, 2 * R, M_ref, ref)
    for M in range(M_rule + 1):
        E = surface.circle_signal(g, DT, 2 * R, D, refocus=ref, n_modes=M)
        allowance = _truncation(g, DT, R, M) + _circle_rounding(g, DT, 2 * R, M, ref) + slack
        assert abs(E - E_ref) <= allowance, (M, abs(E - E_ref), allowance)


def _turning(G, n, cycles, phase=0.0):
    """A gradient of amplitude ``G`` whose direction turns ``cycles`` times in the plane across the axis over ``n``
    steps (no echo across the axis, none along it)."""
    t = 2.0 * np.pi * cycles * (np.arange(n) + 0.5) / n + phase
    return np.stack([G * np.cos(t), G * np.sin(t), np.zeros(n)], axis=1)


def test_a_turning_gradient_obeys_the_truncation_rule_and_the_circle_s_symmetry():
    """A gradient whose direction turns couples both families of modes (``cos k theta`` and ``sin k theta``): the
    bound holds order by order, and rotating the whole waveform about the axis leaves the echo of the circle
    unchanged (the truncated basis is closed under rotations, which fix the uniform mode)."""
    R, dt = 2e-6, 2e-5
    g = _turning(0.3, 240, 3.0)
    M_rule = surface.circle_order(g, dt, 2 * R)
    M_ref = M_rule + 8
    E_ref = surface.circle_signal(g, dt, 2 * R, D, n_modes=M_ref)
    slack = _truncation(g, dt, R, M_ref) + _circle_rounding(g, dt, 2 * R, M_ref)
    for M in range(0, M_rule + 1, 3):
        E = surface.circle_signal(g, dt, 2 * R, D, n_modes=M)
        assert abs(E - E_ref) <= _truncation(g, dt, R, M) + _circle_rounding(g, dt, 2 * R, M) + slack
    E_rot = surface.circle_signal(_turning(0.3, 240, 3.0, phase=0.7), dt, 2 * R, D, n_modes=M_rule)
    E_rule = surface.circle_signal(g, dt, 2 * R, D, n_modes=M_rule)
    assert abs(E_rot - E_rule) <= 2 * _circle_rounding(g, dt, 2 * R, M_rule)


# ======================================================================== the 180 as a conjugation; the refusals
def test_the_180_as_a_conjugation_is_the_effective_waveform_conjugated():
    """The waveform as played with its 180s conjugating the magnetisation gives the complex conjugate of the effective
    waveform's echo after an odd number of 180s and the same echo after an even number: on the circle, on an
    outline, on a capped tube and on an infinite one."""
    outline = np.array([[2e-6, 0.0], [0.4e-6, 1.5e-6], [-1.8e-6, 0.5e-6], [-1.0e-6, -1.4e-6], [1.0e-6, -1.2e-6]])
    g, ref = surface.pgse_waveform(0.3, 2e-3, 6e-3, 1e-4, [0.6, -0.2, 0.5], ramp=5e-4)
    double = np.concatenate([g, g])                                   # a double spin echo: two 180s
    double_ref = (ref[0], len(g), len(g) + ref[0])
    for played, refocus in ((g, ref), (double, double_ref)):
        effective = _effective(played, refocus)
        conj = len(refocus) % 2 == 1
        for length in (None, 3e-6):
            for pore, form in ((PeriodicPore.circle(3e-6, D, n_modes=12),
                                lambda w, **kw: surface.circle_signal(w, 1e-4, 3e-6, D, n_modes=12, **kw)),
                               (PeriodicPore.polygon(outline, D, n_modes=12),
                                lambda w, **kw: surface.outline_signal(w, 1e-4, outline, D, n_modes=12, **kw))):
                E_played = form(played, refocus=refocus, length=length)
                E_eff = form(effective, length=length)
                budget = (_tube_rounding(pore, played, 1e-4, refocus, length)
                          + _tube_rounding(pore, effective, 1e-4, (), length))
                assert abs(E_played - (np.conj(E_eff) if conj else E_eff)) <= budget


def test_the_refusals_name_what_is_wrong():
    g, ref = surface.pgse_waveform(0.3, 2e-3, 6e-3, 1e-4, [0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="axial moment"):
        surface.circle_signal(g, 1e-4, 2e-6, D)                     # the 180 left out: no echo along the axis
    with pytest.raises(ValueError, match="whole number of steps"):
        surface.pgse_waveform(0.3, 2.05e-3, 6e-3, 1e-4, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="ramp <= delta"):
        surface.pgse_waveform(0.3, 2e-3, 3e-3, 1e-4, [1.0, 0.0, 0.0], ramp=1.5e-3)
    with pytest.raises(ValueError, match="strictly increasing"):
        surface.circle_signal(g, 1e-4, 2e-6, D, refocus=(len(g) + 1,))
    with pytest.raises(ValueError, match="distinct"):
        surface.outline_signal(g, 1e-4, [[0.0, 0.0], [1e-6, 0.0], [1e-6, 0.0], [0.0, 1e-6]], D, n_modes=4,
                               refocus=ref)
    with pytest.raises(ValueError, match=r"\(n_t, 3\)"):
        surface.circle_signal(g[:, :2], 1e-4, 2e-6, D)


# ================================================================================ 5: the outline family
def _ellipse(a, b, n):
    t = 2.0 * np.pi * np.arange(n) / n
    return np.stack([a * np.cos(t), b * np.sin(t)], axis=1)


def _rounded_triangle(R, eps, n):
    t = 2.0 * np.pi * np.arange(n) / n
    r = R * (1.0 + eps * np.cos(3.0 * t))
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _arc_table(V, n_tab):
    """The closed polygon ``V`` sampled at ``n_tab + 1`` points equally spaced in arc length: ``(P, X, Y)``."""
    Vc = np.vstack([V, V[:1]])
    s_v = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Vc, axis=0).T))])
    grid = np.linspace(0.0, s_v[-1], n_tab + 1)
    return s_v[-1], np.interp(grid, s_v, Vc[:, 0]), np.interp(grid, s_v, Vc[:, 1])


def _real_fourier_basis(s, P, M):
    """The basis of :class:`PeriodicPore` at the arc lengths ``s``: ``1, sqrt2 cos(k alpha), sqrt2 sin(k alpha)``,
    ``alpha = 2 pi s/P``, interleaved by order."""
    alpha = 2.0 * np.pi * s / P
    U = [np.ones_like(s)]
    for k in range(1, M + 1):
        U += [np.sqrt(2.0) * np.cos(k * alpha), np.sqrt(2.0) * np.sin(k * alpha)]
    return np.stack(U, axis=1)


def test_the_position_matrices_are_the_multiplication_by_the_coordinates():
    """The coupling of a periodic pore is the truncated multiplication by ``x(s)`` and ``y(s)``.

    * The circle: multiplication operators commute and ``x^2 + y^2 = R^2``, so on the orders below the cut (where a
      product of two matrices of bandwidth one meets no dropped mode) ``[B_x, B_y] = 0`` and ``B_x^2 + B_y^2 = R^2``,
      to rounding.
    * A polygon (a rounded triangle, no symmetry to hide a sign): ``<u_i| x |u_j>`` by the trapezoid rule over the
      perimeter on ``n`` points, ``x(s)`` interpolated along the edges -- a route independent of the coefficients'
      closed form and of the matrices' assembly. The rule's error over a cell is ``h^2/12`` times the integral of
      ``|f''|`` there, and a cell holding a vertex adds ``h^2/8`` times the jump of ``f'``; with ``f = x u_i u_j``,
      ``|u| <= sqrt2``, ``|u'| <= sqrt2 w``, ``|u''| <= sqrt2 w^2`` (``w = 2 pi M/P``) and ``|x'| <= 1``: ``|f''| <=
      8 w^2 max|x| + 8 w`` and the jumps ``2 |turn|`` at each vertex.
    """
    M = 9
    circle = PeriodicPore.circle(4e-6, D, n_modes=M)
    R = 2e-6
    inner = slice(0, 2 * M - 1)                                    # the orders below M
    for product in (circle.B[0] @ circle.B[1] - circle.B[1] @ circle.B[0],
                    circle.B[0] @ circle.B[0] + circle.B[1] @ circle.B[1] - R ** 2 * np.eye(2 * M + 1)):
        assert np.max(np.abs(product[inner, inner])) <= 4 * (2 * M + 1) * U64 * R ** 2
    V = _rounded_triangle(2.5e-6, 0.15, 96)
    pore = PeriodicPore.polygon(V, D, n_modes=M)
    n = 2 ** 16
    P, X, Y = _arc_table(V, n)
    s = np.linspace(0.0, P, n + 1)[:-1]
    U = _real_fourier_basis(s, P, M)
    h, w = P / n, 2.0 * np.pi * M / P
    edge = np.diff(np.vstack([V, V[:1]]), axis=0)
    tangent = edge / np.hypot(edge[:, 0], edge[:, 1])[:, None]
    turns = float(np.hypot(*(tangent - np.roll(tangent, 1, axis=0)).T).sum())
    for axis, coordinate in enumerate((X[:-1], Y[:-1])):
        quadrature = (U * coordinate[:, None]).T @ U / n
        f2 = 8.0 * w ** 2 * float(np.max(np.abs(coordinate))) + 8.0 * w
        rounding = n * U64 * 2.0 * float(np.max(np.abs(coordinate)))     # a mean of n terms of size 2 max|x|
        allowance = h ** 2 / 12.0 * f2 + h ** 2 / 8.0 * 2.0 * turns / P + rounding
        assert np.max(np.abs(pore.B[axis] - quadrature)) <= allowance, axis


def test_the_aspect_one_ellipse_is_the_circle_of_its_perimeter():
    """The aspect-1 ellipse sampled at ``n`` points is the regular ``n``-gon; against the circle of its perimeter,
    both walked at unit speed from the same point, it has the same eigenvalues, and

    * its position matrices differ from the circle's by at most the distance between the two curves: ``B_poly -
      B_circle`` is the truncated multiplication by ``x_poly(s) - x_circle(s)`` (and ``y``), whose norm is at most
      that function's largest value, taken on a grid of 64 points per edge plus the grid's error ``h^2/(8 R)`` (the
      circle's curvature);
    * its echo differs from the circle's by at most ``gamma int |G_perp| dt ||B_poly - B_circle||``: the two
      truncated propagators share their decay and are contractions, so their difference is the integral of the
      difference of the couplings (Duhamel).
    """
    r, n = 2e-6, 1024
    V = _ellipse(r, r, n)
    P, _ = polygon_fourier(V, 0)
    assert P == pytest.approx(2 * n * r * np.sin(np.pi / n), rel=n * U64)
    Rc = P / (2.0 * np.pi)
    delta, Delta = B3
    g, ref = surface.pgse_waveform(0.5, delta, Delta, DT, [0.6, 0.8, 0.0], ramp=RAMP)
    M = surface.circle_order(g, DT, 2 * Rc)
    poly = PeriodicPore.polygon(V, D, n_modes=M)
    circle = PeriodicPore.circle(2 * Rc, D, n_modes=M)
    assert np.array_equal(poly.lam, circle.lam)
    s = np.linspace(0.0, P, 64 * n + 1)
    _, X, Y = _arc_table(V, 64 * n)
    grid = (P / (64 * n)) ** 2 / (8 * Rc)
    for axis, (on_polygon, on_circle) in enumerate(((X, Rc * np.cos(s / Rc)), (Y, Rc * np.sin(s / Rc)))):
        distance = float(np.max(np.abs(on_polygon - on_circle))) + grid
        # rounding: a coefficient sums n terms of total size ~ R, an entry two coefficients, the norm N entries
        assert np.linalg.norm(poly.B[axis] - circle.B[axis], 2) <= distance + 2 * n * len(poly.lam) * U64 * Rc
    dB = np.hypot(np.linalg.norm(poly.B[0] - circle.B[0], 2), np.linalg.norm(poly.B[1] - circle.B[1], 2))
    E_n = surface.outline_signal(g, DT, V, D, n_modes=M, refocus=ref)
    E_c = surface.circle_signal(g, DT, 2 * Rc, D, n_modes=M, refocus=ref)
    winding = GAMMA * float(np.hypot(g[:, 0], g[:, 1]).sum()) * DT
    allowance = winding * dB + _tube_rounding(poly, g, DT, ref) + _tube_rounding(circle, g, DT, ref)
    assert abs(E_n - E_c) <= allowance, (abs(E_n - E_c), allowance)


def _monte_carlo(V, g, dt, refocus, n_walkers, rng, n_sub, n_tab):
    """The echo of the exact Brownian walk on the tube's surface: ``s`` (on the circle of length ``P``) and ``z``
    take Gaussian steps of variance ``2 D h``, the outline maps ``s`` to ``(x, y)`` through a table of ``n_tab``
    points equally spaced in arc length, the phase of each step of constant gradient is the trapezoid rule over its
    ``n_sub`` sub-steps, a 180 negates the phase, and a stretch without gradient is one exact step. Returns the mean
    of ``cos(phi)``, its standard error and the mean of ``sin(phi)``."""
    P, X, Y = _arc_table(V, n_tab)
    dX, dY, scale = np.diff(X), np.diff(Y), n_tab / P

    def position(s):
        u = np.mod(s, P) * scale
        i = np.minimum(u.astype(np.int64), n_tab - 1)
        f = u - i
        return X[i] + f * dX[i], Y[i] + f * dY[i]

    s = rng.uniform(0.0, P, n_walkers)
    z = np.zeros(n_walkers)
    phase = np.zeros(n_walkers)
    flips, h, idle = set(refocus), dt / n_sub, 0.0
    for k in range(len(g)):
        if k in flips:
            phase = -phase
        if not np.any(g[k]):
            idle += dt
            continue
        if idle > 0.0:
            s += np.sqrt(2.0 * D * idle) * rng.standard_normal(n_walkers)
            z += np.sqrt(2.0 * D * idle) * rng.standard_normal(n_walkers)
            idle = 0.0
        x, y = position(s)
        for _ in range(n_sub):
            s1 = s + np.sqrt(2.0 * D * h) * rng.standard_normal(n_walkers)
            z1 = z + np.sqrt(2.0 * D * h) * rng.standard_normal(n_walkers)
            x1, y1 = position(s1)
            phase += GAMMA * h * 0.5 * (g[k, 0] * (x + x1) + g[k, 1] * (y + y1) + g[k, 2] * (z + z1))
            s, z, x, y = s1, z1, x1, y1
    if len(g) in flips:
        phase = -phase
    c = np.cos(phase)
    return float(c.mean()), float(c.std(ddof=1)) / np.sqrt(n_walkers), float(np.sin(phase).mean())


MC_CASES = [
    ("ellipse 3x2 um, along its major axis", _ellipse(3e-6, 2e-6, 512), [1.0, 0.0, 0.0]),
    ("ellipse 3x2 um, 52 deg to the axis", _ellipse(3e-6, 2e-6, 512), [0.0, np.sin(0.9), np.cos(0.9)]),
    ("rounded triangle, 70 deg to the axis", _rounded_triangle(2.5e-6, 0.15, 512),
     [np.sin(1.22) * np.cos(0.35), np.sin(1.22) * np.sin(0.35), np.cos(1.22)]),
]


@pytest.mark.parametrize("case", range(len(MC_CASES)))
def test_an_outline_is_its_monte_carlo_walk(case):
    """The matrix route against the Monte Carlo of the exact walk in ``(s, z)`` (:func:`_monte_carlo`, 20,000
    walkers, the paper's PGSE at b = 1: 500 mT/m, 1 ms ramps), which assumes nothing about factorisation. Agreement
    within ``z SE + bias``: ``z`` the two-sided family-wise level 1e-3 (3.29 for one comparison) Bonferroni-corrected
    over the comparisons, the walks on independent streams, and the bias of the trapezoid rule per sub-step ``h``:
    the bridge's fluctuation ``gamma h sqrt(D int G^2 dt/6)`` and its conditional mean through the outline's
    curvature, ``gamma int |G| dt kappa D h/3`` (``kappa`` the polygon's largest turn per length), plus the table's
    distance from the polygon times ``gamma int |G_perp| dt``. The outline's order is the circle's at ``P/(2 pi)``
    plus 30, confirmed by raising it ten orders: the echo moves by less than a hundredth of the allowance."""
    name, V, direction = MC_CASES[case]
    delta, Delta = B1
    g, ref = surface.pgse_waveform(0.5, delta, Delta, DT, direction, ramp=RAMP)
    P, _ = polygon_fourier(V, 0)
    M = surface.circle_order(g, DT, P / np.pi) + 30
    E = surface.outline_signal(g, DT, V, D, n_modes=M, refocus=ref)
    rng = np.random.default_rng(np.random.SeedSequence(20261102).spawn(len(MC_CASES))[case])
    n_sub, n_tab = 2, 2 ** 16
    mean, se, _ = _monte_carlo(V, g, DT, ref, 20_000, rng, n_sub, n_tab)
    z = stats.norm.isf(0.5e-3 / len(MC_CASES))
    h = DT / n_sub
    Vc = np.vstack([V, V[:1]])
    edge = np.diff(Vc, axis=0)
    length = np.hypot(edge[:, 0], edge[:, 1])
    tangent = edge / length[:, None]
    turn = np.hypot(*(tangent - np.roll(tangent, 1, axis=0)).T)
    kappa = float(np.max(turn / (0.5 * (length + np.roll(length, 1)))))
    area = GAMMA * float(np.linalg.norm(g, axis=1).sum()) * DT
    bias = (GAMMA * h * np.sqrt(D * float(np.sum(g * g)) * DT / 6.0) + area * kappa * D * h / 3.0
            + GAMMA * float(np.hypot(g[:, 0], g[:, 1]).sum()) * DT * float(np.max(turn)) * (P / n_tab) / 4.0)
    allowance = z * se + bias
    assert abs(E.real - mean) <= allowance, (name, E.real, mean, se, bias)
    E_up = surface.outline_signal(g, DT, V, D, n_modes=M + 10, refocus=ref)
    assert abs(E_up - E) <= 0.01 * allowance


# ================================================================================== 6: factorisation
@pytest.mark.parametrize("beta", [0.0, 0.3, np.pi / 4, 1.2, np.pi / 2])
def test_an_angled_pgse_is_the_paper_s_product_of_the_two_factors(beta):
    """A rectangular PGSE at the angle ``beta`` to an infinite tube is ``E_perp(q sin beta) exp(-q^2 (Delta - delta/3)
    D cos^2 beta)`` with ``q = gamma G delta`` (the paper's Eqs. 36 and 37): the factor across the axis from the
    gradient's transverse part alone, the one along it Stejskal and Tanner's."""
    R, G = 2e-6, 0.5
    delta, Delta = B3
    g, ref = surface.pgse_waveform(G, delta, Delta, DT, [np.sin(beta), 0.0, np.cos(beta)])
    M = surface.circle_order(g, DT, 2 * R)
    E = surface.circle_signal(g, DT, 2 * R, D, refocus=ref, n_modes=M)
    g_perp, _ = surface.pgse_waveform(G * np.sin(beta), delta, Delta, DT, [1.0, 0.0, 0.0])
    E_perp = surface.circle_signal(g_perp, DT, 2 * R, D, refocus=ref, n_modes=M)
    q = GAMMA * G * delta
    E_par = np.exp(-q ** 2 * (Delta - delta / 3.0) * D * np.cos(beta) ** 2)
    # the two transverse waveforms differ by the rounding of their amplitudes, 2 U64 each, which moves the echo by
    # at most gamma R int |dG| dt (the coupling of two walks); the closed form is a handful of operations
    pore = PeriodicPore.circle(2 * R, D, n_modes=M)
    Y = GAMMA * R * float(np.hypot(g[:, 0], g[:, 1]).sum()) * DT
    budget = (_tube_rounding(pore, g, DT, ref) + _tube_rounding(pore, g_perp, DT, ref) + 2 * U64 * Y
              + 8 * U64)
    assert abs(E - E_perp * E_par) <= budget


def test_a_capped_tube_is_the_joint_propagation_in_the_product_of_the_two_bases():
    """The surface of a tube of length ``L`` capped by reflecting ends has the eigenfunctions ``u_k(s) v_n(z)`` with
    the eigenvalues summed, and the gradient couples them through ``g_perp . C(s) + g_z z``: the generator is the
    Kronecker sum of the outline's and the plates', so propagating jointly in the product basis, with an oblique
    trapezoidal PGSE and the 180 as a conjugation, gives the product of the two factors to rounding."""
    V = _ellipse(1.5e-6, 1e-6, 64)
    L = 3e-6
    M, K = 8, 10
    g, ref = surface.pgse_waveform(0.25, 2e-3, 5e-3, 1e-4, [0.5, 0.3, 0.81], ramp=5e-4)
    E = surface.outline_signal(g, 1e-4, V, D, n_modes=M, length=L, refocus=ref, axial_modes=K)
    ring = PeriodicPore.polygon(V, D, n_modes=M)
    plates = MatrixPore("plane", L, D, n_modes=K)
    I_s, I_z = np.eye(len(ring.lam)), np.eye(len(plates.lam))
    lam = (ring.lam[:, None] + plates.lam[None, :]).ravel()
    B = np.stack([np.kron(ring.B[0], I_z), np.kron(ring.B[1], I_z), np.kron(I_s, plates.B)])
    E_joint = mm._echo(lam, D, B, g, 1e-4, refocus=ref)
    budget = (_rounding(lam, B, g, 1e-4, ref) + _rounding(ring.lam, ring.B, g[:, :2], 1e-4, ref)
              + _rounding(plates.lam, plates.B[None], g[:, 2:], 1e-4, ref))
    assert abs(E - E_joint) <= budget
    assert abs(E.imag) <= budget and 0.0 < E.real < 1.0


# ================================================== 7: beyond PGSE -- trapezoidal OGSE and a turning gradient
def _ogse(G, f, cycles, dt, ramp):
    """A trapezoidal oscillating gradient along ``x`` as the effective waveform: ``cycles`` periods of ``1/f``, each a
    positive and a negative lobe of amplitude ``G`` whose linear ramps last ``ramp`` (midpoint-sampled, so each step
    keeps its area)."""
    n_half, n_ramp = int(round(0.5 / f / dt)), int(round(ramp / dt))
    up = G * (np.arange(n_ramp) + 0.5) / n_ramp
    lobe = np.concatenate([up, np.full(n_half - 2 * n_ramp, G), up[::-1]])
    g = np.tile(np.concatenate([lobe, -lobe]), cycles)
    return np.stack([g, np.zeros_like(g), np.zeros_like(g)], axis=1)


def _lamellae_walk(g, dt, r_in, r_out, n_walkers, rng, n_sub):
    """The echo of the exact Brownian walks on the lamellae of a sheath: each walker starts at a radius drawn with
    density proportional to ``r`` on ``[r_in, r_out]`` and an angle uniform on the circle, and stays on its lamella,
    where the angle takes Gaussian steps of variance ``2 D h / r^2``; the phase is the trapezoid rule over the ``n_sub``
    sub-steps of each step of the gradient. Returns the mean of ``cos(phi)`` and its standard error."""
    r = np.sqrt(rng.uniform(r_in ** 2, r_out ** 2, n_walkers))
    theta = rng.uniform(0.0, 2.0 * np.pi, n_walkers)
    phase = np.zeros(n_walkers)
    h = dt / n_sub
    sd = np.sqrt(2.0 * D * h) / r
    x, y = r * np.cos(theta), r * np.sin(theta)
    for k in range(len(g)):
        for _ in range(n_sub):
            theta = theta + sd * rng.standard_normal(n_walkers)
            x1, y1 = r * np.cos(theta), r * np.sin(theta)
            phase += GAMMA * h * 0.5 * (g[k, 0] * (x + x1) + g[k, 1] * (y + y1))
            x, y = x1, y1
    c = np.cos(phase)
    return float(c.mean()), float(c.std(ddof=1)) / np.sqrt(n_walkers)


BEYOND_DT = 1e-4          # s; D dt / R^2 <= 0.01 on the smallest tube below
BEYOND_R = 3e-6           # m, the circle's radius: D / R^2 = 89 / s
BEYOND_SHEATH = (1.8e-6, 3.0e-6)       # m, the lamellae's innermost and outermost radii
BEYOND_N = 20_000
BEYOND_SUB = 5
#: (name, effective waveform): 12.5 Hz is below the circle's diffusion rate (2 pi f = 79 < 89 / s), 25 and 50 Hz are past it
#: (157 and 314 / s); the amplitudes bring the echo of each to the middle of (0, 1), where a wrong exponent shows.
BEYOND_WAVEFORMS = [
    ("OGSE 12.5 Hz, 70 mT/m", _ogse(0.07, 12.5, 2, BEYOND_DT, 5e-4)),
    ("OGSE 25 Hz, 100 mT/m", _ogse(0.10, 25.0, 2, BEYOND_DT, 5e-4)),
    ("OGSE 50 Hz, 200 mT/m", _ogse(0.20, 50.0, 2, BEYOND_DT, 5e-4)),
    ("gradient turning 2 cycles in the plane, 100 mT/m", _turning(0.10, 800, 2.0)),
]
#: The rows sharing the family-wise level: every waveform on the circle and on the lamellae.
BEYOND_ROWS = 2 * len(BEYOND_WAVEFORMS)


def _beyond_allowance(g, se, kappa):
    """``z SE`` plus the bias of the trapezoid rule per sub-step ``h``, as in :func:`test_an_outline_is_its_monte_carlo_walk`:
    the bridge's fluctuation ``gamma h sqrt(D int G^2 dt / 6)`` and its conditional mean through the curvature
    ``kappa`` of the path, ``gamma int |G| dt kappa D h / 3`` (``kappa`` the largest curvature, ``1 / r_min``). ``z`` is
    the two-sided family-wise level 1e-3 Bonferroni-corrected over the :data:`BEYOND_ROWS` rows."""
    h = BEYOND_DT / BEYOND_SUB
    area = GAMMA * float(np.linalg.norm(g, axis=1).sum()) * BEYOND_DT
    bias = GAMMA * h * np.sqrt(D * float(np.sum(g * g)) * BEYOND_DT / 6.0) + area * kappa * D * h / 3.0
    return stats.norm.isf(0.5e-3 / BEYOND_ROWS) * se + bias


@pytest.mark.parametrize("case", range(len(BEYOND_WAVEFORMS)))
def test_the_circle_is_its_monte_carlo_walk_under_ogse_and_a_turning_gradient(case):
    """The circle's matrix route (:func:`surface.circle_signal`) against the Monte Carlo of the exact walk on the
    ring (:func:`_monte_carlo`, 20,000 walkers on a 4096-gon, which the 2^16-point arc table follows to
    ``1e-9 R``), for a trapezoidal OGSE below and past the circle's diffusion rate and for a gradient turning in the
    plane. Seeds ``SeedSequence(20261108)`` spawn 0-3; agreement within :func:`_beyond_allowance`."""
    name, g = BEYOND_WAVEFORMS[case]
    ring = np.stack([BEYOND_R * np.cos(2.0 * np.pi * np.arange(4096) / 4096),
                     BEYOND_R * np.sin(2.0 * np.pi * np.arange(4096) / 4096)], axis=1)
    E = surface.circle_signal(g, BEYOND_DT, 2 * BEYOND_R, D)
    rng = np.random.default_rng(np.random.SeedSequence(20261108).spawn(BEYOND_ROWS)[case])
    mean, se, _ = _monte_carlo(ring, g, BEYOND_DT, (), BEYOND_N, rng, BEYOND_SUB, 2 ** 16)
    assert 0.05 < E.real < 0.95, name
    assert abs(E.real - mean) <= _beyond_allowance(g, se, 1.0 / BEYOND_R), (name, E.real, mean, se)


@pytest.mark.parametrize("case", range(len(BEYOND_WAVEFORMS)))
def test_the_lamellae_are_their_monte_carlo_walk_under_ogse_and_a_turning_gradient(case):
    """The lamellae's average (:func:`sheath.lamellae_signal`, the Gauss-Legendre rule in the radius) against walkers
    that start at a radius drawn with density proportional to ``r`` and stay on their lamella
    (:func:`_lamellae_walk`, 20,000 walkers), for the waveforms of the circle's test, on seeds 4-7 of
    ``SeedSequence(20261108)``; agreement within :func:`_beyond_allowance`, whose curvature is the innermost lamella's."""
    name, g = BEYOND_WAVEFORMS[case]
    r_in, r_out = BEYOND_SHEATH
    E = sheath.lamellae_signal(g, BEYOND_DT, 2 * r_out, 2 * r_in, D)
    rng = np.random.default_rng(np.random.SeedSequence(20261108).spawn(BEYOND_ROWS)[len(BEYOND_WAVEFORMS) + case])
    mean, se = _lamellae_walk(g, BEYOND_DT, r_in, r_out, BEYOND_N, rng, BEYOND_SUB)
    assert 0.05 < E.real < 0.95, name
    assert abs(E.real - mean) <= _beyond_allowance(g, se, 1.0 / r_in), (name, E.real, mean, se)
