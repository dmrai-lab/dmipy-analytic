"""The eigenmode (matrix) solution of restricted diffusion in a planar, cylindrical or spherical pore under a
generalised gradient waveform -- Callaghan's 1997 matrix formalism (J. Magn. Reson. 129:74-84), extended to these
three pores by Codd & Callaghan 1999 (J. Magn. Reson. 137:358-372, doi:10.1006/jmre.1998.1679), and restated by
Barzykin 1999 (J. Magn. Reson. 139:342-353) and Grebenkov's review (Rev. Mod. Phys. 79:1077-1137, 2007).

The Bloch-Torrey equation ``dm/dt = D grad^2 m - i gamma (g(t).r) m`` with reflecting (Neumann) walls is solved in the
Laplacian eigenbasis: expand ``m`` in the orthonormal eigenfunctions of the pore (``u_0`` constant, eigenvalue 0).
Over a step of duration ``tau`` at a constant gradient ``g`` along the measured axis the coefficients advance by
``c <- exp(-tau (D Lambda + i gamma g B)) c``, with ``Lambda`` the eigenvalues and ``B_{mn} = <u_m| ghat.r |u_n>`` the
position-projection matrix; the signal is the echo's ``(0, 0)`` entry of the ordered propagator product. This is exact
up to the number of eigenmodes kept, whose convergence is an argument rather than assumed; the Gaussian-phase models
(Van Gelderen for the cylinder, Murday-Cotts for the sphere) are its low-b limit, Callaghan's narrow-pulse expression
its short-delta limit.

A piecewise-constant waveform has a handful of distinct RUNS of constant amplitude, so the exact product costs one
matrix exponential per run (:func:`_runs`) rather than one per sample -- a slew-limited PGSE has a few amplitudes from
its ramps but just as few runs and stays exact at that cost. A waveform whose amplitude changes at (almost) every
sample (an OGSE cosine) is genuinely dense; past ``_MAX_EXACT_RUNS`` runs the signal is a Strang split of the
diffusion and precession operators, whose Trotter error is ``O(dt^2)``. A waveform is the effective one (the sign of
each lobe after a refocusing pulse folded in), or the one as played with the ideal 180s as conjugations of the
magnetisation (``refocus``): one propagation (:func:`_echo`) serves every pore.

:class:`PeriodicPore` is the closed curve walked along its arc length: the Fourier modes of the circle of its
perimeter, and a position matrix per axis of the curve's plane built from the Fourier coefficients of its coordinates
(the circle, or the closed polygon through given points). It is the cross-section of the surface of a straight tube
(:mod:`dmipy_analytic.restricted.surface`).

The eigenvalue roots and the dimensionless position matrix are quadratured once per ``(shape, n_modes)`` on the UNIT
geometry (radius or thickness 1) and cached; a pore of size ``a`` reads ``lambda = lambda_unit / a**2`` and
``B = a * B_unit`` (a scalar times a cached matrix, not a requadrature), so a sweep over pore size pays one
eigendecomposition. The eigenfunction norms and ``B`` come from direct quadrature rather than closed-form Lommel
integrals, so the only approximations are the number of modes and the quadrature resolution, both arguments (module
constants, since both shapes and sizes share one resolution).

**Assumptions.** Impermeable walls without surface relaxivity (the Neumann condition), the magnetisation uniform in
the pore at the start, one free diffusivity ``D`` inside, no relaxation, and a gradient along one fixed axis given as
its amplitude per step (piecewise constant over each step ``dt``): the plane's normal, a direction perpendicular to the
cylinder's axis, any direction for the sphere; the periodic pore takes both axes of its curve's plane, the gradient's
direction free to turn. A pore's size is the plates' separation (plane) or its diameter (cylinder, sphere); a periodic
pore is the circle's diameter or the outline's points, in metres.

The forms need scipy (the ``special`` extra), imported inside them.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .._extras import need
from ..constants import GAMMA

#: Quadrature nodes across the unit pore's radius/thickness and around its angle.
RADIAL_NODES = 4000
ANGULAR_NODES = 2048

#: The run-product path costs one matrix exponential per run of constant amplitude (``_runs``), not per
#: distinct amplitude -- a slew-limited PGSE has a handful of amplitudes from its ramps but just as few runs
#: (12 at 200 T/m/s, 32 at 50 T/m/s) and is exact at that cost. A waveform whose amplitude changes at (almost) every
#: sample -- an OGSE cosine, ~1000 runs at n_t=1000 -- is genuinely dense and this ceiling sends it to the Strang
#: split instead.
_MAX_EXACT_RUNS = 64

__all__ = ["MatrixPore", "AnnulusPore", "PeriodicPore", "polygon_fourier", "matrix_restricted_signal",
           "matrix_restricted_batch", "cylinder_neumann_roots", "sphere_neumann_roots", "annulus_neumann_roots",
           "DEFAULT_MODES", "ANNULUS_DEFAULT_MODES"]


def cylinder_neumann_roots(m, n_roots):
    """The first ``n_roots`` non-negative roots of ``J_m'(x) = 0``; ``m = 0`` includes the constant mode at 0: the
    Neumann eigenvalues of the unit disk, ``lambda = x^2`` for the mode ``J_m(x r) cos(m theta)``. ``n_roots`` is at
    least 1; for ``m = 0`` one root is the constant mode alone."""
    special = need("scipy.special")
    if n_roots < 1:
        raise ValueError(f"n_roots must be at least 1, not {n_roots}")
    if m == 0:
        return np.concatenate([[0.0], special.jnp_zeros(0, n_roots - 1)]) if n_roots > 1 else np.zeros(1)
    return special.jnp_zeros(m, n_roots)


def sphere_neumann_roots(l, n_roots):
    """The first ``n_roots`` non-negative roots of ``j_l'(x) = 0``; ``l = 0`` includes the constant mode at 0: the
    Neumann eigenvalues of the unit ball, ``lambda = x^2`` for the mode ``j_l(x r) P_l(cos theta)``. Bracketed on a grid
    of 40,000 points and refined by Brent's method. ``n_roots`` is at least 1; for ``l = 0`` one root is the constant
    mode alone."""
    special = need("scipy.special")
    brentq = need("scipy.optimize").brentq
    if n_roots < 1:
        raise ValueError(f"n_roots must be at least 1, not {n_roots}")
    wanted = n_roots - 1 if l == 0 else n_roots               # the constant mode of l = 0 is the first of the n_roots
    if wanted == 0:
        return np.zeros(1)
    xs = np.linspace(1e-6, 4 * (n_roots + l + 2) + 20, 40000)
    sign = np.sign(special.spherical_jn(l, xs, derivative=True))
    roots = []
    for i in np.where(np.diff(sign) != 0)[0]:
        try:
            r = brentq(lambda x: special.spherical_jn(l, x, derivative=True), xs[i], xs[i + 1])
        except Exception:
            continue
        if not roots or r - roots[-1] > 1e-3:
            roots.append(r)
        if len(roots) >= wanted:
            break
    roots = np.asarray(roots)
    return np.concatenate([[0.0], roots]) if l == 0 else roots


def annulus_neumann_roots(m, ratio, n_roots):
    """The first ``n_roots`` non-negative roots of the Neumann condition on the annulus ``ratio <= r <= 1`` at the
    angular order ``m``, ``J_m'(x ratio) Y_m'(x) - J_m'(x) Y_m'(x ratio) = 0``: the eigenvalues ``lambda = x^2`` of the
    modes ``[J_m(x r) Y_m'(x) - Y_m(x r) J_m'(x)] cos(m theta)``; ``m = 0`` includes the constant mode at 0.
    ``0 < ratio < 1``; ``n_roots`` is at least 1.

    The cross product has no root below ``x = m`` (the Rayleigh quotient of a mode of order ``m`` on radii at most 1
    is at least ``m^2``) and its roots are separated by about ``pi / (1 - ratio)`` (the wall-to-wall wavenumber), so
    it is bracketed from ``m`` on a grid of a hundredth of that spacing and each sign change refined by Brent's
    method."""
    special = need("scipy.special")
    brentq = need("scipy.optimize").brentq
    if n_roots < 1:
        raise ValueError(f"n_roots must be at least 1, not {n_roots}")
    rho = float(ratio)
    if not 0.0 < rho < 1.0:
        raise ValueError(f"an annulus's ratio of inner to outer radius lies strictly between 0 and 1, not {ratio!r}")

    def cross(x):
        return (special.jvp(m, x * rho) * special.yvp(m, x) - special.jvp(m, x) * special.yvp(m, x * rho))

    roots = [0.0] if m == 0 else []
    h = np.pi / (1.0 - rho) / 100.0
    lo = max(float(m), 1e-6)
    while len(roots) < n_roots:
        xs = lo + h * np.arange(1, 1001)
        f = cross(xs)
        if not np.all(np.isfinite(f)):
            raise ValueError(f"the annulus's cross product overflows at order {m} and ratio {rho}: the inner wall is "
                             f"too close to the axis for the Bessel functions of the second kind")
        f0 = cross(lo)
        prev = np.concatenate([[f0], f[:-1]])
        for i in np.flatnonzero(np.sign(prev) * np.sign(f) < 0):
            a = lo if i == 0 else xs[i - 1]
            roots.append(brentq(cross, a, xs[i], xtol=1e-14, rtol=1e-15))
            if len(roots) >= n_roots:
                break
        lo = xs[-1]
    return np.asarray(roots[:n_roots])


def _plane_modes_unit(n_max, nr):
    """The unit slab ``x in [0, 1]`` with reflecting walls, gradient along its normal: ``cos(n pi x)`` with
    eigenvalue ``(n pi)^2``, and ``B`` the position operator centred on the slab's midplane -- the same
    convention the cylinder/sphere get for free from their radial coordinate running outward from the pore's
    centre, so ``B[0, 0] == 0`` and a non-refocused waveform carries no spurious global phase."""
    x = np.linspace(0.0, 1.0, nr)
    xc = x - 0.5
    U = [np.cos(n * np.pi * x) for n in range(n_max + 1)]
    lam = np.asarray([(n * np.pi) ** 2 for n in range(n_max + 1)])
    norm = [np.trapezoid(u * u, x) for u in U]
    N = len(U)
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            B[i, j] = np.trapezoid(U[i] * xc * U[j], x) / np.sqrt(norm[i] * norm[j])
    return lam, B


def _cyl_modes_unit(m_max, n_max, nr, nt):
    """The unit disk ``r in [0, 1]``, gradient across it: states ``(m, n)`` with radial ``J_m(alpha r)`` and
    angular ``cos(m theta)``, orthonormal on ``r dr dtheta``, and ``B`` the operator ``x = r cos theta``."""
    special = need("scipy.special")
    r = np.linspace(0.0, 1.0, nr)
    states, lam, rnorm, rfun = [], [], [], []
    for m in range(m_max + 1):
        for n, al in enumerate(cylinder_neumann_roots(m, n_max)):
            R = special.jv(m, al * r)
            states.append((m, n)); lam.append(al ** 2)
            rnorm.append(np.trapezoid(R * R * r, r)); rfun.append(R)
    t = np.linspace(0.0, 2 * np.pi, nt)
    ang_norm = [2 * np.pi if m == 0 else np.pi for m in range(m_max + 2)]
    N = len(states)
    B = np.zeros((N, N))
    for i in range(N):
        mi, _ = states[i]
        for j in range(N):
            mj, _ = states[j]
            if abs(mi - mj) != 1:                       # cos t couples only neighbouring angular orders
                continue
            rad = np.trapezoid(rfun[i] * rfun[j] * r * r, r)
            ang = np.trapezoid(np.cos(mi * t) * np.cos(t) * np.cos(mj * t), t)
            B[i, j] = rad * ang / np.sqrt(rnorm[i] * ang_norm[mi] * rnorm[j] * ang_norm[mj])
    return np.asarray(lam), B


#: Gauss-Legendre nodes across the annulus's wall-to-wall interval: its radial integrands are smooth and oscillate at
#: most ``n_max`` times across it, so the rule is exact to rounding (checked against twice the nodes in the tests).
ANNULUS_NODES = 512


@lru_cache(maxsize=None)
def _annulus_modes_unit(ratio, m_max, n_max, nodes):
    """The annulus ``ratio <= r <= 1``, gradient across it: states ``(m, n)`` with radial
    ``J_m(x r) Y_m'(x) - Y_m(x r) J_m'(x)`` (the root ``x`` of :func:`annulus_neumann_roots`; the constant for the
    ground mode) and angular ``cos(m theta)``, orthonormal on ``r dr dtheta``, and ``B`` the operator
    ``x = r cos theta``. ``(lambda_unit, B_unit, beta_unit, U_unit)`` as :func:`_unit_modes`, cached per ratio."""
    special = need("scipy.special")
    t, wq = np.polynomial.legendre.leggauss(int(nodes))
    r = ratio + 0.5 * (1.0 - ratio) * (t + 1.0)
    wr = 0.5 * (1.0 - ratio) * wq
    states, lam, rfun = [], [], []
    for m in range(m_max + 1):
        for n, x in enumerate(annulus_neumann_roots(m, ratio, n_max)):
            if x == 0.0:
                R = np.ones_like(r)
            else:
                R = special.jv(m, x * r) * special.yvp(m, x) - special.yv(m, x * r) * special.jvp(m, x)
            R = R / np.sqrt(np.sum(wr * R * R * r))
            states.append(m); lam.append(x * x); rfun.append(R)
    N = len(states)
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            mi, mj = states[i], states[j]
            if abs(mi - mj) != 1:                       # cos t couples only neighbouring angular orders
                continue
            rad = np.sum(wr * rfun[i] * rfun[j] * r * r)
            # int cos(m t) cos(t) cos((m+1) t) dt = pi/2 (pi when m = 0), over the angular norms 2 pi (m = 0) and pi
            ang = np.pi * (1.0 if min(mi, mj) == 0 else 0.5)
            B[i, j] = rad * ang / np.sqrt((2 * np.pi if mi == 0 else np.pi) * (2 * np.pi if mj == 0 else np.pi))
    lam = np.asarray(lam)
    beta, U = np.linalg.eigh(B)
    return lam, B, beta, U


def _sph_modes_unit(l_max, n_max, nr, nt):
    """The unit ball ``r in [0, 1]``: states ``(l, n)`` with radial ``j_l(beta r)`` and angular ``P_l(cos th)``,
    orthonormal on ``r^2 dr sin th dth dphi``, and ``B`` the operator ``z = r cos th``."""
    special = need("scipy.special")
    r = np.linspace(0.0, 1.0, nr)
    states, lam, rnorm, rfun = [], [], [], []
    for l in range(l_max + 1):
        for n, be in enumerate(sphere_neumann_roots(l, n_max)):
            R = special.spherical_jn(l, be * r)
            states.append((l, n)); lam.append(be ** 2)
            rnorm.append(np.trapezoid(R * R * r * r, r)); rfun.append(R)
    th = np.linspace(0.0, np.pi, nt)
    ct, st = np.cos(th), np.sin(th)
    P = {l: special.eval_legendre(l, ct) for l in range(l_max + 2)}
    ang_norm = {l: 2 * np.pi * np.trapezoid(P[l] * P[l] * st, th) for l in range(l_max + 2)}
    N = len(states)
    B = np.zeros((N, N))
    for i in range(N):
        li, _ = states[i]
        for j in range(N):
            lj, _ = states[j]
            if abs(li - lj) != 1:
                continue
            rad = np.trapezoid(rfun[i] * rfun[j] * r * r * r, r)
            ang = 2 * np.pi * np.trapezoid(P[li] * ct * P[lj] * st, th)
            B[i, j] = rad * ang / np.sqrt(rnorm[i] * ang_norm[li] * rnorm[j] * ang_norm[lj])
    return np.asarray(lam), B


@lru_cache(maxsize=None)
def _unit_modes(shape, n_modes, radial_nodes, angular_nodes):
    """``(lambda_unit, B_unit, beta_unit, U_unit)`` on the UNIT geometry (radius/thickness 1), cached per
    ``(shape, n_modes, radial_nodes, angular_nodes)`` -- computed once however many pore SIZES are evaluated,
    since a pore of size ``a`` scales as ``lambda = lambda_unit / a**2``, ``B = a * B_unit`` (eigenvectors of
    ``B`` are unchanged by the positive scalar ``a``, so ``beta = a * beta_unit``, ``U = U_unit``)."""
    if shape == "plane":
        if not isinstance(n_modes, (int, np.integer)):
            raise ValueError(f"a plane has one mode index: n_modes is the count of cosine modes (an int), not {n_modes!r}")
        lam, B = _plane_modes_unit(int(n_modes), radial_nodes)
    elif shape in ("cylinder", "sphere"):
        if isinstance(n_modes, (int, np.integer)) or len(n_modes) != 2:
            raise ValueError(f"a {shape} has an angular and a radial mode index: n_modes is the pair (angular, radial), not {n_modes!r}")
        m_max, n_max = (int(x) for x in n_modes)
        if shape == "cylinder":
            lam, B = _cyl_modes_unit(m_max, n_max, radial_nodes, angular_nodes)
        else:
            lam, B = _sph_modes_unit(m_max, n_max, radial_nodes, angular_nodes)
    else:
        raise ValueError(f"a closed form is known for a plane, a cylinder and a sphere, not {shape!r}")
    beta, U = np.linalg.eigh(B)
    return lam, B, beta, U


def _runs(g, breaks=()):
    """A per-step amplitude array as ``[(amplitude, n_steps)]``: what makes a piecewise-constant waveform cost
    a handful of matrix exponentials rather than one per step. ``g`` is ``(n_t,)``, an amplitude being a float, or
    ``(n_t, n_axes)``, an amplitude being the tuple of its axes; a run also ends before each step index of
    ``breaks``."""
    g = np.asarray(g)
    rows = g if g.ndim == 2 else g[:, None]
    starts = np.flatnonzero(np.any(rows[1:] != rows[:-1], axis=1)) + 1
    starts = np.union1d(starts, [int(k) for k in breaks if 0 < int(k) < len(g)]).astype(int)
    bounds = np.concatenate([[0], starts, [len(g)]])
    first = [tuple(float(v) for v in rows[a]) for a in bounds[:-1]]
    return [(amp if g.ndim == 2 else amp[0], int(b - a)) for amp, a, b in zip(first, bounds[:-1], bounds[1:])]


def _refocus_steps(refocus, n_t):
    """The step indices of the ideal 180s as a set, each ``0 <= k <= n_t`` (a 180 acts before step ``k``; ``n_t``
    is after the last step), strictly increasing."""
    steps = [int(k) for k in refocus]
    if any(k != float(r) for k, r in zip(steps, refocus)):
        raise ValueError(f"refocus holds step indices (integers), not {tuple(refocus)!r}")
    if any(k < 0 or k > n_t for k in steps) or any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError(f"refocus holds strictly increasing step indices in [0, {n_t}], not {tuple(refocus)!r}")
    return set(steps)


def _generator(D, lam, B, amp):
    """``D Lambda + i gamma (g . B)`` of a run of the amplitude ``amp`` (one entry per axis of the stack ``B``)."""
    H = D * np.diag(lam) + 1j * GAMMA * amp[0] * B[0]
    for a, B_axis in zip(amp[1:], B[1:]):
        H = H + 1j * GAMMA * a * B_axis
    return H


def _echo(lam, D, B, g, dt, *, refocus=(), max_exact_runs=None, eig=None):
    """The echo of a pore's mode coefficients under a piecewise-constant gradient: the ground (uniform) mode's
    coefficient after the ordered propagation from it.

    ``lam`` (N,) are the eigenvalues (1/m^2), ``B`` (n_axes, N, N) the real symmetric position matrices (m) of the
    gradient's axes, ``g`` (n_t, n_axes) the amplitudes (T/m) per step of ``dt`` (s). A run of constant amplitude
    advances the coefficients by ``exp(-tau (D Lambda + i gamma g . B))``, one matrix exponential per distinct run
    (a zero run is the diagonal decay). An ideal 180 before step ``k`` (``k`` in ``refocus``) conjugates the
    magnetisation, ``c -> conj(c)`` in a real eigenbasis; since the start is real, the conjugation passes onto the
    product accumulated so far. Beyond ``max_exact_runs`` runs (``None``: no ceiling) a single axis goes through
    the Strang split with the eigendecomposition ``eig = (beta, U)`` of ``B[0]``.
    """
    g = np.asarray(g, np.float64)
    flips = _refocus_steps(refocus, len(g))
    runs = _runs(g, flips)
    if max_exact_runs is not None and len(runs) > max_exact_runs:
        beta, U = eig
        return _strang(lam, D, beta, U, g[:, 0], float(dt), flips)
    expm = need("scipy.linalg").expm
    M = np.eye(len(lam), dtype=complex)
    cache = {}
    k = 0
    for amp, ns in runs:
        if k in flips:
            M = M.conj()
        k += ns
        tau = ns * float(dt)
        if not any(amp):
            M = np.exp(-tau * D * lam)[:, None] * M
            continue
        key = (tuple(round(a, 18) for a in amp), ns)
        if key not in cache:
            cache[key] = expm(-tau * _generator(D, lam, B, amp))
        M = cache[key] @ M
    if k in flips:
        M = M.conj()
    return complex(M[0, 0])


def _strang(lam, D, beta, U, g, dt, flips):
    """The Strang split of :func:`_echo` along one axis: half a step of decay, the precession in the eigenbasis
    ``B = U diag(beta) U^T``, half a step of decay, per step; the 180s of ``flips`` conjugate the coefficients."""
    half = np.exp(-0.5 * dt * D * lam)
    c = np.zeros(len(lam), complex)
    c[0] = 1.0
    for k, gk in enumerate(g):
        if k in flips:
            c = c.conj()
        c = half * c
        if gk != 0.0:
            c = U @ (np.exp(-1j * GAMMA * gk * dt * beta) * (U.T @ c))
        c = half * c
    if len(g) in flips:
        c = c.conj()
    return complex(c[0])


#: the series truncation when none is given: the cosine count for the plane, the (angular, radial) pair for the
#: round pores
DEFAULT_MODES = {"plane": 24, "cylinder": (24, 18), "sphere": (24, 18)}


class MatrixPore:
    """The closed form of one pore: ``MatrixPore(shape, size_m, D).signal(g, dt)``.

    ``shape`` is ``"plane"``, ``"cylinder"`` or ``"sphere"``; ``size_m`` the plate separation (plane) or the
    diameter (cylinder, sphere); ``n_modes`` the truncation of the series -- the count of cosine modes (an int)
    for the plane, whose eigenfunctions carry one index, and the ``(angular, radial)`` pair for the cylinder and
    the sphere; :data:`DEFAULT_MODES` when not given. It is the only approximation besides the quadrature
    resolution, refined to measure the reference's own uncertainty.
    """

    def __init__(self, shape, size_m, D, *, n_modes=None):
        self.shape, self.D, self.size = str(shape), float(D), float(size_m)
        if n_modes is None:
            n_modes = DEFAULT_MODES.get(self.shape)
        key = int(n_modes) if isinstance(n_modes, (int, np.integer)) else tuple(int(x) for x in n_modes)
        lam_unit, B_unit, beta_unit, U_unit = _unit_modes(self.shape, key, RADIAL_NODES, ANGULAR_NODES)
        a = self.size if self.shape == "plane" else 0.5 * self.size
        self.lam = lam_unit / a ** 2
        self.B = a * B_unit
        self._beta = a * beta_unit
        self._U = U_unit

    def signal(self, g_axis, dt, *, refocus=()):
        """``S`` for the per-step gradient amplitudes ``g_axis`` (T/m along the measured axis) at step ``dt``.

        Exact for a piecewise-constant waveform: one matrix exponential per run of constant amplitude, a
        zero-amplitude run being pure mode decay and therefore diagonal -- so the cost is ``len(_runs(g))``,
        not the number of distinct amplitudes, and a slew-limited waveform (many amplitudes, few runs) stays
        on this path. A genuinely dense waveform (an OGSE cosine, amplitude changing almost every sample) goes
        through the Strang split instead, whose ``O(dt^2)`` error the caller bounds by refining ``dt``.

        ``g_axis`` is the effective waveform (the sign of each lobe after a refocusing pulse folded in), or the
        waveform as played with ``refocus`` the step indices before which an ideal 180 conjugates the
        magnetisation; the two give complex-conjugate echoes after an odd number of 180s, the same magnitude.
        """
        g = np.asarray(g_axis, np.float64)
        return _echo(self.lam, self.D, self.B[None], g[:, None], dt, refocus=refocus,
                     max_exact_runs=_MAX_EXACT_RUNS, eig=(self._beta, self._U))


#: the series truncation of the annulus when none is given, the (angular, radial) pair: its radial modes decay at
#: ``(n pi / (1 - ratio))^2`` against the angular ``m^2``, so few are needed beside the angular orders
ANNULUS_DEFAULT_MODES = (24, 8)


class AnnulusPore(MatrixPore):
    """The closed form of the annulus between two coaxial circles, the wall of a hollow cylinder filled with water:
    ``AnnulusPore(outer_diameter_m, inner_diameter_m, D).signal(g, dt)``.

    The pore's Laplacian with reflecting walls on both circles has the modes ``R_mn(r) cos(m theta)``, ``R_mn`` the
    cross product of :func:`annulus_neumann_roots` (the Bessel functions of both kinds, since the axis is not in the
    pore), and the gradient along one fixed direction across the axis couples the angular orders ``m`` and
    ``m +/- 1`` through ``x = r cos theta``; the propagation is :class:`MatrixPore`'s (its :meth:`signal`). The disk
    is the limit of a vanishing inner diameter, the circle of :class:`PeriodicPore` (water on one cylindrical
    surface) the limit of a vanishing thickness.

    ``n_modes`` is the ``(angular, radial)`` truncation, :data:`ANNULUS_DEFAULT_MODES` when not given: the only
    approximation besides the quadrature (:data:`ANNULUS_NODES`). The unit modes are computed once per ratio of the
    diameters and order, and a pore of outer radius ``a`` reads ``lambda = lambda_unit / a^2``, ``B = a B_unit``.
    Assumes impermeable walls without surface relaxivity, a uniform start, one free diffusivity and a gradient along a
    fixed direction perpendicular to the axis, given as its amplitude per step. Source: the eigenfunctions of the
    annulus (Carslaw HS, Jaeger JC (1959), Conduction of Heat in Solids, 2nd ed., Sec. 13.5) in the matrix formalism
    of Callaghan PT (1997), J. Magn. Reson. 129:74-84.
    """

    def __init__(self, outer_diameter_m, inner_diameter_m, D, *, n_modes=None):
        outer, inner = float(outer_diameter_m), float(inner_diameter_m)
        if not 0.0 < inner < outer:
            raise ValueError(f"an annulus has 0 < inner diameter < outer diameter, not inner={inner_diameter_m!r}, "
                             f"outer={outer_diameter_m!r}")
        m_max, n_max = (int(x) for x in (ANNULUS_DEFAULT_MODES if n_modes is None else n_modes))
        self.shape, self.D, self.size, self.inner = "annulus", float(D), outer, inner
        lam_unit, B_unit, beta_unit, U_unit = _annulus_modes_unit(inner / outer, m_max, n_max, ANNULUS_NODES)
        a = 0.5 * outer
        self.lam = lam_unit / a ** 2
        self.B = a * B_unit
        self._beta = a * beta_unit
        self._U = U_unit


def _position_matrix(f_hat, n_modes):
    """The matrix of multiplication by a real periodic coordinate ``f(s)`` in the real Fourier basis of
    :class:`PeriodicPore`, from its Fourier coefficients ``f_hat[n] = (1/P) int f(s) exp(-2 pi i n s/P) ds``,
    ``n = 0 .. 2 n_modes``.

    With ``a_n = Re f_hat[n]``, ``b_n = -Im f_hat[n]`` (``b_{-n} = -b_n``) and the basis ``1, sqrt2 cos(k alpha),
    sqrt2 sin(k alpha)``, ``alpha = 2 pi s/P``, the entries are ``<1|f|1> = a_0``, ``<1|f|c_k> = sqrt2 a_k``,
    ``<1|f|s_k> = sqrt2 b_k``, ``<c_j|f|c_k> = a_|j-k| + a_{j+k}``, ``<s_j|f|s_k> = a_|j-k| - a_{j+k}`` and
    ``<c_j|f|s_k> = b_{k+j} + b_{k-j}``: a Toeplitz-plus-Hankel matrix of the coefficients."""
    M = int(n_modes)
    a, b = f_hat.real, -f_hat.imag
    B = np.zeros((2 * M + 1, 2 * M + 1))
    B[0, 0] = a[0]
    if M == 0:
        return B
    k = np.arange(1, M + 1)
    cos, sin = 2 * k - 1, 2 * k
    B[0, cos] = B[cos, 0] = np.sqrt(2.0) * a[k]
    B[0, sin] = B[sin, 0] = np.sqrt(2.0) * b[k]
    j, kk = np.meshgrid(k, k, indexing="ij")
    B[np.ix_(cos, cos)] = a[np.abs(j - kk)] + a[j + kk]
    B[np.ix_(sin, sin)] = a[np.abs(j - kk)] - a[j + kk]
    cross = b[j + kk] + np.sign(kk - j) * b[np.abs(kk - j)]
    B[np.ix_(cos, sin)] = cross
    B[np.ix_(sin, cos)] = cross.T
    return B


def polygon_fourier(outline_m, n_max):
    """The closed polygon through the points ``outline_m`` (n, 2), walked at unit speed: its perimeter ``P`` (m) and
    the Fourier coefficients in arc length of its two coordinates, shape (2, n_max + 1),
    ``C_hat[:, n] = (1/P) int_0^P C(s) exp(-2 pi i n s/P) ds``.

    The points are the vertices in order around the outline; the closing edge from the last point back to the
    first is implied. The coordinates are piecewise linear in ``s``, so their coefficients are exact: with the
    slope ``m_j`` of the edge leaving vertex ``j`` (at arc length ``s_j``) and ``omega = 2 pi n/P``, integrating
    by parts twice gives ``C_hat[n] = -(1/(P omega^2)) sum_j (m_j - m_{j-1}) exp(-i omega s_j)`` for ``n >= 1``,
    and the mean ``C_hat[0]`` is the edges' midpoints weighted by their lengths. A polygon's coefficients fall as
    ``1/n^2``, a corner's turn being a jump of the slope."""
    V = np.asarray(outline_m, np.float64)
    if V.ndim != 2 or V.shape[1] != 2 or len(V) < 3 or not np.all(np.isfinite(V)):
        raise ValueError(f"an outline is three or more finite points (n, 2) in metres, not an array of shape {V.shape}")
    edge = np.roll(V, -1, axis=0) - V
    length = np.hypot(edge[:, 0], edge[:, 1])
    if np.any(length == 0.0):
        raise ValueError("the outline's consecutive points are distinct (the closing edge back to the first point "
                         "is implied: do not repeat it)")
    P = float(length.sum())
    s = np.concatenate([[0.0], np.cumsum(length)[:-1]])
    slope = edge / length[:, None]
    kink = slope - np.roll(slope, 1, axis=0)
    C_hat = np.zeros((2, int(n_max) + 1), complex)
    C_hat[:, 0] = (length[:, None] * (V + 0.5 * edge)).sum(axis=0) / P
    if n_max >= 1:
        omega = 2.0 * np.pi * np.arange(1, int(n_max) + 1) / P
        C_hat[:, 1:] = -(np.exp(-1j * np.outer(omega, s)) @ kink).T / (P * omega ** 2)
    return P, C_hat


class PeriodicPore:
    """The closed curve walked along its arc length, the periodic pore of the matrix method:
    ``PeriodicPore.circle(diameter_m, D, n_modes=M)`` or ``PeriodicPore.polygon(outline_m, D, n_modes=M)``, then
    ``.signal(g, dt)``.

    A walker on a closed curve ``C(s) = (x(s), y(s))`` of perimeter ``P``, ``s`` its arc length, diffuses along
    ``s`` with the diffusivity ``D``: Brownian motion on the circle of length ``P``, whatever the curve's shape. The
    Laplacian ``-d^2/ds^2`` has the real Fourier modes ``1, sqrt2 cos(2 pi k s/P), sqrt2 sin(2 pi k s/P)``
    (``k = 1 .. M``, interleaved, orthonormal in the mean over the curve) with the eigenvalues ``(2 pi k/P)^2``
    (``lam``), and a gradient ``(g_x, g_y)`` in the curve's plane couples them through the position, the matrices
    ``B = (B_x, B_y)`` of multiplication by ``x(s)`` and ``y(s)``: the Fourier coefficients of the coordinates
    (:func:`_position_matrix`). The circle of radius ``R`` has ``x_hat[1] = R/2``, ``y_hat[1] = -i R/2`` and no
    other coefficient, so ``B`` couples the order ``k`` to ``k +/- 1`` by ``R/2`` (Canales-Rodriguez et al. 2026,
    Phys. Med. Biol. 71:215006, Eq. 17, in the complex basis ``e^{i k theta}``); a polygon's are exact
    (:func:`polygon_fourier`).

    The order ``M`` (``n_modes``, ``2 M + 1`` modes) is the only approximation; the coordinates are the curve's as
    given (the circle is centred on the origin). The matrices are small, so :meth:`signal` is the exact product for
    every piecewise-constant waveform, without a Strang split.
    """

    def __init__(self, perimeter_m, C_hat, D, *, n_modes):
        M = int(n_modes)
        C_hat = np.asarray(C_hat, complex)
        if M < 0 or C_hat.ndim != 2 or C_hat.shape[0] != 2 or C_hat.shape[1] < 2 * M + 1:
            raise ValueError(f"a periodic pore of order {n_modes} needs the coefficients (2, >= {2 * M + 1}) of its "
                             f"two coordinates, not an array of shape {C_hat.shape}")
        self.perimeter, self.D, self.n_modes = float(perimeter_m), float(D), M
        order = np.concatenate([[0], np.repeat(np.arange(1, M + 1), 2)])
        self.lam = (2.0 * np.pi * order / self.perimeter) ** 2
        self.B = np.stack([_position_matrix(C_hat[0, :2 * M + 1], M), _position_matrix(C_hat[1, :2 * M + 1], M)])

    @classmethod
    def circle(cls, diameter_m, D, *, n_modes):
        """The circle of diameter ``diameter_m`` (m) centred on the origin."""
        R = 0.5 * float(diameter_m)
        C_hat = np.zeros((2, 2 * int(n_modes) + 1), complex)
        if n_modes >= 1:
            C_hat[0, 1], C_hat[1, 1] = 0.5 * R, -0.5j * R
        return cls(np.pi * float(diameter_m), C_hat, D, n_modes=n_modes)

    @classmethod
    def polygon(cls, outline_m, D, *, n_modes):
        """The closed polygon through the points ``outline_m`` (n, 2) in metres (:func:`polygon_fourier`)."""
        P, C_hat = polygon_fourier(outline_m, 2 * int(n_modes))
        return cls(P, C_hat, D, n_modes=n_modes)

    def signal(self, g, dt, *, refocus=()):
        """The complex echo for the per-step gradient ``g`` (n_t, 2) in the curve's plane (T/m) at step ``dt``
        (s): the ordered product of one matrix exponential per run of constant gradient, exact for a
        piecewise-constant waveform. ``g`` is the effective waveform, or the waveform as played with ``refocus``
        the step indices before which an ideal 180 conjugates the magnetisation (see :meth:`MatrixPore.signal`)."""
        g = np.asarray(g, np.float64)
        if g.ndim != 2 or g.shape[1] != 2:
            raise ValueError(f"the gradient in the curve's plane is (n_t, 2), not an array of shape {g.shape}")
        return _echo(self.lam, self.D, self.B, g, dt, refocus=refocus)


def matrix_restricted_signal(shape, g_axis, dt, D, size_m, *, n_modes=None):
    """The restricted signal's magnitude for a single projected 1-D gradient schedule (T/m) at step ``dt``
    (s), diffusivity ``D`` (m^2/s) and pore ``size_m`` -- the functional entry point for a caller that wants a
    number rather than :class:`MatrixPore`'s object API."""
    return float(abs(MatrixPore(shape, size_m, D, n_modes=n_modes).signal(g_axis, dt)))


def matrix_restricted_batch(shape, g_axes, dt, D, size_m, *, n_modes=None):
    """``matrix_restricted_signal`` over a stack of projected 1-D schedules ``g_axes`` (n_meas, n_t) sharing
    one ``dt``, ``D`` and ``size_m`` -- one :class:`MatrixPore` (one unit-mode cache hit, one eigendecomposition
    scaled once) evaluated per measurement rather than rebuilt per call."""
    pore = MatrixPore(shape, size_m, D, n_modes=n_modes)
    return np.array([float(abs(pore.signal(g, dt))) for g in g_axes])
