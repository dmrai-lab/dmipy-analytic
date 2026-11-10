"""The Bloch-Torrey equation with radial symmetry on a permeable pore in a finite confining shell, solved by finite
volumes to a stated accuracy: the semi-exact reference of an isolated permeable sphere, cylinder (uniform along its
axis) or slab (the planar twin that cross-checks the method).

**The problem.** The pore ``r < R`` (the intra pool, diffusivity ``D_in``) lies inside the shell ``R < r < R_out`` (the
extra pool, ``D_out``), with a reflecting wall at ``R_out``; ``r`` is the distance from the centre (sphere), from the
axis (cylinder), or ``|x|`` from the mid-plane (slab). The transverse magnetisation density ``m`` obeys

    dm/dt = D(r) lap m - i gamma g(t) x m,

with the gradient ``g(t)`` along the fixed direction ``x`` (any direction for the sphere, perpendicular to the axis for
the cylinder, the normal for the slab); an ideal 180-degree pulse conjugates ``m``. At ``r = R`` the membrane of
permeability ``kappa`` (m/s) gives the exchange condition

    D_in m'(R-) = D_out m'(R+) = kappa (m(R-) - m(R+)),

the flux ``J = -D m'`` (density times m/s) equal to ``kappa`` times the density jump: the Tanner (1978) membrane
condition, the one a random walk reproduces when a walker meeting the membrane crosses with probability ``x/(1+x)``,
``x = 2 kappa d / D`` for a normal penetration ``d`` (Fieremans et al. 2010, NMR Biomed. 23:711, Eq. 43). ``kappa = 0``
seals the pore, ``kappa = inf`` removes the membrane (the density is continuous). The spin density is uniform, so the
density is continuous at equilibrium for any ``kappa``, also when ``D_in`` differs from ``D_out``.

**The angular expansion.** The magnetisation is expanded in the harmonics of the gradient's direction, ``m = sum_k
u_k(r, t) Y_k``: Legendre polynomials ``P_l(cos theta)`` for the sphere, ``cos(k phi)`` for the cylinder, the even and
odd parts for the slab. Each harmonic obeys the radial diffusion equation with the centrifugal term ``l (l + 1) / r^2``
(sphere) or ``k^2 / r^2`` (cylinder), and the gradient couples the neighbours: ``x Y_k`` is a combination of ``Y_{k-1}``
and ``Y_{k+1}`` (``cos theta P_l = [(l + 1) P_{l+1} + l P_{l-1}] / (2 l + 1)``, ``cos phi cos k phi = [cos (k + 1) phi +
cos (k - 1) phi] / 2``), through a real tridiagonal matrix ``T`` times ``r``. The expansion is exact for any gradient
strength when all harmonics are kept; the ``n_harmonics`` retained (the uniform start is ``k = 0``, and a harmonic
``k`` is reached at order ``k`` in the gradient) is an argument, and the truncation is an error the caller refines.
The signal is the volume integral of ``u_0``.

**The method.** Cell-centred finite volumes on uniform cells in each pool (``n_in`` and ``n_out`` of them), the face
areas ``r^(d-1)`` (``d = 1, 2, 3`` for the slab, cylinder, sphere) and the cell volumes ``(r_+^d - r_-^d) / d``. The
conductance of a face between two cells is that of the two half-cells in series, and at ``R`` the permeability adds its
own resistance ``1 / kappa`` in series. The centrifugal term acts on the cell as ``c_k D (r_i)^(d-3) dr`` (the exact
cell integral for the sphere). The operator of each harmonic is symmetric in the cell volumes, so the diffusion over
a step is applied exactly in the eigenbasis of each harmonic (``numpy.linalg.eigh``); the gradient is applied
exactly, cell by cell, in the eigenbasis of ``T`` (its eigenvalues are the Gauss nodes of the truncation); a step is the
Strang split diffusion/2, gradient, diffusion/2. The ``k = 0`` operator also gives the exchange of the pools without
any time stepping (its eigenvalues).

**Its accuracy.** Second order in the cell width and in the step (a waveform sample is divided into ``n_sub`` equal
steps): refining either by two leaves a quarter of the error. :func:`reference` refines both together, checks the
observed order on three resolutions, and returns the Richardson error estimate; the harmonic truncation adds its own
estimate. The tests give the refinement tables.

**Conventions.** SI units; ``g`` in T/m per waveform sample of ``dt`` s; ``refocus`` the sample indices before which an
ideal 180 acts (``len(g)`` is after the last sample); signals are complex, normalised to the magnetisation at the
excitation. Only numpy is needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import GAMMA

__all__ = ["ShellSpec", "PermeableShell", "ShellSignal", "Reference", "reference", "pgse_waveform"]

_DIM = {"slab": 1, "cylinder": 2, "sphere": 3}
_STARTS = ("uniform", "intra", "extra")


def pgse_waveform(G, delta, Delta, dt):
    """A pulsed gradient pair as played: two lobes of amplitude ``G`` (T/m) and duration ``delta`` (s) whose onsets are
    ``Delta`` (s) apart, an ideal 180 midway between them, sampled on ``dt`` (s).

    Returns ``(g, refocus)``: the amplitudes per sample (the first lobe is samples ``[0, delta/dt)``, the second
    ``[Delta/dt, Delta/dt + delta/dt)``) and the one-element tuple of the sample before which the 180 acts, for
    :meth:`PermeableShell.signal`. ``delta`` and ``Delta`` must be whole multiples of ``dt``, ``Delta >= delta``, and
    ``(delta + Delta) / dt`` even so that the 180 falls on a sample boundary; a ValueError names the one that is not.
    """
    nd, nD = delta / dt, Delta / dt
    for name, n in (("delta", nd), ("Delta", nD)):
        if abs(n - round(n)) > 1e-9 * max(1.0, abs(n)):
            raise ValueError(f"{name} is not a whole multiple of dt: {n} steps")
    nd, nD = int(round(nd)), int(round(nD))
    if nD < nd:
        raise ValueError(f"Delta ({nD} steps) is shorter than delta ({nd} steps): the lobes overlap")
    if (nd + nD) % 2:
        raise ValueError(f"delta + Delta is an odd number of steps ({nd + nD}): the 180 would fall between samples")
    g = np.zeros(nD + nd)
    g[:nd] = G
    g[nD:nD + nd] = G
    return g, ((nd + nD) // 2,)


@dataclass(frozen=True)
class ShellSpec:
    """The geometry and transport of one permeable pore in its shell: every field is required.

    ``geometry`` is ``"sphere"``, ``"cylinder"`` or ``"slab"``; ``R`` and ``R_out`` the pore's and the shell's outer
    radius (half-width for the slab), m, ``0 < R < R_out``; ``D_in``, ``D_out`` the diffusivities, m^2/s; ``kappa`` the
    permeability, m/s (``0``: sealed, ``numpy.inf``: no membrane)."""
    geometry: str
    R: float
    R_out: float
    D_in: float
    D_out: float
    kappa: float


@dataclass(frozen=True)
class ShellSignal:
    """The complex signals of the pools and of their sum, each normalised to the magnetisation at the excitation."""
    intra: complex
    extra: complex

    @property
    def total(self):
        return self.intra + self.extra


def _check_spec(spec):
    if spec.geometry not in _DIM:
        raise ValueError(f"geometry is one of {tuple(_DIM)}, not {spec.geometry!r}")
    for name in ("R", "R_out", "D_in", "D_out"):
        v = float(getattr(spec, name))
        if not (np.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} is positive and finite, not {v}")
    if not spec.R < spec.R_out:
        raise ValueError(f"R ({spec.R}) must be smaller than R_out ({spec.R_out})")
    k = float(spec.kappa)
    if math.isnan(k) or k < 0.0:
        raise ValueError(f"kappa is non-negative (numpy.inf for no membrane), not {k}")


def _tridiagonal(geometry, n_harmonics):
    """``(c, T, parity_odd)`` of a harmonic set: ``c[k]`` the centrifugal coefficient, ``T`` the symmetric coupling
    of ``x / r`` between the normalised harmonics (the matrix of the cell-wise precession is ``-i gamma g r T``), and,
    for the slab, which harmonics vanish at the mid-plane."""
    L = int(n_harmonics)
    k = np.arange(L)
    if geometry == "sphere":
        c = k * (k + 1.0)
        up = (k[:-1] + 1.0) / np.sqrt((2 * k[:-1] + 1.0) * (2 * k[:-1] + 3.0))
    elif geometry == "cylinder":
        c = k.astype(float) ** 2
        up = np.full(L - 1, 0.5)
        if L > 1:
            up[0] = 0.5 * math.sqrt(2.0)
    else:
        if L != 2:
            raise ValueError(f"the slab has two harmonics (even, odd): n_harmonics is 2, not {L}")
        c = np.zeros(2)
        up = np.ones(1)
    T = np.diag(up, 1) + np.diag(up, -1)
    return c, T, (k % 2 == 1) if geometry == "slab" else np.zeros(L, bool)


class PermeableShell:
    """A permeable pore in a finite shell on a finite-volume grid:
    ``PermeableShell(spec, n_in=, n_out=, n_harmonics=).signal(g, dt, n_sub=, start=)``.

    ``spec`` is a :class:`ShellSpec`; ``n_in`` and ``n_out`` the numbers of cells of the pore and of the shell
    (uniform within each); ``n_harmonics`` the harmonics kept (2 for the slab). Everything is required. The object
    holds the eigendecompositions of its operators (``n_harmonics`` dense ``(n_in + n_out)^2`` matrices), so a
    resolution is built once and a waveform, a start and a step count are arguments of the call.

    Methods: :meth:`signal` (a gradient waveform), :meth:`exchange` (``P_ij(t)``, the fraction of spins that started
    uniformly in pool ``i`` found in pool ``j``), :meth:`exchange_rate`, :meth:`mean_residence_time`.
    """

    def __init__(self, spec, *, n_in, n_out, n_harmonics):
        _check_spec(spec)
        self.spec = spec
        self.n_in, self.n_out, self.n_harmonics = int(n_in), int(n_out), int(n_harmonics)
        if min(self.n_in, self.n_out) < 1 or self.n_harmonics < 1:
            raise ValueError(f"n_in, n_out and n_harmonics are positive integers, not {n_in}, {n_out}, {n_harmonics}")
        d = _DIM[spec.geometry]
        faces = np.concatenate([np.linspace(0.0, spec.R, self.n_in + 1),
                                np.linspace(spec.R, spec.R_out, self.n_out + 1)[1:]])
        lo, hi = faces[:-1], faces[1:]
        self.dr = hi - lo
        self.mid = 0.5 * (lo + hi)
        self.volume = (hi ** d - lo ** d) / d
        self.centroid = d * (hi ** (d + 1) - lo ** (d + 1)) / ((d + 1) * (hi ** d - lo ** d))
        n = self.n_in + self.n_out
        self.n = n
        self.D = np.where(np.arange(n) < self.n_in, spec.D_in, spec.D_out)
        area = faces ** (d - 1) if d > 1 else np.ones_like(faces)
        # conductances of the n-1 inner faces: half-cells in series, the permeability's resistance at the pore's face
        resist = 0.5 * self.dr[:-1] / self.D[:-1] + 0.5 * self.dr[1:] / self.D[1:]
        resist = resist.copy()
        if np.isfinite(spec.kappa):
            resist[self.n_in - 1] += (1.0 / spec.kappa) if spec.kappa > 0.0 else np.inf
        self._G = area[1:-1] / resist                                         # 0 where kappa == 0
        self._area0 = float(area[0])
        self._c, self._T, self._odd = _tridiagonal(spec.geometry, self.n_harmonics)
        self._d = d
        sv = np.sqrt(self.volume)
        self._sv = sv
        lam, U = [], []
        for k in range(self.n_harmonics):
            M = self._operator(k)
            w, V = np.linalg.eigh(M / sv[:, None] / sv[None, :])
            lam.append(w)
            U.append(V)
        self.lam = np.array(lam)                                              # (L, n)
        self._U = np.array(U)                                                 # (L, n, n)
        mu, Q = np.linalg.eigh(self._T)
        self._mu, self._Q = mu, Q

    def _operator(self, k):
        """The conductance matrix of harmonic ``k`` (``h dm/dt = -C m``), before the symmetrisation by the volumes."""
        n = self.n
        C = np.zeros((n, n))
        i = np.arange(n - 1)
        C[i, i] += self._G
        C[i + 1, i + 1] += self._G
        C[i, i + 1] -= self._G
        C[i + 1, i] -= self._G
        if self._c[k] != 0.0:
            C[np.arange(n), np.arange(n)] += self._c[k] * self.D * self.dr * self.mid ** (self._d - 3)
        if self._odd[k]:
            C[0, 0] += self.D[0] * self._area0 / (0.5 * self.dr[0])
        return C

    def _start_vector(self, start):
        if start not in _STARTS:
            raise ValueError(f"start is one of {_STARTS}, not {start!r}")
        rho = np.ones(self.n)
        if start == "intra":
            rho[self.n_in:] = 0.0
        elif start == "extra":
            rho[:self.n_in] = 0.0
        return rho * self.volume                                              # magnetisation per cell

    def _propagator(self, tau):
        """``exp(-tau A_k)`` of every harmonic as a dense ``(L, n, n)`` array (symmetric, in the volume-symmetrised
        cell variables)."""
        return np.matmul(self._U * np.exp(-tau * self.lam)[:, None, :], np.swapaxes(self._U, 1, 2))

    @staticmethod
    def _apply(E, y):
        """``E y`` per harmonic for the complex ``y`` of shape ``(L, n)``."""
        z = np.matmul(E, np.stack([y.real, y.imag], axis=-1))
        return z[..., 0] + 1j * z[..., 1]

    def signal(self, g, dt, *, n_sub, start, refocus=()):
        """The pools' signals :class:`ShellSignal` after the waveform ``g`` (T/m per sample of ``dt`` s).

        ``n_sub`` divides each sample into equal steps (the Strang split's step is ``dt / n_sub``); ``start`` is
        ``"uniform"`` (the density 1 in both pools), ``"intra"`` or ``"extra"`` (density 1 in that pool only); the
        signals are normalised to the start's total magnetisation, so ``total`` is the echo of the start and
        ``intra`` and ``extra`` are the parts of it in each pool at the end. ``refocus`` is the sample indices before
        which an ideal 180 conjugates the magnetisation."""
        g = np.asarray(g, np.float64)
        if g.ndim != 1:
            raise ValueError(f"g is a vector of amplitudes per sample, not an array of shape {g.shape}")
        n_sub = int(n_sub)
        if n_sub < 1:
            raise ValueError(f"n_sub is a positive integer, not {n_sub}")
        flips = [int(k) for k in refocus]
        if any(k < 0 or k > len(g) for k in flips) or any(b <= a for a, b in zip(flips, flips[1:])):
            raise ValueError(f"refocus holds strictly increasing sample indices in [0, {len(g)}], not {tuple(refocus)}")
        flips = set(flips)
        h = float(dt) / n_sub
        m0 = self._start_vector(start)
        y = np.zeros((self.n_harmonics, self.n), complex)
        y[0] = self._sv * m0 / self.volume                                    # sqrt(V) * density
        E_half, E_full = self._propagator(0.5 * h), self._propagator(h)
        y = self._apply(E_half, y)
        n_steps = len(g) * n_sub
        s = 0
        for k, gk in enumerate(g):
            if k in flips:
                y = y.conj()
            if gk != 0.0:
                phase = np.exp(-1j * GAMMA * gk * h * self.centroid[None, :] * self._mu[:, None])   # (L, n)
                prec = True
            else:
                prec = False
            for _ in range(n_sub):
                if prec:
                    y = self._Q @ (phase * (self._Q.T @ y))
                s += 1
                y = self._apply(E_full if s < n_steps else E_half, y)
        if len(g) in flips:
            y = y.conj()
        mass = self._sv * y[0]
        total0 = float(m0.sum())
        return ShellSignal(intra=complex(mass[:self.n_in].sum() / total0), extra=complex(mass[self.n_in:].sum() / total0))

    def _l0(self):
        return self.lam[0], self._U[0]

    def exchange(self, times):
        """``P[i, j, t]``: the fraction of the spins that started uniformly in pool ``i`` (0 intra, 1 extra) found in
        pool ``j`` at ``times`` (s), by the eigenmodes of the ``k = 0`` operator (no time step: the error is the
        cells' alone). Rows sum to 1."""
        t = np.atleast_1d(np.asarray(times, np.float64))
        if np.any(t < 0.0):
            raise ValueError("times are non-negative")
        lam, U = self._l0()
        pool = np.zeros((2, self.n))
        pool[0, :self.n_in] = 1.0
        pool[1, self.n_in:] = 1.0
        a = (pool * self._sv) @ U                                             # (2, n): <pool i | mode>
        mass = np.array([a[i] * a[j] * 1.0 for i in range(2) for j in range(2)]).reshape(2, 2, -1)
        vol = np.array([self.volume[:self.n_in].sum(), self.volume[self.n_in:].sum()])
        P = np.einsum("ijm,mt->ijt", mass, np.exp(-np.outer(lam, t))) / vol[:, None, None]
        return P

    def exchange_rate(self):
        """The rate (1/s) of the slowest relaxation of the pools' populations: the smallest non-zero eigenvalue of the
        ``k = 0`` operator, ``k_in->out + k_out->in`` of the two-site model when exchange is the slow process. It
        is the second eigenvalue (the first is the conserved total), so it is ``0`` for ``kappa = 0``."""
        return float(np.sort(self._l0()[0])[1])

    def mean_residence_time(self, pool):
        """The mean time (s) a spin that starts uniformly in ``pool`` (``"intra"`` or ``"extra"``) spends there before
        its first arrival at the other side of the membrane, i.e. the exit through the membrane into an absorbing
        neighbour: ``kappa`` as a Robin sink at the face (``kappa = inf``: an absorbing face). Infinite for
        ``kappa = 0``. The linear solve ``C T = V`` on the pool's cells."""
        if pool not in ("intra", "extra"):
            raise ValueError(f"pool is 'intra' or 'extra', not {pool!r}")
        if self.spec.kappa == 0.0:
            return math.inf
        n_in = self.n_in
        sl = slice(0, n_in) if pool == "intra" else slice(n_in, self.n)
        C = self._operator(0)[sl, sl].copy()
        j = n_in - 1 if pool == "intra" else 0
        # the face's conductance to the absorbing far side: its half-cell and the membrane in series
        k_i = n_in - 1 if pool == "intra" else n_in
        Df = self.D[k_i]
        r = 0.5 * self.dr[k_i] / Df + ((1.0 / self.spec.kappa) if np.isfinite(self.spec.kappa) else 0.0)
        # the symmetric operator has the coupling to the other pool in its diagonal; the sink replaces it
        C[j, j] -= self._G[n_in - 1]
        C[j, j] += (1.0 / r) * (self._area_R())
        T = np.linalg.solve(C, self.volume[sl])
        return float((self.volume[sl] * T).sum() / self.volume[sl].sum())

    def _area_R(self):
        return float(self.spec.R ** (self._d - 1)) if self._d > 1 else 1.0


@dataclass(frozen=True)
class Reference:
    """A reference value with its numerical error: ``intra``, ``extra`` and ``total`` the complex signals at the
    finest resolution, ``error_intra``, ``error_extra`` and ``error_total`` the estimated absolute errors of each
    (Richardson estimate of the grid and the step, plus the harmonic truncation), ``n_in``, ``n_out``, ``n_sub``,
    ``n_harmonics`` the resolution they were computed at, and ``order`` the observed order of convergence (the base-2
    logarithm of the ratio of the last two changes of the signal that changes most; ``nan`` when both changes are below
    the rounding of the steps)."""
    intra: complex
    extra: complex
    total: complex
    error_intra: float
    error_extra: float
    error_total: float
    n_in: int
    n_out: int
    n_sub: int
    n_harmonics: int
    order: float


def reference(spec, g, dt, *, refocus, start, tol, cells, sub_steps, harmonics, max_levels):
    """The finite-volume reference of a permeable pore under a waveform, with its error bar:
    ``reference(spec, g, dt, refocus=(), start="uniform", tol=1e-6, cells=(40, 40), sub_steps=1, harmonics=8,
    max_levels=5)`` (the values shown are examples: every argument is required).

    Runs :class:`PermeableShell` on the finite-volume grid ``cells = (n_in, n_out)`` with ``sub_steps`` steps per
    sample, then raises the harmonics by 2 until the change of the signals falls below ``tol`` (that change is the
    truncation's error estimate), then refines the cells and the steps together by 2 until the Richardson error
    estimate of every pool's signal is below ``tol``. The estimate of a level is ``|S_fine - S_coarse| / 3`` (second
    order) and the observed order is ``log2`` of the ratio of the last two changes: it has to lie in ``(1.5, 2.5)``
    (a ValueError says otherwise) unless the changes are below the rounding of the steps taken (``n_steps *
    eps``). At least three resolutions are run, at most ``max_levels``; failing to reach ``tol`` raises. The stated
    error of each value is the Richardson estimate of the finest resolution plus the truncation's change.

    ``start`` is ``"uniform"``, ``"intra"`` or ``"extra"`` (see :meth:`PermeableShell.signal`).
    Returns a :class:`Reference`."""
    harmonics = int(harmonics)
    step = 2 if spec.geometry != "slab" else 0
    n_in, n_out, n_sub = int(cells[0]), int(cells[1]), int(sub_steps)
    tol = float(tol)
    if tol <= 0.0:
        raise ValueError(f"tol is positive, not {tol}")

    def run(ni, no, ns, L):
        s = PermeableShell(spec, n_in=ni, n_out=no, n_harmonics=L).signal(g, dt, n_sub=ns, start=start, refocus=refocus)
        return np.array([s.intra, s.extra, s.total])

    trunc = np.zeros(3)
    L = harmonics
    if spec.geometry != "slab":
        prev = run(n_in, n_out, n_sub, L)
        while True:
            nxt = run(n_in, n_out, n_sub, L + step)
            trunc = np.abs(nxt - prev)
            L += step
            prev = nxt
            if np.all(trunc < tol):
                break
            if L > harmonics + 8 * step * max(int(max_levels), 1):
                raise RuntimeError(f"the harmonic truncation did not converge to {tol}: change {trunc} at {L} harmonics")
    levels = [run(n_in, n_out, n_sub, L)]
    ni, no, ns = n_in, n_out, n_sub
    err = None
    for lev in range(1, int(max_levels) + 1):
        ni, no, ns = 2 * ni, 2 * no, 2 * ns
        levels.append(run(ni, no, ns, L))
        if len(levels) >= 3:
            d1, d2 = np.abs(levels[-2] - levels[-3]), np.abs(levels[-1] - levels[-2])
            err = d2 / 3.0
            floor = len(g) * ns * np.finfo(float).eps
            big = np.argmax(d1)
            if d2[big] > floor:
                order = math.log2(d1[big] / d2[big])
                if not 1.5 < order < 2.5:
                    raise ValueError(f"the refinement is not second order: observed {order:.2f}; changes {d1}, {d2}")
            else:
                order = float("nan")
            if np.all(err + trunc < tol):
                break
    else:
        raise RuntimeError(f"no convergence to {tol} in {max_levels} levels: error estimate {err + trunc}")
    S = levels[-1]
    e = err + trunc
    return Reference(intra=complex(S[0]), extra=complex(S[1]), total=complex(S[2]), error_intra=float(e[0]),
                     error_extra=float(e[1]), error_total=float(e[2]), n_in=ni, n_out=no, n_sub=ns, n_harmonics=L,
                     order=order)
