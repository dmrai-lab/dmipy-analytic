"""Relaxation and the exchange of magnetization between a free and a bound pool, in closed form.

**The two-region exchange relations.** A free pool exchanges with a bound (macromolecular) pool on a reactive surface
of surface-to-volume ratio ``S/V`` and surface reactivity ``kappa_MT`` (m/s), the bound spins dwelling a mean
``dwell_time`` before release:

    k_f = kappa_MT * (S/V)          the forward (free -> bound) pseudo-first-order rate
    k_r = 1 / dwell_time            the backward (bound -> free) rate
    f_bound = k_f / (k_f + k_r)     the equilibrium bound-pool fraction

``k_f = kappa_MT (S/V)`` is the analogue of the surface-relaxation rate ``1/T2_surf = rho (S/V)`` of the fast-diffusion
(Brownstein-Tarr) regime, and it assumes that regime: the pool is well mixed on the time scale ``1/k_f``. These
relations convert the microscopic ``(kappa_MT, dwell_time)`` into the macroscopic ``(f_bound, k)`` of quantitative MT.

**The two-pool Bloch-McConnell system.** The free pool ``a`` and the bound pool ``b``, each with its own ``R1``, ``R2``
and off-resonance, exchange every magnetization component at ``k_f`` and ``k_r``, under a continuous RF of nutation
rate ``w1`` about an axis in the transverse plane. The augmented state ``x = [Mxa, Mya, Mza, Mxb, Myb, Mzb, 1]`` obeys
``dx/dt = A x`` with a 7x7 generator, so over an interval with everything constant the solution is ``expm(A t) x0``,
exact. Assumptions: the bound pool is a short-T2 spin like the free one (no absorption lineshape is assumed for it),
the pool sizes are in detailed balance unless given (``M0b = M0a k_f / k_r``), standard Bloch conventions
(precession ``dMx = -dw My``, ``dMy = +dw Mx``; RF about ``(cos phi, sin phi, 0)``).

Sources: McConnell HM (1958), *Reaction rates by nuclear magnetic resonance*, J. Chem. Phys. 28:430-431; Henkelman RM
et al. (1993), *Quantitative interpretation of magnetization transfer*, Magn. Reson. Med. 29:759-766; Graham SJ,
Henkelman RM (1997), JMRI 7:903-912; Stanisz GJ et al. (2005), Magn. Reson. Med. 54:507-512 (tissue parameters).

``evolve_two_pool`` and the forms built on it need scipy's matrix exponential (the ``special`` extra), imported inside
them.
"""
import numpy as np

from ._extras import need

__all__ = [
    "forward_rate", "backward_rate", "bound_fraction", "kappa_MT_from_forward_rate", "dwell_time_from_fraction",
    "two_pool_generator", "evolve_two_pool", "bloch_mcconnell_transverse", "bloch_mcconnell_longitudinal",
    "mt_z_spectrum",
]


def forward_rate(kappa_MT, S_over_V):
    """Forward (free->bound) exchange rate k_f = kappa_MT * (S/V), s^-1."""
    return float(kappa_MT) * float(S_over_V)


def backward_rate(dwell_time):
    """Backward (bound->free) exchange rate k_r = 1/dwell_time, s^-1."""
    if dwell_time <= 0.0:
        raise ValueError(f"dwell_time must be > 0, got {dwell_time}")
    return 1.0 / float(dwell_time)


def bound_fraction(kappa_MT, dwell_time, S_over_V):
    """Equilibrium bound-pool fraction f_b = k_f / (k_f + k_r)."""
    kf = forward_rate(kappa_MT, S_over_V)
    kr = backward_rate(dwell_time)
    return kf / (kf + kr) if (kf + kr) > 0 else 0.0


def kappa_MT_from_forward_rate(k_f, S_over_V):
    """Invert k_f = kappa_MT * (S/V) -> kappa_MT (m/s)."""
    if S_over_V <= 0.0:
        raise ValueError(f"S_over_V must be > 0, got {S_over_V}")
    return float(k_f) / float(S_over_V)


def dwell_time_from_fraction(f_bound, k_f):
    """Dwell time for a target equilibrium bound fraction at forward rate k_f.

    From detailed balance k_f*(1-f_b) relation: k_r = k_f*(1-f_b)/f_b, so
    dwell_time = 1/k_r = f_b / (k_f * (1 - f_b)).
    """
    if not (0.0 < f_bound < 1.0):
        raise ValueError(f"f_bound must be in (0,1), got {f_bound}")
    if k_f <= 0.0:
        raise ValueError(f"k_f must be > 0, got {k_f}")
    k_r = k_f * (1.0 - f_bound) / f_bound
    return 1.0 / k_r


def two_pool_generator(*, R1a, R2a, R1b, R2b, k_f, k_r,
                       M0a=1.0, M0b=None, dw_a=0.0, dw_b=0.0,
                       w1=0.0, rf_phase=0.0):
    """Build the 7x7 generator A of the two-pool Bloch--McConnell system.

    Rates in s^-1, angular frequencies (dw_a, dw_b, w1) in rad/s.  ``w1 = gamma*B1``
    is a continuous RF nutation rate; ``rf_phase`` (rad) sets the B1 axis in the
    transverse plane (0 = +x).  If ``M0b`` is None it is set from detailed balance
    ``M0b = M0a * k_f / k_r`` (so the given rates and pool sizes are consistent).

    Conventions (standard Bloch): precession dM/dt = M x (0,0,dw) -> dMx=-dw My,
    dMy=+dw Mx; RF about axis phi at rate w1 -> rotation about (cos phi, sin phi, 0).
    """
    if M0b is None:
        M0b = M0a * (k_f / k_r) if k_r > 0 else 0.0
    A = np.zeros((7, 7), dtype=float)
    ux, uy = np.cos(rf_phase), np.sin(rf_phase)

    def _block(base, R1, R2, dw, M0):
        ix, iy, iz = base, base + 1, base + 2
        # relaxation
        A[ix, ix] += -R2
        A[iy, iy] += -R2
        A[iz, iz] += -R1
        A[iz, 6] += R1 * M0                 # affine +R1*M0 recovery
        # off-resonance precession about z
        A[ix, iy] += -dw
        A[iy, ix] += +dw
        # RF nutation about axis (ux, uy, 0) at rate w1:  dM/dt = w1 * (u x M)
        # u x M = (uy*Mz - 0, 0 - ux*Mz, ux*My - uy*Mx)
        A[ix, iz] += w1 * uy
        A[iy, iz] += -w1 * ux
        A[iz, ix] += -w1 * uy
        A[iz, iy] += w1 * ux

    _block(0, R1a, R2a, dw_a, M0a)
    _block(3, R1b, R2b, dw_b, M0b)

    # exchange a <-> b (applies to all three components of each pool)
    for c in range(3):
        ia, ib = c, c + 3
        A[ia, ia] += -k_f
        A[ia, ib] += k_r
        A[ib, ib] += -k_r
        A[ib, ia] += k_f
    return A


def evolve_two_pool(state0, t, A):
    """Evolve the augmented state ``state0`` (7,) by time ``t`` under generator A: ``expm(A t) state0``."""
    expm = need("scipy.linalg").expm
    state0 = np.asarray(state0, dtype=float)
    return expm(A * float(t)) @ state0


def _equilibrium_state(M0a, M0b):
    return np.array([0, 0, M0a, 0, 0, M0b, 1.0], dtype=float)


def bloch_mcconnell_transverse(t, *, T2a, T2b, k_f, k_r, T1a=1.0, T1b=1.0,
                               M0a=1.0, M0b=None, dw_a=0.0, dw_b=0.0):
    """Free-pool transverse magnitude |Mxy,a|(t) after a 90 on both pools (no RF).

    Both pools start fully transverse (Mx=M0).  Returns |Mxa+iMya| at each ``t``.
    In the limit T2b -> 0 with dw=0 this reduces to exp(-(1/T2a + k_f) t) --
    the classic result that the forward exchange rate adds to the free-pool R2.
    """
    if M0b is None:
        M0b = M0a * (k_f / k_r) if k_r > 0 else 0.0
    A = two_pool_generator(R1a=1.0 / T1a, R2a=1.0 / T2a, R1b=1.0 / T1b,
                           R2b=1.0 / T2b, k_f=k_f, k_r=k_r, M0a=M0a, M0b=M0b,
                           dw_a=dw_a, dw_b=dw_b)
    s0 = np.array([M0a, 0, 0, M0b, 0, 0, 1.0], dtype=float)   # 90 -> transverse
    ts = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty(ts.shape, dtype=float)
    for i, tt in enumerate(ts):
        s = evolve_two_pool(s0, tt, A)
        out[i] = np.hypot(s[0], s[1])
    return out if out.size > 1 else float(out[0])


def bloch_mcconnell_longitudinal(t, *, T1a, T1b, k_f, k_r, M0a=1.0, M0b=None,
                                 Mza0=None, Mzb0=None):
    """Free-pool longitudinal Mza(t) under exchange (no RF, no transverse).

    Default initial condition inverts BOTH pools (Mz = -M0), i.e. an inversion
    recovery with exchange.  Pass ``Mza0``/``Mzb0`` for other preparations (e.g.
    saturate the bound pool: Mzb0=0, Mza0=M0a).
    """
    if M0b is None:
        M0b = M0a * (k_f / k_r) if k_r > 0 else 0.0
    if Mza0 is None:
        Mza0 = -M0a
    if Mzb0 is None:
        Mzb0 = -M0b
    A = two_pool_generator(R1a=1.0 / T1a, R2a=1e-9, R1b=1.0 / T1b, R2b=1e-9,
                           k_f=k_f, k_r=k_r, M0a=M0a, M0b=M0b)
    s0 = np.array([0, 0, Mza0, 0, 0, Mzb0, 1.0], dtype=float)
    ts = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty(ts.shape, dtype=float)
    for i, tt in enumerate(ts):
        out[i] = evolve_two_pool(s0, tt, A)[2]
    return out if out.size > 1 else float(out[0])


def mt_z_spectrum(offsets_hz, *, w1_hz, t_sat, T1a, T2a, T1b, T2b, k_f, k_r,
                  M0a=1.0, M0b=None, read_pool="a"):
    """Steady-ish Z-spectrum: free-pool Mz after a CW saturation pulse vs offset.

    A continuous-wave RF of nutation ``w1_hz`` (Hz, = gamma*B1/2pi) and duration
    ``t_sat`` is applied at each frequency ``offset`` (Hz) from the free-pool
    resonance; returns Mz of the read pool (default free ``a``) normalised to M0.
    The bound pool's short T2b makes it saturate over a broad offset range while
    the free pool saturates only near resonance -- the MT dip.  (No analytic
    lineshape is assumed; the bound pool is a real short-T2b spin, matching the MC.)
    """
    if M0b is None:
        M0b = M0a * (k_f / k_r) if k_r > 0 else 0.0
    offsets = np.atleast_1d(np.asarray(offsets_hz, dtype=float))
    w1 = 2.0 * np.pi * float(w1_hz)
    out = np.empty(offsets.shape, dtype=float)
    idx = 2 if read_pool == "a" else 5
    norm = M0a if read_pool == "a" else M0b
    for i, off in enumerate(offsets):
        dw = 2.0 * np.pi * off          # RF offset seen by both pools (same nucleus)
        A = two_pool_generator(R1a=1.0 / T1a, R2a=1.0 / T2a, R1b=1.0 / T1b,
                               R2b=1.0 / T2b, k_f=k_f, k_r=k_r, M0a=M0a, M0b=M0b,
                               dw_a=dw, dw_b=dw, w1=w1)
        s = evolve_two_pool(_equilibrium_state(M0a, M0b), t_sat, A)
        out[i] = s[idx] / norm if norm > 0 else 0.0
    return out if out.size > 1 else float(out[0])
