"""Diffusion inside an impermeable sphere in closed form: the short-gradient-pulse limit and the Gaussian-phase
approximation of the signal.

The sphere has the diameter ``diameter`` (metres) and the signal is isotropic, so the forms take the wave-vector ``q``
(1/m) or the gradient ``G`` (T/m) along the measured direction. The forms need scipy (the ``special`` extra), imported
inside them.
"""
from functools import lru_cache

import numpy as np

from .._extras import need
from ..constants import GAMMA

__all__ = ["sgp_long_time", "sgp", "GAUSSIAN_PHASE_ROOTS", "gaussian_phase", "gaussian_phase_waveform"]

GAUSSIAN_PHASE_ROOTS = np.array([
    2.081575978, 5.940369990, 9.205840145, 12.40444502,
    15.57923641, 18.74264558, 21.89969648, 25.05282528,
    28.20336100, 31.35209173, 34.49951492, 37.64596032,
    40.79165523, 43.93676147, 47.08139741, 50.22565165,
    53.36959180, 56.51327045, 59.65672900, 62.80000055,
    65.94311190, 69.08608495, 72.22893775, 75.37168540,
    78.51434055, 81.65691380, 84.79941440, 87.94185005,
    91.08422750, 94.22655255, 97.36883035, 100.5110653,
    103.6532613, 106.7954217, 109.9375497, 113.0796480,
    116.2217188, 119.3637645, 122.5057870, 125.6477880,
    128.7897690, 131.9317315, 135.0736768, 138.2156061,
    141.3575204, 144.4994207, 147.6413080, 150.7831829,
    153.9250463, 157.0668989, 160.2087413, 163.3505741,
    166.4923978, 169.6342129, 172.7760200, 175.9178194,
    179.0596116, 182.2013968, 185.3431756, 188.4849481,
    191.6267147, 194.7684757, 197.9102314, 201.0519820,
    204.1937277, 207.3354688, 210.4772054, 213.6189378,
    216.7606662, 219.9023907, 223.0441114, 226.1858287,
    229.3275425, 232.4692530, 235.6109603, 238.7526647,
    241.8943662, 245.0360648, 248.1777608, 251.3194542,
    254.4611451, 257.6028336, 260.7445198, 263.8862038,
    267.0278856, 270.1695654, 273.3112431, 276.4529189,
    279.5945929, 282.7362650, 285.8779354, 289.0196041,
    292.1612712, 295.3029367, 298.4446006, 301.5862631,
    304.7279241, 307.8695837, 311.0112420, 314.1528990,
])
"""The first 100 roots of ``j_1'(x) = 0`` (the solutions of ``J_{3/2}(x) / x = J_{5/2}(x)``), tabulated to ten digits
by Balinov et al. (1993), J. Magn. Reson. A 104:17 for the Gaussian-phase signal of the sphere (``x`` up to ``100 pi``)."""
GAUSSIAN_PHASE_ROOTS.setflags(write=False)


def sgp_long_time(q, diameter):
    r"""The signal inside a sphere in the short-gradient-pulse limit at a diffusion time long enough for the spins to
    explore the whole sphere, ``(3 (sin x / x - cos x) / x^2)^2`` with ``x = 2 pi q R``, ``R = d/2``; one at
    ``q = 0``.

    Parameters
    ----------
    q : array_like
        The wave-vector, 1/m.
    diameter : float
        The diameter of the sphere, m.

    Assumes impermeable walls, gradient pulses short against the time to diffuse across the sphere
    (``delta << R^2/D``) and a diffusion time long against it: the signal is the squared modulus of the Fourier
    transform of the uniform density in the ball, the diffraction pattern of a sphere. Source: Balinov B, Jonsson B,
    Linse P, Soderman O (1993), J. Magn. Reson. A 104:17-25, Eq. 5.
    """
    q = np.asarray(q, dtype=float)
    radius = diameter / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = 2 * np.pi * q * radius
        E = (
            3 / (factor ** 2) *
            (
                np.sin(factor) / factor -
                np.cos(factor)
            )
        ) ** 2
    return np.where(q != 0, E, 1.0)


def _spherical_jn_prime_roots(n, n_roots):
    """The first ``n_roots`` positive roots of ``d/dx j_n(x) = 0``, bracketed on a grid of six points per unit ``x``
    up to ``(n_roots + n + 2) pi`` and refined by Brent's method."""
    special = need("scipy.special")
    brentq = need("scipy.optimize").brentq
    xmax = (n_roots + n + 2) * np.pi
    xs = np.linspace(1e-6, xmax, int(xmax * 6) + 2)
    f = special.spherical_jn(n, xs, derivative=True)
    roots = []
    for i in range(len(xs) - 1):
        if f[i] == 0.0:
            roots.append(xs[i])
        elif f[i] * f[i + 1] < 0:
            roots.append(brentq(
                lambda x: special.spherical_jn(n, x, derivative=True),
                xs[i], xs[i + 1]))
        if len(roots) >= n_roots:
            break
    return np.array(roots[:n_roots])


@lru_cache(maxsize=None)
def _callaghan_roots(n_roots, n_functions):
    """The roots ``alpha[n, k]`` of ``j_n'(x) = 0`` for the orders ``n = 0 ... n_functions - 1``; the order ``n = 0``
    starts with the constant mode ``alpha = 0``."""
    alpha = np.zeros((n_functions, n_roots))
    for n in range(n_functions):
        if n == 0:
            alpha[0, 1:] = _spherical_jn_prime_roots(0, n_roots - 1)
        else:
            alpha[n, :] = _spherical_jn_prime_roots(n, n_roots)
    alpha.setflags(write=False)
    return alpha


def sgp(q, tau, diameter, diffusivity, n_roots=25, n_functions=16):
    r"""The signal inside a sphere in the short-gradient-pulse limit at a finite diffusion time ``tau`` (Callaghan's
    series): ``E = F(x)^2 + 6 sum_{n,k: alpha_nk > 0} (2n + 1) alpha_nk^2 / (alpha_nk^2 - n(n+1))
    [x j_n'(x) / (x^2 - alpha_nk^2)]^2 exp(-alpha_nk^2 D tau / R^2)`` with ``x = 2 pi q R``, ``R = d/2``,
    ``F(x) = 3 (sin x - x cos x) / x^3`` the uniform-sphere structure factor (the ground state, the long-time limit),
    ``alpha_nk`` the roots of ``j_n'`` and ``j_n`` the spherical Bessel function; one at ``q = 0``.

    Parameters
    ----------
    q : array_like
        The wave-vector, 1/m.
    tau : array_like
        The diffusion time, s (``Delta - delta/3`` of a pulsed gradient).
    diameter : float
        The diameter ``2 R`` of the sphere, m.
    diffusivity : float
        The diffusivity inside the sphere, m^2/s.
    n_roots, n_functions : int
        The number of radial and of angular modes of the series.

    Assumes impermeable walls and gradient pulses short against the diffusion time. The factor 6 and the ground
    state together satisfy completeness, ``E(tau -> 0) = 1``. At the resonance ``x = alpha_nk`` the factor
    ``x j_n'(x) / (x^2 - alpha^2)`` is replaced by its limit ``-(alpha^2 - n(n+1)) j_n(x) / (2 alpha^2)``. It needs
    scipy (the ``special`` extra). Source: Callaghan PT (1995), J. Magn. Reson. A 113:53-59, Eqs. 22-24 (the spherical
    pore); Grebenkov DS (2007), Rev. Mod. Phys. 79:1077 for the eigenfunctions.
    """
    special = need("scipy.special")
    shape = np.shape(q)
    q = np.atleast_1d(np.abs(np.asarray(q, dtype=float)))
    tau = np.asarray(tau, dtype=float)
    alpha = _callaghan_roots(n_roots, n_functions)
    radius = diameter / 2.0
    radius2 = radius ** 2
    D = diffusivity
    x = 2 * np.pi * q * radius
    x2 = x ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        struct = np.where(x > 0,
                          3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3,
                          1.0)
        E = struct ** 2

        for n in range(alpha.shape[0]):
            jn = special.spherical_jn(n, x)
            jn_prime = special.spherical_jn(n, x, derivative=True)
            for k in range(alpha.shape[1]):
                a = alpha[n, k]
                if a == 0.0:
                    continue
                a2 = a * a
                Cn = a2 / (a2 - n * (n + 1))
                denom = x2 - a2
                near = np.abs(denom) < 1e-9 * (1.0 + a2)
                M_reg = x * jn_prime / np.where(near, 1.0, denom)
                M_lim = -(a2 - n * (n + 1)) * jn / (2.0 * a2)
                M = np.where(near, M_lim, M_reg)
                E = E + 6.0 * (2 * n + 1) * Cn * M ** 2 * \
                    np.exp(-a2 * D * tau / radius2)
    return np.where(x > 0, E, 1.0).reshape(shape)


def gaussian_phase(G, delta, Delta, diameter, diffusivity, n_roots=100, gamma=GAMMA):
    r"""The signal inside a sphere of a pulsed gradient in the Gaussian-phase approximation,
    ``exp(-2 gamma^2 G^2 / D sum_k alpha_k^-4 / (R^2 alpha_k^2 - 2) (2 delta - (2 + e^{-a_k D (Delta - delta)}
    - 2 e^{-a_k D delta} - 2 e^{-a_k D Delta} + e^{-a_k D (Delta + delta)}) / (a_k D)))`` with ``alpha_k = x_k / R``,
    ``a_k = alpha_k^2`` and ``x_k`` the roots of ``j_1'``.

    Parameters
    ----------
    G : array_like
        The gradient amplitude, T/m.
    delta, Delta : array_like
        The gradient pulse duration and separation, s; they broadcast against ``G``.
    diameter : float
        The diameter ``2 R`` of the sphere, m.
    diffusivity : float
        The diffusivity inside the sphere, m^2/s.
    n_roots : int
        The number of roots summed, at most 100.
    gamma : float
        The gyromagnetic ratio, rad/(s T).

    Assumes impermeable walls and that the phase accumulated by the spins is Gaussian distributed, which holds when
    many wall collisions occur during a pulse (``delta >> R^2/D``); the matrix method is exact beyond it. The roots
    are :data:`GAUSSIAN_PHASE_ROOTS`. Source: Murday JS, Cotts RM (1968), J. Chem. Phys. 48:4938-4945 (the sphere);
    Balinov et al. (1993), J. Magn. Reson. A 104:17, Eqs. 8-9.
    """
    D = diffusivity
    radius = diameter / 2
    roots = GAUSSIAN_PHASE_ROOTS[:n_roots]
    delta = np.asarray(delta, dtype=float)[..., None]
    Delta = np.asarray(Delta, dtype=float)[..., None]

    alpha = roots / radius
    alpha2 = alpha ** 2
    alpha2D = alpha2 * D

    first_factor = -2 * (gamma * np.asarray(G, dtype=float)) ** 2 / D
    summands = (
        alpha ** (-4) / (alpha2 * radius ** 2 - 2) *
        (
            2 * delta - (
                2 +
                np.exp(-alpha2D * (Delta - delta)) -
                2 * np.exp(-alpha2D * delta) -
                2 * np.exp(-alpha2D * Delta) +
                np.exp(-alpha2D * (Delta + delta))
            ) / (alpha2D)
        )
    )
    return np.exp(first_factor * summands.sum(axis=-1))


def gaussian_phase_waveform(G, dt, diameter, diffusivity, n_roots=100, gamma=GAMMA):
    r"""The signal inside a sphere of a sampled gradient waveform in the Gaussian-phase approximation,
    ``exp(-(gamma^2/2) sum_k B_k int int G(t) . G(t') exp(-a_k D |t - t'|) dt dt')`` with ``B_k = 2 (R/x_k)^2 /
    (x_k^2 - 2)``, ``a_k = (x_k/R)^2`` and ``x_k`` the roots of ``j_1'``.

    Parameters
    ----------
    G : array_like, shape (n_t,) or (n_t, c)
        The gradient at the samples, T/m: one component, or ``c`` Cartesian components whose phases add (the sphere
        is isotropic, so any waveform, rotating or tensor-valued, is covered).
    dt : float
        The sampling interval, s.
    diameter : float
        The diameter ``2 R`` of the sphere, m.
    diffusivity : float
        The diffusivity inside the sphere, m^2/s.
    n_roots : int
        The number of roots summed, at most 100.
    gamma : float
        The gyromagnetic ratio, rad/(s T).

    Assumes impermeable walls and a Gaussian phase distribution. The double integral over a sampled waveform is
    evaluated by the causal recursion ``H_i = H_{i-1} e^{-a_k D dt} + G_i dt`` with the diagonal counted once,
    ``I_k = 2 sum_i G_i dt H_i - dt^2 sum_i G_i^2``. Source: Stepisnik J (1981), Physica B+C 104:350; Xu J et al.
    (2009), J. Magn. Reson. 200:189-197, Eq. 3 (the sphere); Murday and Cotts (1968), J. Chem. Phys. 48:4938.
    """
    G = np.asarray(G, dtype=np.float64)
    components = G[:, None] if G.ndim == 1 else G
    n_t = components.shape[0]
    R = diameter / 2.0
    D = diffusivity

    mu_k = np.asarray(GAUSSIAN_PHASE_ROOTS[:n_roots], dtype=np.float64)
    lam_k = (mu_k / R) ** 2
    B_k = 2.0 * (R / mu_k) ** 2 / (mu_k ** 2 - 2.0)

    phi = 0.0
    for c in range(components.shape[1]):
        G_f64 = np.ascontiguousarray(components[:, c])     # one contiguous waveform per component (the dot products)
        for k in range(len(mu_k)):
            lkd = lam_k[k] * D
            decay_step = np.exp(-lkd * dt)

            H = np.zeros(n_t)
            H[0] = G_f64[0] * dt
            for n in range(1, n_t):
                H[n] = H[n - 1] * decay_step + G_f64[n] * dt

            I_k = 2.0 * np.dot(G_f64 * dt, H) - np.dot(G_f64, G_f64) * dt ** 2
            phi += B_k[k] * I_k

    phi *= 0.5 * gamma ** 2
    return np.exp(-phi)
