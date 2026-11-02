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
from .matrix_method import sphere_neumann_roots

__all__ = ["sgp_long_time", "sgp", "gaussian_phase", "gaussian_phase_waveform"]

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


@lru_cache(maxsize=None)
def _callaghan_roots(n_roots, n_functions):
    """The roots ``alpha[n, k]`` of ``j_n'(x) = 0`` for the orders ``n = 0 ... n_functions - 1``
    (:func:`dmipy_analytic.restricted.matrix_method.sphere_neumann_roots`); the order ``n = 0`` starts with the
    constant mode ``alpha = 0``."""
    alpha = np.array([sphere_neumann_roots(n, n_roots) for n in range(n_functions)])
    alpha.setflags(write=False)
    return alpha


@lru_cache(maxsize=None)
def _gaussian_phase_roots(n_roots):
    """The first ``n_roots`` roots of ``j_1'(x) = 0``, the solutions of ``J_{3/2}(x) / x = J_{5/2}(x)``."""
    roots = sphere_neumann_roots(1, n_roots)
    roots.setflags(write=False)
    return roots


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
        The number of roots summed.
    gamma : float
        The gyromagnetic ratio, rad/(s T).

    Assumes impermeable walls and that the phase accumulated by the spins is Gaussian distributed, which holds when
    many wall collisions occur during a pulse (``delta >> R^2/D``); the matrix method is exact beyond it. The roots
    are those of :func:`dmipy_analytic.restricted.matrix_method.sphere_neumann_roots` for ``l = 1``. Source: Murday JS,
    Cotts RM (1968), J. Chem. Phys. 48:4938-4945 (the sphere);
    Balinov et al. (1993), J. Magn. Reson. A 104:17, Eqs. 8-9.
    """
    D = diffusivity
    radius = diameter / 2
    roots = _gaussian_phase_roots(n_roots)
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
        The number of roots summed.
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

    mu_k = _gaussian_phase_roots(n_roots)
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
