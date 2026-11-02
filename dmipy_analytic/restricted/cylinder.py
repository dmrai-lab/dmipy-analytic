"""Diffusion inside an impermeable cylinder in closed form: the short-gradient-pulse limit and the Gaussian-phase
approximation of the signal perpendicular to the axis, combined with free diffusion along it.

The cylinder has the diameter ``diameter`` (metres). The forms take the wave-vector ``q`` (1/m) or the gradient ``G``
(T/m) PERPENDICULAR to the axis, the one plane in which the spins are restricted; the signal of a gradient at an
angle to the axis is the product of this and the free diffusion along the axis (:func:`dmipy_analytic.gaussian.stick`)
for the components of the gradient along and across it. The capped cylinder is the cylinder across and a pair of
plates (:mod:`dmipy_analytic.restricted.plane`) along the axis.

The roots of the Bessel-function equations are those of
:func:`dmipy_analytic.restricted.matrix_method.cylinder_neumann_roots`. The forms need scipy (the ``special``
extra), imported inside them.
"""
from functools import lru_cache

import numpy as np

from .._extras import need
from ..constants import GAMMA
from . import plane
from .matrix_method import cylinder_neumann_roots

__all__ = ["sgp_long_time", "sgp", "neumann_roots_table", "gaussian_phase", "gaussian_phase_waveform",
           "gaussian_phase_cosine", "geometry_factor", "gaussian_phase_gamma_lm", "gaussian_phase_exponent_coefficients",
           "capped_sgp_long_time", "capped_sgp"]


def sgp_long_time(q, diameter):
    r"""The signal across a cylinder in the short-gradient-pulse limit at a diffusion time long enough for the spins to
    explore the whole cross-section, ``(2 J_1(2 pi q R) / (2 pi q R))^2`` with ``R = d/2``; one at ``q = 0``.

    Parameters
    ----------
    q : array_like
        The wave-vector perpendicular to the axis, 1/m.
    diameter : float
        The diameter of the cylinder, m.

    Assumes impermeable walls, gradient pulses short against the time to diffuse across the cylinder
    (``delta << R^2/D``) and a diffusion time long against it: the signal is the squared modulus of the Fourier
    transform of the uniform density over the disk, the diffraction pattern of a circular aperture. It needs
    scipy (the ``special`` extra). Source: Soderman O, Jonsson B (1995), J. Magn. Reson. A 117:94-97, Eq. 6.
    """
    special = need("scipy.special")
    q = np.asarray(q, dtype=float)
    radius = diameter / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        E = ((2 * special.jn(1, 2 * np.pi * q * radius)) ** 2 /
             (2 * np.pi * q * radius) ** 2)
    return np.where(q != 0, E, 1.0)


@lru_cache(maxsize=None)
def neumann_roots_table(n_roots, n_functions):
    """The roots of ``J_m'(x) = 0`` for the orders ``m = 0 ... n_functions - 1`` as a table of shape
    ``(n_roots, n_functions)`` (column ``m`` holds the first ``n_roots`` roots of order ``m``; the ``m = 0`` column
    starts with the constant mode at zero)."""
    table = np.stack([cylinder_neumann_roots(m, n_roots) for m in range(n_functions)], axis=1)
    table.setflags(write=False)
    return table


def sgp(q, tau, diameter, diffusivity, n_roots=20, n_functions=50):
    r"""The signal across a cylinder in the short-gradient-pulse limit at a finite diffusion time ``tau``
    (Callaghan's series): the diffraction pattern plus the radial and angular modes of the disk, each decaying as
    ``exp(-alpha^2 D tau / R^2)``; one at ``q = 0``.

    Parameters
    ----------
    q : array_like
        The wave-vector perpendicular to the axis, 1/m.
    tau : array_like
        The diffusion time, s (``Delta - delta/3`` of a pulsed gradient).
    diameter : float
        The diameter ``2 R`` of the cylinder, m.
    diffusivity : float
        The diffusivity across the cylinder, m^2/s.
    n_roots, n_functions : int
        The number of radial and of angular modes of the series.

    Assumes impermeable walls and gradient pulses short against the diffusion time. The modes are the Neumann
    eigenfunctions ``J_m(alpha r / R) cos(m phi)`` with ``J_m'(alpha) = 0`` and the series is truncated at
    ``n_roots x n_functions`` terms. It needs scipy (the ``special`` extra). Source: Callaghan PT (1995), J. Magn.
    Reson. A 113:53-59, Eqs. 16-17 (the cylindrical pore); Neuman CH (1974), J. Chem. Phys. 60:4508.
    """
    special = need("scipy.special")
    shape = np.shape(q)
    q = np.atleast_1d(np.asarray(q, dtype=float))
    tau = np.asarray(tau, dtype=float)
    alpha = neumann_roots_table(n_roots, n_functions)
    radius = diameter / 2.
    q_argument = 2 * np.pi * q * radius
    q_argument_2 = q_argument ** 2
    res = np.zeros_like(q)

    with np.errstate(divide="ignore", invalid="ignore"):
        J = special.j1(q_argument) ** 2
        for k in range(0, alpha.shape[0]):
            alpha2 = alpha[k, 0] ** 2
            update = (
                4 * np.exp(-alpha2 * diffusivity * tau / radius ** 2) *
                q_argument_2 /
                (q_argument_2 - alpha2) ** 2 * J
            )
            res += update

        for m in range(1, alpha.shape[1]):
            J = special.jvp(m, q_argument, 1)
            q_argument_J = (q_argument * J) ** 2
            for k in range(alpha.shape[0]):
                alpha2 = alpha[k, m] ** 2
                update = (
                    8 * np.exp(-alpha2 * diffusivity * tau / radius ** 2) *
                    alpha2 / (alpha2 - m ** 2) *
                    q_argument_J /
                    (q_argument_2 - alpha2) ** 2
                )
                res += update
    return np.where(q != 0, res, 1.0).reshape(shape)


def _gaussian_phase_roots(n_roots):
    return cylinder_neumann_roots(1, n_roots)


def gaussian_phase(G, delta, Delta, diameter, diffusivity, n_roots=100, gamma=GAMMA):
    r"""The signal across a cylinder of a pulsed gradient in the Gaussian-phase approximation,
    ``exp(-2 gamma^2 G^2 sum_k (2 a_k D delta - 2 + 2 e^{-a_k D delta} + 2 e^{-a_k D Delta}
    - e^{-a_k D (Delta - delta)} - e^{-a_k D (Delta + delta)}) / (D^2 alpha_k^6 (R^2 alpha_k^2 - 1)))`` with
    ``alpha_k = mu_k / R``, ``a_k = alpha_k^2`` and ``mu_k`` the roots of ``J_1'``.

    Parameters
    ----------
    G : array_like
        The gradient amplitude perpendicular to the axis, T/m.
    delta, Delta : array_like
        The gradient pulse duration and separation, s; they broadcast against ``G``.
    diameter : float
        The diameter ``2 R`` of the cylinder, m.
    diffusivity : float
        The diffusivity across the cylinder, m^2/s.
    n_roots : int
        The number of roots of ``J_1'`` summed.
    gamma : float
        The gyromagnetic ratio, rad/(s T).

    Assumes impermeable walls and that the phase accumulated by the spins is Gaussian distributed, which holds when
    many wall collisions occur during a pulse (``delta >> R^2/D``); the matrix method is exact beyond it. It needs
    scipy (the ``special`` extra). Source: Van Gelderen P, DesPres D, van Zijl PCM, Moonen CTW (1994), J. Magn.
    Reson. B 103:255-260, Eq. 7; the Gaussian phase approximation of Neuman (1974), J. Chem. Phys. 60:4508.
    """
    roots = _gaussian_phase_roots(n_roots)
    D = diffusivity
    radius = diameter / 2.
    delta = np.asarray(delta, dtype=float)[..., None]
    Delta = np.asarray(Delta, dtype=float)[..., None]
    first_factor = -2 * (np.asarray(G, dtype=float) * gamma) ** 2
    alpha = roots / radius
    alpha2 = alpha ** 2
    alpha2D = alpha2 * D

    summands = (
        2 * alpha2D * delta - 2 +
        2 * np.exp(-alpha2D * delta) +
        2 * np.exp(-alpha2D * Delta) -
        np.exp(-alpha2D * (Delta - delta)) -
        np.exp(-alpha2D * (Delta + delta))
    ) / (D ** 2 * alpha ** 6 * (radius ** 2 * alpha2 - 1))

    return np.exp(first_factor * summands.sum(axis=-1))


def gaussian_phase_waveform(G, dt, diameter, diffusivity, n_roots=100, gamma=GAMMA):
    r"""The signal across a cylinder of a sampled gradient waveform in the Gaussian-phase approximation,
    ``exp(-(gamma^2/2) sum_k B_k int int G(t) G(t') exp(-a_k D |t - t'|) dt dt')`` with ``B_k = 2 (R/mu_k)^2 /
    (mu_k^2 - 1)``, ``a_k = (mu_k/R)^2`` and ``mu_k`` the roots of ``J_1'``.

    Parameters
    ----------
    G : array_like, shape (n_t,)
        The gradient along one direction perpendicular to the axis at the samples, T/m.
    dt : float
        The sampling interval, s.
    diameter : float
        The diameter ``2 R`` of the cylinder, m.
    diffusivity : float
        The diffusivity across the cylinder, m^2/s.
    n_roots : int
        The number of roots of ``J_1'`` summed.
    gamma : float
        The gyromagnetic ratio, rad/(s T).

    Assumes impermeable walls, a Gaussian phase distribution and a gradient of fixed direction. The double integral
    over a sampled waveform is evaluated by the causal recursion ``H_i = H_{i-1} e^{-a_k D dt} + G_i dt`` with the
    diagonal counted once, ``I_k = 2 sum_i G_i dt H_i - dt^2 sum_i G_i^2``. It covers any oscillating or trapezoidal
    waveform; :func:`gaussian_phase` is its rectangular-pulse closed form. Source: Stepisnik J (1981), Physica B+C
    104:350-364 (the Gaussian-phase signal of an arbitrary waveform); Xu J et al. (2009), J. Magn. Reson. 200:189-197,
    Eq. 3.
    """
    G_f64 = np.array(G, dtype=np.float64)                  # a contiguous copy (the dot products)
    n_t = len(G_f64)
    R = diameter / 2.0
    D = diffusivity

    mu_n = _gaussian_phase_roots(n_roots)
    lam_n = (mu_n / R) ** 2
    B_n = 2.0 * (R / mu_n) ** 2 / (mu_n ** 2 - 1.0)

    phi = 0.0
    for k in range(len(mu_n)):
        lk_D = lam_n[k] * D
        decay_step = np.exp(-lk_D * dt)
        H = np.zeros(n_t, dtype=np.float64)
        H[0] = G_f64[0] * dt
        for i in range(1, n_t):
            H[i] = H[i - 1] * decay_step + G_f64[i] * dt
        I_k = 2.0 * np.dot(G_f64 * dt, H) - np.dot(G_f64, G_f64) * dt ** 2
        phi += B_n[k] * I_k

    phi *= 0.5 * gamma ** 2
    return np.exp(-phi)


def gaussian_phase_cosine(G, omega, sigma, diameter, diffusivity, n_roots=100, gamma=GAMMA):
    r"""The signal across a cylinder of a cosine gradient ``G cos(omega t)`` on ``[0, sigma]`` in the Gaussian-phase
    approximation, in closed form: the double integral of :func:`gaussian_phase_waveform` for the cosine.

    Parameters
    ----------
    G : float
        The amplitude of the gradient perpendicular to the axis, T/m.
    omega : float
        The angular frequency, rad/s.
    sigma : float
        The duration of the gradient, s.
    diameter : float
        The diameter ``2 R`` of the cylinder, m.
    diffusivity : float
        The diffusivity across the cylinder, m^2/s.
    n_roots : int
        The number of roots of ``J_1'`` summed.
    gamma : float
        The gyromagnetic ratio, rad/(s T).

    Assumes impermeable walls and a Gaussian phase distribution. With ``alpha_k = a_k D``, ``d = alpha_k^2 +
    omega^2``, ``c = cos(omega sigma)``, ``s = sin(omega sigma)`` and ``F_s = (alpha_k (1 - c e^{-alpha_k sigma}) +
    omega s e^{-alpha_k sigma}) / d``, the double integral is ``(2/d)(alpha_k (sigma/2 + s c / (2 omega)) - alpha_k
    F_s + s^2/2)``. Source: Stepisnik (1981), Physica B+C 104:350; Ianus A, Siow B, Drobnjak I, Zhang H, Alexander DC
    (2013), J. Magn. Reson. 237:128-136 (the oscillating-gradient Gaussian phase of a cylinder).
    """
    R = diameter / 2.0
    D = diffusivity
    mu_n = _gaussian_phase_roots(n_roots)
    lam_n = (mu_n / R) ** 2
    B_n = 2.0 * (R / mu_n) ** 2 / (mu_n ** 2 - 1.0)

    alpha = lam_n * D
    denom = alpha ** 2 + omega ** 2

    c = np.cos(omega * sigma)
    s = np.sin(omega * sigma)
    e_asm = np.exp(-alpha * sigma)

    F_s = (alpha * (1.0 - c * e_asm) + omega * s * e_asm) / denom

    I_n = (2.0 / denom) * (
        alpha * (sigma / 2.0 + s * c / (2.0 * omega))
        - alpha * F_s
        + s ** 2 / 2.0
    )

    phi = 0.5 * gamma ** 2 * G ** 2 * np.sum(B_n * I_n)
    return np.exp(-phi)


def geometry_factor(diameter, diffusivity, n_roots=100, gamma=GAMMA):
    r"""The geometry factor of the fast-eigenmode Gaussian-phase signal, ``gamma^2 sum_k B_k / alpha_k`` with
    ``B_k = 2 (R/mu_k)^2 / (mu_k^2 - 1)`` and ``alpha_k = (mu_k/R)^2 D``, s^3/(kg^2 m) in SI units.

    Parameters
    ----------
    diameter : float
        The diameter ``2 R`` of the cylinder, m.
    diffusivity : float
        The diffusivity across the cylinder, m^2/s.
    n_roots : int
        The number of roots of ``J_1'`` summed.
    gamma : float
        The gyromagnetic ratio, rad/(s T).

    The factor multiplies the integrated squared gradient in :func:`gaussian_phase_gamma_lm`; it needs the
    eigenmodes to decay within the waveform's time scale (``alpha_1 T >> 1``).
    """
    R = diameter / 2.0
    mu_n = _gaussian_phase_roots(n_roots)
    B_k = 2.0 * (R / mu_n) ** 2 / (mu_n ** 2 - 1.0)
    alpha_k = (mu_n / R) ** 2 * diffusivity
    return gamma ** 2 * np.sum(B_k / alpha_k)


def _real_harmonics_l2(axis):
    """The five real spherical harmonics of order two at the direction ``axis``: ``[Y2-2, Y2-1, Y20, Y21, Y22]``."""
    x, y, z = axis[0], axis[1], axis[2]
    Y2m2 = np.sqrt(15.0 / (4.0 * np.pi)) * x * y
    Y2m1 = np.sqrt(15.0 / (4.0 * np.pi)) * y * z
    Y20 = np.sqrt(5.0 / (16.0 * np.pi)) * (2.0 * z ** 2 - x ** 2 - y ** 2)
    Y21 = np.sqrt(15.0 / (4.0 * np.pi)) * x * z
    Y22 = np.sqrt(15.0 / (16.0 * np.pi)) * (x ** 2 - y ** 2)
    return np.array([Y2m2, Y2m1, Y20, Y21, Y22], dtype=np.float64)


def gaussian_phase_gamma_lm(gamma_lm, b_par, axis, diameter, diffusivity_perp, diffusivity_par, n_roots=100,
                            gamma=GAMMA):
    r"""The signal of a cylinder of axis ``u`` for any gradient waveform in the fast-eigenmode Gaussian-phase limit,
    ``exp(-(phi_perp + phi_par))`` with ``phi_perp = C (2/3 int |G|^2 dt - (8 pi/15) sum_m Gamma_2m Y_2m(u))``,
    ``C =`` :func:`geometry_factor` and ``phi_par = b_par D_par``.

    Parameters
    ----------
    gamma_lm : array_like, shape (m, 6)
        The angular power spectrum of each measurement's gradient, ``Gamma_lm = int |G(t)|^2 Y_lm(G^(t)) dt`` for
        ``l = 0`` and ``l = 2`` in the order ``Y00, Y2-2, Y2-1, Y20, Y21, Y22`` (real harmonics, orthonormal).
    b_par : array_like, shape (m,)
        The b-value along the axis, ``b (n . u)^2``, s/m^2.
    axis : array_like, shape (3,)
        The unit vector ``u`` of the cylinder's axis.
    diameter : float
        The diameter ``2 R`` of the cylinder, m.
    diffusivity_perp, diffusivity_par : float
        The diffusivity across and along the cylinder, m^2/s.
    n_roots : int
        The number of roots of ``J_1'`` summed.
    gamma : float
        The gyromagnetic ratio, rad/(s T).

    Assumes impermeable walls, a Gaussian phase distribution and eigenmodes that relax within a gradient time scale
    (``alpha_1 T >> 1``, ``alpha_1 = (mu_1/R)^2 D``): the cylinder is then an order-two low-pass filter of the
    encoding, of which only ``l = 0`` and ``l = 2`` contribute, so rotating and tensor-valued waveforms are covered.
    Source: the Gaussian phase of Stepisnik (1981), Physica B+C 104:350 for the cylinder's eigenmodes, expanded in the
    order-two harmonics of the encoding.
    """
    C_geom = geometry_factor(diameter, diffusivity_perp, n_roots, gamma)
    gamma_lm = np.asarray(gamma_lm)
    Gamma00 = gamma_lm[:, 0]
    int_G2_dt = Gamma00 * np.sqrt(4.0 * np.pi)
    Gamma2m = gamma_lm[:, 1:]
    Y2m_mu = _real_harmonics_l2(np.asarray(axis, dtype=float))
    phi_perp = C_geom * (
        (2.0 / 3.0) * int_G2_dt
        - (8.0 * np.pi / 15.0) * Gamma2m.dot(Y2m_mu)
    )
    phi_par = b_par * diffusivity_par
    return np.exp(-(phi_perp + phi_par))


def gaussian_phase_exponent_coefficients(gamma_lm, diameter, diffusivity, n_roots=100, gamma=GAMMA):
    r"""The coefficients ``(c_0, c_2m)`` of the cylinder's perpendicular Gaussian-phase exponent in the harmonics of
    the axis direction, ``phi_perp(u) = c_0 + sum_m c_2m Y_2m(u)``, with ``c_0 = C (2/3) int |G|^2 dt`` and
    ``c_2m = -C (8 pi/15) Gamma_2m``.

    Parameters
    ----------
    gamma_lm : array_like, shape (m, 6)
        The angular power spectrum of each measurement's gradient (see :func:`gaussian_phase_gamma_lm`).
    diameter : float
        The diameter ``2 R`` of the cylinder, m.
    diffusivity : float
        The diffusivity across the cylinder, m^2/s.
    n_roots : int
        The number of roots of ``J_1'`` summed.
    gamma : float
        The gyromagnetic ratio, rad/(s T).

    Returns
    -------
    c0 : ndarray, shape (m,)
    c2m : ndarray, shape (m, 5)

    The exponent is an order-two function of the axis, so the signal is a Gaussian kernel on the sphere whose
    spherical-harmonic coefficients are closed forms. Same assumptions as :func:`gaussian_phase_gamma_lm`.
    """
    C_geom = geometry_factor(diameter, diffusivity, n_roots, gamma)
    gamma_lm = np.asarray(gamma_lm)
    Gamma00 = gamma_lm[:, 0]
    int_G2_dt = Gamma00 * np.sqrt(4.0 * np.pi)
    Gamma2m = gamma_lm[:, 1:]
    c0 = C_geom * (2.0 / 3.0) * int_G2_dt
    c2m = -C_geom * (8.0 * np.pi / 15.0) * Gamma2m
    return c0, c2m


def capped_sgp_long_time(q_perp, q_par, diameter, length):
    r"""The signal of a capped cylinder in the short-gradient-pulse limit at a long diffusion time: the cylinder across
    (:func:`sgp_long_time`) times a pair of plates ``length`` apart along the axis
    (:func:`dmipy_analytic.restricted.plane.sgp_long_time`).

    Parameters
    ----------
    q_perp, q_par : array_like
        The wave-vector across and along the axis, 1/m.
    diameter : float
        The diameter of the cylinder, m.
    length : float
        The distance between the caps, m.

    Assumes the restriction across and along the axis are independent (the displacement distribution of a cylinder
    capped by planes factorises) and each is in its short-pulse, long-time limit. Source: Soderman and Jonsson (1995),
    J. Magn. Reson. A 117:94 for the cross-section; Balinov et al. (1993), J. Magn. Reson. A 104:17, Eq. 6 along it.
    """
    return plane.sgp_long_time(q_par, length) * sgp_long_time(q_perp, diameter)


def capped_sgp(q_perp, q_par, tau, diameter, length, diffusivity, n_roots_cylinder=20, n_functions_cylinder=50,
               n_roots_plane=40):
    r"""The signal of a capped cylinder in the short-gradient-pulse limit at a finite diffusion time ``tau``: the
    cylinder across (:func:`sgp`) times a pair of plates ``length`` apart along the axis
    (:func:`dmipy_analytic.restricted.plane.sgp`), both with the diffusivity ``diffusivity``.

    Parameters
    ----------
    q_perp, q_par : array_like
        The wave-vector across and along the axis, 1/m.
    tau : array_like
        The diffusion time, s.
    diameter : float
        The diameter of the cylinder, m.
    length : float
        The distance between the caps, m.
    diffusivity : float
        The diffusivity inside the cylinder, m^2/s.
    n_roots_cylinder, n_functions_cylinder, n_roots_plane : int
        The modes of the two series.

    Assumes the restriction across and along the axis are independent. Source: Callaghan (1995), J. Magn. Reson. A
    113:53, Eqs. 8-9 along the axis and 16-17 across.
    """
    return (plane.sgp(q_par, tau, length, diffusivity, n_roots_plane)
            * sgp(q_perp, tau, diameter, diffusivity, n_roots_cylinder, n_functions_cylinder))
