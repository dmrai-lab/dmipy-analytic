"""Diffusion between two parallel impermeable plates in closed form: the short-gradient-pulse limit and the
Gaussian-phase approximation of the signal along the plates' normal.

The plates are a distance ``diameter`` apart (metres); the gradient is taken along their normal, the one direction in
which the spins are restricted. Diffusion parallel to the plates is free and does not enter. ``q`` is the
wave-vector of the pulsed gradient ``q = gamma G delta / (2 pi)``, in 1/m.
"""
import numpy as np

from ..constants import GAMMA

__all__ = ["sgp_long_time", "sgp", "gaussian_phase"]


def sgp_long_time(q, diameter):
    r"""The signal between two plates in the short-gradient-pulse limit at a diffusion time long enough for the
    spins to explore the whole gap, ``2 (1 - cos(2 pi q d)) / (2 pi q d)^2``; one at ``q = 0``.

    Parameters
    ----------
    q : array_like
        The wave-vector of the gradient along the normal, 1/m.
    diameter : float
        The separation ``d`` of the plates, m.

    Assumes impermeable walls, gradient pulses short against the time to diffuse across the gap (``delta << d^2/D``)
    and a diffusion time long against it (``Delta >> d^2/D``): the signal is the squared modulus of the Fourier
    transform of the uniform density across the gap, the diffraction pattern of a slit. Source: Balinov B, Jonsson B,
    Linse P, Soderman O (1993), J. Magn. Reson. A 104:17-25, Eq. 6.
    """
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        q_argument = 2 * np.pi * q * diameter
        E = 2 * (1 - np.cos(q_argument)) / q_argument ** 2
    return np.where(q != 0, E, 1.0)


def sgp(q, tau, diameter, diffusivity, n_roots=40):
    r"""The signal between two plates in the short-gradient-pulse limit at a finite diffusion time ``tau`` (Callaghan's
    series): the diffraction pattern of the slit plus the modes of the gap, each decaying as
    ``exp(-xi^2 D tau / a^2)``, ``a = d/2``, with ``xi = n pi`` (the cosine modes about the midplane) and
    ``xi = n pi + pi/2`` (the sine modes); one at ``q = 0``.

    Parameters
    ----------
    q : array_like
        The wave-vector of the gradient along the normal, 1/m.
    tau : array_like
        The diffusion time, s (``Delta - delta/3`` of a pulsed gradient).
    diameter : float
        The separation ``d`` of the plates, m.
    diffusivity : float
        The diffusivity between the plates, m^2/s.
    n_roots : int
        The number of modes of each parity.

    Assumes impermeable walls and gradient pulses short against the diffusion time. A term of the series whose
    denominator vanishes (a resonance ``q a = xi``) is dropped from the sum. Source: Callaghan PT (1995), J. Magn.
    Reson. A 113:53-59 (the planar pore).
    """
    shape = np.shape(q)
    q = np.atleast_1d(np.asarray(q, dtype=float))
    tau = np.asarray(tau, dtype=float)
    xi = np.arange(n_roots) * np.pi
    zeta = np.arange(n_roots) * np.pi + np.pi / 2.0
    radius = diameter / 2.0
    q_argument = 2 * np.pi * q * radius
    q_argument_2 = q_argument ** 2
    res = np.zeros_like(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        for n in range(len(xi)):
            xi_n = xi[n]
            xi_n2 = xi[n] ** 2

            if xi_n == 0.:
                div = 1.
            else:
                div = np.sin(2 * xi_n) / 2 * xi_n

            update = (
                2 * np.exp(-xi_n2 * diffusivity * tau / radius ** 2) /
                (1 + div) *
                (q_argument * np.sin(q_argument) * np.cos(xi_n) - xi_n *
                 np.cos(q_argument) * np.sin(xi_n)) ** 2 /
                (q_argument_2 - xi_n2) ** 2
            )

            update[~np.isfinite(update)] = 0.

            res += update

        for m in range(len(zeta)):
            zeta_m = zeta[m]
            zeta_m2 = zeta[m] ** 2

            update = (
                2 * np.exp(-zeta_m2 * diffusivity * tau / radius ** 2) /
                (1 - np.sin(2 * zeta_m) / (2 * zeta_m)) *
                (q_argument * np.cos(q_argument) * np.sin(zeta_m) - zeta_m *
                 np.sin(q_argument) * np.cos(zeta_m)) ** 2 /
                (q_argument_2 - zeta_m2) ** 2
            )

            update[~np.isfinite(update)] = 0.
            res += update
    return np.where(q != 0, res, 1.0).reshape(shape)


def gaussian_phase(G, delta, Delta, diameter, diffusivity, n_modes=100, gamma=GAMMA):
    r"""The signal between two plates of a pulsed gradient in the Gaussian-phase approximation,
    ``exp(-gamma^2 G^2 sum_{k odd} (8 d^2 / (k pi)^4) I_k)`` with the bracket of the Gaussian phase of a pair of
    rectangular pulses ``I_k = (2 lambda_k delta - 2 + 2 e^{-lambda_k delta} + 2 e^{-lambda_k Delta}
    - e^{-lambda_k (Delta - delta)} - e^{-lambda_k (Delta + delta)}) / lambda_k^2`` and the decay rates
    ``lambda_k = D (k pi / d)^2``.

    Parameters
    ----------
    G : array_like
        The gradient amplitude along the normal, T/m.
    delta, Delta : array_like
        The gradient pulse duration and separation, s; they broadcast against ``G``.
    diameter : float
        The separation ``d`` of the plates, m.
    diffusivity : float
        The diffusivity between the plates, m^2/s.
    n_modes : int
        The number of odd modes summed (the terms fall as ``k^-4``).
    gamma : float
        The gyromagnetic ratio, rad/(s T).

    Assumes impermeable walls and that the phase accumulated by the spins is Gaussian distributed, which holds when
    many wall collisions occur during a pulse (``delta >> d^2/D``) and degrades at short pulses and high b, where
    the matrix method is exact. Only the odd modes couple the uniform ground state to the position operator, with
    weight ``|<k|x|0>|^2 = 8 d^2/(k pi)^4``. Source: the Gaussian phase approximation of Murday JS, Cotts RM (1968),
    J. Chem. Phys. 48:4938, applied to the planar pore; Balinov et al. (1993), J. Magn. Reson. A 104:17.
    """
    D = diffusivity
    L = diameter
    k = np.arange(1, 2 * int(n_modes), 2)
    lam_D = D * (k * np.pi / L) ** 2
    delta = np.asarray(delta, dtype=float)[..., None]
    Delta = np.asarray(Delta, dtype=float)[..., None]
    bracket = (2 * lam_D * delta - 2 + 2 * np.exp(-lam_D * delta)
               + 2 * np.exp(-lam_D * Delta)
               - np.exp(-lam_D * (Delta - delta))
               - np.exp(-lam_D * (Delta + delta))) / lam_D ** 2
    weight = 8.0 * L ** 2 / (k * np.pi) ** 4
    phi2 = (gamma * np.asarray(G, dtype=float)) ** 2 * np.sum(weight * bracket, axis=-1)
    return np.exp(-phi2)
