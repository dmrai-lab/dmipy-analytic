"""Diffusion with exchange between compartments: the Karger model of exchanging Gaussian pools and its propagators
under spin-echo and stimulated-echo encoding, and the forms of diffusion through permeable membranes.

**The Karger model.** Two pools of Gaussian diffusion, an intra-cellular pool of volume fraction ``f`` and an
extra-cellular pool ``1 - f``, exchange water at the rate ``kappa`` (s^-1) from the first to the second and
``kappa f / (1 - f)`` back (detailed balance). Over the diffusion time ``t_d`` each pool dephases with its exponent
``R = -ln E`` (the log-attenuation it would have alone), and the magnetisation vector ``(M_1, M_2)`` obeys the
two-pool rate equation ``dM/dt = (K - R/t_d) M`` with the exchange generator ``K = [[-kappa, k_ee], [kappa, -k_ee]]``.
:func:`karger` is its solution in closed form for a diffusion time at which the dephasing is constant, the sum of two
exponentials in the eigenvalues of ``K t_d + R``; the propagators :func:`spin_echo_signal` and
:func:`stimulated_echo_signal` are its exact solution through the intervals of a sequence, one matrix exponential
per interval, with a relaxation time ``T2`` and ``T1`` per pool and the radiofrequency pulses of finite duration
(a pulse relaxes both ways at ``(2/pi)(1/T2 + 1/T1)``).

Sources: Karger J, Pfeifer H, Heink W (1988), Adv. Magn. Reson. 12:1-89 (the two-site exchange of the diffusion
attenuation); Karger J (1985), Adv. Colloid Interface Sci. 23:129-148; Jelescu IO et al. (2022), NeuroImage 256:119277
(NEXI: the anisotropic Karger signal of the neurites and the extra-neurite space, Eqs. 3-6); the matrix propagators are
the exchange generalisation of the multi-interval Bloch-Torrey solution.

The propagators need scipy's matrix exponential (the ``special`` extra), imported inside them.
"""
import numpy as np

from ._extras import need

__all__ = ["diffusion_time", "karger", "exchange_matrix", "spin_echo_signal", "stimulated_echo_signal"]


def diffusion_time(delta, Delta):
    r"""The effective diffusion time of a pulsed-gradient measurement, ``Delta - delta/3``.

    Parameters
    ----------
    delta, Delta : float or array_like
        The gradient pulse duration and the pulse separation, s.

    Assumes rectangular gradient pulses; the time over which the encoded displacement is correlated. Source:
    Stejskal EO, Tanner JE (1965), J. Chem. Phys. 42:288 (the factor ``Delta - delta/3`` of the attenuation).
    """
    return Delta - delta / 3.0


def karger(exponent_intra, exponent_extra, kappa, f, diffusion_time):
    r"""The Karger signal of two exchanging Gaussian pools, ``sigma_+ exp(-lambda_+) + (1 - sigma_+) exp(-lambda_-)``
    with ``lambda_+/-`` the eigenvalues of the exchange-weighted dephasing matrix.

    Parameters
    ----------
    exponent_intra, exponent_extra : array_like
        The dephasing exponents ``R = -ln E`` the intra- and extra-cellular pool would have alone, per measurement.
    kappa : float
        The exchange rate from the intra- to the extra-cellular pool, s^-1.
    f : float
        The intra-cellular volume fraction, in (0, 1).
    diffusion_time : float or array_like
        The effective diffusion time, s (:func:`diffusion_time`).

    Assumes Gaussian diffusion in each pool, exchange at a constant rate between the pools in detailed balance
    (``kappa f / (1 - f)`` back), and a dephasing that accumulates over the diffusion time as if constant (the
    narrow-pulse picture). At ``kappa = 0`` the weights are the volume fractions, ``f`` on the larger exponent when
    it is the intra-cellular one, and a vanishing eigenvalue gap gives the mean exponent with weight one half.
    Source: Karger et al. (1988), Adv. Magn. Reson. 12:1; Jelescu et al. (2022), NeuroImage 256:119277, Eqs. 3-5.
    """
    EPS = 1e-10
    kee = kappa * f / (1.0 - f)
    kt = kappa * diffusion_time
    keet = kee * diffusion_time

    Tr = exponent_intra + exponent_extra + kt + keet
    Det = exponent_intra * exponent_extra + exponent_intra * keet + exponent_extra * kt

    disc_sq = np.maximum(Tr ** 2 - 4.0 * Det, 0.0)
    disc = np.sqrt(disc_sq)

    lam_plus = (Tr + disc) / 2.0
    lam_minus = (Tr - disc) / 2.0

    safe_disc = np.where(disc > EPS, disc, np.ones_like(disc))
    safe_keet = keet if np.ndim(keet) == 0 else np.where(keet > EPS, keet, np.ones_like(keet))

    with np.errstate(divide="ignore", invalid="ignore"):         # the branch not taken may divide by zero
        sp_full = f * (exponent_intra - lam_minus) * (lam_minus - exponent_extra) / (safe_keet * safe_disc)
        sp_kappa0 = (f * exponent_intra + (1.0 - f) * exponent_extra - lam_minus) / safe_disc
    sigma_plus = np.where(
        disc <= EPS,
        0.5,
        np.where(np.asarray(keet) <= EPS, sp_kappa0, sp_full),
    )

    return sigma_plus * np.exp(-lam_plus) + (1.0 - sigma_plus) * np.exp(-lam_minus)


def exchange_matrix(kappa, f):
    r"""The exchange generator of two pools in detailed balance, ``[[-kappa, k_ee], [kappa, -k_ee]]`` with
    ``k_ee = kappa f / (1 - f)``.

    Parameters
    ----------
    kappa : float
        The exchange rate from the first (intra-cellular) to the second pool, s^-1.
    f : float
        The volume fraction of the first pool, in (0, 1).

    The columns sum to zero (the exchange conserves magnetisation) and ``K (f, 1 - f)^T = 0`` (the volume fractions
    are the equilibrium). Source: Karger et al. (1988), Adv. Magn. Reson. 12:1.
    """
    kee = kappa * f / (1.0 - f)
    return np.array([[-kappa, kee],
                     [kappa, -kee]], dtype=float)


def spin_echo_signal(K, M0, D, T2, T1, b, dt1, dt2, tau_exc=0.0, tau_180=0.0):
    r"""The signal of exchanging pools at a spin echo of finite radiofrequency pulses: the magnetisation summed over
    the pools after ``P_fp2 P_180 P_fp1 P_exc M0``, each ``P = expm(.)`` the propagator of one interval.

    Parameters
    ----------
    K : array_like, shape (N, N)
        The exchange generator (columns sum to zero), s^-1.
    M0 : array_like, shape (N,)
        The initial magnetisation of each pool, the volume fractions.
    D, T2, T1 : array_like, shape (N,)
        The effective diffusivity (m^2/s) and the relaxation times (s) of each pool; ``1e10`` s is no relaxation.
    b : float
        The b-value, s/m^2, split equally between the two free-precession intervals.
    dt1, dt2 : float
        The durations of the free precession before and after the refocusing pulse, s.
    tau_exc, tau_180 : float
        The durations of the excitation and refocusing pulses, s; zero is an instantaneous pulse.

    Assumes Gaussian diffusion in each pool, constant gradient-weighting rate ``b/2`` over each free-precession
    interval, and during a pulse relaxation at ``(2/pi)(1/T2 + 1/T1)`` with exchange. Needs scipy (``special``).
    Source: the multi-interval solution of the exchange-coupled Bloch-Torrey equation; Karger et al. (1988), Adv. Magn.
    Reson. 12:1 for the exchange generator.
    """
    expm = need("scipy.linalg").expm
    RT2 = np.diag(1.0 / np.asarray(T2, dtype=float))
    RT1 = np.diag(1.0 / np.asarray(T1, dtype=float))
    RD = np.diag(np.asarray(D, dtype=float))
    R12 = (2.0 / np.pi) * (RT2 + RT1)
    B = b / 2.0
    P_exc = expm((K - R12) * tau_exc)
    P_fp1 = expm((K - RT2) * dt1 - B * RD)
    P_180 = expm((K - R12) * tau_180)
    P_fp2 = expm((K - RT2) * dt2 - B * RD)
    M_TE = P_fp2 @ P_180 @ P_fp1 @ P_exc @ np.asarray(M0, dtype=float)
    return np.sum(M_TE)


def stimulated_echo_signal(K, M0, D, T2, T1, b, delta, TM, dt6, tau_90=0.0):
    r"""The signal of exchanging pools at a stimulated echo of finite radiofrequency pulses, including the factor
    one half of the stimulated pathway: ``P_enc2 P_rec P_mix P_store P_enc1 P_exc M0``.

    Parameters
    ----------
    K : array_like, shape (N, N)
        The exchange generator, s^-1.
    M0 : array_like, shape (N,)
        The initial magnetisation of each pool.
    D, T2, T1 : array_like, shape (N,)
        The effective diffusivity (m^2/s) and the relaxation times (s) of each pool.
    b : float
        The b-value, s/m^2, split equally between the two encoding intervals.
    delta : float
        The duration of the first encoding interval, s.
    TM : float
        The mixing time during which the magnetisation is longitudinal (``T1`` relaxation and exchange only), s.
    dt6 : float
        The duration of the second encoding interval, s.
    tau_90 : float
        The duration of each of the three 90-degree pulses, s.

    Assumes the transverse encoding is two intervals of constant gradient-weighting rate ``b/2`` bracketing the
    mixing time, and the pathway factor one half of a stimulated echo. Needs scipy (``special``). Source: the
    multi-interval solution of the exchange-coupled Bloch-Torrey equation; Karger et al. (1988), Adv. Magn. Reson.
    12:1.
    """
    expm = need("scipy.linalg").expm
    RT2 = np.diag(1.0 / np.asarray(T2, dtype=float))
    RT1 = np.diag(1.0 / np.asarray(T1, dtype=float))
    RD = np.diag(np.asarray(D, dtype=float))
    R12 = (2.0 / np.pi) * (RT2 + RT1)
    B = b / 2.0
    P_exc = expm((K - R12) * tau_90)
    P_enc1 = expm((K - RT2) * delta - B * RD)
    P_store = expm((K - R12) * tau_90)
    P_mix = expm((K - RT1) * TM)
    P_rec = expm((K - R12) * tau_90)
    P_enc2 = expm((K - RT2) * dt6 - B * RD)
    M_TE = 0.5 * P_enc2 @ P_rec @ P_mix @ P_store @ P_enc1 @ P_exc @ np.asarray(M0, dtype=float)
    return np.sum(M_TE)
