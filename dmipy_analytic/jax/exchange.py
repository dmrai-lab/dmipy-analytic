"""The JAX twins of the forms of :mod:`dmipy_analytic.exchange`, each held to its numpy definition.

The twins have no Python branch on a value: every conditional is a ``jnp.where``. The propagators use
``jax.scipy.linalg.expm`` and are written for any number of pools, ``K`` an ``(N, N)`` generator.
"""
import jax.numpy as jnp
import jax.scipy.linalg as jsl

__all__ = ["diffusion_time", "karger", "exchange_matrix", "spin_echo_signal", "stimulated_echo_signal"]


def diffusion_time(delta, Delta):
    """The twin of :func:`dmipy_analytic.exchange.diffusion_time`."""
    return Delta - delta / 3.0


def karger(exponent_intra, exponent_extra, kappa, f, diffusion_time):
    """The twin of :func:`dmipy_analytic.exchange.karger`, with the same limits at ``kappa = 0`` and at a vanishing
    eigenvalue gap."""
    EPS = 1e-10
    kee = kappa * f / (1.0 - f)
    kt = kappa * diffusion_time
    keet = kee * diffusion_time

    Tr = exponent_intra + exponent_extra + kt + keet
    Det = exponent_intra * exponent_extra + exponent_intra * keet + exponent_extra * kt
    disc_sq = jnp.maximum(Tr ** 2 - 4.0 * Det, 0.0)
    disc = jnp.sqrt(disc_sq)

    lam_plus = (Tr + disc) / 2.0
    lam_minus = (Tr - disc) / 2.0

    safe_disc = jnp.where(disc > EPS, disc, jnp.ones_like(disc))
    safe_keet = jnp.where(keet > EPS, keet, jnp.ones_like(keet))

    sp_full = f * (exponent_intra - lam_minus) * (lam_minus - exponent_extra) / (safe_keet * safe_disc)
    sp_kappa0 = (f * exponent_intra + (1.0 - f) * exponent_extra - lam_minus) / safe_disc
    sigma_plus = jnp.where(disc <= EPS, 0.5, jnp.where(keet <= EPS, sp_kappa0, sp_full))
    sigma_minus = 1.0 - sigma_plus

    return sigma_plus * jnp.exp(-lam_plus) + sigma_minus * jnp.exp(-lam_minus)


def exchange_matrix(kappa, f):
    """The twin of :func:`dmipy_analytic.exchange.exchange_matrix`."""
    kee = kappa * f / (1.0 - f)
    return jnp.array([[-kappa, kee],
                      [kappa, -kee]])


def spin_echo_signal(K, M0, D, T2, T1, b, dt1, dt2, tau_exc=0.0, tau_180=0.0):
    """The twin of :func:`dmipy_analytic.exchange.spin_echo_signal`."""
    RT2 = jnp.diag(1.0 / T2)
    RT1 = jnp.diag(1.0 / T1)
    RD = jnp.diag(D)
    R12 = (2.0 / jnp.pi) * (RT2 + RT1)
    B = b / 2.0
    P_exc = jsl.expm((K - R12) * tau_exc)
    P_fp1 = jsl.expm((K - RT2) * dt1 - B * RD)
    P_180 = jsl.expm((K - R12) * tau_180)
    P_fp2 = jsl.expm((K - RT2) * dt2 - B * RD)
    return jnp.sum(P_fp2 @ P_180 @ P_fp1 @ P_exc @ M0)


def stimulated_echo_signal(K, M0, D, T2, T1, b, delta, TM, dt6, tau_90=0.0):
    """The twin of :func:`dmipy_analytic.exchange.stimulated_echo_signal`."""
    RT2 = jnp.diag(1.0 / T2)
    RT1 = jnp.diag(1.0 / T1)
    RD = jnp.diag(D)
    R12 = (2.0 / jnp.pi) * (RT2 + RT1)
    B = b / 2.0
    P_exc = jsl.expm((K - R12) * tau_90)
    P_enc1 = jsl.expm((K - RT2) * delta - B * RD)
    P_store = jsl.expm((K - R12) * tau_90)
    P_mix = jsl.expm((K - RT1) * TM)
    P_rec = jsl.expm((K - R12) * tau_90)
    P_enc2 = jsl.expm((K - RT2) * dt6 - B * RD)
    M_TE = 0.5 * P_enc2 @ P_rec @ P_mix @ P_store @ P_enc1 @ P_exc @ M0
    return jnp.sum(M_TE)
