"""The JAX twins of the forms of :mod:`dmipy_analytic.exchange`, each held to its numpy definition.

The twins have no Python branch on a value: every conditional is a ``jnp.where``. The propagators use
``jax.scipy.linalg.expm`` and are written for any number of pools, ``K`` an ``(N, N)`` generator.
"""
import jax.numpy as jnp
import jax.scipy.linalg as jsl

from ..constants import GAMMA

__all__ = ["diffusion_time", "karger", "exchange_matrix", "spin_echo_signal", "stimulated_echo_signal",
           "karger_waveform", "fexi_narrow_pulse", "fexi_adc", "fexi_signal", "fexi_two_site"]


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


def _expm2(M):
    """The twin of ``dmipy_analytic.exchange._expm2``: the closed 2 x 2 exponential, branches as ``jnp.where``."""
    a, b, c, d = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
    tau = 0.5 * (a + d)
    s2 = (0.5 * (a - d)) ** 2 + b * c
    x = jnp.sqrt(jnp.abs(s2))
    small = x < 1e-3
    xs = jnp.where(small, 1.0, x)
    ep, em, et = jnp.exp(tau + x), jnp.exp(tau - x), jnp.exp(tau)
    C = jnp.where(s2 >= 0.0, 0.5 * (ep + em), et * jnp.cos(x))
    S_big = jnp.where(s2 >= 0.0, 0.5 * (ep - em) / xs, et * jnp.sin(x) / xs)
    S_small = et * (1.0 + jnp.sign(s2) * x * x / 6.0 + x ** 4 / 120.0)
    S = jnp.where(small, S_small, S_big)
    row0 = jnp.stack([C + S * (a - tau), S * b], axis=-1)
    row1 = jnp.stack([S * c, C + S * (d - tau)], axis=-1)
    return jnp.stack([row0, row1], axis=-2)


def _ordered_product(P):
    """The twin of ``dmipy_analytic.exchange._ordered_product``: the same balanced tree of pairwise products."""
    while P.shape[-3] > 1:
        if P.shape[-3] % 2:
            eye = jnp.broadcast_to(jnp.eye(2, dtype=P.dtype), P.shape[:-3] + (1, 2, 2))
            P = jnp.concatenate([P, eye], axis=-3)
        P = P[..., 1::2, :, :] @ P[..., 0::2, :, :]
    return P[..., 0, :, :]


def _rates(T):
    if T is None:
        return jnp.zeros(2)
    T = jnp.broadcast_to(jnp.asarray(T), (2,))
    return jnp.where(jnp.isfinite(T), 1.0 / jnp.where(jnp.isfinite(T), T, 1.0), 0.0)


def karger_waveform(G, dt, D, K, M0, *, stored=None, T2=None, T1=None, sub_steps=1, gamma=GAMMA):
    """The twin of :func:`dmipy_analytic.exchange.karger_waveform`: the same Magnus steps and the same tree of products.
    ``G``'s shape and ``stored`` are static (they set the steps)."""
    G = jnp.asarray(G)
    squeeze = G.ndim < 3
    if G.ndim == 1:
        G = G[None, :, None]
    elif G.ndim == 2:
        G = G[None]
    n_t = G.shape[1]
    sub = int(sub_steps)
    st = jnp.zeros(n_t, dtype=bool) if stored is None else jnp.asarray(stored, dtype=bool).reshape(-1)
    h = dt / sub
    D, K, M0 = jnp.asarray(D), jnp.asarray(K), jnp.asarray(M0)
    g = jnp.repeat(G[:, :-1, :], sub, axis=1)
    st = jnp.repeat(st[:-1], sub)
    R = jnp.where(st[:, None], _rates(T1)[None, :], _rates(T2)[None, :])
    v = gamma * g
    q1 = jnp.cumsum(v, axis=1) * h
    q0 = q1 - v * h
    alpha = jnp.sum(q0 * q0, axis=-1)
    beta = 2.0 * jnp.sum(q0 * v, axis=-1)
    c = jnp.sum(v * v, axis=-1)
    I1 = alpha * h + beta * h ** 2 / 2.0 + c * h ** 3 / 3.0
    w2 = (h ** 3 / 12.0) * (beta + c * h)
    o00 = (K[0, 0] - R[:, 0]) * h - D[0] * I1
    o11 = (K[1, 1] - R[:, 1]) * h - D[1] * I1
    o01 = K[0, 1] * h + w2 * K[0, 1] * (D[1] - D[0])
    o10 = K[1, 0] * h + w2 * K[1, 0] * (D[0] - D[1])
    Om = jnp.stack([jnp.stack([o00, o01], axis=-1), jnp.stack([o10, o11], axis=-1)], axis=-2)
    P = _ordered_product(_expm2(Om))
    out = jnp.sum(P @ M0, axis=-1)
    return out[0] if squeeze else out


def fexi_narrow_pulse(q_filter, Delta_filter, t_m, q_detect, Delta_detect, D, K, M0):
    """The twin of :func:`dmipy_analytic.exchange.fexi_narrow_pulse`."""
    qf, tm, qd = jnp.broadcast_arrays(jnp.asarray(q_filter), jnp.asarray(t_m), jnp.asarray(q_detect))
    D, K, M0 = jnp.asarray(D), jnp.asarray(K), jnp.asarray(M0)
    Dm = jnp.diag(D)
    Mf = (K[None] - qf.reshape(-1)[:, None, None] ** 2 * Dm[None]) * Delta_filter
    Mx = K[None] * tm.reshape(-1)[:, None, None]
    Md = (K[None] - qd.reshape(-1)[:, None, None] ** 2 * Dm[None]) * Delta_detect
    S = jnp.sum(_expm2(Md) @ _expm2(Mx) @ _expm2(Mf) @ M0, axis=-1)
    return S.reshape(qf.shape)


def fexi_adc(t_m, adc_eq, sigma, axr):
    """The twin of :func:`dmipy_analytic.exchange.fexi_adc`."""
    return adc_eq * (1.0 - sigma * jnp.exp(-axr * jnp.asarray(t_m)))


def fexi_signal(b, t_m, filtered, adc_eq, sigma, axr):
    """The twin of :func:`dmipy_analytic.exchange.fexi_signal`."""
    b, t_m, filtered = jnp.broadcast_arrays(jnp.asarray(b), jnp.asarray(t_m), jnp.asarray(filtered, dtype=bool))
    adc = jnp.where(filtered, fexi_adc(t_m, adc_eq, sigma, axr), adc_eq)
    return jnp.exp(-b * adc)


def fexi_two_site(f, kappa, D, b_filter):
    """The twin of :func:`dmipy_analytic.exchange.fexi_two_site`, as a tuple of arrays."""
    D = jnp.asarray(D).reshape(-1)
    w = jnp.stack([f, 1.0 - f]) * jnp.exp(-b_filter * D)
    adc_eq = f * D[0] + (1.0 - f) * D[1]
    adc_0 = jnp.sum(w * D) / jnp.sum(w)
    return adc_eq, 1.0 - adc_0 / adc_eq, kappa / (1.0 - f)
