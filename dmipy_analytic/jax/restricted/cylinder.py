"""The JAX twins of the forms of :mod:`dmipy_analytic.restricted.cylinder`, each held to its numpy definition.

The Bessel functions come from ``jax.scipy.special.bessel_jn`` (integer orders); the derivative of order ``m`` is
``(J_{m-1} - J_{m+1}) / 2``. The Callaghan series needs ``float64``.
"""
import jax
import jax.numpy as jnp

from ...constants import GAMMA
from ...restricted.cylinder import neumann_roots_table
from ...restricted.matrix_method import cylinder_neumann_roots
from . import plane

__all__ = ["sgp_long_time", "sgp", "gaussian_phase", "gaussian_phase_waveform", "gaussian_phase_cosine",
           "geometry_factor", "gaussian_phase_gamma_lm", "gaussian_phase_exponent_coefficients",
           "capped_sgp_long_time", "capped_sgp"]


def sgp_long_time(q, diameter):
    """The twin of :func:`dmipy_analytic.restricted.cylinder.sgp_long_time`; the limit ``1`` is taken for
    ``2 pi q R <= 1e-10``."""
    radius = diameter / 2.0
    q_arg = 2.0 * jnp.pi * q * radius
    safe_q_arg = jnp.where(q_arg > 1e-10, q_arg, jnp.ones_like(q_arg))
    j1_vals = jax.scipy.special.bessel_jn(safe_q_arg, v=1)[-1]
    E_perp_nonzero = (2.0 * j1_vals / safe_q_arg) ** 2
    return jnp.where(q_arg > 1e-10, E_perp_nonzero, jnp.ones_like(q_arg))


def sgp(q, tau, diameter, diffusivity, n_roots=20, n_functions=50):
    """The twin of :func:`dmipy_analytic.restricted.cylinder.sgp`: the Bessel functions of every order of the series
    in one call, the series summed over its radial modes per order. Needs ``float64``."""
    alpha_jax = jnp.array(neumann_roots_table(n_roots, n_functions))
    max_order = n_functions + 1
    D = diffusivity

    q = jnp.atleast_1d(jnp.asarray(q))
    tau = jnp.broadcast_to(jnp.asarray(tau), q.shape)
    radius = diameter / 2.0
    q_arg = 2.0 * jnp.pi * q * radius
    q2 = q_arg ** 2

    safe_q_arg = jnp.where(q_arg > 1e-10, q_arg, jnp.ones_like(q_arg))
    J_all = jax.scipy.special.bessel_jn(safe_q_arg, v=max_order)

    alpha2_0 = alpha_jax[:, 0] ** 2
    J1 = J_all[1]
    exp_0 = jnp.exp(-alpha2_0[:, None] * D * tau[None, :] / radius ** 2)
    numer_0 = 4.0 * exp_0 * q2[None, :] * J1[None, :] ** 2
    denom_0_raw = (q2[None, :] - alpha2_0[:, None]) ** 2
    denom_0 = jnp.where(denom_0_raw > 1e-60, denom_0_raw, jnp.ones_like(denom_0_raw))
    res = jnp.sum(numer_0 / denom_0, axis=0)

    for m in range(1, n_functions):
        alpha2_m = alpha_jax[:, m] ** 2
        Jderiv_m = (J_all[m - 1] - J_all[m + 1]) / 2.0
        q_Jd_sq = (q_arg * Jderiv_m) ** 2

        exp_m = jnp.exp(-alpha2_m[:, None] * D * tau[None, :] / radius ** 2)
        coeff = 8.0 * alpha2_m[:, None] / (alpha2_m[:, None] - float(m) ** 2)
        numer_m = coeff * exp_m * q_Jd_sq[None, :]
        denom_m_raw = (q2[None, :] - alpha2_m[:, None]) ** 2
        denom_m = jnp.where(denom_m_raw > 1e-60, denom_m_raw, jnp.ones_like(denom_m_raw))
        res = res + jnp.sum(numer_m / denom_m, axis=0)

    return jnp.where(q > 1e-10, res, jnp.ones_like(q))


def gaussian_phase(G, delta, Delta, diameter, diffusivity, n_roots=100, gamma=GAMMA):
    """The twin of :func:`dmipy_analytic.restricted.cylinder.gaussian_phase`: the roots run along the first axis and
    the measurements along the second. Needs ``float64`` (the terms fall as ``alpha^-6``)."""
    D = diffusivity
    radius = diameter / 2.0
    roots_jax = jnp.asarray(cylinder_neumann_roots(1, n_roots))
    alpha = roots_jax / radius
    alpha2 = alpha ** 2
    alpha2D = alpha2 * D

    a2D = alpha2D[:, None]
    d = jnp.asarray(delta)[None, ...]
    D_ = jnp.asarray(Delta)[None, ...]

    numer = (
        2.0 * a2D * d - 2.0 +
        2.0 * jnp.exp(-a2D * d) +
        2.0 * jnp.exp(-a2D * D_) -
        jnp.exp(-a2D * (D_ - d)) -
        jnp.exp(-a2D * (D_ + d))
    )
    denom = D ** 2 * alpha2[:, None] ** 3 * (radius ** 2 * alpha2[:, None] - 1.0)
    sum_over_roots = jnp.sum(numer / denom, axis=0)

    first_factor = -2.0 * (jnp.asarray(G) * gamma) ** 2
    return jnp.exp(first_factor * sum_over_roots)


def gaussian_phase_waveform(G, dt, diameter, diffusivity, n_roots=100, gamma=GAMMA):
    """The twin of :func:`dmipy_analytic.restricted.cylinder.gaussian_phase_waveform`: the causal recursion of each
    root is a ``jax.lax.scan``. Needs ``float64``."""
    G = jnp.asarray(G)
    R = diameter / 2.0
    D = diffusivity
    mu_n = jnp.asarray(cylinder_neumann_roots(1, n_roots))
    lam_n = (mu_n / R) ** 2
    B_n = 2.0 * (R / mu_n) ** 2 / (mu_n ** 2 - 1.0)

    def compute_I_k(lkd):
        decay_step = jnp.exp(-lkd * dt)

        def scan_fn(H_prev, G_n):
            H_n = H_prev * decay_step + G_n * dt
            return H_n, H_n

        _, H = jax.lax.scan(scan_fn, jnp.zeros(()), G)
        return 2.0 * jnp.dot(G * dt, H) - jnp.dot(G, G) * dt ** 2

    I_k = jax.vmap(compute_I_k)(lam_n * D)
    phi = 0.5 * gamma ** 2 * jnp.sum(B_n * I_k)
    return jnp.exp(-phi)


def gaussian_phase_cosine(G, omega, sigma, diameter, diffusivity, n_roots=100, gamma=GAMMA):
    """The twin of :func:`dmipy_analytic.restricted.cylinder.gaussian_phase_cosine`."""
    R = diameter / 2.0
    D = diffusivity
    mu_n = jnp.asarray(cylinder_neumann_roots(1, n_roots))
    lam_n = (mu_n / R) ** 2
    B_n = 2.0 * (R / mu_n) ** 2 / (mu_n ** 2 - 1.0)

    alpha = lam_n * D
    denom = alpha ** 2 + omega ** 2

    c = jnp.cos(omega * sigma)
    s = jnp.sin(omega * sigma)
    e_asm = jnp.exp(-alpha * sigma)

    F_s = (alpha * (1.0 - c * e_asm) + omega * s * e_asm) / denom
    I_n = (2.0 / denom) * (
        alpha * (sigma / 2.0 + s * c / (2.0 * omega))
        - alpha * F_s
        + s ** 2 / 2.0
    )
    phi = 0.5 * gamma ** 2 * G ** 2 * jnp.sum(B_n * I_n)
    return jnp.exp(-phi)


def geometry_factor(diameter, diffusivity, n_roots=100, gamma=GAMMA):
    """The twin of :func:`dmipy_analytic.restricted.cylinder.geometry_factor`."""
    R = diameter / 2.0
    mu_n = jnp.asarray(cylinder_neumann_roots(1, n_roots))
    B_k = 2.0 * (R / mu_n) ** 2 / (mu_n ** 2 - 1.0)
    alpha_k = (mu_n / R) ** 2 * diffusivity
    return gamma ** 2 * jnp.sum(B_k / alpha_k)


def _real_harmonics_l2(axis):
    x, y, z = axis[0], axis[1], axis[2]
    Y2m2 = jnp.sqrt(15.0 / (4.0 * jnp.pi)) * x * y
    Y2m1 = jnp.sqrt(15.0 / (4.0 * jnp.pi)) * y * z
    Y20 = jnp.sqrt(5.0 / (16.0 * jnp.pi)) * (2.0 * z ** 2 - x ** 2 - y ** 2)
    Y21 = jnp.sqrt(15.0 / (4.0 * jnp.pi)) * x * z
    Y22 = jnp.sqrt(15.0 / (16.0 * jnp.pi)) * (x ** 2 - y ** 2)
    return jnp.stack([Y2m2, Y2m1, Y20, Y21, Y22])


def gaussian_phase_gamma_lm(gamma_lm, b_par, axis, diameter, diffusivity_perp, diffusivity_par, n_roots=100,
                            gamma=GAMMA):
    """The twin of :func:`dmipy_analytic.restricted.cylinder.gaussian_phase_gamma_lm`."""
    C_geom = geometry_factor(diameter, diffusivity_perp, n_roots, gamma)
    gamma_lm = jnp.asarray(gamma_lm)
    int_G2_dt = gamma_lm[:, 0] * jnp.sqrt(4.0 * jnp.pi)
    Gamma2m = gamma_lm[:, 1:]
    Y2m_mu = _real_harmonics_l2(jnp.asarray(axis))
    phi_perp = C_geom * (
        (2.0 / 3.0) * int_G2_dt
        - (8.0 * jnp.pi / 15.0) * Gamma2m.dot(Y2m_mu)
    )
    phi_par = b_par * diffusivity_par
    return jnp.exp(-(phi_perp + phi_par))


def gaussian_phase_exponent_coefficients(gamma_lm, diameter, diffusivity, n_roots=100, gamma=GAMMA):
    """The twin of :func:`dmipy_analytic.restricted.cylinder.gaussian_phase_exponent_coefficients`."""
    C_geom = geometry_factor(diameter, diffusivity, n_roots, gamma)
    gamma_lm = jnp.asarray(gamma_lm)
    int_G2_dt = gamma_lm[:, 0] * jnp.sqrt(4.0 * jnp.pi)
    c0 = C_geom * (2.0 / 3.0) * int_G2_dt
    c2m = -C_geom * (8.0 * jnp.pi / 15.0) * gamma_lm[:, 1:]
    return c0, c2m


def capped_sgp_long_time(q_perp, q_par, diameter, length):
    """The twin of :func:`dmipy_analytic.restricted.cylinder.capped_sgp_long_time`."""
    return plane.sgp_long_time(q_par, length) * sgp_long_time(q_perp, diameter)


def capped_sgp(q_perp, q_par, tau, diameter, length, diffusivity, n_roots_cylinder=20, n_functions_cylinder=50,
               n_roots_plane=40):
    """The twin of :func:`dmipy_analytic.restricted.cylinder.capped_sgp`."""
    return (plane.sgp(q_par, tau, length, diffusivity, n_roots_plane)
            * sgp(q_perp, tau, diameter, diffusivity, n_roots_cylinder, n_functions_cylinder))
