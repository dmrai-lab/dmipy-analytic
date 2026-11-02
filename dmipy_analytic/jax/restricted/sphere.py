"""The JAX twins of the forms of :mod:`dmipy_analytic.restricted.sphere`, each held to its numpy definition.

:func:`dmipy_analytic.restricted.sphere.sgp` has no twin: its terms are spherical Bessel functions and their
derivatives of orders up to ``n_functions - 1``, which ``jax.scipy.special`` does not provide.
"""
import jax
import jax.numpy as jnp

from ...constants import GAMMA
from ...restricted.matrix_method import sphere_neumann_roots

__all__ = ["sgp_long_time", "gaussian_phase", "gaussian_phase_waveform"]


def sgp_long_time(q, diameter):
    """The twin of :func:`dmipy_analytic.restricted.sphere.sgp_long_time`; the limit ``1`` is taken for
    ``2 pi q R <= 1e-7``."""
    radius = diameter / 2.0
    factor = 2.0 * jnp.pi * q * radius
    safe_factor = jnp.where(factor > 1e-7, factor, jnp.ones_like(factor))
    E_nonzero = (3.0 / safe_factor ** 2 *
                 (jnp.sin(safe_factor) / safe_factor - jnp.cos(safe_factor))) ** 2
    return jnp.where(factor > 1e-7, E_nonzero, jnp.ones_like(q))


def gaussian_phase(G, delta, Delta, diameter, diffusivity, n_roots=100, gamma=GAMMA):
    """The twin of :func:`dmipy_analytic.restricted.sphere.gaussian_phase`."""
    R = diameter / 2.0
    D = diffusivity
    roots = jnp.asarray(sphere_neumann_roots(1, n_roots))
    delta = jnp.asarray(delta)[..., None]
    Delta = jnp.asarray(Delta)[..., None]

    alpha = roots / R
    alpha2 = alpha ** 2
    alpha2D = alpha2 * D

    first_factor = -2.0 * (gamma * jnp.asarray(G)) ** 2 / D

    summands = (
        alpha ** (-4) / (alpha2 * R ** 2 - 2.0) *
        (
            2.0 * delta - (
                2.0
                + jnp.exp(-alpha2D * (Delta - delta))
                - 2.0 * jnp.exp(-alpha2D * delta)
                - 2.0 * jnp.exp(-alpha2D * Delta)
                + jnp.exp(-alpha2D * (Delta + delta))
            ) / alpha2D
        )
    )
    return jnp.exp(first_factor * jnp.sum(summands, axis=-1))


def gaussian_phase_waveform(G, dt, diameter, diffusivity, n_roots=100, gamma=GAMMA):
    """The twin of :func:`dmipy_analytic.restricted.sphere.gaussian_phase_waveform` for one waveform ``G`` of shape
    ``(n_t,)`` or ``(n_t, c)``: the causal recursion of each root is a ``jax.lax.scan``, and the components' phases
    add."""
    G = jnp.asarray(G)
    components = G[:, None] if G.ndim == 1 else G
    R = diameter / 2.0
    D = diffusivity

    mu_k = jnp.asarray(sphere_neumann_roots(1, n_roots))
    lam_k = (mu_k / R) ** 2
    B_k = 2.0 * (R / mu_k) ** 2 / (mu_k ** 2 - 2.0)

    def compute_I_k(lkd, G_t):
        decay_step = jnp.exp(-lkd * dt)

        def scan_fn(H_prev, G_n):
            H_n = H_prev * decay_step + G_n * dt
            return H_n, H_n

        _, H = jax.lax.scan(scan_fn, jnp.zeros(()), G_t)
        return 2.0 * jnp.dot(G_t * dt, H) - jnp.dot(G_t, G_t) * dt ** 2

    lkD = lam_k * D
    I_k_all = sum(
        jax.vmap(lambda lkd, c=c: compute_I_k(lkd, components[:, c]))(lkD)
        for c in range(components.shape[1]))

    phi = 0.5 * gamma ** 2 * jnp.sum(B_k * I_k_all)
    return jnp.exp(-phi)
