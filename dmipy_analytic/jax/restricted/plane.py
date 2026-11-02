"""The JAX twins of the forms of :mod:`dmipy_analytic.restricted.plane`, each held to its numpy definition."""
import jax.numpy as jnp
import numpy as np
from jax import vmap

from ...constants import GAMMA

__all__ = ["sgp_long_time", "sgp", "gaussian_phase"]


def sgp_long_time(q, diameter):
    """The twin of :func:`dmipy_analytic.restricted.plane.sgp_long_time`."""
    arg = 2.0 * jnp.pi * q * diameter
    safe = jnp.where(arg == 0.0, 1.0, arg)
    return jnp.where(q != 0, 2.0 * (1.0 - jnp.cos(arg)) / safe ** 2, 1.0)


def sgp(q, tau, diameter, diffusivity, n_roots=40):
    """The twin of :func:`dmipy_analytic.restricted.plane.sgp`: the cosine modes ``xi_n = n pi`` and the sine modes
    ``zeta_n = n pi + pi/2`` are summed with ``vmap``; a term with a vanishing denominator is dropped."""
    xi = jnp.asarray(np.arange(n_roots) * np.pi, dtype=float)
    zeta = jnp.asarray(np.arange(n_roots) * np.pi + np.pi / 2.0, dtype=float)
    radius = diameter / 2.0
    qa = 2.0 * jnp.pi * q * radius
    qa2 = qa ** 2

    def xi_term(xi_n):
        xi_n2 = xi_n ** 2
        div = jnp.where(xi_n == 0.0, 1.0, jnp.sin(2 * xi_n) / 2 * xi_n)
        num = (qa * jnp.sin(qa) * jnp.cos(xi_n)
               - xi_n * jnp.cos(qa) * jnp.sin(xi_n)) ** 2
        t = (2 * jnp.exp(-xi_n2 * diffusivity * tau / radius ** 2) / (1 + div)
             * num / (qa2 - xi_n2) ** 2)
        return jnp.where(jnp.isfinite(t), t, 0.0)

    def zeta_term(zeta_m):
        zeta_m2 = zeta_m ** 2
        div = jnp.sin(2 * zeta_m) / (2 * zeta_m)
        num = (qa * jnp.cos(qa) * jnp.sin(zeta_m)
               - zeta_m * jnp.sin(qa) * jnp.cos(zeta_m)) ** 2
        t = (2 * jnp.exp(-zeta_m2 * diffusivity * tau / radius ** 2) / (1 - div)
             * num / (qa2 - zeta_m2) ** 2)
        return jnp.where(jnp.isfinite(t), t, 0.0)

    res = vmap(xi_term)(xi).sum(0) + vmap(zeta_term)(zeta).sum(0)
    return jnp.where(q != 0, res, 1.0)


def gaussian_phase(G, delta, Delta, diameter, diffusivity, n_modes=100, gamma=GAMMA):
    """The twin of :func:`dmipy_analytic.restricted.plane.gaussian_phase`."""
    D = diffusivity
    L = diameter
    k = jnp.asarray(np.arange(1, 2 * int(n_modes), 2), dtype=float)
    lam_D = D * (k * jnp.pi / L) ** 2
    delta = jnp.asarray(delta)[..., None]
    Delta = jnp.asarray(Delta)[..., None]
    bracket = (2 * lam_D * delta - 2 + 2 * jnp.exp(-lam_D * delta)
               + 2 * jnp.exp(-lam_D * Delta)
               - jnp.exp(-lam_D * (Delta - delta))
               - jnp.exp(-lam_D * (Delta + delta))) / lam_D ** 2
    weight = 8.0 * L ** 2 / (k * jnp.pi) ** 4
    phi2 = (gamma * jnp.asarray(G)) ** 2 * jnp.sum(weight * bracket, axis=-1)
    return jnp.exp(-phi2)
