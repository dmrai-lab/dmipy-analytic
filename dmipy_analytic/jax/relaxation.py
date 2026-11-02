"""The JAX twins of the forms of :mod:`dmipy_analytic.relaxation`, each held to its numpy definition.

The Gamma-averaged surface relaxation has no Bessel function of the second kind in JAX; its twin evaluates the same
integral ``int x^(a-1) exp(-beta x - c/x) dx / Gamma(a)`` by generalised Gauss-Laguerre quadrature, with the nodes
and weights of the (static) shape computed once with numpy and scipy when the twin is traced.
"""
import jax.numpy as jnp

from .._extras import need

__all__ = [
    "transverse_decay", "longitudinal_decay", "surface_relaxation", "surface_to_volume",
    "surface_relaxation_gamma_cylinders", "surface_relaxation_gamma_spheres",
    "exterior_surface_to_volume",
]

_SURFACE_TO_VOLUME_OVER_DIAMETER = {"sphere": 6.0, "cylinder": 4.0, "plane": 2.0}
_EXTERIOR_GEOMETRY = {"sphere": (6.0, 3), "cylinder": (4.0, 2), "plane": (2.0, 1)}


def transverse_decay(tau, T2):
    """The twin of :func:`dmipy_analytic.relaxation.transverse_decay`."""
    return jnp.exp(-tau / T2)


def longitudinal_decay(tau, T1):
    """The twin of :func:`dmipy_analytic.relaxation.longitudinal_decay`."""
    return jnp.exp(-tau / T1)


def surface_to_volume(diameter, geometry):
    """The twin of :func:`dmipy_analytic.relaxation.surface_to_volume`."""
    try:
        coefficient = _SURFACE_TO_VOLUME_OVER_DIAMETER[geometry]
    except KeyError:
        raise ValueError("geometry must be one of {}; got {!r}".format(sorted(_SURFACE_TO_VOLUME_OVER_DIAMETER),
                                                                       geometry))
    return coefficient / diameter


def surface_relaxation(rho, surface_to_volume, tau):
    """The twin of :func:`dmipy_analytic.relaxation.surface_relaxation`."""
    return jnp.exp(-rho * surface_to_volume * tau)


def _gamma_average_of_surface_relaxation(a, beta, c, n_nodes):
    """``E_d[exp(-c/d)]`` for ``d ~ Gamma(a, rate beta)`` by ``n_nodes``-point generalised Gauss-Laguerre quadrature
    of weight ``x^(a-1) exp(-x)``: ``sum_k w_k exp(-c beta / x_k) / Gamma(a)``. ``a`` is a Python float."""
    special = need("scipy.special")
    nodes, weights = special.roots_genlaguerre(n_nodes, a - 1.0)
    x = jnp.asarray(nodes)
    w = jnp.asarray(weights)
    inv_gamma_a = 1.0 / float(special.gamma(a))
    c = jnp.asarray(c)
    xr = x.reshape((-1,) + (1,) * c.ndim)
    wr = w.reshape((-1,) + (1,) * c.ndim)
    arg = -(c[None, ...] * beta) / xr
    average = (wr * jnp.exp(arg)).sum(0) * inv_gamma_a
    return jnp.where(c <= 0, 1.0, average)


def surface_relaxation_gamma_cylinders(shape, scale, rho, tau, volume_weighted=True, n_nodes=64):
    """The twin of :func:`dmipy_analytic.relaxation.surface_relaxation_gamma_cylinders`, by ``n_nodes``-point
    Gauss-Laguerre quadrature. ``shape`` and ``volume_weighted`` are static; ``scale``, ``rho`` and ``tau`` may be
    traced."""
    a = shape + 2.0 if volume_weighted else shape
    return _gamma_average_of_surface_relaxation(a, 1.0 / scale, 4.0 * rho * tau, n_nodes)


def surface_relaxation_gamma_spheres(shape, scale, rho, tau, volume_weighted=True, n_nodes=64):
    """The twin of :func:`dmipy_analytic.relaxation.surface_relaxation_gamma_spheres`, by ``n_nodes``-point
    Gauss-Laguerre quadrature."""
    a = shape + 3.0 if volume_weighted else shape
    return _gamma_average_of_surface_relaxation(a, 1.0 / scale, 6.0 * rho * tau, n_nodes)


def exterior_surface_to_volume(f_cell, shape, scale, *, geometry):
    """The twin of :func:`dmipy_analytic.relaxation.exterior_surface_to_volume`."""
    try:
        coefficient, power = _EXTERIOR_GEOMETRY[geometry]
    except KeyError:
        raise ValueError("geometry must be one of {}; got {!r}".format(sorted(_EXTERIOR_GEOMETRY), geometry))
    return coefficient * f_cell / ((1.0 - f_cell) * (shape + power - 1.0) * scale)
