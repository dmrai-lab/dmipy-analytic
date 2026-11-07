"""The JAX twins of the forms of :mod:`dmipy_analytic.restricted.sheath`, each held to its numpy definition.

A twin propagates the mode coefficients step by step, one ``jax.scipy.linalg.expm`` per step in a ``jax.lax.scan``
(:mod:`dmipy_analytic.jax.restricted.surface`'s propagation): the same ordered product as the numpy form's, exact for a
piecewise-constant waveform, differentiable in the diffusivity and the gradient. The modes are those of
:class:`~dmipy_analytic.restricted.matrix_method.AnnulusPore` and
:class:`~dmipy_analytic.restricted.matrix_method.PeriodicPore`, computed once on the host: the diameters, the orders,
``n_radii``, ``length``, ``refocus`` and the number of steps are static. The annulus's direction across the axis is that
of the waveform's largest sample; the numpy form refuses a waveform that leaves it, the twin does not check (the
waveform may be traced). ``float64`` (``jax_enable_x64``) is the reference precision.
"""
import jax.numpy as jnp
import numpy as np

from ...restricted import sheath as _sheath
from ...restricted import surface as _surface
from ...restricted.matrix_method import AnnulusPore, PeriodicPore
from .surface import _flip_mask, _propagate, axial_signal

__all__ = ["annulus_signal", "lamellae_signal"]


def annulus_signal(g, dt, outer_diameter, inner_diameter, diffusivity, *, length=None, refocus=(), n_modes=None,
                   axial_modes=None):
    """The twin of :func:`dmipy_analytic.restricted.sheath.annulus_signal`. ``g``, ``dt`` and ``diffusivity`` may be
    traced."""
    g = jnp.asarray(g)
    outer, inner = _sheath._diameters(outer_diameter, inner_diameter)
    pore = AnnulusPore(outer, inner, 1.0, n_modes=n_modes)                 # lam and B do not depend on D
    norm = jnp.hypot(g[:, 0], g[:, 1])
    k = jnp.argmax(norm)
    u = g[k, :2] / jnp.where(norm[k] > 0, norm[k], 1.0)
    mask = _flip_mask(refocus, g.shape[0])
    across = _propagate(pore.lam, pore.B[None], (g[:, :2] @ u)[:, None], dt, diffusivity, mask)
    return across * axial_signal(g[:, 2], dt, diffusivity, length=length, refocus=refocus, axial_modes=axial_modes)


def lamellae_signal(g, dt, outer_diameter, inner_diameter, diffusivity, *, n_radii=_sheath.LAMELLAE_RADII, length=None,
                    refocus=(), n_modes=None, axial_modes=None):
    """The twin of :func:`dmipy_analytic.restricted.sheath.lamellae_signal`. ``g``, ``dt`` and ``diffusivity`` may be
    traced; with ``n_modes=None`` the order is the truncation rule's at the outer diameter for the concrete ``g``."""
    outer, _ = _sheath._diameters(outer_diameter, inner_diameter)
    if n_modes is None:
        n_modes = _surface.circle_order(np.asarray(g), float(dt), outer)
    g = jnp.asarray(g)
    mask = _flip_mask(refocus, g.shape[0])
    r, w = _sheath.lamellae_radii(outer_diameter, inner_diameter, n_radii)
    across = 0.0
    for ri, wi in zip(r, w):
        pore = PeriodicPore.circle(2.0 * ri, 1.0, n_modes=n_modes)
        across = across + wi * _propagate(pore.lam, pore.B, g[:, :2], dt, diffusivity, mask)
    return across * axial_signal(g[:, 2], dt, diffusivity, length=length, refocus=refocus, axial_modes=axial_modes)
