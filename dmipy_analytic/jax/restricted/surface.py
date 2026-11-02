"""The JAX twins of the forms of :mod:`dmipy_analytic.restricted.surface`, each held to its numpy definition.

A twin propagates the mode coefficients step by step, one ``jax.scipy.linalg.expm`` of
``-dt (D Lambda + i gamma g_k . B)`` per step in a ``jax.lax.scan`` (the numpy form takes one per run of constant
gradient): the same ordered product, exact for a piecewise-constant waveform, differentiable in the diffusivity and the
gradient. An ideal 180 conjugates the coefficients before its step. The eigenvalues and position matrices are those of
:class:`~dmipy_analytic.restricted.matrix_method.PeriodicPore` (and of the plate pore for a capped tube), computed once
on the host: the diameter or the outline, the order ``n_modes``, ``length``, ``refocus`` and the number of steps are
static, and an order left to the truncation rule is computed on the host from a concrete waveform. The axial moment of
an infinite tube's waveform is not checked here (the waveform may be traced); the numpy form refuses a waveform that
leaves one. ``float64`` (``jax_enable_x64``) is the reference precision.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ...constants import GAMMA
from ...restricted import surface as _surface
from ...restricted.matrix_method import MatrixPore, PeriodicPore, _refocus_steps

__all__ = ["circle_signal", "outline_signal", "axial_signal"]


def _flip_mask(refocus, n_t):
    """``(n_t + 1,)`` booleans: a 180 before step ``k`` (the last entry: after the last step)."""
    mask = np.zeros(n_t + 1, bool)
    mask[sorted(_refocus_steps(refocus, n_t))] = True
    return mask


def _propagate(lam, B, g, dt, diffusivity, mask):
    """The ground mode's coefficient after the per-step product from it, for the eigenvalues ``lam`` (N,), the
    position matrices ``B`` (n_axes, N, N) and the gradient ``g`` (n_t, n_axes)."""
    cdtype = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
    lam = jnp.asarray(lam)
    B = jnp.asarray(B)
    decay = diffusivity * jnp.diag(lam)
    c0 = jnp.zeros(lam.shape[0], dtype=cdtype).at[0].set(1.0)

    def step(c, inputs):
        gk, flip = inputs
        c = jnp.where(flip, jnp.conj(c), c)
        c = jax.scipy.linalg.expm(-dt * (decay + 1j * GAMMA * jnp.tensordot(gk, B, axes=1))) @ c
        return c, None

    c, _ = jax.lax.scan(step, c0, (jnp.asarray(g), jnp.asarray(mask[:-1])))
    return jnp.where(mask[-1], jnp.conj(c), c)[0]


def axial_signal(g_z, dt, diffusivity, *, length=None, refocus=(), axial_modes=None):
    """The twin of :func:`dmipy_analytic.restricted.surface.axial_signal`: free diffusion along an infinite tube,
    ``exp(-gamma^2 D int F_z^2 dt)`` integrated exactly over each step, or the plate pore of a capped tube."""
    g_z = jnp.asarray(g_z)
    mask = _flip_mask(refocus, g_z.shape[0])
    if length is not None:
        plates = MatrixPore("plane", float(length), 1.0, n_modes=axial_modes)       # lam and B do not depend on D
        return _propagate(plates.lam, plates.B[None], g_z[:, None], dt, diffusivity, mask)
    sign = (-1.0) ** np.cumsum(mask[:-1])
    eff = g_z * sign
    F = jnp.concatenate([jnp.zeros(1, eff.dtype), jnp.cumsum(eff)[:-1] * dt])
    exponent = dt * jnp.sum(F * F + F * eff * dt + eff * eff * dt * dt / 3.0)
    return jnp.exp(-GAMMA ** 2 * diffusivity * exponent)


def _tube(pore, g, dt, diffusivity, length, refocus, axial_modes):
    g = jnp.asarray(g)
    mask = _flip_mask(refocus, g.shape[0])
    across = _propagate(pore.lam, pore.B, g[:, :2], dt, diffusivity, mask)
    return across * axial_signal(g[:, 2], dt, diffusivity, length=length, refocus=refocus, axial_modes=axial_modes)


def circle_signal(g, dt, diameter, diffusivity, *, length=None, refocus=(), n_modes=None, axial_modes=None):
    """The twin of :func:`dmipy_analytic.restricted.surface.circle_signal`. ``g``, ``dt`` and ``diffusivity`` may be
    traced; with ``n_modes=None`` the order is the truncation rule's for the concrete ``g``."""
    if n_modes is None:
        n_modes = _surface.circle_order(np.asarray(g), float(dt), diameter)
    pore = PeriodicPore.circle(diameter, 1.0, n_modes=n_modes)
    return _tube(pore, g, dt, diffusivity, length, refocus, axial_modes)


def outline_signal(g, dt, outline, diffusivity, *, n_modes, length=None, refocus=(), axial_modes=None):
    """The twin of :func:`dmipy_analytic.restricted.surface.outline_signal`. ``g``, ``dt`` and ``diffusivity`` may be
    traced; ``outline`` and ``n_modes`` are static."""
    pore = PeriodicPore.polygon(np.asarray(outline, np.float64), 1.0, n_modes=n_modes)
    return _tube(pore, g, dt, diffusivity, length, refocus, axial_modes)
