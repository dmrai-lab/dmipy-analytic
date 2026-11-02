"""The JAX twins of the forms of :mod:`dmipy_analytic.restricted.matrix_method`, each held to its numpy definition.

The twin propagates the mode coefficients of a pore through a sampled gradient waveform by a Strang split in a
``jax.lax.scan``: half a step of diffusion (diagonal), a step of precession (diagonal in the eigenbasis of the position
matrix ``B``), half a step of diffusion. It is differentiable in the diffusivity and the gradient. The pore's
eigenvalues ``lam`` and position matrix ``B`` are those of :class:`dmipy_analytic.restricted.matrix_method.MatrixPore`
(they depend on the shape, the size and the number of modes, and not on the diffusivity), computed once on the host;
``float64`` (``jax_enable_x64``) is the reference precision.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ...constants import GAMMA
from ...restricted.matrix_method import MatrixPore

__all__ = ["strang_signal", "matrix_restricted_signal", "matrix_restricted_batch"]


def strang_signal(g_axis, dt, diffusivity, lam, beta, U):
    """The magnitude of the echo of the Strang-split propagation of one projected gradient schedule, for the pore's
    eigenvalues ``lam`` and the eigendecomposition ``B = U diag(beta) U^T`` of its position matrix.

    Parameters
    ----------
    g_axis : array_like, shape (n_t,)
        The gradient along the measured axis at each step, T/m.
    dt : float
        The step, s.
    diffusivity : float
        The diffusivity, m^2/s.
    lam : array_like, shape (N,)
        The eigenvalues of the Laplacian of the pore, 1/m^2 (``MatrixPore.lam``).
    beta, U : array_like
        The eigenvalues, shape ``(N,)``, and the eigenvectors, shape ``(N, N)``, of the pore's position matrix
        (``numpy.linalg.eigh(MatrixPore.B)``).
    """
    half = jnp.exp(-0.5 * dt * diffusivity * lam)
    rot = GAMMA * dt * beta
    Ut = U.T
    cdtype = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
    c0 = jnp.zeros(lam.shape[0], dtype=cdtype).at[0].set(1.0 + 0.0j)

    def step(c, gk):
        c = half * c
        c = U @ (jnp.exp(-1j * gk * rot) * (Ut @ c))
        c = half * c
        return c, None

    cN, _ = jax.lax.scan(step, c0, g_axis)
    return jnp.abs(cN[0])


def _pore_modes(shape, size_m, n_modes):
    """``(lam, beta, U)`` of the pore on the host, as arrays of the active JAX precision."""
    pore = MatrixPore(shape, size_m, 1.0, n_modes=n_modes)                   # lam and B do not depend on D
    beta, U = np.linalg.eigh(pore.B)
    return jnp.asarray(pore.lam), jnp.asarray(beta), jnp.asarray(U)


def matrix_restricted_signal(shape, g_axis, dt, D, size_m, *, n_modes=None):
    """The twin of :func:`dmipy_analytic.restricted.matrix_method.matrix_restricted_signal`: the Strang-split signal
    of one projected schedule. ``shape``, ``size_m`` and ``n_modes`` are static; ``g_axis``, ``dt`` and ``D`` may be
    traced."""
    lam, beta, U = _pore_modes(shape, size_m, n_modes)
    return strang_signal(jnp.asarray(g_axis), dt, D, lam, beta, U)


def matrix_restricted_batch(shape, g_axes, dt, D, size_m, *, n_modes=None):
    """The twin of :func:`dmipy_analytic.restricted.matrix_method.matrix_restricted_batch`: ``vmap`` of
    :func:`matrix_restricted_signal` over the stack of schedules ``g_axes`` of shape ``(n_meas, n_t)``."""
    lam, beta, U = _pore_modes(shape, size_m, n_modes)
    return jax.vmap(lambda g: strang_signal(g, dt, D, lam, beta, U))(jnp.asarray(g_axes))
