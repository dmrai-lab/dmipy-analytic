"""The JAX twin of :mod:`dmipy_analytic.bloch_torrey`, held to its numpy definition.

The twin runs the same Strang split as :meth:`dmipy_analytic.bloch_torrey.BlochTorrey1D.signal` in a ``jax.lax.scan``:
half a step of diffusion in the eigenbasis of the symmetrised operator, the potential applied exactly, half a step of
diffusion, with an ideal 180 between two scans conjugating the magnetisation. It is differentiable in the gradient.
The operator (the eigendecomposition of the layers' diffusion, the cell positions, the relaxation and off-resonance of
each cell) is the numpy model's, built once on the host; ``float64`` (``jax_enable_x64``) is the reference precision.
"""
import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["signal"]


def signal(model, G, dt, *, refocus=(), sub_steps=1):
    """The total complex signal of ``model`` (a :class:`dmipy_analytic.bloch_torrey.BlochTorrey1D`) after the physical
    gradient ``G`` (n_t,) along ``x`` (T/m per step of ``dt`` s), with ideal 180s before the step indices ``refocus``
    and ``sub_steps`` Strang steps per step of ``G``: :meth:`~dmipy_analytic.bloch_torrey.BlochTorrey1D.signal` for one
    waveform, in ``jax.numpy``."""
    sub = int(sub_steps)
    if sub < 1:
        raise ValueError(f"sub_steps is a positive integer, not {sub_steps!r}")
    G = jnp.asarray(G)
    n_t = G.shape[0]
    flips = sorted(int(k) for k in refocus)
    if any(k < 0 or k > n_t for k in flips) or len(set(flips)) != len(flips):
        raise ValueError(f"refocus holds distinct step indices in [0, {n_t}], not {tuple(refocus)!r}")
    tau = float(dt) / sub
    half = jnp.exp(0.5 * tau * jnp.asarray(model.lam))
    U = jnp.asarray(model.U)
    Ut = U.T
    x = jnp.asarray(model.x)
    base = jnp.asarray(model.R2_cells) + 1j * jnp.asarray(model.omega_cells)
    gamma = float(model.gamma)
    sqrt_h = jnp.sqrt(jnp.asarray(model.h))
    c = Ut @ (sqrt_h * jnp.asarray(model.m0)).astype(base.dtype)

    def step(c, gk):
        V = jnp.exp(-tau * (base + 1j * gamma * gk * x))

        def one(c, _):
            return half * (Ut @ (V * (U @ (half * c)))), None

        c, _ = jax.lax.scan(one, c, None, length=sub)
        return c, None

    bounds = [0] + [k for k in flips if 0 < k < n_t] + [n_t]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a in flips:
            c = Ut @ jnp.conj(U @ c)
        c, _ = jax.lax.scan(step, c, G[a:b])
    y = U @ c
    if n_t in flips:
        y = jnp.conj(y)
    return jnp.sum(y * sqrt_h) / float(np.sum(model.h * model.m0))
