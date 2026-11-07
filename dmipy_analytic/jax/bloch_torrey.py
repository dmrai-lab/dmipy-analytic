"""The JAX twin of :mod:`dmipy_analytic.bloch_torrey`, held to its numpy definition.

The twin runs the same Strang split as :meth:`dmipy_analytic.bloch_torrey.BlochTorrey1D.signal` in a ``jax.lax.scan``:
half a step of diffusion in the eigenbasis of the symmetrised operator, the potential applied exactly, half a step of
diffusion, with an ideal 180 between two scans conjugating the magnetisation. It is differentiable in the gradient.
The operator (the eigendecomposition of the layers' diffusion and the faces' bound pools on each interval of the
permeability, each state's position, relaxation and off-resonance) is the numpy model's, built once on the host; at a
permeability breakpoint the state is carried into the next interval's eigenbasis between two scans, as the 180 is
applied. ``float64`` (``jax_enable_x64``) is the reference precision.
"""
import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["signal"]


def signal(model, G, dt, *, refocus=(), sub_steps=1, pool="total"):
    """The complex signal of ``model`` (a :class:`dmipy_analytic.bloch_torrey.BlochTorrey1D`) after the physical
    gradient ``G`` (n_t,) along ``x`` (T/m per step of ``dt`` s), with ideal 180s before the step indices ``refocus``
    and ``sub_steps`` Strang steps per step of ``G``, read from ``pool`` (``"total"``, ``"free"`` or ``"bound"``):
    :meth:`~dmipy_analytic.bloch_torrey.BlochTorrey1D.signal` for one waveform, in ``jax.numpy``. Every permeability
    breakpoint of ``model`` lies on a boundary of the steps of ``G`` (``t = k dt``); one inside a step is refused."""
    sub = int(sub_steps)
    if sub < 1:
        raise ValueError(f"sub_steps is a positive integer, not {sub_steps!r}")
    if pool not in ("total", "free", "bound"):
        raise ValueError(f"pool is 'total', 'free' or 'bound', not {pool!r}")
    G = jnp.asarray(G)
    n_t = G.shape[0]
    flips = sorted(int(k) for k in refocus)
    if any(k < 0 or k > n_t for k in flips) or len(set(flips)) != len(flips):
        raise ValueError(f"refocus holds distinct step indices in [0, {n_t}], not {tuple(refocus)!r}")
    tau = float(dt) / sub
    steps = model.breakpoint_steps(float(dt))                           # the step of G at whose start each interval begins
    x = jnp.asarray(model.x_state)
    base = jnp.asarray(model.R2_state) + 1j * jnp.asarray(model.omega_state)
    gamma = float(model.gamma)
    sqrt_w = jnp.asarray(model.sqrt_w)
    keep = np.ones(len(model.free), bool) if pool == "total" else (model.free if pool == "free" else ~model.free)

    def scan_of(piece):
        """The scan over steps of ``G`` in the eigenbasis of the interval ``piece``."""
        half = jnp.exp(0.5 * tau * jnp.asarray(model.lam[piece]))
        U = jnp.asarray(model.U[piece])
        Ut = U.T

        def step(c, gk):
            V = jnp.exp(-tau * (base + 1j * gamma * gk * x))

            def one(c, _):
                return half * (Ut @ (V * (U @ (half * c)))), None

            c, _ = jax.lax.scan(one, c, None, length=sub)
            return c, None
        return step, U

    piece = 0
    step, U = scan_of(piece)
    c = U.T @ (jnp.asarray(model.mass0) / sqrt_w).astype(base.dtype)
    bounds = sorted({0, n_t} | {k for k in flips if 0 < k < n_t} | {k for k in steps if 0 < k < n_t})
    for a, b in zip(bounds[:-1], bounds[1:]):
        while piece < len(steps) and steps[piece] == a:                 # the next interval: carry the state over
            y = U @ c
            piece += 1
            step, U = scan_of(piece)
            c = U.T @ y
        if a in flips:
            c = U.T @ jnp.conj(U @ c)
        c, _ = jax.lax.scan(step, c, G[a:b])
    y = U @ c
    if n_t in flips:
        y = jnp.conj(y)
    return jnp.sum(jnp.where(jnp.asarray(keep), y * sqrt_w, 0.0)) / float(np.sum(model.mass0))
