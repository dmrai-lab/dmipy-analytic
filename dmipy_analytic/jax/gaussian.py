"""The JAX twins of the forms of :mod:`dmipy_analytic.gaussian`, each held to its numpy definition.

Each twin has the name and the arguments of its numpy form. The contractions over the three components of a direction
are sums of products (``jnp.sum`` over the last axis), which XLA does not lower to a batched matrix multiplication, so
a twin evaluates a batch of one on every device.
"""
import jax
import jax.numpy as jnp

__all__ = [
    "ball", "stick", "stick_btensor", "zeppelin", "zeppelin_btensor", "tensor", "tensor_btensor",
    "time_dependent_diffusivity", "temporal_zeppelin", "stick_powder_average", "zeppelin_powder_average",
]


def ball(b, diffusivity):
    """The twin of :func:`dmipy_analytic.gaussian.ball`."""
    return jnp.exp(-b * diffusivity)


def stick(b, direction, axis, diffusivity):
    """The twin of :func:`dmipy_analytic.gaussian.stick`."""
    cos_theta = jnp.sum(direction * axis, axis=-1)
    return jnp.exp(-b * diffusivity * cos_theta ** 2)


def stick_btensor(btensor, axis, diffusivity):
    """The twin of :func:`dmipy_analytic.gaussian.stick_btensor`."""
    return jnp.exp(-diffusivity * jnp.einsum("i,mij,j->m", axis, btensor, axis))


def zeppelin(b, direction, axis, diffusivity_par, diffusivity_perp):
    """The twin of :func:`dmipy_analytic.gaussian.zeppelin`. The perpendicular part of the direction is
    ``|n|^2 - (n . u)^2``, floored at zero, so no outer-product projector is built."""
    mag_par = jnp.sum(direction * axis, axis=-1)
    mag_sq = jnp.sum(direction ** 2, axis=-1)
    mag_perp_sq = jnp.maximum(0.0, mag_sq - mag_par ** 2)
    return jnp.exp(-b * (diffusivity_par * mag_par ** 2 + diffusivity_perp * mag_perp_sq))


def zeppelin_btensor(btensor, axis, diffusivity_par, diffusivity_perp):
    """The twin of :func:`dmipy_analytic.gaussian.zeppelin_btensor`."""
    b_trace = jnp.trace(btensor, axis1=1, axis2=2)
    u_dot_B_u = jnp.einsum("i,mij,j->m", axis, btensor, axis)
    return jnp.exp(-(diffusivity_perp * b_trace + (diffusivity_par - diffusivity_perp) * u_dot_B_u))


def tensor(b, direction, diffusion_tensor):
    """The twin of :func:`dmipy_analytic.gaussian.tensor`."""
    return jnp.exp(-b * jnp.einsum("mi,ij,mj->m", direction, diffusion_tensor, direction))


def tensor_btensor(btensor, diffusion_tensor):
    """The twin of :func:`dmipy_analytic.gaussian.tensor_btensor`."""
    return jnp.exp(-jnp.einsum("mij,ij->m", btensor, diffusion_tensor))


def time_dependent_diffusivity(diffusivity_inf, A, delta, Delta):
    """The twin of :func:`dmipy_analytic.gaussian.time_dependent_diffusivity`."""
    return diffusivity_inf + A * (jnp.log(Delta / delta) + 1.5) / (Delta - delta / 3.0)


def temporal_zeppelin(b, direction, axis, diffusivity_par, diffusivity_inf, A, delta, Delta):
    """The twin of :func:`dmipy_analytic.gaussian.temporal_zeppelin`."""
    diffusivity_perp = time_dependent_diffusivity(diffusivity_inf, A, delta, Delta)
    n_dot_u = jnp.dot(direction, axis)
    return jnp.exp(-b * (diffusivity_perp + (diffusivity_par - diffusivity_perp) * n_dot_u ** 2))


def stick_powder_average(b, diffusivity):
    """The twin of :func:`dmipy_analytic.gaussian.stick_powder_average`. A product ``b D`` of at most ``1e-7`` is
    the limit ``1``."""
    bl = b * diffusivity
    sqrt_bl = jnp.sqrt(bl)
    return jnp.where(bl > 1e-7, jnp.sqrt(jnp.pi) * jax.scipy.special.erf(sqrt_bl) / (2.0 * sqrt_bl),
                     jnp.ones_like(b))


def zeppelin_powder_average(b, diffusivity_par, diffusivity_perp):
    """The twin of :func:`dmipy_analytic.gaussian.zeppelin_powder_average` for a prolate tensor
    (``diffusivity_par >= diffusivity_perp``; at equal diffusivities it is the ball). Its domain ends there: the oblate
    average needs ``erfi``, which JAX does not provide, and the twin returns NaN for it, never a finite value of the
    wrong tensor."""
    exp_bl = jnp.exp(-b * diffusivity_perp)
    sqrt_bl = jnp.sqrt(b * jnp.abs(diffusivity_par - diffusivity_perp))
    safe_sqrt_bl = jnp.where(sqrt_bl > 0, sqrt_bl, jnp.ones_like(sqrt_bl))
    anisotropic = exp_bl * jnp.sqrt(jnp.pi) * jax.scipy.special.erf(safe_sqrt_bl) / (2.0 * safe_sqrt_bl)
    isotropic = jnp.exp(-b * diffusivity_par)
    return jnp.where(diffusivity_par > diffusivity_perp, anisotropic,
                     jnp.where(diffusivity_par == diffusivity_perp, isotropic, jnp.nan))
