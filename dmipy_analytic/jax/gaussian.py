"""The JAX twins of the forms of :mod:`dmipy_analytic.gaussian`, each held to its numpy definition.

Each twin has the name and the arguments of its numpy form. The contractions over the three components of a direction
are sums of products (``jnp.sum`` over the last axis), which XLA does not lower to a batched matrix multiplication, so
a twin evaluates a batch of one on every device.

The oblate zeppelin's powder average has no closed form in JAX (it is an ``erfi``, which ``jax.scipy.special`` does not
provide); :func:`zeppelin_powder_average` evaluates it by Gauss-Legendre quadrature, with the number of nodes derived
from a stated bound on the quadrature error (:func:`_quadrature_bound`).
"""
import functools
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "ball", "stick", "stick_squared_cosine", "stick_btensor", "zeppelin", "zeppelin_squared_cosines",
    "zeppelin_btensor", "tensor", "tensor_btensor", "time_dependent_diffusivity", "temporal_zeppelin",
    "stick_powder_average", "zeppelin_powder_average",
]


def ball(b, diffusivity):
    """The twin of :func:`dmipy_analytic.gaussian.ball`."""
    return jnp.exp(-b * diffusivity)


def stick(b, direction, axis, diffusivity):
    """The twin of :func:`dmipy_analytic.gaussian.stick`."""
    cos_theta = jnp.sum(direction * axis, axis=-1)
    return stick_squared_cosine(b, cos_theta ** 2, diffusivity)


def stick_squared_cosine(b, cos2, diffusivity):
    """The twin of :func:`dmipy_analytic.gaussian.stick_squared_cosine`."""
    return jnp.exp(-b * diffusivity * cos2)


def stick_btensor(btensor, axis, diffusivity):
    """The twin of :func:`dmipy_analytic.gaussian.stick_btensor`."""
    return jnp.exp(-diffusivity * jnp.einsum("i,mij,j->m", axis, btensor, axis))


def zeppelin(b, direction, axis, diffusivity_par, diffusivity_perp):
    """The twin of :func:`dmipy_analytic.gaussian.zeppelin`. The perpendicular part of the direction is
    ``|n|^2 - (n . u)^2``, floored at zero, so no outer-product projector is built."""
    mag_par = jnp.sum(direction * axis, axis=-1)
    mag_sq = jnp.sum(direction ** 2, axis=-1)
    mag_perp_sq = jnp.maximum(0.0, mag_sq - mag_par ** 2)
    return zeppelin_squared_cosines(b, mag_par ** 2, mag_perp_sq, diffusivity_par, diffusivity_perp)


def zeppelin_squared_cosines(b, cos2, sin2, diffusivity_par, diffusivity_perp):
    """The twin of :func:`dmipy_analytic.gaussian.zeppelin_squared_cosines`."""
    return jnp.exp(-b * (diffusivity_par * cos2 + diffusivity_perp * sin2))


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


OBLATE_KAPPA_MAX = 64.0
"""The largest ``kappa = b (D_perp - D_par)`` of an oblate tensor that :func:`zeppelin_powder_average` evaluates (a
b-value of 20,000 s/mm^2 at a diffusivity contrast of 3.2e-9 m^2/s); beyond it the twin returns NaN."""

OBLATE_TOLERANCE = 2.0 ** -53
"""The relative error of the quadrature of the oblate average, the unit roundoff of float64: the number of nodes is
the smallest for which :func:`_quadrature_bound` holds it for every ``kappa`` up to :data:`OBLATE_KAPPA_MAX`."""


def _quadrature_bound(kappa, n_nodes):
    r"""An upper bound of the relative error of the ``n_nodes``-point Gauss-Legendre rule for
    ``J(kappa) = int_0^1 exp(kappa t^2) dt = (1/2) int_{-1}^{1} exp(kappa t^2) dt``, ``kappa >= 0``.

    The integrand is even and entire. The rule has positive weights of sum 2 and is exact for polynomials of degree
    ``2 n - 1``, so its error on a function ``g`` is at most ``4 max|g|``; with ``g = f - p``, ``p`` the Chebyshev
    series of ``f`` cut at degree ``2 n - 1``, and ``f`` bounded by ``M`` inside the Bernstein ellipse of parameter
    ``rho > 1`` (foci at ``+-1``, semi-axes ``(rho +- 1/rho)/2``), the Chebyshev coefficients are at most
    ``2 M rho^{-k}``, so ``max|g| <= 2 M rho^{1-2n} / (rho - 1)`` (Trefethen, *Approximation Theory and Approximation
    Practice*, Thms. 8.1-8.2). On that ellipse ``|exp(kappa z^2)| <= M = exp(kappa ((rho + 1/rho)/2)^2)``, and the
    integral is at least ``max(1, sinh(kappa)/kappa)`` (``t^2 >= 2 t - 1``). The error of ``J`` relative to ``J`` is
    therefore at most ``4 M rho^{1-2n} / ((rho - 1) max(1, sinh(kappa)/kappa))``, here minimised over ``rho``: it
    decays geometrically in ``n`` once ``n`` exceeds ``sqrt(kappa)``-scaled terms (``exp(-n^2/kappa)`` for
    ``n < kappa``)."""
    rho = 1.0 + np.logspace(-4, 1, 4000)
    if kappa > 1.0:
        log_integral = kappa + np.log1p(-np.exp(-2.0 * kappa)) - np.log(2.0 * kappa)     # log(sinh(kappa) / kappa)
    else:
        log_integral = 0.0
    log_bound = (np.log(4.0) + kappa * ((rho + 1.0 / rho) / 2.0) ** 2 + (1 - 2 * n_nodes) * np.log(rho)
                 - np.log(rho - 1.0) - log_integral)
    return float(np.exp(log_bound.min()))


@functools.lru_cache(maxsize=None)
def _oblate_nodes():
    """The number of nodes (even) of the rule of :func:`zeppelin_powder_average`: the smallest ``n`` for which
    :func:`_quadrature_bound` is at most :data:`OBLATE_TOLERANCE` on a grid of 64 values of ``kappa`` up to
    :data:`OBLATE_KAPPA_MAX`."""
    grid = OBLATE_KAPPA_MAX * np.arange(1, 65) / 64.0
    n = 2
    while max(_quadrature_bound(k, n) for k in grid) > OBLATE_TOLERANCE:
        n += 2
    return n


def _legendre(n, x):
    """``P_n(x)`` and ``P_{n-1}(x)`` by the three-term recurrence (any numeric type)."""
    p_previous, p = 1, x
    for k in range(2, n + 1):
        p_previous, p = p, ((2 * k - 1) * x * p - (k - 1) * p_previous) / k
    return p, p_previous


@functools.lru_cache(maxsize=None)
def _half_rule_exact(n_nodes):
    """The positive nodes ``x_i`` and weights ``w_i`` of the ``n_nodes``-point Gauss-Legendre rule (``n_nodes`` even), as
    ``Decimal`` numbers of 50 digits, by Newton's method on ``P_n`` from the nodes of numpy's rule. ``sum w_i = 1`` and
    ``sum w_i f(x_i)`` is the rule for ``int_0^1 f(t) dt`` of an even ``f``."""
    guesses = np.polynomial.legendre.leggauss(n_nodes)[0]
    rule = []
    with localcontext() as context:
        context.prec = 50
        for guess in guesses[guesses > 0]:
            x = Decimal(float(guess))
            for _ in range(6):
                p, p_previous = _legendre(n_nodes, x)
                x = x - p / (n_nodes * (x * p - p_previous) / (x * x - 1))
            p, p_previous = _legendre(n_nodes, x)
            derivative = n_nodes * (x * p - p_previous) / (x * x - 1)
            rule.append((x, 2 / ((1 - x * x) * derivative * derivative)))
    return tuple(sorted(rule))


@functools.lru_cache(maxsize=None)
def _half_rule(n_nodes):
    """The squares ``x_i^2``, their complements ``1 - x_i^2`` and the weights ``w_i`` of :func:`_half_rule_exact`, each
    rounded once to float64 (the weights of numpy's ``leggauss`` carry relative errors of 1e-12)."""
    rule = _half_rule_exact(n_nodes)
    squares = np.array([float(x * x) for x, _ in rule])
    complements = np.array([float(1 - x * x) for x, _ in rule])
    weights = np.array([float(w) for _, w in rule])
    return squares, complements, weights


def _oblate_average(b, diffusivity_par, diffusivity_perp):
    """``exp(-b D_perp) J(kappa)``, ``kappa = b (D_perp - D_par)``, by the Gauss-Legendre rule of
    :data:`OBLATE_TOLERANCE`, as the sum of ``w_i exp(-b (D_par x_i^2 + D_perp (1 - x_i^2)))`` (every exponent is
    negative); NaN beyond :data:`OBLATE_KAPPA_MAX`."""
    squares, complements, weights = (jnp.asarray(a) for a in _half_rule(_oblate_nodes()))
    b_, par, perp = (jnp.asarray(a)[..., None] for a in (b, diffusivity_par, diffusivity_perp))
    average = jnp.sum(weights * jnp.exp(-b_ * (par * squares + perp * complements)), axis=-1)
    return jnp.where(b * (diffusivity_perp - diffusivity_par) <= OBLATE_KAPPA_MAX, average, jnp.nan)


def zeppelin_powder_average(b, diffusivity_par, diffusivity_perp):
    """The twin of :func:`dmipy_analytic.gaussian.zeppelin_powder_average`. A prolate tensor
    (``diffusivity_par > diffusivity_perp``) is the closed form with ``erf``; equal diffusivities are the ball; an oblate
    tensor, whose closed form is an ``erfi`` that JAX does not provide, is the Gauss-Legendre rule of
    ``_oblate_nodes()`` nodes (58 at present, 29 exponentials) for ``int_0^1 exp(-b (D_par t^2 + D_perp (1 - t^2))) dt``.

    The integrand is smooth, so the rule's error decays geometrically in the number of nodes: for
    ``kappa = b (D_perp - D_par)`` up to :data:`OBLATE_KAPPA_MAX` it is at most :data:`OBLATE_TOLERANCE` relative to the
    average (:func:`_quadrature_bound`, with the proof), and the nodes and weights are exact to the rounding of float64.
    On top of it come the rounding of the sum, ``n/2`` roundings, and of the exponents, about ``6 u b D_max`` relative
    (``u`` the unit roundoff, ``D_max`` the larger diffusivity). Beyond :data:`OBLATE_KAPPA_MAX` the twin returns NaN,
    never a value of a different accuracy."""
    exp_bl = jnp.exp(-b * diffusivity_perp)
    sqrt_bl = jnp.sqrt(b * jnp.abs(diffusivity_par - diffusivity_perp))
    safe_sqrt_bl = jnp.where(sqrt_bl > 0, sqrt_bl, jnp.ones_like(sqrt_bl))
    anisotropic = exp_bl * jnp.sqrt(jnp.pi) * jax.scipy.special.erf(safe_sqrt_bl) / (2.0 * safe_sqrt_bl)
    isotropic = jnp.exp(-b * diffusivity_par)
    return jnp.where(diffusivity_par > diffusivity_perp, anisotropic,
                     jnp.where(diffusivity_par == diffusivity_perp, isotropic,
                               _oblate_average(b, diffusivity_par, diffusivity_perp)))
