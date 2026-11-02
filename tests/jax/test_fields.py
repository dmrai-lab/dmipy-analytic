"""The JAX twins of the field forms against their numpy definitions, in float32 against float64.

The tolerance is the first-order rounding bound of the twin's float32 evaluation, never a fitted number: each output
is a sum of terms, each a product of a coefficient and a few unit-vector components, and a float32 evaluation carries
at most ``n u`` relative error on each term, with ``u = eps32 / 2`` the unit roundoff and ``n`` the roundings along
its longest dependency chain, inputs included. For the infinite cylinder that chain (the norm, the radial unit vector,
its cross product with the axis, the outer products, the coefficient and the sums) is below 100 roundings, so the
bound is ``64 eps32`` (128 ``u``) times the sum of the terms' magnitudes. For the segment the primitives are
differenced at the two ends, so the magnitudes summed are those of the primitives at both ends, before the
difference; their chains are below 100 roundings too. A point within 1e-4 of a wall is not compared: its region is
decided on ``rho``, which float32 rounding moves."""
import numpy as np
import pytest

pytest.importorskip("jax")

from dmipy_analytic import fields
from dmipy_analytic.jax import fields as twins

EPS32 = float(np.finfo(np.float32).eps)
K = 64


def _points(n, a, b, seed):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    t = rng.normal(size=(n, 3)); t -= np.sum(t * u, axis=1, keepdims=True) * u
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    r = np.exp(rng.uniform(np.log(0.05 * a), np.log(8.0 * b), n))
    keep = (np.abs(r / a - 1.0) > 1e-4) & (np.abs(r / b - 1.0) > 1e-4)
    return (r[:, None] * t)[keep], u[keep]


def test_the_hollow_cylinder_twin_is_its_definition_to_float32_rounding():
    a, b = 0.6e-6, 0.9e-6
    rho, u = _points(4000, a, b, 0)
    n = len(rho)
    f32 = lambda x: np.asarray(x, np.float32)
    want = fields.hollow_cylinder_basis(rho, u, np.full(n, a), np.full(n, b))
    got = np.asarray(twins.hollow_cylinder_basis(f32(rho), f32(u), f32(np.full(n, a)), f32(np.full(n, b))), np.float64)
    r2 = np.sum(rho * rho, axis=1)
    # the terms' magnitudes: the coefficients of P, u u^T and S (unit-vector products are at most 1)
    scale = 1.0 + 0.5 * np.log(b / a) + np.abs(np.log(b / np.sqrt(r2))) + (b * b) / r2
    assert np.all(np.abs(got - want) <= K * EPS32 * scale[:, None])


def test_the_segment_twin_is_its_definition_to_float32_rounding():
    a, b = 0.6e-6, 0.9e-6
    rho, u = _points(4000, a, b, 1)
    rho = rho[np.linalg.norm(rho, axis=1) > b * (1 + 1e-4)]
    u = u[:len(rho)]
    n = len(rho)
    rng = np.random.default_rng(2)
    z1 = rng.uniform(-30e-6, 5e-6, n)
    z2 = z1 + rng.uniform(0.1e-6, 40e-6, n)
    f32 = lambda x: np.asarray(x, np.float32)
    want = fields.segment_basis(rho, u, z1, z2, np.full(n, a), np.full(n, b))
    got = np.asarray(twins.segment_basis(f32(rho), f32(u), f32(z1), f32(z2), f32(np.full(n, a)), f32(np.full(n, b))),
                     np.float64)
    # the primitives at both ends, in units of b, and the combination's multipliers
    rr = np.linalg.norm(rho, axis=1) / b
    mag = 0.0
    for z in (z1 / b, z2 / b):
        R = np.sqrt(rr * rr + z * z)
        prims = (np.abs(z) / (rr * rr * R), np.abs(z) * (2 * z * z + 3 * rr * rr) / (3 * rr ** 4 * R ** 3),
                 1.0 / (3 * R ** 3), np.abs(z) ** 3 / (3 * rr * rr * R ** 3))
        mag = mag + prims[0] + 3.0 * (rr * rr * prims[1] + 2.0 * rr * prims[2] + prims[3])
    c = (1.0 - (a / b) ** 2) / 4.0
    assert np.all(np.abs(got - want) <= K * EPS32 * (c * mag)[:, None])


def test_a_segment_is_the_line_up_close_and_a_dipole_far_away():
    """The twin of `segment_basis`, in its own float32: a very long segment seen from beside its middle is the infinite
    cylinder's outside formula; a short one seen from far away is the point dipole of its moment (the isotropic
    channels: `A L (I - 3 r r^T) / 4 pi r^3`); a line cut into pieces sums to the line."""
    a, b = 1.0e-6, 1.5e-6; u = np.array([0.3, -0.2, 1.0]); u /= np.linalg.norm(u)
    rng = np.random.default_rng(3); v = rng.normal(size=3); rv = v - (v @ u) * u; rv *= 6e-6 / np.linalg.norm(rv)
    long_ = np.asarray(twins.segment_basis(rv, u, -1.0, 1.0, a, b)); inf_ = np.asarray(twins.hollow_cylinder_basis(rv, u, a, b))
    np.testing.assert_allclose(long_[1:], inf_[1:], rtol=0, atol=1e-6 * np.abs(inf_[1:]).max())
    L = 2e-6; zc = 60e-6; r = rv + zc * u; rhat = r / np.linalg.norm(r); A = np.pi * (b ** 2 - a ** 2)   # (L / r)^2 = 1e-3; float32 holds a 2 um piece at 60 um
    far = np.asarray(twins.segment_basis(rv, u, -zc - L / 2, -zc + L / 2, a, b))
    dip = A * L * (np.eye(3) - 3 * np.outer(rhat, rhat)) / (4 * np.pi * np.linalg.norm(r) ** 3)
    want = np.array([dip[0, 0], dip[1, 1], dip[0, 1], dip[0, 2], dip[1, 2]])                 # zz is the trace's
    np.testing.assert_allclose(far[1:6], want, rtol=5e-3, atol=2e-3 * np.abs(want).max())
    cuts = np.linspace(-40e-6, 40e-6, 9)
    whole = np.asarray(twins.segment_basis(rv, u, cuts[0], cuts[-1], a, b))
    pieces = sum(np.asarray(twins.segment_basis(rv, u, z1, z2, a, b)) for z1, z2 in zip(cuts[:-1], cuts[1:]))
    np.testing.assert_allclose(pieces, whole, rtol=0, atol=1e-6 * np.abs(whole).max())
