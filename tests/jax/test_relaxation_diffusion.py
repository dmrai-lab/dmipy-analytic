"""The JAX twins of the relaxation factors of a diffusion measurement against their float64 numpy definitions, in
float32.

The tolerances are derived from the rounding of the forms (``rounding.exp_rtol``):

* ``transverse_decay`` and ``longitudinal_decay``: ``x = tau / T`` from two rounded inputs and one quotient;
* ``surface_relaxation``: ``x = rho (S/V) tau`` from three rounded inputs and two products;
* the Gamma averages: a sum of ``n`` positive terms ``w_k exp(-c beta / x_k)`` is evaluated with a relative error of at
  most ``(n + 3) U32`` from its terms (the pairwise sum is better; the bound is the worst case), each term having the
  error of ``exp(-y)`` for ``y = c beta / x_k`` of at most ``y_max`` (``c`` and ``beta`` rounded, two products and a
  quotient: ``(3 + 3)``); the quadrature truncation of the 64-node rule is compared with the 192-node rule in float64
  and must be below one tenth of the rounding bound;
* ``surface_to_volume`` and ``exterior_surface_to_volume``: a few products and quotients of rounded inputs, ``8 U32``
  at the worst.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy.special")

from dmipy_analytic import relaxation
from dmipy_analytic.jax import relaxation as jrelaxation
from .rounding import U32, exp_rtol

TAU = np.array([0.0, 0.02, 0.05, 0.12, 0.3])


def _close(twin, form, rtol):
    twin = np.asarray(twin, dtype=np.float64)
    assert np.all(np.abs(twin - form) <= rtol * np.abs(form)), np.max(np.abs(twin - form) / np.abs(form))


def test_decays():
    _close(jrelaxation.transverse_decay(TAU, 0.08), relaxation.transverse_decay(TAU, 0.08),
           exp_rtol((TAU / 0.08).max(), 2, 1))
    _close(jrelaxation.longitudinal_decay(TAU, 1.2), relaxation.longitudinal_decay(TAU, 1.2),
           exp_rtol((TAU / 1.2).max(), 2, 1))


def test_surface_relaxation_and_geometry():
    rho, sv = 12e-6, 1.1e6
    _close(jrelaxation.surface_relaxation(rho, sv, TAU), relaxation.surface_relaxation(rho, sv, TAU),
           exp_rtol(rho * sv * TAU.max(), 3, 2))
    for geometry in ("sphere", "cylinder", "plane"):
        _close(jrelaxation.surface_to_volume(2.5e-6, geometry), relaxation.surface_to_volume(2.5e-6, geometry),
               2 * U32)
        _close(jrelaxation.exterior_surface_to_volume(0.55, 2.0, 0.3e-6, geometry=geometry),
               relaxation.exterior_surface_to_volume(0.55, 2.0, 0.3e-6, geometry=geometry), 8 * U32)


@pytest.mark.parametrize("name", ["surface_relaxation_gamma_cylinders", "surface_relaxation_gamma_spheres"])
def test_gamma_average(name):
    shape, scale, rho = 2.0, 0.4e-6, 10e-6
    tau = TAU[1:]
    form = getattr(relaxation, name)(shape, scale, rho, tau)
    twin = getattr(jrelaxation, name)(shape, scale, rho, tau)
    n = 64
    y_max = 6.0 * rho * tau.max() / scale / 3.0                  # c beta / x at the smallest node, x ~ 3 for a=4..5
    bound = exp_rtol(y_max, 6, 0) + (n + 3) * U32
    quad_error = np.max(np.abs(np.asarray(getattr(jrelaxation, name)(shape, scale, rho, tau, n_nodes=192)) - form)
                        / form)
    _close(twin, form, bound)
    assert quad_error < bound / 10.0
