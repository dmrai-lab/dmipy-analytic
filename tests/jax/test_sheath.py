"""The JAX twins of ``dmipy_analytic.restricted.sheath`` against the numpy forms.

The twin is the numpy form's ordered product of matrix exponentials taken step by step rather than run by run, so the
two differ by rounding alone (the waveforms here have fewer runs than the numpy form's ceiling, so it takes the exact
product). The rounding is :mod:`tests.jax.test_surface`'s: ``U64 sum ||tau H||_1`` and ``N U64`` per product, per run for
the numpy form and per step for the twin, ``5 n_t U64`` for an infinite tube's axial exponent, ``U64`` for the product;
the lamellae's average adds ``U64`` per node.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy.linalg")

import jax
import jax.numpy as jnp

import dmipy_analytic.restricted.matrix_method as mm
from dmipy_analytic.jax.restricted import sheath as jsheath
from dmipy_analytic.restricted import sheath, surface
from dmipy_analytic.restricted.matrix_method import AnnulusPore, PeriodicPore
from .rounding import U64, x64

D = 2.0e-9
DT = 1e-4
OUTER, INNER = 6e-6, 4.2e-6
MODES = (10, 4)


def _budget(lam, B, g, refocus, per_step):
    flips = mm._refocus_steps(refocus, len(g))
    runs = [(tuple(x), 1) for x in g] if per_step else mm._runs(g, flips)
    total = 0.0
    for amp, ns in runs:
        H = mm._generator(D, lam, B, amp)
        total += U64 * ns * DT * float(np.abs(H).sum(axis=0).max()) + len(lam) * U64
    return total


def _both(lam, B, g, refocus):
    return sum(_budget(lam, B, g, refocus, s) for s in (False, True)) + 2 * (5 * len(g) * U64) + 2 * U64


def _waveforms():
    g, ref = surface.pgse_waveform(0.25, 3e-3, 8e-3, DT, [0.6, -0.2, 0.5], ramp=5e-4)
    eff = np.asarray(g) * np.where(np.arange(len(g)) < ref[0], 1.0, -1.0)[:, None]
    return [("oblique trapezoidal PGSE as played", g, ref), ("its effective waveform", eff, ())]


@pytest.mark.parametrize("case", range(2))
def test_the_annulus_twin_is_the_numpy_form_to_rounding(case):
    name, g, ref = _waveforms()[case]
    exact = sheath.annulus_signal(g, DT, OUTER, INNER, D, refocus=ref, n_modes=MODES)
    pore = AnnulusPore(OUTER, INNER, D, n_modes=MODES)
    a = sheath.fixed_direction(np.asarray(g)[:, :2])
    with x64():
        twin = complex(jsheath.annulus_signal(jnp.asarray(g), DT, OUTER, INNER, D, refocus=ref, n_modes=MODES))
    assert abs(twin - exact) <= _both(pore.lam, pore.B[None], a[:, None], ref), name


@pytest.mark.parametrize("case", range(2))
def test_the_lamellae_twin_is_the_numpy_form_to_rounding(case):
    name, g, ref = _waveforms()[case]
    M = surface.circle_order(np.asarray(g), DT, OUTER)
    exact = sheath.lamellae_signal(g, DT, OUTER, INNER, D, refocus=ref, n_radii=6)
    with x64():
        twin = complex(jsheath.lamellae_signal(jnp.asarray(g), DT, OUTER, INNER, D, refocus=ref, n_radii=6))
    r, _ = sheath.lamellae_radii(OUTER, INNER, 6)
    budget = max(_both(p.lam, p.B, np.asarray(g)[:, :2], ref) for p in
                 (PeriodicPore.circle(2 * ri, D, n_modes=M) for ri in r)) + 6 * U64
    assert abs(twin - exact) <= budget, name


def test_the_annulus_twin_is_differentiable_in_the_diffusivity():
    """``jax.grad`` against central differences of the twin: the difference at step ``h`` is off by
    ``|fd(2h) - fd(h)|/3`` to leading order (Richardson) and by the rounding of the two echoes over ``2 h``."""
    _, g, ref = _waveforms()[0]
    pore = AnnulusPore(OUTER, INNER, D, n_modes=MODES)
    a = sheath.fixed_direction(np.asarray(g)[:, :2])

    def echo(d):
        return jnp.real(jsheath.annulus_signal(jnp.asarray(g), DT, OUTER, INNER, d, refocus=ref, n_modes=MODES))

    with x64():
        slope = float(jax.grad(echo)(D))
        h = 1e-4 * D
        fd = [float((echo(D + s) - echo(D - s)) / (2 * s)) for s in (h, 2 * h)]
    rounding = _both(pore.lam, pore.B[None], a[:, None], ref)
    assert abs(slope - fd[0]) <= abs(fd[1] - fd[0]) / 3.0 + rounding / h
