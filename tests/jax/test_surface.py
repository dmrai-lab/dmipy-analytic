"""The JAX twins of ``dmipy_analytic.restricted.surface`` against the numpy forms.

The twin is the numpy form's ordered product of matrix exponentials taken step by step rather than run by run, so the
two differ by rounding alone. A matrix exponential by scaling and squaring is designed to a backward error of
``U64 ||A||_1`` (Higham 2005, SIAM J. Matrix Anal. Appl. 26:1179, JAX's; Al-Mohy and Higham 2009, 31:970, scipy's),
which a product of contractions passes on unamplified: the numpy form is allowed ``U64 sum_runs ||tau H||_1`` and
``N U64`` per product, the twin ``U64 sum_steps ||dt H||_1`` and ``N U64`` per step; an infinite tube's axial
exponent, a sum of ``n_t`` terms of five operations, ``5 n_t U64`` (``x e^{-x} <= 1``); their product ``U64``.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy.linalg")

import jax
import jax.numpy as jnp

import dmipy_analytic.restricted.matrix_method as mm
from dmipy_analytic.constants import GAMMA
from dmipy_analytic.jax.restricted import surface as jsurface
from dmipy_analytic.restricted import surface
from dmipy_analytic.restricted.matrix_method import MatrixPore, PeriodicPore
from .rounding import U64, x64

D = 0.8e-9
DT = 1e-4
OUTLINE = np.array([[2e-6, 0.0], [0.4e-6, 1.5e-6], [-1.8e-6, 0.5e-6], [-1.0e-6, -1.4e-6], [1.0e-6, -1.2e-6]])


def _budget(lam, B, g, dt, refocus, per_step):
    """The rounding of one echo across (or along) the axis: per run for the numpy form, per step for the twin."""
    flips = mm._refocus_steps(refocus, len(g))
    runs = [(tuple(x), 1) for x in g] if per_step else mm._runs(g, flips)
    total = 0.0
    for amp, ns in runs:
        H = mm._generator(D, lam, B, amp)
        total += U64 * ns * dt * float(np.abs(H).sum(axis=0).max()) + len(lam) * U64
    return total


def _tube_budget(pore, g, dt, refocus, length):
    total = 2 * U64
    for per_step in (False, True):
        total += _budget(pore.lam, pore.B, g[:, :2], dt, refocus, per_step)
        if length is None:
            total += 5 * len(g) * U64
        else:
            plates = MatrixPore("plane", length, D)
            total += _budget(plates.lam, plates.B[None], g[:, 2:], dt, refocus, per_step)
    return total


def _cases():
    g, ref = surface.pgse_waveform(0.3, 2e-3, 6e-3, DT, [0.6, -0.2, 0.5], ramp=5e-4)
    t = 2.0 * np.pi * 2.0 * (np.arange(60) + 0.5) / 60
    turning = np.stack([0.2 * np.cos(t), 0.2 * np.sin(t), np.zeros(60)], axis=1)
    return [("circle, oblique trapezoidal PGSE", "circle", g, ref, None),
            ("circle capped, oblique trapezoidal PGSE", "circle", g, ref, 3e-6),
            ("circle, a turning gradient", "circle", turning, (), None),
            ("pentagon, oblique trapezoidal PGSE", "outline", g, ref, None),
            ("pentagon capped, the effective waveform", "outline", np.asarray(g) * np.where(
                np.arange(len(g)) < ref[0], 1.0, -1.0)[:, None], (), 2.5e-6)]


@pytest.mark.parametrize("case", range(5))
def test_the_twin_is_the_numpy_form_to_rounding(case):
    name, kind, g, ref, length = _cases()[case]
    if kind == "circle":
        M = surface.circle_order(g, DT, 3e-6)
        exact = surface.circle_signal(g, DT, 3e-6, D, length=length, refocus=ref, n_modes=M)
        pore = PeriodicPore.circle(3e-6, D, n_modes=M)
        with x64():
            twin = complex(jsurface.circle_signal(jnp.asarray(g), DT, 3e-6, D, length=length, refocus=ref,
                                                  n_modes=M))
    else:
        exact = surface.outline_signal(g, DT, OUTLINE, D, n_modes=12, length=length, refocus=ref)
        pore = PeriodicPore.polygon(OUTLINE, D, n_modes=12)
        with x64():
            twin = complex(jsurface.outline_signal(jnp.asarray(g), DT, OUTLINE, D, n_modes=12, length=length,
                                                   refocus=ref))
    assert abs(twin - exact) <= _tube_budget(pore, np.asarray(g), DT, ref, length), name


def test_the_twin_is_differentiable_in_the_diffusivity():
    """``jax.grad`` against central differences of the twin: the difference at step ``h`` is off by
    ``|fd(2h) - fd(h)|/3`` to leading order (Richardson) and by the rounding of the two echoes over ``2 h``."""
    g, ref = surface.pgse_waveform(0.3, 2e-3, 6e-3, DT, [0.6, -0.2, 0.5], ramp=5e-4)
    M = surface.circle_order(g, DT, 3e-6)
    pore = PeriodicPore.circle(3e-6, D, n_modes=M)

    def echo(d):
        return jnp.real(jsurface.circle_signal(jnp.asarray(g), DT, 3e-6, d, refocus=ref, n_modes=M))

    with x64():
        slope = float(jax.grad(echo)(D))
        h = 1e-4 * D
        fd = [float((echo(D + s) - echo(D - s)) / (2 * s)) for s in (h, 2 * h)]
    rounding = _tube_budget(pore, np.asarray(g), DT, ref, None)
    assert abs(slope - fd[0]) <= abs(fd[1] - fd[0]) / 3.0 + rounding / h
