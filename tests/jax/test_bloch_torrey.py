"""The JAX twin of ``dmipy_analytic.bloch_torrey`` against the numpy definition, in float64.

The twin runs the numpy form's own steps on the numpy form's own operator, so the two differ by rounding alone. Each
step of ``N`` cells is two real-by-complex matrix-vector products (``2 N`` operations per entry each) and four
elementwise products, on a state whose entries are at most the unit scale of the start; a 180 adds two products.
Rounding errors of a product of contractions add, so over ``n`` steps the state is within ``n (4 N + 4) U64`` of the
numpy form's (absolute, on the unit scale), and the sum over the cells adds ``N U64``.
"""
import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from dmipy_analytic.bloch_torrey import BlochTorrey1D
from dmipy_analytic.jax import bloch_torrey as jbt
from .rounding import U64, x64


def _model():
    return BlochTorrey1D([0.0, 2e-6, 4e-6], 2e-9, permeability=1e-5, membrane_relaxivity=1e-5, wall_relaxivity=2e-5,
                         R2=(12.5, 50.0), omega=(0.0, 150.0), background_gradient=0.01, density=(1.0, 0.0), n_cells=64)


def _waveform():
    G = np.zeros(200)
    G[20:60] = 0.08
    G[120:160] = 0.08
    return G


@pytest.mark.parametrize("refocus, sub_steps", [((), 1), ((100,), 2), ((0, 100, 200), 1)])
def test_the_twin_is_the_numpy_form_to_rounding(refocus, sub_steps):
    model, G = _model(), _waveform()
    want = model.signal(G, 1e-4, refocus=refocus, sub_steps=sub_steps)
    with x64():
        got = complex(jbt.signal(model, jnp.asarray(G), 1e-4, refocus=refocus, sub_steps=sub_steps))
    N = len(model.h)
    n = len(G) * sub_steps + len(refocus)
    assert abs(got - want) <= (n * (4 * N + 4) + N) * U64, (got, want)


def test_the_twin_carries_a_permeability_breakpoint():
    """A permeability that changes twice (at whole steps of the waveform): the twin carries the state into each
    interval's eigenbasis as the numpy form does, so the two differ by rounding alone (the two basis changes count as
    two more products)."""
    model = BlochTorrey1D([0.0, 2e-6, 4e-6], 2e-9, permeability=(1e-5, 1e-4, 0.0), permeability_breakpoints=(0.005, 0.012),
                          R2=(12.5, 50.0), density=(1.0, 0.0), n_cells=64)
    G = _waveform()
    want = model.signal(G, 1e-4, refocus=(100,), sub_steps=2)
    with x64():
        got = complex(jbt.signal(model, jnp.asarray(G), 1e-4, refocus=(100,), sub_steps=2))
    N = len(model.h)
    n = len(G) * 2 + 1 + 2
    assert abs(got - want) <= (n * (4 * N + 4) + N) * U64, (got, want)
    with pytest.raises(ValueError, match="not on a boundary"):
        jbt.signal(BlochTorrey1D([0.0, 2e-6, 4e-6], 2e-9, permeability=(1e-5, 1e-4), permeability_breakpoints=(0.00505,),
                                 n_cells=16), jnp.asarray(G), 1e-4, sub_steps=2)


def test_the_twin_is_differentiable_in_the_gradient():
    """The derivative of the echo's real part with respect to one gradient sample, against the central difference of
    the numpy form: a central difference of step ``h`` is exact to ``h^2 / 6`` times the third derivative (estimated by
    the third difference, and allowed twice), and its rounding is that of the two values over ``2 h``."""
    model, G = _model(), _waveform()
    with x64():
        grad = jax.grad(lambda g: jnp.real(jbt.signal(model, g, 1e-4, refocus=(100,))))(jnp.asarray(G))
        d_twin = float(grad[40])
    h = 1e-3
    f = [model.signal(G + s * h * (np.arange(len(G)) == 40), 1e-4, refocus=(100,)).real for s in (-2, -1, 1, 2)]
    d_central = (f[2] - f[1]) / (2 * h)
    third = abs((f[3] - 2 * f[2] + 2 * f[1] - f[0]) / (2 * h ** 3))
    rounding = 2 * (len(G) + 1) * (4 * len(model.h) + 4) * U64 / (2 * h)
    assert abs(d_twin - d_central) <= 2.0 * third * h ** 2 / 6.0 + rounding, (d_twin, d_central)
