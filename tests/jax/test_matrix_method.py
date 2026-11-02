"""The JAX twins of ``dmipy_analytic.restricted.matrix_method`` against the numpy forms.

The twin is a Strang split (half a step of diffusion, a step of precession, half a step of diffusion) of the mode
coefficients; the numpy form is the exact product of matrix exponentials over the runs of a piecewise-constant
waveform. The twin differs from the exact signal by the splitting error and by rounding, each derived:

* the Strang split has the local error ``dt^3 / 12`` times the norm of the double commutator of the diffusion and the
  precession generators, ``|[A, [A, B]]| ~ (D lam_max)^2 gamma g beta_max`` for the largest eigenvalue ``lam_max`` and
  position eigenvalue ``beta_max`` kept, and the global error is ``n_t`` times it; the test evaluates it for the
  waveform and the pore in question and requires the twin within it, plus rounding;
* the rounding of the scan is ``n_t (2 N + 4) U64`` in float64 (a complex matrix-vector product with ``N`` modes twice
  per step and four elementwise operations), absolute on the unit scale of the state.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy.linalg")

import jax.numpy as jnp

from dmipy_analytic.constants import GAMMA
from dmipy_analytic.restricted import matrix_method as mm
from dmipy_analytic.jax.restricted import matrix_method as jmm
from .rounding import U64, x64

D = 2e-9


def _waveform(G, delta, Delta, steps):
    dt = delta / steps
    nd, nD = steps, int(round(Delta / dt))
    g = np.zeros(nD + nd)
    g[:nd] = G
    g[nD:nD + nd] = -G
    return g, dt


def _splitting_bound(pore, g, dt):
    lam_max = float(np.max(pore.lam))
    beta_max = float(np.max(np.abs(np.linalg.eigvalsh(pore.B))))
    return len(g) * dt ** 3 / 12.0 * (D * lam_max) ** 2 * GAMMA * float(np.max(np.abs(g))) * beta_max


@pytest.mark.parametrize("shape, size, n_modes", [("plane", 4e-6, 8), ("cylinder", 4e-6, (3, 5)),
                                                  ("sphere", 6e-6, (3, 5))])
def test_the_strang_split_is_the_exact_signal_within_its_splitting_error(shape, size, n_modes):
    g, dt = _waveform(0.05, 0.004, 0.01, 40)
    pore = mm.MatrixPore(shape, size, D, n_modes=n_modes)
    exact = abs(pore.signal(g, dt))
    with x64():
        twin = float(jmm.matrix_restricted_signal(shape, jnp.asarray(g), dt, D, size, n_modes=n_modes))
    bound = _splitting_bound(pore, g, dt) + len(g) * (2 * len(pore.lam) + 4) * U64
    assert abs(twin - exact) <= bound


def test_the_batch_is_the_signal_of_each_schedule_and_matches_the_numpy_batch():
    g, dt = _waveform(0.05, 0.004, 0.01, 40)
    gs = np.stack([g, 0.5 * g, np.zeros_like(g)])
    n_modes = (3, 5)
    with x64():
        batch = np.asarray(jmm.matrix_restricted_batch("cylinder", jnp.asarray(gs), dt, D, 4e-6, n_modes=n_modes))
        singles = np.array([float(jmm.matrix_restricted_signal("cylinder", jnp.asarray(x), dt, D, 4e-6,
                                                               n_modes=n_modes)) for x in gs])
    pore = mm.MatrixPore("cylinder", 4e-6, D, n_modes=n_modes)
    rounding = len(g) * (2 * len(pore.lam) + 4) * U64                  # the batch fuses the products differently
    assert np.all(np.abs(batch - singles) <= rounding)
    exact = mm.matrix_restricted_batch("cylinder", gs, dt, D, 4e-6, n_modes=n_modes)
    assert np.all(np.abs(batch - exact) <= _splitting_bound(pore, g, dt) + len(g) * (2 * len(pore.lam) + 4) * U64)
    assert batch[2] == pytest.approx(1.0, abs=1e-12)


def test_the_signal_is_differentiable_in_the_diffusivity():
    import jax
    g, dt = _waveform(0.05, 0.004, 0.01, 40)
    with x64():
        slope = float(jax.grad(lambda d: jmm.matrix_restricted_signal(
            "plane", jnp.asarray(g), dt, d, 4e-6, n_modes=8))(D))
        step = 1e-12
        finite = float((jmm.matrix_restricted_signal("plane", jnp.asarray(g), dt, D + step, 4e-6, n_modes=8)
                        - jmm.matrix_restricted_signal("plane", jnp.asarray(g), dt, D - step, 4e-6, n_modes=8))
                       / (2 * step))
    assert slope == pytest.approx(finite, rel=1e-5)
