r"""Independent finite-kappa SGP validation; predictions recorded BEFORE comparison.

Predicted absolute tolerance: agreement must be within the SUM of the stated
SGP and finite-volume error estimates, with no fitted tolerance floor. Expected
combined errors are below 2e-4 for the fast grid and 5e-5 for the full study.
If either budget is exceeded, or agreement fails, the test fails; do not loosen
these predictions after seeing the comparison.

We predict leading O(delta) finite-pulse error at fixed angular q=gamma G delta,
with a possible O(delta**1.5) reflecting-boundary correction. On four halved
pulse widths, use L(delta)=2 E(delta/2)-E(delta) and extrapolate the last L pair
with power 1.5. Its error estimate is the change from the last L, conservatively
retaining that entire correction, plus the difference from a quadratic-in-delta
extrapolation. Grid error is |intercept(fine)-intercept(middle)|/3 (verified
second-order on three grids); time-split and harmonic errors are checked
separately. These are numerical asymptotic error estimates, not rigorous bounds.

The finite-volume code is called through its public signal API with a PGSE
waveform and a refocusing pulse. Four decreasing deltas have exactly the same
q and onset separation Delta. No package eigenvector/private attribute is read.
The full 36-row table can be regenerated with pytest -s -m slow; the default
invocation also runs the marked study (use -m 'not slow' for the fast subset).
The full study reports all q,Delta,kappa,geometry combinations and delta orders.
The full grid uses four times shorter pulses than the fast grid; q is chosen
from the SGP curve alone to target E=(0.9,0.5,0.05) for each parameter set.
An initial coarser study exceeded the predicted accuracy budget and encountered
a spatial cancellation (order 3.66); the tolerances and estimators are unchanged.
Only the new tests and their independent oracle implement this validation.

Measured full study (CPU, 2026-11-11)
--------------------------------------
23 tests passed in 1043.70 s; the slow full grid took 1018.95 s.
The final fast subset passed: 22 tests, 1 slow test deselected, in 24.42 s.
Fast grids: (15,9), (30,18), (60,36) radial cells; full grids:
(30,18), (60,36), (120,72). Both use 12 FV harmonics, checked at 14;
the split-step counts are 1,2,4 with an independent check at 8.
Full pulse widths: 250,125,62.5,31.25 microseconds (onsets fixed at Delta).
SGP: 10 radial modes/order, 12 angular orders, 64/128 quadrature nodes/region.
Leading delta orders 0.934--1.026; spatial orders 1.979--2.282.
Maximum |FV-SGP|=3.664e-6; maximum combined error=2.84e-5.
All estimates are absolute. The err_SGP column combines quadrature/root roundoff
(about 2e-12) and the stated modal tail bound; the largest SGP bound is
2.76e-7 at the highest-q cylinder point. q is rounded in this printed table.

geometry kappa Delta_s q_rad/m SGP FV_extrap FV-SGP err_SGP err_FV p_delta p_grid
sphere   1e-05 0.020 1.0729488e+05 0.9000000000 0.9000003963 +3.963e-07 2.00e-12 4.49e-06 0.973 2.030
sphere   1e-05 0.020 2.7522314e+05 0.5000000000 0.5000010940 +1.094e-06 2.00e-12 1.26e-05 0.980 2.027
sphere   1e-05 0.020 6.3305692e+05 0.0500000000 0.0500009552 +9.552e-07 1.53e-09 1.93e-06 1.002 2.014
sphere   1e-05 0.050 9.2920643e+04 0.9000000000 0.9000002974 +2.974e-07 2.00e-12 3.39e-06 0.973 2.029
sphere   1e-05 0.050 2.3565703e+05 0.5000000000 0.5000006659 +6.658e-07 2.00e-12 8.86e-06 0.981 2.025
sphere   1e-05 0.050 5.1182193e+05 0.0500000000 0.0499999993 -6.605e-10 1.45e-11 2.29e-06 1.007 1.999
sphere   1e-04 0.020 1.0368827e+05 0.9000000000 0.9000003249 +3.249e-07 2.00e-12 4.00e-06 0.974 2.037
sphere   1e-04 0.020 2.6328184e+05 0.5000000000 0.5000008318 +8.318e-07 2.00e-12 1.06e-05 0.981 2.035
sphere   1e-04 0.020 5.4317745e+05 0.0500000000 0.0500004015 +4.015e-07 5.06e-11 5.15e-07 1.000 2.014
sphere   1e-04 0.050 9.1787739e+04 0.9000000000 0.9000003021 +3.021e-07 2.01e-12 3.21e-06 0.973 2.030
sphere   1e-04 0.050 2.3073982e+05 0.5000000000 0.5000006850 +6.850e-07 2.00e-12 7.86e-06 0.982 2.026
sphere   1e-04 0.050 4.4090559e+05 0.0500000000 0.0499994958 -5.042e-07 2.41e-12 4.87e-06 1.024 2.042
sphere   1e-03 0.020 9.8571042e+04 0.9000000000 0.9000004934 +4.934e-07 2.00e-12 3.03e-06 0.979 2.030
sphere   1e-03 0.020 2.4923304e+05 0.5000000000 0.5000013931 +1.393e-06 2.00e-12 7.37e-06 0.986 2.028
sphere   1e-03 0.020 4.9508092e+05 0.0500000000 0.0500002559 +2.559e-07 7.85e-12 1.90e-06 1.009 2.102
sphere   1e-03 0.050 9.0840906e+04 0.9000000000 0.9000004310 +4.310e-07 2.00e-12 2.63e-06 0.979 2.028
sphere   1e-03 0.050 2.2810956e+05 0.5000000000 0.5000011122 +1.112e-06 2.00e-12 5.94e-06 0.987 2.024
sphere   1e-03 0.050 4.3238632e+05 0.0500000000 0.0499994385 -5.615e-07 2.26e-12 5.30e-06 1.026 2.040
cylinder 1e-05 0.020 1.0990764e+05 0.9000000000 0.9000008363 +8.363e-07 2.00e-12 6.46e-06 0.967 2.020
cylinder 1e-05 0.020 2.8630112e+05 0.5000000000 0.5000028575 +2.857e-06 2.00e-12 1.98e-05 0.975 2.018
cylinder 1e-05 0.020 8.1003099e+05 0.0500000000 0.0500036642 +3.664e-06 2.76e-07 2.81e-05 0.939 2.018
cylinder 1e-05 0.050 8.9415294e+04 0.9000000000 0.9000005578 +5.578e-07 2.00e-12 4.30e-06 0.967 2.020
cylinder 1e-05 0.050 2.3008633e+05 0.5000000000 0.5000017558 +1.756e-06 2.00e-12 1.26e-05 0.976 2.016
cylinder 1e-05 0.050 6.7939107e+05 0.0500000000 0.0500024552 +2.455e-06 6.31e-09 1.28e-05 0.965 2.018
cylinder 1e-04 0.020 1.0105854e+05 0.9000000000 0.9000006212 +6.212e-07 2.00e-12 5.17e-06 0.968 2.025
cylinder 1e-04 0.020 2.5816236e+05 0.5000000000 0.5000020566 +2.057e-06 2.00e-12 1.53e-05 0.975 2.022
cylinder 1e-04 0.020 6.2170871e+05 0.0500000000 0.0500026749 +2.675e-06 8.91e-10 1.71e-05 0.934 2.022
cylinder 1e-04 0.050 8.4604907e+04 0.9000000000 0.9000004710 +4.710e-07 2.00e-12 3.68e-06 0.968 2.023
cylinder 1e-04 0.050 2.1264617e+05 0.5000000000 0.5000013945 +1.395e-06 2.00e-12 9.99e-06 0.977 2.018
cylinder 1e-04 0.050 4.1190812e+05 0.0500000000 0.0500002786 +2.786e-07 2.08e-12 2.28e-06 1.011 1.979
cylinder 1e-03 0.020 9.1149152e+04 0.9000000000 0.9000008009 +8.009e-07 2.00e-12 3.44e-06 0.975 2.020
cylinder 1e-03 0.020 2.3065553e+05 0.5000000000 0.5000026415 +2.642e-06 2.00e-12 9.58e-06 0.982 2.017
cylinder 1e-03 0.020 4.7166957e+05 0.0500000000 0.0500016877 +1.688e-06 3.70e-12 3.50e-06 0.986 2.015
cylinder 1e-03 0.050 8.1656512e+04 0.9000000000 0.9000005925 +5.925e-07 2.00e-12 2.75e-06 0.975 2.022
cylinder 1e-03 0.050 2.0444705e+05 0.5000000000 0.5000017568 +1.757e-06 2.00e-12 6.90e-06 0.984 2.017
cylinder 1e-03 0.050 3.8360863e+05 0.0500000000 0.0499999101 -8.991e-08 2.02e-12 3.70e-06 1.022 2.282

The initial coarser full-study attempt, with widths 1,0.5,0.25,0.125 ms,
gave sphere kappa=1e-5, Delta=20 ms, q=2.8e5: SGP=0.4880827138,
FV=0.4880799005, difference=-2.813e-6, errors=(2.00e-12,9.89e-5).
It agreed within errors but exceeded the predeclared 5e-5 accuracy budget.
A later row had spatial order 3.66. Refining the PULSE RESOLUTION by four
met the same budget and convergence guards, without changing the methods
or the tolerances. A separate q-selection attempt stopped at the overly
short search endpoint 8e5 before cylinder FV comparisons; the explicit
first-crossing search now covers 0--1e6 rad/m.

DECISION: the passive membrane sign and angular-q convention are explicit in
_sgp_permeable's derivation. The printed membrane sign in the brief/package
is inconsistent with the package implementation and with passive diffusion.
"""
from functools import lru_cache
from time import monotonic

from scipy.optimize import brentq

import numpy as np
import pytest

from dmipy_analytic.constants import GAMMA
from dmipy_analytic.permeable_shell import PermeableShell, ShellSpec, pgse_waveform
from tests._sgp_permeable import Problem, Spectrum, single_pore

R, R_OUT = 5e-6, 8e-6
D_IN, D_OUT = 2e-9, 1.2e-9
KAPPAS = (1e-5, 1e-4, 1e-3)
GEOMETRIES = ("sphere", "cylinder")
# Fixed angular-q probes for closed forms and the fast subset.
# The full study chooses its q values by the three target echoes.
QS = (9e4, 2.8e5, 5.5e5)
DELTAS = np.array([1e-3, 5e-4, 2.5e-4, 1.25e-4])


@lru_cache(maxsize=None)
def spectral(geometry, kappa, diffusion_in, diffusion_out, modes, orders):
    return Spectrum(Problem(geometry, R, R_OUT, diffusion_in, diffusion_out, kappa),
                    modes_per_order=modes, orders=orders)


@lru_cache(maxsize=None)
def finite_volume(geometry, kappa, cells, harmonics):
    return PermeableShell(ShellSpec(geometry, R, R_OUT, D_IN, D_OUT, kappa),
                          n_in=5 * cells, n_out=3 * cells, n_harmonics=harmonics)


def pulse_echo(solver, q, Delta, delta, substeps):
    # Choose dt=delta/2 so that the refocussing index is always integral.
    g, refocus = pgse_waveform(q / (GAMMA * delta), delta, Delta, delta / 2)
    result = solver.signal(g, delta / 2, n_sub=substeps, start="uniform", refocus=refocus).total
    assert abs(result.imag) < 1e-12
    return result.real


def extrapolate(values):
    """Four widths h,h/2,h/4,h/8; linear then delta**1.5 Richardson."""
    linear = 2 * values[1:] - values[:-1]
    factor = 2 ** 1.5
    estimate = (factor * linear[-1] - linear[-2]) / (factor - 1)
    quadratic = (8 * values[-1] - 6 * values[-2] + values[-3]) / 3
    error = abs(estimate - linear[-1]) + abs(estimate - quadratic)
    differences = np.diff(values)
    order = np.log2(abs(differences[-2] / differences[-1]))
    return float(estimate), float(error), float(order)


def compare(geometry, kappa, q, Delta, *, base_cells, deltas):
    sgp = spectral(geometry, kappa, D_IN, D_OUT, 10, 12).echo(q, Delta, nodes=64, start="uniform")
    estimates, pulse_errors, orders = [], [], []
    for level in range(3):
        solver = finite_volume(geometry, kappa, base_cells * 2 ** level, 12)
        values = np.array([pulse_echo(solver, q, Delta, delta, 2 ** level) for delta in deltas])
        intercept, pulse_error, order = extrapolate(values)
        estimates.append(intercept)
        pulse_errors.append(pulse_error)
        orders.append(order)
    grid_changes = np.abs(np.diff(estimates))
    grid_order = np.log2(grid_changes[0] / grid_changes[1])
    grid_error = grid_changes[-1] / 3
    solver = finite_volume(geometry, kappa, 4 * base_cells, 12)
    extra_time = np.array([pulse_echo(solver, q, Delta, delta, 8) for delta in deltas])
    time_error = abs(extrapolate(extra_time)[0] - estimates[-1])
    extra_solver = finite_volume(geometry, kappa, 4 * base_cells, 14)
    extra_angular = np.array([pulse_echo(extra_solver, q, Delta, delta, 8) for delta in deltas])
    angular_error = abs(extrapolate(extra_angular)[0] - extrapolate(extra_time)[0])
    fv_error = grid_error + pulse_errors[-1] + time_error + angular_error
    return dict(geometry=geometry, kappa=kappa, q=q, Delta=Delta, sgp=sgp.value,
                fv=estimates[-1], difference=estimates[-1] - sgp.value,
                sgp_error=sgp.error, fv_error=fv_error, delta_order=orders[-1],
                grid_order=grid_order, grid_error=grid_error, pulse_error=pulse_errors[-1],
                time_error=time_error, angular_error=angular_error)


def report(row):
    print("{geometry:8s} {kappa:.0e} {Delta:.3f} {q:.7e} {sgp:.10f} {fv:.10f} "
          "{difference:+.3e} {sgp_error:.2e} {fv_error:.2e} {delta_order:.3f} "
          "{grid_order:.3f}".format(**row), flush=True)


def q_values(sp, Delta):
    """First crossing of each target on an explicit SGP-only search grid."""
    grid = np.linspace(0.0, 1e6, 41)
    signals = np.array([sp.integrate(q, Delta, nodes=64, start="uniform")[0] for q in grid])
    result = []
    for target in (.9, .5, .05):
        crossings = np.flatnonzero((signals[:-1] - target) * (signals[1:] - target) <= 0)
        if len(crossings) == 0:
            raise ValueError(("target outside explicit q search", target, signals))
        j = crossings[0]
        result.append(brentq(lambda q: sp.integrate(q, Delta, nodes=64, start="uniform")[0] - target,
                             grid[j], grid[j + 1], xtol=1e-4))
    return result


def assert_comparison(row, budget):
    assert 1.5 < row["grid_order"] < 2.5, ("spatial refinement", row)
    assert row["sgp_error"] + row["fv_error"] <= budget, row
    assert abs(row["difference"]) <= row["sgp_error"] + row["fv_error"], row
    assert 0.7 < row["delta_order"] < 1.2, ("leading delta order", row)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_closed_form_sealed_intra(geometry):
    sp = spectral(geometry, 0., D_IN, D_OUT, 10, 12)
    for q in (*QS, 0.0):
        echo = sp.echo(q, .02, nodes=64, start="intra")
        exact = single_pore(geometry, q, .02, R, D_IN, modes_per_order=14, orders=16)
        assert abs(echo.value - exact) <= echo.error, (geometry, q, echo, exact)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_continuous_equal_diffusivity_is_single_outer_pore(geometry):
    sp = spectral(geometry, np.inf, D_IN, D_IN, 10, 12)
    for q in (*QS, 0.0):
        echo = sp.echo(q, .02, nodes=64, start="uniform")
        exact = single_pore(geometry, q, .02, R_OUT, D_IN, modes_per_order=14, orders=16)
        assert abs(echo.value - exact) <= echo.error, (geometry, q, echo, exact)


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("kappa", KAPPAS)
def test_zero_q_and_completeness_counts(geometry, kappa):
    sp = spectral(geometry, kappa, D_IN, D_OUT, 10, 12)
    assert all(count == (10, 10, 10) for count in sp.counts)
    for q in (0.0, 1e-3):
        echo = sp.echo(q, .05, nodes=64, start="uniform")
        assert abs(echo.value - 1) <= echo.error


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("kappa", KAPPAS)
def test_finite_kappa_narrow_pulse_fast(geometry, kappa):
    row = compare(geometry, kappa, QS[1], .02, base_cells=3, deltas=DELTAS)
    report(row)
    assert_comparison(row, 2e-4)


@pytest.mark.slow
def test_full_requested_grid():
    started = monotonic()
    print("geometry kappa Delta_s q_rad/m SGP FV_extrap FV-SGP err_SGP err_FV delta_order grid_order", flush=True)
    rows = []
    for geometry in GEOMETRIES:
        for kappa in KAPPAS:
            for Delta in (.02, .05):
                sp = spectral(geometry, kappa, D_IN, D_OUT, 10, 12)
                # Pick angular q from SGP alone, before accessing FV at this point.
                qs = q_values(sp, Delta)
                for q in qs:
                    row = compare(geometry, kappa, q, Delta, base_cells=6, deltas=DELTAS / 4)
                    report(row)
                    rows.append(row)
    # Print every requested row before asserting, preserving diagnostics on failure.
    print(f"full grid seconds: {monotonic() - started:.2f}", flush=True)
    failures = []
    for row in rows:
        try:
            assert_comparison(row, 5e-5)
        except AssertionError as error:
            failures.append(str(error))
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("kappa", KAPPAS)
def test_radial_modes_are_orthogonal_in_plain_volume(geometry, kappa):
    sp = spectral(geometry, kappa, D_IN, D_OUT, 10, 12)
    z, w = np.polynomial.legendre.leggauss(96)
    a = R / R_OUT
    si, so = (z + 1) * a / 2, a + (z + 1) * (1 - a) / 2
    p = {"sphere": 3, "cylinder": 2}[geometry]
    weights = np.r_[w * a / 2 * si ** (p - 1), w * (1 - a) / 2 * so ** (p - 1)]
    for order in (0, 1, 4):
        functions = np.array([np.concatenate(sp.radial(order, root, pool, si, so))
                              for root, pool in sp.modes[order]])
        gram = (functions * weights) @ functions.T
        norms = np.sqrt(np.diag(gram))
        gram /= norms[:, None] * norms[None, :]
        assert np.max(np.abs(gram - np.eye(len(gram)))) < 2e-9
