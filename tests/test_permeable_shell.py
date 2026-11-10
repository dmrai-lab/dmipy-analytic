"""The radial finite-volume Bloch-Torrey solver against independent routes. Every test states its prediction first;
every tolerance is the solver's own error (the Richardson estimate of :func:`reference`, which refines the cells and
the steps together and checks the observed order) plus the error of the route it is held to.

Predictions, written before the first run:

1. refining the cells by two, or the steps by two, leaves a quarter of the change: the ratio of two successive changes is
   4 (accepted between 3 and 5, the range in which the Richardson estimate is within a third of the error it estimates);
2. a sealed pore is the matrix method of the sphere and of the cylinder, and its shell (cylinder) the matrix method of
   the annulus;
3. at small gradient the exponent ``-ln S`` of the sealed pore is the Gaussian-phase exponent (the exact second cumulant
   of the sphere and of the cylinder): their difference is the fourth cumulant, which falls by ``2^4`` when the gradient
   halves;
4. ``kappa = inf`` with one diffusivity is free diffusion in ``r < R_out``: the matrix method at the combined radius;
5. the slab (the planar twin) is :class:`~dmipy_analytic.bloch_torrey.BlochTorrey1D` on three layers, and its
   populations a membrane between equal layers: two Brownstein-Tarr series;
6. the uniform density is stationary: a uniform start leaves each pool's share at its volume fraction for any ``kappa``;
7. the mean residence time before the first exit is ``R^2 / (15 D) + R / (3 kappa)`` (sphere) and ``R^2 / (8 D) + R / (2
   kappa)`` (cylinder), exact for the equation, and the second-order grid reproduces it with a quartered error per
   doubling;
8. as ``kappa R / D -> 0`` the pools exchange as the two-site model, ``k = (kappa S / V_in) / (1 - f)``, the relative
   deviation of the rate being linear in ``kappa R / D``.
"""
import math

import numpy as np
import pytest

from dmipy_analytic.constants import GAMMA
from dmipy_analytic.permeable_shell import PermeableShell, ShellSpec, pgse_waveform, reference

D = 2e-9
R, R_OUT = 5e-6, 8e-6
DT = 1e-4
G_PGSE, REFOCUS = pgse_waveform(0.1, 8e-3, 20e-3, DT)          # q-space point with strong restriction (R^2/D = 12.5 ms)
TOL = 1e-5                      # the error bar the fast tests ask of the reference; the slow study goes to 1e-8
TOL_GPD = 2e-7
REF_ARGS = dict(refocus=REFOCUS, cells=(10, 6), sub_steps=1, harmonics=8, max_levels=6)


def _grid(n):
    return dict(n_in=n, n_out=int(round(n * (R_OUT - R) / R)))


def _richardson(values):
    """``(error estimate of the finest, observed ratio)`` of three results at a resolution doubled twice."""
    v = np.asarray(values)
    d1, d2 = abs(v[0] - v[1]), abs(v[1] - v[2])
    return d2 / 3.0, d1 / d2                  # elementwise for arrays


@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
@pytest.mark.parametrize("kappa", [0.0, 3e-5, 3e-4, np.inf])
def test_refining_the_cells_or_the_steps_by_two_leaves_a_quarter_of_the_change(geometry, kappa):
    spec = ShellSpec(geometry, R, R_OUT, D, 0.7 * D, kappa)
    for name in ("cells", "steps"):
        values = []
        for lev in range(3):
            n, ns = (10 * 2 ** lev, 8) if name == "cells" else (80, 1 * 2 ** lev)
            sig = PermeableShell(spec, n_harmonics=8, **_grid(n)).signal(G_PGSE, DT, n_sub=ns, start="uniform",
                                                                         refocus=REFOCUS)
            values.append(sig.total)
        err, ratio = _richardson(values)
        assert 3.0 < ratio < 5.0, (geometry, kappa, name, ratio, values)


@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
def test_a_sealed_pore_is_the_matrix_method(geometry):
    from dmipy_analytic.restricted.matrix_method import MatrixPore
    pytest.importorskip("scipy.linalg")
    spec = ShellSpec(geometry, R, R_OUT, D, D, 0.0)
    ref = reference(spec, G_PGSE, DT, start="intra", tol=TOL, **REF_ARGS)
    mm = [MatrixPore(geometry, 2 * R, D, n_modes=m).signal(G_PGSE, DT, refocus=REFOCUS) for m in ((24, 18), (36, 30))]
    assert abs(ref.intra - mm[1]) <= 2.0 * (ref.error_intra + abs(mm[0] - mm[1])), (ref, mm)
    assert abs(ref.extra) <= ref.error_extra + 1e-15          # nothing starts outside, nothing leaks out


def test_the_shell_of_a_sealed_cylinder_is_the_annulus_of_the_matrix_method():
    from dmipy_analytic.restricted.matrix_method import AnnulusPore
    pytest.importorskip("scipy.linalg")
    spec = ShellSpec("cylinder", R, R_OUT, D, D, 0.0)
    ref = reference(spec, G_PGSE, DT, start="extra", tol=TOL, **REF_ARGS)
    an = [AnnulusPore(2 * R_OUT, 2 * R, D, n_modes=m).signal(G_PGSE, DT, refocus=REFOCUS) for m in ((24, 8), (28, 10))]
    assert abs(ref.extra - an[1]) <= 2.0 * (ref.error_extra + abs(an[0] - an[1])), (ref, an)


@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
def test_the_exponent_at_small_gradient_is_the_gaussian_phase_exponent_to_the_fourth_cumulant(geometry):
    pytest.importorskip("scipy.special")
    if geometry == "sphere":
        from dmipy_analytic.restricted.sphere import gaussian_phase
    else:
        from dmipy_analytic.restricted.cylinder import gaussian_phase
    spec = ShellSpec(geometry, R, R_OUT, D, D, 0.0)
    gaps, errs = [], []
    for G in (0.08, 0.04):
        g, rf = pgse_waveform(G, 8e-3, 20e-3, DT)
        ref = reference(spec, g, DT, refocus=rf, start="intra", tol=TOL_GPD, cells=(10, 6), sub_steps=1, harmonics=8,
                        max_levels=8)
        gpd = float(gaussian_phase(G, 8e-3, 20e-3, 2 * R, D))
        gaps.append(abs(-math.log(ref.intra.real) + math.log(gpd)))
        errs.append(ref.error_intra / ref.intra.real)
    assert 12.0 < gaps[0] / gaps[1] < 20.0, (gaps, errs)
    assert max(errs) < 0.02 * gaps[1]


@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
def test_an_open_membrane_with_one_diffusivity_is_free_diffusion_in_the_whole_shell(geometry):
    from dmipy_analytic.restricted.matrix_method import MatrixPore
    pytest.importorskip("scipy.linalg")
    spec = ShellSpec(geometry, R, R_OUT, D, D, np.inf)
    ref = reference(spec, G_PGSE, DT, start="uniform", tol=TOL, **REF_ARGS)
    mm = [MatrixPore(geometry, 2 * R_OUT, D, n_modes=m).signal(G_PGSE, DT, refocus=REFOCUS) for m in ((24, 18), (36, 30))]
    assert abs(ref.total - mm[1]) <= 2.0 * (ref.error_total + abs(mm[0] - mm[1])), (ref, mm)


def test_the_slab_is_the_one_dimensional_solver_on_three_layers():
    from dmipy_analytic.bloch_torrey import BlochTorrey1D
    a, kappa = 2e-6, 3e-4
    spec = ShellSpec("slab", a, 2 * a, D, D, kappa)
    ref = reference(spec, G_PGSE, DT, refocus=REFOCUS, start="uniform", tol=TOL, cells=(10, 10), sub_steps=1,
                    harmonics=2, max_levels=6)
    edges = [-2 * a, -a, a, 2 * a]
    other = [BlochTorrey1D(edges, D, permeability=kappa, n_cells=n).signal(G_PGSE, DT, refocus=REFOCUS, sub_steps=s)
             for n, s in ((100, 4), (200, 8), (400, 16))]
    err_other, ratio = _richardson(other)
    assert 3.0 < ratio < 5.0
    assert abs(ref.total - other[2]) <= 2.0 * (ref.error_total + err_other), (ref, other)


@pytest.mark.parametrize("kappa", [1e-5, 1e-4, 1e-3])
def test_the_slab_populations_are_a_membrane_between_equal_layers(kappa):
    from dmipy_analytic.relaxation import surface_relaxation_plane
    a = 2e-6
    t = np.array([0.1, 0.4, 1.6]) * a / (2.0 * kappa)         # fractions of the membrane's own exchange time
    spec = ShellSpec("slab", a, 2 * a, D, D, kappa)
    solvers = [PermeableShell(spec, n_in=n, n_out=n, n_harmonics=2) for n in (50, 100, 200)]
    got = [s.exchange(t) for s in solvers]
    err, ratio = _richardson(np.array(got)[:, 0, 0])
    odd = surface_relaxation_plane(t, 2 * a, D, 2 * kappa)
    pred = (1.0 + odd) / 2.0                                  # the share of the first layer that started in it
    assert np.all((ratio > 3.0) & (ratio < 5.0)), ratio
    assert np.all(np.abs(got[2][0, 0] - pred) <= 2.0 * (err + 2.0 / (np.pi ** 2 * 400))), (got[2][0, 0], pred)


@pytest.mark.parametrize("geometry", ["sphere", "cylinder", "slab"])
@pytest.mark.parametrize("kappa", [0.0, 1e-5, np.inf])
def test_the_uniform_density_is_stationary_and_the_rows_of_the_exchange_matrix_sum_to_one(geometry, kappa):
    spec = ShellSpec(geometry, R, R_OUT, D, 0.4 * D, kappa)
    s = PermeableShell(spec, n_in=30, n_out=20, n_harmonics=2)
    t = np.array([0.0, 1e-3, 1e-2, 1.0])
    P = s.exchange(t)
    d = {"slab": 1, "cylinder": 2, "sphere": 3}[geometry]
    f = (R / R_OUT) ** d
    lam = s.lam[0]
    nonzero = lam[lam > s.n * np.finfo(float).eps * lam.max()]
    # rounding: the eigenvectors of the near-null modes are known to eps * ||A|| / gap
    atol = s.n ** 1.5 * np.finfo(float).eps * lam.max() / nonzero.min()
    assert np.allclose(P.sum(axis=1), 1.0, rtol=0, atol=atol)
    # a uniform start: f P_ii + (1 - f) P_ji stays f
    assert np.allclose(f * P[0, 0] + (1 - f) * P[1, 0], f, rtol=0, atol=atol)
    sig = s.signal(np.zeros(5), 2e-3, n_sub=2, start="uniform")
    assert abs(sig.intra - f) < atol and abs(sig.extra - (1 - f)) < atol


@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
@pytest.mark.parametrize("kappa", [1e-6, 1e-4, np.inf])
def test_the_mean_residence_time_is_its_closed_form_to_second_order(geometry, kappa):
    spec = ShellSpec(geometry, R, R_OUT, 1.3 * D, D, kappa)
    c, k = (1.0 / 15.0, 3.0) if geometry == "sphere" else (1.0 / 8.0, 2.0)
    exact = c * R ** 2 / (1.3 * D) + (R / (k * kappa) if np.isfinite(kappa) else 0.0)
    got = [PermeableShell(spec, n_harmonics=1, **_grid(n)).mean_residence_time("intra") for n in (25, 50, 100)]
    errs = [abs(x - exact) for x in got]
    assert errs[2] < errs[1] < errs[0]
    assert 3.0 < errs[1] / errs[2] < 5.0 or errs[2] < 8 * np.finfo(float).eps * exact, errs
    assert errs[2] <= 1.5 * abs(got[2] - got[1]) / 3.0 + 4 * np.finfo(float).eps * exact


@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
def test_slow_exchange_is_the_two_site_kinetics_at_first_order_in_the_permeability(geometry):
    d, k_surface = (3, 3.0) if geometry == "sphere" else (2, 2.0)
    f = (R / R_OUT) ** d
    dev = []
    kappas = (1e-7, 1e-6)
    for kappa in kappas:
        s = PermeableShell(ShellSpec(geometry, R, R_OUT, D, D, kappa), n_in=100, n_out=60, n_harmonics=1)
        pred = (k_surface * kappa / R) / (1.0 - f)             # k_ie + k_ei, k_ie = kappa S / V, k_ei = k_ie f / (1-f)
        dev.append((s.exchange_rate() / pred - 1.0) / kappa)
        t = np.array([0.1 / pred])
        P = s.exchange(t)
        two_site = f + (1 - f) * np.exp(-pred * t)
        assert abs(P[0, 0, 0] - two_site[0]) < 1.1 * (1 - f) * abs(dev[-1]) * kappa * 0.5 + 1e-9
    assert dev[0] < 0.0 and abs(dev[0] / dev[1] - 1.0) < 0.1, dev


def test_a_missing_parameter_is_an_error_named_after_it():
    with pytest.raises(TypeError, match="kappa"):
        ShellSpec(geometry="sphere", R=R, R_out=R_OUT, D_in=D, D_out=D)
    spec = ShellSpec("sphere", R, R_OUT, D, D, 0.0)
    with pytest.raises(TypeError, match="n_harmonics"):
        PermeableShell(spec, n_in=10, n_out=10)
    with pytest.raises(TypeError, match="start"):
        PermeableShell(spec, n_in=10, n_out=10, n_harmonics=2).signal(G_PGSE, DT, n_sub=1)
    with pytest.raises(TypeError, match="tol"):
        reference(spec, G_PGSE, DT, refocus=(), start="uniform", cells=(10, 10), sub_steps=1, harmonics=4, max_levels=3)
    with pytest.raises(ValueError, match="kappa"):
        PermeableShell(ShellSpec("sphere", R, R_OUT, D, D, -1.0), n_in=10, n_out=10, n_harmonics=2)
    with pytest.raises(ValueError, match="geometry"):
        PermeableShell(ShellSpec("torus", R, R_OUT, D, D, 0.0), n_in=10, n_out=10, n_harmonics=2)
    with pytest.raises(ValueError, match="start"):
        PermeableShell(spec, n_in=10, n_out=10, n_harmonics=2).signal(G_PGSE, DT, n_sub=1, start="everywhere")
    with pytest.raises(ValueError, match="whole multiple"):
        pgse_waveform(0.1, 8.5e-5, 2e-2, DT)


@pytest.mark.slow
@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
@pytest.mark.parametrize("kappa", [1e-5, 1e-4, 1e-3])
def test_the_refinement_tables_of_a_permeable_pore_under_a_strong_gradient(geometry, kappa, capsys):
    """The error bar of the reference where no closed form exists (finite ``kappa``, two diffusivities): three
    resolutions, the observed order, the truncation in harmonics, at the gradient where the restricted signal has lost
    half its amplitude."""
    spec = ShellSpec(geometry, R, R_OUT, D, 0.6 * D, kappa)
    g, rf = pgse_waveform(0.25, 8e-3, 20e-3, DT)
    rows = []
    for lev in range(4):
        ni, no, ns = 10 * 2 ** lev, 6 * 2 ** lev, 2 ** lev
        sig = PermeableShell(spec, n_in=ni, n_out=no, n_harmonics=14).signal(g, DT, n_sub=ns, start="uniform", refocus=rf)
        rows.append((ni, no, ns, sig.total.real))
    ch = [abs(rows[i][3] - rows[i + 1][3]) for i in range(3)]
    with capsys.disabled():
        print(f"\n{geometry} kappa={kappa:g}: " + "; ".join(f"n={r[0]}+{r[1]} sub={r[2]} S={r[3]:.9f}" for r in rows)
              + f"; changes {ch}, ratios {ch[0] / ch[1]:.3f} {ch[1] / ch[2]:.3f}")
    assert 3.0 < ch[0] / ch[1] < 5.0 and 3.0 < ch[1] / ch[2] < 5.0
    trunc = [PermeableShell(spec, n_harmonics=L, **_grid(20)).signal(g, DT, n_sub=2, start="uniform",
                                                                     refocus=rf).total for L in (8, 10, 12, 14)]
    assert abs(trunc[3] - trunc[2]) < abs(trunc[2] - trunc[1]) < abs(trunc[1] - trunc[0]) or abs(trunc[3] - trunc[2]) < 1e-13
