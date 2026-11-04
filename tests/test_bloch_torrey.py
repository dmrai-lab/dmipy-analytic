"""The 1-D Bloch-Torrey reference against independent routes, each tolerance derived from the scheme's own error.

The scheme is second order in the cell width and in the step, so its error at a resolution is estimated by
Richardson's rule from the change when the resolution doubles, ``e(n) ~ (S(n) - S(2n)) * 4/3`` (``1/3`` of the change
between the doubled resolution and the one after). :func:`_richardson` measures that change and the ratio of two
successive changes, and every comparison first asserts that the ratio is the second order's 4 (between 3 and 5: the
estimate is then within a third of the error it estimates), then allows twice the estimated error of each side (the
reference's cells, its steps, and the form's own truncation) and nothing else.

The independent routes:

* a sealed membrane (and no membrane at all) under a pulsed gradient with a 180: the plane of the matrix method
  (:class:`~dmipy_analytic.restricted.matrix_method.MatrixPore`), exact up to its modes;
* Robin walls: the Brownstein-Tarr series of two plates (:func:`~dmipy_analytic.relaxation.surface_relaxation_plane`);
* a membrane between two equal layers, every spin starting in the first: by symmetry the even part of the magnetisation
  never crosses and the odd part sees a Robin wall of relaxivity ``2 kappa`` (plus the membrane's own ``rho_e`` on
  both parts), so each layer's population is a sum of two Brownstein-Tarr series, exactly; and when the permeability
  changes at a time ``t_b``, the odd part's series at ``t_b`` projected on the Robin modes of the new ``2 kappa'`` and
  continued in them (the overlaps of two cosine families in closed form);
* slow exchange and slow relaxation against diffusion: the two-site exchange of Karger's model
  (:func:`~dmipy_analytic.exchange.karger`), approached at first order in ``kappa a / D`` and ``R2 a^2 / D``;
* a static gradient next to a wall: the localisation regime, the Hahn echo decaying at long times at the rate
  ``|a'_1| / 2 (gamma^2 g^2 D)^(1/3)`` per unit echo time, ``a'_1 = -1.01879297`` the first zero of ``Ai'``. Stoller,
  Happer & Dyson (1991), Phys. Rev. A 44:7459, Eq. 3.19: the poles of the one-boundary resolvent are
  "s = s_n = il - a'_n e^{i alpha} for n = 1, 2, 3, ..., where a'_n is one of the zeros of the entire function Ai'
  [...] a'_1 = -1.01879297", with ``alpha = 2 pi / 3`` (their Eq. 2.16) in the units ``x = b^(1/3) zeta``,
  ``t = b^(2/3) tau`` (Eq. 2.1), so the slowest mode decays at ``-a'_1 cos(alpha) = |a'_1| / 2`` in units of
  ``(gamma^2 g^2 D)^(1/3)``. De Swiet & Sen (1994), J. Chem. Phys. 100:5597, state the same zeros ("a_1 = -1.0188,
  a_2 = -3.2482, a_3 = -4.8201") and, in the caption of their Fig. 2 (the Hahn echo of a slab), "the long-time decay
  rate -0.5094 D^(1/3) g^(2/3) 2 tau" at the echo time ``2 tau``, ``g`` the gyromagnetic ratio times the gradient.
"""
import numpy as np
import pytest

from dmipy_analytic.bloch_torrey import BlochTorrey1D
from dmipy_analytic.constants import GAMMA

D = 2e-9
A1 = 1.01879297                     # |a'_1|, Stoller, Happer & Dyson (1991) Eq. 3.19
A2 = 3.24819758                     # |a'_2|


def _richardson(values):
    """``(error estimate of values[0], ratio)`` for three results (scalars or arrays of them) at a resolution doubled
    twice: the error of the first is ``4/3`` of its change to the second, and the ratio of the two changes, read on the
    entry that changes most, is 4 for a second-order scheme."""
    v = np.asarray(values)
    d1, d2 = v[0] - v[1], v[1] - v[2]
    worst = np.unravel_index(np.argmax(np.abs(d1)), np.shape(d1)) if np.ndim(d1) else ()
    return np.abs(d1) * 4.0 / 3.0, float(np.abs(d1[worst]) / max(float(np.abs(d2[worst])), 1e-300))


def _pgse(G, delta, Delta, dt):
    """A pulsed gradient pair of amplitude ``G`` as played (both lobes positive, the 180 between them at TE/2)
    on steps of ``dt``: ``(G, refocus step, n_t)``."""
    nd, nD = int(round(delta / dt)), int(round(Delta / dt))
    n_t = nD + nd + 2 * nd
    g = np.zeros(n_t)
    start = nd
    g[start:start + nd] = G
    g[start + nD:start + nD + nd] = G
    return g, (start + start + nD + nd) // 2


def _second_order(values):
    err, ratio = _richardson(values)
    assert 3.0 < ratio < 5.0, f"the refinement is not in the second-order regime: ratio {ratio:.2f}, values {values}"
    return err


def test_a_sealed_membrane_and_no_membrane_are_the_plane_of_the_matrix_method():
    from dmipy_analytic.restricted.matrix_method import MatrixPore
    pytest.importorskip("scipy.linalg")
    a, dt = 4e-6, 1e-4
    G, k180 = _pgse(0.12, 0.008, 0.02, dt)
    for edges, kwargs, width in (([0.0, a, 2 * a], dict(permeability=0.0, density=(1.0, 0.0)), a),
                                 ([0.0, a, 2 * a], dict(permeability=np.inf, R2=(5.0, 5.0)), 2 * a)):
        decay = np.exp(-kwargs.get("R2", (0.0,))[0] * len(G) * dt)
        by_cells = [BlochTorrey1D(edges, D, n_cells=n, **kwargs).signal(G, dt, refocus=(k180,))
                    for n in (100, 200, 400)]
        by_steps = [BlochTorrey1D(edges, D, n_cells=100, **kwargs).signal(G, dt, refocus=(k180,), sub_steps=s)
                    for s in (1, 2, 4)]
        e_cells, e_steps = _second_order(by_cells), _second_order(by_steps)
        mm = [MatrixPore("plane", width, D, n_modes=m).signal(G, dt, refocus=(k180,)) * decay for m in (24, 48)]
        e_modes = abs(mm[0] - mm[1])
        assert abs(by_cells[0] - mm[1]) <= 2.0 * (e_cells + e_steps + e_modes), (by_cells, by_steps, mm)


@pytest.mark.parametrize("rho_a_over_D", [0.1, 1.0, 10.0])
def test_robin_walls_are_the_brownstein_tarr_series(rho_a_over_D):
    from dmipy_analytic.relaxation import surface_relaxation_plane
    L, dt = 6e-6, 2e-4
    rho = rho_a_over_D * D / (L / 2)
    n_t = (25, 100)                                           # 5 and 20 ms
    got = [np.array([BlochTorrey1D([0.0, L], D, wall_relaxivity=rho, n_cells=n).signal(np.zeros(m), dt).real
                     for m in n_t]) for n in (100, 200, 400)]
    e_cells = _second_order(got)
    ref = surface_relaxation_plane(np.array(n_t) * dt, L, D, rho)
    tail = 2.0 / (np.pi ** 2 * 400)                           # the series' truncation at t = 0, a bound at t > 0
    assert np.all(np.abs(got[0] - ref) <= 2.0 * (e_cells + tail)), (got[0], ref, e_cells)


@pytest.mark.parametrize("kappa, rho_e", [(1e-5, 0.0), (1e-4, 0.0), (1e-5, 2e-5)])
def test_a_membrane_between_equal_layers_is_two_brownstein_tarr_series(kappa, rho_e):
    """Every spin starts in the first layer: the even part (1/2 everywhere) sees a Robin wall of ``rho_e`` at the
    membrane, the odd part (+1/2, -1/2) one of ``2 kappa + rho_e``; each part of a layer of width ``a`` with a
    reflecting outer wall is half of two plates ``2a`` apart. The first layer holds ``(S(rho_e) + S(2 kappa + rho_e)) /
    2`` and the second ``(S(rho_e) - S(2 kappa + rho_e)) / 2`` of the start."""
    from dmipy_analytic.relaxation import surface_relaxation_plane
    a, dt = 2e-6, 2e-4
    n_t = (10, 50, 250)                                       # 2, 10 and 50 ms
    got = [np.array([BlochTorrey1D([0.0, a, 2 * a], D, permeability=kappa, membrane_relaxivity=rho_e,
                                   density=(1.0, 0.0), n_cells=n).signal(np.zeros(m), dt, by_layer=True).real
                     for m in n_t]) for n in (100, 200, 400)]
    e_cells = _second_order(got)
    t = np.array(n_t) * dt
    even, odd = surface_relaxation_plane(t, 2 * a, D, rho_e), surface_relaxation_plane(t, 2 * a, D, 2 * kappa + rho_e)
    ref = np.stack([(even + odd) / 2, (even - odd) / 2], axis=-1)
    tail = 2.0 / (np.pi ** 2 * 400)
    assert np.all(np.abs(got[0] - ref) <= 2.0 * (e_cells + tail)), (got[0], ref, e_cells)


def test_slow_exchange_and_relaxation_tend_to_the_two_site_karger_model_at_first_order():
    """Two layers of widths ``a_1``, ``a_2``, uniform start, ``R2`` per layer, no gradient: when diffusion mixes each
    layer faster than the membrane and the relaxation act (``kappa a / D`` and ``R2 a^2 / D`` small), each layer is a
    well-mixed site and the populations obey ``dM/dt = (K - R2) M`` with the rates ``kappa / a_1`` and ``kappa / a_2``:
    Karger's model with the exponents ``R2 t``. Both small parameters scale as ``1 / D``, so the departure falls as
    ``1 / D``: four times the diffusivity, a quarter of the departure (first order: the ratio within 3 to 5)."""
    from dmipy_analytic.exchange import karger
    a1, a2, kappa, R2, t = 2e-6, 3e-6, 2e-5, (40.0, 10.0), 0.03
    dt = 2e-4
    n_t = int(round(t / dt))
    f = a1 / (a1 + a2)
    ref = karger(R2[0] * t, R2[1] * t, kappa / a1, f, t)
    departure = []
    for Dx in (2e-9, 8e-9, 32e-9):
        S = BlochTorrey1D([0.0, a1, a1 + a2], Dx, permeability=kappa, R2=R2, n_cells=500).signal(np.zeros(n_t), dt)
        departure.append(abs(S.real - ref))
    ratios = departure[0] / departure[1], departure[1] / departure[2]
    assert all(3.0 < r < 5.0 for r in ratios), (departure, ratios)


def test_a_background_gradient_next_to_a_wall_decays_at_the_airy_rate():
    """Plates 20 ``l_g`` apart (``l_g = (D / gamma g)^(1/3)``), a static gradient ``g`` and nothing else: the Hahn echo's
    logarithmic slope between the echo times 20 and 24 (in units of ``(gamma^2 g^2 D)^(-1/3)``) is the Airy rate
    ``|a'_1| / 2``. Its errors, each estimated and summed: the next mode, whose share of the echo falls as
    ``exp(-(|a'_2| - |a'_1|) TE / 4)`` (its slope error between the pairs (12, 16) and (20, 24) shrinks by that factor
    over 8 units, measured on the first pair), the cells and the step (Richardson). The two walls' coupling is
    transcendentally small at 20 ``l_g`` (de Swiet & Sen 1994, Eq. 17)."""
    l_g = 1e-6
    g0 = D / l_g ** 3 / GAMMA
    unit = (GAMMA ** 2 * g0 ** 2 * D) ** (1.0 / 3.0)
    L, dt = 20 * l_g, 0.02 / unit
    theory = A1 / 2.0

    def rate(n, sub, te):
        bt = BlochTorrey1D([-L / 2, L / 2], D, background_gradient=g0, n_cells=n)
        lnE = []
        for TE in te:
            n_t = 2 * int(round(TE / unit / dt / 2))
            lnE.append(np.log(abs(bt.signal(np.zeros(n_t), dt, refocus=(n_t // 2,), sub_steps=sub))))
        return -(lnE[1] - lnE[0]) / ((te[1] - te[0]) / unit) / unit

    late, early = (20.0, 24.0), (12.0, 16.0)
    r = rate(300, 2, late)
    e_mode = abs(rate(300, 2, early) - r) * np.exp(-(A2 - A1) / 4.0 * 8.0)
    e_cells = abs(r - rate(600, 2, late)) * 4.0 / 3.0
    e_steps = abs(r - rate(300, 4, late)) * 4.0 / 3.0
    assert abs(r - theory) <= 2.0 * (e_mode + e_cells + e_steps), (r, theory, e_mode, e_cells, e_steps)
    assert 2.0 * (e_mode + e_cells + e_steps) < 1e-3 * theory          # the check resolves the constant to 0.1 %


def test_the_180_conjugates_and_the_off_resonance_precesses_as_exp_minus_i_omega_t():
    """No gradient, uniform ``omega`` and ``R2``: the free induction decays as ``exp(-(R2 + i omega) t)`` and an
    ideal 180 at half the echo time refocuses the phase exactly, to rounding."""
    R2, omega, dt, n_t = 25.0, 300.0, 1e-4, 200
    bt = BlochTorrey1D([0.0, 2e-6, 5e-6], D, permeability=3e-5, R2=R2, omega=omega, n_cells=60)
    t = n_t * dt
    assert bt.signal(np.zeros(n_t), dt) == pytest.approx(np.exp(-(R2 + 1j * omega) * t), rel=1e-12, abs=1e-14)
    assert bt.signal(np.zeros(n_t), dt, refocus=(n_t // 2,)) == pytest.approx(np.exp(-R2 * t), rel=1e-12, abs=1e-14)


def test_a_waveform_stack_is_its_rows():
    bt = BlochTorrey1D([0.0, 2e-6, 4e-6], D, permeability=1e-5, R2=(12.5, 50.0), omega=(0.0, 200.0),
                       density=(1.0, 0.0), n_cells=80)
    G = np.zeros((3, 120))
    G[1, 10:30] = G[1, 70:90] = 0.1
    G[2, 10:30] = 0.2
    rows = bt.signal(G, 1e-4, refocus=(60,))
    for k in range(3):
        assert rows[k] == pytest.approx(bt.signal(G[k], 1e-4, refocus=(60,)), rel=1e-12, abs=1e-15)
    layers = bt.signal(G, 1e-4, refocus=(60,), by_layer=True)
    assert layers.shape == (3, 2) and np.allclose(layers.sum(-1), rows, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("kwargs", [dict(edges_m=[0.0, 0.0]), dict(edges_m=[0.0, 1e-6], diffusivity=-1.0),
                                    dict(edges_m=[0.0, 1e-6, 2e-6], permeability=(1e-5, 1e-5)),
                                    dict(edges_m=[0.0, 1e-6, 2e-6], permeability=np.inf, membrane_relaxivity=1e-5),
                                    dict(edges_m=[0.0, 1e-6], density=0.0)])
def test_an_ill_posed_interval_is_refused(kwargs):
    kw = dict(diffusivity=D)
    kw.update(kwargs)
    with pytest.raises(ValueError):
        BlochTorrey1D(kw.pop("edges_m"), kw.pop("diffusivity"), **kw)


def _robin_odd_part(a, kappas, t_b, t, n_modes=400):
    """The odd part's share of the first of two equal layers of width ``a`` (every spin starting in it) at the times
    ``t``, the permeability ``kappas[0]`` before ``t_b`` and ``kappas[1]`` after: the odd part ``+-1/2`` of the start
    is the even (cosine) family of a plate pore of half-width ``a`` with the Robin relaxivity ``2 kappa``,
    ``cos(xi x / a)``, ``xi tan xi = 2 kappa a / D`` (:func:`~dmipy_analytic.relaxation.plane_robin_roots`). Its
    coefficients at ``t_b`` are projected on the second family by the overlaps ``int_0^a cos(xi x/a) cos(xi' x/a) dx``
    in closed form, then continued in it. Returns the first layer's population, ``1/2`` plus the odd part's mass."""
    from dmipy_analytic.relaxation import plane_robin_roots

    def family(kappa):
        xi = plane_robin_roots(2.0 * kappa * a / D, n_modes)
        norm = a * (0.5 + 0.5 * np.sinc(2.0 * xi / np.pi))             # int_0^a cos^2; a at xi = 0 (a sealed membrane)
        return xi, norm, D * (xi / a) ** 2, a * np.sinc(xi / np.pi)     # roots, norms, rates, the mode's mass

    xi1, n1, r1, m1 = family(kappas[0])
    xi2, n2, r2, m2 = family(kappas[1])
    c1 = 0.5 * m1 / n1                                                  # the start +1/2 on the first family
    p, q = xi1[:, None], xi2[None, :]
    overlap = 0.5 * a * (np.sinc((p - q) / np.pi) + np.sinc((p + q) / np.pi))
    c2 = (c1 * np.exp(-r1 * t_b)) @ overlap / n2                        # the state at t_b on the second family
    t = np.asarray(t, dtype=np.float64)
    before = (c1 * m1 * np.exp(-np.multiply.outer(np.minimum(t, t_b), r1))).sum(-1)
    after = (c2 * m2 * np.exp(-np.multiply.outer(np.maximum(t - t_b, 0.0), r2))).sum(-1)
    return 0.5 + np.where(t <= t_b, before, after) / a


def test_a_permeability_that_keeps_its_value_at_a_breakpoint_is_the_constant_one():
    """A breakpoint at which every edge keeps its permeability changes nothing but the eigenbasis the state is carried
    in, which is the same one: the populations of three layers are the constant model's to rounding."""
    edges, kappa = [0.0, 2e-6, 3e-6, 5e-6], (1e-5, 4e-5)
    const = BlochTorrey1D(edges, D, permeability=kappa, density=(1.0, 0.0, 0.0), n_cells=120)
    split = BlochTorrey1D(edges, D, permeability=(kappa, kappa), permeability_breakpoints=(0.004,),
                          density=(1.0, 0.0, 0.0), n_cells=120)
    for n_t in (10, 40, 60):
        a = const.signal(np.zeros(n_t), 2e-4, by_layer=True)
        b = split.signal(np.zeros(n_t), 2e-4, by_layer=True)
        assert np.allclose(a, b, rtol=0, atol=1e-13), (n_t, a, b)


@pytest.mark.parametrize("kappas", [(1e-5, 1e-4), (1e-4, 0.0), (0.0, 3e-5)])
def test_a_step_change_of_permeability_is_the_restarted_brownstein_tarr_series(kappas):
    """Two equal layers, every spin starting in the first, the permeability ``kappas[0]`` before ``t_b = 6 ms`` and
    ``kappas[1]`` after (sealing the membrane, and opening a sealed one, among them): the populations at 2, 6, 10 and
    40 ms against the odd part's series restarted on the new Robin modes at ``t_b`` (:func:`_robin_odd_part`), within
    twice the scheme's own error (Richardson on the cells) and the series' truncation."""
    a, dt, t_b = 2e-6, 2e-4, 0.006
    n_t = (10, 30, 50, 200)
    got = [np.array([BlochTorrey1D([0.0, a, 2 * a], D, permeability=kappas, permeability_breakpoints=(t_b,),
                                   density=(1.0, 0.0), n_cells=n).signal(np.zeros(m), dt, by_layer=True).real
                     for m in n_t]) for n in (100, 200, 400)]
    e_cells = _second_order(got)
    first = _robin_odd_part(a, kappas, t_b, np.array(n_t) * dt)
    ref = np.stack([first, 1.0 - first], axis=-1)
    tail = 2.0 / (np.pi ** 2 * 400)
    assert np.all(np.abs(got[0] - ref) <= 2.0 * (e_cells + tail)), (got[0], ref, e_cells)


def test_the_restart_is_needed():
    """The same step change walked at the first permeability throughout misses the restarted series by far more than
    the scheme's error: the comparison above resolves the change."""
    a, dt, t_b = 2e-6, 2e-4, 0.006
    stuck = BlochTorrey1D([0.0, a, 2 * a], D, permeability=1e-5, density=(1.0, 0.0), n_cells=200)
    first = _robin_odd_part(a, (1e-5, 1e-4), t_b, 0.01)
    assert abs(stuck.signal(np.zeros(50), dt, by_layer=True).real[0] - first) > 0.02


@pytest.mark.parametrize("kwargs", [dict(permeability=(1e-5, 2e-5), permeability_breakpoints=(0.001, 0.002)),
                                    dict(permeability=1e-5, permeability_breakpoints=(0.001,)),
                                    dict(permeability=(1e-5, 2e-5), permeability_breakpoints=(-0.001,)),
                                    dict(permeability=(1e-5, 2e-5, 3e-5), permeability_breakpoints=(0.002, 0.001)),
                                    dict(permeability=((1e-5, 1e-5), 2e-5), permeability_breakpoints=(0.001,))])
def test_an_ill_posed_permeability_schedule_is_refused(kwargs):
    with pytest.raises(ValueError, match="permeability"):
        BlochTorrey1D([0.0, 1e-6, 2e-6], D, **kwargs)


def test_a_breakpoint_inside_a_step_is_refused():
    bt = BlochTorrey1D([0.0, 1e-6, 2e-6], D, permeability=(1e-5, 2e-5), permeability_breakpoints=(0.00105,))
    with pytest.raises(ValueError, match="not on a boundary"):
        bt.signal(np.zeros(20), 1e-4)
    bt.signal(np.zeros(20), 1e-4, sub_steps=2)                         # 1.05 ms is a whole number of 50 us steps

