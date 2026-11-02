"""The call table of the parity record of dmipy-analytic's forms (dmrai-lab/dmipy-sim#834): one group per module the
forms come from, each a function of a spelling (:mod:`apis`) returning ``{name: value}`` on fixed inputs.

A value is an array, a number, a string, ``None``, bytes (recorded by their sha256: the large matrices) or a dict or
list of these (flattened by the record). The inputs are built by arithmetic alone (no random stream), and every JAX
form is called on float32 arrays, so the precision is the call's and not a configuration's."""
import hashlib

import numpy as np


def _fib(n):
    """``n`` unit vectors on a Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = i * np.pi * (1.0 + np.sqrt(5.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _rot(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


R1 = _rot([1.0, 2.0, 0.5], 0.7)
R2 = _rot([0.0, 1.0, 1.0], -1.1)
POLES = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.0, 0.8]])


def _digest(a):
    """A large array by its bytes (its sha256 is recorded), with its dtype and shape beside it."""
    a = np.ascontiguousarray(a)
    return {"bytes": hashlib.sha256(a.tobytes()).digest(), "dtype": a.dtype.str, "shape": list(a.shape)}


def _refusal(fn, *args, **kw):
    """The exception ``fn(*args, **kw)`` raises, as ``Type: message``; ``None`` when it returns."""
    try:
        fn(*args, **kw)
    except Exception as exc:                                     # the refusal is the value recorded
        return f"{type(exc).__name__}: {exc}"
    return None


def g01_constants(api):
    return {"GAMMA": api.GAMMA, "GAMMA_BAR": api.GAMMA_BAR, "MU0": api.MU0}


def g02_harmonics(api):
    dirs = np.concatenate([_fib(40), POLES])
    out = {
        "cart2sphere/one": api.cart2sphere(np.array([0.3, -0.4, 0.5])),
        "cart2sphere/zero": api.cart2sphere(np.array([0.0, 0.0, 0.0])),
        "cart2sphere/many": api.cart2sphere(np.concatenate([2.5 * dirs, np.zeros((1, 3))])),
        "cart2sphere/refused": _refusal(api.cart2sphere, np.zeros((2, 2, 3))),
        "n_sh_coeffs": [[int(api.n_sh_coeffs(l, full)) for l in range(13)] for full in (False, True)],
        "sh_block": [[[api.sh_block(l, full).start, api.sh_block(l, full).stop] for l in range(13)]
                     for full in (False, True)],
        "real_sh/even8": api.real_sh(8, dirs),
        "real_sh/full7": api.real_sh(7, dirs, full=True),
        "real_sh/even16": api.real_sh(16, dirs[:12]),
        "real_sh/one": api.real_sh(4, dirs[3]),
    }
    Rs = np.stack([np.eye(3), R1, R2, R1 @ R2, R2 @ R1.T])
    out["wigner_blocks"] = list(api.wigner_blocks(6, Rs))
    out["wigner_blocks/one"] = list(api.wigner_blocks(3, R1))
    return out


KAPPAS = (0.0, 1e-13, 0.5, 2.0, 16.0, 128.0, 699.0, 701.0, 2000.0)
PAIRS = ((0.0, 0.0), (1.0, 4.0), (8.0, 2.0), (16.0, 16.0), (0.5, 64.0), (40.0, 0.0))


def g03_orientation(api):
    out = {}
    for k in KAPPAS:
        for l_max in (8, 16):
            out[f"watson_zonal_ratios/{k!r}/{l_max}"] = api.watson_zonal_ratios(k, l_max)
    for i, mu in enumerate([(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.3, -0.4, 0.5), (-2.0, 1.0, 0.5)]):
        for k in (0.5, 16.0, 128.0, 1000.0):
            out[f"watson_sh/{i}/{k!r}"] = api.watson_sh(np.array(mu), k)
        out[f"watson_sh/{i}/l12"] = api.watson_sh(np.array(mu), 7.0, l_max=12)
    for k1, k2 in PAIRS:
        out[f"bingham_normalization/{k1!r}/{k2!r}"] = api.bingham_normalization(k1, k2)
        out[f"bingham_canonical_sh/{k1!r}/{k2!r}"] = api.bingham_canonical_sh(k1, k2)
        out[f"bingham_canonical_sh/{k1!r}/{k2!r}/l12"] = api.bingham_canonical_sh(k1, k2, l_max=12)
        out[f"bingham_sh/{k1!r}/{k2!r}"] = api.bingham_sh(R1, (k1, k2))
    out["bingham_sh/scalar"] = api.bingham_sh(R2, 9.0, l_max=10)
    return out


def _pathways(paths, dt):
    return [{"eta": complex(p.eta), "intervals": repr(p.intervals), "readout_idx": p.readout_idx,
             "chi_perp": p.chi_perp(dt), "eps_P": p.eps_P(dt)} for p in paths]


def g04_epg(api):
    schedules = {
        "cpmg_120": api.cpmg_schedule(4, 120.0, TE=0.08),
        "cpmg_180": api.cpmg_schedule(3, TE=0.06),
        "cpmg_150_x": api.cpmg_schedule(3, 150.0, TE=0.04, refocus_phase_deg=0.0),
        "splice_140": api.splice_schedule(3, 140.0, ESP=0.01),
        "ste": api.ste_schedule(delta=0.01, TM=0.05),
        "ste_spoiled": api.ste_schedule(60.0, 75.0, 80.0, delta=0.01, TM=0.05, spoil=4),
        "hand": api.Schedule((api.Pulse(70.0, 20.0), api.Winding(2, 0.004), api.Pulse(130.0, 95.0),
                              api.Winding(-1, 0.003, readout=True), api.Pulse(45.0), api.Winding(1, 0.002, readout=True))),
    }
    out = {}
    for name, sch in schedules.items():
        out[f"{name}/schedule"] = repr(sch)
        out[f"{name}/pathways"] = _pathways(api.enumerate_pathways(sch), 1e-3)
        out[f"{name}/pathways_all"] = _pathways(api.enumerate_pathways(sch, threshold=1e-12), 5e-4)
    for flips in ((90.0, 90.0, 90.0), (60.0, 75.0, 80.0), (90.0, 0.0, 90.0)):
        out[f"ste_amplitude/{flips!r}"] = api.ste_amplitude(*flips)
    return out


def _state(st):
    return {"F": st.F.copy(), "Z": st.Z.copy(), "signal": st.signal, "n": st.n}


def g05_epg_state(api):
    out = {}
    for flip, phase in ((90.0, 0.0), (120.0, 90.0), (33.3, 47.0), (180.0, 0.0), (75.0, -30.0)):
        out[f"rf_operator/{flip!r}/{phase!r}"] = api.rf_operator(flip, phase)
    st = api.EPGState.equilibrium(6)
    steps = [("pulse", (90.0, 0.0)), ("shift", (1,)), ("off_resonance", (0.01, 30.0)),
             ("relax", (0.01, 1.0, 0.08)), ("pulse", (120.0, 90.0)), ("shift", (-2,)), ("relax", (0.01, None, 0.05)),
             ("pulse", (90.0, 90.0)), ("shift", (3,)), ("off_resonance", (0.004, -12.0)), ("pulse", (160.0, 10.0)),
             ("shift", (-3,)), ("relax", (0.02, 0.9, None))]
    for i, (op, args) in enumerate(steps):
        getattr(st, op)(*args)
        out[f"state/{i}/{op}"] = _state(st)
    out["state/copy"] = _state(st.copy())
    out["state/empty"] = _state(api.EPGState(3))
    prep = [("pulse", 90.0, 0.0), ("interval", 1, 0), ("pulse", 180.0, 90.0), ("interval", 1, 1),
            ("pulse", 90.0, 90.0), ("interval", 2, 2), ("pulse", 90.0, 90.0), ("interval", -1, 3)]
    gated = {0, 1, 3}
    branches = api.split_by_gate(prep, 8, lambda i: i in gated)
    out["split_by_gate"] = {"/".join(g) or "none": _state(s) for g, s in sorted(branches.items())}
    train = []
    for k in range(5):
        train += [("pulse", 140.0, 90.0), ("interval", 1, 4 + 2 * k), ("interval", 1, 5 + 2 * k)]
    readouts = [2, 5, 8, 11, 14]
    w = api.train_weights(prep, train, 10, lambda i: i in gated, readouts)
    out["train_weights"] = {"/".join(g) or "none": a for g, a in sorted(w.items())}
    durations = [0.002] * 20
    w = api.train_weights(prep, train, 10, lambda i: i in gated, readouts, dw=25.0, durations=durations)
    out["train_weights/off_resonance"] = {"/".join(g) or "none": a for g, a in sorted(w.items())}
    return out


def _pgse(G, n_on, n_gap, n_lead=3):
    z = np.zeros
    return np.concatenate([z(n_lead), np.full(n_on, G), z(n_gap), np.full(n_on, -G), z(n_lead)])


def _ramped(G, n_ramp, n_flat, n_gap):
    up = G * (np.arange(1, n_ramp + 1) / n_ramp)
    lobe = np.concatenate([up, np.full(n_flat, G), up[::-1]])
    return np.concatenate([np.zeros(2), lobe, np.zeros(n_gap), -lobe, np.zeros(2)])


def _ogse(G, n, cycles):
    t = (np.arange(n) + 0.5) / n
    return G * np.cos(2.0 * np.pi * cycles * t)


PORES = (("plane", 8e-6, 16), ("cylinder", 6e-6, (8, 6)), ("sphere", 10e-6, (8, 6)),
         ("plane", 12e-6, None), ("cylinder", 6e-6, None), ("sphere", 10e-6, None))


def g06_matrix_method(api):
    out = {"DEFAULT_MODES": repr(api.DEFAULT_MODES), "RADIAL_NODES": api.RADIAL_NODES,
           "ANGULAR_NODES": api.ANGULAR_NODES}
    for m in range(4):
        out[f"cylinder_neumann_roots/{m}"] = api.cylinder_neumann_roots(m, 6)
        out[f"sphere_neumann_roots/{m}"] = api.sphere_neumann_roots(m, 6)
    dt = 1e-4
    waveforms = {"pgse": _pgse(0.06, 80, 120), "ramped": _ramped(0.05, 6, 60, 90), "ogse": _ogse(0.08, 400, 4.0),
                 "zero": np.zeros(50), "one_step": np.array([0.2])}
    for shape, size, modes in PORES:
        key = f"{shape}/{size!r}/{modes!r}"
        pore = api.MatrixPore(shape, size, 2e-9, n_modes=modes)
        small = modes is not None
        out[f"{key}/lam"] = pore.lam if small else _digest(pore.lam)
        out[f"{key}/B"] = pore.B if small else _digest(pore.B)
        for name, g in waveforms.items():
            out[f"{key}/signal/{name}"] = pore.signal(g, dt)
        out[f"{key}/restricted_signal"] = api.matrix_restricted_signal(shape, waveforms["pgse"], dt, 2e-9, size,
                                                                         n_modes=modes)
        out[f"{key}/restricted_batch"] = api.matrix_restricted_batch(
            shape, np.stack([waveforms["pgse"] * s for s in (0.25, 1.0, 1.5)]), dt, 1.7e-9, size, n_modes=modes)
    out["refused/plane_pair"] = _refusal(api.MatrixPore, "plane", 1e-5, 2e-9, n_modes=(4, 4))
    out["refused/cylinder_int"] = _refusal(api.MatrixPore, "cylinder", 1e-5, 2e-9, n_modes=8)
    out["refused/shape"] = _refusal(api.MatrixPore, "cube", 1e-5, 2e-9, n_modes=8)
    return out


def _cylinder_points(a, b, u):
    """Points around an axis ``u``: radii from inside the lumen to past the sheath, at several azimuths, as the
    perpendicular vectors ``rho_vec``."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    e1 = np.cross(u, [0.3, 0.5, 0.8]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    radii = np.concatenate([np.linspace(0.05, 0.95, 5) * a, np.array([1.0]) * a,
                            a + np.linspace(0.05, 0.95, 5) * (b - a), np.linspace(1.0, 4.0, 6) * b])
    phis = np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False) + 0.1
    rho = np.array([r * (np.cos(p) * e1 + np.sin(p) * e2) for r in radii for p in phis])
    return rho, np.broadcast_to(u, rho.shape)


def g07_fields(api):
    out = {}
    a = np.array([0.4e-6, 0.7e-6, 1.0e-6, 2.0e-6])
    b = np.array([0.5e-6, 1.0e-6, 1.1e-6, 3.5e-6])
    out["annulus_mean_log"] = api.annulus_mean_log(a, b)
    out["annulus_mean_log/scalar"] = api.annulus_mean_log(0.6e-6, 0.9e-6)
    for name, u in (("z", (0.0, 0.0, 1.0)), ("tilted", (0.3, -0.5, 0.81))):
        for ia, ib in ((0.6e-6, 0.8e-6), (1.2e-6, 1.9e-6)):
            rho, uu = _cylinder_points(ia, ib, u)
            f32 = lambda x: np.asarray(x, np.float32)
            n = rho.shape[0]
            out[f"hollow_cylinder_basis/{name}/{ia!r}"] = np.asarray(api.jax_hollow_cylinder_basis(
                f32(rho), f32(uu), f32(np.full(n, ia)), f32(np.full(n, ib))))
            far = rho[np.linalg.norm(rho, axis=1) >= ib]
            m = far.shape[0]
            for z1, z2 in ((-2e-6, 3e-6), (0.5e-6, 9e-6), (-40e-6, 40e-6)):
                out[f"segment_basis/{name}/{ia!r}/{z1!r}/{z2!r}"] = np.asarray(api.jax_segment_basis(
                    f32(far), f32(uu[:m]), f32(np.full(m, z1)), f32(np.full(m, z2)), f32(np.full(m, ia)),
                    f32(np.full(m, ib))))
    return out


def g08_coil(api):
    a, b = np.array([0.0, 0.0, -0.1]), np.array([0.02, 0.01, 0.15])
    P = np.concatenate([0.3 * _fib(30), np.array([[0.05, -0.02, 0.0], [0.5, 0.5, 0.5]])])
    return {"segment_field": api.segment_field(a, b, P), "segment_field/current": api.segment_field(b, a, P, 2.5),
            "segment_field/one": api.segment_field(a, b, np.array([0.1, 0.0, 0.0])),
            "segment_field/on_the_wire": _refusal(api.segment_field, a, b, 0.5 * (a + b))}


def g09_relaxation(api):
    out = {"forward_rate": api.forward_rate(1.5e-5, 2.0e6), "backward_rate": api.backward_rate(0.02),
           "bound_fraction": api.bound_fraction(1.5e-5, 0.02, 2.0e6), "bound_fraction/zero": api.bound_fraction(0.0, 0.02, 0.0),
           "kappa_MT_from_forward_rate": api.kappa_MT_from_forward_rate(30.0, 2.0e6),
           "dwell_time_from_fraction": api.dwell_time_from_fraction(0.15, 30.0),
           "refused/backward_rate": _refusal(api.backward_rate, 0.0),
           "refused/kappa": _refusal(api.kappa_MT_from_forward_rate, 1.0, 0.0),
           "refused/fraction": _refusal(api.dwell_time_from_fraction, 1.0, 3.0),
           "refused/rate": _refusal(api.dwell_time_from_fraction, 0.2, 0.0)}
    A = api.two_pool_generator(R1a=1.0, R2a=12.5, R1b=1.0, R2b=1e5, k_f=4.3, k_r=25.0, dw_a=30.0, dw_b=-200.0,
                               w1=150.0, rf_phase=0.4)
    out["two_pool_generator"] = A
    out["two_pool_generator/M0b"] = api.two_pool_generator(R1a=0.9, R2a=14.0, R1b=1.1, R2b=8e4, k_f=3.0, k_r=0.0,
                                                           M0a=0.8)
    s0 = np.array([0.3, -0.1, 0.7, 0.0, 0.0, 0.17, 1.0])
    out["evolve_two_pool"] = [api.evolve_two_pool(s0, t, A) for t in (0.0, 1e-3, 0.02, 0.4)]
    t = np.array([0.0, 1e-3, 5e-3, 0.02, 0.08])
    out["bloch_mcconnell_transverse"] = api.bloch_mcconnell_transverse(t, T2a=0.08, T2b=1e-5, k_f=4.3, k_r=25.0)
    out["bloch_mcconnell_transverse/one"] = api.bloch_mcconnell_transverse(0.01, T2a=0.07, T2b=0.02, k_f=2.0, k_r=9.0,
                                                                           T1a=0.9, T1b=0.8, M0b=0.2, dw_a=5.0,
                                                                           dw_b=-40.0)
    out["bloch_mcconnell_longitudinal"] = api.bloch_mcconnell_longitudinal(t * 10, T1a=0.9, T1b=1.0, k_f=4.3, k_r=25.0)
    out["bloch_mcconnell_longitudinal/saturated"] = api.bloch_mcconnell_longitudinal(
        t * 10, T1a=0.9, T1b=1.0, k_f=4.3, k_r=25.0, Mza0=1.0, Mzb0=0.0)
    offsets = np.array([-20000.0, -4000.0, -500.0, 0.0, 300.0, 2000.0, 15000.0])
    kw = dict(w1_hz=250.0, t_sat=0.3, T1a=1.0, T2a=0.08, T1b=1.0, T2b=1.2e-5, k_f=4.3, k_r=25.0)
    out["mt_z_spectrum"] = api.mt_z_spectrum(offsets, **kw)
    out["mt_z_spectrum/bound"] = api.mt_z_spectrum(offsets, read_pool="b", **kw)
    out["mt_z_spectrum/one"] = api.mt_z_spectrum(1000.0, **kw)
    return out


GROUPS = {f.__name__: f for f in (g01_constants, g02_harmonics, g03_orientation, g04_epg, g05_epg_state,
                                  g06_matrix_method, g07_fields, g08_coil, g09_relaxation)}
