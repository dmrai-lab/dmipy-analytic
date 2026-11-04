"""Diffusion with exchange between compartments: the Karger model of exchanging Gaussian pools and its propagators
under spin-echo and stimulated-echo encoding, and the forms of diffusion through permeable membranes.

**The Karger model.** Two pools of Gaussian diffusion, an intra-cellular pool of volume fraction ``f`` and an
extra-cellular pool ``1 - f``, exchange water at the rate ``kappa`` (s^-1) from the first to the second and
``kappa f / (1 - f)`` back (detailed balance). Over the diffusion time ``t_d`` each pool dephases with its exponent
``R = -ln E`` (the log-attenuation it would have alone), and the magnetisation vector ``(M_1, M_2)`` obeys the
two-pool rate equation ``dM/dt = (K - R/t_d) M`` with the exchange generator ``K = [[-kappa, k_ee], [kappa, -k_ee]]``.
:func:`karger` is its solution in closed form for a diffusion time at which the dephasing is constant, the sum of two
exponentials in the eigenvalues of ``K t_d + R``; the propagators :func:`spin_echo_signal` and
:func:`stimulated_echo_signal` are its exact solution through the intervals of a sequence, one matrix exponential
per interval, with a relaxation time ``T2`` and ``T1`` per pool and the radiofrequency pulses of finite duration
(a pulse relaxes both ways at ``(2/pi)(1/T2 + 1/T1)``).

**Any gradient waveform.** With the effective gradient sampled in time, the two pools obey the generalised Karger
equation ``dM/dt = (K - R(t) - |q(t)|^2 diag(D)) M`` with ``q(t) = gamma int_0^t G_eff``, which holds the narrow-pulse
form as its limit (Ning L et al. (2018), J. Chem. Phys. 148:074109, Eq. 5). :func:`karger_waveform` propagates it
along a sampled waveform, relaxation at ``T2`` while transverse and ``T1`` while stored; :func:`fexi_narrow_pulse` is
its closed form for a filter-exchange (double-encoding) sequence of narrow pulses (Ning et al. 2018, Eq. 7).

**Filter-exchange imaging (FEXI).** A diffusion filter, a mixing time ``t_m`` with the magnetisation stored, and a
detection block. The apparent exchange rate model ``ADC'(t_m) = ADC_eq [1 - sigma exp(-AXR t_m)]`` (Lasic S et al.
(2011), Magn. Reson. Med. 66:356-365) is :func:`fexi_adc` and its signal :func:`fexi_signal`; :func:`fexi_two_site` is
what it reads on two exchanging Gaussian pools.

Sources: Karger J, Pfeifer H, Heink W (1988), Adv. Magn. Reson. 12:1-89 (the two-site exchange of the diffusion
attenuation); Karger J (1985), Adv. Colloid Interface Sci. 23:129-148; Jelescu IO et al. (2022), NeuroImage 256:119277
(NEXI: the anisotropic Karger signal of the neurites and the extra-neurite space, Eqs. 3-6); the matrix propagators are
the exchange generalisation of the multi-interval Bloch-Torrey solution.

The interval propagators need scipy's matrix exponential (the ``special`` extra), imported inside them; the
two-pool forms of a waveform and of FEXI are numpy alone (a 2 x 2 exponential has a closed form).
"""
import numpy as np

from ._extras import need
from .constants import GAMMA

__all__ = ["diffusion_time", "karger", "exchange_matrix", "spin_echo_signal", "stimulated_echo_signal",
           "karger_waveform", "fexi_narrow_pulse", "fexi_adc", "fexi_signal", "fexi_two_site"]


def diffusion_time(delta, Delta):
    r"""The effective diffusion time of a pulsed-gradient measurement, ``Delta - delta/3``.

    Parameters
    ----------
    delta, Delta : float or array_like
        The gradient pulse duration and the pulse separation, s.

    Assumes rectangular gradient pulses; the time over which the encoded displacement is correlated. Source:
    Stejskal EO, Tanner JE (1965), J. Chem. Phys. 42:288 (the factor ``Delta - delta/3`` of the attenuation).
    """
    return Delta - delta / 3.0


def karger(exponent_intra, exponent_extra, kappa, f, diffusion_time):
    r"""The Karger signal of two exchanging Gaussian pools, ``sigma_+ exp(-lambda_+) + (1 - sigma_+) exp(-lambda_-)``
    with ``lambda_+/-`` the eigenvalues of the exchange-weighted dephasing matrix.

    Parameters
    ----------
    exponent_intra, exponent_extra : array_like
        The dephasing exponents ``R = -ln E`` the intra- and extra-cellular pool would have alone, per measurement.
    kappa : float
        The exchange rate from the intra- to the extra-cellular pool, s^-1.
    f : float
        The intra-cellular volume fraction, in (0, 1).
    diffusion_time : float or array_like
        The effective diffusion time, s (:func:`diffusion_time`).

    Assumes Gaussian diffusion in each pool, exchange at a constant rate between the pools in detailed balance
    (``kappa f / (1 - f)`` back), and a dephasing that accumulates over the diffusion time as if constant (the
    narrow-pulse picture). At ``kappa = 0`` the weights are the volume fractions, ``f`` on the larger exponent when
    it is the intra-cellular one, and a vanishing eigenvalue gap gives the mean exponent with weight one half.
    Source: Karger et al. (1988), Adv. Magn. Reson. 12:1; Jelescu et al. (2022), NeuroImage 256:119277, Eqs. 3-5.
    """
    EPS = 1e-10
    kee = kappa * f / (1.0 - f)
    kt = kappa * diffusion_time
    keet = kee * diffusion_time

    Tr = exponent_intra + exponent_extra + kt + keet
    Det = exponent_intra * exponent_extra + exponent_intra * keet + exponent_extra * kt

    disc_sq = np.maximum(Tr ** 2 - 4.0 * Det, 0.0)
    disc = np.sqrt(disc_sq)

    lam_plus = (Tr + disc) / 2.0
    lam_minus = (Tr - disc) / 2.0

    safe_disc = np.where(disc > EPS, disc, np.ones_like(disc))
    safe_keet = keet if np.ndim(keet) == 0 else np.where(keet > EPS, keet, np.ones_like(keet))

    with np.errstate(divide="ignore", invalid="ignore"):         # the branch not taken may divide by zero
        sp_full = f * (exponent_intra - lam_minus) * (lam_minus - exponent_extra) / (safe_keet * safe_disc)
        sp_kappa0 = (f * exponent_intra + (1.0 - f) * exponent_extra - lam_minus) / safe_disc
    sigma_plus = np.where(
        disc <= EPS,
        0.5,
        np.where(np.asarray(keet) <= EPS, sp_kappa0, sp_full),
    )

    return sigma_plus * np.exp(-lam_plus) + (1.0 - sigma_plus) * np.exp(-lam_minus)


def exchange_matrix(kappa, f):
    r"""The exchange generator of two pools in detailed balance, ``[[-kappa, k_ee], [kappa, -k_ee]]`` with
    ``k_ee = kappa f / (1 - f)``.

    Parameters
    ----------
    kappa : float
        The exchange rate from the first (intra-cellular) to the second pool, s^-1.
    f : float
        The volume fraction of the first pool, in (0, 1).

    The columns sum to zero (the exchange conserves magnetisation) and ``K (f, 1 - f)^T = 0`` (the volume fractions
    are the equilibrium). Source: Karger et al. (1988), Adv. Magn. Reson. 12:1.
    """
    kee = kappa * f / (1.0 - f)
    return np.array([[-kappa, kee],
                     [kappa, -kee]], dtype=float)


def spin_echo_signal(K, M0, D, T2, T1, b, dt1, dt2, tau_exc=0.0, tau_180=0.0):
    r"""The signal of exchanging pools at a spin echo of finite radiofrequency pulses: the magnetisation summed over
    the pools after ``P_fp2 P_180 P_fp1 P_exc M0``, each ``P = expm(.)`` the propagator of one interval.

    Parameters
    ----------
    K : array_like, shape (N, N)
        The exchange generator (columns sum to zero), s^-1.
    M0 : array_like, shape (N,)
        The initial magnetisation of each pool, the volume fractions.
    D, T2, T1 : array_like, shape (N,)
        The effective diffusivity (m^2/s) and the relaxation times (s) of each pool; ``1e10`` s is no relaxation.
    b : float
        The b-value, s/m^2, split equally between the two free-precession intervals.
    dt1, dt2 : float
        The durations of the free precession before and after the refocusing pulse, s.
    tau_exc, tau_180 : float
        The durations of the excitation and refocusing pulses, s; zero is an instantaneous pulse.

    Assumes Gaussian diffusion in each pool, constant gradient-weighting rate ``b/2`` over each free-precession
    interval, and during a pulse relaxation at ``(2/pi)(1/T2 + 1/T1)`` with exchange. Needs scipy (``special``).
    Source: the multi-interval solution of the exchange-coupled Bloch-Torrey equation; Karger et al. (1988), Adv. Magn.
    Reson. 12:1 for the exchange generator.
    """
    expm = need("scipy.linalg").expm
    RT2 = np.diag(1.0 / np.asarray(T2, dtype=float))
    RT1 = np.diag(1.0 / np.asarray(T1, dtype=float))
    RD = np.diag(np.asarray(D, dtype=float))
    R12 = (2.0 / np.pi) * (RT2 + RT1)
    B = b / 2.0
    P_exc = expm((K - R12) * tau_exc)
    P_fp1 = expm((K - RT2) * dt1 - B * RD)
    P_180 = expm((K - R12) * tau_180)
    P_fp2 = expm((K - RT2) * dt2 - B * RD)
    M_TE = P_fp2 @ P_180 @ P_fp1 @ P_exc @ np.asarray(M0, dtype=float)
    return np.sum(M_TE)


def stimulated_echo_signal(K, M0, D, T2, T1, b, delta, TM, dt6, tau_90=0.0):
    r"""The signal of exchanging pools at a stimulated echo of finite radiofrequency pulses, including the factor
    one half of the stimulated pathway: ``P_enc2 P_rec P_mix P_store P_enc1 P_exc M0``.

    Parameters
    ----------
    K : array_like, shape (N, N)
        The exchange generator, s^-1.
    M0 : array_like, shape (N,)
        The initial magnetisation of each pool.
    D, T2, T1 : array_like, shape (N,)
        The effective diffusivity (m^2/s) and the relaxation times (s) of each pool.
    b : float
        The b-value, s/m^2, split equally between the two encoding intervals.
    delta : float
        The duration of the first encoding interval, s.
    TM : float
        The mixing time during which the magnetisation is longitudinal (``T1`` relaxation and exchange only), s.
    dt6 : float
        The duration of the second encoding interval, s.
    tau_90 : float
        The duration of each of the three 90-degree pulses, s.

    Assumes the transverse encoding is two intervals of constant gradient-weighting rate ``b/2`` bracketing the
    mixing time, and the pathway factor one half of a stimulated echo. Needs scipy (``special``). Source: the
    multi-interval solution of the exchange-coupled Bloch-Torrey equation; Karger et al. (1988), Adv. Magn. Reson.
    12:1.
    """
    expm = need("scipy.linalg").expm
    RT2 = np.diag(1.0 / np.asarray(T2, dtype=float))
    RT1 = np.diag(1.0 / np.asarray(T1, dtype=float))
    RD = np.diag(np.asarray(D, dtype=float))
    R12 = (2.0 / np.pi) * (RT2 + RT1)
    B = b / 2.0
    P_exc = expm((K - R12) * tau_90)
    P_enc1 = expm((K - RT2) * delta - B * RD)
    P_store = expm((K - R12) * tau_90)
    P_mix = expm((K - RT1) * TM)
    P_rec = expm((K - R12) * tau_90)
    P_enc2 = expm((K - RT2) * dt6 - B * RD)
    M_TE = 0.5 * P_enc2 @ P_rec @ P_mix @ P_store @ P_enc1 @ P_exc @ np.asarray(M0, dtype=float)
    return np.sum(M_TE)


# ── two pools under any sampled waveform, and filter-exchange imaging ───────────────────────────────────────────────

def _expm2(M):
    r"""``exp(M)`` of real 2 x 2 matrices ``(..., 2, 2)`` in closed form: with ``tau = tr M / 2`` and
    ``s^2 = ((M_11 - M_22) / 2)^2 + M_12 M_21``, ``exp(M) = e^tau [C I + S (M - tau I)]``, ``C = cosh s`` and
    ``S = sinh(s) / s`` for ``s^2 >= 0`` (``cos`` and ``sin(s) / s`` of ``sqrt(-s^2)`` otherwise), written as
    ``(e^(tau + s) +- e^(tau - s)) / 2`` so neither factor overflows, and ``S`` by its series below ``s = 1e-3``. Source:
    the Cayley-Hamilton theorem for a 2 x 2 matrix (``(M - tau I)^2 = s^2 I``)."""
    a, b, c, d = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
    tau = 0.5 * (a + d)
    s2 = (0.5 * (a - d)) ** 2 + b * c
    x = np.sqrt(np.abs(s2))
    small = x < 1e-3
    xs = np.where(small, 1.0, x)
    ep, em = np.exp(tau + x), np.exp(tau - x)
    et = np.exp(tau)
    C = np.where(s2 >= 0.0, 0.5 * (ep + em), et * np.cos(x))
    S_big = np.where(s2 >= 0.0, 0.5 * (ep - em) / xs, et * np.sin(x) / xs)
    S_small = et * (1.0 + np.sign(s2) * x * x / 6.0 + x ** 4 / 120.0)
    S = np.where(small, S_small, S_big)
    out = np.empty(M.shape, dtype=np.float64)
    out[..., 0, 0] = C + S * (a - tau)
    out[..., 1, 1] = C + S * (d - tau)
    out[..., 0, 1] = S * b
    out[..., 1, 0] = S * c
    return out


def _ordered_product(P):
    """``P[..., n-1, :, :] @ ... @ P[..., 0, :, :]``, the propagators applied in time order, by pairwise products
    (a balanced tree, so the rounding grows with ``log2 n``)."""
    while P.shape[-3] > 1:
        if P.shape[-3] % 2:
            eye = np.broadcast_to(np.eye(2), P.shape[:-3] + (1, 2, 2))
            P = np.concatenate([P, eye], axis=-3)
        P = P[..., 1::2, :, :] @ P[..., 0::2, :, :]
    return P[..., 0, :, :]


def _two_pools(D, K, M0):
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    K = np.asarray(K, dtype=np.float64)
    M0 = np.asarray(M0, dtype=np.float64).reshape(-1)
    if D.shape != (2,) or K.shape != (2, 2) or M0.shape != (2,):
        raise ValueError(f"two pools: D and M0 of shape (2,) and K of shape (2, 2); got {D.shape}, {K.shape}, {M0.shape}")
    return D, K, M0


def _rates(T, default):
    """Per-pool relaxation rates ``1/T`` (s^-1) of ``T`` (two times, s), zero for ``None`` or an infinite time."""
    if T is None:
        return np.zeros(2)
    T = np.broadcast_to(np.asarray(T, dtype=np.float64), (2,))
    return np.where(np.isfinite(T), 1.0 / np.where(np.isfinite(T), T, 1.0), 0.0)


def karger_waveform(G, dt, D, K, M0, *, stored=None, T2=None, T1=None, sub_steps=1, gamma=GAMMA):
    r"""The signal of two exchanging Gaussian pools under a sampled effective gradient: the generalised Karger equation
    ``dM/dt = (K - R(t) - |q(t)|^2 diag(D)) M``, ``q(t) = gamma int_0^t G``, propagated from ``M0`` and summed over the
    pools.

    Parameters
    ----------
    G : array_like, shape (n_t,), (n_t, 3) or (n_meas, n_t, 3)
        The EFFECTIVE gradient (T/m): the played gradient through the coherence sign and zero while the magnetisation is
        stored. Sample ``k`` acts over ``[k dt, (k + 1) dt)`` and the last sample is the readout, which acts over
        nothing, so the signal is read at ``(n_t - 1) dt``. A 1-D ``G`` is along one axis.
    dt : float
        The sampling interval, s.
    D : array_like, shape (2,)
        The diffusivity of each pool, m^2/s.
    K : array_like, shape (2, 2)
        The exchange generator, s^-1: ``dM_1/dt = K_11 M_1 + K_12 M_2`` with columns summing to zero
        (:func:`exchange_matrix`).
    M0 : array_like, shape (2,)
        The magnetisation of each pool at ``t = 0`` (the volume fractions at equilibrium).
    stored : array_like of bool, shape (n_t,), optional
        Where the magnetisation is stored along B0 (sample ``k`` for the step it starts): it relaxes at ``T1`` there and
        at ``T2`` elsewhere. ``None`` is transverse throughout.
    T2, T1 : array_like, shape (2,), optional
        The relaxation times of each pool, s; ``None`` or ``inf`` is no relaxation.
    sub_steps : int
        Equal sub-steps per sample (the gradient is constant over a sample, so ``q`` is linear over each sub-step).
    gamma : float
        The gyromagnetic ratio, rad/(s T).

    Returns
    -------
    float or ndarray, shape (n_meas,)
        ``sum(M(T))``: the coherent pathway's attenuation, without the pathway's own amplitude (a stimulated echo's 1/2,
        :func:`dmipy_analytic.epg.ste_amplitude`).

    Assumes Gaussian diffusion in each pool, exchange at constant rates between well-mixed pools (barrier-limited
    exchange), and the gradient constant over each sample. Each sub-step of length ``h`` is propagated by the exponential
    of the first two terms of its Magnus expansion, both in closed form for the quadratic ``|q|^2 = alpha + beta t + c
    t^2``: ``Omega = (K - R) h - diag(D) int |q|^2 + (h^3 / 12)(beta + c h) [K, diag(D)]``, with ``int |q|^2 = alpha h +
    beta h^2 / 2 + c h^3 / 3``. The local error is ``O(h^5)`` and the global ``O(h^4)``: halving the sub-step divides
    the error by 16, and a step over which ``q`` is constant (no gradient) is exact. Exact without exchange or with equal
    diffusivities, whatever the step. Each exponential is the closed 2 x 2 form. Sources: Ning L et al. (2018), J. Chem.
    Phys. 148:074109, Eq. 5 (the equation); Magnus W (1954), Commun. Pure Appl. Math. 7:649-673, and Blanes S et al.
    (2009), Phys. Rep. 470:151-238, Sec. 2 (the expansion).
    """
    D, K, M0 = _two_pools(D, K, M0)
    G = np.asarray(G, dtype=np.float64)
    squeeze = G.ndim < 3
    if G.ndim == 1:
        G = G[None, :, None]
    elif G.ndim == 2:
        G = G[None]
    if G.ndim != 3:
        raise ValueError(f"G is (n_t,), (n_t, 3) or (n_meas, n_t, 3); got shape {np.shape(G)}")
    n_meas, n_t = G.shape[0], G.shape[1]
    if n_t < 2:
        raise ValueError("G needs at least two samples: one step and the readout")
    sub = int(sub_steps)
    if sub < 1:
        raise ValueError(f"sub_steps is a positive integer; got {sub_steps!r}")
    st = np.zeros(n_t, dtype=bool) if stored is None else np.asarray(stored, dtype=bool).reshape(-1)
    if st.shape != (n_t,):
        raise ValueError(f"stored is one flag per sample ({n_t}); got shape {st.shape}")
    h = float(dt) / sub
    g = np.repeat(G[:, :-1, :], sub, axis=1)                        # (n_meas, n_steps, 3)
    st = np.repeat(st[:-1], sub)
    R = np.where(st[:, None], _rates(T1, None)[None, :], _rates(T2, None)[None, :])   # (n_steps, 2)
    comm = np.array([[0.0, K[0, 1] * (D[1] - D[0])], [K[1, 0] * (D[0] - D[1]), 0.0]])  # [K, diag(D)]
    out = np.empty(n_meas)
    chunk = max(1, int(4e6 // max(1, g.shape[1])))
    for m0 in range(0, n_meas, chunk):
        gm = g[m0:m0 + chunk]
        v = gamma * gm
        q1 = np.cumsum(v, axis=1) * h
        q0 = q1 - v * h
        alpha = np.sum(q0 * q0, axis=-1)
        beta = 2.0 * np.sum(q0 * v, axis=-1)
        c = np.sum(v * v, axis=-1)
        I1 = alpha * h + beta * h ** 2 / 2.0 + c * h ** 3 / 3.0
        w2 = (h ** 3 / 12.0) * (beta + c * h)
        Om = np.empty(I1.shape + (2, 2))
        Om[..., 0, 0] = (K[0, 0] - R[:, 0]) * h - D[0] * I1
        Om[..., 1, 1] = (K[1, 1] - R[:, 1]) * h - D[1] * I1
        Om[..., 0, 1] = K[0, 1] * h + w2 * comm[0, 1]
        Om[..., 1, 0] = K[1, 0] * h + w2 * comm[1, 0]
        P = _ordered_product(_expm2(Om))                              # (chunk, 2, 2)
        out[m0:m0 + chunk] = np.sum(P @ M0, axis=-1)
    return float(out[0]) if squeeze else out


def fexi_narrow_pulse(q_filter, Delta_filter, t_m, q_detect, Delta_detect, D, K, M0):
    r"""The two-pool signal of a filter-exchange sequence of narrow pulses: ``1^T exp(M(q_d) Delta_d) exp(K t_m)
    exp(M(q_f) Delta_f) M0`` with ``M(q) = K - q^2 diag(D)``.

    Parameters
    ----------
    q_filter, q_detect : float or array_like
        The wave numbers ``gamma g delta`` of the filter and the detection pair, rad/m (broadcast together).
    Delta_filter, Delta_detect : float
        The separation of each pair's two narrow pulses, s: the time over which its wave number is held.
    t_m : float or array_like
        The mixing time between the two pairs, s (broadcast with the wave numbers).
    D, K, M0
        As :func:`karger_waveform`.

    Assumes narrow pulses (the wave number constant between the two pulses of a pair and zero elsewhere), Gaussian
    diffusion in each pool, barrier-limited exchange at constant rates, and no relaxation. The limit of
    :func:`karger_waveform` as each pair's pulse duration vanishes at a fixed wave number. Source: Ning L et al. (2018),
    J. Chem. Phys. 148:074109, Eq. 7 (the double-diffusion-encoding signal of the Karger model).
    """
    D, K, M0 = _two_pools(D, K, M0)
    qf, tm, qd = np.broadcast_arrays(np.asarray(q_filter, np.float64), np.asarray(t_m, np.float64),
                                     np.asarray(q_detect, np.float64))
    Dm = np.diag(D)
    Mf = (K[None] - qf.reshape(-1)[:, None, None] ** 2 * Dm[None]) * float(Delta_filter)
    Mx = K[None] * tm.reshape(-1)[:, None, None]
    Md = (K[None] - qd.reshape(-1)[:, None, None] ** 2 * Dm[None]) * float(Delta_detect)
    S = np.sum(_expm2(Md) @ _expm2(Mx) @ _expm2(Mf) @ M0, axis=-1)
    return S.reshape(qf.shape) if qf.ndim else float(S[0])


def fexi_adc(t_m, adc_eq, sigma, axr):
    r"""The apparent diffusivity a filtered FEXI measurement reads after a mixing time ``t_m``:
    ``ADC'(t_m) = ADC_eq [1 - sigma exp(-AXR t_m)]``.

    Parameters
    ----------
    t_m : float or array_like
        The mixing time, s.
    adc_eq : float
        The equilibrium apparent diffusivity (the unfiltered one), m^2/s.
    sigma : float
        The filter efficiency, ``1 - ADC'(0) / ADC_eq``.
    axr : float
        The apparent exchange rate, s^-1.

    Exact for two exchanging Gaussian pools of equal ``T1`` in the limit of a small detection b-value, whatever the
    filter and detection blocks: the two-pool generator has the spectral form ``exp(K t) = Pi + exp(-(k_12 + k_21) t)
    (I - Pi)`` with ``Pi`` the projector on equilibrium, so a detected signal is ``A(b) + B(b) exp(-AXR t_m)`` with
    ``B(0) = 0`` (exchange conserves the magnetisation), and its initial slope in ``b`` is this form with ``AXR = k_12 +
    k_21``. At a finite detection b the log of that sum is not linear in ``b``, and the form is the first cumulant.
    Source: Lasic S et al. (2011), Magn. Reson. Med. 66:356-365 (the AXR model); md-dmri ``methods/fexi11/
    fexi11_1d_fit2data.m`` (Nilsson M, Topgaard D et al.; the reference implementation).
    """
    return adc_eq * (1.0 - sigma * np.exp(-axr * np.asarray(t_m, dtype=np.float64)))


def fexi_signal(b, t_m, filtered, adc_eq, sigma, axr):
    r"""The normalised FEXI signal of the AXR model, ``exp(-b ADC)`` with ``ADC = ADC'(t_m)`` (:func:`fexi_adc`) on a
    filtered measurement and ``ADC_eq`` on an unfiltered one.

    Parameters
    ----------
    b : array_like
        The detection b-value, s/m^2.
    t_m : array_like
        The mixing time, s (broadcast with ``b``).
    filtered : array_like of bool
        Whether the measurement's filter block is on (broadcast with ``b``).
    adc_eq, sigma, axr : float
        As :func:`fexi_adc`.

    Each series of measurements sharing a filter and a mixing time is scaled by its own signal at no detection
    weighting, which this form leaves out. Source: md-dmri ``methods/fexi11/fexi11_1d_fit2data.m`` (``ADC = ADC0 (1 -
    ind_non_eq sigma exp(-AXR tm12))``, ``s = exp(-ADC b2)``); Lasic et al. (2011), Magn. Reson. Med. 66:356-365.
    """
    b, t_m, filtered = np.broadcast_arrays(np.asarray(b, np.float64), np.asarray(t_m, np.float64),
                                           np.asarray(filtered, bool))
    adc = np.where(filtered, fexi_adc(t_m, adc_eq, sigma, axr), adc_eq)
    return np.exp(-b * adc)


def fexi_two_site(f, kappa, D, b_filter):
    r"""``(ADC_eq, sigma, AXR)`` of two exchanging Gaussian pools under a filter block of b-value ``b_filter`` whose
    duration is short against the exchange: ``ADC_eq = f D_1 + (1 - f) D_2``, ``ADC'(0)`` the same mean over the
    filtered fractions ``f_i exp(-b_filter D_i)``, ``sigma = 1 - ADC'(0) / ADC_eq`` and ``AXR = k_12 + k_21 =
    kappa / (1 - f)``.

    Parameters
    ----------
    f : float
        The fraction of pool 1, in (0, 1).
    kappa : float
        The exchange rate out of pool 1, s^-1 (``kappa f / (1 - f)`` back, detailed balance, :func:`exchange_matrix`).
    D : array_like, shape (2,)
        The diffusivities of the two pools, m^2/s.
    b_filter : float
        The filter's b-value, s/m^2.

    Assumes no exchange during the filter and detection blocks and the small-b limit of the detection (where
    :func:`fexi_adc` is exact); with blocks of finite duration ``sigma`` is smaller and ``AXR`` unchanged. Source:
    Lasic S et al. (2011), Magn. Reson. Med. 66:356-365 (the filter efficiency and the two-site exchange rate);
    Karger et al. (1988), Adv. Magn. Reson. 12:1 (the two-site exchange).
    """
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    w = np.array([f, 1.0 - f]) * np.exp(-float(b_filter) * D)
    adc_eq = f * D[0] + (1.0 - f) * D[1]
    adc_0 = float(np.sum(w * D) / np.sum(w))
    return float(adc_eq), float(1.0 - adc_0 / adc_eq), float(kappa / (1.0 - f))
