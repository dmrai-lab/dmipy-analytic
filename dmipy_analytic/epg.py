"""Extended phase graphs: the coherence pathways of a schedule of RF pulses and gradient windings, and their amplitudes.

A sequence of RF pulses and gradient windings does not produce one signal pathway but many. A perfect refocusing
pulse leaves a single spin-echo pathway; any other flip angle splits the magnetisation at every pulse, and the branches
that come back to coherence order zero at a readout all contribute to the echo -- each having spent a different part
of the sequence transverse (decaying at T2, accruing gradient phase) or stored longitudinally (decaying at T1,
accruing none). The amplitudes of those branches are what an extended phase graph computes; they are not free
parameters and they are not to be written down by hand.

**Enumerated.** :func:`enumerate_pathways` turns a :class:`Schedule` into a list of :class:`Pathway`, each carrying
its complex amplitude ``eta`` and, laid onto a time grid, the two arrays a replay of a walk consumes:
:meth:`Pathway.chi_perp` (1 while transverse, 0 while stored) and :meth:`Pathway.eps_P` (+1 / -1 / 0, the sign with
which phase accumulates, so the pathway's own effective gradient is ``G * eps_P``)::

    S = sum_p  eta_p * replay(trajectory, ..., chi_perp=p.chi_perp(dt), eps_P=p.eps_P(dt))

**As a state.** :class:`EPGState` carries the same amplitudes as a vector over coherence orders, one matrix multiply
per pulse and one shift per interval, which is what a long train needs (49,714 pathways reach the last readout of a
twelve-echo train at 120 degrees). What it gives up is the pathway labels, which say how much gradient phase a pathway
accumulated; :func:`split_by_gate` keeps exactly those, splitting the state only where the gradient is on, so a
diffusion-prepared train collapses onto a handful of gates and :func:`train_weights` gives each gate's amplitude at
every echo.

**Phases are the lab's.** ``phase_deg`` is the B1 phase, 0 about x and 90 about y, for every pulse including the
excitation, and the equilibrium magnetisation is tipped into the transverse plane exactly as that pulse leaves it. So
the CPMG geometry -- refocusing about the axis the magnetisation lies along -- is a train whose refocusing pulses are
90 degrees from its excitation. An excitation's own phase multiplies every pathway alike, so it cancels from any ratio;
a perfect refocusing train gives ``|eta| = 1`` at every echo, its sign alternating with each pulse.

**Assumptions.** Pulses are instantaneous, windings are integer multiples of one coherence order, and relaxation is
applied by the consumer (or by :meth:`EPGState.relax`), so the amplitudes depend on the flip angles and phases alone.
Source: Weigel M (2015), *Extended phase graphs: dephasing, RF pulses, and echoes -- pure and simple*, JMRI
41(2):266-295, doi:10.1002/jmri.24619; Hennig J (1988), *Multiecho imaging sequences with low refocusing flip angles*,
J. Magn. Reson. 78:397-407.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Pulse", "Winding", "Schedule", "Pathway", "enumerate_pathways", "splice_schedule", "cpmg_schedule",
           "ste_schedule", "ste_amplitude", "rf_operator", "EPGState", "split_by_gate", "train_weights"]


@dataclass(frozen=True)
class Pulse:
    """An instantaneous RF pulse on the coherence lattice: its flip and its B1 phase, in degrees.

    ``phase_deg`` is the B1 phase in the lab, 0 about x and 90 about y, and it is read on every pulse
    including the excitation.
    """
    flip_deg: float
    phase_deg: float = 0.0


@dataclass(frozen=True)
class Winding:
    """A free-precession interval: the gradient winds every transverse coherence by ``dk`` orders.

    ``dk`` is the interval's net winding in whole coherence orders, signed by the lobe's polarity; a
    crusher or an unbalanced readout is simply a winding of more than one. ``duration`` is the interval's
    length in seconds, which is what lays a pathway's gates onto a trajectory grid, and ``readout`` marks an
    echo acquisition at the end of the interval.
    """
    dk: int
    duration: float = 0.0
    readout: bool = False


@dataclass(frozen=True)
class Schedule:
    """A sequence as an ordered timeline of :class:`Pulse` and :class:`Winding` events.

    The gradient's shape is not here: this is the RF and winding structure alone, which is what decides which
    pathways exist.
    """
    events: tuple


@dataclass(frozen=True)
class Pathway:
    """One branch that reaches coherence order zero at a readout: its amplitude and its history.

    ``intervals`` is ``((state, duration), ...)`` over the windings up to the readout, with ``state`` one of
    ``'F+'``, ``'F-'`` (transverse) or ``'Z'`` (stored).
    """
    eta: complex
    intervals: tuple
    readout_idx: int

    def _lay(self, dt, value):
        segs = [np.full(max(1, int(round(d / float(dt)))), value(st), float) for st, d in self.intervals]
        return np.concatenate(segs) if segs else np.zeros(0)

    def chi_perp(self, dt):
        """The transverse gate on a grid of step ``dt``: 1 while this pathway is transverse, 0 while stored."""
        return self._lay(dt, lambda st: 1.0 if st in ("F+", "F-") else 0.0)

    def eps_P(self, dt):
        """The sign with which phase accumulates on a grid of step ``dt``: +1 on ``F+``, -1 on ``F-``, 0 stored.

        The pathway's own effective gradient is the sequence's ``G`` times this array.
        """
        return self._lay(dt, lambda st: {"F+": 1.0, "F-": -1.0, "Z": 0.0}[st])


def _branch(paths, flip_deg, phase_deg, threshold):
    """Every pathway through one pulse of flip ``flip_deg`` and phase ``phase_deg`` (Weigel's operator).

    At phase zero this is the rotation about x; a phase ``phi`` carries the three factors ``exp(2i phi)`` on
    the transverse-to-conjugate term and ``exp(+-i phi)`` on the terms that couple to and from the
    longitudinal store.
    """
    a, phi = np.radians(float(flip_deg)), np.radians(float(phase_deg))
    c2, s2, sa, ca = np.cos(a / 2) ** 2, np.sin(a / 2) ** 2, np.sin(a), np.cos(a)
    e1, e2 = np.exp(1j * phi), np.exp(2j * phi)
    out = []
    for k, st, amp, hist in paths:
        if st == "F+":
            cand = ((k, "F+", c2 * amp, hist), (k, "F-", np.conj(e2) * s2 * amp, hist),
                    (k, "Z", -0.5j * sa * np.conj(e1) * amp, hist))
        elif st == "F-":
            cand = ((k, "F+", e2 * s2 * amp, hist), (k, "F-", c2 * amp, hist),
                    (k, "Z", 0.5j * sa * e1 * amp, hist))
        else:
            cand = ((k, "F+", -1j * sa * e1 * amp, hist), (k, "F-", 1j * sa * np.conj(e1) * amp, hist),
                    (k, "Z", ca * amp, hist))
        out.extend(c for c in cand if abs(c[2]) >= threshold)
    return out


def _wind_once(paths, step):
    """One unit winding: transverse orders move by ``step``, the store does not, and an order crossing zero
    becomes its conjugate partner (``F+`` at order 0 wound down is ``F-`` at order 1)."""
    out = []
    for k, st, amp, hist in paths:
        if st == "Z":
            out.append((k, "Z", amp, hist))
        elif st == "F+":
            nk = k + step
            out.append((nk, "F+", amp, hist) if nk >= 0 else (1, "F-", np.conj(amp), hist))
        else:
            nk = k - step
            out.append((nk, "F-", amp, hist) if nk >= 0 else (1, "F+", np.conj(amp), hist))
    return out


def _wind(paths, dk):
    """A winding of any whole number of orders, as the unit step repeated."""
    step = 1 if dk > 0 else -1
    for _ in range(abs(int(dk))):
        paths = _wind_once(paths, step)
    return paths


def enumerate_pathways(schedule, threshold=1e-2):
    """Every pathway of ``schedule`` that reaches coherence order zero at a readout.

    Walks the timeline once from equilibrium, branching at each :class:`Pulse` and winding at each
    :class:`Winding`, and collects the transverse coherence at order zero wherever a readout is marked. A
    branch whose amplitude falls below ``threshold`` is dropped, so the cost is bounded by what actually
    contributes. Returns the pathways sorted by readout and then by descending amplitude.
    """
    paths, out, readout_idx, excited = [], [], 0, False
    for ev in schedule.events:
        if isinstance(ev, Pulse):
            if not excited:
                # The excitation is seeded rather than branched: at order zero the two transverse states are
                # one hermitian coherence, and branching would count the initial magnetisation twice. The
                # seed is the physical one a pulse of this flip and phase makes out of equilibrium,
                # ``-i exp(i phi) sin(a)``, so that a phase is a LAB B1 phase and a refocusing pulse 90
                # degrees from the excitation is the CPMG geometry.
                a, phi = np.radians(float(ev.flip_deg)), np.radians(float(ev.phase_deg))
                paths = [p for p in ((0, "F+", -1j * np.exp(1j * phi) * np.sin(a), ()),
                                     (0, "Z", complex(np.cos(a)), ()))
                         if abs(p[2]) >= threshold]
                excited = True
            else:
                paths = _branch(paths, ev.flip_deg, ev.phase_deg, threshold)
        else:
            paths = [(k, st, amp, hist + ((st, ev.duration),)) for k, st, amp, hist in paths]
            paths = _wind(paths, ev.dk)
            if ev.readout:
                out.extend(Pathway(eta=amp, intervals=hist, readout_idx=readout_idx)
                           for k, st, amp, hist in paths if k == 0 and st in ("F+", "F-") and abs(amp) >= threshold)
                readout_idx += 1
    out.sort(key=lambda p: (p.readout_idx, -abs(p.eta)))
    return out


def splice_schedule(n_echoes, beta_deg, ESP=1.0, refocus_phase_deg=90.0):
    """A SPLICE train: the split acquisition of fast spin-echo signals (Schick 1997).

    The readout gradient is **prolonged and unbalanced** -- the pre-phaser carries a quarter of the area the
    readout then plays, not the half an ordinary fast spin echo uses -- so the interval winds three orders
    instead of two and TWO echoes form in it rather than one. Written as the winding structure alone:

        pre-phaser   +1 over ESP / 4
        refocus      beta
        E1           +1 over ESP / 4   read here
        E2           +1 over ESP / 2   read here

    What the two echoes separate is **not** spin echoes from stimulated echoes: each family carries both.
    They are the two CONJUGATION PARITIES -- pathways refocused an even and an odd number of times -- which
    is why the split works at all after a diffusion preparation. The preparation leaves every spin an
    arbitrary phase, and that phase enters the two families as ``+phi`` and ``-phi``, constant within each,
    so each family's MAGNITUDE is insensitive to it. The two are reconstructed separately and their
    magnitude images summed. That is what lets this family violate the CPMG condition and survive.

    At ``beta_deg = 180`` the split degenerates: nothing is stored along B0, one pathway survives, and it
    lands alternately in one family and the other (1, 0, 1, 0 against 0, 1, 0, 1), so each k-space would get
    only every second line. A real SPLICE train therefore runs below 180.

    Amplitudes reproduce the reference implementation of Rahbek et al. 2023 (MRM 89:1469, ``epg_splice.m``)
    exactly; the first few have closed forms: ``sin^2(beta/2)`` and ``(1/2) sin^2(beta)`` for E1's first two
    echoes, ``sin^4(beta/2)`` and ``sin^2(beta/2) sin^2(beta)`` for E2's second and third.
    """
    ev = [Pulse(90.0)]
    for _ in range(int(n_echoes)):
        ev += [Winding(+1, float(ESP) / 4),
               Pulse(float(beta_deg), float(refocus_phase_deg)),
               Winding(+1, float(ESP) / 4, readout=True),
               Winding(+1, float(ESP) / 2, readout=True)]
    return Schedule(tuple(ev))


def cpmg_schedule(n_echoes, beta_deg=180.0, TE=1.0, refocus_phase_deg=90.0):
    """A CPMG train: an excitation, then ``n_echoes`` refocusing pulses each between two unit windings.

    ``refocus_phase_deg`` is the refocusing pulses' B1 phase against an excitation about x: 90 is the CPMG
    condition, refocusing about the axis the magnetisation lies along, and 0 is not. The two agree at a
    perfect 180 and part company at every other flip, which is what the condition is about.
    """
    ev = [Pulse(90.0)]
    for _ in range(int(n_echoes)):
        ev += [Winding(+1, TE / 2), Pulse(float(beta_deg), float(refocus_phase_deg)),
               Winding(+1, TE / 2, readout=True)]
    return Schedule(tuple(ev))


def ste_schedule(alpha1_deg=90.0, alpha2_deg=90.0, alpha3_deg=90.0, delta=1.0, TM=1.0, spoil=0):
    """A three-pulse stimulated echo: excite, dephase, store, mix, recall, rephase, read.

    Both gradient lobes wind the SAME way, as a stimulated echo's lobes physically do; there is no 180 to
    fold, and what rephases the stored magnetisation is the recall pulse turning it into the conjugate
    coherence, which then winds back down to order zero. That is why the echoing pathway reads
    ``F+ / Z / F-`` and its ``eps_P`` is ``+1 / 0 / -1``: the sign the replay needs lives on the pathway,
    not on the lobe.

    ``spoil`` is the crusher's winding over the mixing time, in whole coherence orders. Zero leaves the
    competing transverse branches in place and they contribute to the echo. A crusher removes them only if
    the rest of the sequence cannot wind them back to order zero, so its size is not free: on this schedule
    a winding of 2 is rewound by the two lobes and returns an unwanted pathway to the readout, which is the
    ordinary reason a real crusher's area is chosen with care rather than merely made large.
    """
    return Schedule((Pulse(float(alpha1_deg)), Winding(+1, delta), Pulse(float(alpha2_deg)),
                     Winding(int(spoil), TM), Pulse(float(alpha3_deg)), Winding(+1, delta, readout=True)))


def ste_amplitude(alpha1_deg=90.0, alpha2_deg=90.0, alpha3_deg=90.0):
    """The stimulated echo's amplitude: ``0.5 sin(a1) sin(a2) sin(a3)``, as the pathway product gives it.

    The one half of a stimulated echo is this number at three 90 degree pulses. It is the magnitude of the
    stored pathway's rotation product, not a factor applied by hand.
    """
    stored = [p for p in enumerate_pathways(ste_schedule(alpha1_deg, alpha2_deg, alpha3_deg, spoil=4),
                                            threshold=1e-12)
              if any(st == "Z" for st, _ in p.intervals)]
    return float(abs(stored[0].eta)) if stored else 0.0


# ------------------------------------------------------------------ the configuration-state vector
def rf_operator(flip_deg, phase_deg=0.0):
    """The 3x3 mixing a pulse applies to ``(F+, F-, Z)`` at every coherence order (Weigel 2015, eq. 2.4)."""
    a = np.deg2rad(float(flip_deg))
    p = np.deg2rad(float(phase_deg))
    ca2, sa2 = np.cos(a / 2) ** 2, np.sin(a / 2) ** 2
    sa, ca = np.sin(a), np.cos(a)
    e1, e2 = np.exp(1j * p), np.exp(2j * p)
    return np.array([
        [ca2,            e2 * sa2,         -1j * e1 * sa],
        [np.conj(e2) * sa2, ca2,            1j * np.conj(e1) * sa],
        [-0.5j * np.conj(e1) * sa, 0.5j * e1 * sa, ca],
    ], dtype=np.complex128)


class EPGState:
    """The configuration states of a spin ensemble: ``F`` over coherence orders ``-N..N`` and ``Z`` over
    ``0..N``.

    Stored over the FULL integer range rather than folded, because folding needs a reality condition applied
    at exactly the right moment and getting that wrong is silent -- it reproduces the first echo of a train
    and then quietly drifts. ``F[0]`` is the observable transverse magnetisation.
    """

    def __init__(self, n_orders):
        self.n = int(n_orders)
        self.F = np.zeros(2 * self.n + 1, np.complex128)      # index n + k holds order k
        self.Z = np.zeros(self.n + 1, np.complex128)

    @classmethod
    def equilibrium(cls, n_orders):
        st = cls(n_orders)
        st.Z[0] = 1.0
        return st

    def copy(self):
        out = EPGState(self.n)
        out.F, out.Z = self.F.copy(), self.Z.copy()
        return out

    def pulse(self, flip_deg, phase_deg=0.0):
        """Mix ``(F_k, F_{-k}^*, Z_k)`` at every order -- the one place the three states talk to each other."""
        T = rf_operator(flip_deg, phase_deg)
        n = self.n
        Fp = self.F[n:]                                  # orders 0..N
        Fm = np.conj(self.F[n::-1])                      # conj(F_{-k}) for k = 0..N
        Z = self.Z
        new_p = T[0, 0] * Fp + T[0, 1] * Fm + T[0, 2] * Z
        new_m = T[1, 0] * Fp + T[1, 1] * Fm + T[1, 2] * Z
        new_z = T[2, 0] * Fp + T[2, 1] * Fm + T[2, 2] * Z
        self.F[n:] = new_p
        self.F[n::-1] = np.conj(new_m)
        self.Z = new_z
        return self

    def shift(self, dk=1):
        """Wind every transverse coherence by ``dk`` orders: what an interval's gradient does at voxel scale.

        ``Z`` does not move, which is the whole reason a stored pathway comes back where it left."""
        k = int(dk)
        if k:
            self.F = np.roll(self.F, k)
            if k > 0:
                self.F[:k] = 0.0
            else:
                self.F[k:] = 0.0
        return self

    def off_resonance(self, dt, dw):
        """Advance EVERY transverse state by ``dw * dt`` radians, and the stored ones by nothing.

        Off-resonance is gated like the gradient in one respect only -- magnetisation parked along B0
        accumulates none of it -- and NOT in the other. The dephasing index does not enter. This array holds
        the F+ coefficients over the whole integer range, and they are all ordinary transverse
        magnetisation, so a uniform offset advances them all by the same angle whatever their winding.

        What reverses the accumulated phase is the RF, not the index: a refocusing pulse conjugates the
        state, which negates everything accrued so far. That is why a spin echo refocuses an offset and a
        gradient echo does not, and why the answer is the same whether a bipolar pair was wound +/- or -/+.
        Keying the sign to the index instead gets the second of those backwards -- it reports zero for a
        gradient echo wound negative-first, where the truth is the same phase as positive-first.
        """
        if dw:
            self.F *= np.exp(1j * float(dw) * float(dt))
        return self

    def relax(self, dt, T1=None, T2=None):
        """Per-interval relaxation; longitudinal order zero regrows toward 1."""
        if T2:
            self.F *= np.exp(-float(dt) / float(T2))
        if T1:
            e1 = np.exp(-float(dt) / float(T1))
            self.Z *= e1
            self.Z[0] += 1.0 - e1
        return self

    @property
    def signal(self):
        """The observable transverse magnetisation: the order-zero ``F``."""
        return complex(self.F[self.n])


def split_by_gate(events, n_orders, gradient_on):
    """Propagate through ``events``, splitting the state whenever the gradient is ON.

    ``events`` is a list of ``("pulse", flip, phase)`` and ``("interval", dk, index)``. ``gradient_on`` says
    whether interval ``index`` carries a microscopic gradient. Where it does, the state is split into its
    three kinds -- because they accumulate ``+phase``, ``-phase`` and nothing -- and the kind is appended to
    that branch's gate. Where it does not, nothing splits: that interval cannot tell pathways apart.

    Returns ``{gate: EPGState}``. For a diffusion-prepared train the gradient is on only in the preparation,
    so this terminates with a handful of branches however long the train that follows is.
    """
    branches = {(): EPGState.equilibrium(n_orders)}
    for ev in events:
        if ev[0] == "pulse":
            for st in branches.values():
                st.pulse(ev[1], ev[2])
            continue
        dk, idx = ev[1], ev[2]
        if not gradient_on(idx):
            for st in branches.values():
                st.shift(dk)
            continue
        out = {}
        n = n_orders
        for gate, st in branches.items():
            for kind, sl in (("F+", slice(n, None)), ("F-", slice(None, n)), ("Z", None)):
                part = EPGState(n)
                if kind == "Z":
                    part.Z = st.Z.copy()
                else:
                    part.F[sl] = st.F[sl]
                if not np.any(np.abs(part.F)) and not np.any(np.abs(part.Z)):
                    continue
                part.shift(dk)
                key = gate + (kind,)
                if key in out:
                    out[key].F += part.F
                    out[key].Z += part.Z
                else:
                    out[key] = part
        branches = out
    return branches


def train_weights(prep_events, train_events, n_orders, gradient_on, readouts, dw=0.0, durations=None):
    """``{gate: [amplitude at each readout]}``: what each microscopic gate contributes to each echo.

    The preparation is split by gate; each branch is then propagated through the train as a state vector,
    which is where the pathway explosion would otherwise live and where it costs nothing.
    """
    branches = split_by_gate(prep_events, n_orders, gradient_on)
    out = {}
    for gate, st in branches.items():
        s = st.copy()
        amps, ro = [], set(readouts)
        for i, ev in enumerate(train_events):
            if ev[0] == "pulse":
                s.pulse(ev[1], ev[2])
            else:
                s.shift(ev[1])
                if dw and durations is not None:
                    s.off_resonance(durations[ev[2]] if ev[2] < len(durations) else 0.0, dw)
            if i in ro:
                amps.append(s.signal)
        out[gate] = np.asarray(amps, np.complex128)
    return out
