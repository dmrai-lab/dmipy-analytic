"""Water confined to the surface of a straight tube: the signal of any gradient waveform, exact up to a stated
truncation, for the circular tube (the model of myelin water in the 3-4 nm aqueous gaps between the lamellae of the
sheath, Canales-Rodriguez et al. 2026) and for a tube of any closed outline.

**The walk.** The surface of a straight tube is its outline ``C(s)`` (a closed curve of perimeter ``P`` in the
``(x, y)`` plane, ``s`` its arc length) swept along the axis ``z``. It is developable: its metric is ``ds^2 + dz^2``,
flat, so a walker on it performs Brownian motion in ``(s, z)`` with the diffusivity ``D`` in both, the two
independent, ``s`` on the circle of length ``P``, whatever the outline's shape. The phase of a gradient ``G(t)``,
``gamma int G(t).(C(s(t)), z(t)) dt``, is the sum of a part across the axis, which depends on ``s`` alone, and a part
along it, which depends on ``z`` alone; the two are independent, so the signal factorises exactly,
``E = E_perp(G_perp) E_par(G_z)`` (for the circle, Canales-Rodriguez et al. 2026, Eq. 35).

**Across the axis.** ``E_perp`` is the matrix method of the outline,
:class:`~dmipy_analytic.restricted.matrix_method.PeriodicPore`: the Fourier modes of the circle of length ``P`` with
the eigenvalues ``D (2 pi k/P)^2``, coupled by the position matrices of the outline's coordinates, the matrices of the
Fourier coefficients of ``x(s)`` and ``y(s)``. For the circle of radius ``R``, the eigenvalues ``D k^2/R^2`` and the coupling ``R/2`` of the order ``k``
to ``k +/- 1`` (Eqs. 15 and 17 of the paper, in the basis ``e^{i k theta}``); for an outline, the closed polygon
through its points, whose coefficients are exact. The propagation is the ordered product of one matrix exponential
per run of constant gradient: exact for any piecewise-constant waveform, including one whose direction turns in the
plane of the outline. The rectangular PGSE across the axis is the paper's Eq. 34 (its Eq. 11 for the general pore).

**Along the axis.** ``E_par`` is free diffusion for an infinite tube, ``exp(-gamma^2 D int F_z(t)^2 dt)`` with
``F_z(t) = int_0^t G_z`` the moment of the effective waveform, exact for a piecewise-constant waveform at an echo
(``F_z = 0`` at its end; the rectangular PGSE gives ``exp(-q^2 (Delta - delta/3) D cos^2 beta)``, the paper's
Eq. 36); for a tube of length ``L`` capped by reflecting ends, it is the plate pore of separation ``L``
(:class:`~dmipy_analytic.restricted.matrix_method.MatrixPore`, ``"plane"``), as the paper's discussion states.

**The refocusing pulse.** The waveform is either the effective one (the sign of each lobe after a refocusing pulse
folded in) or the one as played with ``refocus``, the step indices before which an ideal 180 conjugates the
magnetisation. For the PGSE as played (:func:`pgse_waveform`), the two lobes then share their matrix exponentials,
the conjugation symmetry of the paper's Section 3.4.

**The truncation rule (the circle).** Keeping the orders ``|k| <= M`` changes the echo by at most
``sqrt(2) I_{M+2}(Y)``, with ``I`` the modified Bessel function and ``Y = gamma R int |G_perp(t)| dt`` the total
variation of the transverse winding: the truncated propagator is a contraction, so its error is the integral of the
flux the exact coefficients send across the cut, ``gamma |G_perp| (R/2) |c_{+/-(M+1)}|``; dropping the decay, the
coefficients are bounded by the comparison system ``b' = (gamma |G_perp| R/2)(b_{k-1} + b_{k+1})``, whose solution
from the ground mode is ``I_k(y(t))``, and ``int_0^Y I_{M+1} = 2 sum_j (-1)^j I_{M+2+2j}(Y) <= 2 I_{M+2}(Y)``. The
diffusion during the gradients only damps the high orders, and a 180 maps ``|c_k|`` onto ``|c_{-k}|``, so the bound
holds for every waveform. :func:`circle_order` is the smallest ``M`` that keeps it below ``TRUNCATION_TOLERANCE``,
the default of :func:`circle_signal`.

**The truncation of an outline.** Its order is the caller's (``n_modes``). The phase pattern wound along the outline
has the local wavenumber ``gamma F_perp(t).C'(s)``, at most that of the circle of the same perimeter, so the circle's
order at the radius ``P/(2 pi)`` is a floor; the outline's own coefficients set how far beyond it the coupling
reaches. They fall geometrically while the polygon follows a smooth curve and as ``1/n^2`` where its corners show (a
hexagon at once), where the convergence in ``M`` is algebraic. The order is confirmed by raising it.

**Assumptions.** The walker is on the surface: a gap thin against the outline's radii of curvature, with reflecting
faces and a uniform diffusivity ``D`` along it, the same along and across the axis. The tube is straight and either
infinite or capped by reflecting ends; no exchange with the lumen or the outside, no relaxation; the magnetisation is
uniform on the surface at the start. The gradient is uniform and piecewise constant over each step ``dt``, given in
the tube's frame: ``x`` and ``y`` in the outline's plane (the outline's coordinates, the circle centred on the axis),
``z`` along the axis.

**Source.** Canales-Rodriguez EJ, Tax CMW, Gorriz JM, Jones DK, Thiran J-P, Rafael-Patino J (2026), Exact analytical
PGSE signal for diffusion confined to a cylindrical surface using a spectral Laplacian formalism, Phys. Med. Biol.
71:215006, doi:10.1088/1361-6560/aeb2a0 (the circle under a rectangular PGSE); the matrix formalism of Callaghan PT
(1997), J. Magn. Reson. 129:74-84, and Barzykin AV (1999), J. Magn. Reson. 139:342-353. The forms need scipy (the
``special`` extra), imported inside them.
"""
import numpy as np

from .._extras import need
from ..constants import GAMMA
from .matrix_method import MatrixPore, PeriodicPore, _echo, _refocus_steps

__all__ = ["circle_signal", "outline_signal", "axial_signal", "circle_order", "pgse_waveform",
           "TRUNCATION_TOLERANCE", "ECHO_RTOL"]

TRUNCATION_TOLERANCE = 1e-12
"""The bound on the circle's truncation error that :func:`circle_order` keeps the echo within (the echo is at most
one)."""

ECHO_RTOL = 1e-6
"""The axial moment an infinite tube's waveform may leave at its end, relative to ``int |G_z| dt``, taken as the
rounding of an echo: a waveform stored in float32 leaves at most ``2^-24`` of it."""


def _tube_waveform(g):
    g = np.asarray(g, np.float64)
    if g.ndim != 2 or g.shape[1] != 3 or len(g) == 0 or not np.all(np.isfinite(g)):
        raise ValueError(f"the gradient is finite and (n_t, 3) in the tube's frame (T/m), not an array of shape "
                         f"{g.shape}")
    return g


def circle_order(g, dt, diameter, *, tolerance=TRUNCATION_TOLERANCE):
    """The truncation order ``M`` of the circular tube for a waveform: the smallest ``M`` with
    ``sqrt(2) I_{M+2}(Y) <= tolerance``, ``Y = gamma (d/2) int |G_perp(t)| dt``, the bound on the change of the echo
    when the orders above ``M`` are dropped (the module's truncation rule).

    Parameters
    ----------
    g : array_like, shape (n_t, 3)
        The gradient per step in the tube's frame, T/m (only its part across the axis enters).
    dt : float
        The step, s.
    diameter : float
        The diameter of the tube, m.
    tolerance : float
        The bound to keep, on the scale of the echo.

    Holds for any waveform on the circle under the module's assumptions; the bound is derived in the module's
    docstring (a comparison of the mode coefficients with the modified Bessel functions). It needs scipy (the
    ``special`` extra).
    """
    ive = need("scipy.special").ive
    g = _tube_waveform(g)
    Y = GAMMA * 0.5 * float(diameter) * float(np.hypot(g[:, 0], g[:, 1]).sum()) * float(dt)
    target = np.log(float(tolerance) / np.sqrt(2.0))
    M = 0
    with np.errstate(divide="ignore"):
        while np.log(ive(M + 2, Y)) + Y > target:      # I_v(Y) = ive(v, Y) e^Y, decreasing in v
            M += 1
    return M


def axial_signal(g_z, dt, diffusivity, *, length=None, refocus=(), axial_modes=None):
    """The factor of the echo along the tube's axis: free diffusion on an infinite tube, or the plate pore of
    separation ``length`` for a tube capped by reflecting ends.

    Parameters
    ----------
    g_z : array_like, shape (n_t,)
        The gradient along the axis per step, T/m (effective, or as played with ``refocus``).
    dt : float
        The step, s.
    diffusivity : float
        The diffusivity on the surface, m^2/s.
    length : float or None
        The length of a capped tube, m; ``None`` for an infinite tube.
    refocus : sequence of int
        The step indices before which an ideal 180 conjugates the magnetisation.
    axial_modes : int or None
        The cosine modes of the plate pore (:data:`~dmipy_analytic.restricted.matrix_method.DEFAULT_MODES` when
        ``None``); unused for an infinite tube.

    For an infinite tube the factor is ``exp(-gamma^2 D int F_z(t)^2 dt)``, ``F_z`` the moment of the effective
    waveform, integrated exactly over each step, ``dt (F_k^2 + F_k G_k dt + G_k^2 dt^2 / 3)``; the uniform density
    along an infinite axis cancels every echo that leaves an axial moment, so a waveform whose moment does not return
    to zero (to :data:`ECHO_RTOL`) is refused. For a capped tube it is the plate pore's exact product of matrix
    exponentials. Assumes a straight axis with the walker's motion along it independent of its motion around the
    outline, a uniform diffusivity, reflecting ends for a capped tube and a uniform start. It needs scipy (the
    ``special`` extra) for a capped tube. Source: the Gaussian phase of free diffusion (Stejskal EO, Tanner JE
    (1965), J. Chem. Phys. 42:288; ``b (Delta - delta/3)`` for rectangular pulses, Canales-Rodriguez et al. (2026),
    Phys. Med. Biol. 71:215006, Eq. 36); the plate pore of Codd SL, Callaghan PT (1999), J. Magn. Reson. 137:358.
    """
    gz = np.asarray(g_z, np.float64)
    if gz.ndim != 1 or len(gz) == 0 or not np.all(np.isfinite(gz)):
        raise ValueError(f"the axial gradient is finite and (n_t,), not an array of shape {gz.shape}")
    dt = float(dt)
    if length is not None:
        pore = MatrixPore("plane", float(length), float(diffusivity), n_modes=axial_modes)
        return _echo(pore.lam, pore.D, pore.B[None], gz[:, None], dt, refocus=refocus)
    flips = _refocus_steps(refocus, len(gz))
    marks = np.zeros(len(gz), int)
    for k in flips:
        if k < len(gz):
            marks[k] = 1
    eff = gz * (-1.0) ** np.cumsum(marks)
    F = np.concatenate([[0.0], np.cumsum(eff) * dt])
    if abs(F[-1]) > ECHO_RTOL * float(np.abs(eff).sum()) * dt:
        raise ValueError(f"the waveform leaves the axial moment {F[-1]:.3e} T s/m at its end: on an infinite tube the "
                         "echo is then zero; give a refocused waveform or the length of a capped tube")
    F = F[:-1]
    exponent = dt * float(np.sum(F * F + F * eff * dt + eff * eff * dt * dt / 3.0))
    return complex(np.exp(-GAMMA ** 2 * float(diffusivity) * exponent))


def _tube(pore, g, dt, diffusivity, length, refocus, axial_modes):
    return pore.signal(g[:, :2], dt, refocus=refocus) * axial_signal(
        g[:, 2], dt, diffusivity, length=length, refocus=refocus, axial_modes=axial_modes)


def circle_signal(g, dt, diameter, diffusivity, *, length=None, refocus=(), n_modes=None, axial_modes=None):
    """The complex echo of water on the surface of a straight circular tube of diameter ``diameter``, for any
    piecewise-constant gradient waveform: ``E_perp E_par``, the matrix method across the axis times the axial factor.

    Parameters
    ----------
    g : array_like, shape (n_t, 3)
        The gradient per step in the tube's frame, T/m: ``x``, ``y`` across the axis, ``z`` along it; the effective
        waveform, or the waveform as played with ``refocus``.
    dt : float
        The step, s.
    diameter : float
        The diameter of the tube, m (the circle is centred on the axis).
    diffusivity : float
        The diffusivity on the surface, m^2/s.
    length : float or None
        The length of a tube capped by reflecting ends, m; ``None`` for an infinite tube (:func:`axial_signal`).
    refocus : sequence of int
        The step indices before which an ideal 180 conjugates the magnetisation.
    n_modes : int or None
        The truncation order ``M`` (``2 M + 1`` modes); :func:`circle_order` of the waveform when ``None``.
    axial_modes : int or None
        The cosine modes of a capped tube's plate pore.

    The echo is normalised to the signal without gradient and is real and in ``[0, 1]`` for a PGSE; it is the
    paper's Eq. 37 for a rectangular PGSE at an angle to an infinite tube, exact (to the order ``M``) for any waveform
    the steps can hold. Assumes water confined to the surface (a gap thin against the radius) with a uniform
    diffusivity, no exchange with the lumen or the outside, no relaxation, a straight tube and a uniform start. It
    needs scipy (the ``special`` extra). Source: Canales-Rodriguez et al. (2026), Phys. Med. Biol. 71:215006
    (Eqs. 15, 17, 34-37 for the rectangular PGSE).
    """
    g = _tube_waveform(g)
    if n_modes is None:
        n_modes = circle_order(g, dt, diameter)
    pore = PeriodicPore.circle(diameter, diffusivity, n_modes=n_modes)
    return _tube(pore, g, dt, diffusivity, length, refocus, axial_modes)


def outline_signal(g, dt, outline, diffusivity, *, n_modes, length=None, refocus=(), axial_modes=None):
    """The complex echo of water on the surface of a straight tube whose cross-section is the closed polygon through
    the points ``outline``, for any piecewise-constant gradient waveform: ``E_perp E_par``.

    Parameters
    ----------
    g : array_like, shape (n_t, 3)
        The gradient per step in the tube's frame, T/m: ``x``, ``y`` in the outline's plane, ``z`` along the axis;
        the effective waveform, or the waveform as played with ``refocus``.
    dt : float
        The step, s.
    outline : array_like, shape (n, 2)
        The outline's points in order, m: the vertices of the closed polygon walked along its arc length, the
        closing edge back to the first point implied. A smooth outline sampled at ``n`` points is approximated to
        second order in its spacing.
    diffusivity : float
        The diffusivity on the surface, m^2/s.
    n_modes : int
        The truncation order ``M`` (``2 M + 1`` Fourier modes of the perimeter); see the module's truncation of an
        outline.
    length : float or None
        The length of a tube capped by reflecting ends, m; ``None`` for an infinite tube.
    refocus : sequence of int
        The step indices before which an ideal 180 conjugates the magnetisation.
    axial_modes : int or None
        The cosine modes of a capped tube's plate pore.

    The echo is normalised to the signal without gradient. The circle is the outline whose only Fourier coefficients
    are those of order one; the walk along the perimeter depends on the outline only through ``P``, the coupling
    through the coefficients of the coordinates. Assumes water confined to the surface (a gap thin against the
    outline's radii of curvature) with a uniform diffusivity, no exchange with the lumen or the outside, no
    relaxation, a straight tube and a uniform start. It needs scipy (the ``special`` extra). Source: the circle's
    spectral solution (Canales-Rodriguez et al. (2026), Phys. Med. Biol. 71:215006) carried to the flat surface of
    any straight outline, in the matrix formalism of Callaghan PT (1997), J. Magn. Reson. 129:74.
    """
    g = _tube_waveform(g)
    pore = PeriodicPore.polygon(outline, diffusivity, n_modes=n_modes)
    return _tube(pore, g, dt, diffusivity, length, refocus, axial_modes)


def _steps(x, dt, name):
    n = int(round(float(x) / float(dt)))
    if abs(n * float(dt) - float(x)) > 1e-9 * float(dt):
        raise ValueError(f"{name} = {x} s is not a whole number of steps of {dt} s")
    return n


def pgse_waveform(G, delta, Delta, dt, direction, *, ramp=0.0):
    """The pulsed-gradient spin echo as played, on steps of ``dt``: ``(g, refocus)``, two lobes of amplitude ``G``
    along ``direction`` and the ideal 180 between them.

    Parameters
    ----------
    G : float
        The plateau amplitude, T/m.
    delta : float
        The duration of a lobe from the start of its ramp up to the start of its ramp down, s: its area is ``G delta``.
    Delta : float
        The separation of the starts of the two lobes, s.
    dt : float
        The step, s; ``delta``, ``Delta`` and ``ramp`` are whole numbers of steps.
    direction : array_like, shape (3,)
        The gradient's direction in the tube's frame (normalised here).
    ramp : float
        The duration of each linear ramp, s (``G / slew rate``); zero for rectangular lobes. A ramp is sampled at the
        midpoints of its steps, which keeps the area of each step.

    Returns
    -------
    g : ndarray, shape (n_t, 3)
        The gradient per step, both lobes positive, ending with the second lobe (the echo's amplitude is unchanged
        until the echo).
    refocus : tuple of int
        The step index of the 180, the middle of the gap between the lobes.

    The waveform of Stejskal and Tanner (1965) with trapezoidal lobes; the paper's Monte Carlo used 500 mT/m at
    500 T/m/s, ``ramp = 1 ms``.
    """
    n_delta, n_Delta, n_ramp = _steps(delta, dt, "delta"), _steps(Delta, dt, "Delta"), _steps(ramp, dt, "ramp")
    if n_delta < 1 or n_ramp > n_delta or n_Delta < n_delta + n_ramp:
        raise ValueError(f"a PGSE needs 0 <= ramp <= delta and delta + ramp <= Delta, not delta={delta}, "
                         f"Delta={Delta}, ramp={ramp}")
    u = np.asarray(direction, np.float64)
    if u.shape != (3,) or not np.linalg.norm(u) > 0.0:
        raise ValueError(f"the direction is a non-zero vector (3,), not {direction!r}")
    up = float(G) * (np.arange(n_ramp) + 0.5) / max(n_ramp, 1)
    lobe = np.concatenate([up, np.full(n_delta - n_ramp, float(G)), up[::-1]])
    amp = np.zeros(n_Delta + len(lobe))
    amp[:len(lobe)] = lobe
    amp[n_Delta:] = lobe
    return amp[:, None] * (u / np.linalg.norm(u))[None, :], ((len(lobe) + n_Delta) // 2,)
