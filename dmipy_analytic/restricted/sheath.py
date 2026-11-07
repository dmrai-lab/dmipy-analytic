"""Water in the sheath of a straight tube, the annulus between an inner and an outer circle swept along the axis: the
signal of a gradient waveform for the two models of myelin water, the annulus filled with freely diffusing water and
the water confined to the lamellae.

**The annulus** (:func:`annulus_signal`). Water fills the sheath and diffuses isotropically with ``D``, reflected at
both walls. Its motion across the axis is reflecting Brownian motion in the annulus ``r_in <= r <= r_out`` and along
the axis free (or between reflecting caps), the two independent, so the echo factorises, ``E = E_perp E_par``.
``E_perp`` is the matrix method of the annulus, :class:`~dmipy_analytic.restricted.matrix_method.AnnulusPore`: the
Neumann modes ``R_mn(r) cos(m theta)`` (the Bessel functions of both kinds) coupled by ``x = r cos theta`` along the
gradient's fixed direction across the axis. ``E_par`` is :func:`~dmipy_analytic.restricted.surface.axial_signal`.

**The lamellae** (:func:`lamellae_signal`). Water lies in the thin aqueous gaps between the lamellae of the sheath and
moves along them, around and along the axis, never across them (Canales-Rodriguez et al. 2026): each lamella is the
surface of a circular tube (:func:`~dmipy_analytic.restricted.surface.circle_signal`) and the sheath the lamellae
evenly spaced between the two radii, so that a lamella's water is proportional to its radius. The echo is the
average of the lamellae's, ``int r E_circle(r) dr / int r dr`` over ``[r_in, r_out]``, by Gauss-Legendre quadrature
in the radius on ``n_radii`` nodes (the integrand is smooth in ``r``; the nodes are confirmed by doubling them). The
axial factor is common to every lamella.

**What separates them.** The annulus's water also diffuses across the sheath, whose thickness ``r_out - r_in`` it
explores in ``(r_out - r_in)^2 / D``; the lamellae's never does. Both tend to one circle as the sheath thins.

**Assumptions.** A straight tube, a uniform start, impermeable walls (no exchange with the lumen or the outside), no
relaxation and no surface relaxivity, one diffusivity ``D``; for the annulus a gradient along one fixed direction
across the axis (its amplitude may change in time and sign, a turning direction is refused), for the lamellae any
piecewise-constant gradient. The gradient is uniform and piecewise constant over each step ``dt``, given in the tube's
frame: ``x``, ``y`` across the axis, ``z`` along it; the effective waveform, or the waveform as played with
``refocus``. Sizes are diameters, in metres.

**Sources.** The eigenfunctions of the annulus: Carslaw HS, Jaeger JC (1959), Conduction of Heat in Solids, 2nd ed.,
Oxford, Sec. 13.5; the matrix formalism: Callaghan PT (1997), J. Magn. Reson. 129:74-84; the lamellae:
Canales-Rodriguez EJ, Tax CMW, Gorriz JM, Jones DK, Thiran J-P, Rafael-Patino J (2026), Phys. Med. Biol. 71:215006.
The forms need scipy (the ``special`` extra), imported inside them.
"""
import numpy as np

from .matrix_method import AnnulusPore, PeriodicPore
from .surface import _tube_waveform, axial_signal, circle_order

__all__ = ["annulus_signal", "lamellae_signal", "fixed_direction", "DIRECTION_RTOL", "LAMELLAE_RADII"]

DIRECTION_RTOL = 1e-9
"""How far a waveform's gradient across the axis may leave its one direction, relative to the largest sample, for the
annulus's form: the rounding of a direction stated in float32 or rotated in float64."""

LAMELLAE_RADII = 16
"""The Gauss-Legendre nodes in the radius of :func:`lamellae_signal` when none are given."""


def _diameters(outer_diameter, inner_diameter):
    outer, inner = float(outer_diameter), float(inner_diameter)
    if not 0.0 < inner < outer:
        raise ValueError(f"a sheath has 0 < inner diameter < outer diameter, not inner={inner_diameter!r}, "
                         f"outer={outer_diameter!r}")
    return outer, inner


def fixed_direction(g_perp):
    """The signed amplitude ``a(t) = G_perp(t) . u`` (n_t,) of a gradient across the axis that keeps one direction
    ``u``, the direction of its largest sample; a gradient that leaves it by more than :data:`DIRECTION_RTOL` of that
    sample is refused by name. All zero gives zero."""
    g_perp = np.asarray(g_perp, np.float64)
    norm = np.hypot(g_perp[:, 0], g_perp[:, 1])
    k = int(np.argmax(norm))
    if norm[k] == 0.0:
        return np.zeros(len(g_perp))
    u = g_perp[k] / norm[k]
    off = np.abs(g_perp[:, 0] * u[1] - g_perp[:, 1] * u[0])
    if off.max() > DIRECTION_RTOL * norm[k]:
        raise ValueError(f"the annulus's form takes a gradient along one fixed direction across the axis; this one "
                         f"leaves it by {off.max() / norm[k]:.3g} of its largest sample (a turning or a two-direction "
                         f"encoding). Give each direction's waveform on its own, or the lamellae's form, which takes "
                         f"any direction")
    return g_perp @ u


def annulus_signal(g, dt, outer_diameter, inner_diameter, diffusivity, *, length=None, refocus=(), n_modes=None,
                   axial_modes=None):
    """The complex echo of water filling the annulus between two coaxial circles of a straight tube, diffusing freely
    in it and along the axis: ``E_perp E_par``, the annulus's matrix method across the axis times the axial factor.

    Parameters
    ----------
    g : array_like, shape (n_t, 3)
        The gradient per step in the tube's frame, T/m: ``x``, ``y`` across the axis (one fixed direction), ``z``
        along it; the effective waveform, or the waveform as played with ``refocus``.
    dt : float
        The step, s.
    outer_diameter, inner_diameter : float
        The diameters of the outer and the inner wall, m.
    diffusivity : float
        The diffusivity in the sheath, m^2/s, the same across and along the axis.
    length : float or None
        The length of a tube capped by reflecting ends, m; ``None`` for an infinite tube.
    refocus : sequence of int
        The step indices before which an ideal 180 conjugates the magnetisation.
    n_modes : (int, int) or None
        The ``(angular, radial)`` truncation of the annulus,
        :data:`~dmipy_analytic.restricted.matrix_method.ANNULUS_DEFAULT_MODES` when ``None``.
    axial_modes : int or None
        The cosine modes of a capped tube's plate pore.

    The echo is normalised to the signal without gradient. Assumes the module's: a uniform start, impermeable walls,
    one diffusivity, no relaxation, a straight tube and a gradient across the axis along one fixed direction (a
    turning one is refused, :func:`fixed_direction`). It needs scipy (the ``special`` extra). Source: the annulus's
    eigenfunctions (Carslaw and Jaeger 1959, Sec. 13.5) in the matrix formalism of Callaghan (1997), J. Magn. Reson.
    129:74.
    """
    g = _tube_waveform(g)
    outer, inner = _diameters(outer_diameter, inner_diameter)
    a = fixed_direction(g[:, :2])
    across = AnnulusPore(outer, inner, diffusivity, n_modes=n_modes).signal(a, dt, refocus=refocus)
    return across * axial_signal(g[:, 2], dt, diffusivity, length=length, refocus=refocus, axial_modes=axial_modes)


def lamellae_radii(outer_diameter, inner_diameter, n_radii):
    """``(r, w)``: the Gauss-Legendre nodes in the radius over ``[r_in, r_out]`` and the weights of the lamellae's
    average, ``w ~ r`` times the rule's weight, summing to one."""
    outer, inner = _diameters(outer_diameter, inner_diameter)
    t, wq = np.polynomial.legendre.leggauss(int(n_radii))
    r_in, r_out = 0.5 * inner, 0.5 * outer
    r = r_in + 0.5 * (r_out - r_in) * (t + 1.0)
    w = wq * r
    return r, w / w.sum()


def lamellae_signal(g, dt, outer_diameter, inner_diameter, diffusivity, *, n_radii=LAMELLAE_RADII, length=None,
                    refocus=(), n_modes=None, axial_modes=None):
    """The complex echo of water confined to the lamellae of a sheath between two coaxial circles: the average over
    the lamellae, evenly spaced in the radius, of the echo of water on one circular tube.

    Parameters
    ----------
    g : array_like, shape (n_t, 3)
        The gradient per step in the tube's frame, T/m (any direction, turning or not); the effective waveform, or the
        waveform as played with ``refocus``.
    dt : float
        The step, s.
    outer_diameter, inner_diameter : float
        The diameters of the outermost and the innermost lamella, m.
    diffusivity : float
        The diffusivity along a lamella, m^2/s, the same around and along the axis.
    n_radii : int
        The Gauss-Legendre nodes of the average in the radius (:data:`LAMELLAE_RADII`).
    length : float or None
        The length of a tube capped by reflecting ends, m; ``None`` for an infinite tube.
    refocus : sequence of int
        The step indices before which an ideal 180 conjugates the magnetisation.
    n_modes : int or None
        The circle's truncation order on every lamella; the truncation rule's at the outer diameter
        (:func:`~dmipy_analytic.restricted.surface.circle_order`), which holds on every smaller lamella, when ``None``.
    axial_modes : int or None
        The cosine modes of a capped tube's plate pore.

    The echo is ``E_par sum_i w_i E_circle(r_i)``, with :func:`lamellae_radii`'s nodes and weights. Assumes the
    module's: water on the lamellae only, a uniform start, no exchange between lamellae or with the lumen or the
    outside, one diffusivity along them, no relaxation, a straight tube. It needs scipy (the ``special`` extra).
    Source: one lamella's spectral solution, Canales-Rodriguez et al. (2026), Phys. Med. Biol. 71:215006; the sheath
    as lamellae evenly spaced in the radius.
    """
    g = _tube_waveform(g)
    outer, _ = _diameters(outer_diameter, inner_diameter)
    if n_modes is None:
        n_modes = circle_order(g, dt, outer)
    r, w = lamellae_radii(outer_diameter, inner_diameter, n_radii)
    across = sum(wi * PeriodicPore.circle(2.0 * ri, diffusivity, n_modes=n_modes).signal(g[:, :2], dt, refocus=refocus)
                 for ri, wi in zip(r, w))
    return across * axial_signal(g[:, 2], dt, diffusivity, length=length, refocus=refocus, axial_modes=axial_modes)
