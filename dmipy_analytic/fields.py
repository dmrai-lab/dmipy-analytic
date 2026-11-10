"""The magnetic fields of simple shapes in closed form: the field basis of a hollow cylinder of isotropic and
anisotropic susceptibility (the myelin sheath), infinite and as a finite segment, and the field of a straight current
segment.

**The hollow cylinder.** The sheath ``a < r < b`` carries the susceptibility tensor ``chi_I I + chi_A (n n^T - I/3)``
with ``n`` the radial (lipid) unit director and ``I`` the identity tensor. Here ``chi_I = tr(chi)/3`` is the mean
isotropic susceptibility and ``chi_A = chi_radial - chi_tangential`` is the radial-minus-tangential susceptibility
difference; the eigenvalues are ``chi_I + 2 chi_A/3`` radially and ``chi_I - chi_A/3`` in both tangential directions.
Wharton & Bowtell (2012), Fig. 3C, use ``chi = chi_I I + diag(chi_A, -chi_A/2, -chi_A/2)`` in the local radial frame.
Thus ``chi_A(this package) = (3/2) chi_A(W&B)``, with identical ``chi_I`` for the same susceptibility reference:
W&B ``chi_A = -0.1 ppm`` is ``chi_A = -0.15 ppm`` here. The along-B0 field, in a basis of 13 components of 12
degrees of freedom (12 are carried), is

    dB/B0 = chi_I [ m/3 - H.M_P.H ] + chi_A H.M_A.H

with ``m`` the sheath indicator, ``H`` the unit B0 direction and, in the point's own frame (``u`` the axis, ``e1`` the
radial unit vector, ``e2 = u x e1``; ``P = I - u u^T``, ``S = e1 e1^T - e2 e2^T``):

    lumen  (r < a):      M_P = 0                              M_A = (1/2) ln(b/a) P
    sheath (a <= r < b): M_P = P/2 + (a^2 / 2 r^2) S          M_A = ((1/2) ln(b/r) - 5/18) P - u u^T / 9 - (1/12)(1 + a^2/r^2) S
    outside (r >= b):    M_P = -((b^2 - a^2) / 2 r^2) S       M_A = ((b^2 - a^2) / 12 r^2) S

so that ``H.P.H = sin^2 theta``, ``H.S.H = sin^2 theta cos 2 alpha`` (theta the fibre-to-B0 angle, alpha the azimuth of
B0 from the point's radial direction). The lumen field is ``(1/2) chi_A sin^2 theta ln(1/g)`` with ``g = a/b``,
uniform and zero with B0 along the fibre; outside, ``(chi_I + chi_A/6) (b^2 - a^2) sin^2 theta cos 2 alpha / (2 r^2)``.
These fields are the Wharton-Bowtell fields with their anisotropy converted to this package's convention.
Source: Wharton & Bowtell (2012), *Fiber orientation-dependent white matter contrast in gradient echo MRI*, PNAS
109:18559-18564, Table 1 (fields) and Fig. 3C (tensor convention), doi:10.1073/pnas.1211075109. Assumptions: an infinite straight
cylinder in a uniform main field, a susceptibility small enough for the first-order (``chi << 1``) field, and the bare
superposition value: a field computed in k-space carries its domain mean (the Lorentz reference) subtracted, a uniform
offset a magnitude signal cannot see. The forms agree with a k-space dipole solve of one annulus (4096^2 points at
0.02 um) on all 13 components.

**The current segment.** :func:`segment_field` is the Biot-Savart integral over a finite straight wire in closed form,
``B = mu0 I / (4 pi) (r1 x r2) (|r1| + |r2|) / (|r1| |r2| (|r1| |r2| + r1.r2))``, with ``r1``, ``r2`` from the point to
the segment's ends (magnetostatics in free space).
"""
import numpy as np

from .constants import MU0

__all__ = ["hollow_cylinder_basis", "segment_basis", "annulus_mean_log", "segment_field"]


def _sym6(M):
    """``(..., 3, 3)`` symmetric -> ``(..., 6)`` in the (xx, yy, zz, xy, xz, yz) order."""
    return np.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2], M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]], axis=-1)


def _sym5(M):
    """``(..., 3, 3)`` symmetric -> ``(..., 5)`` in the (xx, yy, xy, xz, yz) order: the traceless block's stored
    components, ``zz`` being ``3 iso_local - xx - yy``."""
    return np.stack([M[..., 0, 0], M[..., 1, 1], M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]], axis=-1)


def hollow_cylinder_basis(rho_vec, u, a, b):
    """The 12 basis channels of one infinite hollow cylinder at points given by ``rho_vec`` ``(..., 3)``, the vector
    from the axis to the point (perpendicular to the axis), ``u`` ``(..., 3)`` the unit axis, ``a`` and ``b`` ``(...)``
    the inner and outer radii (metres). Returns ``(..., 12)``: ``iso_local`` (``m / 3``), the five stored components of
    ``M_P`` (``zz`` follows from its trace), the six of ``M_A``; float64, broadcast over leading axes.

    The sheath tensor is ``chi = chi_I I + chi_A (n n^T - I/3)``, with ``I`` the identity tensor and ``n`` the radial
    (lipid) unit director. ``chi_I = tr(chi)/3`` is the mean isotropic susceptibility;
    ``chi_A = chi_radial - chi_tangential`` is the radial-minus-tangential susceptibility difference. The along-B0
    field is ``dB/B0 = chi_I [iso_local - H.M_P.H] + chi_A H.M_A.H``, with ``H`` the unit B0 direction.
    Wharton & Bowtell (2012), PNAS 109:18559-18564, Table 1 (fields) and Fig. 3C (tensor convention),
    doi:10.1073/pnas.1211075109, use ``chi = chi_I I + diag(chi_A, -chi_A/2, -chi_A/2)`` in the local radial frame.
    Thus ``chi_A(this package) = (3/2) chi_A(W&B)``, with identical ``chi_I`` for the same susceptibility reference:
    W&B ``chi_A = -0.1 ppm`` is ``chi_A = -0.15 ppm`` here.
    """
    rho_vec = np.asarray(rho_vec, np.float64)
    u = np.asarray(u, np.float64)
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    rho = np.linalg.norm(rho_vec, axis=-1)
    rho_safe = np.maximum(rho, 1e-30)
    e1 = rho_vec / rho_safe[..., None]
    e2 = np.cross(u, e1)
    UU = u[..., :, None] * u[..., None, :]
    P = np.eye(3) - UU
    S = e1[..., :, None] * e1[..., None, :] - e2[..., :, None] * e2[..., None, :]
    r2 = rho_safe ** 2
    a2 = a ** 2
    b2 = b ** 2
    lumen = rho < a
    sheath = (rho >= a) & (rho < b)
    out = rho >= b
    m = sheath.astype(np.float64)
    with np.errstate(over="ignore"):                  # the branch a region does not select may overflow at the axis
        cS_P = np.where(sheath, a2 / (2.0 * r2), np.where(out, -(b2 - a2) / (2.0 * r2), 0.0))
        MP = np.where(sheath, 0.5, 0.0)[..., None, None] * P + cS_P[..., None, None] * S
        cP_A = np.where(lumen, 0.5 * np.log(b / a), np.where(sheath, 0.5 * np.log(b / rho_safe) - 5.0 / 18.0, 0.0))
        cU_A = np.where(sheath, -1.0 / 9.0, 0.0)
        cS_A = np.where(sheath, -(1.0 + a2 / r2) / 12.0, np.where(out, (b2 - a2) / (12.0 * r2), 0.0))
    MA = cP_A[..., None, None] * P + cU_A[..., None, None] * UU + cS_A[..., None, None] * S
    return np.concatenate([(m / 3.0)[..., None], _sym5(MP), _sym6(MA)], axis=-1)


def segment_basis(rho_vec, u, z1, z2, a, b):
    """The 12 channels of one FINITE segment of the hollow cylinder, seen from outside the sheath: the dipole kernel
    ``D = (I - 3 n n^T) / (4 pi r^3)`` integrated in closed form along the segment, whose ends are at axial coordinates
    ``z1 < z2`` from the point's foot on the axis (``c(z') = foot + z' u``, ``p - c = rho_vec - z' u``), times the
    sheath's cross-section ``A = pi (b^2 - a^2)``: ``M_P = A L``, ``M_A = -A sym(L T)`` with ``L`` the integrated tensor and
    ``T = P / 2 - I / 3`` the annulus-averaged director tensor. For an infinite line it is the outside formula of
    :func:`hollow_cylinder_basis` (``M_P = -A S / 2 pi rho^2``, ``M_A = A S / 12 pi rho^2``), far away the point dipole
    of the segment's moment, ``A L (I - 3 r r^T) / 4 pi r^3``; a line cut into segments sums to the line exactly.
    Within the sheath's radius ``rho`` is continued at the surface (``rho >= b``): the value a point inside another
    segment's extension sees. Traceless in ``M_P``, no local term. Lengths are taken in units of the outer radius, so
    the primitives are of order one; float64, broadcast over leading axes."""
    rho_vec = np.asarray(rho_vec, np.float64)
    u = np.asarray(u, np.float64)
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    z1 = np.asarray(z1, np.float64)
    z2 = np.asarray(z2, np.float64)
    rho_vec = rho_vec / b[..., None]
    z1 = z1 / b
    z2 = z2 / b
    a = a / b
    b = np.ones_like(b)
    rho = np.linalg.norm(rho_vec, axis=-1)
    rho_c = np.maximum(rho, 1.0 + 1e-6)
    rv = rho_vec * (rho_c / np.maximum(rho, 1e-30))[..., None]
    r2 = rho_c * rho_c

    def prim(z):                                       # the four primitives at z (r^2 = rho^2 + z^2)
        rr = np.sqrt(r2 + z * z)
        r3 = rr * rr * rr
        return (z / (r2 * rr), z * (2.0 * z * z + 3.0 * r2) / (3.0 * r2 * r2 * r3), -1.0 / (3.0 * r3),
                z * z * z / (3.0 * r2 * r3))
    p1, p2 = prim(z1), prim(z2)
    i0, j0, j1, j2 = (q2 - q1 for q1, q2 in zip(p1, p2))
    # the six components of L = (I i0 - 3 (rv rv^T j0 - (rv u^T + u rv^T) j1 + u u^T j2)) / 4 pi
    A = np.pi * (b * b - a * a)
    c = A / (4.0 * np.pi)
    rx, ry, rz = rv[..., 0], rv[..., 1], rv[..., 2]
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]

    def L6(ri, rj, ui, uj, diag):
        return c * ((i0 if diag else 0.0) - 3.0 * (ri * rj * j0 - (ri * uj + ui * rj) * j1 + ui * uj * j2))
    Lxx = L6(rx, rx, ux, ux, True)
    Lyy = L6(ry, ry, uy, uy, True)
    Lzz = L6(rz, rz, uz, uz, True)
    Lxy = L6(rx, ry, ux, uy, False)
    Lxz = L6(rx, rz, ux, uz, False)
    Lyz = L6(ry, rz, uy, uz, False)
    # M_A = -A sym(L T) with T = P/2 - I/3 = I/6 - u u^T/2:  -[A L / 6 - sym((A L u) u^T) / 2]
    wx = Lxx * ux + Lxy * uy + Lxz * uz
    wy = Lxy * ux + Lyy * uy + Lyz * uz
    wz = Lxz * ux + Lyz * uy + Lzz * uz
    MA = np.stack([-(Lxx / 6.0 - wx * ux / 2.0), -(Lyy / 6.0 - wy * uy / 2.0), -(Lzz / 6.0 - wz * uz / 2.0),
                   -(Lxy / 6.0 - (wx * uy + wy * ux) / 4.0), -(Lxz / 6.0 - (wx * uz + wz * ux) / 4.0),
                   -(Lyz / 6.0 - (wy * uz + wz * uy) / 4.0)], axis=-1)
    MP = np.stack([Lxx, Lyy, Lxy, Lxz, Lyz], axis=-1)        # A L, its zz being - (xx + yy): traceless
    return np.concatenate([np.zeros(rho.shape + (1,)), MP, MA], axis=-1)


def annulus_mean_log(a, b):
    """The area average of ``ln(b/r)`` over the annulus ``a < r < b``: ``1/2 - a^2 ln(b/a) / (b^2 - a^2)``."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    return 0.5 - a ** 2 * np.log(b / a) / (b ** 2 - a ** 2)


def segment_field(a, b, points, current=1.0):
    """``B`` at ``points`` from a straight current segment running ``a -> b``, in tesla.

    The closed form of the Biot-Savart integral over a finite segment. A point on the wire, or within rounding of
    it, where the integrand diverges, is refused: there the numerator and the denominator are both rounding noise
    and their ratio a finite number of plausible size.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    P = np.atleast_2d(np.asarray(points, dtype=np.float64))
    r1, r2 = a - P, b - P
    n1 = np.linalg.norm(r1, axis=-1, keepdims=True)
    n2 = np.linalg.norm(r2, axis=-1, keepdims=True)
    cross = np.cross(r1, r2)
    denom = n1 * n2 * (n1 * n2 + np.sum(r1 * r2, axis=-1, keepdims=True))
    # ON the wire both the numerator and the denominator are rounding noise, and their ratio is a FINITE
    # number of plausible size -- up to six orders above the field being measured. It is never a NaN, so it
    # is never caught, and one such point in a probe set destroyed a whole fit. An oracle may not return a
    # wrong number that looks right, so a denominator that is not resolvably non-zero is refused.
    floor = 8.0 * np.finfo(np.float64).eps * (n1 * n2) ** 2
    if np.any(np.abs(denom) <= floor):
        bad = int(np.argmax(np.abs(denom) <= floor))
        raise ValueError(
            f"point {tuple(np.round(P[bad], 6))} lies on (or within rounding of) the segment "
            f"{tuple(np.round(a, 4))} -> {tuple(np.round(b, 4))}, where the Biot-Savart integrand diverges. "
            f"The field there is not defined; move the probe off the conductor")
    return (MU0 * current / (4.0 * np.pi)) * cross * (n1 + n2) / denom
