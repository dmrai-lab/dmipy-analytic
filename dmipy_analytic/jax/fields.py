"""The JAX twins of the forms of :mod:`dmipy_analytic.fields`, each held to its numpy definition: the field basis of
the infinite hollow cylinder and of its finite segment, traceable and elementwise (no matmul touches a position), for a
walk that reads the field at each walker's own position on a device."""
import jax.numpy as jnp

__all__ = ["hollow_cylinder_basis", "segment_basis"]


def _sym6(M):
    """``(..., 3, 3)`` symmetric -> ``(..., 6)`` in the (xx, yy, zz, xy, xz, yz) order."""
    return jnp.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2], M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]], axis=-1)


def _sym5(M):
    """``(..., 3, 3)`` symmetric -> ``(..., 5)`` in the (xx, yy, xy, xz, yz) order: the traceless block's stored
    components, ``zz`` being ``3 iso_local - xx - yy``."""
    return jnp.stack([M[..., 0, 0], M[..., 1, 1], M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]], axis=-1)


def hollow_cylinder_basis(rho_vec, u, a, b):
    """The 12 basis channels of one infinite hollow cylinder at points given by ``rho_vec`` ``(..., 3)``, the
    vector from the axis to the point (perpendicular to the axis), ``u`` ``(..., 3)`` the unit axis, ``a`` and
    ``b`` ``(...)`` the inner and outer radii (metres). Returns ``(..., 12)``: ``iso_local``, the five stored
    components of ``M_P`` (``zz`` follows from its trace), the six of ``M_A`` (:mod:`dmipy_analytic.fields`). Traceable
    (jax.numpy), in the precision of its inputs; broadcasts over leading axes."""
    rho_vec = jnp.asarray(rho_vec); u = jnp.asarray(u)
    a = jnp.asarray(a, rho_vec.dtype); b = jnp.asarray(b, rho_vec.dtype)
    rho = jnp.linalg.norm(rho_vec, axis=-1)
    rho_safe = jnp.maximum(rho, jnp.asarray(1e-30, rho_vec.dtype))
    e1 = rho_vec / rho_safe[..., None]
    e2 = jnp.cross(u, e1)
    eye = jnp.eye(3, dtype=rho_vec.dtype)
    UU = u[..., :, None] * u[..., None, :]
    P = eye - UU
    S = e1[..., :, None] * e1[..., None, :] - e2[..., :, None] * e2[..., None, :]
    r2 = rho_safe ** 2
    a2 = a ** 2; b2 = b ** 2
    lumen = rho < a
    sheath = (rho >= a) & (rho < b)
    out = rho >= b
    m = sheath.astype(rho_vec.dtype)
    # isotropic
    cP_sheath = jnp.asarray(0.5, rho_vec.dtype)
    cS_P = jnp.where(sheath, a2 / (2.0 * r2), jnp.where(out, -(b2 - a2) / (2.0 * r2), 0.0))
    MP = jnp.where(sheath, cP_sheath, 0.0)[..., None, None] * P + cS_P[..., None, None] * S
    # anisotropic
    cP_A = jnp.where(lumen, 0.5 * jnp.log(b / a), jnp.where(sheath, 0.5 * jnp.log(b / rho_safe) - 5.0 / 18.0, 0.0))
    cU_A = jnp.where(sheath, -1.0 / 9.0, 0.0)
    cS_A = jnp.where(sheath, -(1.0 + a2 / r2) / 12.0, jnp.where(out, (b2 - a2) / (12.0 * r2), 0.0))
    MA = cP_A[..., None, None] * P + cU_A[..., None, None] * UU + cS_A[..., None, None] * S
    return jnp.concatenate([(m / 3.0)[..., None], _sym5(MP), _sym6(MA)], axis=-1)


def segment_basis(rho_vec, u, z1, z2, a, b):
    """The 12 channels of one FINITE segment of the hollow cylinder, seen from outside the sheath: the dipole kernel
    ``D = (I - 3 n n^T) / (4 pi r^3)`` integrated in closed form along the segment, whose ends are at axial
    coordinates ``z1 < z2`` from the point's foot on the axis (``c(z') = foot + z' u``, ``p - c = rho_vec - z' u``),
    times the sheath's cross-section ``pi (b^2 - a^2)``: ``M_P = A L``, ``M_A = -A sym(L T)`` with ``L`` the
    integrated tensor and ``T = P / 2 - I / 3`` the annulus-averaged director tensor. For an infinite line it is
    the outside formula of :func:`hollow_cylinder_basis` (``M_P = -A S / 2 pi rho^2``, ``M_A = A S / 12 pi rho^2``),
    far away the point dipole of the segment's moment, ``A L (I - 3 r r^T) / 4 pi r^3``; a line cut into segments
    sums to the line exactly. Within the sheath's radius ``rho`` is continued at the surface (``rho >= b``): the
    value a point inside another segment's extension sees. Traceless in ``M_P``, no local term."""
    rho_vec = jnp.asarray(rho_vec); u = jnp.asarray(u); dt = rho_vec.dtype
    a = jnp.asarray(a, dt); b = jnp.asarray(b, dt); z1 = jnp.asarray(z1, dt); z2 = jnp.asarray(z2, dt)
    # in units of the outer radius: the primitives are then of order one (in metres they reach 1e24 and a
    # float32 difference of two of them is noise)
    rho_vec = rho_vec / b[..., None]; z1 = z1 / b; z2 = z2 / b; a = a / b; b = jnp.ones_like(b)
    rho = jnp.linalg.norm(rho_vec, axis=-1)
    rho_c = jnp.maximum(rho, 1.0 + 1e-6)
    rv = rho_vec * (rho_c / jnp.maximum(rho, jnp.asarray(1e-30, dt)))[..., None]
    r2 = rho_c * rho_c

    def prim(z):                                       # the four primitives at z (r^2 = rho^2 + z^2)
        rr = jnp.sqrt(r2 + z * z); r3 = rr * rr * rr
        return (z / (r2 * rr), z * (2.0 * z * z + 3.0 * r2) / (3.0 * r2 * r2 * r3), -1.0 / (3.0 * r3), z * z * z / (3.0 * r2 * r3))
    p1, p2 = prim(z1), prim(z2)
    i0, j0, j1, j2 = (q2 - q1 for q1, q2 in zip(p1, p2))
    # the six components of L = (I i0 - 3 (rv rv^T j0 - (rv u^T + u rv^T) j1 + u u^T j2)) / 4 pi, no 3x3 intermediates
    # (a 1536 x 4096 block of 3x3 tensors is 226 MB per intermediate; this form moves a third of that)
    A = jnp.pi * (b * b - a * a)
    c = A / (4.0 * jnp.pi)
    rx, ry, rz = rv[..., 0], rv[..., 1], rv[..., 2]; ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    def L6(ri, rj, ui, uj, diag):
        return c * ((i0 if diag else 0.0) - 3.0 * (ri * rj * j0 - (ri * uj + ui * rj) * j1 + ui * uj * j2))
    Lxx = L6(rx, rx, ux, ux, True); Lyy = L6(ry, ry, uy, uy, True); Lzz = L6(rz, rz, uz, uz, True)
    Lxy = L6(rx, ry, ux, uy, False); Lxz = L6(rx, rz, ux, uz, False); Lyz = L6(ry, rz, uy, uz, False)
    # M_A = -A sym(L T) with T = P/2 - I/3 = I/6 - u u^T/2:  -[A L / 6 - sym((A L u) u^T) / 2]
    wx = Lxx * ux + Lxy * uy + Lxz * uz; wy = Lxy * ux + Lyy * uy + Lyz * uz; wz = Lxz * ux + Lyz * uy + Lzz * uz   # (A L) u
    MA = jnp.stack([-(Lxx / 6.0 - wx * ux / 2.0), -(Lyy / 6.0 - wy * uy / 2.0), -(Lzz / 6.0 - wz * uz / 2.0),
                    -(Lxy / 6.0 - (wx * uy + wy * ux) / 4.0), -(Lxz / 6.0 - (wx * uz + wz * ux) / 4.0), -(Lyz / 6.0 - (wy * uz + wz * uy) / 4.0)], axis=-1)
    MP = jnp.stack([Lxx, Lyy, Lxy, Lxz, Lyz], axis=-1)                         # A L, its zz being - (xx + yy): traceless
    return jnp.concatenate([jnp.zeros(rho.shape + (1,), dt), MP, MA], axis=-1)
