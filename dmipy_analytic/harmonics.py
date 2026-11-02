"""The orthonormal real spherical-harmonic basis every spherical-harmonic coefficient of the dmipy packages is written
in, its compact layout, and its rotation matrices.

The basis is DIPY's ``real_sh_tournier(legacy=False)``: the orthonormal real harmonics of Tournier et al. (2007,
NeuroImage 35:1459-1472), Condon-Shortley phase included, ``Y_l^0`` for ``m = 0``, ``sqrt(2) K_lm P_l^m cos(m phi)`` for
``m > 0`` and ``sqrt(2) K_l|m| P_l^|m| sin(|m| phi)`` for ``m < 0``. Several conventions in circulation are not
orthonormal -- DIPY's default ``legacy=True`` scales every ``m != 0`` function by ``1/sqrt(2)`` -- and every identity of
the rotation matrices holds only in an orthonormal one. A test pins this basis to DIPY's at 1e-12 where DIPY is
installed (it is an oracle, never a dependency).

The compact layout stores the orders ``l = 0, 2, 4, ...`` (or every order, ``full=True``), each as ``m = -l..l``. The
rotation matrix ``M^l(R)`` of order ``l`` is defined by ``Y_l(R d) = M^l(R) Y_l(d)``: orthogonal, a group
homomorphism, its ``m' = 0`` column the harmonics of the rotated pole.
"""
import functools

import numpy as np

__all__ = ["cart2sphere", "n_sh_coeffs", "sh_block", "real_sh", "wigner_blocks"]


def cart2sphere(cartesian_coordinates):
    """Spherical coordinates ``[r, theta, phi]`` of cartesian ``[x, y, z]``, one point ``(3,)`` or ``N`` points
    ``(N, 3)``: ``theta`` in ``[0, pi]`` the polar angle from ``z`` (0 at the origin), ``phi`` in ``[-pi, pi]`` the
    azimuth from ``x``, DIPY's convention."""
    cartesian_coordinates = np.asarray(cartesian_coordinates)
    if np.ndim(cartesian_coordinates) == 1:
        x, y, z = cartesian_coordinates
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r) if r > 0 else 0.0
        phi = np.arctan2(y, x)
        return np.r_[r, theta, phi]
    elif np.ndim(cartesian_coordinates) == 2:
        x, y, z = cartesian_coordinates.T
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.where(r > 0, np.arccos(z / r), 0.)
        phi = np.arctan2(y, x)
        return np.c_[r, theta, phi]
    raise ValueError("coordinates must be array of size 3 or N x 3.")


def n_sh_coeffs(lmax, full=False):
    """Coefficients in the compact layout up to ``lmax`` (even orders only unless ``full``).

    ``full=True`` keeps odd orders too.  It is needed wherever the two acquisition directions
    are expanded: antipodal symmetry of the response constrains ``l1 + l2`` to be even, which
    admits odd--odd pairs such as ``P_1 P_1`` -- weaker than requiring each order even.
    """
    step = 1 if full else 2
    return sum(2 * l + 1 for l in range(0, lmax + 1, step))


def sh_block(l, full=False):
    """Slice of the compact array holding order ``l`` (``m = -l..+l``)."""
    if full:
        return slice(l * l, l * l + 2 * l + 1)
    M = l // 2
    start = M * (2 * M - 1) if M > 0 else 0
    return slice(start, start + 2 * l + 1)


def real_sh(lmax, dirs, full=False):
    """Orthonormal real spherical harmonics in the compact layout.

    ``dirs`` is ``(n, 3)`` unit Cartesian; returns ``(n, n_sh_coeffs(lmax, full))``. Every order comes from one
    pass of the standard three-term recurrences on the fully normalised associated Legendre functions
    (Condon-Shortley phase included, as ``scipy.special.lpmv`` has it), vectorised over the directions: the
    same numbers as evaluating ``lpmv`` per ``(l, m)`` to 1e-12, at a small fraction of its cost.
    """
    d = np.asarray(dirs, np.float64).reshape(-1, 3)
    x = np.clip(d[:, 2], -1.0, 1.0)
    phi = np.arctan2(d[:, 1], d[:, 0])
    s = np.sqrt(np.clip(1.0 - x * x, 0.0, 1.0))
    L = int(lmax)
    # P[l][m] = K_lm P_l^m(x), K_lm = sqrt((2l+1)/4pi (l-m)!/(l+m)!): the diagonal by the sectoral recurrence, the
    # first off-diagonal by one step, the rest by the standard recurrence in l at fixed m
    P = [[None] * (l + 1) for l in range(L + 1)]
    P[0][0] = np.full(d.shape[0], 1.0 / np.sqrt(4.0 * np.pi))
    for m in range(1, L + 1):
        P[m][m] = -np.sqrt((2.0 * m + 1.0) / (2.0 * m)) * s * P[m - 1][m - 1]
    for m in range(0, L):
        P[m + 1][m] = np.sqrt(2.0 * m + 3.0) * x * P[m][m]
    for m in range(0, L + 1):
        for l in range(m + 2, L + 1):
            a = np.sqrt((4.0 * l * l - 1.0) / (l * l - m * m))
            b = np.sqrt(((l - 1.0) ** 2 - m * m) / (4.0 * (l - 1.0) ** 2 - 1.0))
            P[l][m] = a * x * P[l - 1][m] - a * b * P[l - 2][m]
    cos_m = [np.cos(m * phi) for m in range(L + 1)]
    sin_m = [np.sin(m * phi) for m in range(L + 1)]
    out = np.empty((d.shape[0], n_sh_coeffs(L, full)), np.float64)
    for l in range(0, L + 1, 1 if full else 2):
        col = out[:, sh_block(l, full)]
        col[:, l] = P[l][0]
        for m in range(1, l + 1):
            col[:, l + m] = np.sqrt(2.0) * P[l][m] * cos_m[m]
            col[:, l - m] = np.sqrt(2.0) * P[l][m] * sin_m[m]
    return out


def _sh_l(l, dirs):
    """Order-``l`` block of the orthonormal real spherical harmonics, ``(n, 2l+1)``: the recurrence of
    :func:`real_sh` (``lpmv`` is not used anywhere: at high orders it has crashed the process)."""
    return np.ascontiguousarray(real_sh(int(l), dirs, full=True)[:, sh_block(int(l), True)])


@functools.lru_cache(maxsize=64)
def _anchors(l):
    """Directions and the pseudo-inverse of their harmonics, for reading a rotation matrix off its action.

    ``M^l(R)`` is fixed by ``Y_l(R a) = M^l(R) Y_l(a)`` at ``2l+1`` independent directions. The anchor set is
    over-determined (a Fibonacci spiral of ``4l+4`` directions) and inverted in least squares, which keeps the
    read-off well conditioned without a hand-rolled Wigner recursion to get wrong.
    """
    k = 4 * l + 4
    i = np.arange(k) + 0.5
    z = 1.0 - 2.0 * i / k
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    phi = i * np.pi * (1.0 + np.sqrt(5.0))
    a = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    A = _sh_l(l, a)
    return a, np.linalg.pinv(A)


def wigner_blocks(lmax, R):
    """Per order ``l``, the ``(N, 2l+1, 2l+1)`` real rotation matrices ``M^l(R)`` of the harmonic basis.

    ``M^l`` is defined by ``Y_lm(R d) = sum_m' M^l(R)[m, m'] Y_lm'(d)``, so it is a group homomorphism and
    orthogonal, and its ``m' = 0`` column is the harmonics of the rotated pole.
    """
    R = np.asarray(R, np.float64).reshape(-1, 3, 3)
    out = []
    for l in range(int(lmax) + 1):
        a, Ainv = _anchors(l)
        Ra = np.einsum("nij,kj->nki", R, a)                       # R a_k for every rotation
        Y = _sh_l(l, Ra.reshape(-1, 3)).reshape(R.shape[0], a.shape[0], 2 * l + 1)
        out.append(np.einsum("nkm,jk->nmj", Y, Ainv))
    return out
