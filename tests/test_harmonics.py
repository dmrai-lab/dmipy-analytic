"""The real spherical-harmonic basis: the recurrence is the ``lpmv`` basis to 1e-12, the basis is orthonormal and is
DIPY's non-legacy ``tournier`` (the oracle, where DIPY is installed), and the rotation matrices are the representation
they claim to be."""
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_analytic.harmonics import cart2sphere, n_sh_coeffs, real_sh, sh_block, wigner_blocks

scipy_special = pytest.importorskip("scipy.special")            # the oracle of the recurrence


def _sphere_quadrature(n_theta, n_phi):
    """Gauss-Legendre (cos theta) x uniform (phi) product rule, weights summing to 4 pi: exact for spherical
    polynomials of degree < 2 n_theta in cos theta and < n_phi in phi."""
    x, wx = np.polynomial.legendre.leggauss(n_theta)
    phi = np.arange(n_phi) * (2.0 * np.pi / n_phi)
    st = np.sqrt(np.clip(1.0 - x ** 2, 0.0, 1.0))
    dirs = np.stack([np.outer(st, np.cos(phi)), np.outer(st, np.sin(phi)), np.outer(x, np.ones(n_phi))],
                    axis=-1).reshape(-1, 3)
    w = (wx[:, None] * (2.0 * np.pi / n_phi) * np.ones(n_phi)[None, :]).reshape(-1)
    return dirs, w


def _rot(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


ROTATIONS = np.stack([_rot([1.0, 2.0, 0.5], 0.7), _rot([0.0, 1.0, 1.0], -1.1), _rot([-0.3, 0.2, 0.9], 2.4),
                      _rot([1.0, 0.0, 0.0], np.pi), _rot([0.4, -0.7, 0.1], 0.05)])


def _real_sh_lpmv(lmax, dirs, full=False):
    d = np.asarray(dirs, np.float64).reshape(-1, 3)
    x = np.clip(d[:, 2], -1.0, 1.0); phi = np.arctan2(d[:, 1], d[:, 0])
    out = np.empty((d.shape[0], n_sh_coeffs(lmax, full)), np.float64)
    for l in range(0, lmax + 1, 1 if full else 2):
        col = out[:, sh_block(l, full)]
        col[:, l] = np.sqrt((2 * l + 1) / (4 * np.pi)) * scipy_special.lpmv(0, l, x)
        for m in range(1, l + 1):
            K = np.sqrt((2 * l + 1) / (4 * np.pi) * np.exp(scipy_special.gammaln(l - m + 1)
                                                            - scipy_special.gammaln(l + m + 1)))
            P = scipy_special.lpmv(m, l, x)
            col[:, l + m] = np.sqrt(2.0) * K * P * np.cos(m * phi)
            col[:, l - m] = np.sqrt(2.0) * K * P * np.sin(m * phi)
    return out


def test_the_recurrence_is_the_lpmv_basis_to_1e12():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(2000, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = np.concatenate([d, [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0]]])          # the poles included
    for lmax in (0, 1, 4, 8, 13, 20):
        for full in (False, True):
            npt.assert_allclose(real_sh(lmax, d, full=full), _real_sh_lpmv(lmax, d, full=full), atol=1e-12, rtol=1e-12)


def test_the_basis_is_orthonormal_by_quadrature():
    dirs, w = _sphere_quadrature(24, 48)
    Y = real_sh(12, dirs, full=True)
    npt.assert_allclose((Y * w[:, None]).T @ Y, np.eye(Y.shape[1]), atol=1e-10)


def test_the_compact_layout_counts_and_places_every_order():
    for full in (False, True):
        step = 1 if full else 2
        starts = [sh_block(l, full).start for l in range(0, 13, step)]
        stops = [sh_block(l, full).stop for l in range(0, 13, step)]
        assert starts[0] == 0 and starts[1:] == stops[:-1]                     # contiguous, in order of l
        assert stops[-1] == n_sh_coeffs(12, full) == sum(2 * l + 1 for l in range(0, 13, step))


def _dipy(name, lmax, dirs):
    pytest.importorskip("dipy", reason="dipy pins the normative basis (an oracle, never a dependency)")
    from dipy.reconst.shm import real_sh_descoteaux, real_sh_tournier
    th = np.arccos(np.clip(dirs[:, 2], -1, 1)); ph = np.arctan2(dirs[:, 1], dirs[:, 0])
    if name == "mrtrix":
        return real_sh_tournier(lmax, th, ph, legacy=True)[0]
    if name == "tournier":
        return real_sh_tournier(lmax, th, ph, legacy=False)[0]
    return real_sh_descoteaux(lmax, th, ph, legacy=False)[0]


def test_the_basis_is_dipy_tournier_non_legacy():
    dirs, _w = _sphere_quadrature(48, 96)
    npt.assert_allclose(real_sh(8, dirs), _dipy("tournier", 8, dirs), atol=1e-12)


def test_the_basis_is_orthonormal_where_the_legacy_convention_is_not():
    dirs, w = _sphere_quadrature(48, 96)
    Ym = _dipy("mrtrix", 8, dirs)
    gram = np.einsum("qa,qb,q->ab", Ym, Ym, w)
    assert np.abs(gram - np.eye(gram.shape[0])).max() > 0.4                    # the legacy basis is not orthonormal
    Yd = _dipy("descoteaux", 8, dirs)
    npt.assert_allclose(np.einsum("qa,qb,q->ab", Yd, Yd, w), np.eye(Yd.shape[1]), atol=1e-10)


def test_the_rotation_matrices_are_the_representation_they_claim_to_be():
    """``M^l(R)`` is read off the action of ``R`` on the harmonics, so it must be orthogonal, multiplicative, and
    reproduce that action."""
    lmax = 6
    blocks = wigner_blocks(lmax, ROTATIONS)
    d = ROTATIONS[:, :, 2] @ _rot([0.2, 0.3, 0.9], 0.4)                          # some directions
    for l, M in enumerate(blocks):
        assert M.shape == (len(ROTATIONS), 2 * l + 1, 2 * l + 1)
        Yl = lambda x: real_sh(l, x, full=True)[:, sh_block(l, True)]
        for k in range(len(ROTATIONS)):
            npt.assert_allclose(M[k] @ M[k].T, np.eye(2 * l + 1), atol=1e-9)
            npt.assert_allclose(Yl(d @ ROTATIONS[k].T), Yl(d) @ M[k].T, atol=1e-9)    # Y_l(R d) = M^l(R) Y_l(d)
    A, B = ROTATIONS[0], ROTATIONS[1]
    for Ma, Mb, Mab in zip(wigner_blocks(lmax, A[None]), wigner_blocks(lmax, B[None]), wigner_blocks(lmax, (A @ B)[None])):
        npt.assert_allclose(Mab[0], Ma[0] @ Mb[0], atol=1e-9)                       # a homomorphism


def test_spherical_coordinates_invert_to_the_point():
    rng = np.random.default_rng(3)
    p = rng.normal(size=(50, 3))
    r, th, ph = cart2sphere(p).T
    npt.assert_allclose(np.stack([r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)], 1), p,
                        atol=1e-12)
    npt.assert_array_equal(cart2sphere(np.zeros(3)), [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="size 3 or N x 3"):
        cart2sphere(np.zeros((2, 2, 3)))
