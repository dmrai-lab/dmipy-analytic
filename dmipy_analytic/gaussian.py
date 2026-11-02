"""Gaussian diffusion: the signal of free diffusion (the ball), of the stick, the zeppelin and the full diffusion
tensor under a b-value or a b-tensor, and their orientation (powder) averages.

Every form here is the second-cumulant (Gaussian) signal ``exp(-<B, D>)``: the b-tensor ``B`` (s/m^2) contracted with
the diffusion tensor ``D`` (m^2/s). A pulsed-gradient measurement has ``B = b n n^T`` for the unit gradient direction
``n``; the b-value forms take ``(b, n)`` and the b-tensor forms take ``B`` of shape ``(m, 3, 3)``, which also holds
the tensor-valued encodings (planar, spherical). The ball is isotropic, the stick has diffusion along its axis only,
the zeppelin is axially symmetric about its axis, and the tensor is general. The orientation (powder) average of the
stick and of the zeppelin is the mean over all axis directions at one b-value.

Sources: Stejskal EO, Tanner JE (1965), J. Chem. Phys. 42:288-292 (the Gaussian attenuation of a pulsed gradient);
Basser PJ, Mattiello J, LeBihan D (1994), Biophys. J. 66:259-267 (the tensor, Eq. 1 and the b-matrix attenuation);
Behrens TEJ et al. (2003), Magn. Reson. Med. 50:1077-1088 (the ball and the stick); Panagiotaki E et al. (2012),
NeuroImage 59:2241-2254 (the zeppelin); Kaden E et al. (2016), NeuroImage 139:346-359 (the powder averages).
"""
import numpy as np

from ._extras import need

__all__ = [
    "ball", "stick", "stick_btensor", "zeppelin", "zeppelin_btensor", "tensor", "tensor_btensor",
    "time_dependent_diffusivity", "temporal_zeppelin", "stick_powder_average", "zeppelin_powder_average",
]


def ball(b, diffusivity):
    r"""The attenuation of free isotropic Gaussian diffusion, ``exp(-b D)``.

    Parameters
    ----------
    b : array_like
        The b-value, s/m^2.
    diffusivity : float or array_like
        The diffusivity ``D``, m^2/s.

    Assumes a displacement distribution that is Gaussian and the same in every direction (no restriction, no
    exchange) and an encoding described by its b-value alone. Source: Stejskal and Tanner (1965), J. Chem. Phys.
    42:288, the attenuation of the echo ``exp(-gamma^2 G^2 delta^2 D (Delta - delta/3)) = exp(-b D)``.
    """
    return np.exp(-b * diffusivity)


def stick(b, direction, axis, diffusivity):
    r"""The attenuation of Gaussian diffusion along one axis only (the stick, a cylinder of zero radius),
    ``exp(-b D (n . u)^2)``.

    Parameters
    ----------
    b : array_like, shape (m,)
        The b-value, s/m^2.
    direction : array_like, shape (m, 3)
        The unit gradient direction ``n`` of each measurement.
    axis : array_like, shape (3,)
        The unit vector ``u`` of the stick.
    diffusivity : float
        The diffusivity along the axis, m^2/s.

    Assumes no diffusion perpendicular to the axis (the radius is zero) and Gaussian diffusion along it; a pulsed
    gradient (``B = b n n^T``). Source: Behrens et al. (2003), Magn. Reson. Med. 50:1077, the anisotropic compartment
    of Eq. 1 with ``d_perp = 0``.
    """
    return np.exp(-b * diffusivity * np.dot(direction, axis) ** 2)


def stick_btensor(btensor, axis, diffusivity):
    r"""The attenuation of the stick under a b-tensor, ``exp(-D u^T B u)``.

    Parameters
    ----------
    btensor : array_like, shape (m, 3, 3)
        The b-tensor of each measurement, s/m^2.
    axis : array_like, shape (3,)
        The unit vector ``u`` of the stick.
    diffusivity : float
        The diffusivity along the axis, m^2/s.

    Assumes no diffusion perpendicular to the axis and Gaussian diffusion along it. It is :func:`stick` for
    ``B = b n n^T`` and holds for any tensor-valued encoding. Source: Behrens et al. (2003), Magn. Reson. Med.
    50:1077, with the b-matrix attenuation of Basser et al. (1994), Biophys. J. 66:259 (Eq. 1).
    """
    return np.exp(-diffusivity * np.einsum("i,mij,j->m", axis, btensor, axis))


def zeppelin(b, direction, axis, diffusivity_par, diffusivity_perp):
    r"""The attenuation of an axially symmetric Gaussian tensor (the zeppelin) under a pulsed gradient,
    ``exp(-b (D_par (n . u)^2 + D_perp |n_perp|^2))`` with ``|n_perp|^2 = |n|^2 - (n . u)^2``.

    Parameters
    ----------
    b : array_like, shape (m,)
        The b-value, s/m^2.
    direction : array_like, shape (m, 3)
        The gradient direction ``n`` of each measurement.
    axis : array_like, shape (3,)
        The unit symmetry axis ``u``.
    diffusivity_par, diffusivity_perp : float
        The diffusivity along and perpendicular to the axis, m^2/s.

    Assumes Gaussian diffusion with a tensor symmetric about the axis and a pulsed gradient. The perpendicular part
    of the direction is its squared length less the squared projection, floored at zero. Source: Panagiotaki et al.
    (2012), NeuroImage 59:2241, the zeppelin (Gaussian anisotropic) compartment; Basser et al. (1994), Biophys. J.
    66:259, Eq. 1.
    """
    mag_par = np.sum(direction * axis, axis=-1)
    mag_sq = np.sum(direction ** 2, axis=-1)
    mag_perp_sq = np.maximum(0.0, mag_sq - mag_par ** 2)
    return np.exp(-b * (diffusivity_par * mag_par ** 2 + diffusivity_perp * mag_perp_sq))


def zeppelin_btensor(btensor, axis, diffusivity_par, diffusivity_perp):
    r"""The attenuation of the zeppelin under a b-tensor, ``exp(-(D_perp tr B + (D_par - D_perp) u^T B u))``.

    Parameters
    ----------
    btensor : array_like, shape (m, 3, 3)
        The b-tensor of each measurement, s/m^2.
    axis : array_like, shape (3,)
        The unit symmetry axis ``u``.
    diffusivity_par, diffusivity_perp : float or array_like
        The diffusivity along and perpendicular to the axis, m^2/s; an array of one value per measurement gives each
        measurement its own perpendicular diffusivity (a time-dependent one).

    Assumes Gaussian diffusion with a tensor symmetric about the axis; the encoding enters through ``B`` alone, so
    planar and spherical tensor encodings are covered. The axially symmetric tensor is
    ``D_perp I + (D_par - D_perp) u u^T``, so ``<B, D> = D_perp tr B + (D_par - D_perp) u^T B u``. Source: Basser
    et al. (1994), Biophys. J. 66:259, Eq. 1; Westin CF et al. (2016), NeuroImage 135:345-362 (the b-tensor
    attenuation).
    """
    b_trace = np.trace(btensor, axis1=1, axis2=2)
    u_dot_B_u = np.einsum("i,mij,j->m", axis, btensor, axis)
    return np.exp(-(diffusivity_perp * b_trace + (diffusivity_par - diffusivity_perp) * u_dot_B_u))


def tensor(b, direction, diffusion_tensor):
    r"""The attenuation of a general Gaussian diffusion tensor under a pulsed gradient, ``exp(-b n^T D n)``.

    Parameters
    ----------
    b : array_like, shape (m,)
        The b-value, s/m^2.
    direction : array_like, shape (m, 3)
        The unit gradient direction ``n`` of each measurement.
    diffusion_tensor : array_like, shape (3, 3)
        The symmetric diffusion tensor ``D``, m^2/s.

    Assumes Gaussian diffusion. Source: Basser et al. (1994), Biophys. J. 66:259, Eq. 1 with ``B = b n n^T``.
    """
    return np.exp(-b * np.einsum("mi,ij,mj->m", direction, diffusion_tensor, direction))


def tensor_btensor(btensor, diffusion_tensor):
    r"""The attenuation of a general Gaussian diffusion tensor under a b-tensor, ``exp(-sum_ij B_ij D_ij)``.

    Parameters
    ----------
    btensor : array_like, shape (m, 3, 3)
        The b-tensor of each measurement, s/m^2.
    diffusion_tensor : array_like, shape (3, 3)
        The symmetric diffusion tensor ``D``, m^2/s.

    Assumes Gaussian diffusion. Source: Basser et al. (1994), Biophys. J. 66:259, Eq. 1: the attenuation is the
    exponential of minus the b-matrix contracted with ``D``.
    """
    return np.exp(-np.einsum("mij,ij->m", btensor, diffusion_tensor))


def time_dependent_diffusivity(diffusivity_inf, A, delta, Delta):
    r"""The apparent perpendicular diffusivity of a pulsed-gradient measurement in a medium with one-dimensional
    structural disorder, ``D_inf + A (ln(Delta/delta) + 3/2) / (Delta - delta/3)``.

    Parameters
    ----------
    diffusivity_inf : float
        The diffusivity at infinite diffusion time, m^2/s.
    A : float
        The structural-disorder coefficient, m^2.
    delta, Delta : float or array_like
        The gradient pulse duration and separation, s.

    Assumes a long-time expansion in which the diffusivity approaches its limit as ``ln(t)/t`` (the signature of
    short-range disorder in one dimension, as the extra-axonal space of a packed bundle has perpendicular to the
    axons) and gradient pulses of finite width ``delta``. Source: Burcaw LM, Fieremans E, Novikov DS (2015),
    NeuroImage 114:18-37, the pulsed-gradient diffusivity of Eq. 7 (the ``ln(t)/t`` law integrated over rectangular
    pulses).
    """
    return diffusivity_inf + A * (np.log(Delta / delta) + 3. / 2.) / (Delta - delta / 3.)


def temporal_zeppelin(b, direction, axis, diffusivity_par, diffusivity_inf, A, delta, Delta):
    r"""The attenuation of the zeppelin whose perpendicular diffusivity is time dependent,
    ``exp(-b (D_perp + (D_par - D_perp)(n . u)^2))`` with ``D_perp =`` :func:`time_dependent_diffusivity`.

    Parameters
    ----------
    b : array_like, shape (m,)
        The b-value, s/m^2.
    direction : array_like, shape (m, 3)
        The unit gradient direction ``n`` of each measurement.
    axis : array_like, shape (3,)
        The unit symmetry axis ``u``.
    diffusivity_par : float
        The diffusivity along the axis, m^2/s.
    diffusivity_inf, A : float
        The perpendicular diffusivity at infinite time (m^2/s) and the structural-disorder coefficient (m^2).
    delta, Delta : array_like, shape (m,)
        The gradient pulse duration and separation, s.

    Assumes the parallel diffusivity is time independent and the perpendicular one follows
    :func:`time_dependent_diffusivity`. Source: Burcaw et al. (2015), NeuroImage 114:18, Eq. 7; the compartment of
    Novikov DS et al. (2019), NMR Biomed. 32:e3998.
    """
    diffusivity_perp = time_dependent_diffusivity(diffusivity_inf, A, delta, Delta)
    n_dot_u = np.dot(direction, axis)
    return np.exp(-b * (diffusivity_perp + (diffusivity_par - diffusivity_perp) * n_dot_u ** 2))


def stick_powder_average(b, diffusivity):
    r"""The orientation average of the stick's attenuation over all axis directions,
    ``sqrt(pi) erf(sqrt(b D)) / (2 sqrt(b D))``; ``1`` at ``b = 0``.

    Parameters
    ----------
    b : array_like
        The b-value, s/m^2.
    diffusivity : float
        The diffusivity along the stick, m^2/s.

    Assumes a uniform distribution of the axis over the sphere and a pulsed gradient. The average of
    ``exp(-b D t^2)`` over ``t = cos(theta)`` uniform in ``[-1, 1]`` is the error function. It needs scipy (the
    ``special`` extra). Source: Kaden et al. (2016), NeuroImage 139:346, the stick compartment of Eq. 7.
    """
    erf = need("scipy.special").erf
    b = np.asarray(b, dtype=float)
    average = np.ones_like(b)
    above = b > 0
    b_above = b[above]
    average[above] = (np.sqrt(np.pi) * erf(np.sqrt(b_above * diffusivity))) / (2 * np.sqrt(b_above * diffusivity))
    return average


def zeppelin_powder_average(b, diffusivity_par, diffusivity_perp):
    r"""The orientation average of the zeppelin's attenuation over all axis directions,
    ``exp(-b D_perp) J_0(kappa) / 2`` with ``kappa = -b (D_par - D_perp)`` and
    ``J_0(kappa) = int_{-1}^{1} exp(kappa t^2) dt``.

    Parameters
    ----------
    b : array_like
        The b-value, s/m^2.
    diffusivity_par, diffusivity_perp : float
        The diffusivity along and perpendicular to the axis, m^2/s.

    Assumes a uniform distribution of the axis over the sphere and a pulsed gradient. The integral is
    ``sqrt(pi) erf(sqrt(-kappa)) / sqrt(-kappa)`` for a prolate tensor (``kappa < 0``),
    ``sqrt(pi) erfi(sqrt(kappa)) / sqrt(kappa)`` for an oblate one, and ``2`` for ``|kappa| < 1e-12``. It needs
    scipy (the ``special`` extra). Source: Kaden et al. (2016), NeuroImage 139:346, Eq. 8 (the zeppelin's spherical
    mean).
    """
    special = need("scipy.special")
    b = np.asarray(b, dtype=float)
    kappa = -b * (diffusivity_par - diffusivity_perp)
    J0 = np.full(kappa.shape, 2.0)
    prolate = kappa <= -1e-12
    oblate = kappa >= 1e-12
    root = np.sqrt(np.abs(kappa))
    J0[prolate] = np.sqrt(np.pi) * special.erf(root[prolate]) / root[prolate]
    J0[oblate] = np.sqrt(np.pi) * special.erfi(root[oblate]) / root[oblate]
    return np.exp(-b * diffusivity_perp) * J0 / 2.0
