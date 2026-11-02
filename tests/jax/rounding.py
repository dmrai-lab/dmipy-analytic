"""The float32 rounding budget a JAX twin is held to against its float64 numpy definition.

A twin evaluated in float32 differs from the float64 form by the rounding of its inputs to float32, of each of its
operations, and of the library functions it calls. With ``U32 = 2**-24`` the unit roundoff of float32, a product or a
quotient of rounded inputs has a relative error of at most ``(n_inputs + n_operations) U32``; an exponential
``exp(-x)`` of an argument with a relative error ``e`` has a relative error ``|x| e`` (first order) and adds the
rounding of ``exp`` itself, ``EXP_ULPS`` units in the last place of float32, an ulp being at most ``2 U32``. A twin's
test states the counts of its own form in its docstring and passes them to :func:`exp_rtol`.
"""
import numpy as np

U32 = float(np.finfo(np.float32).eps) / 2.0
"""The unit roundoff of float32, ``2**-24``."""

EXP_ULPS = 2
"""The accuracy budget of the float32 exponential of XLA's CPU backend, in ulps of float32."""


def exp_rtol(exponent, n_rounded, n_operations, ulps=EXP_ULPS):
    """The relative error bound of ``exp(-x)`` for ``|x| <= exponent`` when ``x`` is formed from ``n_rounded`` inputs
    rounded to float32 by ``n_operations`` multiplications, divisions and additions of terms of one sign:
    ``(exponent (n_rounded + n_operations) + 2 ulps) U32``."""
    return (exponent * (n_rounded + n_operations) + 2 * ulps) * U32
