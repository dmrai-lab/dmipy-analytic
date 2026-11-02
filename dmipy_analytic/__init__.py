"""dmipy-analytic: closed and semi-closed forms of the diffusion-MRI signal for special cases.

Every form states the assumptions it holds under and the source it comes from (paper and equation), and is tested
against an independent route where one exists. The package depends on numpy alone at import; a form that needs a
special function imports scipy inside the function (the ``special`` extra), and ``dmipy_analytic.jax`` holds the same
forms for JAX (the ``jax`` extra), each held to its numpy definition.
"""
__version__ = "3.0.0"
