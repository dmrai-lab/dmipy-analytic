"""The JAX twins of the forms (the ``jax`` extra): ``dmipy_analytic.jax.<module>.<form>`` is the form
``dmipy_analytic.<module>.<form>`` written in ``jax.numpy`` for ``jit``, ``grad`` and ``vmap``, with the same
arguments. The numpy float64 form is the definition; its twin is held to it by a parity test at the form whose
tolerance is the float32 rounding of the form's operations, derived in the test."""
