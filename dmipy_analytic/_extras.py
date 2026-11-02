"""The optional extras of dmipy-analytic (``special``: scipy; ``jax``): what a form that needs one imports through,
inside the function, so that every module imports with numpy alone and a call that needs an extra which is not
installed is refused with the extra's pip spelling."""
import importlib

EXTRAS = {"scipy": "special", "jax": "jax"}
"""The extra that installs each optional top-level module."""


def need(module):
    """The module ``module`` (``"scipy.special"``, ``"jax.numpy"``, ...), imported; refused, naming the extra that
    installs it, when its package is not installed."""
    top = module.split(".")[0]
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        if not str(getattr(exc, "name", "") or "").startswith(top):
            raise
        extra = EXTRAS[top]
        raise ImportError(f"{module} is not installed: this form needs dmipy-analytic's {extra!r} extra "
                          f"(pip install 'dmipy-analytic[{extra}]')") from exc
