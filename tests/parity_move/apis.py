"""The two spellings of every name the parity record of dmipy-analytic's forms reads (dmrai-lab/dmipy-sim#834):
:class:`Before` (where each form is on the dmipy-replay and dmipy-sim mains before the move) and :class:`After` (where
the move puts it in dmipy-analytic). The call table (:mod:`calls`) reads every name through one of them, so the two
drivers run the same calls.

A name :class:`After` cannot import yet is :class:`Pending`: the group that needs it is reported as not yet moved,
neither identical nor different."""
import importlib


class Pending(Exception):
    """A name the installed packages do not define yet."""


def _attr(path):
    """The object at ``package.module:attr``."""
    mod, _, name = path.partition(":")
    try:
        obj = importlib.import_module(mod)
    except ImportError as exc:
        raise Pending(f"{path}: {exc}") from None
    try:
        return getattr(obj, name)
    except AttributeError as exc:
        raise Pending(f"{path}: {exc}") from None


class _Spelling:
    """Names resolved lazily from :attr:`PATHS`."""

    PATHS = {}

    def __getattr__(self, name):
        if name not in self.PATHS:
            raise AttributeError(name)
        return _attr(self.PATHS[name])


def _paths(table):
    return {name: f"{module}:{attr}" for module, names in table.items() for name, attr in names.items()}


def _same(*names):
    return {n: n for n in names}


EPG = ("Pulse", "Winding", "Schedule", "Pathway", "enumerate_pathways", "splice_schedule", "cpmg_schedule",
       "ste_schedule", "ste_amplitude")
EPG_STATE = ("rf_operator", "EPGState", "split_by_gate", "train_weights")
ORIENTATION = ("watson_zonal_ratios", "watson_sh", "bingham_normalization", "bingham_canonical_sh", "bingham_sh")
MATRIX = ("MatrixPore", "matrix_restricted_signal", "matrix_restricted_batch", "DEFAULT_MODES", "RADIAL_NODES",
          "ANGULAR_NODES")
MT = ("forward_rate", "backward_rate", "bound_fraction", "kappa_MT_from_forward_rate", "dwell_time_from_fraction",
      "two_pool_generator", "evolve_two_pool", "bloch_mcconnell_transverse", "bloch_mcconnell_longitudinal",
      "mt_z_spectrum")


class Before(_Spelling):
    PATHS = _paths({
        "dmipy_replay.constants": _same("GAMMA", "GAMMA_BAR"),
        "dmipy_sim.acquisition.coil": {**_same("MU0", "segment_field")},
        "dmipy_replay.math.sh_analytical": _same("cart2sphere", *ORIENTATION),
        "dmipy_replay.replay.so3": _same("n_sh_coeffs", "sh_block", "real_sh", "wigner_blocks"),
        "dmipy_replay.acquisition.epg": _same(*EPG),
        "dmipy_replay.acquisition.epg_state": _same(*EPG_STATE),
        "dmipy_sim.math.matrix_method": {**_same(*MATRIX), "cylinder_neumann_roots": "_cyl_neumann_roots",
                                         "sphere_neumann_roots": "_sph_neumann_roots"},
        "dmipy_sim.fields.hollow_cylinder": {"annulus_mean_log": "annulus_mean_log",
                                             "jax_hollow_cylinder_basis": "hollow_cylinder_basis",
                                             "jax_segment_basis": "segment_basis"},
        "dmipy_sim.engine.mt": _same(*MT),
    })


class After(_Spelling):
    PATHS = _paths({
        "dmipy_analytic.constants": _same("GAMMA", "GAMMA_BAR", "MU0"),
        "dmipy_analytic.harmonics": _same("cart2sphere", "n_sh_coeffs", "sh_block", "real_sh", "wigner_blocks"),
        "dmipy_analytic.orientation": _same(*ORIENTATION),
        "dmipy_analytic.epg": _same(*EPG, *EPG_STATE),
        "dmipy_analytic.restricted.matrix_method": _same(*MATRIX, "cylinder_neumann_roots", "sphere_neumann_roots"),
        "dmipy_analytic.fields": _same("annulus_mean_log", "segment_field"),
        "dmipy_analytic.jax.fields": {"jax_hollow_cylinder_basis": "hollow_cylinder_basis",
                                      "jax_segment_basis": "segment_basis"},
        "dmipy_analytic.relaxation": _same(*MT),
    })


SPELLINGS = {"before": Before, "after": After}
