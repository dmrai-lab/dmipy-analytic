"""The physical constants against the CODATA 2018 values their docstrings state."""
import numpy as np

from dmipy_analytic.constants import GAMMA, GAMMA_BAR, MU0

CODATA_GAMMA_P_SHIELDED = 2.675153151e8       # rad/(s T), protons in a spherical water sample at 25 C
CODATA_GAMMA_P = 2.6752218744e8               # rad/(s T), the bare proton
CODATA_MU0 = 1.25663706212e-6                 # N/A^2


def test_the_gyromagnetic_ratio_is_the_proton_s_to_its_stated_offsets():
    assert abs(GAMMA / CODATA_GAMMA_P_SHIELDED - 1.0 + 8.654e-6) < 1e-9
    assert abs(GAMMA / CODATA_GAMMA_P - 1.0 + 3.434e-5) < 1e-8
    assert GAMMA_BAR == GAMMA / (2.0 * np.pi)


def test_the_permeability_is_four_pi_1e_minus_7_and_the_codata_value_to_5_4e_minus_10():
    assert MU0 == 4.0e-7 * np.pi
    assert abs(MU0 / CODATA_MU0 - 1.0) < 6e-10
