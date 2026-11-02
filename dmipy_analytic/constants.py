"""The physical constants of the dmipy packages, defined once, in SI units: the proton gyromagnetic ratio ``GAMMA`` in
rad/(s T), its value per cycle ``GAMMA_BAR`` in Hz/T, and the vacuum permeability ``MU0`` in N/A^2."""
import numpy as np

GAMMA = 267.513e6
"""The proton gyromagnetic ratio, rad/(s T): 267.513e6, the value of the disimpy simulator (Kerkela et al. 2020, J.
Open Source Softw. 5:2527), so that a walk and a disimpy walk dephase alike. It lies 8.7e-6 relative below the CODATA
2018 shielded-proton value ``gamma'_p = 2.675153151e8`` (a spherical water sample at 25 C) and 3.4e-5 below the bare
proton's ``2.6752218744e8``."""

GAMMA_BAR = GAMMA / (2.0 * np.pi)
"""The proton gyromagnetic ratio per cycle, Hz/T: ``GAMMA / (2 pi)``."""

MU0 = 4.0e-7 * np.pi
"""The vacuum magnetic permeability, N/A^2: ``4 pi 1e-7``, its defined value before the 2019 SI revision, within
5.4e-10 relative of the CODATA 2018 value ``1.25663706212e-6``."""
