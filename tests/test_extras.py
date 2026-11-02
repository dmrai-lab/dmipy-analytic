"""A form that needs scipy, called where scipy is not installed, is refused with the ``special`` extra's pip spelling,
and every module imports without it (run with scipy hidden from the interpreter)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _without_scipy(code):
    head = (f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n"
            "for m in ('scipy', 'scipy.special', 'scipy.linalg', 'scipy.optimize'):\n    sys.modules[m] = None\n")
    r = subprocess.run([sys.executable, "-I", "-c", head + code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_a_form_that_needs_scipy_names_the_special_extra():
    out = _without_scipy(
        "import numpy as np\n"
        "from dmipy_analytic.orientation import watson_sh\n"
        "from dmipy_analytic.restricted.matrix_method import MatrixPore\n"
        "from dmipy_analytic.relaxation import mt_z_spectrum\n"
        "for f in (lambda: watson_sh(np.array([0.0, 0.0, 1.0]), 4.0), lambda: MatrixPore('cylinder', 1e-5, 2e-9),\n"
        "          lambda: mt_z_spectrum(0.0, w1_hz=1.0, t_sat=0.1, T1a=1.0, T2a=0.1, T1b=1.0, T2b=1e-5, k_f=1.0,"
        " k_r=10.0)):\n"
        "    try:\n        f()\n    except ImportError as e:\n        print(e)\n")
    lines = out.splitlines()
    assert len(lines) == 3 and all("pip install 'dmipy-analytic[special]'" in l for l in lines), out


def test_the_forms_that_need_none_run_without_it():
    out = _without_scipy("import numpy as np\n"
                         "from dmipy_analytic.harmonics import real_sh\n"
                         "from dmipy_analytic import epg\n"
                         "print(real_sh(4, np.array([[0.0, 0.0, 1.0]])).shape, epg.ste_amplitude())\n")
    assert out == "(1, 15) 0.5", out
