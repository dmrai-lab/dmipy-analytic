"""Record the parity reference of dmipy-analytic's forms (dmrai-lab/dmipy-sim#834) with the import paths before the
move:

    python tests/parity_move/record_reference.py OUT_DIR [GROUP ...]

run with dmipy-replay and dmipy-sim installed at the commits before the move (``reference/manifest.json`` names them).
Each group of :mod:`calls` runs in a fresh interpreter in the environment :data:`parity.ENV` fixes."""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import parity  # noqa: E402

if __name__ == "__main__":
    parity.record(sys.argv[1], "before", sys.argv[2:] or None)
