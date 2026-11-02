# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project follows [Semantic Versioning](https://semver.org/).

## [3.0.0] - 2026-11-02

dmipy-analytic holds the closed and semi-closed forms of the diffusion-MRI signal for special cases, and the physical
constants of the dmipy packages. It is the bottom layer of the stack and the one copy the other packages import:
[dmipy-fit](https://github.com/dmrai-lab/dmipy-fit) fits these forms, and
[dmipy-replay](https://github.com/dmrai-lab/dmipy-replay) and [dmipy-sim](https://github.com/dmrai-lab/dmipy-sim) use the
ones their engines need.

- **The forms, by module.**
  - `constants`: the proton gyromagnetic ratio `GAMMA` (rad/(s T)) and `GAMMA_BAR` (Hz/T), and `MU0`.
  - `harmonics`: the orthonormal real spherical-harmonic basis, its compact layout and its rotation matrices.
  - `orientation`: the Watson and Bingham kernels and their exact harmonic coefficients.
  - `gaussian`: free diffusion, the stick, the zeppelin and the tensor under a b-value or a b-tensor, the
    time-dependent zeppelin and the powder averages.
  - `restricted`: the plane, the cylinder and the sphere pore, by the eigenmode (matrix) method for any sampled gradient
    waveform and in closed form (the short-gradient-pulse limit, the Gaussian-phase approximation, the capped cylinder).
  - `exchange`: the Karger model, its exchange generator, and the spin-echo and stimulated-echo propagators with finite
    pulses.
  - `relaxation`: transverse and longitudinal decay, surface relaxation and the surface-to-volume ratios, and the
    exchange of magnetization between a free and a bound pool (Bloch-McConnell, the Z-spectrum).
  - `epg`: extended phase graphs, the coherence pathways of a schedule of pulses and windings.
  - `fields`: the fields of simple shapes of susceptibility (the hollow cylinder, infinite and as a segment) and of a
    current segment.
- **NumPy definitions, JAX twins.** The NumPy float64 form is the definition. `dmipy_analytic.jax.<module>.<form>` is the
  same form with the same arguments in `jax.numpy` for `jit`, `grad` and `vmap` (`gaussian`, `restricted`, `exchange`,
  `relaxation`, `fields`), held to its definition by a test at a tolerance derived from the float32 rounding of its
  operations. The sphere's Callaghan series has no twin. The zeppelin's powder average
  (`gaussian.zeppelin_powder_average`) covers the prolate, the spherical and the oblate tensor: where JAX has no `erfi`,
  the oblate average is a Gauss-Legendre rule of 58 nodes with a proved bound, 2^-53 relative for
  b (D_perp - D_par) <= 64, and the twin returns NaN beyond.
- **NumPy at import.** `import dmipy_analytic` needs NumPy alone. A form that needs a special function imports SciPy inside
  the function (the `special` extra), and the twins need the `jax` extra; a call whose extra is missing is refused with its
  `pip install` spelling.
- **Sources.** Each form states its assumptions and its source: the paper and, where the equation number was checked
  against the paper's text, the equation; where it was not, the paper alone. Each is tested against an independent route
  where one exists (quadrature, the numerical Fourier transform of the pore, the spectral sum of its modes, the matrix
  method at small gradients; CODATA for the constants; DIPY, as an oracle only, for the harmonic basis).
- **Requires** Python >= 3.10 and NumPy >= 2.0; the extras `special` (SciPy >= 1.11) and `jax` (JAX >= 0.6.2). It depends
  on no other dmipy package.
- **Licence** Apache-2.0.
