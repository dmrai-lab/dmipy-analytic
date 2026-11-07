# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

- **The water of a sheath** (`restricted.sheath`): the echo of water in the annulus between two coaxial circles of a
  straight tube, for the two models of myelin water. `annulus_signal`: water filling the annulus and diffusing in it
  (the matrix method of the annulus, `matrix_method.AnnulusPore`, on its Neumann modes of the Bessel functions of both
  kinds, `annulus_neumann_roots`, times free diffusion or the plate pore along the axis), for a gradient along one
  fixed direction across the axis (a turning one is refused by name). `lamellae_signal`: water confined to the
  lamellae (Canales-Rodriguez et al. 2026), the circular tube's echo averaged over lamellae evenly spaced in the radius
  by Gauss-Legendre quadrature, for any gradient. Checked against the roots' cross product (none missed), the modes'
  orthonormality and the annulus's exact second moment, its long-time narrow-pulse diffraction pattern (first order
  in `delta`), the circle as the sheath thins (second order) and the disk as the hole closes, the convergence in modes
  and nodes, the factorisation and the 180 as a conjugation. The JAX twins are held to the numpy forms in float64.
- **Two exchanging pools under any sampled waveform** (`exchange.karger_waveform`): the generalised Karger equation
  `dM/dt = (K - R(t) - |q(t)|^2 diag(D)) M` (Ning et al. 2018, Eq. 5) along an effective gradient, `T2` while transverse
  and `T1` while stored, each step the exponential of its two-term Magnus exponent in closed form (fourth order; exact
  without a gradient, without exchange or with equal diffusivities). Checked against the mixture of the pools, equal
  pools, an eighth-order ODE solution (the error falls 16-fold per halving) and the narrow-pulse limit.
- **Filter-exchange imaging** (`exchange.fexi_narrow_pulse`, `fexi_adc`, `fexi_signal`, `fexi_two_site`): the narrow-pulse
  double-encoding signal of two pools (Ning et al. 2018, Eq. 7); the AXR model `ADC'(t_m) = ADC_eq [1 - sigma exp(-AXR
  t_m)]` and its signal (Lasic et al. 2011; md-dmri `fexi11`), exact for two pools of equal `T1` at a small detection b
  whatever the blocks, with `AXR = k_12 + k_21`; and the short-block two-site values of `ADC_eq`, `sigma` and `AXR`. The
  JAX twins are held to the numpy forms in float64.
- **The one-dimensional Bloch-Torrey equation** (`bloch_torrey.BlochTorrey1D`): `dm/dt = D m'' - [R2(x) + i omega(x) +
  i gamma (G(t) + g0) x] m` on layers between two walls, with any number of membranes (`D m' = kappa [m]`, a surface
  relaxivity on their faces), Robin walls, `R2` and `omega` per layer, a background gradient `g0` and the 180 as a
  conjugation; cell-centred finite volumes, the diffusion half-steps in the eigenbasis of the symmetrised operator and
  the potential applied exactly (a Strang split), second order in the cells and the step. Checked against the matrix
  method (a sealed and an absent membrane), the Brownstein-Tarr series (Robin walls; a membrane between equal layers),
  the two-site Karger limit and the Airy decay rate of the localisation regime next to a wall. The JAX twin runs the
  same steps.
- **A permeability that changes in time** (`BlochTorrey1D(permeability=, permeability_breakpoints=)`): piecewise
  constant, every interior edge taking its next permeability at each breakpoint. The operator of each interval is built
  once, and at a breakpoint (on a step boundary; one inside a step is refused) the state is carried into the next
  interval's eigenbasis, so the restart is exact. Checked against the Brownstein-Tarr series of two equal layers
  restarted on the new Robin modes at the breakpoint (the membrane opened, sealed, and changed tenfold). The JAX twin
  carries the state at breakpoints on whole steps of the waveform.
- **The Brownstein-Tarr series of two plates** (`relaxation.surface_relaxation_plane`, `relaxation.plane_robin_roots`):
  the exact eigenmode series of the magnetisation between plates with surface relaxivity, for every `rho a / D`.
- **Diffusion on the surface of a straight tube** (`restricted.surface`): the echo of water confined to the surface of a
  circular tube, the model of myelin water in the aqueous gaps between the lamellae (Canales-Rodriguez et al. 2026,
  Phys. Med. Biol. 71:215006), and of a tube of any closed outline (a polygon through given points), for any
  piecewise-constant gradient waveform. The surface is flat, so the signal is the product of the outline's matrix method
  across the axis and free diffusion (or the plate pore of a capped tube) along it. The circle's truncation order
  follows a proved bound; `pgse_waveform` builds the trapezoidal PGSE as played. The JAX twin propagates step by step.
- **The periodic pore** (`matrix_method.PeriodicPore`, `matrix_method.polygon_fourier`): the closed curve walked along
  its arc length, its coupling the exact Fourier coefficients of a polygon's coordinates.
- **The 180 as a conjugation**: `MatrixPore.signal` and the periodic pore take the waveform as played with `refocus`,
  the steps before which an ideal 180 conjugates the magnetisation; the effective waveform gives the same echo,
  conjugated after an odd number of 180s. One propagation serves every pore; `MatrixPore`'s echoes are unchanged.

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
