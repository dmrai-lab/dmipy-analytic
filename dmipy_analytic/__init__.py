"""dmipy-analytic: closed and semi-closed forms of the diffusion-MRI signal for special cases.

Every form states the assumptions it holds under and the source it comes from (paper and equation), and is tested
against an independent route where one exists. The package depends on numpy alone at import; a form that needs a
special function imports scipy inside the function (the ``special`` extra), and ``dmipy_analytic.jax`` holds the same
forms for JAX (the ``jax`` extra), each held to its numpy definition.

The modules:

* :mod:`~dmipy_analytic.constants`: the physical constants (the proton gyromagnetic ratio);
* :mod:`~dmipy_analytic.harmonics`: the real spherical-harmonic basis and its rotations;
* :mod:`~dmipy_analytic.orientation`: the Watson and Bingham kernels and their harmonic coefficients;
* :mod:`~dmipy_analytic.gaussian`: free diffusion, the stick, the zeppelin and the tensor;
* :mod:`~dmipy_analytic.restricted`: the plane, the cylinder, the sphere and the annulus pore, by the eigenmode
  (matrix) method and in closed form, the surface of a straight tube of any outline, and the water of a sheath;
* :mod:`~dmipy_analytic.exchange`: exchange between compartments and permeable membranes;
* :mod:`~dmipy_analytic.bloch_torrey`: the one-dimensional Bloch-Torrey equation on layers between two walls, with
  membranes, surface relaxivity, relaxation, off-resonance and a background gradient, solved to a stated accuracy;
* :mod:`~dmipy_analytic.permeable_shell`: the radial finite-volume Bloch-Torrey equation of a permeable sphere, cylinder
  or slab in a finite shell, under any piecewise-constant waveform, with the pools' exchange and a stated error;
* :mod:`~dmipy_analytic.relaxation`: relaxation, surface relaxivity and the exchange of magnetization between pools;
* :mod:`~dmipy_analytic.epg`: extended phase graphs;
* :mod:`~dmipy_analytic.fields`: the fields of simple shapes of susceptibility;
* :mod:`dmipy_analytic.jax`: the JAX twins, module by module.
"""
__version__ = "3.0.0"
