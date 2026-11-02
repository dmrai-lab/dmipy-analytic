# dmipy-analytic

Closed and semi-closed forms of the diffusion-MRI signal for special cases: free diffusion, restriction in simple
pores, exchange, relaxation and coherence pathways, orientation distributions and the fields of simple shapes.

Each form states the assumptions it holds under and the source it comes from (paper and equation), and is tested
against an independent route where one exists. It is the one copy the dmipy packages import: dmipy-fit fits these
forms, dmipy-replay and dmipy-sim use the ones their engines need, and the rest check replay packs and simulations
where a form and a walk describe the same situation.

```
pip install dmipy-analytic             # numpy only
pip install dmipy-analytic[special]    # the forms that need scipy's special functions
pip install dmipy-analytic[jax]        # dmipy_analytic.jax: the same forms for jit, grad and vmap
```

The modules: `constants` (the physical constants), `harmonics` (the real spherical-harmonic basis and its rotations),
`orientation` (the Watson and Bingham kernels), `gaussian` (free diffusion, the stick, the zeppelin, the tensor),
`restricted` (the plane, cylinder and sphere pores: the eigenmode method and the closed forms; the surface of a straight
tube of any outline), `exchange` (exchanging
compartments, permeable membranes), `bloch_torrey` (the one-dimensional Bloch-Torrey equation on layers with
membranes, Robin walls, relaxation, off-resonance and a background gradient, solved to a stated accuracy), `relaxation` (relaxation, surface relaxivity, magnetization exchange between
pools), `epg` (extended phase graphs) and `fields` (the fields of simple shapes of susceptibility); `dmipy_analytic.jax`
holds the JAX twins, module by module.

Licence: Apache-2.0.
