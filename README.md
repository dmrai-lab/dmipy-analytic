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

Licence: Apache-2.0.
