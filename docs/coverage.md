# Coverage: which form certifies which part of the replay framework

A closed or semi-closed form that a replayed pack reproduces certifies the whole chain in that situation: the
geometry and its walls, the step law, the band representation and its codec, and the contraction. This page records,
for every part of the framework, the form that can certify it, the test that holds a replay (or a walk) to it, and
the route that test exercises. A cell with no test says so. A new form names the empty cell it fills.

Four kinds of check appear below, and only the first certifies the replay framework end to end:

* **replay vs form**: a pack walked and built, replayed, and compared with the form at the pack's own standard error;
* **walk vs form**: the fused engine's signal (no pack, no codec) compared with the form;
* **form vs independent route**: the form's own test in this repository (quadrature, a limit, another series);
* **route vs route**: two implementations of the replay (or of a field) held to each other, no form involved.

Test identifiers are `repository/path::test`, on `main` of each repository. *CPU* runs on every pull request on the
free runners; *GPU tier* is `pytest -m gpu` (dmipy-sim), run on demand; *slow* is deselected by `-m "not slow"`
(dmipy-fit). The tolerance column records what the test asserts, so that a tolerance that is not derived from the
standard error and the number of comparisons is visible.

## Single-tier forms

| part of the framework | form | test | kind | tier | tolerance |
|---|---|---|---|---|---|
| positions and walls (walk, step law, codec, contraction) | matrix method, sphere and cylinder ([`restricted.matrix_method.MatrixPore`](../dmipy_analytic/restricted/matrix_method.py)) | [`dmipy-sim/tests/test_coefficient_walk.py::test_sphere_and_cylinder_agree_with_the_matrix_method`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_coefficient_walk.py) | replay vs form | CPU | `3 se + 2e-3` per measurement, 2 pores x 2 b |
| positions and walls: the plane | matrix method, plane; SGP and GPA of [`restricted.plane`](../dmipy_analytic/restricted/plane.py) | none (replay vs form). Walk vs form: [`dmipy-sim/tests/physics/test_sgp_pore_sizes.py`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/physics/test_sgp_pore_sizes.py) (Box1D, cylinder, sphere; its own SGP formulas, not this package's) | walk vs form | CPU | `z_family` (Bonferroni) |
| the step law | stepped matrix method | [`dmipy-sim/tests/math/test_step_law.py`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/math/test_step_law.py), [`dmipy-sim/tests/test_step_rule.py`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_step_rule.py) | walk vs form (the step operator) | CPU | stated `eps_D` |
| band representation (arbitrary waveforms) | matrix method on OGSE | the same test as the first row: a cosine OGSE at 400 Hz that the pack's band only just holds | replay vs form | CPU | as the first row |
| band representation: arbitrary `G(t)` | matrix method on a non-periodic waveform | none. The three-scanner result is a paper figure, not a test | | | |
| compartments (C1), one pool | `exp(-TE / T2)` ([`relaxation.transverse_decay`](../dmipy_analytic/relaxation.py)) | [`dmipy-sim/tests/test_pathway_weight.py::test_a_spin_echo_is_untouched`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_pathway_weight.py), [`dmipy-sim/tests/test_pack_replay.py::test_relaxation_and_contact_end_at_the_readout`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_pack_replay.py) | replay vs form | CPU | `rel 2e-3`; `rtol 1e-9` |
| compartments (C1), T2 per pool in a multi-pool pack | `exp(-TE / T2)` per pool | none (replay vs form) | | | |
| contact (C2), transverse `rho2` | Brownstein-Tarr (the Robin problem) | [`dmipy-sim/tests/validation/test_surface_relaxivity_ladder.py::test_relaxivity_3d_sphere`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/validation/test_surface_relaxivity_ladder.py) (the sphere's lowest Robin eigenvalue against a decay fitted to the walk) | walk vs form | GPU tier | eigenvalue ratio |
| contact (C2), longitudinal `rho1` | Robin series of a sphere | [`dmipy-sim/tests/physics/test_surface_relaxivity_t1.py::test_stimulated_echo_matches_the_sphere_robin_series`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/physics/test_surface_relaxivity_t1.py) | replay vs form | CPU | `max(5/sqrt(N), 6 % of the reference)` |
| contact (C2), `rho2` at the signal | none (two routes) | [`dmipy-sim/tests/test_coefficient_walk.py::test_the_contact_tier_is_admitted_at_the_signal`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_coefficient_walk.py) (the pack against the dense reading of the walk and against the fused engine) | route vs route | CPU | the replay's floor |
| exchange (membrane crossings) | Karger ([`exchange.karger`](../dmipy_analytic/exchange.py)) | [`dmipy-sim/tests/physics/test_karger_mc_parity.py`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/physics/test_karger_mc_parity.py): **skipped in CI** (module-level skip without dmipy-fit, which the CPU job does not install), and it calls `dmipy_fit.signal_models.exchange_models._karger_formula`, which dmipy-fit does not define | walk vs form | none | `atol 0.03` (fixed) |
| exchange: crossing rate | short-time exchange of a slab | [`dmipy-sim/tests/geometry/test_permeable_crossings.py`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/geometry/test_permeable_crossings.py) | walk vs form | CPU | |
| field (C3): the hollow cylinder | [`fields.hollow_cylinder_basis`](../dmipy_analytic/fields.py) | [`dmipy-sim/tests/test_hollow_cylinder.py`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_hollow_cylinder.py), [`dmipy-sim/tests/test_susceptibility_field_oracle.py`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_susceptibility_field_oracle.py) (the field basis against the k-space route and the Wharton-Bowtell lumen) | form vs field route (the basis, not a signal) | CPU | |
| field (C3) at the signal | none | [`dmipy-sim/tests/test_replay_fields_mt.py::test_susceptibility_replay_gre_and_se`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_replay_fields_mt.py) (replay against the fused forward engine) | route vs route | CPU | noise floor |
| vector Bloch and RF: trains and crushers | EPG ([`epg`](../dmipy_analytic/epg.py)) | [`dmipy-sim/tests/test_crusher_replay.py::test_a_crushed_train_reproduces_the_pathway_enumeration`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_crusher_replay.py) (free-water pack, b = 0), [`dmipy-sim/tests/test_splice_sequence.py::test_the_replayed_split_train_reproduces_both_echo_families`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_splice_sequence.py) (and the reference implementation's numbers), [`dmipy-sim/tests/test_pathway_weight.py::test_a_stimulated_echo_replays_to_its_closed_form`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_pathway_weight.py) | replay vs form | CPU | `1e-9` (numpy route), `1e-3` (default route); `3/sqrt(4000)` |
| vector Bloch: B1 scale | `sin(90 kappa) sin^2(90 kappa)` by enumeration | [`dmipy-sim/tests/test_pathway_weight.py::test_a_transmit_scale_on_both_pulses_scales_the_excitation_too`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_pathway_weight.py) (the replay's pathway amplitude against the enumeration; no pack is replayed) | form vs the replay's rule | CPU | `1e-9` |
| pose (orientation distributions) | Watson, Bingham SH coefficients ([`orientation`](../dmipy_analytic/orientation.py)) | [`dmipy-sim/tests/test_analytic_pose.py::test_a_watson_field_on_the_form_is_the_dispersed_integral`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_analytic_pose.py) (closed form vs quadrature); [`dmipy-replay/tests/test_fit_oracle.py`](https://github.com/dmrai-lab/dmipy-replay/blob/main/tests/test_fit_oracle.py) (the coefficients against dmipy-fit's density); [`dmipy-sim/tests/test_pose_closed_form.py`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_pose_closed_form.py) (the pose expansion against the direct posed replay) | form vs quadrature; route vs route | CPU | |
| MT bound pool | Bloch-McConnell, the Z-spectrum ([`relaxation.mt_z_spectrum`](../dmipy_analytic/relaxation.py)) | [`dmipy-sim/tests/test_replay_fields_mt.py::test_mt_replay_zspectrum_matches_oracle`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_replay_fields_mt.py) (CPU); [`dmipy-sim/tests/test_mt_zspectrum.py::test_emergent_zspectrum_matches_oracle`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_mt_zspectrum.py) (GPU tier) | replay vs form; walk vs form | CPU; GPU tier | |
| walks on surfaces | the cylinder surface and straight outlines ([`restricted.surface`](../dmipy_analytic/restricted/surface.py)) | [`dmipy-analytic/tests/restricted/test_surface.py`](../tests/restricted/test_surface.py) (form vs its independent routes); no replay check (waits for the surface family) | form vs independent route | CPU | |

## Cross-tier forms

| cross term | form | test | status |
|---|---|---|---|
| compartments x exchange x field x RF (the permeable slab) | the 1-D Bloch-Torrey equation with a membrane | none in any repository: the comparison is a figure of the replayable Monte Carlo paper, made by a script outside the test suites | open |
| restriction x internal gradient (1) | the localisation regime next to a wall (Stoller, Happer & Dyson 1991; de Swiet & Sen 1994) | none | open |
| relaxivity x exchange (2) | the Robin and transmission conditions together | none. [`dmipy-sim/tests/geometry/test_permeability_crossing.py::test_permeability_with_relaxivity_reduces_signal`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/geometry/test_permeability_crossing.py) asserts only that the signal falls | open |
| B1 x diffusion (3) | EPG with its diffusion operator (Weigel) | none | open |
| field x pose (4) | the hollow-cylinder field at an angle to B0 | none against a form; [`dmipy-sim/tests/test_pose_closed_form.py::test_the_field_is_composed_in_closed_form_and_agrees_with_every_other_route`](https://github.com/dmrai-lab/dmipy-sim/blob/main/tests/test_pose_closed_form.py) holds the pose routes to each other with the field on | route vs route only |
| MT x exchange (5) | two-pool Bloch-McConnell with permeability | none | open |
| dispersion x restriction (6) | Watson-dispersed cylinders (SH of the GPA cylinder) | [`dmipy-fit/dmipy_fit/signal_models/tests/test_sh_dispersion.py::test_watson_mc_validation`](https://github.com/dmrai-lab/dmipy-fit/blob/main/dmipy_fit/signal_models/tests/test_sh_dispersion.py) (a cylinder pack replayed over a Watson cone of axes against the dispersed GPA): marked **slow**, deselected in CI; [`dmipy-fit/dmipy_fit/signal_models/tests/test_cylinder_fixture_ogse_dispersed.py::test_ogse_dispersed_vs_mc_fixture`](https://github.com/dmrai-lab/dmipy-fit/blob/main/dmipy_fit/signal_models/tests/test_cylinder_fixture_ogse_dispersed.py) compares stored Monte-Carlo signals, not a pack | slow only |
| surface x lumen exchange | | | later |

## In review

The 1-D Bloch-Torrey reference ([dmipy-analytic#14](https://github.com/dmrai-lab/dmipy-analytic/pull/14),
`bloch_torrey.BlochTorrey1D`, against the matrix method, the Brownstein-Tarr series, the two-site Karger limit and the
Airy rate of the localisation regime) and the replay checks against it
([dmipy-replay#30](https://github.com/dmrai-lab/dmipy-replay/pull/30), `tests/test_cross_tier.py`, two committed
8000-walker packs, tolerance `z se + e_ref` with `z` Bonferroni over 59 comparisons) fill these cells:

| cross term | test | result |
|---|---|---|
| restriction x internal gradient (1) | `dmipy-replay/tests/test_cross_tier.py::test_a_background_gradient_between_plates_is_the_exact_signal_by_the_effective_waveform`, `::test_a_linear_field_between_plates_is_the_exact_signal_by_the_field_channel`, `::test_a_linear_field_between_plates_dephases_a_gradient_echo_as_the_exact_signal` | agrees by both routes (the effective waveform, the field channel), max abs(z) 1.44 over 27 comparisons |
| compartments x exchange (the slab, C1) | `::test_the_permeable_slab_is_the_exact_signal[C1-*]`, `::test_the_permeable_slab_under_pgse_is_the_exact_signal` | agrees, max abs(z) 0.11 |
| relaxivity x exchange (2) | `::test_the_permeable_slab_is_the_exact_signal[C2-*]` (the membrane's faces) | agrees, max abs(z) 1.13 |
| compartments x exchange x field (the slab, C3) | `::test_the_permeable_slab_is_the_exact_signal[C3 3T-*]`, `[C3 7T-*]` | **fails**: the derived field band (2 modes over 50 ms) keeps 16 % of the spin echo's cross-membrane dephasing at 20 ms (z = 217 at 7 T) and doubles a 10 ms gradient echo's (z = -11); the same walk with 64 field modes agrees (abs(z) < 0.9) |

## Gaps and weak checks

* **The permeable slab certifies nothing in CI.** No test in dmipy-analytic, dmipy-replay, dmipy-sim or dmipy-fit replays a slab pack against the 1-D
  reference. The per-pool T2 of a multi-pool pack, the membrane crossings at the signal and the field at the signal
  have no replay-vs-form check.
* **The Karger check does not run.** `tests/physics/test_karger_mc_parity.py` is skipped at collection on the CPU
  runners (no dmipy-fit), and dmipy-fit does not define the helper it imports; its tolerance is a fixed `atol 0.03`.
* **The matrix-method oracle carries an absolute allowance** of `2e-3` beside `3 se`, and the `rho1` Robin check a
  relative allowance of 6 % of the reference: neither is derived from the standard error and the number of comparisons.
* **The vector-Bloch / RF row is covered** for trains and crushers at b = 0 (replay against the EPG enumeration, and
  against the reference implementation of SPLICE); B1 x diffusion is open.
* **The hollow cylinder is verified as a field basis**, against the k-space dipole route; no replay of a field pack is
  held to a signal form.
* **The relaxivity ladder runs only on the GPU tier**, and it holds the fused engine (not a replay) to the Robin
  eigenvalue.
* **The plane has no replay-vs-form check**; the matrix-method oracle covers the sphere and the cylinder.
